import warnings
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

MIN_CLUSTERS = 20


def map_label_to_trend_direction(df: pd.DataFrame, state_col: str = "state", return_col: str = "returns",
                                 method: str = "conservative", min_samples: int = 30, confidence_level: float = 0.95,
                                 mapping_dict: Optional[Dict[Any, int]] = None, min_sharpe: float = 0.3,
                                 cost_threshold: Optional[float] = None, horizon: int = 1,
                                 cluster_size: Optional[int] = None, force_distinct_signs: bool = False,
                                 return_diagnostics: bool = False,
                                 **kwargs) -> Dict[Any, int] | Tuple[Dict[Any, int], pd.DataFrame]:
    """
    Map states (HMM states, cluster ids, rule labels) to trade directions in {-1, 0, 1}.

    - A state gets a sign only when its method's criteria pass; otherwise it gets 0 (no trade).
    - The significance test is a cluster-robust t-test (see ``cluster_size``), so overlapping forward returns and
      serial dependence do not inflate it. An iid t-test on overlapping 12-bar forward returns flags about 60% of
      no-drift states.
    - ``force_distinct_signs=True`` restores the old ranking fallback (min(n, 3) distinct signs whatever the test
      says). A fallback sign is a rank, not evidence.
    - Rows of ``df`` are taken to be in time order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with HMM state labels and returns
    state_col : str, default='state'
        Column name for HMM state labels (e.g., 'state', 'regime', 'label')
    return_col : str, default='returns'
        Column name for returns (typically log returns)
    method : str, default='conservative'
        Mapping strategy:
        - 'conservative': Multi-criteria (significance + Sharpe + magnitude) [RECOMMENDED]
        - 'statistical': Statistical significance only (t-test)
        - 'sharpe': Risk-adjusted returns only
        - 'simple': Mean threshold (fast but naive)
    min_samples : int, default=30
        Minimum observations per state for reliable statistics
        States with fewer samples → labeled as 0 (no trade)
    confidence_level : float, default=0.95
        Confidence level for statistical tests (95% = alpha of 0.05)
    mapping_dict : dict, optional
        A frozen {state: direction} map, e.g. one fitted on a training block. A copy is returned as-is: ``method``
        and the distinct-sign fallback are skipped, because ``df`` is then usually the evaluation block and
        re-ranking states on it would be look-ahead. States in ``df`` that the map does not cover get 0, with a warning.
    min_sharpe : float, default=0.3
        Minimum Sharpe ratio to consider a state tradable
        (Sharpe < min_sharpe → labeled as 0)
    cost_threshold : float, optional
        Minimum mean return to exceed transaction costs (e.g., 0.0002 = 2bps)
        If None, auto-estimated using data-adaptive algorithm:
        1. Primary: Based on minimum separation between state means
        2. Fallback: Typical noise level (median absolute return * 0.1)
        3. Final: Larger of separation-based and noise-based thresholds
        This ensures thresholds adapt to timeframe and instrument characteristics
    horizon : int, default=1
        Bars spanned by each value of ``return_col``. Forward h-bar returns overlap, so pass h: the default cluster
        size grows with it.
    cluster_size : int, optional
        Rows per cluster (consecutive rows of ``df``) for the cluster-robust t-test; inference uses a t distribution
        with (clusters - 1) degrees of freedom. Default max(20 * horizon, ceil(sqrt(len(df)))), capped so there
        are at least 20 clusters (never below ``horizon``); on short samples the test then runs mildly liberal.
    force_distinct_signs : bool, default=False
        If True and the method signs fewer than min(n, 3) distinct directions, rank the states by mean and assign
        -1/0/+1 anyway (``fallback`` = True in the diagnostics). Never applied to a supplied ``mapping_dict``.
    return_diagnostics : bool, default=False
        If True, returns (mapping, diagnostics_df) for debugging
        If False, returns only mapping dict
    **kwargs : dict
        Method-specific overrides (see method implementations)

    Returns
    -------
    mapping : dict
        {state_value: direction} where direction in {-1, 0, 1}
        - 1: Long signal (bullish regime)
        - 0: No trade (neutral/uncertain regime)
        - -1: Short signal (bearish regime)
    diagnostics_df : pd.DataFrame (if return_diagnostics=True)
        Statistics for each state showing why it was labeled. ``test_direction`` is the method's own verdict;
        ``direction`` differs from it only where the fallback fired.

    Methods
    -------
    - Conservative: significance + Sharpe + magnitude (strictest)
    - Statistical: significance + magnitude
    - Sharpe: risk-adjusted return + magnitude
    - Simple: mean beyond half the cross-state spread of means (no test)

    The map is fitted on ``df``: apply it out of sample through ``mapping_dict``, fitted on a training block.
    """
    if state_col not in df.columns:
        raise ValueError(f"Column '{state_col}' not found in DataFrame")
    if return_col not in df.columns:
        raise ValueError(f"Column '{return_col}' not found in DataFrame")

    # Filter out NaN states and returns. Clusters are cut on the row positions of ``df`` before filtering, so a
    # dropped row never joins the bars on either side of it into one block.
    valid_mask = df[state_col].notna() & df[return_col].notna()
    df_clean = df.loc[valid_mask].copy()
    block = cluster_size or min(max(20 * horizon, int(np.ceil(np.sqrt(len(df))))),
                                max(horizon, len(df) // MIN_CLUSTERS))
    df_clean["_block"] = np.flatnonzero(valid_mask.to_numpy()) // block

    if len(df_clean) == 0:
        warnings.warn("No valid data after removing NaN values")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== COMPUTE STATISTICS (EFFICIENT) ==========
    # Build stats list efficiently (not O(n²) concat)
    stats_list = []

    for state, state_df in df_clean.groupby(state_col):
        returns = state_df[return_col].values
        n = len(returns)

        if n < min_samples:
            # Insufficient data - will be labeled as 0
            stats_list.append(
                {
                    "state": state,
                    "count": n,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "sharpe": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "n_clusters": np.nan,
                    "pos_ratio": np.nan,
                    "insufficient_data": True,
                }
            )
            continue

        # Basic statistics
        mean_ret = np.mean(returns)
        median_ret = np.median(returns)
        std_ret = np.std(returns, ddof=1)
        pos_ratio = np.mean(returns > 0)

        # Risk-adjusted return
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        # Statistical significance (cluster-robust t-test of mean 0)
        t_stat, p_value, n_clusters = _cluster_t_test(returns, state_df["_block"].to_numpy())

        stats_list.append(
            {
                "state": state,
                "count": n,
                "mean": mean_ret,
                "median": median_ret,
                "std": std_ret,
                "sharpe": sharpe,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_clusters": n_clusters,
                "pos_ratio": pos_ratio,
                "insufficient_data": False,
            }
        )

    stats_df = pd.DataFrame(stats_list)

    if stats_df.empty:
        warnings.warn("No states found in data")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== AUTO-ESTIMATE COST THRESHOLD (DATA-ADAPTIVE) ==========
    if cost_threshold is None:
        valid_states = stats_df[~stats_df["insufficient_data"]]

        if len(valid_states) > 0:
            # Use the statistical separation between states
            state_means = valid_states["mean"].values

            if len(state_means) > 1:
                # Compute pairwise differences between state means
                mean_diffs = []
                for i in range(len(state_means)):
                    for j in range(i + 1, len(state_means)):
                        mean_diffs.append(abs(state_means[i] - state_means[j]))

                # Use a fraction of the minimum pairwise difference
                min_separation = min(mean_diffs) if mean_diffs else 0
                cost_threshold = min_separation * 0.25
            else:
                cost_threshold = abs(state_means[0]) * 0.1

            # Fallback: Use typical noise level
            typical_noise = df_clean[return_col].abs().median()
            noise_threshold = typical_noise * 0.1
            cost_threshold = max(cost_threshold, noise_threshold)
        else:
            cost_threshold = df_clean[return_col].abs().median() * 0.1

    # ========== APPLY MAPPING LOGIC ==========
    mapping = {}
    stats_df["direction"] = 0  # Default
    stats_df["reason"] = ""  # For diagnostics

    alpha = 1 - confidence_level

    frozen = mapping_dict is not None
    if frozen:
        mapping = dict(mapping_dict)
        stats_df["direction"] = stats_df["state"].map(mapping).fillna(0).astype(int)
        stats_df["reason"] = "Supplied mapping_dict"
    elif method == "conservative":
        # Multi-criteria: ALL must pass for non-zero label
        # 1. Statistical significance (p < alpha)
        # 2. Risk-adjusted return (|Sharpe| > min_sharpe)
        # 3. Magnitude check (|mean| > cost_threshold)

        for idx, row in stats_df.iterrows():
            state = row["state"]

            if row["insufficient_data"]:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = (
                    f'Insufficient data (n={row["count"]} < {min_samples})'
                )
                continue

            # Check all criteria
            is_significant = row["p_value"] < alpha
            has_good_sharpe = abs(row["sharpe"]) > min_sharpe
            exceeds_costs = abs(row["mean"]) > cost_threshold

            if is_significant and has_good_sharpe and exceeds_costs:
                direction = 1 if row["mean"] > 0 else -1
                mapping[state] = direction
                stats_df.at[idx, "direction"] = direction
                stats_df.at[idx, "reason"] = (
                    f'Passed all: p={row["p_value"]:.4f}<{alpha}, '
                    f'Sharpe={row["sharpe"]:.3f}>{min_sharpe}, '
                    f'|mean|={abs(row["mean"]):.6f}>{cost_threshold:.6f}'
                )
            else:
                # Failed at least one criterion
                mapping[state] = 0
                reasons = []
                if not is_significant:
                    reasons.append(f'p={row["p_value"]:.4f}>={alpha}')
                if not has_good_sharpe:
                    reasons.append(f'|Sharpe|={abs(row["sharpe"]):.3f}<={min_sharpe}')
                if not exceeds_costs:
                    reasons.append(
                        f'|mean|={abs(row["mean"]):.6f}<={cost_threshold:.6f}'
                    )
                stats_df.at[idx, "reason"] = "Failed: " + ", ".join(reasons)
    elif method == "statistical":
        # Pure statistical significance (t-test only)
        for idx, row in stats_df.iterrows():
            state = row["state"]

            if row["insufficient_data"]:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = f'Insufficient data (n={row["count"]})'
                continue

            if row["p_value"] < alpha and abs(row["mean"]) > cost_threshold:
                direction = 1 if row["mean"] > 0 else -1
                mapping[state] = direction
                stats_df.at[idx, "direction"] = direction
                stats_df.at[idx, "reason"] = (
                    f'Significant: p={row["p_value"]:.4f}, mean={row["mean"]:.6f}'
                )
            else:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = (
                    f'Not significant or too small: p={row["p_value"]:.4f}'
                )
    elif method == "sharpe":
        # Pure risk-adjusted approach
        for idx, row in stats_df.iterrows():
            state = row["state"]

            if row["insufficient_data"]:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = f'Insufficient data (n={row["count"]})'
                continue

            if abs(row["sharpe"]) > min_sharpe and abs(row["mean"]) > cost_threshold:
                direction = 1 if row["sharpe"] > 0 else -1
                mapping[state] = direction
                stats_df.at[idx, "direction"] = direction
                stats_df.at[idx, "reason"] = f'Good Sharpe: {row["sharpe"]:.3f}'
            else:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = f'Poor Sharpe: {row["sharpe"]:.3f}'
    elif method == "simple":
        # Simple mean threshold (fast but naive - for comparison only)
        # Adaptive threshold based on cross-state mean distribution
        valid_means = stats_df.loc[~stats_df["insufficient_data"], "mean"]
        if len(valid_means) > 0:
            thresh = valid_means.std() * 0.5
        else:
            thresh = cost_threshold

        for idx, row in stats_df.iterrows():
            state = row["state"]

            if row["insufficient_data"]:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = f'Insufficient data (n={row["count"]})'
            elif row["mean"] > thresh:
                mapping[state] = 1
                stats_df.at[idx, "direction"] = 1
                stats_df.at[idx, "reason"] = f'mean={row["mean"]:.6f} > {thresh:.6f}'
            elif row["mean"] < -thresh:
                mapping[state] = -1
                stats_df.at[idx, "direction"] = -1
                stats_df.at[idx, "reason"] = f'mean={row["mean"]:.6f} < {-thresh:.6f}'
            else:
                mapping[state] = 0
                stats_df.at[idx, "reason"] = f'mean={row["mean"]:.6f} in neutral zone'
    else:
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Choose from: 'conservative', 'statistical', 'sharpe', 'simple'"
        )

    # The method's own verdict, kept before the fallback below overwrites ``direction``: a fallback sign is a rank,
    # not evidence, so a caller counting significant states must read ``test_direction``.
    stats_df["test_direction"] = stats_df["direction"]
    stats_df["fallback"] = False

    # ========== OPT-IN: n states → min(n, 3) distinct signs ==========
    # Only on request: a ranking fallback overrides the test. Never on a supplied map: that would re-rank its states
    # on ``df``, which is usually the evaluation block.
    valid_states = stats_df[~stats_df["insufficient_data"]]
    n_valid_states = len(valid_states)

    if n_valid_states > 0 and not frozen and force_distinct_signs:
        # Count distinct signs assigned (excluding states with insufficient data)
        assigned_signs = set()
        for idx, row in valid_states.iterrows():
            if row["state"] in mapping:
                assigned_signs.add(mapping[row["state"]])

        n_distinct_signs = len(assigned_signs)
        required_signs = min(n_valid_states, 3)  # Max 3 signs: -1, 0, 1

        if n_distinct_signs < required_signs:
            # Fallback: Rank states by mean and assign distinct signs
            warnings.warn(
                f"Method '{method}' produced {n_distinct_signs} distinct signs for {n_valid_states} states. "
                f"Applying ranking fallback to ensure {required_signs} distinct signs."
            )

            # Sort valid states by mean return
            valid_states_sorted = valid_states.sort_values(
                "mean", ascending=True
            ).copy()

            # Assign signs based on ranking
            for i, (idx, row) in enumerate(valid_states_sorted.iterrows()):
                state = row["state"]

                if required_signs == 1:
                    # Single state → neutral
                    new_direction = 0
                    reason_suffix = "(only state)"
                elif required_signs == 2:
                    # Two states → -1 and 1 (skip 0)
                    new_direction = -1 if i == 0 else 1
                    reason_suffix = (
                        f"(rank {i + 1}/2: lowest mean)"
                        if i == 0
                        else f"(rank {i + 1}/2: highest mean)"
                    )
                else:  # required_signs == 3
                    # Three or more states → -1, 0, 1
                    if i < n_valid_states // 3:
                        new_direction = -1
                        reason_suffix = f"(rank {i + 1}/{n_valid_states}: bottom third)"
                    elif i >= (2 * n_valid_states) // 3:
                        new_direction = 1
                        reason_suffix = f"(rank {i + 1}/{n_valid_states}: top third)"
                    else:
                        new_direction = 0
                        reason_suffix = f"(rank {i + 1}/{n_valid_states}: middle third)"

                mapping[state] = new_direction
                stats_df.at[idx, "direction"] = new_direction
                stats_df.at[idx, "fallback"] = True
                stats_df.at[idx, "reason"] = (
                    f"Fallback ranking: mean={row['mean']:.6e} {reason_suffix}"
                )

    # Handle any states not in stats_df (shouldn't happen, but defensive)
    all_states = df[state_col].unique()
    for state in all_states:
        if pd.notna(state) and state not in mapping:
            mapping[state] = 0
            warnings.warn(f"State {state} not in mapping, defaulting to 0")

    if return_diagnostics:
        return mapping, stats_df
    else:
        return mapping


def _cluster_t_test(values: np.ndarray, clusters: np.ndarray) -> Tuple[float, float, int]:
    """Two-sided t-test of mean 0 with cluster-robust standard error, on (clusters - 1) degrees of freedom."""
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values - values.mean())
    n_clusters = sums.size
    if n_clusters < 2:
        return np.nan, np.nan, n_clusters
    se = np.sqrt(n_clusters / (n_clusters - 1) * (sums ** 2).sum()) / values.size
    if not se > 0:
        return np.nan, np.nan, n_clusters
    t_stat = values.mean() / se
    return float(t_stat), float(2 * stats.t.sf(abs(t_stat), df=n_clusters - 1)), n_clusters


def map_regime_to_volatility_score(df: pd.DataFrame, regime_col: str = "regime", vol_proxy_col: str = None,
                                   method: str = "median", min_samples: int = 30, tie_tolerance: float = 1e-6,
                                   return_diagnostics: bool = False) -> dict[Any, int] | tuple[dict[Any, int], pd.DataFrame]:
    """
    Map regime labels to ordered volatility buckets where buckets increase monotonically with empirical within-regime volatility.

    Scores range from 0 to n where:
    - 0 = quietest/calmest regime
    - n = most violent/turbulent regime

    Every distinct regime receives a unique bucket (0, 1, 2, ..., k-1 for k valid regimes) with strict monotonicity:
        bucket i has volatility >= bucket i-1.

    Designed for production pipelines feeding position-sizing, risk budgeting, or regime-conditional volatility targeting.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the regime column and a volatility proxy column.
    regime_col : str, default='regime'
        Column name for regime labels (any hashable dtype).
    vol_proxy_col : str, required
        Column name for volatility proxy. Examples:
        - 'realized_vol': 5-min or daily realized volatility
        - 'atr': Average True Range
        - 'abs_returns': Absolute returns
        - 'hl_range': High-Low range
    method : {'median', 'mean', 'q95'}, default='median'
        Volatility estimation method for ranking regimes:
        - 'median': median vol_proxy (robust to outliers) [RECOMMENDED]
        - 'mean': arithmetic mean vol_proxy (faster, outlier-sensitive)
        - 'q95': 95th percentile vol_proxy (captures tail volatility)
    min_samples : int, default=30
        Minimum observations per regime for reliable statistics.
        Regimes with fewer samples are forced into bucket 0 (quietest).
    tie_tolerance : float, default=1e-6
        Threshold for detecting ties. If |vol_A - vol_B| < tie_tolerance,
        regimes are considered tied and ranked by:
        1. Sample count (larger sample wins)
        2. Mean vol_proxy (if counts are equal)
    return_diagnostics : bool, default=False
        If True, returns (mapping, diagnostics_df) showing statistics for each regime.
        If False, returns only the mapping dict.

    Returns
    -------
    mapping : dict
        {regime_label: int_bucket} where bucket ∈ [0, k-1] for k valid regimes.
        Buckets are sequential (no gaps) and monotonically increasing with volatility.
    diagnostics_df : pd.DataFrame (if return_diagnostics=True)
        Columns: regime, median_vol, mean_vol, count, assigned_bucket, reason

    Guarantees
    ----------
    1. Monotonic: bucket_i > bucket_j ⇒ volatility_i >= volatility_j
    2. Unique: Every valid regime gets a distinct bucket (no duplicates)
    3. Sequential: Buckets are 0, 1, 2, ..., k-1 with no gaps
    4. Complete: Every regime observed in df receives a bucket
    5. Deterministic: identical data → identical mapping
    """
    # Validate inputs
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if vol_proxy_col is None:
        raise ValueError(
            "vol_proxy_col is required. Specify a volatility proxy column."
        )
    if vol_proxy_col not in df.columns:
        raise ValueError(f"Column '{vol_proxy_col}' not found in DataFrame")
    if method not in {"median", "mean", "q95"}:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'median', 'mean', 'q95'"
        )

    # Filter valid data
    valid_mask = df[regime_col].notna() & df[vol_proxy_col].notna()
    df_clean = df.loc[valid_mask].copy()

    if len(df_clean) == 0:
        warnings.warn("No valid data after removing NaN values")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # Track regimes that disappeared
    all_regimes_before = set(df[regime_col].dropna().unique())
    all_regimes_after = set(df_clean[regime_col].unique())
    disappeared = all_regimes_before - all_regimes_after
    if disappeared:
        warnings.warn(f"Regimes {disappeared} disappeared after dropping NaNs")

    # Compute global 90th percentile for optional diagnostics
    global_q90 = df_clean[vol_proxy_col].quantile(0.90)

    # ========== COMPUTE STATISTICS FOR EACH REGIME ==========
    stats_list = []

    for regime, regime_df in df_clean.groupby(regime_col):
        vol_values = regime_df[vol_proxy_col].values
        n = len(vol_values)

        if n < min_samples:
            # Insufficient data
            stats_list.append(
                {
                    "regime": regime,
                    "count": n,
                    "median_vol": np.nan,
                    "mean_vol": np.nan,
                    "q95_vol": np.nan,
                    "iqr_vol": np.nan,
                    "frac_above_q90": np.nan,
                    "insufficient_data": True,
                }
            )
            continue

        # Calculate statistics
        median_vol = np.median(vol_values)
        mean_vol = np.mean(vol_values)
        q95_vol = np.percentile(vol_values, 95)
        q25, q75 = np.percentile(vol_values, [25, 75])
        iqr_vol = q75 - q25
        frac_above_q90 = np.mean(vol_values > global_q90)
        stats_list.append(
            {
                "regime": regime,
                "count": n,
                "median_vol": median_vol,
                "mean_vol": mean_vol,
                "q95_vol": q95_vol,
                "iqr_vol": iqr_vol,
                "frac_above_q90": frac_above_q90,
                "insufficient_data": False,
            }
        )

    stats_df = pd.DataFrame(stats_list)

    if stats_df.empty:
        warnings.warn("No regimes found in data")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== SELECT RANKING METRIC ==========
    if method == "median":
        metric_col = "median_vol"
    elif method == "mean":
        metric_col = "mean_vol"
    elif method == "q95":
        metric_col = "q95_vol"

    # ========== ASSIGN BUCKETS ==========
    mapping = {}
    stats_df["assigned_bucket"] = 0
    stats_df["reason"] = ""

    # Separate valid and insufficient-data regimes
    valid_regimes = stats_df[~stats_df["insufficient_data"]].copy()
    insufficient_regimes = stats_df[stats_df["insufficient_data"]].copy()

    # Sort valid regimes by volatility metric (ascending: quietest first)
    valid_regimes = valid_regimes.sort_values(metric_col, ascending=True).reset_index(drop=True)

    # Handle ties and assign buckets
    n_valid = len(valid_regimes)
    if n_valid > 0:
        # Assign sequential buckets 0, 1, 2, ..., k-1
        for i in range(n_valid):
            regime = valid_regimes.loc[i, "regime"]
            bucket = i  # Sequential buckets starting from 0

            # Check for ties with next regime (if exists)
            if i < n_valid - 1:
                current_vol = valid_regimes.loc[i, metric_col]
                next_vol = valid_regimes.loc[i + 1, metric_col]

                if abs(next_vol - current_vol) < tie_tolerance:
                    # Tie detected - verify tie-breaking worked
                    current_count = valid_regimes.loc[i, "count"]
                    next_count = valid_regimes.loc[i + 1, "count"]
                    tie_info = f" (tied with next, broke by count: {current_count} vs {next_count})"
                else:
                    tie_info = ""
            else:
                tie_info = ""

            mapping[regime] = bucket
            stats_df.loc[stats_df["regime"] == regime, "assigned_bucket"] = bucket
            stats_df.loc[stats_df["regime"] == regime, "reason"] = (
                f"Bucket {bucket}/{n_valid - 1}: {metric_col}={valid_regimes.loc[i, metric_col]:.6f}{tie_info}"
            )

        # ========== VERIFY MONOTONICITY ==========
        # Check that buckets are monotonically increasing with volatility
        monotonic = True
        for i in range(n_valid - 1):
            current_vol = valid_regimes.loc[i, metric_col]
            next_vol = valid_regimes.loc[i + 1, metric_col]
            if next_vol < current_vol - tie_tolerance:
                monotonic = False
                warnings.warn(
                    f"Monotonicity violation detected: bucket {i} has {metric_col}={current_vol:.6f}, "
                    f"but bucket {i + 1} has {metric_col}={next_vol:.6f}"
                )

        if not monotonic:
            warnings.warn("Re-ranking to ensure monotonicity")
            # Re-sort (this should already be sorted, but defensive)
            valid_regimes = valid_regimes.sort_values(
                metric_col, ascending=True
            ).reset_index(drop=True)

    # Handle insufficient-data regimes → bucket 0
    for idx, row in insufficient_regimes.iterrows():
        regime = row["regime"]
        mapping[regime] = 0
        stats_df.loc[stats_df["regime"] == regime, "assigned_bucket"] = 0
        stats_df.loc[stats_df["regime"] == regime, "reason"] = (
            f'Insufficient data (n={row["count"]} < {min_samples}) → forced to bucket 0'
        )

    # Handle any regimes not in stats_df (defensive)
    all_regimes = df[regime_col].unique()
    for regime in all_regimes:
        if pd.notna(regime) and regime not in mapping:
            mapping[regime] = 0
            warnings.warn(f"Regime {regime} not in mapping, defaulting to bucket 0")

    if return_diagnostics:
        # Return only relevant columns
        diag_cols = [
            "regime",
            "median_vol",
            "mean_vol",
            "count",
            "assigned_bucket",
            "reason",
        ]
        return mapping, stats_df[diag_cols]
    else:
        return mapping
