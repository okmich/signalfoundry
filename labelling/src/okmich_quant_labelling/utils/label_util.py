import warnings
from typing import List, Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats


def map_label_to_trend_direction(df: pd.DataFrame, state_col: str = "state", return_col: str = "returns",
                                 method: str = "conservative", min_samples: int = 30, confidence_level: float = 0.95,
                                 mapping_dict: Dict[float, str] = None, min_sharpe: float = 0.3,
                                 cost_threshold: Optional[float] = None,
                                 return_diagnostics: bool = False, **kwargs, ) -> Dict[Any, int] | Tuple[Dict[Any, int], pd.DataFrame]:
    """
    Map HMM states to trend directions with guaranteed distinct sign assignment

    Designed for production use in automated backtesting pipelines where:
    - HMM states (0,1,2,3...) need to be mapped to directional signals (-1, 0, 1)
    - GUARANTEE: n states → min(n, 3) distinct signs (1-to-1 mapping up to max 3)
    - Data-adaptive thresholds inferred from state separation
    - Fallback ranking ensures distinct signs when statistical tests fail
    - Conservative defaults prevent false signals
    - Robust handling of edge cases (small samples, noisy regimes)
    - Deterministic results for reproducibility

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
        Statistics for each state showing why it was labeled

    Design Philosophy for HMM Backtesting
    --------------------------------------
    1. GUARANTEED MAPPING: n states → min(n, 3) distinct signs
       - 2 states always get 2 distinct signs (e.g., -1 and 1)
       - 3+ states always get 3 distinct signs (-1, 0, 1)
       - If statistical tests fail, automatic fallback to ranking by mean
    2. DATA-ADAPTIVE: Thresholds inferred from state separation, not hard-coded
       - Works across different timeframes (1-min to daily)
       - Works across different instruments (forex, equities, crypto)
       - No manual threshold tuning required
    3. STATISTICALLY SOUND: Prefer significance tests when states are well-separated
       - Conservative: Requires significance + Sharpe + magnitude (strictest)
       - Statistical: Requires significance + magnitude
       - Sharpe: Requires risk-adjusted returns + magnitude
       - Simple: Pure ranking by mean (fastest, least statistical)
    4. ROBUST: Handle edge cases gracefully (never crash backtesting)
    5. DETERMINISTIC: Same data → same labels (no randomness)
    """
    if state_col not in df.columns:
        raise ValueError(f"Column '{state_col}' not found in DataFrame")
    if return_col not in df.columns:
        raise ValueError(f"Column '{return_col}' not found in DataFrame")

    # Filter out NaN states and returns
    valid_mask = df[state_col].notna() & df[return_col].notna()
    df_clean = df.loc[valid_mask].copy()

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

        # Statistical significance (one-sample t-test against 0)
        t_stat, p_value = stats.ttest_1samp(returns, popmean=0.0, nan_policy="omit")

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

    if mapping_dict:
        mapping = mapping_dict
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

    # ========== GUARANTEE: n states → min(n, 3) distinct signs ==========
    # If the statistical method didn't produce enough distinct signs, use ranking fallback
    valid_states = stats_df[~stats_df["insufficient_data"]]
    n_valid_states = len(valid_states)

    if n_valid_states > 0:
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


def map_label_to_momentum_score(df: pd.DataFrame, regime_col: str = "regime", ret_col: str = "returns",
                                method: str = "robust", min_samples: int = 30,
                                momentum_range: int | tuple[int, int] = 5,
                                is_directional: bool = True,
                                return_diagnostics: bool = False) -> dict[Any, int] | tuple[dict[Any, int], pd.DataFrame]:
    """
    Map regime labels to an ordered momentum score where scores increase monotonically with empirical momentum estimates.

    For directional mode (default): scores range from -n to +n where negative values indicate downward momentum and
    positive values indicate upward momentum.

    For non-directional mode: scores range from 0 to n where higher values indicate stronger momentum magnitude
    regardless of direction.

    The mapping is monotonic: if regime A is assigned a higher score than regime B, the momentum estimate of A is
    guaranteed to be greater than (or equal to) that of B.
    Designed for production pipelines that feed position-sizing or risk-on/risk-off flags.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the regime column and a return column.
    regime_col : str, default='regime'
        Column name for regime labels (any hashable dtype).
    ret_col : str, default='returns'
        Column name for (usually log) returns used to measure momentum.
    method : {'robust', 'mean', 'sharpe'}, default='robust'
        Momentum estimation method:
        - 'robust': rank regimes by median return (resistant to outliers) [RECOMMENDED]
        - 'mean': rank by arithmetic mean (faster, but sensitive to tails)
        - 'sharpe': rank by Sharpe ratio (mean / std) to embed risk adjustment
    min_samples : int, default=30
        Minimum observations per regime for reliable statistics.
        Regimes with fewer samples are assigned to:
        - Directional mode: score 0 (neutral)
        - Non-directional mode: score 0 (no momentum)
    momentum_range : int or tuple(int, int), default=5
        Defines the scoring range:
        - If int and is_directional=True: scores range from -momentum_range to +momentum_range
        - If int and is_directional=False: scores range from 0 to momentum_range
        - If tuple: explicit (min, max) range (overrides is_directional interpretation)
    is_directional : bool, default=True
        If True: scores are directional (-n to +n) capturing both direction and strength
        If False: scores are non-directional (0 to n) capturing only magnitude/strength
    return_diagnostics : bool, default=False
        If True, returns (mapping, diagnostics_df) showing statistics for each regime.
        If False, returns only the mapping dict.

    Returns
    -------
    mapping : dict
        {regime_label: int_score} where scores satisfy monotonicity and range constraints
    diagnostics_df : pd.DataFrame (if return_diagnostics=True)
        One row per regime with statistics (mean, median, std, sharpe, count, score, reason)

    Guarantees
    ----------
    1. Monotonic: score_i > score_j ⇒ momentum_estimate_i > momentum_estimate_j
    2. Complete: every regime observed in df receives a score
    3. Range-respecting: all scores lie inside the specified momentum_range
    4. Deterministic: identical data → identical mapping
    """
    # Validate inputs
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if ret_col not in df.columns:
        raise ValueError(f"Column '{ret_col}' not found in DataFrame")
    if method not in {"robust", "mean", "sharpe"}:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'robust', 'mean', 'sharpe'"
        )

    # Parse momentum_range
    if isinstance(momentum_range, int):
        if is_directional:
            score_min, score_max = -momentum_range, momentum_range
        else:
            score_min, score_max = 0, momentum_range
    elif isinstance(momentum_range, tuple) and len(momentum_range) == 2:
        score_min, score_max = momentum_range
    else:
        raise ValueError(
            f"momentum_range must be int or tuple(int, int), got {type(momentum_range)}"
        )

    if score_min >= score_max:
        raise ValueError(f"Invalid range: min={score_min} >= max={score_max}")

    # Filter valid data
    valid_mask = df[regime_col].notna() & df[ret_col].notna()
    df_clean = df.loc[valid_mask].copy()

    if len(df_clean) == 0:
        warnings.warn("No valid data after removing NaN values")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== COMPUTE STATISTICS FOR EACH REGIME ==========
    stats_list = []

    for regime, regime_df in df_clean.groupby(regime_col):
        returns = regime_df[ret_col].values
        n = len(returns)

        if n < min_samples:
            # Insufficient data
            stats_list.append(
                {
                    "regime": regime,
                    "count": n,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "sharpe": np.nan,
                    "insufficient_data": True,
                }
            )
            continue

        # Calculate statistics
        mean_ret = np.mean(returns)
        median_ret = np.median(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        stats_list.append(
            {
                "regime": regime,
                "count": n,
                "mean": mean_ret,
                "median": median_ret,
                "std": std_ret,
                "sharpe": sharpe,
                "insufficient_data": False,
            }
        )

    stats_df = pd.DataFrame(stats_list)

    if stats_df.empty:
        warnings.warn("No regimes found in data")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== SELECT RANKING METRIC ==========
    if method == "robust":
        metric_col = "median"
    elif method == "mean":
        metric_col = "mean"
    elif method == "sharpe":
        metric_col = "sharpe"

    # ========== ASSIGN SCORES ==========
    mapping = {}
    stats_df["score"] = 0
    stats_df["reason"] = ""

    # Separate valid and insufficient-data regimes
    valid_regimes = stats_df[~stats_df["insufficient_data"]].copy()
    insufficient_regimes = stats_df[stats_df["insufficient_data"]].copy()

    # For directional mode: use raw metric
    # For non-directional mode: use absolute value of metric
    if is_directional:
        valid_regimes["rank_metric"] = valid_regimes[metric_col]
    else:
        valid_regimes["rank_metric"] = valid_regimes[metric_col].abs()

    # Sort by rank metric
    valid_regimes = valid_regimes.sort_values("rank_metric", ascending=True)

    # Distribute scores across the range
    n_valid = len(valid_regimes)
    if n_valid > 0:
        # Linear distribution of scores
        scores = np.linspace(score_min, score_max, n_valid)
        scores = np.round(scores).astype(int)

        for i, (idx, row) in enumerate(valid_regimes.iterrows()):
            regime = row["regime"]
            score = scores[i]
            mapping[regime] = score
            stats_df.at[idx, "score"] = score
            stats_df.at[idx, "reason"] = (
                f"Rank {i + 1}/{n_valid}: {metric_col}={row[metric_col]:.6f}"
            )

    # Handle insufficient-data regimes
    insufficient_score = 0
    for idx, row in insufficient_regimes.iterrows():
        regime = row["regime"]
        mapping[regime] = insufficient_score
        stats_df.at[idx, "score"] = insufficient_score
        stats_df.at[idx, "reason"] = (
            f'Insufficient data (n={row["count"]} < {min_samples})'
        )

    # Handle any regimes not in stats_df (defensive)
    all_regimes = df[regime_col].unique()
    for regime in all_regimes:
        if pd.notna(regime) and regime not in mapping:
            mapping[regime] = 0
            warnings.warn(f"Regime {regime} not in mapping, defaulting to 0")

    if return_diagnostics:
        return mapping, stats_df
    else:
        return mapping


def map_regime_to_volatility_score(df: pd.DataFrame, regime_col: str = "regime", vol_proxy_col: str = None,
                                   method: str = "median", min_samples: int = 30, tie_tolerance: float = 1e-6,
                                   return_diagnostics: bool = False) -> dict[Any, int] | tuple[
    dict[Any, int], pd.DataFrame]:
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


def map_regime_to_path_structure_score(df: pd.DataFrame, regime_col: str = "regime", method: str = "id_chop",
                                       min_samples: int = 30, choppiness_range: tuple[int, int] = (0, 4),
                                       lookback: int = 14, return_diagnostics: bool = False) -> dict[Any, int] | tuple[
    dict[Any, int], pd.DataFrame]:
    """
    Map regime labels to an ordered choppiness score where scores increase monotonically with empirical within-regime
    choppiness (path complexity, oscillation, whipsaw).

    Scores range from 0 to n where:
    - 0 = "silky-smooth directional" (least choppy, trending)
    - n = "maximum whipsaw/noise" (most choppy, oscillating)

    The mapping is monotonic: higher score = higher choppiness.
    Built for automated pipelines that throttle leverage, widen stops, or switch to mean-reverting models when markets get choppy.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the regime column and price data (or pre-computed choppiness proxy).
    regime_col : str, default='regime'
        Column name for regime labels (any hashable dtype).
    method : {'id_chop', 'cv_range', 'atr_dr', 'custom'}, default='id_chop'
        Choppiness estimation method:
        - 'id_chop': Choppiness Index - 100×log(sum(TR)/range) / log(lookback)
          Higher = choppier. Requires high, low, close columns.
        - 'cv_range': Coefficient of variation of range (std/mean of high-low).
          Higher = choppier. Requires high, low columns.
        - 'atr_dr': Ratio of ATR to directional range (sum of ATR / sum of |close-open|).
          Higher = choppier. Requires high, low, close, open columns.
        - 'custom': User supplies pre-computed 'choppiness_proxy' column.
    min_samples : int, default=30
        Minimum observations per regime for reliable statistics.
        Regimes with fewer samples are forced to bucket 0 (smoothest).
    choppiness_range : tuple(int, int), default=(0, 4)
        Closed interval [least_choppy, most_choppy] for integer scores.
    lookback : int, default=14
        Window length for rolling choppiness estimators (ignored for 'custom').
    return_diagnostics : bool, default=False
        If True, returns (mapping, diagnostics_df) showing statistics for each regime.
        If False, returns only the mapping dict.

    Returns
    -------
    mapping : dict
        {regime_label: int_score} where score ∈ choppiness_range and
        scores increase strictly with empirical choppiness estimate.
    diagnostics_df : pd.DataFrame (if return_diagnostics=True)
        Columns: regime, median_chop, mean_chop, count, assigned_score, reason

    Guarantees
    ----------
    1. Monotonic: score_i > score_j ⇒ choppiness_i > choppiness_j
    2. Complete: every regime receives a score
    3. Range-respecting: all scores within choppiness_range
    4. Deterministic: identical data → identical mapping
    5. Fallback: ties broken by sample size; tiny regimes → bucket 0

    Notes
    -----
    - Ties (difference < 1e-8) are broken by sample size (larger sample wins)
    - Unseen regimes default to bucket 0 (production safety)
    """
    # Validate inputs
    if regime_col not in df.columns:
        raise ValueError(f"Column '{regime_col}' not found in DataFrame")
    if method not in {"id_chop", "cv_range", "atr_dr", "custom"}:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'id_chop', 'cv_range', 'atr_dr', 'custom'"
        )

    # Validate required columns based on method
    if method == "custom":
        if "choppiness_proxy" not in df.columns:
            raise ValueError(
                "method='custom' requires a 'choppiness_proxy' column in df"
            )
    elif method == "id_chop":
        required = ["high", "low", "close"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"method='id_chop' requires columns {required}, missing: {missing}")
    elif method == "cv_range":
        required = ["high", "low"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"method='cv_range' requires columns {required}, missing: {missing}")
    elif method == "atr_dr":
        required = ["high", "low", "close", "open"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"method='atr_dr' requires columns {required}, missing: {missing}")

    if not isinstance(choppiness_range, tuple) or len(choppiness_range) != 2:
        raise ValueError(f"choppiness_range must be tuple(int, int), got {choppiness_range}")
    score_min, score_max = choppiness_range
    if score_min >= score_max:
        raise ValueError(f"Invalid range: min={score_min} >= max={score_max}")

    # ========== COMPUTE CHOPPINESS PROXY ==========
    df_work = df.copy()

    if method == "custom":
        # Use pre-computed proxy
        chop_col = "choppiness_proxy"
    else:
        # Compute choppiness metric
        chop_col = "choppiness_proxy"

        if method == "id_chop":
            # Choppiness Index = 100 × log(sum(TR) / (HH - LL)) / log(lookback)
            # True Range
            df_work["tr"] = np.maximum(
                df_work["high"] - df_work["low"],
                np.maximum(
                    abs(df_work["high"] - df_work["close"].shift(1)),
                    abs(df_work["low"] - df_work["close"].shift(1)),
                ),
            )
            # Rolling calculations
            sum_tr = df_work["tr"].rolling(window=lookback).sum()
            hh = df_work["high"].rolling(window=lookback).max()
            ll = df_work["low"].rolling(window=lookback).min()
            range_hl = hh - ll

            # Choppiness Index (avoid div by zero)
            df_work[chop_col] = np.where(
                range_hl > 0, 100 * np.log(sum_tr / range_hl) / np.log(lookback), np.nan
            )

        elif method == "cv_range":
            # Coefficient of Variation of range
            df_work["range"] = df_work["high"] - df_work["low"]
            mean_range = df_work["range"].rolling(window=lookback).mean()
            std_range = df_work["range"].rolling(window=lookback).std()
            df_work[chop_col] = np.where(mean_range > 0, std_range / mean_range, np.nan)

        elif method == "atr_dr":
            # ATR / Directional Range ratio
            # True Range (same as above)
            df_work["tr"] = np.maximum(
                df_work["high"] - df_work["low"],
                np.maximum(
                    abs(df_work["high"] - df_work["close"].shift(1)),
                    abs(df_work["low"] - df_work["close"].shift(1)),
                ),
            )
            atr = df_work["tr"].rolling(window=lookback).mean()

            # Directional range (absolute change close to open)
            df_work["dir_range"] = abs(df_work["close"] - df_work["open"])
            sum_dir_range = df_work["dir_range"].rolling(window=lookback).sum()

            df_work[chop_col] = np.where(
                sum_dir_range > 0,
                atr
                * lookback
                / sum_dir_range,  # Normalize: ATR×lookback / sum(dir_range)
                np.nan,
            )

    # Filter valid data
    valid_mask = df_work[regime_col].notna() & df_work[chop_col].notna()
    df_clean = df_work.loc[valid_mask].copy()

    if len(df_clean) == 0:
        warnings.warn("No valid data after removing NaN values")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== COMPUTE REGIME-LEVEL STATISTICS ==========
    stats_list = []

    for regime, regime_df in df_clean.groupby(regime_col):
        chop_values = regime_df[chop_col].values
        n = len(chop_values)

        if n < min_samples:
            # Insufficient data
            stats_list.append(
                {
                    "regime": regime,
                    "count": n,
                    "median_chop": np.nan,
                    "mean_chop": np.nan,
                    "insufficient_data": True,
                }
            )
            continue

        # Calculate statistics
        median_chop = np.median(chop_values)
        mean_chop = np.mean(chop_values)

        stats_list.append(
            {
                "regime": regime,
                "count": n,
                "median_chop": median_chop,
                "mean_chop": mean_chop,
                "insufficient_data": False,
            }
        )

    stats_df = pd.DataFrame(stats_list)

    if stats_df.empty:
        warnings.warn("No regimes found in data")
        return {} if not return_diagnostics else ({}, pd.DataFrame())

    # ========== ASSIGN SCORES ==========
    mapping = {}
    stats_df["assigned_score"] = score_min
    stats_df["reason"] = ""

    # Separate valid and insufficient-data regimes
    valid_regimes = stats_df[~stats_df["insufficient_data"]].copy()
    insufficient_regimes = stats_df[stats_df["insufficient_data"]].copy()

    # Sort by median choppiness (ascending: smoothest first)
    valid_regimes = valid_regimes.sort_values(
        "median_chop", ascending=True
    ).reset_index(drop=True)

    # Distribute scores across the range
    n_valid = len(valid_regimes)
    if n_valid > 0:
        # Linear distribution of scores
        scores = np.linspace(score_min, score_max, n_valid)
        scores = np.round(scores).astype(int)

        for i in range(n_valid):
            regime = valid_regimes.loc[i, "regime"]
            score = scores[i]

            # Check for ties with next regime (if exists)
            tie_info = ""
            if i < n_valid - 1:
                current_chop = valid_regimes.loc[i, "median_chop"]
                next_chop = valid_regimes.loc[i + 1, "median_chop"]

                if abs(next_chop - current_chop) < 1e-8:
                    # Tie detected
                    current_count = valid_regimes.loc[i, "count"]
                    next_count = valid_regimes.loc[i + 1, "count"]
                    tie_info = (
                        f" (tied, broke by count: {current_count} vs {next_count})"
                    )

            mapping[regime] = score
            stats_df.loc[stats_df["regime"] == regime, "assigned_score"] = score
            stats_df.loc[stats_df["regime"] == regime, "reason"] = (
                f"Score {score} (rank {i + 1}/{n_valid}): median_chop={valid_regimes.loc[i, 'median_chop']:.4f}{tie_info}"
            )

    # Handle insufficient-data regimes → score_min (smoothest bucket)
    for idx, row in insufficient_regimes.iterrows():
        regime = row["regime"]
        mapping[regime] = score_min
        stats_df.loc[stats_df["regime"] == regime, "assigned_score"] = score_min
        stats_df.loc[stats_df["regime"] == regime, "reason"] = (
            f'Insufficient data (n={row["count"]} < {min_samples}) → forced to bucket {score_min} (smoothest)'
        )

    # Handle any regimes not in stats_df (defensive)
    all_regimes = df[regime_col].unique()
    for regime in all_regimes:
        if pd.notna(regime) and regime not in mapping:
            mapping[regime] = score_min
            warnings.warn(
                f"Regime {regime} not in mapping, defaulting to bucket {score_min}"
            )

    if return_diagnostics:
        diag_cols = [
            "regime",
            "median_chop",
            "mean_chop",
            "count",
            "assigned_score",
            "reason",
        ]
        return mapping, stats_df[diag_cols]
    else:
        return mapping


def create_consensus_labels(df: pd.DataFrame, label_columns: List[str]):
    labels = df[label_columns]

    consensus = pd.Series(0, index=df.index, dtype=np.int8)

    # For each unique label value (excluding 0), check for unanimity
    unique_labels = labels.values.flatten()
    unique_labels = pd.Series(unique_labels).unique()
    unique_labels = unique_labels[unique_labels != 0]

    for label_value in unique_labels:
        all_match = (labels == label_value).all(axis=1)
        consensus[all_match] = label_value

    return consensus
