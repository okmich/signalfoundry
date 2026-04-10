from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress

from .label_util import map_label_to_trend_direction

"""
Path Structure Regime Classification - Evaluation Metrics Documentation

This module provides comprehensive evaluation metrics for assessing the quality
and economic meaningfulness of market regime classifications.

REGIME_EVALUATION_METRICS = {
    # =========================================================================
    # CORE DISCRIMINABILITY METRICS
    # =========================================================================
    'regime_discriminability': {
        'description': 'Composite score measuring how well-separated regimes are',
        'calculation': 'Weighted combination of volatility, efficiency, persistence differences',
        'interpretation': {
            'excellent': '> 0.8 - Regimes are clearly distinct',
            'good': '0.6 - 0.8 - Moderate separation',
            'poor': '< 0.5 - Regimes are too similar',
            'warning': 'Identical scores across algorithms indicate arbitrary splitting'
        },
        'ideal_pattern': 'Consistent high scores across all regimes in a model'
    },

    'efficiency_ratio': {
        'description': 'Quality of price movement within regime',
        'calculation': 'abs(total_return) / sum(abs_returns)',
        'interpretation': {
            'trending': '> 0.3 - Clean directional moves',
            'grinding': '0.1 - 0.3 - Noisy but directional',
            'choppy': '< 0.1 - Mean-reverting, whipsaw action',
            'perfect_trend': '1.0 - Straight line moves',
            'pure_noise': '0.0 - Complete randomness'
        },
        'ideal_pattern': 'Clear separation between high/low efficiency regimes'
    },

    # =========================================================================
    # VOLATILITY & RISK METRICS
    # =========================================================================
    'volatility': {
        'description': 'Standard deviation of returns within regime',
        'interpretation': {
            'high_vol': '> 1.2 - High energy periods (trends OR chop)',
            'medium_vol': '0.8 - 1.2 - Normal market activity',
            'low_vol': '< 0.8 - Quiet, compressed markets'
        },
        'ideal_pattern': 'Clear volatility clustering across regimes',
        'warning': 'All regimes with similar volatility indicates poor separation'
    },

    'volatility_to_mean_ratio': {
        'description': 'Risk-adjusted return potential',
        'calculation': 'volatility / mean_abs_return',
        'interpretation': {
            'efficient': '1.2 - 1.3 - Healthy risk/return',
            'inefficient': '> 1.4 - Too much noise for direction',
            'warning': 'Extreme values indicate unreliable regimes'
        }
    },

    'max_drawdown': {
        'description': 'Maximum peak-to-trough decline within regime',
        'interpretation': 'Lower values indicate more stable regimes',
        'warning': 'Extreme values (>10) may indicate regime miscalssification'
    },

    # =========================================================================
    # REGIME PERSISTENCE METRICS
    # =========================================================================
    'mean_duration': {
        'description': 'Average number of periods regime persists',
        'interpretation': {
            'short_term': '1-5 periods - Noise, transient states',
            'medium_term': '5-20 periods - Meaningful regimes',
            'long_term': '>20 periods - Structural market states'
        },
        'ideal_pattern': 'Realistic durations (not too sticky, not too jumpy)'
    },

    'regime_stability': {
        'description': 'Consistency of regime durations',
        'calculation': 'mean_duration / duration_std',
        'interpretation': {
            'stable': '> 1.0 - Consistent regime lengths',
            'unstable': '< 0.5 - Erratic regime changes'
        }
    },

    # =========================================================================
    # PATH STRUCTURE VALIDATION METRICS
    # =========================================================================
    'autocorrelation_lag1': {
        'description': 'Serial correlation of returns (momentum vs mean-reversion)',
        'interpretation': {
            'momentum': '> 0.1 - Positive persistence',
            'mean_reversion': '< -0.1 - Negative persistence',
            'random_walk': '-0.1 to 0.1 - No clear pattern'
        },
        'ideal_pattern': 'Varies meaningfully across regimes'
    },

    'trend_persistence': {
        'description': 'R-squared of cumulative returns trend',
        'interpretation': {
            'strong_trend': '> 0.3 - Consistent directional movement',
            'weak_trend': '0.1 - 0.3 - Some directionality',
            'no_trend': '< 0.1 - Directionless movement'
        }
    },

    'return_consistency': {
        'description': 'Stability of return magnitudes',
        'calculation': 'std(abs_returns) / mean(abs_returns)',
        'interpretation': {
            'consistent': '< 0.8 - Stable return patterns',
            'erratic': '> 1.2 - Unstable, spiky returns'
        }
    },

    # =========================================================================
    # REGIME FREQUENCY METRICS
    # =========================================================================
    'frequency_pct': {
        'description': 'Percentage of time spent in each regime',
        'interpretation': {
            'common': '25-50% - Healthy frequency',
            'rare': '< 10% - Possibly noise or rare events',
            'dominant': '> 60% - One regime may be too broad'
        },
        'ideal_pattern': 'Balanced distribution across regimes'
    },

    'n_observations': {
        'description': 'Absolute count of observations in regime',
        'warning': '< 100 observations may indicate unreliable statistics'
    }
}

# =============================================================================
# ALGORITHM COMPARISON FRAMEWORK
# =============================================================================
ALGORITHM_EVALUATION_CRITERIA = {
    'discriminability': {
        'weight': 0.3,
        'calculation': 'mean(regime_discriminability)',
        'description': 'Overall regime separation quality'
    },

    'efficiency_separation': {
        'weight': 0.25,
        'calculation': 'max(efficiency_ratio) - min(efficiency_ratio)',
        'description': 'Ability to distinguish trending vs choppy markets'
    },

    'volatility_clustering': {
        'weight': 0.2,
        'calculation': '1 - (volatility_std / volatility_mean)',
        'description': 'Clear separation of high/low volatility periods'
    },

    'regime_persistence': {
        'weight': 0.15,
        'calculation': 'mean(regime_stability)',
        'description': 'Realistic and stable regime durations'
    },

    'frequency_balance': {
        'weight': 0.1,
        'calculation': '1 - abs(max(frequency_pct) - 33.3) / 33.3',
        'description': 'Reasonable time spent in each regime'
    }
}

# =============================================================================
# INTERPRETATION GUIDELINES
# =============================================================================
SUCCESS PATTERNS:
✅ Clear trending regime: High efficiency (>0.3), good discriminability (>0.8)
✅ Clear choppy regime: Low efficiency (<0.1), high volatility, good discriminability  
✅ Volatility regimes: Clear high/medium/low volatility separation
✅ Realistic durations: 5-20 periods average, good stability (>1.0)

FAILURE PATTERNS:
❌ All regimes similar: Near-identical metrics across regimes
❌ Poor discriminability: Scores <0.5 indicate arbitrary splitting
❌ Unrealistic durations: Avg duration <2 (too jumpy) or >50 (too sticky)
❌ Extreme frequencies: One regime dominates (>60%) or is too rare (<5%)

ASSET-CLASS EXPECTATIONS:
• Equities: Often show clear trend/choppy/transition regimes  
• FX: More volatility-driven regimes, lower efficiency overall
• Gold: Hybrid - can trend like equities but with FX-like volatility regimes
"""


def trend_label_statistics(df, state_col="label", return_col="return"):
    """
    Calculate comprehensive statistics for each label state.

    Enhanced version with additional metrics:
    - Return statistics (sum, mean, std, percentiles)
    - Run length statistics (persistence periods)
    - Transition probabilities (what follows each state)
    - Drawdown metrics (worst streaks within each state)
    """
    # Basic return statistics
    stats = (
        df.groupby(state_col)[return_col]
        .agg(
            [
                "sum",
                "count",
                "min",
                "max",
                "mean",
                "std",
                ("median", "median"),
                ("q25", lambda x: x.quantile(0.25)),
                ("q75", lambda x: x.quantile(0.75)),
                ("skew", lambda x: x.skew()),
                (
                    "win_rate",
                    lambda x: (x > 0).mean() if state_col != 0 else (x != 0).mean(),
                ),
            ]
        )
        .reset_index()
    )

    # Run length statistics (persistence/persistence)
    runs = df[state_col].ne(df[state_col].shift()).cumsum().rename("run_id")

    df_runs = df.assign(run_id=runs)

    run_lens = df_runs.groupby([state_col, "run_id"]).size().reset_index(name="run_len")

    run_stats = (
        run_lens.groupby(state_col)["run_len"]
        .agg(
            [
                "min",
                ("q25", lambda x: x.quantile(0.25)),
                "median",
                "mean",
                "std",
                ("q75", lambda x: x.quantile(0.75)),
                "max",
            ]
        )
        .rename(
            columns={
                "min": "min_persistence",
                "q25": "q25_persistence",
                "median": "median_persistence",
                "mean": "avg_persistence",
                "std": "std_persistence",
                "max": "max_persistence",
                "q75": "q75_persistence",
            }
        )
        .reset_index()
    )

    # Cumulative returns within each run (detect best/worst streaks)
    run_returns = (
        df_runs.groupby([state_col, "run_id"])[return_col]
        .sum()
        .reset_index(name="run_return")
    )

    run_return_stats = (
        run_returns.groupby(state_col)["run_return"]
        .agg(
            [
                ("best_run", "max"),
                ("worst_run", "min"),
                ("avg_run_return", "mean"),
                ("std_run_return", "std"),
            ]
        )
        .reset_index()
    )

    df_transitions = df.copy()
    df_transitions["next_state"] = df_transitions[state_col].shift(-1)

    transition_matrix = pd.crosstab(
        df_transitions[state_col], df_transitions["next_state"], normalize="index"
    )

    persistence_prob = {}
    for state in df[state_col].unique():
        if state in transition_matrix.index and state in transition_matrix.columns:
            persistence_prob[state] = transition_matrix.loc[state, state]
        else:
            persistence_prob[state] = 0.0

    persistence_df = pd.DataFrame(
        [
            {"label": k, "persistence_probability": v}
            for k, v in persistence_prob.items()
        ]
    )

    # Merge all stats
    result = (
        stats.merge(run_stats, on=state_col, how="left")
        .merge(run_return_stats, on=state_col, how="left")
        .merge(persistence_df, left_on=state_col, right_on="label", how="left")
    )

    return result


def label_path_structure_statistics(df: pd.DataFrame, state_col: str, returns_col: str = "returns", price_col: str = "close",
                                    sort_by_col: str = "frequency_pct", min_regime_samples: int = 5) -> pd.DataFrame:
    """
    Evaluate regime labels by calculating comprehensive statistics for each regime state.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing regime labels and market data
    state_col : str
        Name of the column containing regime labels
    returns_col : str
        Name of the column containing returns
    price_col : str
        Name of the column containing prices
    sort_by_col : str
        Name of the column with which we should sort the output dataframe
    min_regime_samples : int
        Minimum number of samples required to analyze a regime

    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame with regime states as rows and evaluation metrics as columns
    """
    required_cols = [state_col, returns_col, price_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if df[state_col].isna().all():
        raise ValueError(f"Regime column '{state_col}' contains only NaN values")

    # Filter out NaN regimes
    valid_data = df.dropna(subset=[state_col]).copy()
    if len(valid_data) == 0:
        raise ValueError("No valid regime labels found after removing NaN values")

    regime_stats = {}
    unique_regimes = valid_data[state_col].unique()

    for regime in unique_regimes:
        regime_mask = valid_data[state_col] == regime
        regime_data = valid_data[regime_mask]

        if len(regime_data) < min_regime_samples:
            continue

        returns = regime_data[returns_col]
        prices = regime_data[price_col]

        # Basic regime properties
        stats_dict = {
            "regime": regime,
            "n_observations": len(regime_data),
            "frequency_pct": round(len(regime_data) / len(valid_data), 4) * 100,
        }

        # Returns characteristics (agnostic to direction)
        stats_dict.update(_calculate_return_stats(returns))

        # Volatility and risk metrics
        stats_dict.update(_calculate_volatility_stats(returns, prices))

        # Regime persistence and transitions
        stats_dict.update(_calculate_persistence_stats(valid_data, state_col, regime))

        # Path structure validation metrics
        stats_dict.update(_calculate_path_validation_stats(returns, prices))

        regime_stats[regime] = stats_dict

    if not regime_stats:
        raise ValueError("No regimes met the minimum sample requirement")

    # Create results DataFrame
    stats_df = pd.DataFrame.from_dict(regime_stats, orient="index")
    stats_df["regime_discriminability"] = _calculate_discriminability_score(stats_df)

    # Sort by frequency for better readability
    stats_df = stats_df.sort_values(sort_by_col, ascending=False)
    return stats_df.reset_index(drop=True)


def all_labels_path_structure_statistics(df: pd.DataFrame, state_cols: List[str], returns_col="return",
                                         price_col="close"):
    result_df = None
    if returns_col not in df.columns:
        if price_col not in df.columns:
            raise ValueError(f"Missing price column: {price_col}")
        else:
            df[returns_col] = np.log(df[price_col] / df[price_col].shift(1))

    for col in state_cols:
        stats_df = label_path_structure_statistics(
            df, state_col=col, returns_col=returns_col, price_col=price_col
        )
        stats_df.insert(0, "algo", col)
        result_df = stats_df if result_df is None else pd.concat([result_df, stats_df])
    return result_df


def _calculate_return_stats(returns: pd.Series) -> Dict[str, float]:
    """Calculate return-based statistics for a regime"""
    if len(returns) == 0:
        return {}

    abs_returns = returns.abs()
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    return {
        "mean_return": returns.mean(),
        "median_return": returns.median(),
        "mean_abs_return": abs_returns.mean(),
        "median_abs_return": abs_returns.median(),
        "return_skewness": returns.skew(),
        "return_kurtosis": returns.kurtosis(),
        "positive_return_ratio": (
            len(positive_returns) / len(returns) if len(returns) > 0 else 0
        ),
        "negative_return_ratio": (
            len(negative_returns) / len(returns) if len(returns) > 0 else 0
        ),
        "max_return": returns.max(),
        "min_return": returns.min(),
        "return_consistency": (
            abs_returns.std() / abs_returns.mean()
            if abs_returns.mean() != 0
            else np.nan
        ),
    }


def _calculate_volatility_stats(returns: pd.Series, prices: pd.Series) -> Dict[str, float]:
    """Calculate volatility and risk metrics"""
    if len(returns) < 2:
        return {}

    vol = returns.std()
    abs_returns = returns.abs()

    # Drawdown analysis (using prices)
    if len(prices) > 1:
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
    else:
        max_drawdown = np.nan
        avg_drawdown = np.nan

    return {
        "volatility": vol,
        "volatility_stability": (
            returns.rolling(min(5, len(returns))).std().std() / vol
            if vol != 0
            else np.nan
        ),
        "volatility_to_mean_ratio": (
            vol / abs_returns.mean() if abs_returns.mean() != 0 else np.nan
        ),
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
    }


def _calculate_persistence_stats(df: pd.DataFrame, regime_col: str, regime: Any) -> Dict[str, float]:
    """Calculate regime persistence and transition statistics"""
    regime_series = df[regime_col]
    regime_changes = (regime_series == regime).astype(int)

    # Find regime periods
    regime_periods = _find_contiguous_regimes(regime_series, regime)

    if not regime_periods:
        return {
            "min_duration": 0,
            "q5_duration": 0,
            "q10_duration": 0,
            "q25_duration": 0,
            "mean_duration": 0,
            "median_duration": 0,
            "q75_duration": 0,
            "max_duration": 0,
            "duration_std": 0,
            "regime_stability": 0,
        }

    durations = [period["length"] for period in regime_periods]

    return {
        "min_duration": np.min(durations),
        "q5_duration": np.quantile(durations, 0.05),
        "q10_duration": np.quantile(durations, 0.1),
        "q25_duration": np.quantile(durations, 0.25),
        "mean_duration": np.round(np.mean(durations), 0),
        "median_duration": np.median(durations),
        "q75_duration": np.quantile(durations, 0.75),
        "max_duration": np.max(durations),
        "duration_std": np.round(np.std(durations), 1),
        "regime_stability": (
            np.mean(durations) / np.std(durations)
            if np.std(durations) > 0
            else np.mean(durations)
        ),
    }


def _find_contiguous_regimes(regime_series: pd.Series, regime: Any) -> list:
    """Find contiguous blocks of a specific regime"""
    regimes = regime_series.values
    is_regime = regimes == regime

    periods = []
    i = 0
    while i < len(is_regime):
        if is_regime[i]:
            start = i
            while i < len(is_regime) and is_regime[i]:
                i += 1
            end = i - 1
            periods.append({"start": start, "end": end, "length": end - start + 1})
        else:
            i += 1

    return periods


def _calculate_path_validation_stats(returns: pd.Series, prices: pd.Series) -> Dict[str, float]:
    """Calculate path-specific validation metrics"""
    if len(returns) < 5:
        return {}

    # Serial correlation (should vary by regime)
    acf1 = returns.autocorr(lag=1)

    # Hurst-like metrics (simplified)
    if len(returns) >= 10:
        # Simple trend persistence metric
        cumulative = (1 + returns).cumprod()
        if len(cumulative) > 1:
            trend_strength = abs(
                linregress(np.arange(len(cumulative)), cumulative).rvalue
            )
        else:
            trend_strength = np.nan
    else:
        trend_strength = np.nan

    return {
        "autocorrelation_lag1": acf1 if not np.isnan(acf1) else 0,
        "trend_persistence": trend_strength,
        "noise_to_signal_ratio": (
            returns.std() / returns.abs().mean()
            if returns.abs().mean() != 0
            else np.nan
        ),
        "efficiency_ratio": (
            abs(returns.sum()) / returns.abs().sum() if returns.abs().sum() != 0 else 0
        ),
    }


def _calculate_discriminability_score(stats_df: pd.DataFrame) -> pd.Series:
    """Calculate how discriminable each regime is from others"""
    discriminability_scores = []

    for idx, regime_row in stats_df.iterrows():
        score = 0

        # Volatility discriminability
        vol_diff = (
                abs(regime_row["volatility"] - stats_df["volatility"].mean())
                / stats_df["volatility"].std()
        )

        # Return pattern discriminability
        return_consistency_diff = (
                abs(
                    regime_row["return_consistency"] - stats_df["return_consistency"].mean()
                )
                / stats_df["return_consistency"].std()
        )

        # Persistence discriminability
        duration_diff = (
                abs(regime_row["mean_duration"] - stats_df["mean_duration"].mean())
                / stats_df["mean_duration"].std()
        )

        # Path structure discriminability
        autocorr_diff = (
                abs(
                    regime_row["autocorrelation_lag1"]
                    - stats_df["autocorrelation_lag1"].mean()
                )
                / stats_df["autocorrelation_lag1"].std()
        )

        # Composite score (equal weighting)
        score = (vol_diff + return_consistency_diff + duration_diff + autocorr_diff) / 4
        discriminability_scores.append(score)

    return pd.Series(discriminability_scores, index=stats_df.index)


def evaluate_regime_returns_potentials(df, label_col="label", progressive_skip: int = 1, whipsaw_cost: float = 0.0,
                                       wrong_direction_penalty: float = 0.0, label_sign_map: Dict[Any, int] = None,
                                       include_overall: bool = True):
    """
    Evaluate regime classification based on persistent returns with realistic trading constraints.

    This function assesses how well regime labels predict directional movements by:
    1. Calculating signed returns (label_sign * log_return)
    2. Grouping returns into contiguous regimes
    3. Applying realistic trading penalties (entry lag, transaction costs, wrong predictions)
    4. Computing statistics on persistent returns per regime

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: open, high, low, close, and label column
        - open, high, low, close: OHLC price data
        - label column: regime classification (e.g., -1=bearish, 0=flat, 1=bullish)

    label_col : str, default='label'
        Name of the column containing regime labels

    progressive_skip : int, default=1
        Number of periods to skip at the start of each new regime.
        Simulates realistic entry lag - positions are opened AFTER detecting regime change.
        - 1 = skip first period (standard practice)
        - 2-3 = more conservative, accounts for confirmation delays
        - 0 = no skip (unrealistic, assumes instantaneous entry)

    whipsaw_cost : float, default=0.0
        Transaction cost deducted per regime change (as fraction/percentage).
        Penalizes frequent regime flips. Typical values: 0.001-0.01 (0.1%-1%)

    wrong_direction_penalty : float, default=0.0
        Additional penalty multiplier applied when signed_return is negative.
        Example: 0.5 means losing trades cost 1.5x (1.0 + 0.5)
        Useful to emphasize minimizing wrong predictions.

    label_sign_map : dict, optional
        Custom mapping from label values to directional signs {label: sign}.
        - If None, defaults to {-1: -1, 0: 0, 1: 1}
        - Example: {0: -1, 1: 0, 2: 1} for labels [0, 1, 2]
        - Allows flexibility in label encoding schemes

    Returns:
    --------
    metrics_df : pd.DataFrame
        DataFrame indexed by label values, with an 'overall' row for aggregate stats.
        Columns include:

        OVERALL PERFORMANCE:
        - total_return: Sum of all persistent returns (primary metric)
        - count: Number of regime periods identified
        - mean/median: Central tendency of persistent returns
        - std: Volatility of persistent returns
        - min/max: Range of persistent returns
        - 25p/75: Quartile boundaries
        - skew: Asymmetry (>0 = more large gains, <0 = more large losses)
        - kurtosis: Tail heaviness (>0 = more outliers)

        WIN/LOSS ANALYSIS:
        - n_wins: Count of profitable regimes
        - n_losses: Count of losing regimes
        - win_rate: Fraction of profitable regimes (target: >0.5)
        - profit_factor: Total gains / Total losses (target: >1.5)
        - sharpe_ratio: Risk-adjusted return (target: >1.0)

        REGIME STABILITY:
        - avg_dwell: Average periods per regime (higher = more stable)
        - persistence_score: avg_dwell * win_rate (combined quality metric)
        - regime_purity: Avg fraction of correct-direction returns within regimes (target: >0.6, measures within-regime consistency)

        DRAWDOWN ANALYSIS:
        - dd_min/max/mean/median: Drawdown statistics per regime
        - dd_25p/75: Drawdown quartiles
        - dd_std: Drawdown volatility

        MARKET EXPOSURE (only in 'overall' row):
        - no_trade_pct: Percentage of time not in market (label_sign == 0)

        COST IMPACTS (only in 'overall' row):
        - total_whipsaw_cost: Cumulative transaction costs
        - total_wrong_direction_penalty: Cumulative penalties from wrong predictions
        - net_return_after_costs: total_return - costs - penalties

        CUMULATIVE CONSISTENCY METRICS:
        - cum_signed_ret_raw: Sum of raw per-bar signed returns (before wrong_direction_penalty)
        - adv_ret_cum_raw: Cumulative absolute adverse (wrong-direction) raw returns
        - cum_const_ret: cum_signed_ret_raw - adv_ret_cum_raw
          (extra penalty for bars whose movement opposes the assigned regime sign)

    regime_stats : pd.DataFrame
        Detailed per-regime breakdown with columns:
        - persistent_return: Cumulative return for the regime
        - label: Regime type
        - regime_length: Number of periods in regime
        - regime_purity: Fraction of correct-direction returns
        - max_drawdown: Maximum drawdown within regime

    Interpretation Guidelines:
    -------------------------
    WATCH OUT FOR:
    1. Low persistence_score (<5): Model flips too frequently, poor stability
    2. Low regime_purity (<0.55): Predictions lack within-regime consistency
    3. Win_rate <0.5: Model is not better than random
    4. High std relative to mean: Volatile, unreliable performance
    5. Negative skew: Risk of large losses
    6. Large drawdowns: Check dd_max, dd_mean - risk management concern
    7. Flat regime metrics all zero: Expected behavior, not an error

    GOOD TARGETS:
    - persistence_score: >10
    - regime_purity: >0.6
    - win_rate: >0.55
    - profit_factor: >1.5
    - sharpe_ratio: >1.0
    - avg_dwell: >5-10 periods
    """
    # Filter out NaN values in required columns BEFORE processing
    required_cols = [label_col, "close", "open"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df[label_col].dropna().value_counts()) < 2:
        return None, None

    # Remove rows with NaN in label_col, close, or open
    df = df.dropna(subset=required_cols).copy()

    if len(df) == 0:
        raise ValueError(
            f"No valid data after removing NaN values from {required_cols}"
        )

    # Set up label sign mapping
    if label_sign_map is None:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        label_sign_map = map_label_to_trend_direction(
            df, state_col=label_col, return_col="log_return", method="simple"
        )

    # Map labels to signs
    df[f"{label_col}_sign"] = df[label_col].map(label_sign_map)

    # Check if all signs are 0 (no trade regimes)
    unique_signs = df[f"{label_col}_sign"].dropna().unique()
    if len(unique_signs) == 1 and unique_signs[0] == 0:
        # All regimes are no-trade, no point in analyzing
        return None, None

    # Calculate log returns
    df["log_return"] = np.log(df["close"] / df["open"])

    # FIX: Shift signal by 1 to avoid look-ahead bias
    # We observe regime at close of bar i-1, trade on bar i
    df[f"{label_col}_sign_shifted"] = df[f"{label_col}_sign"].shift(1)

    # FIX: Implement consecutive confirmation logic (matching NaiveSignalGenerator)
    # Require progressive_skip consecutive observations before entering a trade
    if progressive_skip > 0:
        # Create lagged signal columns for confirmation check
        for i in range(progressive_skip):
            df[f"signal_lag_{i}"] = df[f"{label_col}_sign"].shift(i + 1)

        # Check if all lagged signals are the same and not zero (no-trade)
        lag_cols = [f"signal_lag_{i}" for i in range(progressive_skip)]

        # Must have NO NaN values (all consecutive observations available)
        no_nans = df[lag_cols].notna().all(axis=1)

        # All signals must be non-zero (not no-trade regime)
        all_non_zero = (df[lag_cols] != 0).all(axis=1)

        # All signals must be identical (consecutive confirmations)
        all_same = df[lag_cols].nunique(axis=1) == 1

        # Can trade only when all conditions are met
        df["can_trade"] = no_nans & all_non_zero & all_same

        # Clean up temporary columns
        for col in lag_cols:
            df.drop(columns=[col], inplace=True)
    else:
        # If no progressive_skip, can trade anytime signal is not 0
        df["can_trade"] = df[f"{label_col}_sign_shifted"] != 0

    # Calculate signed returns using shifted signal
    df["signed_return"] = np.where(
        df["can_trade"],
        df[f"{label_col}_sign_shifted"] * df["log_return"],
        np.nan,  # Don't trade when can_trade is False
    )
    # Keep an unpenalized copy for cumulative consistency diagnostics.
    df["signed_return_raw"] = df["signed_return"]

    # Assign regime IDs based on SHIFTED signal changes (for statistics grouping)
    df["regime_change"] = (
            df[f"{label_col}_sign_shifted"] != df[f"{label_col}_sign_shifted"].shift(1)
    ).astype(int)
    df.loc[df.index[0], "regime_change"] = 1  # First row is always a new regime
    df["regime_id"] = df["regime_change"].cumsum()

    # Apply wrong direction penalty - only to actual trades (sign != 0)
    if wrong_direction_penalty > 0:
        wrong_direction_mask = (df["signed_return"] < 0) & (
                df[f"{label_col}_sign_shifted"] != 0
        )
        df.loc[wrong_direction_mask, "signed_return"] *= 1 + wrong_direction_penalty

    # Calculate regime statistics
    regime_groups = df.groupby("regime_id")

    # Persistent returns per regime — vectorized aggregations
    valid_trade_mask = df["signed_return"].notna()
    signed_valid = df.loc[valid_trade_mask, "signed_return"]
    signed_regime_ids = df.loc[valid_trade_mask, "regime_id"].to_numpy()
    signed_groups = signed_valid.groupby(signed_regime_ids)

    regime_stats = pd.DataFrame({
        "persistent_return": signed_groups.sum(),
        "correct_direction_count": signed_groups.apply(lambda x: (x > 0).sum()),
        "total_trades": signed_groups.count(),
    })

    # Add regime metadata from all rows (including NaN signed_return)
    regime_stats[label_col] = regime_groups[label_col].first()
    regime_stats[f"{label_col}_sign"] = regime_groups[f"{label_col}_sign_shifted"].first()
    regime_stats["regime_length"] = regime_groups.size()

    # Fill regimes that had no valid trades
    regime_stats["persistent_return"] = regime_stats["persistent_return"].fillna(0)
    regime_stats["correct_direction_count"] = regime_stats["correct_direction_count"].fillna(0)
    regime_stats["total_trades"] = regime_stats["total_trades"].fillna(0)

    # Calculate regime purity (fraction of correct-direction returns within each regime)
    regime_stats["regime_purity"] = np.where(
        regime_stats["total_trades"] > 0,
        regime_stats["correct_direction_count"] / regime_stats["total_trades"],
        np.nan,
    )

    # Calculate drawdowns within each regime — vectorized approach
    # Compute cumulative signed returns within each regime group
    df["_cum_signed"] = 0.0
    df.loc[valid_trade_mask, "_cum_signed"] = signed_valid.groupby(signed_regime_ids).cumsum().to_numpy()
    # Running max within each regime
    df["_regime_cummax"] = df.groupby("regime_id")["_cum_signed"].cummax()
    df["_drawdown"] = df["_cum_signed"] - df["_regime_cummax"]
    regime_stats["max_drawdown"] = df.groupby("regime_id")["_drawdown"].min().reindex(regime_stats.index, fill_value=0)
    df.drop(columns=["_cum_signed", "_regime_cummax", "_drawdown"], inplace=True)

    # Calculate whipsaw costs - only count regime changes between actual trades (sign != 0)
    # No-trade regimes (sign == 0) don't incur transaction costs
    trade_regimes = regime_stats[regime_stats[f"{label_col}_sign"] != 0]
    num_trade_regime_changes = (
        max(0, len(trade_regimes) - 1) if len(trade_regimes) > 0 else 0
    )
    total_whipsaw_cost = num_trade_regime_changes * whipsaw_cost

    # Calculate total wrong direction penalty
    total_wrong_direction_penalty = (
        abs(
            df.loc[df["signed_return"] < 0, "signed_return"].sum()
            * wrong_direction_penalty
            / (1 + wrong_direction_penalty)
        )
        if wrong_direction_penalty > 0
        else 0
    )

    # Calculate percentage of regimes that are no-trade (sign == 0)
    # Based on contiguous regime blocks (count), not time periods
    no_trade_regimes = len(regime_stats[regime_stats[f"{label_col}_sign"] == 0])
    total_regimes = len(regime_stats)
    no_trade_pct = (
        round((no_trade_regimes / total_regimes * 100), 2) if total_regimes > 0 else 0
    )

    def compute_cumulative_consistency_metrics(
            signed_returns: np.ndarray,
    ) -> tuple[float, float, float]:
        if len(signed_returns) == 0:
            return 0.0, 0.0, 0.0
        cum_signed_ret_raw = float(np.sum(signed_returns))
        adv_ret_cum_raw = float(
            np.sum(np.abs(signed_returns[signed_returns < 0]))
        )
        cum_const_ret = (
                cum_signed_ret_raw - adv_ret_cum_raw
        )
        return (
            cum_signed_ret_raw,
            adv_ret_cum_raw,
            cum_const_ret,
        )

    # Function to calculate metrics for a set of persistent returns
    def calculate_metrics(persistent_returns, regime_lengths, regime_purities):
        if len(persistent_returns) == 0:
            return {
                "count": 0,
                "n_trades": 0,
                "total_return": 0,
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "25p": 0,
                "75p": 0,
                "skew": np.nan,
                "kurtosis": np.nan,
                "n_wins": 0,
                "n_flat": 0,
                "n_losses": 0,
                "win_rate": 0,
                "profit_factor": np.nan,
                "sharpe_ratio": 0,
                "avg_dwell": 0,
                "persistence_score": 0,
                "regime_purity": 0,
                "dd_min": 0,
                "dd_max": 0,
                "dd_mean": 0,
                "dd_median": 0,
                "dd_25p": 0,
                "dd_75p": 0,
                "dd_std": 0,
                "cum_signed_ret_raw": 0,
                "adv_ret_cum_raw": 0,
                "cum_const_ret": 0,
            }

        return {
            "count": len(persistent_returns),
            "n_trades": len(persistent_returns),
            "total_return": np.sum(persistent_returns),
            "mean": np.mean(persistent_returns),
            "median": np.median(persistent_returns),
            "std": (
                np.std(persistent_returns, ddof=1) if len(persistent_returns) > 1 else 0
            ),
            "min": np.min(persistent_returns),
            "max": np.max(persistent_returns),
            "25p": np.percentile(persistent_returns, 25),
            "75p": np.percentile(persistent_returns, 75),
            "skew": (
                stats.skew(persistent_returns)
                if len(persistent_returns) > 2
                else np.nan
            ),
            "kurtosis": (
                stats.kurtosis(persistent_returns)
                if len(persistent_returns) > 3
                else np.nan
            ),
            "n_wins": np.sum(persistent_returns > 0),
            "n_flat": np.sum(persistent_returns == 0),
            "n_losses": np.sum(persistent_returns < 0),
            "win_rate": round(
                np.sum(persistent_returns > 0) / len(persistent_returns) * 100, 2
            ),
            "profit_factor": (
                np.sum(persistent_returns[persistent_returns > 0])
                / abs(np.sum(persistent_returns[persistent_returns < 0]))
                if np.sum(persistent_returns[persistent_returns < 0]) < 0
                else np.inf
            ),
            "sharpe_ratio": round(
                (
                    np.mean(persistent_returns) / np.std(persistent_returns)
                    if np.std(persistent_returns) > 0
                    else 0
                ),
                3,
            ),
            "avg_dwell": round(np.mean(regime_lengths), 2),
            "persistence_score": np.mean(regime_lengths)
                                 * np.mean(persistent_returns > 0),
            "regime_purity": (
                np.nanmean(regime_purities)
                if len(regime_purities) > 0 and not np.all(np.isnan(regime_purities))
                else 0.0
            ),
            "dd_min": (
                np.min(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ]
                )
                if len(persistent_returns) > 0
                else 0
            ),
            "dd_max": (
                np.max(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ]
                )
                if len(persistent_returns) > 0
                else 0
            ),
            "dd_mean": (
                np.mean(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ]
                )
                if len(persistent_returns) > 0
                else 0
            ),
            "dd_median": (
                np.median(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ]
                )
                if len(persistent_returns) > 0
                else 0
            ),
            "dd_25p": (
                np.percentile(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ],
                    25,
                )
                if len(persistent_returns) > 0
                else 0
            ),
            "dd_75p": (
                np.percentile(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ],
                    75,
                )
                if len(persistent_returns) > 0
                else 0
            ),
            "dd_std": (
                np.std(
                    regime_stats.loc[
                        regime_stats[label_col].isin([persistent_returns]),
                        "max_drawdown",
                    ]
                )
                if len(persistent_returns) > 0
                else 0
            ),
        }

    # Calculate metrics per label
    metrics_list = []
    # Filter out NaN values from unique labels
    unique_labels = sorted(
        [label for label in df[label_col].unique() if pd.notna(label)]
    )

    for label_val in unique_labels:
        label_mask = regime_stats[label_col] == label_val
        label_regime_data = regime_stats.loc[label_mask]
        label_regime_ids = label_regime_data.index

        # For win_rate calculation, only consider actual trade regimes (sign != 0)
        label_sign = (
            label_regime_data[f"{label_col}_sign"].iloc[0]
            if len(label_regime_data) > 0
            else 0
        )
        is_trade_regime = label_sign != 0

        label_persistent_returns = label_regime_data["persistent_return"].values
        label_regime_lengths = label_regime_data["regime_length"].values
        label_regime_purities = label_regime_data["regime_purity"].values
        label_drawdowns = label_regime_data["max_drawdown"].values
        label_signed_returns_raw = df.loc[
            df["regime_id"].isin(label_regime_ids), "signed_return_raw"
        ].dropna().values
        (
            label_cum_signed_ret_raw,
            label_adv_ret_cum_raw,
            label_cum_const_ret,
        ) = compute_cumulative_consistency_metrics(label_signed_returns_raw)

        if len(label_persistent_returns) > 0:
            # For trade regimes (sign != 0), calculate win_rate normally
            # For no-trade regimes (sign == 0), win_rate should be 0
            if is_trade_regime:
                win_rate = round(
                    np.sum(label_persistent_returns > 0)
                    / len(label_persistent_returns)
                    * 100,
                    2,
                )
                persistence_score = np.mean(label_regime_lengths) * np.mean(
                    label_persistent_returns > 0
                )
            else:
                win_rate = 0
                persistence_score = 0

            label_metrics = {
                "count": len(label_persistent_returns),
                "n_trades": len(label_persistent_returns),
                "total_return": np.sum(label_persistent_returns),
                "mean": np.mean(label_persistent_returns),
                "median": np.median(label_persistent_returns),
                "std": (
                    np.std(label_persistent_returns, ddof=1)
                    if len(label_persistent_returns) > 1
                    else 0
                ),
                "min": np.min(label_persistent_returns),
                "max": np.max(label_persistent_returns),
                "25p": np.percentile(label_persistent_returns, 25),
                "75p": np.percentile(label_persistent_returns, 75),
                "skew": (
                    stats.skew(label_persistent_returns)
                    if len(label_persistent_returns) > 2
                    else np.nan
                ),
                "kurtosis": (
                    stats.kurtosis(label_persistent_returns)
                    if len(label_persistent_returns) > 3
                    else np.nan
                ),
                "n_wins": np.sum(label_persistent_returns > 0),
                "n_flat": np.sum(label_persistent_returns == 0),
                "n_losses": np.sum(label_persistent_returns < 0),
                "win_rate": win_rate,
                "profit_factor": round(
                    (
                        np.sum(label_persistent_returns[label_persistent_returns > 0])
                        / abs(
                            np.sum(
                                label_persistent_returns[label_persistent_returns < 0]
                            )
                        )
                        if np.sum(
                            label_persistent_returns[label_persistent_returns < 0]
                        )
                           < 0
                        else np.inf
                    ),
                    2,
                ),
                "sharpe_ratio": round(
                    (
                        np.mean(label_persistent_returns)
                        / np.std(label_persistent_returns)
                        if np.std(label_persistent_returns) > 0
                        else 0
                    ),
                    3,
                ),
                "avg_dwell": round(np.mean(label_regime_lengths), 2),
                "persistence_score": persistence_score,
                "regime_purity": (
                    np.nanmean(label_regime_purities)
                    if len(label_regime_purities) > 0
                       and not np.all(np.isnan(label_regime_purities))
                    else 0.0
                ),
                "dd_min": np.min(label_drawdowns),
                "dd_max": np.max(label_drawdowns),
                "dd_mean": np.mean(label_drawdowns),
                "dd_median": np.median(label_drawdowns),
                "dd_25p": np.percentile(label_drawdowns, 25),
                "dd_75p": np.percentile(label_drawdowns, 75),
                "dd_std": np.std(label_drawdowns),
                "cum_signed_ret_raw": label_cum_signed_ret_raw,
                "adv_ret_cum_raw": label_adv_ret_cum_raw,
                "cum_const_ret": label_cum_const_ret,
            }
        else:
            label_metrics = calculate_metrics([], [], [])

        label_metrics["label"] = label_val
        metrics_list.append(label_metrics)

    # Calculate overall metrics
    # For overall win_rate, only consider actual trade regimes (sign != 0)
    all_persistent_returns = regime_stats["persistent_return"].values
    all_regime_lengths = regime_stats["regime_length"].values
    all_regime_purities = regime_stats["regime_purity"].values
    all_drawdowns = regime_stats["max_drawdown"].values

    # Filter for actual trade regimes (sign != 0) for win_rate calculation
    trade_regimes_mask = regime_stats[f"{label_col}_sign"] != 0
    trade_persistent_returns = regime_stats.loc[
        trade_regimes_mask, "persistent_return"
    ].values
    trade_regime_lengths = regime_stats.loc[trade_regimes_mask, "regime_length"].values
    all_signed_returns_raw = df["signed_return_raw"].dropna().values
    (
        overall_cum_signed_ret_raw,
        overall_adv_ret_cum_raw,
        overall_cum_const_ret,
    ) = compute_cumulative_consistency_metrics(all_signed_returns_raw)

    # Calculate win_rate only for actual trade regimes
    if len(trade_persistent_returns) > 0:
        win_rate = round(
            np.sum(trade_persistent_returns > 0) / len(trade_persistent_returns) * 100,
            2,
        )
        persistence_score = np.mean(trade_regime_lengths) * np.mean(
            trade_persistent_returns > 0
        )
    else:
        win_rate = 0
        persistence_score = 0

    overall_metrics = {
        "count": len(all_persistent_returns),
        "n_trades": len(trade_persistent_returns),
        "total_return": np.sum(all_persistent_returns),
        "mean": np.mean(trade_persistent_returns),
        "median": np.median(trade_persistent_returns),
        "std": (
            np.std(trade_persistent_returns, ddof=1)
            if len(trade_persistent_returns) > 1
            else 0
        ),
        "min": np.min(trade_persistent_returns),
        "max": np.max(trade_persistent_returns),
        "25p": np.percentile(trade_persistent_returns, 25),
        "75p": np.percentile(trade_persistent_returns, 75),
        "skew": (
            stats.skew(trade_persistent_returns)
            if len(trade_persistent_returns) > 2
            else np.nan
        ),
        "kurtosis": (
            stats.kurtosis(trade_persistent_returns)
            if len(trade_persistent_returns) > 3
            else np.nan
        ),
        "n_wins": np.sum(trade_persistent_returns > 0),
        "n_flat": np.sum(trade_persistent_returns == 0),
        "n_losses": np.sum(trade_persistent_returns < 0),
        "win_rate": win_rate,
        "profit_factor": (
            np.sum(trade_persistent_returns[trade_persistent_returns > 0])
            / abs(np.sum(trade_persistent_returns[trade_persistent_returns < 0]))
            if np.sum(trade_persistent_returns[trade_persistent_returns < 0]) < 0
            else np.inf
        ),
        "sharpe_ratio": round(
            (
                np.mean(trade_persistent_returns) / np.std(trade_persistent_returns)
                if np.std(all_persistent_returns) > 0
                else 0
            ),
            3,
        ),
        "avg_dwell": round(np.mean(all_regime_lengths), 2),
        "persistence_score": persistence_score,
        "regime_purity": (
            np.nanmean(all_regime_purities)
            if len(all_regime_purities) > 0
               and not np.all(np.isnan(all_regime_purities))
            else 0.0
        ),
        "no_trade%": no_trade_pct,
        "dd_min": np.min(all_drawdowns),
        "dd_max": np.max(all_drawdowns),
        "dd_mean": np.mean(all_drawdowns),
        "dd_median": np.median(all_drawdowns),
        "dd_25p": np.percentile(all_drawdowns, 25),
        "dd_75p": np.percentile(all_drawdowns, 75),
        "dd_std": np.std(all_drawdowns),
        "total_whipsaw_cost": total_whipsaw_cost,
        "total_wrong_direction_penalty": total_wrong_direction_penalty,
        "net_return_after_costs": np.sum(all_persistent_returns)
                                  - total_whipsaw_cost
                                  - total_wrong_direction_penalty,
        "cum_signed_ret_raw": overall_cum_signed_ret_raw,
        "adv_ret_cum_raw": overall_adv_ret_cum_raw,
        "cum_const_ret": overall_cum_const_ret,
        "label": "overall",
    }

    if include_overall:
        metrics_list.append(overall_metrics)

    return pd.DataFrame(metrics_list), regime_stats


def evaluate_all_labels_regime_returns_potentials(df, labels_cols: List[str], progressive_skip: int = 1,
                                                  whipsaw_cost: float = 0.0, wrong_direction_penalty: float = 0.0,
                                                  label_sign_mapping_method: str = "simple",
                                                  label_sign_mapping: dict[float, int] = None, include_overall: bool = True):
    df = df.copy()  # prevent mutation of caller's DataFrame
    result_df = None
    for col in labels_cols:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        label_sign_map = map_label_to_trend_direction(
            df,
            state_col=col,
            return_col="log_return",
            method=label_sign_mapping_method,
            mapping_dict=label_sign_mapping,
        )
        stats_df, _ = evaluate_regime_returns_potentials(
            df,
            label_col=col,
            progressive_skip=progressive_skip,
            include_overall=include_overall,
            whipsaw_cost=whipsaw_cost,
            label_sign_map=label_sign_map,
            wrong_direction_penalty=wrong_direction_penalty,
        )
        if stats_df is None:
            continue

        stats_df.insert(0, "algo", col)
        stats_df.insert(1, "regime", stats_df["label"])
        stats_df.drop("label", axis=1, inplace=True)
        result_df = stats_df if result_df is None else pd.concat([result_df, stats_df])

    # Return None if no labels were processed (empty labels_cols)
    if result_df is None:
        return None

    return result_df.set_index("algo")
