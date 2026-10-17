from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats

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
        'description': 'How distinguishable the episodes of a regime are from those of its closest other regime',
        'calculation': 'Per metric (volatility, return_consistency, mean_duration, autocorrelation_lag1) and per '
                       'other regime: |2*AUC - 1| over the per-episode values of the two regimes. A pair is scored by '
                       'its best metric; the regime takes its lowest pair score (nearest neighbour). Absolute 0-1 scale.',
        'interpretation': {
            'distinct': '> 0.8 (AUC > 0.9) - Episodes almost always separable',
            'moderate': '0.4 - 0.8 (AUC 0.7 - 0.9) - Partial overlap',
            'poor': '< 0.4 (AUC < 0.7) - Episodes largely indistinguishable',
        },
        'companion': 'discriminability_metric names the separating metric; for a single-axis label it should be '
                     'that axis (e.g. volatility for a vol label), not mean_duration alone',
        'warning': 'Descriptive only: separable episodes say nothing about forward returns'
    },

    'efficiency_ratio': {
        'description': 'Quality of price movement within the episodes of the regime',
        'calculation': 'sum over episodes of abs(net episode return) / sum over episodes of sum(abs_returns)',
        'interpretation': {
            'perfect_trend': '1.0 - Every episode is a straight line',
            'pure_noise': 'about 1/sqrt(episode length) - e.g. 0.31 at 10 bars, 0.10 at 100 bars',
        },
        'warning': 'Depends on episode length, so it does not compare regimes; use efficiency_ratio_rw for that'
    },

    'efficiency_ratio_rw': {
        'description': 'Efficiency relative to a random walk with the same bar sizes',
        'calculation': 'sum of abs(net episode return) / its mean over 64 seeded draws that keep the absolute '
                       'return of every bar and randomise its sign',
        'interpretation': {
            'trending': '> 1 - Drift or positive autocorrelation within episodes',
            'random_walk': 'about 1.0 at any episode length, fat tails and volatility clustering included',
            'mean_reverting': '< 1 - Moves within an episode tend to undo each other'
        },
        'ideal_pattern': 'Clear separation between trending and mean-reverting regimes'
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
        'description': 'Maximum peak-to-trough decline within one episode (each episode starts a fresh curve at 1.0)',
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
        'calculation': 'Lag-1 pairs inside episodes only, both sides centred on the regime mean',
        'interpretation': {
            'momentum': '> 0.1 - Positive persistence',
            'mean_reversion': '< -0.1 - Negative persistence',
            'random_walk': '-0.1 to 0.1 - No clear pattern'
        },
        'ideal_pattern': 'Varies meaningfully across regimes'
    },

    'trend_persistence': {
        'description': 'Linearity of the compounded path within episodes',
        'calculation': 'Bar-weighted mean, over episodes of 10+ bars, of |r| of a straight-line fit to the path',
        'interpretation': {
            'strong_trend': '> 0.3 - Consistent directional movement',
            'weak_trend': '0.1 - 0.3 - Some directionality',
            'no_trend': '< 0.1 - Directionless movement'
        },
        'warning': 'A pure random walk scores about 0.67 at any episode length, so these thresholds do not '
                   'separate trend from noise; prefer efficiency_ratio_rw'
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
        'calculation': 'max(efficiency_ratio_rw) - min(efficiency_ratio_rw)',
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
✅ Clear trending regime: efficiency_ratio_rw well above 1, good discriminability (>0.8)
✅ Clear choppy regime: efficiency_ratio_rw well below 1, high volatility, good discriminability
✅ Volatility regimes: Clear high/medium/low volatility separation
✅ Realistic durations: 5-20 periods average, good stability (>1.0)

FAILURE PATTERNS:
❌ All regimes similar: Near-identical metrics across regimes
❌ Poor discriminability: Scores <0.4 mean the episodes of neighbouring regimes are largely indistinguishable
❌ Unrealistic durations: Avg duration <2 (too jumpy) or >50 (too sticky)
❌ Extreme frequencies: One regime dominates (>60%) or is too rare (<5%)

ASSET-CLASS EXPECTATIONS:
• Equities: Often show clear trend/choppy/transition regimes  
• FX: More volatility-driven regimes, lower efficiency overall
• Gold: Hybrid - can trend like equities but with FX-like volatility regimes
"""


@dataclass(frozen=True)
class LabelTransitionStatistics:
    """Occupancy, transitions and dwell of a categorical label (descriptive; no returns). See
    ``label_transition_statistics``."""
    occupancy: pd.DataFrame  # per label: bars, share
    transition_matrix: pd.DataFrame  # from label (rows) to label (columns), row-normalised; NaN row = never left
    dwell: pd.DataFrame  # per label: episodes, censored, and completed-episode length statistics
    class_entropy: float
    imbalance_ratio: float
    mean_self_transition: float  # mean of the diagonal: each state's probability of staying one more bar


def label_transition_statistics(labels: pd.Series, segment: pd.Series | None = None) -> LabelTransitionStatistics:
    """Occupancy, transition matrix and episode (dwell) lengths of a categorical label.

    Bars are taken in order. A transition is counted between two consecutive known bars of the same ``segment`` (for
    example a trading-day or session id); a NaN label or a segment change ends an episode instead. Episodes cut by
    the sample's ends, a NaN or a segment change are censored: they are counted but kept out of the length
    statistics, which would otherwise be biased short.
    """
    categorical = pd.Categorical(labels)
    states = list(categorical.categories)
    codes = categorical.codes.astype(np.int64)
    seg = (pd.factorize(segment)[0] if segment is not None else np.zeros(codes.size, dtype=np.int64))
    known = codes >= 0
    k = len(states)

    counts = np.bincount(codes[known], minlength=k)
    share = counts / counts.sum() if counts.sum() else np.full(k, np.nan)
    occupancy = pd.DataFrame({"bars": counts, "share": share}, index=pd.Index(states, name="label"))
    present = share[share > 0]
    class_entropy = float(-(present * np.log(present)).sum())
    imbalance_ratio = float(counts.max() / counts[counts > 0].min()) if (counts > 0).any() else np.nan

    linked = known[1:] & known[:-1] & (seg[1:] == seg[:-1])
    pairs = np.bincount(codes[:-1][linked] * k + codes[1:][linked], minlength=k * k).reshape(k, k).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        transition = pairs / pairs.sum(axis=1, keepdims=True)
    transition_matrix = pd.DataFrame(transition, index=pd.Index(states, name="from"),
                                     columns=pd.Index(states, name="to"))

    starts = known & ~np.r_[False, linked & (codes[1:] == codes[:-1])]
    ends = known & ~np.r_[linked & (codes[1:] == codes[:-1]), False]
    run_state = codes[starts]
    run_length = np.flatnonzero(ends) - np.flatnonzero(starts) + 1
    open_start = ~np.r_[False, linked][starts]  # no state change observed before the first bar
    open_end = ~np.r_[linked, False][ends]  # no state change observed after the last bar
    censored = open_start | open_end

    rows = []
    for code, state in enumerate(states):
        mine = run_state == code
        done = run_length[mine & ~censored]
        rows.append({"label": state, "episodes": int(mine.sum()), "censored": int((mine & censored).sum()),
                     "mean": done.mean() if done.size else np.nan,
                     "median": np.median(done) if done.size else np.nan,
                     "q25": np.percentile(done, 25) if done.size else np.nan,
                     "q75": np.percentile(done, 75) if done.size else np.nan,
                     "std": done.std(ddof=1) if done.size > 1 else np.nan,
                     "max": done.max() if done.size else np.nan})
    dwell = pd.DataFrame(rows).set_index("label")
    return LabelTransitionStatistics(occupancy, transition_matrix, dwell, class_entropy, imbalance_ratio,
                                     float(np.nanmean(np.diag(transition))) if k else np.nan)


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

    Notes
    -----
    An episode is a contiguous run of one label in the row order of ``df``; a NaN label ends a run. Point statistics
    (mean, std, skew, ...) pool all bars of a regime. Path statistics (autocorrelation, efficiency, trend persistence,
    volatility stability, drawdown) are measured inside each episode and then aggregated, so no statistic ever spans
    the gap between two separate episodes of the same regime.
    """
    required_cols = [state_col, returns_col, price_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if df[state_col].isna().all():
        raise ValueError(f"Regime column '{state_col}' contains only NaN values")

    # Episodes are numbered before NaN labels are dropped, so the runs on either side of a NaN stay separate
    labels = df[state_col]
    episode_ids = (labels != labels.shift()).cumsum()
    valid_mask = labels.notna()
    valid_data = df.loc[valid_mask].copy()
    valid_data["_episode_id"] = episode_ids[valid_mask]
    if len(valid_data) == 0:
        raise ValueError("No valid regime labels found after removing NaN values")

    regime_stats = {}
    episode_tables = {}
    unique_regimes = valid_data[state_col].unique()

    for regime in unique_regimes:
        regime_mask = valid_data[state_col] == regime
        regime_data = valid_data[regime_mask]

        if len(regime_data) < min_regime_samples:
            continue

        returns = regime_data[returns_col]
        episodes = regime_data["_episode_id"]

        # Basic regime properties
        stats_dict = {
            "regime": regime,
            "n_observations": len(regime_data),
            "frequency_pct": round(len(regime_data) / len(valid_data), 4) * 100,
        }

        # Returns characteristics (agnostic to direction)
        stats_dict.update(_calculate_return_stats(returns))

        # Volatility and risk metrics
        stats_dict.update(_calculate_volatility_stats(returns, episodes))

        # Regime persistence and transitions
        stats_dict.update(_calculate_persistence_stats(episodes))

        # Path structure validation metrics
        stats_dict.update(_calculate_path_validation_stats(returns, episodes))

        regime_stats[regime] = stats_dict
        episode_tables[regime] = _episode_metrics(returns, episodes)

    if not regime_stats:
        raise ValueError("No regimes met the minimum sample requirement")

    # Create results DataFrame
    stats_df = pd.DataFrame.from_dict(regime_stats, orient="index")
    discriminability = _calculate_discriminability_score(episode_tables)
    stats_df["regime_discriminability"] = stats_df["regime"].map(discriminability["regime_discriminability"])
    stats_df["discriminability_metric"] = stats_df["regime"].map(discriminability["discriminability_metric"])

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


# Episode-level evaluation settings
_MIN_BARS_SCALE = 5  # shortest episode whose volatility / return consistency enters the discriminability score
_MIN_BARS_ACF = 10  # shortest episode whose lag-1 autocorrelation enters the discriminability score
_MIN_BARS_TREND = 10  # shortest episode whose trend persistence is measured
_MIN_EPISODES = 5  # fewest qualifying episodes per regime before a metric is compared across regimes
_STABILITY_WINDOW = 5
_SIGN_FLIP_DRAWS = 64
_SIGN_FLIP_SEED = 0
_DISCRIMINABILITY_METRICS = ("volatility", "return_consistency", "mean_duration", "autocorrelation_lag1")


def _calculate_volatility_stats(returns: pd.Series, episodes: pd.Series) -> Dict[str, float]:
    """Calculate volatility and risk metrics; drawdowns and volatility stability are measured within episodes"""
    if len(returns) < 2:
        return {}

    vol = returns.std()
    abs_returns = returns.abs()
    ep = episodes.to_numpy()

    # Each episode starts a fresh equity curve at 1.0, and that starting level is its first peak
    equity = (1 + returns.fillna(0)).groupby(ep).cumprod()
    peak = equity.groupby(ep).cummax().clip(lower=1.0)
    drawdown = equity / peak - 1

    # A rolling window may not straddle two episodes: keep windows that end at least (window - 1) bars into one
    rolling_vol = returns.rolling(_STABILITY_WINDOW).std()
    rolling_vol = rolling_vol[(episodes.groupby(ep).cumcount() >= _STABILITY_WINDOW - 1).to_numpy()]

    return {
        "volatility": vol,
        "volatility_stability": rolling_vol.std() / vol if vol != 0 else np.nan,
        "volatility_to_mean_ratio": vol / abs_returns.mean() if abs_returns.mean() != 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "avg_drawdown": drawdown.mean(),
    }


def _calculate_persistence_stats(episodes: pd.Series) -> Dict[str, float]:
    """Calculate regime persistence statistics from the lengths of the regime's episodes"""
    durations = episodes.value_counts(sort=False).to_numpy()

    if len(durations) == 0:
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


def _calculate_path_validation_stats(returns: pd.Series, episodes: pd.Series) -> Dict[str, float]:
    """Calculate path-structure metrics; each one is measured inside episodes, never across two of them"""
    if returns.notna().sum() < 5:
        return {}

    ep = episodes.to_numpy()

    # Lag-1 autocorrelation from within-episode pairs only, both sides centred on the regime mean
    deviation = returns - returns.mean()
    lag_products = deviation * deviation.groupby(ep).shift(1)
    variance = (deviation ** 2).mean()
    acf1 = lag_products.mean() / variance if lag_products.notna().any() and variance > 0 else np.nan

    has_return = returns.notna().to_numpy()
    clean, clean_ep = returns[has_return], ep[has_return]
    net_move = clean.groupby(clean_ep).sum().abs()
    path_length = clean.abs().groupby(clean_ep).sum()

    return {
        "autocorrelation_lag1": acf1 if not np.isnan(acf1) else 0,
        "trend_persistence": _episode_trend_persistence(clean, clean_ep),
        "noise_to_signal_ratio": returns.std() / returns.abs().mean() if returns.abs().mean() != 0 else np.nan,
        "efficiency_ratio": net_move.sum() / path_length.sum() if path_length.sum() != 0 else 0,
        "efficiency_ratio_rw": _sign_flip_efficiency_ratio(clean, clean_ep),
    }


def _episode_trend_persistence(returns: pd.Series, episodes: np.ndarray) -> float:
    """Bar-weighted mean |r| of a straight-line fit to each episode's compounded path, over episodes of 10+ bars"""
    frame = pd.DataFrame({"episode": episodes, "growth": 1 + returns.to_numpy()})
    frame["growth"] = frame.groupby("episode")["growth"].cumprod() - 1
    frame["t"] = frame.groupby("episode").cumcount().astype(float)
    frame = frame[frame.groupby("episode")["t"].transform("size") >= _MIN_BARS_TREND]
    if frame.empty:
        return np.nan

    # Pearson r of (t, growth) per episode from grouped sums
    frame["tt"], frame["gg"], frame["tg"] = frame["t"] ** 2, frame["growth"] ** 2, frame["t"] * frame["growth"]
    sums = frame.groupby("episode")[["t", "growth", "tt", "gg", "tg"]].sum()
    n = frame.groupby("episode").size()
    cov = sums["tg"] - sums["t"] * sums["growth"] / n
    var_t = sums["tt"] - sums["t"] ** 2 / n
    var_g = sums["gg"] - sums["growth"] ** 2 / n
    r_value = (cov / np.sqrt(var_t * var_g)).abs()

    valid = np.isfinite(r_value)  # a flat episode has no defined fit
    return float(np.average(r_value[valid], weights=n[valid])) if valid.any() else np.nan


def _sign_flip_efficiency_ratio(returns: pd.Series, episodes: np.ndarray) -> float:
    """
    Observed total |net episode move| over its expected value when every bar keeps its size but takes a random sign.

    Keeping each bar's absolute return preserves fat tails and volatility clustering, so a random walk scores ~1.0 at
    any episode length. Drift or positive autocorrelation lifts it above 1; mean reversion pushes it below 1.
    """
    sizes = returns.abs().to_numpy()
    if sizes.sum() == 0:
        return np.nan

    codes = pd.factorize(episodes)[0]
    observed = np.abs(np.bincount(codes, weights=returns.to_numpy())).sum()
    rng = np.random.default_rng(_SIGN_FLIP_SEED)
    null_moves = [np.abs(np.bincount(codes, weights=sizes * (rng.integers(0, 2, len(sizes)) * 2.0 - 1.0))).sum()
                  for _ in range(_SIGN_FLIP_DRAWS)]
    return observed / np.mean(null_moves)


def _episode_metrics(returns: pd.Series, episodes: pd.Series) -> pd.DataFrame:
    """
    Per-episode values of the discriminability metrics: one row per episode, each column named after the regime-level
    statistic it mirrors. A metric is NaN on episodes too short to estimate it.
    """
    ep = episodes.to_numpy()
    grouped = returns.groupby(ep)
    abs_grouped = returns.abs().groupby(ep)
    abs_mean = abs_grouped.mean()
    n_bars = grouped.count()

    deviation = returns - returns.mean()
    lag_products = (deviation * deviation.groupby(ep).shift(1)).groupby(ep).mean()
    variance = (deviation ** 2).groupby(ep).mean()

    return pd.DataFrame({
        "volatility": grouped.std().where(n_bars >= _MIN_BARS_SCALE),
        "return_consistency": (abs_grouped.std() / abs_mean[abs_mean > 0]).where(n_bars >= _MIN_BARS_SCALE),
        "mean_duration": grouped.size(),
        "autocorrelation_lag1": (lag_products / variance[variance > 0]).where(n_bars >= _MIN_BARS_ACF),
    })


def _auc_separation(a: pd.Series, b: pd.Series) -> float:
    """|2·AUC - 1| of two samples: 0 = indistinguishable, 1 = every value of one exceeds every value of the other"""
    a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
    if len(a) < _MIN_EPISODES or len(b) < _MIN_EPISODES:
        return np.nan

    ranks = stats.rankdata(np.concatenate([a, b]))
    auc = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return abs(2 * auc - 1)


def _calculate_discriminability_score(episode_tables: Dict[Any, pd.DataFrame]) -> pd.DataFrame:
    """
    Score how distinguishable each regime's episodes are from those of its closest other regime.

    Against every other regime, each metric's separation is |2·AUC - 1| over the two regimes' per-episode values, and
    the pair is scored by its best-separating metric. A regime's score is its lowest pair score (its nearest
    neighbour) on an absolute 0-1 scale: 0 = episodes indistinguishable on every metric, 1 = fully separable on at
    least one. ``discriminability_metric`` names the metric that separates the regime from that neighbour.
    """
    scores = {}
    for regime, table in episode_tables.items():
        pair_scores = []
        for other, other_table in episode_tables.items():
            if other == regime:
                continue
            separations = {m: _auc_separation(table[m], other_table[m]) for m in _DISCRIMINABILITY_METRICS}
            separations = {m: s for m, s in separations.items() if not np.isnan(s)}
            if separations:
                best = max(separations, key=separations.get)
                pair_scores.append((separations[best], best))
        scores[regime] = min(pair_scores) if pair_scores else (np.nan, None)

    columns = ["regime_discriminability", "discriminability_metric"]
    return pd.DataFrame.from_dict(scores, orient="index", columns=columns)


def evaluate_regime_returns_potentials(df, label_col="label", progressive_skip: int = 1, whipsaw_cost: float = 0.0,
                                       wrong_direction_penalty: float = 0.0, label_sign_map: Dict[Any, int] = None,
                                       include_overall: bool = True):
    """
    P&L potential of trading a label's sign map, per regime and per label, with realistic entry timing.

    Timing:
    - The label known at the close of bar t-1 sets the position held over bar t, which earns bar t's open-to-close
      log return. Nothing trades on the bar its label forms.
    - ``progressive_skip`` is the number of consecutive observations of the same non-zero sign required before the
      position is held. 0 and 1 both mean "from the first bar after the label is known"; k >= 2 waits k - 1 more bars.
    - A NaN label (or price) breaks the confirmation chain and ends a regime; nothing is bridged across it.

    A regime is a run of the decision label (the label at t-1). Every regime is kept, including those mapped to 0
    and those too short to trade, so ``avg_dwell``, ``count`` and ``no_trade%`` describe the whole label.

    In-sample by default: without ``label_sign_map`` the signs are fitted on the same sample, on the next-bar return a
    position earns (``fit_label_sign_map``). That is an upper bound, not an out-of-sample estimate; for one, pass a map
    fitted on a disjoint training block.

    Parameters:
    -----------
    df : pd.DataFrame
        Time-ordered rows with columns open, close and the label column.

    label_col : str, default='label'
        Name of the column containing regime labels

    progressive_skip : int, default=1
        Consecutive observations of the same non-zero sign required before a position is held (see Timing).

    whipsaw_cost : float, default=0.0
        Cost charged per entry into a position (a flip is one entry), in log-return units.

    wrong_direction_penalty : float, default=0.0
        Extra cost on losing bars: the penalty is wrong_direction_penalty x the summed losses, charged once, in
        ``total_wrong_direction_penalty`` and ``net_return_after_costs`` only. ``total_return`` stays unpenalised.

    label_sign_map : dict, optional
        {label: sign in -1/0/1}. Labels it does not cover trade flat. If None, fitted in-sample (see above).

    include_overall : bool, default=True
        Append the 'overall' row.

    Returns:
    --------
    metrics_df : pd.DataFrame
        One row per label (and 'overall'). Return statistics are over TRADED regimes (at least one bar in position);
        ``count`` and ``avg_dwell`` are over all regimes.

        - count: regimes; n_trades: traded regimes
        - total_return, mean, median, std, min, max, 25p, 75p, skew, kurtosis: of per-regime summed returns
        - n_wins / n_flat / n_losses, win_rate (PERCENT of traded regimes that made money), profit_factor
        - sharpe_ratio: mean / std of per-regime returns (per regime, not annualised)
        - avg_dwell: mean regime length in bars; persistence_score: avg_dwell x win fraction
        - regime_purity: mean share of winning bars inside traded regimes
        - dd_*: statistics of the per-regime maximum drawdown (traded regimes)
        - cum_signed_ret_raw, adv_ret_cum_raw, cum_const_ret: summed bar returns, summed losing bars, their difference
        - overall only: no_trade% (PERCENT of bars out of the market), total_whipsaw_cost,
          total_wrong_direction_penalty, net_return_after_costs

    regime_stats : pd.DataFrame
        One row per regime: persistent_return, correct_direction_count, total_trades, the label, its sign,
        regime_length, regime_purity, max_drawdown.

    Returns (None, None) when the label has fewer than two states or every state maps to 0.
    """
    required_cols = [label_col, "close", "open"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if not df[required_cols].notna().all(axis=1).any():
        raise ValueError(f"No valid data after removing NaN values from {required_cols}")
    if df[label_col].dropna().nunique() < 2:
        return None, None

    if label_sign_map is None:
        label_sign_map = fit_label_sign_map(df, label_col, method="simple")
    label = df[label_col]
    sign = label.map(label_sign_map).astype(float)
    sign[label.notna() & sign.isna()] = 0.0  # a known label the map does not cover trades flat
    sign = sign.to_numpy()
    if np.all(sign[np.isfinite(sign)] == 0):
        return None, None

    bar_return = np.log(df["close"] / df["open"]).to_numpy(dtype=float)

    # Position over bar t: the signs observed at closes t-c..t-1 all equal and non-zero.
    confirmations = max(1, int(progressive_skip))
    lags = np.column_stack([pd.Series(sign).shift(i + 1).to_numpy() for i in range(confirmations)])
    with np.errstate(invalid="ignore"):
        confirmed = np.isfinite(lags).all(axis=1) & (lags != 0).all(axis=1) & (lags == lags[:, :1]).all(axis=1)
    position = np.where(confirmed, lags[:, 0], 0.0)
    in_market = (position != 0) & np.isfinite(bar_return)
    signed = np.where(in_market, position * bar_return, np.nan)

    # Regimes: runs of the decision label (the label known at the close of t-1, which drives bar t).
    decision = label.shift(1)
    known = decision.notna().to_numpy()
    codes = pd.factorize(decision)[0]
    regime_id = np.cumsum(known & np.r_[True, codes[1:] != codes[:-1]])

    bars = pd.DataFrame({"regime_id": regime_id, "label": decision.to_numpy(),
                         "sign": pd.Series(sign).shift(1).to_numpy(), "signed": signed, "pnl": np.nan_to_num(signed),
                         "win": signed > 0})[known]
    groups = bars.groupby("regime_id", sort=True)
    cum = groups["pnl"].cumsum()
    peak = cum.groupby(bars["regime_id"]).cummax().clip(lower=0.0)
    regime_stats = pd.DataFrame({
        "persistent_return": groups["pnl"].sum(),
        "correct_direction_count": groups["win"].sum().astype(int),
        "total_trades": groups["signed"].count(),
        label_col: groups["label"].first(),
        f"{label_col}_sign": groups["sign"].first(),
        "regime_length": groups.size(),
        "max_drawdown": (cum - peak).groupby(bars["regime_id"]).min(),
    })
    regime_stats["regime_purity"] = np.where(regime_stats["total_trades"] > 0,
                                             regime_stats["correct_direction_count"]
                                             / regime_stats["total_trades"].clip(lower=1), np.nan)
    regime_stats = regime_stats[["persistent_return", "correct_direction_count", "total_trades", label_col,
                                 f"{label_col}_sign", "regime_length", "regime_purity", "max_drawdown"]]

    metrics_list = []
    for label_val in sorted(label.dropna().unique()):
        in_label = (regime_stats[label_col] == label_val).to_numpy()
        label_bars = np.isin(regime_id, regime_stats.index[in_label]) & known
        metrics = _potential_metrics(regime_stats[in_label], signed[label_bars])
        metrics["label"] = label_val
        metrics_list.append(metrics)

    if include_overall:
        previous = np.r_[0.0, position[:-1]]
        entries = int(((position != 0) & (position != previous)).sum())
        losses = signed[signed < 0]
        total_whipsaw_cost = entries * whipsaw_cost
        total_wrong_direction_penalty = float(wrong_direction_penalty * np.abs(losses).sum())
        overall = _potential_metrics(regime_stats, signed[known])
        decided = known & np.isfinite(bar_return)
        overall["no_trade%"] = round(float((position[decided] == 0).mean() * 100), 2) if decided.any() else 0.0
        overall["total_whipsaw_cost"] = total_whipsaw_cost
        overall["total_wrong_direction_penalty"] = total_wrong_direction_penalty
        overall["net_return_after_costs"] = overall["total_return"] - total_whipsaw_cost - total_wrong_direction_penalty
        overall["label"] = "overall"
        metrics_list.append(overall)

    return pd.DataFrame(metrics_list), regime_stats


def _potential_metrics(regimes: pd.DataFrame, signed: np.ndarray) -> Dict[str, Any]:
    """The per-label / overall row of ``evaluate_regime_returns_potentials`` from its regimes and bar returns."""
    traded = regimes[regimes["total_trades"] > 0]
    r = traded["persistent_return"].to_numpy(dtype=float)
    drawdowns = traded["max_drawdown"].to_numpy(dtype=float)
    lengths = regimes["regime_length"].to_numpy(dtype=float)
    n = r.size
    gains, losses = r[r > 0].sum(), r[r < 0].sum()
    win_fraction = float((r > 0).mean()) if n else 0.0
    bar_signed = signed[np.isfinite(signed)]
    adverse = float(np.abs(bar_signed[bar_signed < 0]).sum())

    def stat(fn, *args):
        return fn(r, *args) if n else 0

    return {
        "count": len(regimes),
        "n_trades": n,
        "total_return": float(r.sum()),
        "mean": stat(np.mean),
        "median": stat(np.median),
        "std": float(np.std(r, ddof=1)) if n > 1 else 0,
        "min": stat(np.min),
        "max": stat(np.max),
        "25p": stat(np.percentile, 25),
        "75p": stat(np.percentile, 75),
        "skew": stats.skew(r) if n > 2 else np.nan,
        "kurtosis": stats.kurtosis(r) if n > 3 else np.nan,
        "n_wins": int((r > 0).sum()),
        "n_flat": int((r == 0).sum()),
        "n_losses": int((r < 0).sum()),
        "win_rate": round(win_fraction * 100, 2),
        "profit_factor": gains / -losses if losses < 0 else (np.inf if gains > 0 else np.nan),
        "sharpe_ratio": round(float(r.mean() / r.std()), 3) if n and r.std() > 0 else 0,
        "avg_dwell": round(float(lengths.mean()), 2) if lengths.size else 0,
        "persistence_score": float(lengths.mean()) * win_fraction if lengths.size else 0,
        "regime_purity": float(np.nanmean(traded["regime_purity"])) if n else 0.0,
        "dd_min": float(drawdowns.min()) if n else 0,
        "dd_max": float(drawdowns.max()) if n else 0,
        "dd_mean": float(drawdowns.mean()) if n else 0,
        "dd_median": float(np.median(drawdowns)) if n else 0,
        "dd_25p": float(np.percentile(drawdowns, 25)) if n else 0,
        "dd_75p": float(np.percentile(drawdowns, 75)) if n else 0,
        "dd_std": float(drawdowns.std()) if n else 0,
        "cum_signed_ret_raw": float(bar_signed.sum()),
        "adv_ret_cum_raw": adverse,
        "cum_const_ret": float(bar_signed.sum()) - adverse,
    }


def fit_label_sign_map(df: pd.DataFrame, label_col: str, method: str = "simple",
                       mapping_dict: Dict[Any, int] = None) -> Dict[Any, int]:
    """{label: sign} fitted on the return a position decided at the close of bar t first earns: bar t+1's open-to-close
    log return. Fitting on the same-bar return would sign a mean-reverting label as trending. With ``mapping_dict`` the
    map is applied as given (uncovered labels get 0). In-sample when ``df`` is also the evaluation block."""
    frame = pd.DataFrame({label_col: df[label_col], "_next_return": np.log(df["close"] / df["open"]).shift(-1)},
                         index=df.index)
    return map_label_to_trend_direction(frame, state_col=label_col, return_col="_next_return", method=method,
                                        mapping_dict=mapping_dict)


def evaluate_all_labels_regime_returns_potentials(df, labels_cols: List[str], progressive_skip: int = 1,
                                                  whipsaw_cost: float = 0.0, wrong_direction_penalty: float = 0.0,
                                                  label_sign_mapping_method: str = "simple",
                                                  label_sign_mapping: dict[float, int] = None,
                                                  include_overall: bool = True):
    """``evaluate_regime_returns_potentials`` for several label columns. Each column's sign map is fitted with
    ``label_sign_mapping_method`` on the next-bar return (in-sample), unless ``label_sign_mapping`` is given; then that
    one map is applied to every column."""
    result_df = None
    for col in labels_cols:
        label_sign_map = fit_label_sign_map(df, col, method=label_sign_mapping_method,
                                            mapping_dict=label_sign_mapping)
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
