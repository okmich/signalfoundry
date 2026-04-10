"""
Oracle input smoother selection.

Finds the best price smoother per symbol for the AmplitudeBasedLabeler oracle.
The goal is to produce clean, stable labels that downstream meta-learners can learn
to replicate — NOT to maximize the oracle's own profitability.

Scoring is based on label learnability:
- Regime count (fewer = simpler target)
- q5 duration (longer = fewer whipsaw labels)
- Regime stability (higher = consistent within-regime behavior)
- Regime discriminability (regimes must remain economically distinct)

This is a one-time operation per symbol. The smoothed price is discarded after
parameter generation — only the smoother config is stored.
"""

import enum
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from okmich_quant_features.filters import (
    smooth_median, smooth_gaussian, smooth_savitzky_golay,
    smooth_wavelet, smooth_kalman,
)

# Lazy imports to avoid circular dependency:
# utils.__init__ -> _smoother_selection -> diagnostics.regime -> utils.label_eval_util -> utils.__init__
# AmplitudeBasedLabeler and label_path_structure_statistics are imported inside functions that use them.


class SmootherType(enum.StrEnum):
    RAW = "raw"
    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    SAVITZKY_GOLAY = "savitzky_golay"
    WAVELET = "wavelet"
    KALMAN = "kalman"


# Smoother candidates — all non-causal (centered) where supported.
# Ordered from light to aggressive smoothing within each type.
DEFAULT_SMOOTHER_CANDIDATES: List[Dict] = [
    {"name": "Raw", "type": SmootherType.RAW, "params": {}},
    {"name": "Median_11", "type": SmootherType.MEDIAN, "params": {"window": 11, "causal": False}},
    {"name": "Median_21", "type": SmootherType.MEDIAN, "params": {"window": 21, "causal": False}},
    {"name": "Gaussian_11", "type": SmootherType.GAUSSIAN, "params": {"window": 11, "sigma": 2.0, "causal": False}},
    {"name": "Gaussian_21", "type": SmootherType.GAUSSIAN, "params": {"window": 21, "sigma": 3.0, "causal": False}},
    {"name": "SavGol_11_2", "type": SmootherType.SAVITZKY_GOLAY, "params": {"window": 11, "polyorder": 2, "causal": False}},
    {"name": "SavGol_21_2", "type": SmootherType.SAVITZKY_GOLAY, "params": {"window": 21, "polyorder": 2, "causal": False}},
    {"name": "SavGol_21_3", "type": SmootherType.SAVITZKY_GOLAY, "params": {"window": 21, "polyorder": 3, "causal": False}},
    {"name": "Wavelet_db4_2", "type": SmootherType.WAVELET, "params": {"wavelet": "db4", "level": 2}},
    {"name": "Wavelet_db4_3", "type": SmootherType.WAVELET, "params": {"wavelet": "db4", "level": 3}},
    {"name": "Kalman_0.5_1", "type": SmootherType.KALMAN, "params": {"process_noise": 0.5, "measurement_noise": 1.0}},
]

# Learnability scoring weights
REGIME_COUNT_WEIGHT = 0.25
Q5_DURATION_WEIGHT = 0.30
STABILITY_WEIGHT = 0.20
DISCRIMINABILITY_WEIGHT = 0.25


_SMOOTHER_FUNCTIONS = {
    SmootherType.MEDIAN: smooth_median,
    SmootherType.GAUSSIAN: smooth_gaussian,
    SmootherType.SAVITZKY_GOLAY: smooth_savitzky_golay,
    SmootherType.WAVELET: smooth_wavelet,
    SmootherType.KALMAN: smooth_kalman,
}


def apply_smoother(close: pd.Series, smoother_type: SmootherType, params: Dict) -> pd.Series:
    """Apply a smoother to a close price series."""
    if smoother_type == SmootherType.RAW:
        return close.copy()
    func = _SMOOTHER_FUNCTIONS[smoother_type]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return func(close, **params)


def _rank_normalize(s: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = s.rank(ascending=ascending)
    rng = ranked.max() - ranked.min()
    if rng == 0:
        return pd.Series(0.5, index=s.index)
    return (ranked - ranked.min()) / rng


def _evaluate_smoother_labels(df: pd.DataFrame, label_col: str) -> Optional[Dict]:
    """Compute learnability metrics for one smoother's labels."""
    from okmich_quant_labelling.utils.label_eval_util import label_path_structure_statistics

    try:
        ps = label_path_structure_statistics(df, state_col=label_col, returns_col="_log_ret")
        regime_count = int((df[label_col].diff().fillna(0) != 0).sum())
        return {
            "regime_count": regime_count,
            "q5_duration_min": float(ps["q5_duration"].min()),
            "median_duration": float(ps["median_duration"].median()),
            "regime_stability_avg": float(ps["regime_stability"].mean()),
            "discriminability_avg": float(ps["regime_discriminability"].mean()),
            "noise_to_signal_avg": float(ps["noise_to_signal_ratio"].mean()),
        }
    except Exception:
        return None


def find_best_smoother(close: pd.Series, df: pd.DataFrame, minamp: int, tinactive: int, scale: float,
                       candidates: Optional[List[Dict]] = None) -> Tuple[Dict, pd.Series, pd.DataFrame]:
    """
    Find the best price smoother for oracle labelling.

    Runs all smoother candidates, labels each with AmplitudeBasedLabeler using the
    given (minamp, tinactive), and ranks by learnability metrics.

    Parameters
    ----------
    close : pd.Series
        Raw close price series.
    df : pd.DataFrame
        Full OHLCV DataFrame (needed for label evaluation). Must contain 'close', 'open'.
    minamp : int
        Amplitude threshold for the labeller.
    tinactive : int
        Inactive period for the labeller.
    scale : float
        Price scale factor for the labeller.
    candidates : list of dict, optional
        Smoother candidates to evaluate. Defaults to DEFAULT_SMOOTHER_CANDIDATES.

    Returns
    -------
    best_config : dict
        The winning smoother config: {"name", "type", "params"}.
    best_smoothed : pd.Series
        The smoothed close price series from the winning smoother.
    results_df : pd.DataFrame
        Full results table with all candidates and their scores.
    """
    from okmich_quant_labelling.diagnostics.regime import AmplitudeBasedLabeler

    if candidates is None:
        candidates = DEFAULT_SMOOTHER_CANDIDATES

    labeler = AmplitudeBasedLabeler(minamp=minamp, Tinactive=tinactive)

    # Work positionally to avoid duplicate datetime index issues
    df_pos = df.reset_index(drop=True)
    close_pos = close.reset_index(drop=True)
    df_pos["_log_ret"] = np.log(df_pos["close"] / df_pos["close"].shift(1))

    smoothed_series = {}
    results = []

    for candidate in candidates:
        name = candidate["name"]
        smoother_type = candidate["type"]
        params = candidate["params"]

        try:
            smoothed = apply_smoother(close_pos, smoother_type, params)
            smoothed_series[name] = smoothed

            # Label with the smoothed price
            s = smoothed.dropna()
            df_temp = df_pos.loc[s.index].copy()
            df_temp["_smooth_close"] = s.values
            labels = labeler.label(df_temp, price_col="_smooth_close", scale=scale)

            # Evaluate learnability on original close (not smoothed)
            df_eval = df_pos.loc[s.index].copy()
            df_eval["_label"] = labels.values
            df_eval = df_eval.dropna(subset=["_label", "_log_ret"])

            metrics = _evaluate_smoother_labels(df_eval, "_label")
            if metrics is None:
                continue

            results.append({"name": name, "type": str(smoother_type), "params": params, **metrics})
        except Exception:
            continue

    if not results:
        # Fallback to Raw if everything fails
        return candidates[0], close.copy(), pd.DataFrame()

    results_df = pd.DataFrame(results).set_index("name")

    # Compute learnability score
    results_df["learnability_score"] = (
        _rank_normalize(results_df["regime_count"], ascending=True) * REGIME_COUNT_WEIGHT
        + _rank_normalize(results_df["q5_duration_min"], ascending=False) * Q5_DURATION_WEIGHT
        + _rank_normalize(results_df["regime_stability_avg"], ascending=False) * STABILITY_WEIGHT
        + _rank_normalize(results_df["discriminability_avg"], ascending=False) * DISCRIMINABILITY_WEIGHT
    )

    best_name = results_df["learnability_score"].idxmax()

    # Find the matching candidate config
    best_config = next(c for c in candidates if c["name"] == best_name)

    # Get the smoothed series with original index
    best_smoothed = apply_smoother(close, best_config["type"], best_config["params"])

    return best_config, best_smoothed, results_df


def smoother_config_to_metastore(config: Dict) -> Dict:
    """Convert a smoother config to a flat dict suitable for metastore storage."""
    return {
        "smoother_name": config["name"],
        "smoother_type": str(config["type"]),
        "smoother_params": config["params"],
    }


def smoother_config_from_metastore(meta: Dict) -> Dict:
    """Reconstruct a smoother config from metastore data."""
    return {
        "name": meta["smoother_name"],
        "type": SmootherType(meta["smoother_type"]),
        "params": meta["smoother_params"],
    }
