"""
Batch computation layer for all 38 Timothy Masters single-market indicators.

Usage
-----
>>> from okmich_quant_features.timothymasters.utils.single_features_computer import compute_features
>>> result = compute_features(df)                       # all 38 columns
>>> result = compute_features(df, groups="momentum")   # 11 columns
>>> result = compute_features(df, groups=["rsi","adx"]) # 2 columns
>>> result = compute_features(df, params={"rsi": {"period": 21}})
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from okmich_quant_features.timothymasters.single.momentum import rsi, detrended_rsi, stochastic, stoch_rsi, \
    ma_difference, macd, ppo,price_change_osc, close_minus_ma, price_intensity, reactivity
from okmich_quant_features.timothymasters.single.trend import linear_trend, quadratic_trend, cubic_trend, \
    linear_deviation, quadratic_deviation, cubic_deviation, adx, aroon_up, aroon_down, aroon_diff
from okmich_quant_features.timothymasters.single.variance import price_variance_ratio, change_variance_ratio
from okmich_quant_features.timothymasters.single.volume import intraday_intensity, money_flow, price_volume_fit, \
    vwma_ratio, normalized_obv, delta_obv, normalized_pvi, normalized_nvi, volume_momentum
from okmich_quant_features.timothymasters.single.information import entropy, mutual_information
from okmich_quant_features.timothymasters.single.fti import fti_lowpass, fti_best_width, fti_best_period, fti_best_fti


# ---------------------------------------------------------------------------
# Group constants (public)
# ---------------------------------------------------------------------------

MOMENTUM_INDICATORS: list[str] = [
    "rsi", "detrended_rsi", "stochastic", "stoch_rsi", "ma_difference",
    "macd", "ppo", "price_change_osc", "close_minus_ma", "price_intensity",
    "reactivity",
]

TREND_INDICATORS: list[str] = [
    "linear_trend", "quadratic_trend", "cubic_trend",
    "linear_deviation", "quadratic_deviation", "cubic_deviation",
    "adx", "aroon_up", "aroon_down", "aroon_diff",
]

VARIANCE_INDICATORS: list[str] = [
    "price_variance_ratio", "change_variance_ratio",
]

VOLUME_INDICATORS: list[str] = [
    "intraday_intensity", "money_flow", "price_volume_fit", "vwma_ratio",
    "normalized_obv", "delta_obv", "normalized_pvi", "normalized_nvi",
    "volume_momentum",
]

INFORMATION_INDICATORS: list[str] = [
    "entropy", "mutual_information",
]

FTI_INDICATORS: list[str] = [
    "fti_lowpass", "fti_best_width", "fti_best_period", "fti_best_fti",
]

ALL_INDICATORS: list[str] = (
    MOMENTUM_INDICATORS
    + TREND_INDICATORS
    + VARIANCE_INDICATORS
    + VOLUME_INDICATORS
    + INFORMATION_INDICATORS
    + FTI_INDICATORS
)

GROUPS: dict[str, list[str]] = {
    "momentum": MOMENTUM_INDICATORS,
    "trend": TREND_INDICATORS,
    "variance": VARIANCE_INDICATORS,
    "volume": VOLUME_INDICATORS,
    "information": INFORMATION_INDICATORS,
    "fti": FTI_INDICATORS,
    "all": ALL_INDICATORS,
}


# ---------------------------------------------------------------------------
# Default parameters (public)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, dict] = {
    # Momentum (#1–11)
    "rsi":              {"period": 14},
    "detrended_rsi":    {"short_period": 7, "long_period": 14, "reg_len": 32},
    "stochastic":       {"period": 14, "smoothing": 1},
    "stoch_rsi":        {"rsi_period": 14, "stoch_period": 14, "smooth_period": 1},
    "ma_difference":    {"short_period": 10, "long_period": 40, "lag": 0},
    "macd":             {"short_period": 18, "long_period": 48, "signal_period": 9},
    "ppo":              {"short_period": 12, "long_period": 26, "signal_period": 9},
    "price_change_osc": {"short_period": 10, "multiplier": 4.0},
    "close_minus_ma":   {"period": 20, "atr_period": 60},
    "price_intensity":  {"smooth_period": 1},
    "reactivity":       {"period": 10, "multiplier": 4},
    # Trend (#12–21)
    "linear_trend":        {"period": 20, "atr_period": 60},
    "quadratic_trend":     {"period": 20, "atr_period": 60},
    "cubic_trend":         {"period": 20, "atr_period": 60},
    "linear_deviation":    {"period": 20},
    "quadratic_deviation": {"period": 20},
    "cubic_deviation":     {"period": 20},
    "adx":                 {"period": 14},
    "aroon_up":            {"period": 25},
    "aroon_down":          {"period": 25},
    "aroon_diff":          {"period": 25},
    # Variance (#22–23)
    "price_variance_ratio":  {"short_period": 10, "multiplier": 4.0},
    "change_variance_ratio": {"short_period": 10, "multiplier": 4.0},
    # Volume (#24–32)
    "intraday_intensity": {"period": 14, "smooth_period": 0},
    "money_flow":         {"period": 14},
    "price_volume_fit":   {"period": 20},
    "vwma_ratio":         {"period": 20},
    "normalized_obv":     {"period": 20},
    "delta_obv":          {"period": 20, "delta_period": 5},
    "normalized_pvi":     {"period": 20},
    "normalized_nvi":     {"period": 20},
    "volume_momentum":    {"short_period": 10, "multiplier": 4.0},
    # Information (#33–34)
    "entropy":             {"word_length": 3, "mult": 10},
    "mutual_information":  {"word_length": 3, "mult": 10},
    # FTI (#35–38)
    "fti_lowpass":     {"lookback": 60, "half_length": 40, "min_period": 8, "max_period": 40},
    "fti_best_width":  {"lookback": 60, "half_length": 40, "min_period": 8, "max_period": 40},
    "fti_best_period": {"lookback": 60, "half_length": 40, "min_period": 8, "max_period": 40},
    "fti_best_fti":    {"lookback": 60, "half_length": 40, "min_period": 8, "max_period": 40},
}


# ---------------------------------------------------------------------------
# Required columns per indicator (private)
# ---------------------------------------------------------------------------

_REQUIRED_COLS: dict[str, set[str]] = {
    # close only
    "rsi":                  {"close"},
    "detrended_rsi":        {"close"},
    "stoch_rsi":            {"close"},
    "ppo":                  {"close"},
    "linear_deviation":     {"close"},
    "quadratic_deviation":  {"close"},
    "cubic_deviation":      {"close"},
    "price_variance_ratio": {"close"},
    "change_variance_ratio":{"close"},
    "entropy":              {"close"},
    "mutual_information":   {"close"},
    "fti_lowpass":          {"close"},
    "fti_best_width":       {"close"},
    "fti_best_period":      {"close"},
    "fti_best_fti":         {"close"},
    # high, low, close
    "stochastic":     {"high", "low", "close"},
    "ma_difference":  {"high", "low", "close"},
    "macd":           {"high", "low", "close"},
    "price_change_osc": {"high", "low", "close"},
    "close_minus_ma": {"high", "low", "close"},
    "linear_trend":   {"high", "low", "close"},
    "quadratic_trend":{"high", "low", "close"},
    "cubic_trend":    {"high", "low", "close"},
    "adx":            {"high", "low", "close"},
    # high, low only
    "aroon_up":   {"high", "low"},
    "aroon_down": {"high", "low"},
    "aroon_diff": {"high", "low"},
    # open, high, low, close
    "price_intensity": {"open", "high", "low", "close"},
    # high, low, close, volume
    "reactivity":         {"high", "low", "close", "volume"},
    "intraday_intensity": {"high", "low", "close", "volume"},
    "money_flow":         {"high", "low", "close", "volume"},
    # close, volume
    "price_volume_fit": {"close", "volume"},
    "vwma_ratio":       {"close", "volume"},
    "normalized_obv":   {"close", "volume"},
    "delta_obv":        {"close", "volume"},
    "normalized_pvi":   {"close", "volume"},
    "normalized_nvi":   {"close", "volume"},
    # volume only
    "volume_momentum": {"volume"},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_ALL_INDICATOR_SET: set[str] = set(ALL_INDICATORS)
_GROUP_NAMES: set[str] = set(GROUPS.keys())


def _resolve_groups(groups: str | list[str]) -> list[str]:
    """Expand group name(s) or indicator name(s) to an ordered deduplicated list.

    Parameters
    ----------
    groups : str | list[str]
        A group name ("all", "momentum", …), a single indicator name, or a
        mixed list of group names and indicator names.

    Returns
    -------
    list[str]
        Ordered, deduplicated indicator names.

    Raises
    ------
    ValueError
        If an unknown group or indicator name is encountered.
    """
    if isinstance(groups, str):
        groups = [groups]

    seen: set[str] = set()
    result: list[str] = []
    for item in groups:
        if item in GROUPS:
            for name in GROUPS[item]:
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        elif item in _ALL_INDICATOR_SET:
            if item not in seen:
                seen.add(item)
                result.append(item)
        else:
            raise ValueError(
                f"Unknown group or indicator name: {item!r}. "
                f"Valid groups: {sorted(_GROUP_NAMES)}. "
                f"Call list_indicators() for the full indicator list."
            )
    return result


def _validate_cols(names: list[str], col_map: dict[str, str], df_columns: set[str]) -> None:
    """Validate that all required symbolic columns are mapped and present in the DataFrame.

    Parameters
    ----------
    names      : Indicator names to validate.
    col_map    : Mapping from symbolic name (``"open"``, ``"close"``, …) to actual column name.
    df_columns : Set of columns present in the DataFrame.

    Raises
    ------
    KeyError
        If a required column is missing from ``df_columns``.
    """
    for name in names:
        for sym_col in _REQUIRED_COLS.get(name, set()):
            actual = col_map[sym_col]
            if actual not in df_columns:
                raise KeyError(
                    f"Indicator {name!r} requires column {sym_col!r} "
                    f"(mapped to {actual!r}), but it is not in the DataFrame. "
                    f"Available columns: {sorted(df_columns)}"
                )


def _compute_one(name: str, params: dict, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray) -> np.ndarray:
    """Dispatch a single indicator by name, forwarding the resolved params."""
    if name == "rsi":
        return rsi(close, **params)
    elif name == "detrended_rsi":
        return detrended_rsi(close, **params)
    elif name == "stochastic":
        return stochastic(high, low, close, **params)
    elif name == "stoch_rsi":
        return stoch_rsi(close, **params)
    elif name == "ma_difference":
        return ma_difference(high, low, close, **params)
    elif name == "macd":
        return macd(high, low, close, **params)
    elif name == "ppo":
        return ppo(close, **params)
    elif name == "price_change_osc":
        return price_change_osc(high, low, close, **params)
    elif name == "close_minus_ma":
        return close_minus_ma(high, low, close, **params)
    elif name == "price_intensity":
        return price_intensity(open_, high, low, close, **params)
    elif name == "reactivity":
        return reactivity(high, low, close, volume, **params)
    elif name == "linear_trend":
        return linear_trend(high, low, close, **params)
    elif name == "quadratic_trend":
        return quadratic_trend(high, low, close, **params)
    elif name == "cubic_trend":
        return cubic_trend(high, low, close, **params)
    elif name == "linear_deviation":
        return linear_deviation(close, **params)
    elif name == "quadratic_deviation":
        return quadratic_deviation(close, **params)
    elif name == "cubic_deviation":
        return cubic_deviation(close, **params)
    elif name == "adx":
        return adx(high, low, close, **params)
    elif name == "aroon_up":
        return aroon_up(high, low, **params)
    elif name == "aroon_down":
        return aroon_down(high, low, **params)
    elif name == "aroon_diff":
        return aroon_diff(high, low, **params)
    elif name == "price_variance_ratio":
        return price_variance_ratio(close, **params)
    elif name == "change_variance_ratio":
        return change_variance_ratio(close, **params)
    elif name == "intraday_intensity":
        return intraday_intensity(high, low, close, volume, **params)
    elif name == "money_flow":
        return money_flow(high, low, close, volume, **params)
    elif name == "price_volume_fit":
        return price_volume_fit(close, volume, **params)
    elif name == "vwma_ratio":
        return vwma_ratio(close, volume, **params)
    elif name == "normalized_obv":
        return normalized_obv(close, volume, **params)
    elif name == "delta_obv":
        return delta_obv(close, volume, **params)
    elif name == "normalized_pvi":
        return normalized_pvi(close, volume, **params)
    elif name == "normalized_nvi":
        return normalized_nvi(close, volume, **params)
    elif name == "volume_momentum":
        return volume_momentum(volume, **params)
    elif name == "entropy":
        return entropy(close, **params)
    elif name == "mutual_information":
        return mutual_information(close, **params)
    elif name == "fti_lowpass":
        return fti_lowpass(close, **params)
    elif name == "fti_best_width":
        return fti_best_width(close, **params)
    elif name == "fti_best_period":
        return fti_best_period(close, **params)
    elif name == "fti_best_fti":
        return fti_best_fti(close, **params)
    else:
        raise ValueError(f"Unknown indicator: {name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(data: pd.DataFrame, groups: str | list[str] = "all", params: dict[str, dict] | None = None,
                     open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close",
                     volume_col: str = "tick_volume", prefix: str = "tm_") -> pd.DataFrame:
    """Compute a batch of Timothy Masters indicators from an OHLCV DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with OHLCV columns (or any subset required by the selected indicators).
    groups : str | list[str]
        Group name(s) or indicator name(s) to compute. Accepted group names:
        ``"all"``, ``"momentum"``, ``"trend"``, ``"variance"``, ``"volume"``,
        ``"information"``, ``"fti"``. May also be individual indicator names or
        a mixed list. Duplicates are deduplicated preserving order.
    params : dict[str, dict] | None
        Per-indicator parameter overrides. Each key must be a valid indicator
        name; the value dict is shallow-merged over the defaults.
    open_col : str
        Name of the open-price column (default ``"open"``).
    high_col : str
        Name of the high-price column (default ``"high"``).
    low_col : str
        Name of the low-price column (default ``"low"``).
    close_col : str
        Name of the close-price column (default ``"close"``).
    volume_col : str
        Name of the volume column (default ``"volume"``).
    prefix : str
        Prefix for output column names (default ``"tm_"``).

    Returns
    -------
    pd.DataFrame
        Same index as ``data``, one column per selected indicator named
        ``prefix + indicator_name``. Warmup bars contain NaN.

    Raises
    ------
    ValueError
        If an unknown group or indicator name is supplied.
    KeyError
        If a required OHLCV column is missing from ``data``.
    """
    names = _resolve_groups(groups)

    # Merge user params over defaults (shallow per-indicator)
    merged_params: dict[str, dict] = {}
    for name in names:
        base = dict(DEFAULT_PARAMS.get(name, {}))
        if params and name in params:
            base.update(params[name])
        merged_params[name] = base

    # Symbolic → actual column mapping
    col_map: dict[str, str] = {
        "open":   open_col,
        "high":   high_col,
        "low":    low_col,
        "close":  close_col,
        "volume": volume_col,
    }
    df_columns = set(data.columns)

    # Upfront column validation (before any computation)
    _validate_cols(names, col_map, df_columns)

    # Extract numpy arrays once
    n = len(data)
    _nan = np.full(n, np.nan)

    open_  = data[open_col].to_numpy(dtype=np.float64)  if open_col  in df_columns else _nan.copy()
    high   = data[high_col].to_numpy(dtype=np.float64)  if high_col  in df_columns else _nan.copy()
    low    = data[low_col].to_numpy(dtype=np.float64)   if low_col   in df_columns else _nan.copy()
    close  = data[close_col].to_numpy(dtype=np.float64) if close_col in df_columns else _nan.copy()
    volume = data[volume_col].to_numpy(dtype=np.float64)if volume_col in df_columns else _nan.copy()

    # Compute each indicator
    results: dict[str, np.ndarray] = {}
    for name in names:
        results[prefix + name] = _compute_one(name, merged_params[name], open_, high, low, close, volume)
    return pd.DataFrame(results, index=data.index)


def list_indicators(group: str = "all") -> list[str]:
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group!r}. Valid groups: {sorted(GROUPS.keys())}")
    return list(GROUPS[group])


def list_groups() -> list[str]:
    """Return all available group names."""
    return list(GROUPS.keys())
