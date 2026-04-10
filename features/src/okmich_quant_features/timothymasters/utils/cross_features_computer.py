"""
Batch computation layer for Timothy Masters paired-market (cross-market) indicators.

Usage
-----
>>> from okmich_quant_features.timothymasters.utils import compute_cross_features
>>> result = compute_cross_features(data1, data2)                    # all 5 → cm_* columns
>>> result = compute_cross_features(data1, data2, groups="correlation")   # 2 columns
>>> result = compute_cross_features(data1, data2, groups=["deviation", "trend_diff"])
>>> result = compute_cross_features(data1, data2, params={"deviation": {"period": 30}})

Both DataFrames must be **date-aligned** before passing — only include bars where both
markets have data on the same date. Alignment is the caller's responsibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from okmich_quant_features.timothymasters.cross_market.correlation import (
    correlation, delta_correlation,
)
from okmich_quant_features.timothymasters.cross_market.deviation import deviation
from okmich_quant_features.timothymasters.cross_market.purify import purify, log_purify
from okmich_quant_features.timothymasters.cross_market.trend_diff import trend_diff, cmma_diff


# ---------------------------------------------------------------------------
# Group constants (public)
# ---------------------------------------------------------------------------

CORRELATION_INDICATORS: list[str] = ["correlation", "delta_correlation"]
DEVIATION_INDICATORS: list[str] = ["deviation"]
PURIFY_INDICATORS: list[str] = ["purify", "log_purify"]
TREND_INDICATORS: list[str] = ["trend_diff", "cmma_diff"]

ALL_INDICATORS: list[str] = (
    CORRELATION_INDICATORS
    + DEVIATION_INDICATORS
    + PURIFY_INDICATORS
    + TREND_INDICATORS
)

GROUPS: dict[str, list[str]] = {
    "correlation": CORRELATION_INDICATORS,
    "deviation":   DEVIATION_INDICATORS,
    "purify":      PURIFY_INDICATORS,
    "trend":       TREND_INDICATORS,
    "all":         ALL_INDICATORS,
}


# ---------------------------------------------------------------------------
# Default parameters (public)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, dict] = {
    "correlation":       {"period": 63},
    "delta_correlation": {"period": 63, "delta_period": 63},
    "deviation":         {"period": 20, "smooth_period": 0},
    "purify":            {"lookback": 60, "trend_length": 20, "accel_length": 20, "vol_length": 20},
    "log_purify":        {"lookback": 60, "trend_length": 20, "accel_length": 20, "vol_length": 20},
    "trend_diff":        {"period": 20, "atr_period": 60},
    "cmma_diff":         {"period": 20, "atr_period": 60},
}


# ---------------------------------------------------------------------------
# Required columns per indicator (same schema assumed for both DataFrames)
# ---------------------------------------------------------------------------

_REQUIRED_COLS: dict[str, set[str]] = {
    "correlation":       {"close"},
    "delta_correlation": {"close"},
    "deviation":         {"close"},
    "purify":            {"close"},
    "log_purify":        {"close"},
    "trend_diff":        {"high", "low", "close"},
    "cmma_diff":         {"high", "low", "close"},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_ALL_INDICATOR_SET: set[str] = set(ALL_INDICATORS)
_GROUP_NAMES: set[str] = set(GROUPS.keys())


def _resolve_groups(groups: str | list[str]) -> list[str]:
    """Expand group/indicator name(s) to an ordered deduplicated list.

    Raises
    ------
    ValueError  If an unknown group or indicator name is encountered.
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


def _validate_cols(names: list[str], col_map: dict[str, str], df1_columns: set[str], df2_columns: set[str]) -> None:
    """Validate that all required columns are present in both DataFrames.

    Raises
    ------
    KeyError  If a required column is missing from either DataFrame.
    """
    for name in names:
        for sym_col in _REQUIRED_COLS.get(name, set()):
            actual = col_map[sym_col]
            if actual not in df1_columns:
                raise KeyError(
                    f"Indicator {name!r} requires column {sym_col!r} "
                    f"(mapped to {actual!r}), but it is not in data1. "
                    f"Available columns: {sorted(df1_columns)}"
                )
            if actual not in df2_columns:
                raise KeyError(
                    f"Indicator {name!r} requires column {sym_col!r} "
                    f"(mapped to {actual!r}), but it is not in data2. "
                    f"Available columns: {sorted(df2_columns)}"
                )


def _compute_one(name: str, params: dict, high1: np.ndarray, low1: np.ndarray, close1: np.ndarray,
                 high2: np.ndarray, low2: np.ndarray, close2: np.ndarray) -> np.ndarray:
    """Dispatch a single cross-market indicator by name."""
    if name == "correlation":
        return correlation(close1, close2, **params)
    elif name == "delta_correlation":
        return delta_correlation(close1, close2, **params)
    elif name == "deviation":
        return deviation(close1, close2, **params)
    elif name == "purify":
        return purify(close1, close2, **params)
    elif name == "log_purify":
        return log_purify(close1, close2, **params)
    elif name == "trend_diff":
        return trend_diff(high1, low1, close1, high2, low2, close2, **params)
    elif name == "cmma_diff":
        return cmma_diff(high1, low1, close1, high2, low2, close2, **params)
    else:
        raise ValueError(f"Unknown indicator: {name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cross_features(data1: pd.DataFrame, data2: pd.DataFrame, groups: str | list[str] = "all",
                           params: dict[str, dict] | None = None, close_col: str = "close", high_col: str = "high",
                           low_col: str = "low", prefix: str = "cm_") -> pd.DataFrame:
    if len(data1) != len(data2):
        raise ValueError(
            f"data1 and data2 must have the same length; "
            f"got {len(data1)} and {len(data2)}."
        )

    names = _resolve_groups(groups)

    # Merge user params over defaults (shallow per-indicator)
    merged_params: dict[str, dict] = {}
    for name in names:
        base = dict(DEFAULT_PARAMS.get(name, {}))
        if params and name in params:
            base.update(params[name])
        merged_params[name] = base

    # Symbolic → actual column mapping (same schema for both DataFrames)
    col_map: dict[str, str] = {
        "close": close_col,
        "high":  high_col,
        "low":   low_col,
    }
    df1_columns = set(data1.columns)
    df2_columns = set(data2.columns)

    # Upfront column validation (before any computation)
    _validate_cols(names, col_map, df1_columns, df2_columns)

    # Extract numpy arrays once
    n = len(data1)
    _nan = np.full(n, np.nan)

    close1 = data1[close_col].to_numpy(dtype=np.float64) if close_col in df1_columns else _nan.copy()
    high1  = data1[high_col].to_numpy(dtype=np.float64)  if high_col  in df1_columns else _nan.copy()
    low1   = data1[low_col].to_numpy(dtype=np.float64)   if low_col   in df1_columns else _nan.copy()

    close2 = data2[close_col].to_numpy(dtype=np.float64) if close_col in df2_columns else _nan.copy()
    high2  = data2[high_col].to_numpy(dtype=np.float64)  if high_col  in df2_columns else _nan.copy()
    low2   = data2[low_col].to_numpy(dtype=np.float64)   if low_col   in df2_columns else _nan.copy()

    # Compute each indicator
    results: dict[str, np.ndarray] = {}
    for name in names:
        results[prefix + name] = _compute_one(
            name, merged_params[name],
            high1, low1, close1,
            high2, low2, close2,
        )

    return pd.DataFrame(results, index=data1.index)


def list_indicators(group: str = "all") -> list[str]:
    """Return indicator names for a group (or all indicators)."""
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group!r}. Valid groups: {sorted(GROUPS.keys())}")
    return list(GROUPS[group])


def list_groups() -> list[str]:
    """Return all available group names."""
    return list(GROUPS.keys())
