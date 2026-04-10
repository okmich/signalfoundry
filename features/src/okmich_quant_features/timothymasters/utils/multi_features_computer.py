"""
Batch computation layer for Timothy Masters multi-market indicators.

Usage
-----
>>> from okmich_quant_features.timothymasters.utils import compute_multi_features
>>> result = compute_multi_features(markets)                            # all 40 → mm_* columns
>>> result = compute_multi_features(markets, groups="risk")             # 5 columns
>>> result = compute_multi_features(markets, groups=["trend_stats", "risk"])
>>> result = compute_multi_features(markets, params={"mahal": {"lookback": 60}})
>>> result = compute_multi_features(markets, groups="janus")            # 25 columns

All DataFrames must be **date-aligned** before passing — only include bars where all
markets have data on the same date.  Alignment is the caller's responsibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from okmich_quant_features.timothymasters.multi_market.portfolio_stats import (
    trend_rank, trend_median, trend_range, trend_iqr, trend_clump,
    cmma_rank, cmma_median, cmma_range, cmma_iqr, cmma_clump,
)
from okmich_quant_features.timothymasters.multi_market.risk import (
    mahal, abs_ratio, abs_shift, coherence, delta_coherence,
)
from okmich_quant_features.timothymasters.multi_market.janus import (
    Janus,
    janus_market_index,
    janus_rs,
    janus_rs_fractile,
    janus_delta_rs_fractile,
    janus_rss,
    janus_delta_rss,
    janus_dom,
    janus_doe,
    janus_dom_index,
    janus_rm,
    janus_rm_fractile,
    janus_delta_rm_fractile,
    janus_rs_leader_equity,
    janus_rs_laggard_equity,
    janus_rs_ps,
    janus_rs_leader_advantage,
    janus_rs_laggard_advantage,
    janus_rm_leader_equity,
    janus_rm_laggard_equity,
    janus_rm_ps,
    janus_rm_leader_advantage,
    janus_rm_laggard_advantage,
    janus_oos_avg,
    janus_cma_oos,
    janus_leader_cma_oos,
)


# ---------------------------------------------------------------------------
# Group constants (public)
# ---------------------------------------------------------------------------

TREND_STATS_INDICATORS: list[str] = ["trend_rank", "trend_median", "trend_range", "trend_iqr", "trend_clump",]

CMMA_STATS_INDICATORS: list[str] = ["cmma_rank", "cmma_median", "cmma_range", "cmma_iqr", "cmma_clump",]

RISK_INDICATORS: list[str] = ["mahal", "abs_ratio", "abs_shift", "coherence", "delta_coherence",]

JANUS_INDICATORS: list[str] = [
    "janus_market_index",
    "janus_rs",
    "janus_rs_fractile",
    "janus_delta_rs_fractile",
    "janus_rss",
    "janus_delta_rss",
    "janus_dom",
    "janus_doe",
    "janus_dom_index",
    "janus_rm",
    "janus_rm_fractile",
    "janus_delta_rm_fractile",
    "janus_rs_leader_equity",
    "janus_rs_laggard_equity",
    "janus_rs_ps",
    "janus_rs_leader_advantage",
    "janus_rs_laggard_advantage",
    "janus_rm_leader_equity",
    "janus_rm_laggard_equity",
    "janus_rm_ps",
    "janus_rm_leader_advantage",
    "janus_rm_laggard_advantage",
    "janus_oos_avg",
    "janus_cma_oos",
    "janus_leader_cma_oos",
]

ALL_INDICATORS: list[str] = (
    TREND_STATS_INDICATORS
    + CMMA_STATS_INDICATORS
    + RISK_INDICATORS
    + JANUS_INDICATORS
)

GROUPS: dict[str, list[str]] = {
    "trend_stats": TREND_STATS_INDICATORS,
    "cmma_stats":  CMMA_STATS_INDICATORS,
    "risk":        RISK_INDICATORS,
    "janus":       JANUS_INDICATORS,
    "all":         ALL_INDICATORS,
}


# ---------------------------------------------------------------------------
# Default parameters (public)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, dict] = {
    "trend_rank":      {"period": 20, "atr_period": 60},
    "trend_median":    {"period": 20, "atr_period": 60},
    "trend_range":     {"period": 20, "atr_period": 60},
    "trend_iqr":       {"period": 20, "atr_period": 60},
    "trend_clump":     {"period": 20, "atr_period": 60},
    "cmma_rank":       {"period": 20, "atr_period": 60},
    "cmma_median":     {"period": 20, "atr_period": 60},
    "cmma_range":      {"period": 20, "atr_period": 60},
    "cmma_iqr":        {"period": 20, "atr_period": 60},
    "cmma_clump":      {"period": 20, "atr_period": 60},
    "mahal":           {"lookback": 120, "smoothing": 0},
    "abs_ratio":       {"lookback": 120, "fraction": 0.2},
    "abs_shift":       {"lookback": 120, "fraction": 0.2, "long_lookback": 60, "short_lookback": 10},
    "coherence":       {"lookback": 120},
    "delta_coherence": {"lookback": 120, "delta_length": 20},
    # janus
    "janus_market_index":         {"lookback": 252},
    "janus_rs":                   {"lookback": 252, "market": 0},
    "janus_rs_fractile":          {"lookback": 252, "market": 0},
    "janus_delta_rs_fractile":    {"lookback": 252, "market": 0, "delta_length": 20},
    "janus_rss":                  {"lookback": 252, "smoothing": 0},
    "janus_delta_rss":            {"lookback": 252, "smoothing": 0},
    "janus_dom":                  {"lookback": 252, "market": 0},
    "janus_doe":                  {"lookback": 252, "market": 0},
    "janus_dom_index":            {"lookback": 252},
    "janus_rm":                   {"lookback": 252, "market": 0},
    "janus_rm_fractile":          {"lookback": 252, "market": 0},
    "janus_delta_rm_fractile":    {"lookback": 252, "market": 0, "delta_length": 20},
    "janus_rs_leader_equity":     {"lookback": 252},
    "janus_rs_laggard_equity":    {"lookback": 252},
    "janus_rs_ps":                {"lookback": 252},
    "janus_rs_leader_advantage":  {"lookback": 252},
    "janus_rs_laggard_advantage": {"lookback": 252},
    "janus_rm_leader_equity":     {"lookback": 252},
    "janus_rm_laggard_equity":    {"lookback": 252},
    "janus_rm_ps":                {"lookback": 252},
    "janus_rm_leader_advantage":  {"lookback": 252},
    "janus_rm_laggard_advantage": {"lookback": 252},
    "janus_oos_avg":              {"lookback": 252},
    "janus_cma_oos":              {"lookback": 252},
    "janus_leader_cma_oos":       {"lookback": 252},
}


# ---------------------------------------------------------------------------
# Required columns per indicator
# ---------------------------------------------------------------------------

_REQUIRED_COLS: dict[str, set[str]] = {
    "trend_rank":      {"high", "low", "close"},
    "trend_median":    {"high", "low", "close"},
    "trend_range":     {"high", "low", "close"},
    "trend_iqr":       {"high", "low", "close"},
    "trend_clump":     {"high", "low", "close"},
    "cmma_rank":       {"high", "low", "close"},
    "cmma_median":     {"high", "low", "close"},
    "cmma_range":      {"high", "low", "close"},
    "cmma_iqr":        {"high", "low", "close"},
    "cmma_clump":      {"high", "low", "close"},
    "mahal":           {"close"},
    "abs_ratio":       {"close"},
    "abs_shift":       {"close"},
    "coherence":       {"close"},
    "delta_coherence": {"close"},
    # janus — all need only close
    "janus_market_index":         {"close"},
    "janus_rs":                   {"close"},
    "janus_rs_fractile":          {"close"},
    "janus_delta_rs_fractile":    {"close"},
    "janus_rss":                  {"close"},
    "janus_delta_rss":            {"close"},
    "janus_dom":                  {"close"},
    "janus_doe":                  {"close"},
    "janus_dom_index":            {"close"},
    "janus_rm":                   {"close"},
    "janus_rm_fractile":          {"close"},
    "janus_delta_rm_fractile":    {"close"},
    "janus_rs_leader_equity":     {"close"},
    "janus_rs_laggard_equity":    {"close"},
    "janus_rs_ps":                {"close"},
    "janus_rs_leader_advantage":  {"close"},
    "janus_rs_laggard_advantage": {"close"},
    "janus_rm_leader_equity":     {"close"},
    "janus_rm_laggard_equity":    {"close"},
    "janus_rm_ps":                {"close"},
    "janus_rm_leader_advantage":  {"close"},
    "janus_rm_laggard_advantage": {"close"},
    "janus_oos_avg":              {"close"},
    "janus_cma_oos":              {"close"},
    "janus_leader_cma_oos":       {"close"},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_ALL_INDICATOR_SET: set[str] = set(ALL_INDICATORS)
_GROUP_NAMES: set[str] = set(GROUPS.keys())


def _resolve_groups(groups: str | list[str]) -> list[str]:
    """Expand group/indicator name(s) to an ordered deduplicated list."""
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


def _validate_cols(names: list[str], col_map: dict[str, str], all_columns: list[set[str]]) -> None:
    """Validate that all required columns are present in all DataFrames."""
    for name in names:
        for sym_col in _REQUIRED_COLS.get(name, set()):
            actual = col_map[sym_col]
            for i, cols in enumerate(all_columns):
                if actual not in cols:
                    raise KeyError(
                        f"Indicator {name!r} requires column {sym_col!r} "
                        f"(mapped to {actual!r}), but it is missing from markets[{i}]. "
                        f"Available columns: {sorted(cols)}"
                    )


_JANUS_INDICATOR_SET: set[str] = set(JANUS_INDICATORS)


def _compute_one(name: str, params: dict, highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray]) -> np.ndarray:
    """Dispatch a single multi-market indicator by name."""
    if name == "trend_rank":
        return trend_rank(highs, lows, closes, **params)
    elif name == "trend_median":
        return trend_median(highs, lows, closes, **params)
    elif name == "trend_range":
        return trend_range(highs, lows, closes, **params)
    elif name == "trend_iqr":
        return trend_iqr(highs, lows, closes, **params)
    elif name == "trend_clump":
        return trend_clump(highs, lows, closes, **params)
    elif name == "cmma_rank":
        return cmma_rank(highs, lows, closes, **params)
    elif name == "cmma_median":
        return cmma_median(highs, lows, closes, **params)
    elif name == "cmma_range":
        return cmma_range(highs, lows, closes, **params)
    elif name == "cmma_iqr":
        return cmma_iqr(highs, lows, closes, **params)
    elif name == "cmma_clump":
        return cmma_clump(highs, lows, closes, **params)
    elif name == "mahal":
        return mahal(closes, **params)
    elif name == "abs_ratio":
        return abs_ratio(closes, **params)
    elif name == "abs_shift":
        return abs_shift(closes, **params)
    elif name == "coherence":
        return coherence(closes, **params)
    elif name == "delta_coherence":
        return delta_coherence(closes, **params)
    elif name in _JANUS_INDICATOR_SET:
        # Dispatched via _compute_janus_batch for efficiency
        raise RuntimeError(f"JANUS indicator {name!r} should be dispatched via _compute_janus_batch")
    else:
        raise ValueError(f"Unknown indicator: {name!r}")


def _extract_janus_indicator(j: Janus, name: str, params: dict) -> np.ndarray:
    """Extract a single indicator from a pre-built Janus object."""
    from okmich_quant_features.timothymasters.multi_market.janus import _ema_smooth

    if name == "janus_market_index":
        return j.market_index
    elif name == "janus_rs":
        return j.rs(params.get("market", 0))
    elif name == "janus_rs_fractile":
        return j.rs_fractile(params.get("market", 0))
    elif name == "janus_delta_rs_fractile":
        frac = j.rs_fractile(params.get("market", 0))
        dl = params.get("delta_length", 20)
        out = np.full_like(frac, np.nan)
        out[dl:] = frac[dl:] - frac[:-dl]
        return out
    elif name == "janus_rss":
        result = j.rss
        sm = params.get("smoothing", 0)
        return _ema_smooth(result, sm) if sm > 0 else result
    elif name == "janus_delta_rss":
        result = j.rss_change
        sm = params.get("smoothing", 0)
        return _ema_smooth(result, sm) if sm > 0 else result
    elif name == "janus_dom":
        return j.dom(params.get("market", 0))
    elif name == "janus_doe":
        return j.doe(params.get("market", 0))
    elif name == "janus_dom_index":
        return j.dom_index_equity
    elif name == "janus_rm":
        return j.rm(params.get("market", 0))
    elif name == "janus_rm_fractile":
        return j.rm_fractile(params.get("market", 0))
    elif name == "janus_delta_rm_fractile":
        frac = j.rm_fractile(params.get("market", 0))
        dl = params.get("delta_length", 20)
        out = np.full_like(frac, np.nan)
        out[dl:] = frac[dl:] - frac[:-dl]
        return out
    elif name == "janus_rs_leader_equity":
        return j.rs_leader_equity
    elif name == "janus_rs_laggard_equity":
        return j.rs_laggard_equity
    elif name == "janus_rs_ps":
        return j.rs_ps
    elif name == "janus_rs_leader_advantage":
        return j.rs_leader_advantage
    elif name == "janus_rs_laggard_advantage":
        return j.rs_laggard_advantage
    elif name == "janus_rm_leader_equity":
        return j.rm_leader_equity
    elif name == "janus_rm_laggard_equity":
        return j.rm_laggard_equity
    elif name == "janus_rm_ps":
        return j.rm_ps
    elif name == "janus_rm_leader_advantage":
        return j.rm_leader_advantage
    elif name == "janus_rm_laggard_advantage":
        return j.rm_laggard_advantage
    elif name == "janus_oos_avg":
        return j.oos_avg
    elif name == "janus_cma_oos":
        return j.cma_oos
    elif name == "janus_leader_cma_oos":
        return j.leader_cma_oos
    else:
        raise ValueError(f"Unknown JANUS indicator: {name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_multi_features(markets: list[pd.DataFrame], groups: str | list[str] = "all",
                           params: dict[str, dict] | None = None, close_col: str = "close", high_col: str = "high",
                           low_col: str = "low", prefix: str = "mm_") -> pd.DataFrame:
    """
    Compute Timothy Masters multi-market indicators for a list of date-aligned DataFrames.

    Parameters
    ----------
    markets : list of pd.DataFrame
        N date-aligned DataFrames.  All must have the same length.
        ``markets[0]`` is the target market for RANK indicators.
    groups : str or list of str
        Group name(s) or indicator name(s) to compute (default "all").
    params : dict, optional
        Per-indicator parameter overrides, e.g. ``{"mahal": {"lookback": 60}}``.
    close_col : str  Column name for close prices (default "close").
    high_col : str   Column name for high prices (default "high").
    low_col : str    Column name for low prices (default "low").
    prefix : str     Column prefix for output (default "mm_").

    Returns
    -------
    pd.DataFrame
        Columns: ``{prefix}{indicator_name}`` for each requested indicator.
        Index taken from ``markets[0]``.
    """
    if len(markets) < 2:
        raise ValueError(f"At least 2 markets required; got {len(markets)}.")

    n = len(markets[0])
    for i, df in enumerate(markets):
        if len(df) != n:
            raise ValueError(
                f"All DataFrames must have the same length; "
                f"markets[{i}] has {len(df)} rows, expected {n}."
            )

    names = _resolve_groups(groups)

    # Merge user params over defaults
    merged_params: dict[str, dict] = {}
    for name in names:
        base = dict(DEFAULT_PARAMS.get(name, {}))
        if params and name in params:
            base.update(params[name])
        merged_params[name] = base

    # Column mapping
    col_map: dict[str, str] = {
        "close": close_col,
        "high":  high_col,
        "low":   low_col,
    }
    all_columns = [set(df.columns) for df in markets]

    # Upfront column validation
    _validate_cols(names, col_map, all_columns)

    # Extract numpy arrays once
    _nan = np.full(n, np.nan)

    closes: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    lows: list[np.ndarray] = []

    for df in markets:
        df_cols = set(df.columns)
        closes.append(df[close_col].to_numpy(dtype=np.float64) if close_col in df_cols else _nan.copy())
        highs.append(df[high_col].to_numpy(dtype=np.float64) if high_col in df_cols else _nan.copy())
        lows.append(df[low_col].to_numpy(dtype=np.float64) if low_col in df_cols else _nan.copy())

    # Compute each indicator
    results: dict[str, np.ndarray] = {}

    # Separate JANUS indicators for batch optimization (single Janus object)
    janus_names = [n for n in names if n in _JANUS_INDICATOR_SET]
    non_janus_names = [n for n in names if n not in _JANUS_INDICATOR_SET]

    for name in non_janus_names:
        results[prefix + name] = _compute_one(name, merged_params[name], highs, lows, closes)

    if janus_names:
        # Group JANUS indicators by their shared core parameter tuple so that
        # indicators with different lookback/spread_tail/min_cma/max_cma each get
        # their own correctly configured Janus object.
        _janus_groups: dict[tuple, list[str]] = {}
        for name in janus_names:
            p = merged_params[name]
            key = (
                p.get("lookback", 252),
                p.get("spread_tail", 0.1),
                p.get("min_cma", 20),
                p.get("max_cma", 60),
            )
            _janus_groups.setdefault(key, []).append(name)

        for (lookback, spread_tail, min_cma, max_cma), group_names in _janus_groups.items():
            j = Janus(closes, lookback=lookback, spread_tail=spread_tail,
                      min_cma=min_cma, max_cma=max_cma)
            for name in group_names:
                results[prefix + name] = _extract_janus_indicator(j, name, merged_params[name])

    return pd.DataFrame(results, index=markets[0].index)


def list_indicators(group: str = "all") -> list[str]:
    """Return indicator names for a group (or all indicators)."""
    if group not in GROUPS:
        raise ValueError(f"Unknown group: {group!r}. Valid groups: {sorted(GROUPS.keys())}")
    return list(GROUPS[group])


def list_groups() -> list[str]:
    """Return all available group names."""
    return list(GROUPS.keys())
