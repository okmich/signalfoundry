"""Forward returns and descriptive filters for gating a signal.

What is left of the former label-to-signal pipeline after the 2026-10 evaluator review: its label scoring (codes
treated as ordinal, same-bar defaults, in-sample selection, zero-cost backtest) was retired. Forward separability of a
label lives in ``okmich_quant_labelling.utils.regime_diagnostics``.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Dict, Sequence

import numpy as np
import pandas as pd


class FilterType(StrEnum):
    CATEGORICAL = "categorical"
    THRESHOLD = "threshold"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df is indexed by DatetimeIndex (or convert from 'Date' column)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "Date" in df.columns:
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index("Date").sort_index()
        return out
    raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")


def prepare_returns(df: pd.DataFrame, price_col: str = "close", log_returns: bool = True,
                    horizons: Sequence[int] = (10,)) -> pd.DataFrame:
    """Add 'ret' (the same-bar return, close t-1 to t) and forward returns 'fwd_ret_H' (close t to t+H) per horizon.

    Horizons count bars: a window that spans a weekend, session break or the FX roll is not masked here.
    """
    df = _ensure_datetime_index(df.copy())
    px = df[price_col].astype(float)

    if log_returns:
        df["ret"] = np.log(px).diff()
        for h in horizons:
            df[f"fwd_ret_{h}"] = np.log(px.shift(-h) / px)
    else:
        df["ret"] = px.pct_change()
        for h in horizons:
            df[f"fwd_ret_{h}"] = px.shift(-h).div(px).sub(1.0)

    return df


def apply_filters(signal: pd.Series, df: pd.DataFrame, filters: Dict[str, Dict]) -> pd.Series:
    """
    Gate a signal by descriptive filters: categorical filters (e.g. market regimes) and threshold filters (e.g.
    volatility bounds). The signal is set to 0 where a filter is not met. Causal when the filter columns are.

    filters spec example:
      {'vol_regime': {'type': 'categorical', 'allow_values': [0]},
       'ATR': {'type': 'threshold', 'min': 0.001, 'max': 0.01}}
    """
    sig = signal.copy().astype(float)
    mask = pd.Series(True, index=df.index)
    for col, spec in (filters or {}).items():
        if col not in df.columns:
            continue
        if spec.get("type") == FilterType.CATEGORICAL:
            allow = set(spec.get("allow_values", []))
            mask &= df[col].isin(allow)
        elif spec.get("type") == FilterType.THRESHOLD:
            mn, mx = spec.get("min", -np.inf), spec.get("max", np.inf)
            mask &= df[col].between(mn, mx)
    sig[~mask] = 0.0
    return sig
