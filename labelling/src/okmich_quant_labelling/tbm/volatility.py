"""Batch volatility estimation for offline label generation.

Each estimator tags its output `pd.Series` with `series.attrs['vol_kind']`
(one of `VolKind.RETURN`, `PRICE`, `ANNUALIZED`). `get_labels` reads this tag
and rejects incompatible kinds — preventing the silent-but-wrong scenario
where price-unit ATR is multiplied by entry price to produce nonsensical
barriers.

Spec-canonical estimator is `get_daily_vol` (RETURN kind: per-bar log-return std).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from okmich_quant_features.volatility import (
    atr,
    garman_klass_volatility,
    parkinson_volatility,
    rolling_volatility,
)


class VolatilityEstimator(str, Enum):
    EWM = "ewm"
    STD = "std"
    ATR = "atr"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"


class VolKind(str, Enum):
    RETURN = "return"          # unitless per-bar return-vol; valid input to get_labels
    PRICE = "price"             # in price units; NOT valid for get_labels without conversion
    ANNUALIZED = "annualized"   # annualized return-vol; NOT valid for get_labels without conversion


VOL_KIND_ATTR = "vol_kind"


def _tag(series: pd.Series, kind: VolKind, name: str) -> pd.Series:
    out = series.rename(name)
    out.attrs[VOL_KIND_ATTR] = kind.value
    return out


def get_daily_vol(close: pd.Series, span: int = 100, annualize: bool = False,
                  annualization_factor: float = 252.0) -> pd.Series:
    """EWM standard deviation of log returns. Tagged `vol_kind=RETURN` (or
    `ANNUALIZED` when `annualize=True`). The RETURN form is the canonical input
    to `get_labels`.
    """
    if span < 2:
        raise ValueError(f"span must be >= 2, got {span}")
    if not (np.isfinite(close.to_numpy(np.float64)).all()):
        raise ValueError("close contains non-finite values")
    if (close <= 0).any():
        raise ValueError("close must be strictly positive")

    log_ret = np.log(close / close.shift(1))
    vol = log_ret.ewm(span=span, adjust=False).std()
    if annualize:
        vol = vol * np.sqrt(annualization_factor)
        return _tag(vol, VolKind.ANNUALIZED, "daily_vol")
    return _tag(vol, VolKind.RETURN, "daily_vol")


def get_atr_vol(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ATR in price units (vol_kind=PRICE)."""
    _check_ohlc(prices)
    high = prices["high"].to_numpy(np.float64)
    low = prices["low"].to_numpy(np.float64)
    close = prices["close"].to_numpy(np.float64)
    atr_val, _ = atr(high, low, close, period=window)
    return _tag(pd.Series(atr_val, index=prices.index), VolKind.PRICE, "atr_vol")


def get_parkinson_vol(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson high-low range volatility (vol_kind=ANNUALIZED, per upstream)."""
    _check_ohlc(prices, need_open=False, need_close=False)
    high = prices["high"].to_numpy(np.float64)
    low = prices["low"].to_numpy(np.float64)
    vol_ann = parkinson_volatility(high, low, window)
    return _tag(pd.Series(vol_ann, index=prices.index), VolKind.ANNUALIZED, "parkinson_vol")


def get_garman_klass_vol(prices: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass OHLC volatility (vol_kind=ANNUALIZED, per upstream)."""
    _check_ohlc(prices)
    open_ = prices["open"].to_numpy(np.float64)
    high = prices["high"].to_numpy(np.float64)
    low = prices["low"].to_numpy(np.float64)
    close = prices["close"].to_numpy(np.float64)
    vol_ann = garman_klass_volatility(open_, high, low, close, window)
    return _tag(pd.Series(vol_ann, index=prices.index), VolKind.ANNUALIZED, "garman_klass_vol")


def get_std_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling std of log returns scaled to price units (vol_kind=PRICE)."""
    vol = rolling_volatility(close, window=window)
    return _tag(vol * close, VolKind.PRICE, "std_vol")


def _check_ohlc(prices: pd.DataFrame, need_open: bool = True, need_close: bool = True) -> None:
    required = {"high", "low"}
    if need_open:
        required.add("open")
    if need_close:
        required.add("close")
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}")
