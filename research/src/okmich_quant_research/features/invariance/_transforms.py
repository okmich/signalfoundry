"""Exact, model-free transforms of an OHLC price path.

Both are applied in LOG-PRICE space about the first close, so they are exact on log-returns rather
than a linear approximation:

    reflect:  p' = c0**2 / p          # monotone DECREASING -> the bar's high and low MUST swap
    rescale:  p' = c0 * (p/c0)**c     # monotone increasing

Exactness is the whole point. Because the transforms introduce no approximation error, a feature's
response to them is a clean verdict rather than an estimate — which is why the parity round-trip is
asserted at a tolerance of 0.02 while measured values come back at 1e-5. Anything looser is hiding a
bug, not absorbing noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")


def _first_close(raw: pd.DataFrame) -> float:
    """Anchor for both transforms: the first valid close. Both maps fix it, so ``p'[0] == p[0]``."""
    valid = raw["close"].dropna()
    if valid.empty:
        raise ValueError("price frame has no valid 'close' values to anchor the transform on.")
    c0 = float(valid.iloc[0])
    if not np.isfinite(c0) or c0 <= 0.0:
        raise ValueError(f"anchor close must be finite and positive, got {c0!r}. "
                         "Both transforms are defined in log-price space.")
    return c0


def _require_ohlc(raw: pd.DataFrame) -> None:
    """Both transforms are defined in LOG-price space, so a non-positive price is not representable.

    Left unchecked it does not raise: ``c0**2 / 0`` is ``inf`` and ``(p/c0)**c`` on a negative base is
    NaN, so the feature pool silently fills with garbage, every IQR degenerates, and the whole pool
    reports UNSCORED — a result that looks like "these features could not be measured" rather than
    "the input was invalid". That is exactly the kind of quiet-wrong-answer this module exists to
    eliminate, so it is an error here.
    """
    missing = [c for c in OHLC if c not in raw.columns]
    if missing:
        raise ValueError(f"price frame is missing column(s) {missing}; need all of {list(OHLC)}.")
    prices = raw[list(OHLC)].to_numpy(dtype=float)
    if np.nanmin(prices) <= 0.0:
        raise ValueError("price frame contains non-positive OHLC values; both transforms are defined "
                         "in log-price space and cannot represent them.")


def reflect_ohlc(raw: pd.DataFrame) -> pd.DataFrame:
    """Negate every log-return: ``p' = c0**2 / p``, with high/low swapped because the map is decreasing.

    The swap is LOAD-BEARING, not tidiness. ``c0**2 / p`` is order-reversing, so the bar's old low maps
    above its old high. Omit the swap and the frame comes back with ``high < low`` on every bar; every
    true-range feature then silently returns garbage and the whole parity measurement is meaningless
    without ever raising. There is a test asserting ``high >= low`` for exactly this reason.

    Non-price columns (volume, spread) are passed through untouched: reflection is a statement about the
    price path only, which is also why a volume-substrate feature such as anything on the LIQUIDITY axis
    is invariant under it and cannot be classified by it.
    """
    _require_ohlc(raw)
    out = raw.copy()
    c0 = _first_close(raw)
    k = c0 * c0
    hi, lo = raw["high"].to_numpy(), raw["low"].to_numpy()
    for col in OHLC:
        out[col] = k / raw[col].to_numpy()
    out["high"] = k / lo                              # decreasing map: old low becomes the new high
    out["low"] = k / hi
    return out


def rescale_ohlc(raw: pd.DataFrame, c: float = 2.0) -> pd.DataFrame:
    """Multiply every log-deviation from ``c0`` (returns AND intrabar range) by ``c``. Order-preserving.

    A feature's spread under this map is what separates a size measure from a normalised one: if
    ``IQR(f')/IQR(f) ~ c`` the feature carries scale, if the ratio is ~1 it is scale-free. See
    ``_classify.scale_exponent``.
    """
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"rescale factor c must be finite and positive, got {c!r}.")
    _require_ohlc(raw)
    out = raw.copy()
    c0 = _first_close(raw)
    for col in OHLC:
        out[col] = c0 * np.power(raw[col].to_numpy() / c0, c)
    return out
