"""Canonical MACD / Wilder-ATR feature definitions shared with MQL5.

This module is the *reference implementation*. ``HmmFeatures.mqh`` mirrors it
line for line, and ``tests/hmm_export/test_parity.py`` binds the two.

Why this exists rather than a call to TA-Lib or ``iMACD``/``iATR``
------------------------------------------------------------------
The HMM must see the *same* observable at fit time and at inference time. Three
implementations disagree on the details:

* MetaTrader 5's built-in MACD smooths the signal line with a **simple** moving
  average, not an EMA.
* MT5's ``ExponentialMA`` seeds the recursion differently from TA-Lib's ``EMA``.
* TA-Lib's ``MACD`` nudges the fast/slow EMA start indices so the signal EMA
  lines up, which shifts every value slightly relative to the naive definition.

None of those differences raise an error. They shift the feature distribution,
the fitted Gaussian emissions are then evaluated off-distribution, and the
posterior quietly collapses onto one state. So the recursions are pinned here
explicitly and mirrored exactly, with a test to keep them that way.

Definitions (0-based indices over the bar array)::

    EMA(x, n)   k = 2 / (n + 1)
      ema[n-1] = mean(x[0 .. n-1])                 <- SMA seed
      ema[i]   = k * x[i] + (1 - k) * ema[i-1]     for i >= n

    MACD(fast, slow, signal)
      macd = EMA(close, fast) - EMA(close, slow)   valid from slow - 1
      sig  = EMA(macd[slow-1:], signal)            valid from slow + signal - 2
      hist = macd - sig

    TR[i]  = max(H-L, |H - C[i-1]|, |L - C[i-1]|)  for i >= 1
    ATR(n) Wilder, SMA-seeded
      atr[n] = mean(TR[1 .. n])
      atr[i] = (atr[i-1] * (n-1) + TR[i]) / n      for i > n
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Mql5Feature(StrEnum):
    """Feature vocabulary implemented on both sides.

    The ``*_ATR`` / ``*_CLOSE`` variants are dimensionless. Prefer them: raw
    MACD and ATR carry price units, so a Gaussian fitted at one price level
    evaluates its emissions far from the fitted support at another.
    """

    MACD = "macd"
    MACD_SIGNAL = "macd_signal"
    MACD_HIST = "macd_hist"
    MACD_ATR = "macd_atr"
    MACD_CLOSE = "macd_close"
    ATR = "atr"
    ATR_CLOSE = "atr_close"
    LOG_ATR = "log_atr"


#: Features carrying price units - unsafe across price levels and symbols.
PRICE_SCALED_FEATURES = frozenset({Mql5Feature.MACD, Mql5Feature.MACD_SIGNAL, Mql5Feature.MACD_HIST, Mql5Feature.ATR})


@dataclass(frozen=True)
class FeatureSpec:
    """An ordered feature recipe. Travels with the fitted model into MQL5.

    Column order is part of the contract: a transposed column produces a
    plausible-looking series and a meaningless state sequence.
    """

    names: tuple[Mql5Feature, ...]
    macd_fast: int = 18
    macd_slow: int = 40
    macd_signal: int = 11
    atr_period: int = 14

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("FeatureSpec.names must not be empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"FeatureSpec.names contains duplicates: {self.names}")
        for period_name in ("macd_fast", "macd_slow", "macd_signal", "atr_period"):
            if getattr(self, period_name) < 1:
                raise ValueError(f"FeatureSpec.{period_name} must be >= 1, got {getattr(self, period_name)}")
        if self.macd_slow <= self.macd_fast:
            raise ValueError(f"macd_slow ({self.macd_slow}) must exceed macd_fast ({self.macd_fast})")
        object.__setattr__(self, "names", tuple(Mql5Feature(n) for n in self.names))

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def warmup_bars(self) -> int:
        """First bar index at which every column is finite."""
        return max(self.macd_slow + self.macd_signal - 2, self.atr_period)

    @property
    def price_scaled(self) -> tuple[Mql5Feature, ...]:
        return tuple(n for n in self.names if n in PRICE_SCALED_FEATURES)


def ema(x: np.ndarray, period: int) -> np.ndarray:
    """SMA-seeded exponential moving average. NaN before index ``period - 1``.

    The loop is deliberate: the SMA seed makes a vectorised formulation
    awkward, and exactness against the MQL5 mirror matters more here than
    throughput (this runs once per fit, not per tick).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    out[period - 1] = x[:period].mean()
    k = 2.0 / (period + 1.0)
    for i in range(period, n):
        out[i] = k * x[i] + (1.0 - k) * out[i - 1]
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range. NaN at index 0, which has no previous close."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    out = np.full(high.shape[0], np.nan, dtype=np.float64)
    if high.shape[0] < 2:
        return out

    prev_close = close[:-1]
    out[1:] = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)))
    return out


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed ATR, SMA-seeded. NaN before index ``period``.

    The seed averages ``period`` true ranges starting at index 1, so the first
    finite value lands at index ``period`` - matching TA-Lib's lookback.
    """
    tr = true_range(high, low, close)
    n = tr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out

    out[period] = tr[1 : period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def macd(close: np.ndarray, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD line, signal line, histogram. See module docstring for the exact recursion."""
    close = np.asarray(close, dtype=np.float64)
    macd_line = ema(close, fast) - ema(close, slow)

    start = slow - 1
    signal_line = np.full(close.shape[0], np.nan, dtype=np.float64)
    if close.shape[0] > start:
        signal_line[start:] = ema(macd_line[start:], signal)

    return macd_line, signal_line, macd_line - signal_line


def build_features(high: np.ndarray, low: np.ndarray, close: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """Assemble the ``(T, D)`` observation matrix in the spec's column order.

    Rows before ``spec.warmup_bars`` are NaN, as are degenerate bars (zero ATR
    or zero close under a ratio feature). Callers fitting a model should drop
    non-finite rows; the MQL5 side skips exactly the same bars.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if not (high.shape == low.shape == close.shape):
        raise ValueError(f"high/low/close shape mismatch: {high.shape}, {low.shape}, {close.shape}")
    if high.ndim != 1:
        raise ValueError(f"high/low/close must be 1-D, got {high.ndim}-D")

    macd_line, signal_line, hist = macd(close, spec.macd_fast, spec.macd_slow, spec.macd_signal)
    atr = wilder_atr(high, low, close, spec.atr_period)

    with np.errstate(divide="ignore", invalid="ignore"):
        safe_atr = np.where(atr > 0.0, atr, np.nan)
        safe_close = np.where(close != 0.0, close, np.nan)
        columns = {
            Mql5Feature.MACD: macd_line,
            Mql5Feature.MACD_SIGNAL: signal_line,
            Mql5Feature.MACD_HIST: hist,
            Mql5Feature.MACD_ATR: macd_line / safe_atr,
            Mql5Feature.MACD_CLOSE: macd_line / safe_close,
            Mql5Feature.ATR: atr,
            Mql5Feature.ATR_CLOSE: atr / safe_close,
            Mql5Feature.LOG_ATR: np.log(safe_atr),
        }

    out = np.column_stack([columns[name] for name in spec.names]).astype(np.float64)

    # The MQL5 engine publishes nothing until every sub-state is warm, so a
    # column that happens to mature earlier must still be masked here.
    out[: spec.warmup_bars] = np.nan
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out
