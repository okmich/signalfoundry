"""Forward targets per axis, measured from the close of decision bar t over bars t+1..t+h, and HAR regressors.

A target is NaN wherever it is not cleanly measurable: the forward window (or the continuation lookback) runs past
the data, crosses a data gap, or touches an excluded clock window; a volatility window has zero realised variance;
the continuation lookback has no move to continue.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .clock import BarClock


class Axis(StrEnum):
    DIRECTION = "direction"
    PATH = "path"
    VOLATILITY = "volatility"


@dataclass(frozen=True)
class TargetKey:
    axis: Axis
    horizon: int
    lookback: int = 0  # continuation lookback; 0 for the axes that have none

    def __str__(self) -> str:
        return f"{self.axis}_h{self.horizon}" + (f"_L{self.lookback}" if self.lookback else "")


def forward_targets(close: pd.Series, clock: BarClock, axes, horizons, lookbacks) -> dict[TargetKey, np.ndarray]:
    """Raw forward target per (axis, horizon[, lookback]) as float arrays aligned to ``close``; NaN = not measurable."""
    log_close = np.log(close.to_numpy(dtype=float))
    n = len(log_close)
    returns = np.r_[np.nan, np.diff(log_close)]
    sq_cum = np.r_[0.0, np.cumsum(np.nan_to_num(returns[1:]) ** 2)]  # sq_cum[i] = sum of r_1^2..r_i^2
    gap_cum = np.cumsum(clock.gap_before)  # gap_cum[i] = gaps among the steps into bars 1..i
    excl_cum = np.r_[0, np.cumsum(clock.excluded)]  # excl_cum[i] = excluded bars among bars 0..i-1
    positions = np.arange(n)

    out: dict[TargetKey, np.ndarray] = {}
    for h in horizons:
        end = positions + h
        end_c = np.minimum(end, n - 1)
        # Bars t..t+h: the decision close is the entry price, so it must lie outside excluded windows too
        clean = (end < n) & (gap_cum[end_c] == gap_cum) & (excl_cum[end_c + 1] == excl_cum[positions])
        displacement = np.where(clean, log_close[end_c] - log_close, np.nan)

        if Axis.DIRECTION in axes:
            out[TargetKey(Axis.DIRECTION, h)] = displacement
        if Axis.VOLATILITY in axes:
            realised = np.where(clean, sq_cum[end_c] - sq_cum, 0.0)
            out[TargetKey(Axis.VOLATILITY, h)] = _log_vol(realised)
        if Axis.PATH in axes:
            for lookback in lookbacks:
                start = positions - lookback
                start_c = np.maximum(start, 0)
                back_clean = (start >= 0) & (gap_cum == gap_cum[start_c])
                back_clean &= excl_cum[positions + 1] == excl_cum[start_c + 1]
                move = np.sign(log_close - log_close[start_c])
                out[TargetKey(Axis.PATH, h, lookback)] = np.where(back_clean & (move != 0), move * displacement, np.nan)
    return out


def _log_vol(realised_variance: np.ndarray) -> np.ndarray:
    """log of realised volatility; NaN where the variance is zero, negative or missing."""
    positive = realised_variance > 0
    return np.where(positive, 0.5 * np.log(np.where(positive, realised_variance, 1.0)), np.nan)


def har_regressors(close: pd.Series, clock: BarClock, windows=("1h", "4h", "1D")) -> np.ndarray:
    """Log trailing realised volatility over each window, ending at bar t inclusive: (n, len(windows)), NaN if empty."""
    log_close = np.log(close.to_numpy(dtype=float))
    returns = np.r_[np.nan, np.diff(log_close)]
    sq_cum = np.r_[0.0, np.cumsum(np.nan_to_num(returns[1:]) ** 2)]
    positions = np.arange(len(log_close))
    columns = []
    for window in windows:
        start = positions - clock.bars(window)
        columns.append(_log_vol(np.where(start >= 0, sq_cum - sq_cum[np.maximum(start, 0)], 0.0)))
    return np.column_stack(columns)
