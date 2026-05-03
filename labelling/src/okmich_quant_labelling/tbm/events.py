"""Event sampling for triple-barrier labelling.

`cusum_filter` accumulates raw return values (NOT differences). Both arms reset
on any trigger so a single drift event is not double-counted.

`get_vertical_barrier` is iloc-positional in `num_bars` mode — it does not use
time arithmetic. This handles sparse and holiday-adjusted calendars correctly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

_NUMBA_WARNED = False


def _cusum_loop_python(values: np.ndarray, threshold: float) -> np.ndarray:
    s_pos = 0.0
    s_neg = 0.0
    triggers = np.zeros(values.shape[0], dtype=np.bool_)
    for i in range(values.shape[0]):
        v = values[i]
        s_pos = max(0.0, s_pos + v)
        s_neg = min(0.0, s_neg + v)
        if s_pos >= threshold:
            triggers[i] = True
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg <= -threshold:
            triggers[i] = True
            s_pos = 0.0
            s_neg = 0.0
    return triggers


if _NUMBA_AVAILABLE:
    _cusum_loop = njit(cache=True)(_cusum_loop_python)
else:
    _cusum_loop = _cusum_loop_python


def cusum_filter(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM event sampler.

    Input is returns (or vol-normalized returns), NOT prices, indexed by a
    monotonic DatetimeIndex. Do not pass `.diff()`-ed returns — that computes
    second differences and is incorrect.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"series.index must be a DatetimeIndex, got {type(series.index).__name__}")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series.index must be monotonic increasing")
    if not series.index.is_unique:
        raise ValueError("series.index must be unique")

    global _NUMBA_WARNED
    if not _NUMBA_AVAILABLE and not _NUMBA_WARNED:
        logger.warning("numba not available; cusum_filter will use a slower Python loop")
        _NUMBA_WARNED = True

    clean = series.dropna()
    if clean.empty:
        return pd.DatetimeIndex([])
    triggers = _cusum_loop(clean.to_numpy(np.float64), float(threshold))
    return pd.DatetimeIndex(clean.index[triggers])


def get_vertical_barrier(events: pd.DatetimeIndex, close: pd.Series, num_bars: Optional[int] = None,
                         num_days: Optional[float] = None) -> pd.Series:
    """Compute t1 (vertical barrier) for each event timestamp.

    Both `num_bars` and `num_days` modes require `events` to be a subset of
    `close.index`. Tz-awareness must match between `events` and `close.index`.
    """
    if (num_bars is None) == (num_days is None):
        raise ValueError("exactly one of num_bars or num_days must be provided")

    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError(f"close.index must be a DatetimeIndex, got {type(close.index).__name__}")
    if len(close.index) == 0:
        raise ValueError("close must be non-empty")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close.index must be monotonic increasing")
    if not close.index.is_unique:
        raise ValueError("close.index must be unique")

    if len(events) == 0:
        return pd.Series([], index=pd.DatetimeIndex([]), dtype="datetime64[ns]")

    close_tz = getattr(close.index, "tz", None)
    events_tz = getattr(events, "tz", None)
    if (close_tz is None) != (events_tz is None):
        raise ValueError(f"events tz={events_tz} and close.index tz={close_tz} mismatch")

    last_iloc = len(close.index) - 1
    t0_ilocs = close.index.get_indexer(events)
    if (t0_ilocs < 0).any():
        missing = events[t0_ilocs < 0]
        raise KeyError(f"{len(missing)} events not present in close.index (first: {missing[0]})")

    if num_bars is not None:
        if num_bars < 1:
            raise ValueError(f"num_bars must be >= 1, got {num_bars}")
        target_ilocs = np.minimum(t0_ilocs + num_bars, last_iloc)
    else:
        if num_days <= 0:
            raise ValueError(f"num_days must be > 0, got {num_days}")
        delta = pd.Timedelta(days=num_days)
        offsets = events + delta
        target_ilocs = close.index.searchsorted(offsets, side="left").astype(np.int64)
        target_ilocs = np.minimum(target_ilocs, last_iloc)

    collapsed = (target_ilocs <= t0_ilocs).sum()
    if collapsed > 0:
        logger.warning("%d events collapsed to t1==t0 (near end of series); they will be skipped by get_labels", collapsed)

    t1_values = close.index.to_numpy()[target_ilocs]
    return pd.Series(t1_values, index=events, name="t1")
