from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_2d_statistic(statistic_matrix: NDArray) -> NDArray:
    arr = np.asarray(statistic_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"statistic_matrix must be 2D (T, K), got shape {arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError("statistic_matrix contains NaN or Inf values.")
    return arr


def _broadcast_positive(value: float | NDArray, k: int, name: str) -> NDArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(k, float(arr), dtype=np.float64)
    elif arr.shape == (k,):
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(f"{name} shape {arr.shape} incompatible with K={k}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values: {value}")
    if not np.all(out > 0.0):
        raise ValueError(f"{name} must be > 0 per direction, got {value}")
    return out


def soft_alarm_projection(statistic_matrix: NDArray, threshold: float | NDArray,
                          scale: float | NDArray | None = None) -> NDArray:
    """Heuristic (T, K) -> (T, K+1) direction-preserving simplex projection."""
    s = _validate_2d_statistic(statistic_matrix)
    t, k = s.shape
    thr = _broadcast_positive(threshold, k, "threshold")
    if scale is None:
        scl = thr / 4.0
    else:
        scl = _broadcast_positive(scale, k, "scale")

    # Per-direction sigmoid alarm score.
    z = (s - thr[None, :]) / scl[None, :]
    alarm = 1.0 / (1.0 + np.exp(-z))  # shape (T, K)

    out = np.empty((t, k + 1), dtype=np.float64)
    out[:, 0] = 1.0 - np.max(alarm, axis=1) if t > 0 else np.empty(0, dtype=np.float64)
    out[:, 1:] = alarm
    row_sums = out.sum(axis=1, keepdims=True)
    # Row sums are guaranteed > 0: alarm in [0,1] and 1 - max(alarm) in [0,1] sum to >= 1 - max + max = 1.
    out /= row_sums
    return out


def collapse_to_binary(soft_alarm_matrix: NDArray) -> NDArray:
    arr = np.asarray(soft_alarm_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"soft_alarm_matrix must be 2D (T, K+1) with K>=1, got shape {arr.shape}")
    out = np.empty((arr.shape[0], 2), dtype=np.float64)
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 1:].sum(axis=1)
    return out


def first_crossings(statistic_matrix: NDArray, threshold: float | NDArray) -> NDArray:
    s = _validate_2d_statistic(statistic_matrix)
    t, k = s.shape
    thr = np.asarray(threshold, dtype=np.float64)
    if thr.ndim == 0:
        thr_arr = np.full(k, float(thr), dtype=np.float64)
    elif thr.shape == (k,):
        thr_arr = thr.astype(np.float64, copy=False)
    else:
        raise ValueError(f"threshold shape {thr.shape} incompatible with K={k}")

    above = s > thr_arr[None, :]
    out = np.zeros_like(above)
    if t == 0:
        return out
    out[0] = above[0]
    out[1:] = above[1:] & ~above[:-1]
    return out


def accumulation_start(statistic_matrix: NDArray) -> NDArray:
    """First bar of the current positive-evidence run, shape (T, K), int.

    For each (t, k): if S[t, k] > 0, returns one bar past the most recent zero
    of S[:, k] (or 0 if no prior zero). If S[t, k] <= 0, the run has not
    started; returns t (the bar itself is the most recent zero — the next
    positive bar would mark the start).
    """
    s = _validate_2d_statistic(statistic_matrix)
    t, k = s.shape
    out = np.zeros((t, k), dtype=np.int64)
    if t == 0:
        return out
    for j in range(k):
        last_zero = -1  # so that accumulation_start = 0 if S > 0 from bar 0
        for i in range(t):
            if s[i, j] <= 0.0:
                last_zero = i
                out[i, j] = i  # current bar is itself the latest zero; no run yet
            else:
                out[i, j] = last_zero + 1
    return out
