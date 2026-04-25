from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr, xlogy


def _validate_posterior_matrix(probs: NDArray, func_name: str, eps: float = 1e-12,
                               normalize: bool = False) -> NDArray:
    """Validate a posterior matrix and optionally normalize rows onto the simplex.

    With ``normalize=False`` (default), the returned array is the input cast to
    ``float`` with no value-side mutation — use this for pure validation and for
    rearrangement transformers that must not silently alter probabilities.

    With ``normalize=True``, values are clipped to ``eps`` and each row is
    rescaled to sum to 1. Use this for calibration / smoothing transformers that
    need log-safe input.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"{func_name} requires posterior matrix (T, K) with K >= 2, got shape={p.shape}")
    if p.size > 0 and not np.isfinite(p.sum()):
        raise ValueError(f"{func_name}: posterior matrix contains NaN or Inf values.")
    if not normalize:
        return p

    clipped = np.clip(p, eps, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    if row_sums.size > 0 and row_sums.min() <= 0.0:
        raise ValueError(f"{func_name}: posterior rows must have strictly positive sums.")
    clipped /= row_sums
    return clipped


def _validate_window(window: int, func_name: str) -> None:
    if window < 1:
        raise ValueError(f"{func_name}: window must be >= 1, got {window}")


def margin(probs: NDArray) -> NDArray:
    """Return top-minus-second probability margin along the last axis."""
    p = np.asarray(probs)
    if p.shape[-1] < 2:
        raise ValueError(f"margin requires at least 2 classes on the last axis, got {p.shape[-1]}")
    partitioned = np.partition(p, -2, axis=-1)
    return partitioned[..., -1] - partitioned[..., -2]


def entropy(probs: NDArray) -> NDArray:
    """Return Shannon entropy in nats along the last axis."""
    p = np.asarray(probs)
    return -xlogy(p, p).sum(axis=-1)


def step_kl(probs: NDArray, eps: float = 1e-12) -> NDArray:
    """KL(p_t || p_{t-1}) per row, shape ``(T,)``. Row 0 is 0 (no prior row).

    Measures how much the posterior belief shifted from one step to the next.
    Zero when consecutive rows are identical; grows with the magnitude of the shift.
    ``p_{t-1}`` is clipped to ``eps`` to avoid division-by-zero when a prior-row
    component is exactly zero. ``p_t`` is not clipped because ``rel_entr`` already
    handles ``x == 0`` correctly.
    """
    p = _validate_posterior_matrix(probs, "step_kl")
    T = p.shape[0]
    out = np.zeros(T, dtype=float)
    if T <= 1:
        return out
    p_prev = np.clip(p[:-1], eps, None)
    p_curr = p[1:]
    out[1:] = rel_entr(p_curr, p_prev).sum(axis=1)
    return out


def rolling_flip_rate(probs: NDArray, window: int) -> NDArray:
    """Trailing rolling fraction of argmax changes over ``window`` steps, shape ``(T,)``.

    At row ``t`` the output is the mean of argmax-change indicators over
    ``[max(0, t - window + 1), t]``. The first-row change indicator is 0
    (no prior row to compare against), so the rolling rate starts at 0.
    """
    p = _validate_posterior_matrix(probs, "rolling_flip_rate")
    _validate_window(window, "rolling_flip_rate")
    T = p.shape[0]
    if T == 0:
        return np.zeros(0, dtype=float)
    argmax = np.argmax(p, axis=1)
    flips = np.zeros(T, dtype=float)
    if T > 1:
        flips[1:] = argmax[1:] != argmax[:-1]
    return _rolling_mean_1d(flips, window)


def rolling_max_prob_std(probs: NDArray, window: int) -> NDArray:
    """Trailing rolling stdev of top-probability over ``window`` steps, shape ``(T,)``.

    Captures the volatility of model confidence. High values mean the top class
    probability has been moving around; low values mean it has been steady
    regardless of its level.
    """
    p = _validate_posterior_matrix(probs, "rolling_max_prob_std")
    _validate_window(window, "rolling_max_prob_std")
    return _rolling_std_1d(np.max(p, axis=1), window)


def rolling_entropy_std(probs: NDArray, window: int) -> NDArray:
    """Trailing rolling stdev of Shannon entropy over ``window`` steps, shape ``(T,)``."""
    p = _validate_posterior_matrix(probs, "rolling_entropy_std")
    _validate_window(window, "rolling_entropy_std")
    return _rolling_std_1d(entropy(p), window)


def dwell_length(probs: NDArray) -> NDArray:
    """Current-regime run length at each row, shape ``(T,)`` with int dtype.

    Row 0 is 1 (regime just started). Increments while argmax is unchanged; resets
    to 1 when argmax changes.
    """
    p = _validate_posterior_matrix(probs, "dwell_length")
    T = p.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    argmax = np.argmax(p, axis=1)
    positions = np.arange(T)
    change = np.concatenate([[True], argmax[1:] != argmax[:-1]])
    last_change = np.maximum.accumulate(np.where(change, positions, -1))
    return (positions - last_change + 1).astype(np.int64)


def _rolling_mean_1d(x: NDArray, window: int) -> NDArray:
    T = len(x)
    if T == 0:
        return np.zeros(0, dtype=float)
    cumsum = np.cumsum(x, dtype=float)
    prev = np.zeros(T, dtype=float)
    if T > window:
        prev[window:] = cumsum[:-window]
    window_sum = cumsum - prev
    denom = np.minimum(np.arange(1, T + 1), window).astype(float)
    return window_sum / denom


def _rolling_std_1d(x: NDArray, window: int) -> NDArray:
    T = len(x)
    if T == 0:
        return np.zeros(0, dtype=float)
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(x)
    cumsum2 = np.cumsum(x * x)
    prev = np.zeros(T, dtype=float)
    prev2 = np.zeros(T, dtype=float)
    if T > window:
        prev[window:] = cumsum[:-window]
        prev2[window:] = cumsum2[:-window]
    window_sum = cumsum - prev
    window_sum2 = cumsum2 - prev2
    denom = np.minimum(np.arange(1, T + 1), window).astype(float)
    mean = window_sum / denom
    # Catastrophic cancellation in E[X^2] - (E[X])^2 leaves a tiny positive residual
    # (~1e-16 variance) when the input is exactly constant. Snap that to zero so
    # downstream callers do not have to reason about numeric noise. 1e-12 variance
    # equals a stdev of 1e-6, well below any meaningful signal on probability values.
    variance = window_sum2 / denom - mean * mean
    variance[variance < 1e-12] = 0.0
    return np.sqrt(variance)
