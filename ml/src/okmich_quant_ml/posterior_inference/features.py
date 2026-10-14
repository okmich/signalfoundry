from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr, xlogy


def validate_posterior_matrix(probs: NDArray, func_name: str, eps: float = 1e-12,
                              normalize: bool = False, negativity_tol: float = 1e-9,
                              row_sum_tol: float = 1e-6) -> NDArray:
    """Validate a posterior matrix and optionally normalize rows onto the simplex.

    With ``normalize=False`` (default), the returned array is the input cast to ``float`` with no value-side mutation —
    use this for pure validation and for rearrangement transformers that must not silently alter probabilities. Rows must
    already sum to 1 within ``row_sum_tol``: a non-simplex row (e.g. ``[0.75, 0.75]``) would otherwise silently corrupt
    every downstream consumer of this validator — entropy, KL, top-prob, gate thresholds — since none of them renormalize.

    With ``normalize=True``, values are clipped to ``eps`` and each row is rescaled to sum to 1. Use this for calibration /
    smoothing transformers that need log-safe input.

    Values more negative than ``-negativity_tol`` are always rejected as upstream-model corruption regardless of ``normalize``.
    The default ``negativity_tol=1e-9`` accepts floating-point drift
    (cumsum / smoothing noise typically in the ``1e-15 — 1e-12`` range) while flagging actual negative probability mass.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError(f"{func_name} requires posterior matrix (T, K) with K >= 2, got shape={p.shape}")
    if p.size > 0 and not np.isfinite(p.sum()):
        raise ValueError(f"{func_name}: posterior matrix contains NaN or Inf values.")
    if p.size > 0 and p.min() < -negativity_tol:
        raise ValueError(
            f"{func_name}: posterior contains negative values below -{negativity_tol} (min={p.min():.3e}); "
            f"this indicates upstream model corruption, not floating-point drift."
        )
    if not normalize:
        # Tiny FP-noise negatives (within negativity_tol) were accepted above; clip
        # them to 0 silently so downstream log-based math (entropy, KL) is safe.
        # No-op when the input is already strictly non-negative.
        if p.size > 0 and p.min() < 0.0:
            p = np.maximum(p, 0.0)
        if p.size > 0:
            row_sums = p.sum(axis=1)
            max_dev = float(np.abs(row_sums - 1.0).max())
            if max_dev > row_sum_tol:
                raise ValueError(
                    f"{func_name}: posterior rows must sum to 1 (max deviation {max_dev:.3e} > {row_sum_tol}); "
                    f"a non-simplex row would silently corrupt every downstream consumer (entropy, KL, "
                    f"top-prob, gate thresholds), none of which renormalize."
                )
        return p

    if p.size > 0:
        # Checked on the pre-clip row sums deliberately: clip(p, eps, None) floors every entry to
        # at least eps, so a post-clip row sum is always >= K*eps > 0 and this guard would never
        # fire — silently turning a corrupted all-zero (or otherwise non-positive) row into a
        # valid-looking uniform posterior downstream.
        pre_clip_row_sums = p.sum(axis=1)
        if pre_clip_row_sums.min() <= 0.0:
            raise ValueError(
                f"{func_name}: posterior rows must have strictly positive sums (min={pre_clip_row_sums.min():.3e}); "
                f"an all-zero or non-positive row is upstream corruption, not a value clip() should paper over."
            )
    clipped = np.clip(p, eps, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    clipped /= row_sums
    return clipped


# Backwards-compatible internal alias; external packages should import the public ``validate_posterior_matrix``.
_validate_posterior_matrix = validate_posterior_matrix


def _validate_window(window: int, func_name: str) -> int:
    """Validate ``window`` at the API boundary and return it coerced to ``int``.

    Rejects bools and non-integral floats (via ``_require_integral``) before the ``>= 1`` check, so
    e.g. ``window=2.5`` fails here with a clear message instead of later inside a slice/range call.
    """
    window = _require_integral(window, f"{func_name}: window")
    if window < 1:
        raise ValueError(f"{func_name}: window must be >= 1, got {window}")
    return window


def _require_integral(value, name: str) -> int:
    """Coerce ``value`` to ``int``, rejecting bools and non-integral floats.

    ``bool`` is an ``int`` subclass in Python, so a hyperparameter like ``window=True`` would
    otherwise silently become ``window=1`` under a naive ``int(value)`` cast. A non-integral
    float (e.g. ``lag=2.9``) would otherwise silently truncate to ``2``, changing causal-alignment
    or stability behavior with no signal to the caller. Integral floats (``lag=2.0``) are accepted.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int, not bool (got {value!r})")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"{name} must be an integer value, got {value!r}")


def margin(probs: NDArray) -> NDArray:
    """Top-minus-second probability margin per row, shape ``(T,)``.

    Expects a posterior matrix ``(T, K)`` with ``K >= 2``; rejects NaN/Inf at the validation gate consistent with the rest
    of the package's public surface.
    """
    p = _validate_posterior_matrix(probs, "margin")
    partitioned = np.partition(p, -2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def top_prob(probs: NDArray) -> NDArray:
    """Top-class probability per row, shape ``(T,)``.

    Expects a posterior matrix ``(T, K)`` with ``K >= 2``; rejects NaN/Inf at the validation gate consistent with the rest
    of the package's public surface.
    """
    p = _validate_posterior_matrix(probs, "top_prob")
    return np.max(p, axis=1)


def entropy(probs: NDArray) -> NDArray:
    """Shannon entropy in nats per row, shape ``(T,)``.

    Expects a posterior matrix ``(T, K)`` with ``K >= 2``; rejects NaN/Inf at the validation gate consistent with the rest
    of the package's public surface.
    """
    p = _validate_posterior_matrix(probs, "entropy")
    return -xlogy(p, p).sum(axis=1)


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


def posterior_delta(probs: NDArray) -> NDArray:
    """Per-state posterior change from t-1 to t, shape ``(T, K)``.

    Row 0 is zeros (no prior row). Signed per-state change in belief between
    consecutive bars; pairs with ``step_kl`` (scalar magnitude) when consumers
    need direction as well as size of belief shift.
    """
    p = _validate_posterior_matrix(probs, "posterior_delta")
    out = np.zeros_like(p)
    if p.shape[0] > 1:
        out[1:] = p[1:] - p[:-1]
    return out


def rolling_flip_rate(probs: NDArray, window: int) -> NDArray:
    """Trailing rolling fraction of argmax changes over ``window`` steps, shape ``(T,)``.

    At row ``t`` the output is the mean of argmax-change indicators over
    ``[max(0, t - window + 1), t]``. The first-row change indicator is 0
    (no prior row to compare against), so the rolling rate starts at 0.
    """
    p = _validate_posterior_matrix(probs, "rolling_flip_rate")
    window = _validate_window(window, "rolling_flip_rate")
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
    window = _validate_window(window, "rolling_max_prob_std")
    return _rolling_std_1d(np.max(p, axis=1), window)


def rolling_entropy_std(probs: NDArray, window: int) -> NDArray:
    """Trailing rolling stdev of Shannon entropy over ``window`` steps, shape ``(T,)``."""
    p = _validate_posterior_matrix(probs, "rolling_entropy_std")
    window = _validate_window(window, "rolling_entropy_std")
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


class DwellStatistic(StrEnum):
    """Summary statistic for :func:`typical_dwell`."""

    MEDIAN = "median"
    MEAN = "mean"


def _argmax_runs(p: NDArray) -> tuple[NDArray, NDArray]:
    """Return ``(argmax, run_start)``: the argmax per row and the row at which its current run began."""
    argmax = np.argmax(p, axis=1)
    if argmax.size == 0:
        return argmax, np.zeros(0, dtype=np.int64)
    positions = np.arange(p.shape[0])
    change = np.concatenate([[True], argmax[1:] != argmax[:-1]])
    run_start = np.maximum.accumulate(np.where(change, positions, -1))
    return argmax, run_start


def switch_indicator(probs: NDArray) -> NDArray:
    """Hard regime-switch indicator ``1[argmax_t != argmax_{t-1}]`` per row, shape ``(T,)`` with int dtype.

    Row 0 is 0 (no prior row). Causal: row ``t`` reads rows ``t-1`` and ``t`` only.
    """
    p = _validate_posterior_matrix(probs, "switch_indicator")
    T = p.shape[0]
    out = np.zeros(T, dtype=np.int64)
    if T > 1:
        argmax = np.argmax(p, axis=1)
        out[1:] = argmax[1:] != argmax[:-1]
    return out


def posterior_shift(probs: NDArray) -> NDArray:
    """Soft switch magnitude: total-variation distance ``½ Σ_k |p_{t,k} − p_{t-1,k}|`` per row, shape ``(T,)``.

    The posterior mass that moved between consecutive bars, in ``[0, 1]``. Row 0 is 0. Unlike the overlap form
    ``1 − Σ_k p_{t-1,k} p_{t,k}``, it is 0 on a static posterior however uncertain (the overlap form reads 0.5 on a
    static uniform K=2 posterior), so it measures movement, not doubt. Causal.
    """
    p = _validate_posterior_matrix(probs, "posterior_shift")
    out = np.zeros(p.shape[0], dtype=float)
    if p.shape[0] > 1:
        out[1:] = 0.5 * np.abs(p[1:] - p[:-1]).sum(axis=1)
    return out


def last_transition(probs: NDArray) -> NDArray:
    """``(from_state, to_state)`` of the most recent hard switch at or before each row, shape ``(T, 2)`` int.

    Both columns are -1 until the first switch. Feeds transition-identity features (e.g. one-hot of the pair).
    Causal: row ``t`` depends on rows ``<= t`` only.
    """
    p = _validate_posterior_matrix(probs, "last_transition")
    T = p.shape[0]
    out = np.full((T, 2), -1, dtype=np.int64)
    if T < 2:
        return out
    argmax, run_start = _argmax_runs(p)
    has_switch = run_start > 0
    out[has_switch, 0] = argmax[run_start[has_switch] - 1]
    out[has_switch, 1] = argmax[run_start[has_switch]]
    return out


def entry_confidence(probs: NDArray) -> NDArray:
    """Top probability at the bar the current argmax run began, carried forward, shape ``(T,)``.

    How confidently the current regime was entered: a run that opened at 0.51 is a tentative call even if
    ``top_prob`` has since risen. Row 0 opens the first run. Causal.
    """
    p = _validate_posterior_matrix(probs, "entry_confidence")
    if p.shape[0] == 0:
        return np.zeros(0, dtype=float)
    _, run_start = _argmax_runs(p)
    return np.max(p, axis=1)[run_start]


def typical_dwell(probs: NDArray, statistic: DwellStatistic = DwellStatistic.MEDIAN) -> NDArray:
    """Per-state typical length of fully observed argmax runs, shape ``(K,)``; NaN for a state with none.

    The first run (may have begun before the window) and the last run (still open) are censored and excluded.
    This is a *training-window* statistic: fit it on train and pass it to :func:`dwell_ratio` on later data.
    Computing it on the evaluation window leaks that window's future run lengths into every row.
    """
    p = _validate_posterior_matrix(probs, "typical_dwell")
    statistic = DwellStatistic(statistic)
    K = p.shape[1]
    out = np.full(K, np.nan)
    T = p.shape[0]
    if T == 0:
        return out
    argmax = np.argmax(p, axis=1)
    starts = np.flatnonzero(np.concatenate([[True], argmax[1:] != argmax[:-1]]))
    lengths = np.diff(np.concatenate([starts, [T]]))
    states = argmax[starts]
    lengths, states = lengths[1:-1], states[1:-1]
    reducer = np.median if statistic is DwellStatistic.MEDIAN else np.mean
    for k in range(K):
        runs = lengths[states == k]
        if runs.size:
            out[k] = float(reducer(runs))
    return out


def dwell_ratio(probs: NDArray, typical: NDArray) -> NDArray:
    """Current run length over the current state's typical dwell, shape ``(T,)``.

    ``dwell_length(probs) / typical[argmax_t]`` with ``typical`` from :func:`typical_dwell` on a training window.
    Above 1: the regime has outlived its usual hold. NaN where the state's typical dwell is NaN. Causal given a
    frozen ``typical``.
    """
    p = _validate_posterior_matrix(probs, "dwell_ratio")
    typical = np.asarray(typical, dtype=float)
    if typical.shape != (p.shape[1],):
        raise ValueError(f"dwell_ratio: typical must have shape ({p.shape[1]},), got {typical.shape}")
    finite = typical[np.isfinite(typical)]
    if finite.size and finite.min() <= 0.0:
        raise ValueError(f"dwell_ratio: typical dwell lengths must be > 0, got min={finite.min()}")
    argmax, run_start = _argmax_runs(p)
    return (np.arange(p.shape[0]) - run_start + 1) / typical[argmax]


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


def _rolling_mean_2d(x: NDArray, window: int) -> NDArray:
    T = x.shape[0]
    if T == 0:
        return np.zeros_like(x, dtype=float)
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(x, axis=0)
    prev = np.zeros_like(cumsum)
    if T > window:
        prev[window:] = cumsum[:-window]
    window_sum = cumsum - prev
    denom = np.minimum(np.arange(1, T + 1), window).astype(float).reshape(-1, 1)
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
