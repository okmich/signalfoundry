"""Causal BOCPD posterior summaries for ruptures→BOCPD evaluation (spec §6).

Every function takes a ``(T, r_max)`` posterior matrix, validates its 2-D shape, finiteness, and non-negativity
(row sums are not enforced — pass through operations may produce slightly off-simplex inputs and that is a caller
concern), and returns a per-bar feature. These are the canonical formulas; consumers MUST NOT re-derive ``posterior_js_innovation`` ad-hoc —
the growth-predicted posterior has a non-obvious final-bin folding rule that scratch implementations routinely get wrong.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_posterior(posterior: NDArray) -> NDArray:
    arr = np.asarray(posterior, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"posterior must be 2-D (T, r_max), got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"r_max must be >= 2, got posterior.shape[1]={arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("posterior contains NaN or Inf values.")
    if (arr < 0.0).any():
        raise ValueError("posterior contains negative values.")
    return arr


def cp_prob(posterior: NDArray) -> NDArray:
    """``P(r_t = 0 | x_{1:t})`` per bar. Shape ``(T,)``."""
    arr = _validate_posterior(posterior)
    return arr[:, 0].copy()


def map_run_length(posterior: NDArray) -> NDArray:
    """``argmax_r posterior[t, r]`` per bar. Shape ``(T,)``, dtype int64."""
    arr = _validate_posterior(posterior)
    return np.argmax(arr, axis=1).astype(np.int64)


def expected_run_length(posterior: NDArray) -> NDArray:
    """Posterior mean run-length ``Σ_r r · posterior[t, r]`` per bar.

    Shape ``(T,)``. The final column ``r_max - 1`` is a saturated bin representing ``P(r_t >= r_max - 1)``, so this
    statistic systematically underestimates for runs that extend beyond ``r_max - 1``. Caller's responsibility to size
    ``r_max`` accordingly.
    """
    arr = _validate_posterior(posterior)
    r_max = arr.shape[1]
    weights = np.arange(r_max, dtype=np.float64)
    return arr @ weights


def entropy(posterior: NDArray) -> NDArray:
    """Per-bar Shannon entropy in nats. Shape ``(T,)``.

    Uses the convention ``0 · log 0 := 0`` for zero-mass slots.
    """
    arr = _validate_posterior(posterior)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(arr > 0.0, np.log(arr), 0.0)
    return -(arr * log_p).sum(axis=1)


def mass_below_k(posterior: NDArray, k: int) -> NDArray:
    """``Σ_{r < k} posterior[t, r]`` — mass on young runs. Shape ``(T,)``.

    ``k`` is a research input recorded alongside any feature/target study.
    """
    arr = _validate_posterior(posterior)
    if int(k) < 1 or int(k) > arr.shape[1]:
        raise ValueError(f"k must be in [1, r_max={arr.shape[1]}], got {k}")
    return arr[:, : int(k)].sum(axis=1)


def _growth_predicted(p_prev: NDArray, hazard_rate: float) -> NDArray:
    """Slot-by-slot growth-predicted posterior matching the BOCPD recursion's cap.

    Folds incoming mass from both ``R - 2`` and the cap ``R - 1`` itself into the final slot, so the predicted vector sums
    to 1 by construction.
    """
    h = float(hazard_rate)
    r_max = p_prev.shape[0]
    g = np.empty(r_max, dtype=np.float64)
    g[0] = h
    if r_max >= 3:
        g[1 : r_max - 1] = (1.0 - h) * p_prev[: r_max - 2]
    g[r_max - 1] = (1.0 - h) * (p_prev[r_max - 2] + p_prev[r_max - 1])
    return g


def _jsd(p: NDArray, q: NDArray) -> float:
    """Jensen-Shannon divergence in nats with ``0 · log 0 := 0``."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log(p), 0.0)
        log_q = np.where(q > 0.0, np.log(q), 0.0)
        log_m_for_p = np.where(p > 0.0, np.log(m), 0.0)
        log_m_for_q = np.where(q > 0.0, np.log(m), 0.0)
    kl_pm = float(np.sum(p * (log_p - log_m_for_p)))
    kl_qm = float(np.sum(q * (log_q - log_m_for_q)))
    return 0.5 * (kl_pm + kl_qm)


def posterior_js_innovation(posterior: NDArray, hazard_rate: float) -> NDArray:
    """JSD between actual posterior and the growth-only predicted posterior.

    For each bar ``t >= 1``, builds the growth-predicted vector ``g[t]`` from ``posterior[t-1]`` per the BOCPD recursion
    (see ``_growth_predicted``) and returns ``JSD(posterior[t] || g[t])``. The bar at ``t = 0`` is undefined and reported
    as ``NaN`` — callers should mask it.

    Row-to-row ``JSD(posterior[t] || posterior[t-1])`` is **not** equivalent: under no-change BOCPD shifts mass from slot
    ``r`` to ``r + 1`` deterministically, which makes raw row-to-row JSD flag ordinary aging as belief change. KL is not
    used because BOCPD posteriors have zero mass on un-reached run lengths, making ``KL`` infinite or undefined.

    Output shape ``(T,)``; ``out[0] = NaN``.
    """
    arr = _validate_posterior(posterior)
    h = float(hazard_rate)
    if not (0.0 < h < 1.0):
        raise ValueError(f"hazard_rate must be in (0, 1), got {hazard_rate}")
    t_total, r_max = arr.shape
    out = np.full(t_total, np.nan, dtype=np.float64)
    if t_total < 2:
        return out

    p_prev = arr[:-1]
    p = arr[1:]
    growth = np.empty_like(p_prev)
    growth[:, 0] = h
    if r_max >= 3:
        growth[:, 1:-1] = (1.0 - h) * p_prev[:, :-2]
    growth[:, -1] = (1.0 - h) * (p_prev[:, -2] + p_prev[:, -1])

    m = 0.5 * (p + growth)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_m = np.where(m > 0.0, np.log(m), 0.0)
        log_p = np.where(p > 0.0, np.log(p), 0.0)
        log_g = np.where(growth > 0.0, np.log(growth), 0.0)
    kl_pm = np.where(p > 0.0, p * (log_p - log_m), 0.0).sum(axis=1)
    kl_gm = np.where(growth > 0.0, growth * (log_g - log_m), 0.0).sum(axis=1)
    out[1:] = 0.5 * (kl_pm + kl_gm)
    return out
