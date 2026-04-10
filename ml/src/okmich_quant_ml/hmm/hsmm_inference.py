"""Numba-accelerated HSMM inference engine (log-domain throughout).

Implements the pyhsmm-style begin/end variable decomposition with explicit right-censoring via the survivor function.

Message invariant
-----------------
- ``alpha_begin[t, j]`` = log prob of *entering* state j at time t,
  **excluding** any emissions from the new segment (pure routing).
- ``alpha_end[t, j]``   = log prob of a segment in state j *ending* at
  time t, **including** all emissions for that segment.

All functions operate on:
- ``log_emissions`` : shape (T, N), pre-computed from pomegranate dists
- ``log_trans``     : shape (N, N), **zero diagonal** (-inf on diagonal)
- ``log_init``      : shape (N,)
- ``duration_model``: :class:`BaseDuration` instance
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numba as nb
import numpy as np

logger = logging.getLogger(__name__)

from .duration._base import BaseDuration

_NEG_INF = -np.inf


# ------------------------------------------------------------------
# Numba primitives
# ------------------------------------------------------------------
@nb.njit(cache=True)
def _logsumexp_1d(arr: np.ndarray) -> float:
    """Log-sum-exp over a contiguous 1-D float64 array (Kahan-compensated)."""
    n = arr.shape[0]
    max_val = arr[0]
    for i in range(1, n):
        if arr[i] > max_val:
            max_val = arr[i]
    if max_val == -np.inf:
        return -np.inf
    # Kahan compensated summation to reduce floating-point drift
    s = 0.0
    c = 0.0  # running compensation
    for i in range(n):
        y = np.exp(arr[i] - max_val) - c
        t = s + y
        c = (t - s) - y
        s = t
    return max_val + np.log(s)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------
@dataclass
class HSMMInferenceResult:
    """Sufficient statistics returned by ``hsmm_forward_backward``."""
    state_posteriors: np.ndarray    # (T, N) — gamma[t, j]
    expected_transitions: np.ndarray  # (N, N) — xi[i, j]
    expected_durations: np.ndarray   # (N, M) — eta[j, u]
    log_likelihood: float


# ------------------------------------------------------------------
# Precomputation helpers
# ------------------------------------------------------------------
def _precompute_cumulative_emissions(log_emissions: np.ndarray) -> np.ndarray:
    """Cumulative sum of log-emissions per state, shape (T+1, N).

    ``cum[0, j] = 0``; ``cum[t+1, j] = sum_{s=0}^{t} log_emit[s, j]``.
    Segment likelihood ``sum_{s=a}^{b} log_emit[s, j] = cum[b+1, j] - cum[a, j]``.
    """
    T, N = log_emissions.shape
    cum = np.zeros((T + 1, N), dtype=np.float64)
    np.cumsum(log_emissions, axis=0, out=cum[1:])
    return cum


def _precompute_duration_tables(duration_model: BaseDuration, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_d, log_D), each shape (N, M).

    log_d[j] = log PMF, log_D[j] = log survivor function.
    """
    M = duration_model.max_duration
    log_d = np.empty((N, M), dtype=np.float64)
    log_D = np.empty((N, M), dtype=np.float64)
    for j in range(N):
        log_d[j] = duration_model.log_pmf(j)
        log_D[j] = duration_model.log_survivor(j)
    return log_d, log_D


# ------------------------------------------------------------------
# Forward kernel
# ------------------------------------------------------------------
@nb.njit(cache=True)
def _forward_kernel(cum: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray, log_d: np.ndarray,
                    log_D: np.ndarray, T: int, N: int, M: int) -> tuple[np.ndarray, np.ndarray, float]:
    alpha_begin = np.full((T, N), -np.inf)
    alpha_end = np.full((T, N), -np.inf)

    alpha_begin[0] = log_init.copy()

    terms = np.empty(M, dtype=np.float64)
    trans_terms = np.empty(N, dtype=np.float64)

    for t in range(T):
        is_terminal = t == T - 1
        for j in range(N):
            u_max = min(t + 1, M)
            for u_idx in range(u_max):
                u = u_idx + 1
                s = t - u + 1
                seg_emit = cum[t + 1, j] - cum[s, j]
                if is_terminal and (s + u - 1 == T - 1):
                    dur_term = log_D[j, u_idx]
                else:
                    dur_term = log_d[j, u_idx]
                terms[u_idx] = alpha_begin[s, j] + dur_term + seg_emit
            alpha_end[t, j] = _logsumexp_1d(terms[:u_max])

        if t < T - 1:
            for j in range(N):
                for i in range(N):
                    trans_terms[i] = alpha_end[t, i] + log_trans[i, j]
                alpha_begin[t + 1, j] = _logsumexp_1d(trans_terms)

    log_likelihood = _logsumexp_1d(alpha_end[T - 1])
    return alpha_begin, alpha_end, log_likelihood


# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------
def hsmm_forward(log_emissions: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray,
                 duration_model: BaseDuration) -> tuple[np.ndarray, np.ndarray, float]:
    """HSMM forward pass with pyhsmm-style begin/end decomposition.

    Returns
    -------
    alpha_begin : (T, N) — log prob of entering state j at time t (excludes emissions)
    alpha_end   : (T, N) — log prob of segment in state j ending at time t (includes emissions)
    log_likelihood : float
    """
    T, N = log_emissions.shape
    M = duration_model.max_duration
    log_d, log_D = _precompute_duration_tables(duration_model, N)
    cum = _precompute_cumulative_emissions(log_emissions)
    alpha_begin, alpha_end, ll = _forward_kernel(cum, log_trans, log_init, log_d, log_D, T, N, M)
    return alpha_begin, alpha_end, float(ll)


# ------------------------------------------------------------------
# Backward kernel
# ------------------------------------------------------------------
@nb.njit(cache=True)
def _backward_kernel(cum: np.ndarray, log_trans: np.ndarray, log_d: np.ndarray, log_D: np.ndarray,
                     T: int, N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    beta_begin = np.full((T, N), -np.inf)
    beta_end = np.full((T, N), -np.inf)

    beta_end[T - 1] = 0.0

    terms = np.empty(M, dtype=np.float64)
    trans_terms = np.empty(N, dtype=np.float64)

    for t in range(T - 1, -1, -1):
        for j in range(N):
            u_max = min(T - t, M)
            for u_idx in range(u_max):
                u = u_idx + 1
                end_t = t + u - 1
                seg_emit = cum[end_t + 1, j] - cum[t, j]
                if end_t == T - 1:
                    dur_term = log_D[j, u_idx]
                else:
                    dur_term = log_d[j, u_idx]
                terms[u_idx] = dur_term + seg_emit + beta_end[end_t, j]
            beta_begin[t, j] = _logsumexp_1d(terms[:u_max])

        if t > 0:
            for j in range(N):
                for k in range(N):
                    trans_terms[k] = log_trans[j, k] + beta_begin[t, k]
                beta_end[t - 1, j] = _logsumexp_1d(trans_terms)

    return beta_begin, beta_end


# ------------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------------
def hsmm_backward(log_emissions: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray,
                  duration_model: BaseDuration) -> tuple[np.ndarray, np.ndarray]:
    """HSMM backward pass.

    Returns
    -------
    beta_begin : (T, N)
    beta_end   : (T, N)
    """
    T, N = log_emissions.shape
    M = duration_model.max_duration
    log_d, log_D = _precompute_duration_tables(duration_model, N)
    cum = _precompute_cumulative_emissions(log_emissions)
    return _backward_kernel(cum, log_trans, log_d, log_D, T, N, M)


# ------------------------------------------------------------------
# Forward-backward sufficient-statistics kernel
# ------------------------------------------------------------------
@nb.njit(cache=True)
def _fb_stats_kernel(cum: np.ndarray, log_trans: np.ndarray, log_d: np.ndarray, log_D: np.ndarray,
                     alpha_begin: np.ndarray, alpha_end: np.ndarray, beta_begin: np.ndarray,
                     beta_end: np.ndarray, log_Z: float, T: int, N: int, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.zeros((N, M), dtype=np.float64)
    gamma_diff = np.zeros((T + 1, N), dtype=np.float64)

    for j in range(N):
        for s in range(T):
            u_max = min(T - s, M)
            for u_idx in range(u_max):
                u = u_idx + 1
                end_t = s + u - 1
                seg_emit = cum[end_t + 1, j] - cum[s, j]
                if end_t == T - 1:
                    dur_term = log_D[j, u_idx]
                else:
                    dur_term = log_d[j, u_idx]
                log_zeta = alpha_begin[s, j] + dur_term + seg_emit + beta_end[end_t, j] - log_Z
                zeta_val = np.exp(log_zeta)

                eta[j, u_idx] += zeta_val
                gamma_diff[s, j] += zeta_val
                if end_t + 1 < T + 1:
                    gamma_diff[end_t + 1, j] -= zeta_val

    # Convert difference array to gamma via cumsum
    gamma = np.empty((T, N), dtype=np.float64)
    for j in range(N):
        gamma[0, j] = gamma_diff[0, j]
        for t in range(1, T):
            gamma[t, j] = gamma[t - 1, j] + gamma_diff[t, j]

    # Expected transition counts
    xi = np.zeros((N, N), dtype=np.float64)
    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                log_val = alpha_end[t, i] + log_trans[i, j] + beta_begin[t + 1, j] - log_Z
                xi[i, j] += np.exp(log_val)

    return gamma, xi, eta


# ------------------------------------------------------------------
# Forward-backward
# ------------------------------------------------------------------
def hsmm_forward_backward(log_emissions: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray,
                          duration_model: BaseDuration) -> HSMMInferenceResult:
    """Full HSMM E-step: forward-backward with sufficient statistics.

    Returns state posteriors (gamma), expected transition counts (xi),
    and expected duration counts (eta).
    """
    T, N = log_emissions.shape
    M = duration_model.max_duration
    log_d, log_D = _precompute_duration_tables(duration_model, N)
    cum = _precompute_cumulative_emissions(log_emissions)

    alpha_begin, alpha_end, log_Z = hsmm_forward(log_emissions, log_trans, log_init, duration_model)
    beta_begin, beta_end = hsmm_backward(log_emissions, log_trans, log_init, duration_model)

    gamma, xi, eta = _fb_stats_kernel(cum, log_trans, log_d, log_D, alpha_begin, alpha_end, beta_begin, beta_end,
                                      log_Z, T, N, M)

    # Posterior mass check (spec §9.4)
    row_sums = gamma.sum(axis=1)
    max_deviation = np.max(np.abs(row_sums - 1.0))
    if max_deviation > 1e-6:
        logger.warning("Forward-backward gamma rows deviate from 1.0 by up to %.2e.", max_deviation)

    return HSMMInferenceResult(state_posteriors=gamma, expected_transitions=xi, expected_durations=eta,
                               log_likelihood=log_Z)


# ------------------------------------------------------------------
# Viterbi kernel
# ------------------------------------------------------------------
@nb.njit(cache=True)
def _viterbi_kernel(cum: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray, log_d: np.ndarray,
                    log_D: np.ndarray, T: int, N: int, M: int) -> np.ndarray:
    v_begin = np.full((T, N), -np.inf)
    v_end = np.full((T, N), -np.inf)
    bp_dur = np.zeros((T, N), dtype=np.int32)
    bp_pred = np.zeros((T, N), dtype=np.int32)

    v_begin[0] = log_init.copy()

    for t in range(T):
        is_terminal = t == T - 1
        for j in range(N):
            u_max = min(t + 1, M)
            best_val = -np.inf
            best_u = 1
            for u_idx in range(u_max):
                u = u_idx + 1
                s = t - u + 1
                seg_emit = cum[t + 1, j] - cum[s, j]
                if is_terminal and (s + u - 1 == T - 1):
                    dur_term = log_D[j, u_idx]
                else:
                    dur_term = log_d[j, u_idx]
                val = v_begin[s, j] + dur_term + seg_emit
                if val > best_val:
                    best_val = val
                    best_u = u
            v_end[t, j] = best_val
            bp_dur[t, j] = best_u

        if t < T - 1:
            for j in range(N):
                best_val = -np.inf
                best_i = 0
                for i in range(N):
                    val = v_end[t, i] + log_trans[i, j]
                    if val > best_val:
                        best_val = val
                        best_i = i
                v_begin[t + 1, j] = best_val
                bp_pred[t + 1, j] = best_i

    # Backtrack
    states = np.empty(T, dtype=np.int32)
    best_j = 0
    best_score = v_end[T - 1, 0]
    for j in range(1, N):
        if v_end[T - 1, j] > best_score:
            best_score = v_end[T - 1, j]
            best_j = j

    t = T - 1
    while t >= 0:
        dur = bp_dur[t, best_j]
        for k in range(t - dur + 1, t + 1):
            states[k] = best_j
        s = t - dur + 1
        if s > 0:
            best_j = bp_pred[s, best_j]
            t = s - 1
        else:
            break

    return states


# ------------------------------------------------------------------
# Viterbi
# ------------------------------------------------------------------
def hsmm_viterbi(log_emissions: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray,
                 duration_model: BaseDuration) -> np.ndarray:
    """Explicit-duration Viterbi decoding.

    Returns
    -------
    states : (T,) — most likely state sequence
    """
    T, N = log_emissions.shape
    M = duration_model.max_duration
    log_d, log_D = _precompute_duration_tables(duration_model, N)
    cum = _precompute_cumulative_emissions(log_emissions)
    return _viterbi_kernel(cum, log_trans, log_init, log_d, log_D, T, N, M)


# ------------------------------------------------------------------
# Filtering kernel
# ------------------------------------------------------------------
@nb.njit(cache=True)
def _filter_kernel(log_emissions: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray, log_hazard: np.ndarray,
                   log_cond_surv: np.ndarray, T: int, N: int, M: int) -> np.ndarray:
    gamma_filtered = np.empty((T, N), dtype=np.float64)

    psi_prev = np.full((N, M), -np.inf)
    psi_curr = np.full((N, M), -np.inf)

    # t = 0
    for j in range(N):
        psi_prev[j, 0] = log_init[j] + log_emissions[0, j]

    # Normalise t=0
    log_Z0 = _logsumexp_2d(psi_prev)
    if log_Z0 == -np.inf:
        for j in range(N):
            gamma_filtered[0, j] = 0.0
    else:
        for j in range(N):
            gamma_filtered[0, j] = np.exp(_logsumexp_1d_safe(psi_prev[j]) - log_Z0)

    # Pre-allocate buffer for new-segment logsumexp terms
    terms_buf = np.empty((N - 1) * M, dtype=np.float64)

    for t in range(1, T):
        # Reset psi_curr
        for j in range(N):
            for d in range(M):
                psi_curr[j, d] = -np.inf

        # 1. Continue existing segments
        for j in range(N):
            for d_idx in range(1, M):
                psi_curr[j, d_idx] = psi_prev[j, d_idx - 1] + log_emissions[t, j] + log_cond_surv[j, d_idx - 1]

        # 2. Start new segments
        for j in range(N):
            n_terms = 0
            for i in range(N):
                if i == j:
                    continue
                for d_idx in range(M):
                    if psi_prev[i, d_idx] > -np.inf:
                        terms_buf[n_terms] = psi_prev[i, d_idx] + log_hazard[i, d_idx] + log_trans[i, j]
                        n_terms += 1
            if n_terms > 0:
                psi_curr[j, 0] = _logsumexp_1d(terms_buf[:n_terms]) + log_emissions[t, j]

        # Normalise
        log_Zt = _logsumexp_2d(psi_curr)
        if log_Zt == -np.inf:
            for j in range(N):
                gamma_filtered[t, j] = 0.0
        else:
            for j in range(N):
                gamma_filtered[t, j] = np.exp(_logsumexp_1d_safe(psi_curr[j]) - log_Zt)

        # Swap
        psi_prev, psi_curr = psi_curr, psi_prev

    return gamma_filtered


@nb.njit(cache=True)
def _logsumexp_1d_safe(arr: np.ndarray) -> float:
    """Logsumexp that returns -inf for all-neginf arrays."""
    has_valid = False
    for i in range(arr.shape[0]):
        if arr[i] > -np.inf:
            has_valid = True
            break
    if not has_valid:
        return -np.inf
    return _logsumexp_1d(arr)


@nb.njit(cache=True)
def _logsumexp_2d(arr: np.ndarray) -> float:
    """Logsumexp over all elements of a 2-D array, skipping -inf."""
    max_val = -np.inf
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > max_val:
                max_val = arr[i, j]
    if max_val == -np.inf:
        return -np.inf
    s = 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > -np.inf:
                s += np.exp(arr[i, j] - max_val)
    return max_val + np.log(s)


# ------------------------------------------------------------------
# Filtering (causal)
# ------------------------------------------------------------------
def hsmm_filter(log_emissions: np.ndarray, log_trans: np.ndarray, log_init: np.ndarray,
                duration_model: BaseDuration) -> np.ndarray:
    """Forward-only causal filtering for HSMM.

    Tracks joint ``(state, elapsed_duration)`` belief using hazard-rate
    recursion. Only stores current and previous time slices → O(N x M)
    memory.

    Returns
    -------
    gamma_filtered : (T, N) — filtered state occupancy posteriors
    """
    T, N = log_emissions.shape
    M = duration_model.max_duration
    log_d, log_D = _precompute_duration_tables(duration_model, N)

    log_hazard = log_d - log_D
    log_cond_surv = np.full((N, M), _NEG_INF, dtype=np.float64)
    for j in range(N):
        for d_idx in range(M - 1):
            log_cond_surv[j, d_idx] = log_D[j, d_idx + 1] - log_D[j, d_idx]

    gamma_filtered = _filter_kernel(log_emissions, log_trans, log_init, log_hazard, log_cond_surv, T, N, M)

    # Posterior mass check (spec §9.4)
    row_sums = gamma_filtered.sum(axis=1)
    max_deviation = np.max(np.abs(row_sums - 1.0))
    if max_deviation > 1e-6:
        logger.warning("Filtered gamma rows deviate from 1.0 by up to %.2e.", max_deviation)

    return gamma_filtered
