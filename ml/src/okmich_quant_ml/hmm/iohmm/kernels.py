"""
Numba-JIT recursions for HMMs whose transition matrix may change every bar.

``log_A`` has shape ``(T_A, K, K)``. With ``T_A == 1`` the single matrix is used at every bar (a homogeneous HMM, no
``T x K x K`` copy needed). With ``T_A == T``, ``log_A[t]`` governs the transition from bar ``t - 1`` into bar ``t``;
``log_A[0]`` is never read, because bar 0 is drawn from the start distribution.

Several sequences are passed as one stacked array plus ``starts`` / ``ends`` row bounds. The fitting kernels run the
sequences in parallel and in probability space with per-bar scaling, which is several times cheaper than log space;
a sequence whose scaled recursion leaves the guarded range is redone in log space (see :func:`_scaled_pass`).

Results never depend on the thread count: every parallel loop writes disjoint rows, and every sum across sequences or
row chunks is formed afterwards in a fixed order (row chunks have a fixed size, not one per thread).

The last two kernels belong to the input-driven transition model: its per-bar log transition matrices and the
objective of its M-step.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

_SAFE = 1e-100  # guarded range [_SAFE, 1 / _SAFE] of the scaled recursions; see _scaled_pass
_ROW_CHUNK = 4096  # rows per partial sum in the parallel reductions (fixed, so the sums do not depend on threads)


def concurrent_calls_supported() -> bool:
    """
    Whether several Python threads may run the parallel kernels at once. Numba's ``tbb`` and ``omp`` threading layers
    allow it; ``workqueue`` (numba's fallback when neither is installed) does not.
    """
    one = np.ones(1, dtype=np.int64)
    sequences_log_likelihood(np.zeros(1), np.zeros((1, 1, 1)), np.zeros((1, 1)), one - 1, one)  # selects the layer
    return numba.threading_layer() != "workqueue"


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False):
    """
    Stable ``log(sum(exp(a)))`` along ``axis``; a slice that is all ``-inf`` gives ``-inf``.

    Used instead of ``scipy.special.logsumexp``, whose array-API dispatch costs ~0.5 ms a call: it dominated EM on many
    short sequences, where it is called once per sequence per iteration and in every M-step objective evaluation.
    """
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    if keepdims:
        return out
    return out.reshape(())[()] if axis is None else np.squeeze(out, axis=axis)


@njit(cache=True, nogil=True)
def _logsumexp(a: np.ndarray) -> float:
    m = -np.inf
    for v in a:
        if v > m:
            m = v
    if m == -np.inf:
        return -np.inf
    s = 0.0
    for v in a:
        s += np.exp(v - m)
    return m + np.log(s)


@njit(cache=True, nogil=True)
def forward_log(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    """Forward log-alpha, shape ``(T, K)``: ``log P(o_0..o_t, z_t = k)``."""
    T, K = log_B.shape
    time_varying = log_A.shape[0] > 1
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + log_B[0]
    buf = np.empty(K)
    for t in range(1, T):
        a = log_A[t] if time_varying else log_A[0]
        for j in range(K):
            for i in range(K):
                buf[i] = log_alpha[t - 1, i] + a[i, j]
            log_alpha[t, j] = _logsumexp(buf) + log_B[t, j]
    return log_alpha


@njit(cache=True, nogil=True)
def backward_log(log_A: np.ndarray, log_B: np.ndarray, log_end: np.ndarray) -> np.ndarray:
    """Backward log-beta, shape ``(T, K)``: ``log P(o_{t+1}..o_{T-1} | z_t = k)``, started from ``log_end``."""
    T, K = log_B.shape
    time_varying = log_A.shape[0] > 1
    log_beta = np.empty((T, K))
    log_beta[T - 1] = log_end
    buf = np.empty(K)
    for t in range(T - 2, -1, -1):
        a = log_A[t + 1] if time_varying else log_A[0]
        for i in range(K):
            for j in range(K):
                buf[j] = a[i, j] + log_B[t + 1, j] + log_beta[t + 1, j]
            log_beta[t, i] = _logsumexp(buf)
    return log_beta


@njit(cache=True, nogil=True)
def max_product_forward(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
    """Viterbi frontier log-delta, shape ``(T, K)``: the best path ending in each state. Causal (no traceback)."""
    T, K = log_B.shape
    time_varying = log_A.shape[0] > 1
    log_delta = np.empty((T, K))
    log_delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        a = log_A[t] if time_varying else log_A[0]
        for j in range(K):
            best = -np.inf
            for i in range(K):
                v = log_delta[t - 1, i] + a[i, j]
                if v > best:
                    best = v
            log_delta[t, j] = best + log_B[t, j]
    return log_delta


@njit(cache=True, nogil=True)
def viterbi_backpointers(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`max_product_forward` plus ``psi[t, k]``, the best predecessor of state ``k`` at bar ``t``."""
    T, K = log_B.shape
    time_varying = log_A.shape[0] > 1
    log_delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=np.int64)
    log_delta[0] = log_pi + log_B[0]
    for t in range(1, T):
        a = log_A[t] if time_varying else log_A[0]
        for j in range(K):
            best, arg = -np.inf, 0
            for i in range(K):
                v = log_delta[t - 1, i] + a[i, j]
                if v > best:
                    best, arg = v, i
            log_delta[t, j] = best + log_B[t, j]
            psi[t, j] = arg
    return log_delta, psi


# ----------------------------------------------------------------------------------------------------------------------
# Fitting: scaled recursions over many sequences
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _scaled_pass(log_pi: np.ndarray, lA: np.ndarray, lB: np.ndarray, backward: bool):
    """
    Forward (and optionally backward) pass of ONE sequence in probability space, scaled every bar (Rabiner, 1989).

    Returns ``(ok, loglik, alpha, beta, A, B_hat, c)``: the filtered posteriors ``alpha`` (T, K); the scaled backward
    messages ``beta`` (T, K), so that ``gamma = alpha * beta``; ``A = exp(lA)``; the emissions over their row maximum
    ``B_hat`` (T, K); and the per-bar normalisers ``c`` (T,), with ``loglik = sum_t log c_t + max_k lB[t, k]``.

    ``ok`` is False, and the caller must redo the sequence in log space, when the recursion may lose precision: a used
    transition probability below ``_SAFE``, a normaliser below ``_SAFE`` (or a bar where every state is impossible),
    or a backward message above ``1 / _SAFE``. Inside these bounds the dominant term of every sum is at least ~1e-200,
    so nothing that matters can underflow: a tiny filtered probability is always swamped by the transitions out of the
    dominant state, and its product with a backward message stays below 1e-200. The result then equals the log-space
    one to rounding.
    """
    T, K = lB.shape
    TA = lA.shape[0]
    time_varying = TA > 1
    alpha = np.empty((T, K))
    beta = np.empty((T if backward else 0, K))
    B_hat = np.empty((T, K))
    c = np.empty(T)
    A = np.exp(lA)
    for s in range(1 if time_varying else 0, TA):  # row 0 of a per-bar log_A is never read
        for i in range(K):
            for j in range(K):
                if not A[s, i, j] >= _SAFE:
                    return False, 0.0, alpha, beta, A, B_hat, c
    pred = np.empty(K)
    loglik = 0.0
    for t in range(T):
        top = -np.inf
        for k in range(K):
            if lB[t, k] > top:
                top = lB[t, k]
        if not np.isfinite(top):
            return False, 0.0, alpha, beta, A, B_hat, c
        if t == 0:
            for k in range(K):
                pred[k] = np.exp(log_pi[k])
        else:
            a = A[t] if time_varying else A[0]
            for j in range(K):
                acc = 0.0
                for i in range(K):
                    acc += alpha[t - 1, i] * a[i, j]
                pred[j] = acc
        ct = 0.0
        for k in range(K):
            B_hat[t, k] = np.exp(lB[t, k] - top)
            alpha[t, k] = pred[k] * B_hat[t, k]
            ct += alpha[t, k]
        if not ct >= _SAFE:
            return False, 0.0, alpha, beta, A, B_hat, c
        for k in range(K):
            alpha[t, k] /= ct
        c[t] = ct
        loglik += np.log(ct) + top
    if backward:
        for k in range(K):
            beta[T - 1, k] = 1.0
        for t in range(T - 2, -1, -1):
            a = A[t + 1] if time_varying else A[0]
            for i in range(K):
                acc = 0.0
                for j in range(K):
                    acc += a[i, j] * B_hat[t + 1, j] * beta[t + 1, j]
                v = acc / c[t + 1]
                if not v <= 1.0 / _SAFE:
                    return False, 0.0, alpha, beta, A, B_hat, c
                beta[t, i] = v
    return True, loglik, alpha, beta, A, B_hat, c


@njit(cache=True, nogil=True)
def _sequence_statistics(log_pi: np.ndarray, lA: np.ndarray, lB: np.ndarray, gamma: np.ndarray, xi: np.ndarray,
                         row0: int, per_transition: bool) -> float:
    """
    E-step of one sequence: writes its posteriors into ``gamma`` (T, K) and its expected transitions into ``xi`` (rows
    ``row0..row0+T-2`` if ``per_transition``, else added into row ``row0``); returns its log-likelihood. Scaled when
    :func:`_scaled_pass` is ok, otherwise in log space (open end: ``log_beta = 0`` at the last bar).
    """
    T, K = lB.shape
    time_varying = lA.shape[0] > 1
    ok, loglik, alpha, beta, A, B_hat, c = _scaled_pass(log_pi, lA, lB, True)
    if ok:
        for t in range(T):
            for k in range(K):
                gamma[t, k] = alpha[t, k] * beta[t, k]
        for t in range(T - 1):
            a = A[t + 1] if time_varying else A[0]
            target = row0 + t if per_transition else row0
            for i in range(K):
                for j in range(K):
                    xi[target, i, j] += alpha[t, i] * a[i, j] * (B_hat[t + 1, j] * beta[t + 1, j] / c[t + 1])
        return loglik
    la = forward_log(log_pi, lA, lB)
    lb = backward_log(lA, lB, np.zeros(K))
    ll = _logsumexp(la[T - 1])
    for t in range(T):
        for k in range(K):
            gamma[t, k] = np.exp(la[t, k] + lb[t, k] - ll)
    for t in range(T - 1):
        a_next = lA[t + 1] if time_varying else lA[0]
        target = row0 + t if per_transition else row0
        for i in range(K):
            for j in range(K):
                xi[target, i, j] += np.exp(la[t, i] + a_next[i, j] + (lB[t + 1, j] + lb[t + 1, j]) - ll)
    return ll


@njit(cache=True, nogil=True, parallel=True)
def sequences_log_likelihood(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray, starts: np.ndarray,
                             ends: np.ndarray) -> float:
    """Total log-likelihood of the stacked sequences ``[starts[s], ends[s])``, each started from ``log_pi``."""
    time_varying = log_A.shape[0] > 1
    n_seq = len(starts)
    seq_loglik = np.empty(n_seq)
    for s in prange(n_seq):
        a, z = starts[s], ends[s]
        lA = log_A[a:z] if time_varying else log_A
        ok, ll = _scaled_pass(log_pi, lA, log_B[a:z], False)[:2]
        seq_loglik[s] = ll if ok else _logsumexp(forward_log(log_pi, lA, log_B[a:z])[z - a - 1])
    total = 0.0
    for s in range(n_seq):  # fixed order
        total += seq_loglik[s]
    return total


@njit(cache=True, nogil=True, parallel=True)
def expected_statistics(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray, starts: np.ndarray,
                        ends: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    E-step over the stacked sequences ``[starts[s], ends[s])`` (open end: ``log_beta = 0`` at each sequence's last bar).

    Returns ``(loglik, gamma, start_counts, xi)``: the total log-likelihood; state posteriors ``gamma`` (T, K); the
    posteriors summed over each sequence's first bar (K,); and the expected transitions ``xi``. With a time-varying
    ``log_A``, ``xi`` has one row per transition (N, K, K), ordered by sequence and bar, every bar but each sequence's
    first; with a single ``log_A`` it is the (1, K, K) sum.
    """
    T, K = log_B.shape
    n_seq = len(starts)
    time_varying = log_A.shape[0] > 1
    gamma = np.empty((T, K))
    xi = np.zeros((T - n_seq if time_varying else n_seq, K, K))  # single log_A: one partial sum per sequence
    seq_loglik = np.empty(n_seq)
    first_row = np.empty(n_seq, dtype=np.int64)  # each earlier sequence has one transition fewer than bars
    for s in range(n_seq):
        first_row[s] = starts[s] - s if time_varying else s
    for s in prange(n_seq):
        a, z = starts[s], ends[s]
        lA = log_A[a:z] if time_varying else log_A
        seq_loglik[s] = _sequence_statistics(log_pi, lA, log_B[a:z], gamma[a:z], xi, first_row[s], time_varying)
    loglik = 0.0
    start_counts = np.zeros(K)
    for s in range(n_seq):  # fixed order
        loglik += seq_loglik[s]
        for k in range(K):
            start_counts[k] += gamma[starts[s], k]
    if time_varying:
        return loglik, gamma, start_counts, xi
    xi_sum = np.zeros((1, K, K))
    for s in range(n_seq):
        for i in range(K):
            for j in range(K):
                xi_sum[0, i, j] += xi[s, i, j]
    return loglik, gamma, start_counts, xi_sum


# ----------------------------------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True)
def windowed_fixed_lag(log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray, lag: int,
                       window_size: int) -> np.ndarray:
    """
    ``P(z_t | o_start..o_end)`` with ``end = min(t + lag, T - 1)`` and ``start = max(0, end - window_size + 1)``: each
    as-of bar sees only its last ``window_size`` bars and restarts from ``log_pi`` (a live buffer emulation).

    The forward pass runs over ``[start, t]`` and the backward pass over ``[t, end]``, the only bars each one needs.
    Bars are independent, so they run in parallel.
    """
    T, K = log_B.shape
    time_varying = log_A.shape[0] > 1
    out = np.empty((T, K))
    for t in prange(T):
        end = min(t + lag, T - 1)
        start = max(0, end - window_size + 1)
        lA_head = log_A[start:t + 1] if time_varying else log_A
        lA_tail = log_A[t:end + 1] if time_varying else log_A
        log_alpha = forward_log(log_pi, lA_head, log_B[start:t + 1])
        log_beta = backward_log(lA_tail, log_B[t:end + 1], np.zeros(K))
        post = np.empty(K)
        for k in range(K):
            post[k] = log_alpha[t - start, k] + log_beta[0, k]
        norm = _logsumexp(post)
        for k in range(K):
            out[t, k] = np.exp(post[k] - norm)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Input-driven transitions
# ----------------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True)
def input_log_matrices(b: np.ndarray, W: np.ndarray, U: np.ndarray) -> np.ndarray:
    """``log A_t[i, j] = eta_t[i, j] - logsumexp_k eta_t[i, k]`` with ``eta_t = b + W . U[t]``, shape ``(T, K, K)``."""
    T, d = U.shape
    K = b.shape[0]
    out = np.empty((T, K, K))
    for t in prange(T):
        for i in range(K):
            top = -np.inf
            for j in range(K):
                v = b[i, j]
                for m in range(d):
                    v += W[i, j, m] * U[t, m]
                out[t, i, j] = v
                if v > top:
                    top = v
            s = 0.0
            for j in range(K):
                s += np.exp(out[t, i, j] - top)
            log_norm = top + np.log(s)
            for j in range(K):
                out[t, i, j] -= log_norm
    return out


@njit(cache=True, nogil=True, parallel=True)
def logit_row_objective(coef: np.ndarray, X1: np.ndarray, n_out: np.ndarray, moves: np.ndarray,
                        total: float) -> tuple[float, np.ndarray]:
    """
    Unpenalised M-step objective of one origin row and its gradient, both divided by ``total``.

    ``coef`` (K - 1, 1 + d) holds the move logits' intercept and weights against "stay" (logit 0); ``X1`` (N, 1 + d)
    is ``[1, u_t]``; ``n_out`` (N,) the soft count of leaving the origin at each transition and ``moves`` (N, K - 1)
    the soft count of each move. NLL = ``sum_n n_out[n] logsumexp([0, eta_n]) - moves[n] . eta_n``.
    """
    N, p = X1.shape
    M = coef.shape[0]
    n_chunks = (N + _ROW_CHUNK - 1) // _ROW_CHUNK
    chunk_nll = np.zeros(n_chunks)
    chunk_grad = np.zeros((n_chunks, M, p))
    for chunk in prange(n_chunks):
        eta = np.empty(M)
        e = np.empty(M)
        part = 0.0
        for n in range(chunk * _ROW_CHUNK, min(N, (chunk + 1) * _ROW_CHUNK)):
            top = 0.0  # the stay logit
            for k in range(M):
                v = 0.0
                for col in range(p):
                    v += coef[k, col] * X1[n, col]
                eta[k] = v
                if v > top:
                    top = v
            s = np.exp(-top)
            for k in range(M):
                e[k] = np.exp(eta[k] - top)
                s += e[k]
            part += n_out[n] * (top + np.log(s))
            for k in range(M):
                part -= moves[n, k] * eta[k]
                r = n_out[n] * e[k] / s - moves[n, k]
                for col in range(p):
                    chunk_grad[chunk, k, col] += r * X1[n, col]
        chunk_nll[chunk] = part
    nll = 0.0
    grad = np.zeros((M, p))
    for chunk in range(n_chunks):  # fixed order
        nll += chunk_nll[chunk]
        for k in range(M):
            for col in range(p):
                grad[k, col] += chunk_grad[chunk, k, col]
    return nll / total, grad / total
