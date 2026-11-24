"""Numba recursions on the flattened leaf chain of a hierarchical HMM, in log space.

``log_trans`` is a stack (P, N, N) of log leaf-to-leaf transition matrices, one per exit-mode pattern; ``pattern[t]``
selects the matrix of the transition from bar t-1 into bar t. ``starts[t]`` marks the first bar of a sequence: the
recursion restarts there from ``log_pi`` (the root's entry distribution pushed down to the leaves).

Everything runs on log probabilities with log-sum-exp, so a state whose probability falls below the smallest float
(e.g. a regime ruled out for hundreds of bars under a near-identity transition) stays finite in log space and can be
recovered by later evidence, and an observation far in the tail of the only reachable leaf still has its finite
density. Emissions enter as ``log_b`` (T, N); a leaf that cannot emit an observation has log density -inf.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

_NEG_INF = -np.inf


@njit(cache=True, inline="always")
def _lse2(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def _log_prior(prev: np.ndarray, log_trans_t: np.ndarray, out: np.ndarray) -> None:
    """out[j] = log sum_i exp(prev[i] + log_trans_t[i, j])."""
    n = out.shape[0]
    for j in range(n):
        top = _NEG_INF
        for i in range(n):
            v = prev[i] + log_trans_t[i, j]
            if v > top:
                top = v
        if top == _NEG_INF:
            out[j] = _NEG_INF
            continue
        acc = 0.0
        for i in range(n):
            v = prev[i] + log_trans_t[i, j]
            if v != _NEG_INF:
                acc += math.exp(v - top)
        out[j] = top + math.log(acc)


@njit(cache=True)
def _normalise(row: np.ndarray) -> float:
    """Subtract log-sum-exp from ``row`` in place and return it (-inf if every entry is -inf)."""
    top = _NEG_INF
    for j in range(row.shape[0]):
        if row[j] > top:
            top = row[j]
    if top == _NEG_INF:
        return _NEG_INF
    acc = 0.0
    for j in range(row.shape[0]):
        if row[j] != _NEG_INF:
            acc += math.exp(row[j] - top)
    norm = top + math.log(acc)
    for j in range(row.shape[0]):
        row[j] -= norm
    return norm


@njit(cache=True)
def _forward(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
             starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised log filtered posteriors (T, N) and log one-step predictive densities (T,)."""
    n_bars, n_leaves = log_b.shape
    log_alpha = np.empty((n_bars, n_leaves))
    log_pred = np.empty(n_bars)
    prior = np.empty(n_leaves)
    for t in range(n_bars):
        if starts[t] or t == 0:
            for j in range(n_leaves):
                prior[j] = log_pi[j]
        else:
            _log_prior(log_alpha[t - 1], log_trans[pattern[t]], prior)
        for j in range(n_leaves):
            log_alpha[t, j] = prior[j] + log_b[t, j]
        norm = _normalise(log_alpha[t])
        if norm == _NEG_INF:
            raise ValueError("The observation sequence has zero probability under the model")
        log_pred[t] = norm
    return log_alpha, log_pred


@njit(cache=True)
def forward_filter(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                   starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filtered leaf posteriors P(z_t | x_1..t) and the one-step predictive log density log p(x_t | x_1..t-1)."""
    log_alpha, log_pred = _forward(log_b, log_trans, pattern, log_pi, starts)
    return np.exp(log_alpha), log_pred


@njit(cache=True)
def forward_backward(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                     starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Smoothed posteriors, pairwise posteriors summed per pattern (P, N, N), posteriors summed at sequence starts,
    and the log-likelihood.

    The backward message is scaled by the forward pass's predictive densities, so gamma = alpha * beta and
    xi_t(i, j) = alpha_t(i) A(i, j) b_t+1(j) beta_t+1(j) / p(x_t+1 | x_1..t), all formed in log space.
    """
    n_bars, n_leaves = log_b.shape
    n_patterns = log_trans.shape[0]
    log_alpha, log_pred = _forward(log_b, log_trans, pattern, log_pi, starts)
    loglik = 0.0
    for t in range(n_bars):
        loglik += log_pred[t]

    log_beta = np.empty((n_bars, n_leaves))
    gamma = np.empty((n_bars, n_leaves))
    xi = np.zeros((n_patterns, n_leaves, n_leaves))
    start_sum = np.zeros(n_leaves)
    nxt = np.empty(n_leaves)
    for j in range(n_leaves):
        log_beta[n_bars - 1, j] = 0.0
    for t in range(n_bars - 1, -1, -1):
        if t < n_bars - 1:
            if starts[t + 1]:
                for i in range(n_leaves):
                    log_beta[t, i] = 0.0
            else:
                p = pattern[t + 1]
                c = log_pred[t + 1]
                for j in range(n_leaves):
                    nxt[j] = log_b[t + 1, j] + log_beta[t + 1, j]
                for i in range(n_leaves):
                    top = _NEG_INF
                    for j in range(n_leaves):
                        v = log_trans[p, i, j] + nxt[j]
                        if v > top:
                            top = v
                    if top == _NEG_INF:
                        log_beta[t, i] = _NEG_INF
                    else:
                        acc = 0.0
                        for j in range(n_leaves):
                            v = log_trans[p, i, j] + nxt[j]
                            if v != _NEG_INF:
                                acc += math.exp(v - top)
                        log_beta[t, i] = top + math.log(acc) - c
                for i in range(n_leaves):
                    a = log_alpha[t, i]
                    if a == _NEG_INF:
                        continue
                    for j in range(n_leaves):
                        v = a + log_trans[p, i, j] + nxt[j] - c
                        if v != _NEG_INF:
                            xi[p, i, j] += math.exp(v)
        norm = _NEG_INF
        for j in range(n_leaves):
            gamma[t, j] = log_alpha[t, j] + log_beta[t, j]
            norm = _lse2(norm, gamma[t, j])
        for j in range(n_leaves):
            gamma[t, j] = math.exp(gamma[t, j] - norm)
        if starts[t] or t == 0:
            for j in range(n_leaves):
                start_sum[j] += gamma[t, j]
    return gamma, xi, start_sum, loglik


@njit(cache=True)
def _viterbi_forward(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                     starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max-product forward pass: delta (T, N) normalised to max 0 per bar, and back-pointers (T, N)."""
    n_bars, n_leaves = log_b.shape
    delta = np.empty((n_bars, n_leaves))
    back = np.full((n_bars, n_leaves), -1, dtype=np.int64)
    for t in range(n_bars):
        top = _NEG_INF
        for j in range(n_leaves):
            if starts[t] or t == 0:
                best = log_pi[j]
            else:
                best = _NEG_INF
                p = pattern[t]
                for i in range(n_leaves):
                    cand = delta[t - 1, i] + log_trans[p, i, j]
                    if cand > best:
                        best = cand
                        back[t, j] = i
            delta[t, j] = best + log_b[t, j]
            if delta[t, j] > top:
                top = delta[t, j]
        if top == _NEG_INF:
            raise ValueError("The observation sequence has zero probability under the model")
        for j in range(n_leaves):
            delta[t, j] -= top
    return delta, back


@njit(cache=True)
def causal_viterbi(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                   starts: np.ndarray) -> np.ndarray:
    """End state of the most likely path given x_1..t, at every t (no traceback: causal)."""
    delta, _ = _viterbi_forward(log_b, log_trans, pattern, log_pi, starts)
    n_bars, n_leaves = delta.shape
    out = np.empty(n_bars, dtype=np.int64)
    for t in range(n_bars):
        best, arg = _NEG_INF, 0
        for j in range(n_leaves):
            if delta[t, j] > best:
                best, arg = delta[t, j], j
        out[t] = arg
    return out


@njit(cache=True)
def viterbi(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
            starts: np.ndarray) -> np.ndarray:
    """Most likely leaf path per sequence (traceback from each sequence's last bar): hindsight only."""
    delta, back = _viterbi_forward(log_b, log_trans, pattern, log_pi, starts)
    n_bars, n_leaves = delta.shape
    path = np.empty(n_bars, dtype=np.int64)
    end = n_bars - 1
    while end >= 0:
        best, arg = _NEG_INF, 0
        for j in range(n_leaves):
            if delta[end, j] > best:
                best, arg = delta[end, j], j
        path[end] = arg
        t = end
        while t > 0 and not starts[t]:
            path[t - 1] = back[t, path[t]]
            t -= 1
        end = t - 1
    return path


@njit(cache=True)
def fixed_lag(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray, starts: np.ndarray,
              lag: int) -> np.ndarray:
    """P(z_t | x_1..min(t + lag, end of t's sequence)), open-ended: the backward pass starts flat at the horizon."""
    log_alpha, _ = _forward(log_b, log_trans, pattern, log_pi, starts)
    n_bars, n_leaves = log_b.shape
    if lag == 0:
        return np.exp(log_alpha)
    seq_end = np.empty(n_bars, dtype=np.int64)
    last = n_bars - 1
    for t in range(n_bars - 1, -1, -1):
        seq_end[t] = last
        if starts[t]:
            last = t - 1
    out = np.empty((n_bars, n_leaves))
    beta = np.empty(n_leaves)
    nxt = np.empty(n_leaves)
    tmp = np.empty(n_leaves)
    for t in range(n_bars):
        horizon = min(t + lag, seq_end[t])
        for j in range(n_leaves):
            beta[j] = 0.0
        for s in range(horizon - 1, t - 1, -1):
            p = pattern[s + 1]
            for j in range(n_leaves):
                tmp[j] = log_b[s + 1, j] + beta[j]
            for i in range(n_leaves):
                top = _NEG_INF
                for j in range(n_leaves):
                    v = log_trans[p, i, j] + tmp[j]
                    if v > top:
                        top = v
                if top == _NEG_INF:
                    nxt[i] = _NEG_INF
                    continue
                acc = 0.0
                for j in range(n_leaves):
                    v = log_trans[p, i, j] + tmp[j]
                    if v != _NEG_INF:
                        acc += math.exp(v - top)
                nxt[i] = top + math.log(acc)
            _normalise(nxt)
            for i in range(n_leaves):
                beta[i] = nxt[i]
        for j in range(n_leaves):
            out[t, j] = log_alpha[t, j] + beta[j]
        _normalise(out[t])
        for j in range(n_leaves):
            out[t, j] = math.exp(out[t, j])
    return out


@njit(cache=True)
def causal_viterbi_scores(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                          starts: np.ndarray) -> np.ndarray:
    """Max-product forward scores log delta_t(j) (T, N): the log probability of the best path ending in j at t, per
    sequence. Path log-probabilities, not a distribution (rows do not normalise to 1)."""
    delta, _ = _viterbi_forward(log_b, log_trans, pattern, log_pi, starts)
    n_bars, n_leaves = delta.shape
    out = np.empty((n_bars, n_leaves))
    # _viterbi_forward subtracts each bar's maximum; add the running offset back (restarting with each sequence)
    offset = 0.0
    for t in range(n_bars):
        if starts[t] or t == 0:
            best = -np.inf
            for j in range(n_leaves):
                v = log_pi[j] + log_b[t, j]
                if v > best:
                    best = v
            offset = best
        else:
            p = pattern[t]
            best = -np.inf
            for j in range(n_leaves):
                inner = -np.inf
                for i in range(n_leaves):
                    cand = delta[t - 1, i] + log_trans[p, i, j]
                    if cand > inner:
                        inner = cand
                v = inner + log_b[t, j]
                if v > best:
                    best = v
            offset += best
        for j in range(n_leaves):
            out[t, j] = delta[t, j] + offset
    return out


@njit(cache=True)
def path_entropy(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                 starts: np.ndarray) -> float:
    """Exact entropy H(z_1:T | x) of the posterior over leaf paths (nats), summed over sequences.

    The posterior over paths is a Markov chain: H = sum over sequences of H(z_s | x) + sum_t sum_i gamma_t-1(i)
    H(z_t | z_t-1 = i, x), with p(z_t = j | z_t-1 = i, x) = A_ij b_j(x_t) beta_t(j) / (beta_t-1(i) c_t).
    """
    n_bars, n_leaves = log_b.shape
    log_alpha, log_pred = _forward(log_b, log_trans, pattern, log_pi, starts)
    log_beta = np.zeros((n_bars, n_leaves))
    nxt = np.empty(n_leaves)
    for t in range(n_bars - 2, -1, -1):
        if starts[t + 1]:
            continue
        p = pattern[t + 1]
        for j in range(n_leaves):
            nxt[j] = log_b[t + 1, j] + log_beta[t + 1, j]
        for i in range(n_leaves):
            acc = _NEG_INF
            for j in range(n_leaves):
                acc = _lse2(acc, log_trans[p, i, j] + nxt[j])
            log_beta[t, i] = acc - log_pred[t + 1]
    total = 0.0
    for t in range(n_bars):
        if starts[t] or t == 0:
            for j in range(n_leaves):
                g = math.exp(log_alpha[t, j] + log_beta[t, j])
                if g > 0.0:
                    total -= g * (log_alpha[t, j] + log_beta[t, j])
            continue
        p = pattern[t]
        for i in range(n_leaves):
            g_prev = math.exp(log_alpha[t - 1, i] + log_beta[t - 1, i])
            if g_prev <= 0.0:
                continue
            h = 0.0
            for j in range(n_leaves):
                lq = log_trans[p, i, j] + log_b[t, j] + log_beta[t, j] - log_pred[t] - log_beta[t - 1, i]
                if lq != _NEG_INF:
                    q = math.exp(lq)
                    if q > 0.0:
                        h -= q * lq
            total += g_prev * h
    return total


@njit(cache=True)
def fixed_lag_windowed(log_b: np.ndarray, log_trans: np.ndarray, pattern: np.ndarray, log_pi: np.ndarray,
                       starts: np.ndarray, lag: int, window: int) -> np.ndarray:
    """Bounded-history fixed lag: P(z_t | x_s..e) with e = min(t + lag, end of t's sequence) and s = max(start of t's
    sequence, e - window + 1), recomputed per t from the entry distribution at s (a finite live buffer)."""
    n_bars, n_leaves = log_b.shape
    seq_start = np.empty(n_bars, dtype=np.int64)
    first = 0
    for t in range(n_bars):
        if starts[t]:
            first = t
        seq_start[t] = first
    seq_end = np.empty(n_bars, dtype=np.int64)
    last = n_bars - 1
    for t in range(n_bars - 1, -1, -1):
        seq_end[t] = last
        if starts[t]:
            last = t - 1
    out = np.empty((n_bars, n_leaves))
    la = np.empty(n_leaves)
    prior = np.empty(n_leaves)
    beta = np.empty(n_leaves)
    nxt = np.empty(n_leaves)
    tmp = np.empty(n_leaves)
    for t in range(n_bars):
        e = min(t + lag, seq_end[t])
        s = max(seq_start[t], e - window + 1)
        for j in range(n_leaves):
            la[j] = log_pi[j] + log_b[s, j]
        _normalise(la)
        for u in range(s + 1, t + 1):
            _log_prior(la, log_trans[pattern[u]], prior)
            for j in range(n_leaves):
                la[j] = prior[j] + log_b[u, j]
            _normalise(la)
        for j in range(n_leaves):
            beta[j] = 0.0
        for u in range(e - 1, t - 1, -1):
            p = pattern[u + 1]
            for j in range(n_leaves):
                tmp[j] = log_b[u + 1, j] + beta[j]
            for i in range(n_leaves):
                acc = _NEG_INF
                for j in range(n_leaves):
                    acc = _lse2(acc, log_trans[p, i, j] + tmp[j])
                nxt[i] = acc
            _normalise(nxt)
            for i in range(n_leaves):
                beta[i] = nxt[i]
        for j in range(n_leaves):
            out[t, j] = la[j] + beta[j]
        _normalise(out[t])
        for j in range(n_leaves):
            out[t, j] = math.exp(out[t, j])
    return out
