"""Numba recursions on the joint state space of a coupled HMM.

Emissions enter as ``log_b`` (T, J). Each row is shifted by its maximum before exponentiating, so a state an
observation rules out (log density -inf) gets probability 0 and the shifts are added back into the likelihood.
``starts[t]`` marks the first bar of a sequence: the recursion restarts there from the initial distribution.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _scaled_emissions(log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_bars, n_joint = log_b.shape
    b = np.empty((n_bars, n_joint))
    shift = np.empty(n_bars)
    for t in range(n_bars):
        top = -np.inf
        for j in range(n_joint):
            if log_b[t, j] > top:
                top = log_b[t, j]
        shift[t] = top
        for j in range(n_joint):
            b[t, j] = math.exp(log_b[t, j] - top)
    return b, shift


@njit(cache=True)
def forward_filter(log_b: np.ndarray, trans: np.ndarray, pi: np.ndarray,
                   starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filtered joint posteriors P(z_t | x_1..t) and the one-step predictive log density log p(x_t | x_1..t-1)."""
    n_bars, n_joint = log_b.shape
    b, shift = _scaled_emissions(log_b)
    alpha = np.empty((n_bars, n_joint))
    log_pred = np.empty(n_bars)
    for t in range(n_bars):
        total = 0.0
        for j in range(n_joint):
            if starts[t] or t == 0:
                prior = pi[j]
            else:
                prior = 0.0
                for i in range(n_joint):
                    prior += alpha[t - 1, i] * trans[i, j]
            alpha[t, j] = prior * b[t, j]
            total += alpha[t, j]
        for j in range(n_joint):
            alpha[t, j] /= total
        log_pred[t] = math.log(total) + shift[t]
    return alpha, log_pred


@njit(cache=True)
def forward_backward(log_b: np.ndarray, trans: np.ndarray, pi: np.ndarray,
                     starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Smoothed posteriors, summed pairwise posteriors, summed posteriors at sequence starts, log-likelihood."""
    n_bars, n_joint = log_b.shape
    b, shift = _scaled_emissions(log_b)
    alpha = np.empty((n_bars, n_joint))
    scale = np.empty(n_bars)
    loglik = 0.0
    for t in range(n_bars):
        total = 0.0
        for j in range(n_joint):
            if starts[t] or t == 0:
                prior = pi[j]
            else:
                prior = 0.0
                for i in range(n_joint):
                    prior += alpha[t - 1, i] * trans[i, j]
            alpha[t, j] = prior * b[t, j]
            total += alpha[t, j]
        for j in range(n_joint):
            alpha[t, j] /= total
        scale[t] = total
        loglik += math.log(total) + shift[t]

    beta = np.empty((n_bars, n_joint))
    gamma = np.empty((n_bars, n_joint))
    xi = np.zeros((n_joint, n_joint))
    start_sum = np.zeros(n_joint)
    weighted = np.empty(n_joint)
    for j in range(n_joint):
        beta[n_bars - 1, j] = 1.0
    for t in range(n_bars - 1, -1, -1):
        if t < n_bars - 1:
            if starts[t + 1]:
                for i in range(n_joint):
                    beta[t, i] = 1.0
            else:
                for j in range(n_joint):
                    weighted[j] = b[t + 1, j] * beta[t + 1, j] / scale[t + 1]
                for i in range(n_joint):
                    acc = 0.0
                    for j in range(n_joint):
                        acc += trans[i, j] * weighted[j]
                    beta[t, i] = acc
                for i in range(n_joint):
                    a = alpha[t, i]
                    for j in range(n_joint):
                        xi[i, j] += a * trans[i, j] * weighted[j]
        for j in range(n_joint):
            gamma[t, j] = alpha[t, j] * beta[t, j]
        if starts[t] or t == 0:
            for j in range(n_joint):
                start_sum[j] += gamma[t, j]
    return gamma, xi, start_sum, loglik
