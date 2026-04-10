"""
Numba-JIT kernels shared across Markov-switching model families.

All kernels are pure-numeric and have no dependency on model classes, making
them reusable across MarkovSwitchingAR, MarkovSwitchingGARCH, and MarkovSwitchingVAR.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _ar_forecast_kernel(ar_coeffs: np.ndarray, intercept: float, history: np.ndarray, steps: int) -> np.ndarray:
    """Compute multi-step AR(p) point forecast via recursive substitution.

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (p,)
        AR coefficients φ₁, φ₂, …, φ_p
    intercept : float
        Regime intercept μ
    history : np.ndarray, shape (p,)
        Last p observed values [y_{T-p+1}, …, y_T]
    steps : int
        Forecast horizon

    Returns
    -------
    forecasts : np.ndarray, shape (steps,)
    """
    p = len(ar_coeffs)
    buf = history.copy()
    forecasts = np.zeros(steps)
    for h in range(steps):
        y_hat = intercept
        for i in range(p):
            y_hat += ar_coeffs[i] * buf[p - 1 - i]
        forecasts[h] = y_hat
        for i in range(p - 1):
            buf[i] = buf[i + 1]
        buf[p - 1] = y_hat
    return forecasts


@njit(cache=True)
def _ar_variance_kernel(ar_coeffs: np.ndarray, sigma2: float, steps: int) -> np.ndarray:
    """Compute multi-step AR(p) forecast variance via MA(∞) recursion.

    For AR(p):  y_t = Σ ψ_j ε_{t-j}  where ψ_0=1, ψ_j = Σ_{k=1}^{min(j,p)} φ_k ψ_{j-k}
    Var[y_{T+h} | y_{1:T}] = σ² Σ_{j=0}^{h-1} ψ_j²

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (p,)
        AR coefficients φ₁, …, φ_p
    sigma2 : float
        Innovation variance σ²
    steps : int
        Forecast horizon

    Returns
    -------
    variances : np.ndarray, shape (steps,)
    """
    p = len(ar_coeffs)
    psi = np.zeros(steps)
    psi[0] = 1.0
    for j in range(1, steps):
        for k in range(1, min(j, p) + 1):
            psi[j] += ar_coeffs[k - 1] * psi[j - k]

    variances = np.zeros(steps)
    cumsum = 0.0
    for h in range(steps):
        cumsum += psi[h] * psi[h]
        variances[h] = sigma2 * cumsum
    return variances


@njit(cache=True)
def _garch_filter_kernel(residuals: np.ndarray, omega: float, alpha: float, beta: float, sigma2_init: float) -> tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) variance filter and log-likelihood sequence.

    Parameters
    ----------
    residuals : np.ndarray, shape (T,)
        Pre-computed AR residuals ε_t = y_t − AR_mean_t.
    omega, alpha, beta : float
        GARCH parameters.  Caller must ensure ω>0, α≥0, β≥0, α+β<1.
    sigma2_init : float
        Initial conditional variance σ²_0 (typically unconditional variance).

    Returns
    -------
    sigma2 : np.ndarray, shape (T,)
    log_liks : np.ndarray, shape (T,)
        Gaussian log p(ε_t | σ²_t) for each t.
    """
    T = len(residuals)
    LOG_2PI = 1.8378770664093455
    sigma2 = np.empty(T)
    sigma2[0] = sigma2_init
    for t in range(1, T):
        sigma2[t] = omega + alpha * residuals[t - 1] ** 2 + beta * sigma2[t - 1]
        # Guard against numerical underflow
        if sigma2[t] < 1e-300:
            sigma2[t] = sigma2_init

    log_liks = np.empty(T)
    for t in range(T):
        log_liks[t] = -0.5 * (LOG_2PI + np.log(sigma2[t]) + residuals[t] ** 2 / sigma2[t])
    return sigma2, log_liks


@njit(cache=True)
def _forward_step(ar_coeffs_all: np.ndarray, intercepts: np.ndarray, sigma2s: np.ndarray, transition_matrix: np.ndarray, alpha_prev: np.ndarray, history: np.ndarray, y_new: float) -> np.ndarray:
    """One causal forward-pass update: incorporate a single new observation.

    Parameters
    ----------
    ar_coeffs_all : np.ndarray, shape (K, p)
        AR coefficients per regime.
    intercepts : np.ndarray, shape (K,)
    sigma2s : np.ndarray, shape (K,)
        Innovation variances per regime.
    transition_matrix : np.ndarray, shape (K, K)
        Row-stochastic: transition_matrix[i,j] = P(s_{t+1}=j | s_t=i).
    alpha_prev : np.ndarray, shape (K,)
        Normalised filtered probability at the previous bar.
    history : np.ndarray, shape (p,)
        Last p observed values [y_{t-p+1}, …, y_t].
    y_new : float
        New observation y_{t+1}.

    Returns
    -------
    alpha_new : np.ndarray, shape (K,)
        Normalised filtered probability P(s_{t+1} | y_{1:t+1}).
    """
    K = len(alpha_prev)
    p = len(history)
    LOG_2PI = 1.8378770664093455  # ln(2π)

    # 1. Predict: π_{t+1|t}[j] = Σ_i P[i,j] · α_t[i]
    alpha_pred = np.zeros(K)
    for j in range(K):
        for i in range(K):
            alpha_pred[j] += transition_matrix[i, j] * alpha_prev[i]

    # 2. Gaussian log-likelihood per regime (uses log-sum-exp for stability)
    log_liks = np.zeros(K)
    for j in range(K):
        ar_mean = intercepts[j]
        for k in range(p):
            ar_mean += ar_coeffs_all[j, k] * history[p - 1 - k]
        residual = y_new - ar_mean
        log_liks[j] = -0.5 * (LOG_2PI + np.log(sigma2s[j]) + residual ** 2 / sigma2s[j])

    # Shift by max for numerical stability before exp
    max_ll = log_liks[0]
    for j in range(1, K):
        if log_liks[j] > max_ll:
            max_ll = log_liks[j]

    # 3. Update α_{t+1}[j] ∝ p(y|j) · π_{t+1|t}[j]
    alpha_new = np.zeros(K)
    for j in range(K):
        alpha_new[j] = alpha_pred[j] * np.exp(log_liks[j] - max_ll)

    # 4. Normalize
    total = 0.0
    for j in range(K):
        total += alpha_new[j]

    if total > 1e-300:
        for j in range(K):
            alpha_new[j] /= total
    else:
        # Degenerate: uniform fallback
        for j in range(K):
            alpha_new[j] = 1.0 / K

    return alpha_new


@njit(cache=True)
def _propagate_regime_probs(transition_matrix: np.ndarray, initial_probs: np.ndarray, steps: int) -> np.ndarray:
    """Propagate regime probabilities forward via the transition matrix.

    π_{t+1}[j] = Σ_i P[i,j] π_t[i]  (rows of transition_matrix sum to 1)

    Parameters
    ----------
    transition_matrix : np.ndarray, shape (K, K)
        Row-stochastic: transition_matrix[i,j] = P(s_{t+1}=j | s_t=i)
    initial_probs : np.ndarray, shape (K,)
        Regime probabilities at the last observed bar
    steps : int
        Forecast horizon

    Returns
    -------
    regime_probs : np.ndarray, shape (steps, K)
    """
    K = len(initial_probs)
    result = np.zeros((steps, K))
    probs = initial_probs.copy()
    for h in range(steps):
        new_probs = np.zeros(K)
        for j in range(K):
            for i in range(K):
                new_probs[j] += transition_matrix[i, j] * probs[i]
        probs = new_probs
        for j in range(K):
            result[h, j] = probs[j]
    return result