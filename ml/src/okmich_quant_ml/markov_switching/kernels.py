"""
Numba-JIT kernels shared across Markov-switching model families.

All kernels are pure-numeric and have no dependency on model classes, making
them reusable across MarkovSwitchingAR, MarkovSwitchingGARCH, and MarkovSwitchingVAR.
"""

from __future__ import annotations

import numpy as np
from numba import njit


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
    sigma2 = np.empty(T)
    sigma2[0] = sigma2_init
    for t in range(1, T):
        sigma2[t] = _garch_variance_step(omega, alpha, beta, residuals[t - 1], sigma2[t - 1], sigma2_init)

    log_liks = np.empty(T)
    for t in range(T):
        log_liks[t] = _gaussian_loglik(residuals[t], sigma2[t])
    return sigma2, log_liks


@njit(cache=True)
def _garch_variance_step(omega: float, alpha: float, beta: float, resid_prev: float, sigma2_prev: float,
                         sigma2_reset: float) -> float:
    """σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}, reset to ``sigma2_reset`` on numerical underflow."""
    sigma2 = omega + alpha * resid_prev ** 2 + beta * sigma2_prev
    if sigma2 < 1e-300:
        sigma2 = sigma2_reset
    return sigma2


@njit(cache=True)
def _gaussian_loglik(resid: float, sigma2: float) -> float:
    """log N(resid; 0, sigma2)."""
    return -0.5 * (1.8378770664093455 + np.log(sigma2) + resid ** 2 / sigma2)


# ─── Scaled forward filter (shared by E-steps and streaming update()) ────────────


@njit(cache=True)
def _predict_regime_probs(alpha_prev: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    """One-step prior π_{t|t−1}[j] = Σ_i P[i, j] α_{t−1}[i] (P row-stochastic)."""
    K = len(alpha_prev)
    prior = np.zeros(K)
    for j in range(K):
        for i in range(K):
            prior[j] += transition_matrix[i, j] * alpha_prev[i]
    return prior


@njit(cache=True)
def _scaled_update(prior: np.ndarray, log_liks: np.ndarray) -> tuple[np.ndarray, float]:
    """Bayes update α ∝ prior · exp(log_liks), normalised in log space.

    The stabilising shift is the max of ``log(prior) + log_liks`` over states with prior mass. Shifting by the max
    log-likelihood alone would let an impossible (zero-prior) state set the scale, underflow every possible state to 0
    and fall back to uniform. NaN log-likelihoods propagate to NaN outputs.

    Returns
    -------
    posterior : np.ndarray
    log_norm : float
        log Σ prior · exp(log_liks) — the step's predictive log-likelihood; −inf if no state is possible.
    """
    n = len(prior)
    posterior = np.empty(n)
    shift = -np.inf
    for i in range(n):
        if prior[i] > 0.0:
            score = np.log(prior[i]) + log_liks[i]
            posterior[i] = score
            if score > shift:
                shift = score
        else:
            posterior[i] = -np.inf
    if shift == -np.inf:
        for i in range(n):
            posterior[i] = 1.0 / n
        return posterior, -np.inf
    total = 0.0
    for i in range(n):
        posterior[i] = np.exp(posterior[i] - shift)
        total += posterior[i]
    for i in range(n):
        posterior[i] /= total
    return posterior, shift + np.log(total)


@njit(cache=True)
def _scaled_forward_filter(initial_probs: np.ndarray, transition_matrix: np.ndarray,
                           log_liks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filtered P(s_t | y_{1:t}) for all t and the per-step log-normalisers (summing to the log-likelihood).

    Row 0 updates ``initial_probs``; later rows update the one-step prior. Streaming ``update()`` calls the same
    ``_predict_regime_probs`` + ``_scaled_update`` pair, so batch and streaming filters agree bit for bit.
    """
    T, K = log_liks.shape
    alpha = np.empty((T, K))
    log_norm = np.empty(T)
    for t in range(T):
        prior = initial_probs if t == 0 else _predict_regime_probs(alpha[t - 1], transition_matrix)
        alpha[t], log_norm[t] = _scaled_update(prior, log_liks[t])
    return alpha, log_norm


# ─── Mean-form MS-AR(p) (Hamilton 1989) ──────────────────────────────────────────
#
#     y_t − μ_{s_t} = Σ_{l=1}^{p} φ_{s_t,l} (y_{t−l} − μ_{s_{t−l}}) + ε_t,   ε_t ~ N(0, σ²_{s_t})
#
# The conditional mean depends on the last p+1 regimes, so the exact filter runs on the joint path
# (s_t, s_{t−1}, …, s_{t−p}). A joint vector has K^(p+1) entries in C order with s_t as the most
# significant digit — the same layout as statsmodels' ``filtered_joint_probabilities[..., t]``.

# statsmodels' Hamilton filter floors transition probabilities at 1e-20 before taking logs; match it.
_TRANSITION_FLOOR = 1e-20


@njit(cache=True)
def _ms_ar_joint_forward_step(ar_coeffs: np.ndarray, means: np.ndarray, sigma2s: np.ndarray,
                              transition_matrix: np.ndarray, joint_prev: np.ndarray, history: np.ndarray,
                              y_new: float) -> np.ndarray:
    """One causal Hamilton-filter step of a mean-form MS-AR(p) on the joint regime path.

    Parameters
    ----------
    ar_coeffs : np.ndarray, shape (K, p)
        φ_{k,l}: AR coefficient at lag l when the current regime is k.
    means : np.ndarray, shape (K,)
        Regime means μ_k (statsmodels ``const[k]``).
    sigma2s : np.ndarray, shape (K,)
        Innovation variance of the current regime.
    transition_matrix : np.ndarray, shape (K, K)
        Row-stochastic: transition_matrix[i, j] = P(s_{t+1}=j | s_t=i).
    joint_prev : np.ndarray, shape (K^(p+1),)
        P(s_t, s_{t−1}, …, s_{t−p} | y_{1:t}).
    history : np.ndarray, shape (p,)
        Last p observed values [y_{t−p+1}, …, y_t].
    y_new : float
        New observation y_{t+1}.

    Returns
    -------
    joint_new : np.ndarray, shape (K^(p+1),)
        P(s_{t+1}, s_t, …, s_{t+1−p} | y_{1:t+1}).
    """
    K = len(means)
    p = len(history)
    n = len(joint_prev)
    k_pow_p = n // K

    # 1. Predict: drop the oldest regime s_{t−p}, prepend s_{t+1}
    pred = np.zeros(n)
    for idx in range(n):
        w = joint_prev[idx]
        if w == 0.0:
            continue
        s_cur = idx // k_pow_p
        tail = idx // K
        for s_new in range(K):
            pred[s_new * k_pow_p + tail] += max(transition_matrix[s_cur, s_new], _TRANSITION_FLOOR) * w

    # 2. Gaussian log-likelihood conditional on the (p+1)-regime path
    log_liks = np.empty(n)
    for idx in range(n):
        s_new = idx // k_pow_p
        resid = y_new - means[s_new]
        div = k_pow_p
        for lag in range(1, p + 1):
            div //= K
            s_lag = (idx // div) % K
            resid -= ar_coeffs[s_new, lag - 1] * (history[p - lag] - means[s_lag])
        log_liks[idx] = _gaussian_loglik(resid, sigma2s[s_new])

    # 3. Update over paths with predicted mass (log-space shift — see _scaled_update)
    return _scaled_update(pred, log_liks)[0]


@njit(cache=True)
def _ms_ar_deviation_moments(means: np.ndarray, joint: np.ndarray,
                             history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regime-split moments of the deviation state Z_T = (y_T − μ_{s_T}, …, y_{T−p+1} − μ_{s_{T−p+1}}).

    Parameters
    ----------
    means : np.ndarray, shape (K,)
    joint : np.ndarray, shape (K^(p+1),)
        Weights over the path (s_T, …, s_{T−p}); need not be normalised.
    history : np.ndarray, shape (p,)
        [y_{T−p+1}, …, y_T].

    Returns
    -------
    weight : np.ndarray, shape (K,)
        Σ_paths w · 1{s_T = k}
    first : np.ndarray, shape (K, p)
        Σ_paths w · Z_T · 1{s_T = k}
    second : np.ndarray, shape (K, p, p)
        Σ_paths w · Z_T Z_T' · 1{s_T = k}
    """
    K = len(means)
    p = len(history)
    n = len(joint)
    k_pow_p = n // K
    weight = np.zeros(K)
    first = np.zeros((K, p))
    second = np.zeros((K, p, p))
    z = np.empty(p)
    for idx in range(n):
        w = joint[idx]
        if w == 0.0:
            continue
        div = k_pow_p
        for lag in range(p):
            z[lag] = history[p - 1 - lag] - means[(idx // div) % K]
            div //= K
        k = idx // k_pow_p
        weight[k] += w
        for a in range(p):
            first[k, a] += w * z[a]
            for b in range(p):
                second[k, a, b] += w * z[a] * z[b]
    return weight, first, second


@njit(cache=True)
def _companion(ar_row: np.ndarray) -> np.ndarray:
    """Companion matrix F of an AR(p) with coefficients ``ar_row``: Z_t = F Z_{t−1} + ε_t e₁."""
    p = len(ar_row)
    F = np.zeros((p, p))
    for j in range(p):
        F[0, j] = ar_row[j]
    for j in range(1, p):
        F[j, j - 1] = 1.0
    return F


@njit(cache=True)
def _ms_ar_forecast_moments(ar_coeffs: np.ndarray, sigma2s: np.ndarray, transition_matrix: np.ndarray,
                            weight0: np.ndarray, first0: np.ndarray, second0: np.ndarray, steps: int,
                            with_second: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact h-step moments of the deviation state, split by the terminal regime s_{T+h}.

    With Z_{T+h} = F_{s_{T+h}} Z_{T+h−1} + ε e₁ and s_{T+h} ⊥ Z_{T+h−1} | s_{T+h−1}:

        π_h(j) = Σ_i P[i,j] π_{h−1}(i)
        m_h(j) = F_j Σ_i P[i,j] m_{h−1}(i)
        S_h(j) = F_j (Σ_i P[i,j] S_{h−1}(i)) F_j' + π_h(j) σ²_j e₁e₁'

    where m_h(j) = E[Z_{T+h} 1{s_{T+h}=j} | y_{1:T}] and S_h(j) the matching second moment. No
    approximation: the recursion is exact for any K and p. ``with_second=False`` skips S (mean-only forecasts) and
    returns zeros for ``second``.

    Returns
    -------
    probs : np.ndarray, shape (steps, K)     π_h(j)
    first : np.ndarray, shape (steps, K)     m_h(j)[0]   = E[z_{T+h} 1{s_{T+h}=j}]
    second : np.ndarray, shape (steps, K)    S_h(j)[0,0] = E[z²_{T+h} 1{s_{T+h}=j}]
    """
    K, p = ar_coeffs.shape
    companions = np.empty((K, p, p))
    for j in range(K):
        companions[j] = _companion(ar_coeffs[j])
    w = weight0.copy()
    m = first0.copy()
    S = second0.copy()
    probs = np.zeros((steps, K))
    first = np.zeros((steps, K))
    second = np.zeros((steps, K))
    for h in range(steps):
        w_new = np.zeros(K)
        m_new = np.zeros((K, p))
        S_new = np.zeros((K, p, p))
        for j in range(K):
            mix_m = np.zeros(p)
            mix_S = np.zeros((p, p))
            for i in range(K):
                pij = transition_matrix[i, j]
                w_new[j] += pij * w[i]
                mix_m += pij * m[i]
                if with_second:
                    mix_S += pij * S[i]
            F = companions[j]
            m_new[j] = F @ mix_m
            probs[h, j] = w_new[j]
            first[h, j] = m_new[j, 0]
            if with_second:
                S_new[j] = F @ mix_S @ F.T
                S_new[j, 0, 0] += w_new[j] * sigma2s[j]
                second[h, j] = S_new[j, 0, 0]
        w = w_new
        m = m_new
        S = S_new
    return probs, first, second


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