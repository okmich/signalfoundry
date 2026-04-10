"""
Markov-Switching Vector Autoregression (MS-VAR) for multi-asset regime-dependent forecasting.

Model
-----
Regime s_t ∈ {0, …, K-1}, Markov chain with transition matrix P.

Given s_t = k, the n-dimensional observation follows a VAR(p):

    Y_t = μ^(k) + A_1^(k) Y_{t-1} + … + A_p^(k) Y_{t-p} + ε_t
    ε_t ~ N(0, Σ^(k))

where:
    μ^(k)   ∈ ℝⁿ         (intercept)
    A_j^(k) ∈ ℝⁿˣⁿ       (VAR coefficient matrix at lag j)
    Σ^(k)   ∈ ℝⁿˣⁿ PD    (innovation covariance)

Parameter count per regime: n + p·n² + n(n+1)/2.
For K=2, n=3, p=2 this is 2×(3 + 18 + 6) = 54 free parameters.

EM Algorithm
------------
E-step: multivariate Gaussian log-likelihood via Cholesky, then Baum-Welch
    forward-backward for smoothed γ_t(k) and two-slice ξ_t(i,j).

M-step (all closed-form):
    Transition:  P[i,j] = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
    VAR params:  weighted OLS  B_k = (X'WX + λI)^{-1} X'WY
    Covariance:  Σ_k = Σ_t γ_t(k) ε_t ε_t' / Σ_t γ_t(k) + λI

Ridge regularisation (default λ=1e-6) prevents singularity in both the
OLS and the covariance when a regime has low effective occupancy.

Forecast
--------
Mean: recursive VAR substitution.
Covariance: Cov[Y_{T+h}|k] = Σ_{j=0}^{h-1} Ψ_j Σ^(k) Ψ_j'
    where Ψ_0 = I, Ψ_j = Σ_{m=1}^{min(j,p)} A_m^(k) Ψ_{j-m}.
Regime-weighted via law of total expectation / total covariance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from .base import BaseMarkovSwitching


class MarkovSwitchingVAR(BaseMarkovSwitching):
    """
    Markov-Switching VAR for multi-asset regime-dependent forecasting.

    Parameters
    ----------
    n_regimes : int, default=2
    order : int, default=1
        VAR lag order.
    ridge : float, default=1e-6
        Ridge regularisation for the weighted OLS and covariance M-steps.
    random_state : int, default=42

    Attributes
    ----------
    n_assets_ : int
        Number of assets / variables (inferred from data).
    filtered_probabilities_ : np.ndarray, shape (T-order, n_regimes)
    regime_probabilities_ : np.ndarray, shape (T-order, n_regimes)
    transition_matrix_ : np.ndarray, shape (n_regimes, n_regimes)
    aic, bic : float

    Examples
    --------
    >>> ms_var = MarkovSwitchingVAR(n_regimes=2, order=1)
    >>> ms_var.fit(returns_matrix)           # (T, n_assets)
    >>> fc = ms_var.forecast(steps=5, return_covariance=True)
    >>> fc['mean']       # (5, n_assets)
    >>> fc['covariance'] # (5, n_assets, n_assets)
    """

    def __init__(
        self,
        n_regimes: int = 2,
        order: int = 1,
        ridge: float = 1e-6,
        random_state: int = 42,
    ):
        super().__init__(n_regimes=n_regimes, random_state=random_state)
        if order < 1:
            raise ValueError("order must be at least 1")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.order = order
        self.ridge = ridge
        self.n_assets_: int | None = None
        self._params_: dict | None = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Fitting
    # ═══════════════════════════════════════════════════════════════════════════

    def fit(self, Y: np.ndarray, num_restarts: int = 3, maxiter: int = 100, tol: float = 1e-5) -> MarkovSwitchingVAR:
        """
        Fit MS-VAR via EM.

        Parameters
        ----------
        Y : np.ndarray, shape (T, n_assets)
        num_restarts : int, default=3
        maxiter : int, default=100
        tol : float, default=1e-5

        Returns
        -------
        self
        """
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2-D (T, n_assets), got ndim={Y.ndim}")
        T, n = Y.shape
        if n < 2:
            raise ValueError(
                f"n_assets must be >= 2, got {n}. Use MarkovSwitchingAR for univariate data."
            )
        if T < self.order + 20:
            raise ValueError(
                f"Time series too short. Need >= {self.order + 20} rows, got {T}"
            )

        self.data_ = Y
        self.n_assets_ = n

        X, Y_eff = self._build_design_matrix(Y)

        best_loglik = -np.inf
        best_result = None

        for restart in range(num_restarts):
            params = self._initialize_params(Y_eff, X, seed=restart)
            try:
                loglik, params, alpha_fwd, gamma = self._run_em(
                    Y_eff, X, params, maxiter, tol
                )
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_result = (params, alpha_fwd, gamma)
            except Exception as e:
                warnings.warn(f"EM restart {restart} failed: {e}")

        if best_result is None:
            raise RuntimeError(
                "All EM restarts failed. Try fewer regimes, lower order, or more data."
            )

        params, alpha_fwd, gamma = best_result
        self._params_ = params
        self.transition_matrix_ = params["P"]
        self.filtered_probabilities_ = alpha_fwd
        self.regime_probabilities_ = gamma

        self._history_buffer_ = Y[-self.order :].copy()     # (order, n)
        self._last_forward_alpha_ = alpha_fwd[-1].copy()
        self._n_updates_ = 0

        # AIC / BIC
        n_eff = len(Y_eff)
        n_free = (
            self.n_regimes * (n + self.order * n * n + n * (n + 1) // 2)
            + self.n_regimes * (self.n_regimes - 1)
        )
        self.aic = -2.0 * best_loglik + 2.0 * n_free
        self.bic = -2.0 * best_loglik + n_free * np.log(n_eff)

        self._is_fitted = True
        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # EM internals
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_em(self, Y_eff, X, params, maxiter, tol):
        prev_loglik = -np.inf
        alpha_fwd = gamma = None
        for _ in range(maxiter):
            alpha_fwd, gamma, xi, loglik = self._e_step(Y_eff, X, params)
            if abs(loglik - prev_loglik) < tol:
                break
            params = self._m_step(Y_eff, X, gamma, xi)
            prev_loglik = loglik
        return loglik, params, alpha_fwd, gamma

    # ── E-step ──────────────────────────────────────────────────────────────────

    def _e_step(self, Y_eff, X, params):
        K = self.n_regimes
        n_eff = len(Y_eff)

        # Per-regime log-likelihoods
        log_liks = np.empty((n_eff, K))
        for k in range(K):
            B_k = self._pack_B(params["intercepts"][k], params["ar_coeffs"][k])
            log_liks[:, k] = self._mvn_loglik(Y_eff, X, B_k, params["Sigma"][k])

        P = params["P"]
        pi0 = self._stationary_dist(P)

        # Forward pass (scaled)
        alpha = np.empty((n_eff, K))
        log_scale = np.empty(n_eff)

        ll_max = log_liks[0].max()
        liks = np.exp(log_liks[0] - ll_max)
        alpha[0] = pi0 * liks
        c = alpha[0].sum()
        alpha[0] /= max(c, 1e-300)
        log_scale[0] = ll_max + np.log(max(c, 1e-300))

        for t in range(1, n_eff):
            ll_max = log_liks[t].max()
            liks = np.exp(log_liks[t] - ll_max)
            alpha[t] = (alpha[t - 1] @ P) * liks
            c = alpha[t].sum()
            if c > 1e-300:
                alpha[t] /= c
            else:
                alpha[t] = 1.0 / K
                c = 1.0 / K
            log_scale[t] = ll_max + np.log(c)

        loglik = float(log_scale.sum())

        # Backward pass (scaled)
        beta_bw = np.ones((n_eff, K))
        for t in range(n_eff - 2, -1, -1):
            ll_max = log_liks[t + 1].max()
            liks_next = np.exp(log_liks[t + 1] - ll_max)
            v = P @ (liks_next * beta_bw[t + 1])
            s = v.sum()
            beta_bw[t] = v / max(s, 1e-300)

        # Smoothed probabilities γ
        gamma = alpha * beta_bw
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma /= np.where(row_sums > 0, row_sums, 1.0)

        # Two-slice marginals ξ
        xi = np.empty((n_eff - 1, K, K))
        for t in range(n_eff - 1):
            ll_max = log_liks[t + 1].max()
            liks_next = np.exp(log_liks[t + 1] - ll_max)
            xi[t] = np.outer(alpha[t], liks_next * beta_bw[t + 1]) * P
            xi_sum = xi[t].sum()
            if xi_sum > 1e-300:
                xi[t] /= xi_sum
            else:
                xi[t] = 1.0 / (K * K)

        return alpha, gamma, xi, loglik

    # ── M-step ──────────────────────────────────────────────────────────────────

    def _m_step(self, Y_eff, X, gamma, xi):
        K = self.n_regimes
        n = self.n_assets_
        p = self.order

        # Transition matrix (closed form)
        P_new = xi.sum(axis=0)
        row_sums = P_new.sum(axis=1, keepdims=True)
        P_new /= np.where(row_sums > 0, row_sums, 1.0)

        intercepts = np.empty((K, n))
        ar_coeffs = np.empty((K, p, n, n))
        Sigma = np.empty((K, n, n))

        for k in range(K):
            B_k, Sigma_k = self._m_step_regime(X, Y_eff, gamma[:, k])
            intercepts[k] = B_k[0, :]
            for lag in range(p):
                # B stores A' (transposed); recover A = B[block].T
                ar_coeffs[k, lag] = B_k[1 + lag * n : 1 + (lag + 1) * n, :].T
            Sigma[k] = Sigma_k

        return {
            "P": P_new,
            "intercepts": intercepts,
            "ar_coeffs": ar_coeffs,
            "Sigma": Sigma,
        }

    def _m_step_regime(self, X, Y_eff, gamma_k):
        """Closed-form weighted OLS + weighted covariance for one regime.

        B_k = (X'WX + λI)^{-1} X'WY
        Σ_k = Σ_t w_t ε_t ε_t' / Σ_t w_t  + λI
        """
        n = self.n_assets_

        # Efficient weighted least squares: weight by √γ
        sqrt_w = np.sqrt(np.maximum(gamma_k, 0.0))[:, None]
        X_w = X * sqrt_w                                         # (T_eff, d)
        Y_w = Y_eff * sqrt_w                                     # (T_eff, n)

        d = X.shape[1]
        XwX = X_w.T @ X_w + self.ridge * np.eye(d)              # (d, d)
        XwY = X_w.T @ Y_w                                       # (d, n)

        try:
            B_k = np.linalg.solve(XwX, XwY)                     # (d, n)
        except np.linalg.LinAlgError:
            B_k = np.linalg.lstsq(XwX, XwY, rcond=None)[0]

        # Weighted residual covariance
        resid = Y_eff - X @ B_k                                 # (T_eff, n)
        w_sum = gamma_k.sum()

        if w_sum > 1e-10:
            Sigma_k = (resid * gamma_k[:, None]).T @ resid / w_sum
        else:
            Sigma_k = np.eye(n) * np.var(Y_eff)

        # Ensure symmetric PD
        Sigma_k = 0.5 * (Sigma_k + Sigma_k.T) + self.ridge * np.eye(n)

        return B_k, Sigma_k

    # ═══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_design_matrix(self, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y_eff) for weighted OLS.

        X[t] = [1, Y_{t+p-1}', Y_{t+p-2}', …, Y_t']     shape: (T-p, 1+n·p)
        Y_eff = Y[p:]                                       shape: (T-p, n)
        """
        T, n = Y.shape
        p = self.order
        n_eff = T - p
        X = np.ones((n_eff, 1 + n * p))
        for t in range(n_eff):
            for lag in range(p):
                X[t, 1 + lag * n : 1 + (lag + 1) * n] = Y[p + t - 1 - lag]
        return X, Y[p:].copy()

    def _pack_B(self, intercept: np.ndarray, ar_coeffs_k: np.ndarray) -> np.ndarray:
        """Pack intercept + VAR coefficient matrices into OLS B matrix.

        B[0]          = μ'                  (1, n)
        B[1+lag·n …]  = A_{lag+1}'         (n, n) — transposed!
        """
        n = self.n_assets_
        p = self.order
        B = np.empty((1 + n * p, n))
        B[0, :] = intercept
        for lag in range(p):
            B[1 + lag * n : 1 + (lag + 1) * n, :] = ar_coeffs_k[lag].T
        return B

    def _mvn_loglik(
        self,
        Y_eff: np.ndarray,
        X: np.ndarray,
        B_k: np.ndarray,
        Sigma_k: np.ndarray,
    ) -> np.ndarray:
        """log N(Y_t; X_t B_k, Σ_k) for each t, via Cholesky."""
        n = Y_eff.shape[1]
        resid = Y_eff - X @ B_k                             # (T_eff, n)

        # Cholesky of Σ_k (with safety for near-singular cases)
        try:
            L = np.linalg.cholesky(Sigma_k)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(Sigma_k + 1e-6 * np.eye(n))

        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        # Solve L^{-1} resid for each t: inv_L_resid = L^{-1} resid'
        inv_L_resid = solve_triangular(L, resid.T, lower=True).T   # (T_eff, n)
        maha = np.sum(inv_L_resid ** 2, axis=1)                    # (T_eff,)

        return -0.5 * (n * np.log(2.0 * np.pi) + log_det + maha)

    def _initialize_params(self, Y_eff, X, seed):
        rng = np.random.default_rng(seed + self.random_state)
        K = self.n_regimes
        n = self.n_assets_
        p = self.order
        d = X.shape[1]

        # Transition matrix: persistent
        P = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(P, 0.95)

        # OLS on full series as baseline
        XtX = X.T @ X + self.ridge * np.eye(d)
        XtY = X.T @ Y_eff
        try:
            B_ols = np.linalg.solve(XtX, XtY)
        except np.linalg.LinAlgError:
            B_ols = np.zeros((d, n))

        resid_ols = Y_eff - X @ B_ols
        Sigma_ols = resid_ols.T @ resid_ols / max(len(Y_eff), 1)
        Sigma_ols = 0.5 * (Sigma_ols + Sigma_ols.T) + self.ridge * np.eye(n)

        intercepts = np.empty((K, n))
        ar_coeffs = np.empty((K, p, n, n))
        Sigma = np.empty((K, n, n))

        for k in range(K):
            perturb = rng.normal(0, 0.05, B_ols.shape)
            B_k = B_ols + perturb
            intercepts[k] = B_k[0, :]
            for lag in range(p):
                ar_coeffs[k, lag] = B_k[1 + lag * n : 1 + (lag + 1) * n, :].T

            scale = rng.uniform(0.7, 1.3)
            Sigma[k] = Sigma_ols * scale
            Sigma[k] = 0.5 * (Sigma[k] + Sigma[k].T) + self.ridge * np.eye(n)

        return {
            "P": P,
            "intercepts": intercepts,
            "ar_coeffs": ar_coeffs,
            "Sigma": Sigma,
        }

    @staticmethod
    def _stationary_dist(P: np.ndarray) -> np.ndarray:
        K = P.shape[0]
        A = P.T - np.eye(K)
        A[-1] = 1.0
        b = np.zeros(K)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.clip(pi, 0, None)
            pi /= pi.sum()
        except np.linalg.LinAlgError:
            pi = np.ones(K) / K
        return pi

    # ═══════════════════════════════════════════════════════════════════════════
    # Inference
    # ═══════════════════════════════════════════════════════════════════════════

    def predict_regime(self, causal: bool = False) -> np.ndarray:
        """Most likely regime at each time step (length T − order)."""
        self._validate_fitted()
        probs = self.filtered_probabilities_ if causal else self.regime_probabilities_
        return np.argmax(probs, axis=1)

    def predict_regime_proba(self, causal: bool = False) -> np.ndarray:
        """Regime probabilities, shape (T − order, n_regimes)."""
        self._validate_fitted()
        return self.filtered_probabilities_ if causal else self.regime_probabilities_

    # ═══════════════════════════════════════════════════════════════════════════
    # Forecasting
    # ═══════════════════════════════════════════════════════════════════════════

    def forecast(self, steps: int = 1, regime: int | None = None, return_covariance: bool = False, causal: bool = True) -> np.ndarray | dict[str, np.ndarray]:
        """
        Multi-step VAR forecast with regime uncertainty.

        Parameters
        ----------
        steps : int, default=1
        regime : int, optional
            Condition on a single regime.  None → regime-weighted.
        return_covariance : bool, default=False
            If True, return dict with mean, covariance, regime details.
        causal : bool, default=True

        Returns
        -------
        np.ndarray, shape (steps, n_assets)   — if not return_covariance
        dict with:
            'mean'        : (steps, n_assets)
            'covariance'  : (steps, n_assets, n_assets)
            'regime_probabilities': (steps, K)
            'regime_forecasts': dict[int, (steps, n_assets)]
        """
        self._validate_fitted()

        if regime is not None:
            return self._forecast_single_regime(regime, steps, return_covariance)

        regime_probs = self.forecast_regime_probabilities(steps, causal=causal)

        regime_means: dict[int, np.ndarray] = {}
        regime_covs: dict[int, np.ndarray] = {}
        for r in range(self.n_regimes):
            out = self._forecast_single_regime(r, steps, return_covariance=True)
            regime_means[r] = out["mean"]
            regime_covs[r] = out["covariance"]

        n = self.n_assets_

        # Law of total expectation
        mean = np.zeros((steps, n))
        for r in range(self.n_regimes):
            mean += regime_probs[:, r : r + 1] * regime_means[r]

        if not return_covariance:
            return mean

        # Law of total covariance
        covariance = np.zeros((steps, n, n))
        for h in range(steps):
            for r in range(self.n_regimes):
                pr = regime_probs[h, r]
                covariance[h] += pr * regime_covs[r][h]
                diff = (regime_means[r][h] - mean[h])[:, None]
                covariance[h] += pr * (diff @ diff.T)

        return {
            "mean": mean,
            "covariance": covariance,
            "regime_probabilities": regime_probs,
            "regime_forecasts": dict(regime_means),
        }

    def _forecast_single_regime(self, regime: int, steps: int, return_covariance: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        n = self.n_assets_
        p = self.order
        intercept = self._params_["intercepts"][regime]
        ar_mats = self._params_["ar_coeffs"][regime]      # (p, n, n)
        Sigma = self._params_["Sigma"][regime]

        # ── Mean: recursive VAR ──────────────────────────────────────────────
        # history[-1] is the most recent observation
        history = [self._history_buffer_[i] for i in range(self.order)]
        mean = np.empty((steps, n))
        for h in range(steps):
            y_hat = intercept.copy()
            for lag in range(min(p, len(history))):
                y_hat = y_hat + ar_mats[lag] @ history[-(lag + 1)]
            mean[h] = y_hat
            history.append(y_hat)

        if not return_covariance:
            return mean

        # ── Impulse-response matrices Ψ_j ────────────────────────────────────
        # Ψ_0 = I,  Ψ_j = Σ_{m=1}^{min(j,p)} A_m Ψ_{j-m}
        Psi = [np.eye(n)]
        for j in range(1, steps):
            Psi_j = np.zeros((n, n))
            for m in range(1, min(j, p) + 1):
                Psi_j += ar_mats[m - 1] @ Psi[j - m]
            Psi.append(Psi_j)

        # ── Forecast covariance ───────────────────────────────────────────────
        # Cov[Y_{T+h+1} | F_T, s=k] = Σ_{j=0}^{h} Ψ_j Σ Ψ_j'
        covariance = np.zeros((steps, n, n))
        cumul = np.zeros((n, n))
        for h in range(steps):
            cumul = cumul + Psi[h] @ Sigma @ Psi[h].T
            covariance[h] = cumul

        return {"mean": mean, "covariance": covariance}

    # ═══════════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ═══════════════════════════════════════════════════════════════════════════

    def get_regime_parameters(self) -> pd.DataFrame:
        """
        Summary statistics per regime.

        Columns: regime, spectral_radius, avg_volatility, min_volatility,
        max_volatility, avg_correlation, log_det_Sigma.

        For full matrices use :meth:`get_var_matrices`.
        """
        self._validate_fitted()
        rows = []
        for k in range(self.n_regimes):
            A1 = self._params_["ar_coeffs"][k, 0]
            Sigma = self._params_["Sigma"][k]

            eigvals = np.linalg.eigvals(A1)
            spectral_radius = float(np.max(np.abs(eigvals)))

            vols = np.sqrt(np.diag(Sigma))
            D_inv = np.diag(1.0 / np.maximum(vols, 1e-10))
            corr = D_inv @ Sigma @ D_inv
            nn = len(vols)
            avg_corr = float((corr.sum() - nn) / max(nn * (nn - 1), 1))

            rows.append(
                {
                    "regime": k,
                    "spectral_radius": spectral_radius,
                    "avg_volatility": float(vols.mean()),
                    "min_volatility": float(vols.min()),
                    "max_volatility": float(vols.max()),
                    "avg_correlation": avg_corr,
                    "log_det_Sigma": float(np.linalg.slogdet(Sigma)[1]),
                }
            )
        return pd.DataFrame(rows)

    def get_var_matrices(self, regime: int) -> dict:
        """
        Full VAR matrices for a specific regime.

        Returns
        -------
        dict with keys:
            'intercept'  : np.ndarray (n,)
            'ar_coeffs'  : list[np.ndarray (n, n)]  — A_1, …, A_p
            'Sigma'      : np.ndarray (n, n)
        """
        self._validate_fitted()
        if regime < 0 or regime >= self.n_regimes:
            raise ValueError(f"Regime {regime} not in [0, {self.n_regimes})")
        return {
            "intercept": self._params_["intercepts"][regime].copy(),
            "ar_coeffs": [
                self._params_["ar_coeffs"][regime, lag].copy()
                for lag in range(self.order)
            ],
            "Sigma": self._params_["Sigma"][regime].copy(),
        }

    def interpret_regimes(self) -> dict[int, str]:
        """Label regimes by VAR persistence, volatility, and correlation."""
        self._validate_fitted()
        params = self.get_regime_parameters()
        median_vol = params["avg_volatility"].median()
        interpretations = {}
        for _, row in params.iterrows():
            r = int(row["regime"])
            sr = row["spectral_radius"]
            persist = (
                "Persistent" if sr > 0.9 else ("Moderate" if sr > 0.5 else "Transient")
            )
            vol = "high_vol" if row["avg_volatility"] >= median_vol else "low_vol"
            corr = (
                "pos_corr"
                if row["avg_correlation"] > 0.1
                else ("neg_corr" if row["avg_correlation"] < -0.1 else "uncorr")
            )
            interpretations[r] = f"{persist} rho(A1)={sr:.3f} ({vol}, {corr})"
        return interpretations