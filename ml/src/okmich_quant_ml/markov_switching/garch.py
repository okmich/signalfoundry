"""
Markov-Switching GARCH (MS-GARCH) for regime-dependent volatility forecasting.

Model
-----
Regime s_t ~ Markov chain with K states and transition matrix P.

Given regime s_t = k:

    y_t = μ^(k) + Σ_{j=1}^p φ_j^(k) y_{t-j} + ε_t
    ε_t = σ_t^(k) z_t,   z_t ~ N(0, 1)
    σ_t²^(k) = ω^(k) + α^(k) ε_{t-1}² + β^(k) σ_{t-1}²

Identification
--------------
Each regime has its own GARCH filter running on the full residual sequence
(the *regime-conditional* approximation).  The exact Gray (1996) filter is
not implemented; the regime-conditional approach is the standard practical
choice.

EM Algorithm
------------
E-step:
  For each regime k, compute AR residuals and run the GARCH(1,1) filter to
  obtain log p(y_t | s_t=k, F_{t-1}).  Then run the Baum-Welch
  forward-backward pass to obtain smoothed γ_t(k) = P(s_t=k | y_{1:T}) and
  two-slice marginals ξ_t(i, j) = P(s_t=i, s_{t+1}=j | y_{1:T}).

M-step:
  Transition matrix: closed-form update from ξ.
  GARCH-AR parameters: per-regime SLSQP to maximise Σ_t γ_t(k) log p(y_t | k).

Stationarity
------------
GARCH stationarity constraint: α^(k) + β^(k) < 1 enforced via SLSQP
inequality constraint.  ω^(k) > 0 enforced via a log-transform in the
optimisation.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.optimize import minimize

from .base import BaseMarkovSwitching
from .kernels import _garch_filter_kernel, _propagate_regime_probs


class MarkovSwitchingGARCH(BaseMarkovSwitching):
    """
    Markov-Switching GARCH model.

    Unlike MS-AR (constant variance per regime), MS-GARCH models
    time-varying conditional volatility within each regime.

    Parameters
    ----------
    n_regimes : int, default=2
    order : int, default=1
        AR order of the mean equation.
    random_state : int, default=42

    Attributes
    ----------
    filtered_probabilities_ : np.ndarray, shape (n_obs, n_regimes)
        Causal filtered P(s_t | y_{1:t}).
    regime_probabilities_ : np.ndarray, shape (n_obs, n_regimes)
        Smoothed P(s_t | y_{1:T}).
    transition_matrix_ : np.ndarray, shape (n_regimes, n_regimes)
    aic, bic : float

    Notes
    -----
    The first ``order`` observations are used as initial conditions for the
    AR mean equation; the model is estimated on observations p+1 … T.
    Effective sample size is T - order.
    """

    def __init__(self, n_regimes: int = 2, order: int = 1, random_state: int = 42):
        super().__init__(n_regimes=n_regimes, random_state=random_state)
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self._params_: dict | None = None
        self._last_garch_sigma2_: np.ndarray | None = None  # (K,) at last obs
        self._last_ar_residuals_: np.ndarray | None = None  # (K,) at last obs

    # ─── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, y: np.ndarray, num_restarts: int = 3, maxiter: int = 100, tol: float = 1e-5) -> MarkovSwitchingGARCH:
        """
        Fit MS-GARCH via EM algorithm.

        Parameters
        ----------
        y : np.ndarray, shape (T,)
        num_restarts : int, default=3
            Number of random EM initialisations; best log-likelihood wins.
        maxiter : int, default=100
            Maximum EM iterations per restart.
        tol : float, default=1e-5
            Convergence tolerance on log-likelihood change.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64).flatten()
        if len(y) < self.order + 20:
            raise ValueError(
                f"Time series too short. Need at least {self.order + 20} samples, "
                f"got {len(y)}"
            )

        self.data_ = y
        best_loglik = -np.inf
        best_result = None

        for restart in range(num_restarts):
            params = self._initialize_params(y, seed=restart)
            try:
                loglik, params, alpha_fwd, gamma, garch_s2 = self._run_em(
                    y, params, maxiter, tol
                )
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_result = (params, alpha_fwd, gamma, garch_s2)
            except Exception as e:
                warnings.warn(f"EM restart {restart} failed: {e}")

        if best_result is None:
            raise RuntimeError(
                "All EM restarts failed. Try fewer regimes, lower order, or more data."
            )

        params, alpha_fwd, gamma, garch_s2 = best_result
        self._params_ = params
        self.transition_matrix_ = params["P"]
        self.filtered_probabilities_ = alpha_fwd          # (n_obs, K)
        self.regime_probabilities_ = gamma                 # (n_obs, K)  smoothed

        # Store GARCH state at last observation for forecasting
        self._last_garch_sigma2_ = garch_s2[-1]           # (K,)
        self._last_ar_residuals_ = np.array([
            self._compute_ar_residuals(y, params["intercepts"][k], params["ar_coeffs"][k])[-1]
            for k in range(self.n_regimes)
        ])
        self._history_buffer_ = y[-self.order:].copy()

        n_obs = len(y) - self.order
        n_params = self.n_regimes * (1 + self.order + 3) + self.n_regimes * (self.n_regimes - 1)
        self.aic = -2.0 * best_loglik + 2.0 * n_params
        self.bic = -2.0 * best_loglik + n_params * np.log(n_obs)

        self._last_forward_alpha_ = alpha_fwd[-1].copy()
        self._n_updates_ = 0
        self._is_fitted = True
        return self

    # ─── EM internals ───────────────────────────────────────────────────────────

    def _run_em(self, y: np.ndarray, params: dict, maxiter: int, tol: float) -> tuple:
        prev_loglik = -np.inf
        for _ in range(maxiter):
            alpha_fwd, gamma, xi, loglik, garch_s2 = self._e_step(y, params)
            params = self._m_step(y, params, gamma, xi)
            if abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik
        return loglik, params, alpha_fwd, gamma, garch_s2

    def _e_step(self, y: np.ndarray, params: dict) -> tuple:
        K = self.n_regimes
        n = len(y) - self.order  # effective sample size

        # ── GARCH filters: log p(y_t | s_t=k, F_{t-1}) ─────────────────────────
        log_liks = np.zeros((n, K))
        garch_s2 = np.zeros((n, K))
        for k in range(K):
            residuals = self._compute_ar_residuals(
                y, params["intercepts"][k], params["ar_coeffs"][k]
            )
            ab = params["alpha"][k] + params["beta"][k]
            sigma2_init = (params["omega"][k] / (1.0 - ab)) if ab < 1.0 else np.var(residuals)
            s2, ll = _garch_filter_kernel(
                residuals, params["omega"][k], params["alpha"][k], params["beta"][k], sigma2_init
            )
            log_liks[:, k] = ll
            garch_s2[:, k] = s2

        P = params["P"]
        pi0 = self._stationary_dist(P)

        # ── Forward pass (scaled) ────────────────────────────────────────────────
        alpha = np.zeros((n, K))
        log_scale = np.zeros(n)

        ll_shifted = log_liks[0] - log_liks[0].max()
        liks = np.exp(ll_shifted)
        alpha[0] = pi0 * liks
        c = alpha[0].sum()
        alpha[0] /= max(c, 1e-300)
        log_scale[0] = log_liks[0].max() + np.log(max(c, 1e-300))

        for t in range(1, n):
            ll_shifted = log_liks[t] - log_liks[t].max()
            liks = np.exp(ll_shifted)
            alpha_pred = alpha[t - 1] @ P
            alpha[t] = alpha_pred * liks
            c = alpha[t].sum()
            if c > 1e-300:
                alpha[t] /= c
            else:
                alpha[t] = 1.0 / K
                c = 1.0 / K
            log_scale[t] = log_liks[t].max() + np.log(c)

        loglik = float(log_scale.sum())

        # ── Backward pass (scaled) ───────────────────────────────────────────────
        beta_bw = np.ones((n, K))
        for t in range(n - 2, -1, -1):
            ll_shifted = log_liks[t + 1] - log_liks[t + 1].max()
            liks_next = np.exp(ll_shifted)
            beta_pred = P @ (liks_next * beta_bw[t + 1])
            s = beta_pred.sum()
            beta_bw[t] = beta_pred / max(s, 1e-300)

        # ── Smoothed probabilities ───────────────────────────────────────────────
        gamma = alpha * beta_bw
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma /= np.where(row_sums > 0, row_sums, 1.0)

        # ── Two-slice marginals ξ_t(i, j) ────────────────────────────────────────
        xi = np.zeros((n - 1, K, K))
        for t in range(n - 1):
            ll_shifted = log_liks[t + 1] - log_liks[t + 1].max()
            liks_next = np.exp(ll_shifted)
            xi[t] = np.outer(alpha[t], liks_next * beta_bw[t + 1]) * P
            xi_sum = xi[t].sum()
            if xi_sum > 1e-300:
                xi[t] /= xi_sum

        return alpha, gamma, xi, loglik, garch_s2

    def _m_step(self, y: np.ndarray, params: dict, gamma: np.ndarray, xi: np.ndarray) -> dict:
        K = self.n_regimes
        new_params = {}

        # Transition matrix (closed form)
        P_new = xi.sum(axis=0)
        row_sums = P_new.sum(axis=1, keepdims=True)
        P_new /= np.where(row_sums > 0, row_sums, 1.0)
        new_params["P"] = P_new

        # Per-regime AR + GARCH (nested optimisation)
        intercepts = np.empty(K)
        ar_coeffs = np.empty((K, self.order))
        omega = np.empty(K)
        alpha = np.empty(K)
        beta = np.empty(K)

        for k in range(K):
            pk = self._m_step_regime(y, gamma[:, k], {
                "intercept": params["intercepts"][k],
                "ar_coeffs": params["ar_coeffs"][k].copy(),
                "omega": params["omega"][k],
                "alpha": params["alpha"][k],
                "beta": params["beta"][k],
            })
            intercepts[k] = pk["intercept"]
            ar_coeffs[k] = pk["ar_coeffs"]
            omega[k] = pk["omega"]
            alpha[k] = pk["alpha"]
            beta[k] = pk["beta"]

        new_params["intercepts"] = intercepts
        new_params["ar_coeffs"] = ar_coeffs
        new_params["omega"] = omega
        new_params["alpha"] = alpha
        new_params["beta"] = beta
        return new_params

    def _m_step_regime(self, y: np.ndarray, gamma_k: np.ndarray, params_k: dict) -> dict:
        """SLSQP to maximise Σ_t γ_t(k) log p(y_t | k, θ_k)."""
        p = self.order

        def neg_wll(x):
            intercept = x[0]
            ar_ = x[1: 1 + p]
            omega_ = np.exp(x[1 + p])           # log-transform → always positive
            alpha_ = x[2 + p]
            beta_ = x[3 + p]
            ab = alpha_ + beta_
            if ab >= 1.0 or alpha_ < 0 or beta_ < 0 or omega_ <= 0:
                return 1e10
            residuals = self._compute_ar_residuals(y, intercept, ar_)
            sigma2_init = omega_ / (1.0 - ab)
            _, ll = _garch_filter_kernel(residuals, omega_, alpha_, beta_, sigma2_init)
            return -float(np.dot(gamma_k, ll))

        x0 = np.array([
            params_k["intercept"],
            *params_k["ar_coeffs"],
            np.log(max(params_k["omega"], 1e-10)),
            params_k["alpha"],
            params_k["beta"],
        ])

        bounds = (
            [(None, None)]           # intercept
            + [(None, None)] * p     # AR coefficients
            + [(None, None)]         # log(omega)
            + [(1e-6, 0.999)]        # alpha
            + [(1e-6, 0.999)]        # beta
        )
        constraints = [{"type": "ineq", "fun": lambda x: 0.999 - x[-2] - x[-1]}]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                neg_wll, x0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-8},
            )

        x = result.x
        alpha_out = float(np.clip(x[2 + p], 1e-6, 0.999))
        beta_out = float(np.clip(x[3 + p], 1e-6, max(0.999 - alpha_out, 1e-6)))
        return {
            "intercept": float(x[0]),
            "ar_coeffs": x[1: 1 + p].copy(),
            "omega": float(np.exp(x[1 + p])),
            "alpha": alpha_out,
            "beta": beta_out,
        }

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def _compute_ar_residuals(self, y: np.ndarray, intercept: float, ar_coeffs: np.ndarray) -> np.ndarray:
        p = self.order
        T = len(y)
        residuals = np.empty(T - p)
        for t in range(p, T):
            ar_mean = intercept
            for j in range(p):
                ar_mean += ar_coeffs[j] * y[t - 1 - j]
            residuals[t - p] = y[t] - ar_mean
        return residuals

    def _initialize_params(self, y: np.ndarray, seed: int) -> dict:
        rng = np.random.default_rng(seed + self.random_state)
        K = self.n_regimes
        p = self.order
        sample_var = max(np.var(y[p:]), 1e-8)
        sample_mean = float(np.mean(y[p:]))

        P = np.full((K, K), 0.05 / max(K - 1, 1))
        np.fill_diagonal(P, 0.95)

        ar_coeffs = rng.uniform(-0.2, 0.2, (K, p))
        intercepts = sample_mean + rng.normal(0, max(abs(sample_mean) * 0.05, 1e-6), K)

        alpha = rng.uniform(0.03, 0.10, K)
        beta = rng.uniform(0.80, 0.88, K)
        omega = sample_var * (1.0 - alpha - beta) * rng.uniform(0.8, 1.2, K)
        omega = np.clip(omega, 1e-8, None)

        return {"P": P, "intercepts": intercepts, "ar_coeffs": ar_coeffs,
                "omega": omega, "alpha": alpha, "beta": beta}

    @staticmethod
    def _stationary_dist(P: np.ndarray) -> np.ndarray:
        """Stationary distribution of row-stochastic matrix P."""
        K = P.shape[0]
        A = (P.T - np.eye(K))
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

    # ─── Inference ──────────────────────────────────────────────────────────────

    def predict_regime(self, causal: bool = False) -> np.ndarray:
        """Most likely regime at each time step (length T − order)."""
        self._validate_fitted()
        probs = self.filtered_probabilities_ if causal else self.regime_probabilities_
        return np.argmax(probs, axis=1)

    def predict_regime_proba(self, causal: bool = False) -> np.ndarray:
        """Regime probabilities, shape (T − order, n_regimes)."""
        self._validate_fitted()
        return self.filtered_probabilities_ if causal else self.regime_probabilities_

    # ─── Forecasting ────────────────────────────────────────────────────────────

    def forecast(self, steps: int = 1, regime: int | None = None, return_variance: bool = False, causal: bool = True) -> np.ndarray | dict[str, np.ndarray]:
        """
        Multi-step forecast.

        Mean: recursive AR substitution.
        Variance: Var[y_{T+h}|k] = Σ_{j=0}^{h-1} ψ_j² σ²_{T+h−j}
            where σ²_{T+h} follows the GARCH persistence formula.

        Parameters
        ----------
        steps : int, default=1
        regime : int, optional
            Condition on a specific regime; None uses regime-weighted forecast.
        return_variance : bool, default=False
        causal : bool, default=True

        Returns
        -------
        np.ndarray, shape (steps,)  or  dict with 'mean', 'variance', etc.
        """
        self._validate_fitted()

        if regime is not None:
            return self._forecast_single_regime(regime, steps, return_variance)

        regime_probs = self.forecast_regime_probabilities(steps, causal=causal)  # (steps, K)

        regime_means: dict[int, np.ndarray] = {}
        regime_vars: dict[int, np.ndarray] = {}
        for r in range(self.n_regimes):
            out = self._forecast_single_regime(r, steps, return_variance=True)
            regime_means[r] = out["mean"]
            regime_vars[r] = out["variance"]

        p = regime_probs
        mu = np.stack([regime_means[r] for r in range(self.n_regimes)], axis=1)
        sigma2 = np.stack([regime_vars[r] for r in range(self.n_regimes)], axis=1)

        mean = np.sum(p * mu, axis=1)
        variance = np.sum(p * sigma2, axis=1) + np.sum(p * (mu - mean[:, None]) ** 2, axis=1)

        if not return_variance:
            return mean

        return {
            "mean": mean,
            "variance": variance,
            "regime_probabilities": regime_probs,
            "regime_forecasts": dict(regime_means),
        }

    def _forecast_single_regime(self, regime: int, steps: int, return_variance: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        p_dict = self._params_
        intercept = p_dict["intercepts"][regime]
        ar_coeffs = p_dict["ar_coeffs"][regime]
        omega = p_dict["omega"][regime]
        alpha = p_dict["alpha"][regime]
        beta = p_dict["beta"][regime]

        # AR mean forecast via recursive substitution
        hist = self._history_buffer_.copy()
        mean = np.empty(steps)
        for h in range(steps):
            ar_mean = intercept + sum(ar_coeffs[j] * hist[-1 - j] for j in range(self.order))
            mean[h] = ar_mean
            hist = np.append(hist[1:], ar_mean)

        if not return_variance:
            return mean

        # GARCH variance forecasts: σ²_{T+h} for h=1..steps
        ab = alpha + beta
        sigma2_unc = omega / max(1.0 - ab, 1e-8)
        # One-step ahead: ω + α ε_T² + β σ²_T
        eps_T = self._last_ar_residuals_[regime]
        sigma2_T = self._last_garch_sigma2_[regime]
        sigma2_T1 = omega + alpha * eps_T ** 2 + beta * sigma2_T

        garch_var = np.empty(steps)
        for h in range(steps):
            garch_var[h] = sigma2_unc + (ab ** h) * (sigma2_T1 - sigma2_unc)

        # MA(∞) impulse-response coefficients ψ_j
        psi = np.zeros(steps)
        psi[0] = 1.0
        for j in range(1, steps):
            for k in range(1, min(j, self.order) + 1):
                psi[j] += ar_coeffs[k - 1] * psi[j - k]

        # Var[y_{T+h}|k] = Σ_{j=0}^{h-1} ψ_j² σ²_{T+h-j}
        # garch_var[i] = σ²_{T+i+1}, so σ²_{T+h-j} = garch_var[h-j-1]
        variance = np.empty(steps)
        for h in range(steps):
            v = 0.0
            for j in range(h + 1):
                v += psi[j] ** 2 * garch_var[h - j]
            variance[h] = v

        return {"mean": mean, "variance": variance}

    # ─── Diagnostics ────────────────────────────────────────────────────────────

    def get_regime_parameters(self) -> pd.DataFrame:
        """
        Return per-regime parameters as a tidy DataFrame.

        Columns: regime, intercept, ar_L1..p, omega, alpha, beta,
                 persistence (α+β), unconditional_variance (ω/(1−α−β)).
        """
        self._validate_fitted()
        rows = []
        for k in range(self.n_regimes):
            ar = self._params_["ar_coeffs"][k]
            ab = self._params_["alpha"][k] + self._params_["beta"][k]
            unc_var = self._params_["omega"][k] / max(1.0 - ab, 1e-8)
            rows.append({
                "regime": k,
                "intercept": self._params_["intercepts"][k],
                **{f"ar_L{i + 1}": ar[i] for i in range(self.order)},
                "omega": self._params_["omega"][k],
                "alpha": self._params_["alpha"][k],
                "beta": self._params_["beta"][k],
                "garch_persistence": ab,
                "unconditional_variance": unc_var,
            })
        return pd.DataFrame(rows)

    def interpret_regimes(self) -> dict[int, str]:
        """Label regimes by GARCH persistence and unconditional volatility."""
        self._validate_fitted()
        params = self.get_regime_parameters()
        median_vol = params["unconditional_variance"].median()
        interpretations = {}
        for _, row in params.iterrows():
            r = int(row["regime"])
            ab = row["garch_persistence"]
            speed = "High-persist" if ab >= 0.95 else "Low-persist"
            vol = "high_vol" if row["unconditional_variance"] >= median_vol else "low_vol"
            interpretations[r] = f"{speed} (α+β={ab:.3f}, {vol})"
        return interpretations