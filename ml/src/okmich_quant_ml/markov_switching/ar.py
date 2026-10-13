"""
Markov-Switching Autoregression (MS-AR) for regime-dependent forecasting.

This module provides a wrapper around statsmodels' MarkovAutoregression that offers:
- Direct return forecasts (not just regime labels)
- Uncertainty quantification (forecast variance)
- Regime-weighted predictions
- Different AR models per regime

Model (Hamilton 1989 mean form, as statsmodels fits it)
-------------------------------------------------------
    y_t − μ_{s_t} = Σ_{l=1}^{p} φ_{s_t,l} (y_{t−l} − μ_{s_{t−l}}) + ε_t,   ε_t ~ N(0, σ²_{s_t})

``const[k]`` in statsmodels is the regime MEAN μ_k, not an intercept. Each lag is centred on the regime that was
active at that lag, so the conditional mean depends on the last p+1 regimes and the exact filter, streaming
``update()`` and forecasts all run on the joint path (s_t, …, s_{t−p}).

statsmodels 0.14.5 (the pinned version) keys the switching innovation variance on s_{t−p+1} instead of s_t whenever order ≥ 2 (a
broadcasting slip in ``_conditional_loglikelihoods``). ``_HamiltonMarkovAutoregression`` corrects it, so σ² always
follows the current regime.

Causality notes
---------------
The EM algorithm uses ALL data to fit parameters. After fitting, two regime probability sequences are available:

- ``filtered_marginal_probabilities`` — P(s_t | y_{1:t}), CAUSAL.
  Uses only data up to time t. Equivalent to HMM FILTERING mode.
  Use this for live trading, rolling-window backtests, and ML training targets.

- ``smoothed_marginal_probabilities`` — P(s_t | y_{1:T}), LOOK-AHEAD.
  Uses the full data series including future observations. Equivalent to HMM SMOOTHING mode. Useful for diagnostics and
  visualization, but introduces look-ahead bias if used as training labels.

``predict_regime()`` / ``predict_regime_proba()`` default to the causal (filtered) sequence.
The ``forecast()`` method is always causal: it projects forward from the last observed bar using only past data and the transition matrix.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from .base import BaseMarkovSwitching
from .kernels import _ar_variance_kernel, _companion, _ms_ar_deviation_moments, _ms_ar_forecast_moments, \
    _ms_ar_joint_forward_step


class _HamiltonMarkovAutoregression(MarkovAutoregression):
    """statsmodels ``MarkovAutoregression`` with the switching innovation variance keyed on the current regime.

    Upstream reshapes the variance to ``(k_regimes, 1, 1)``, which broadcasts against the trailing three axes of the
    ``(k_regimes,) * (order + 1) + (nobs,)`` residual array: correct for order 1, but for order ≥ 2 σ² follows
    s_{t−order+1}. Its EM variance step already weights by P(s_t), so this also makes EM and the likelihood agree.
    """

    def _conditional_loglikelihoods(self, params):
        resid = self._resid(params)
        variance = params[self.parameters["variance"]].squeeze()
        if self.switching_variance:
            variance = np.reshape(variance, (self.k_regimes,) + (1,) * (self.order + 1))
        return -0.5 * resid ** 2 / variance - 0.5 * np.log(2 * np.pi * variance)


class MarkovSwitchingAR(BaseMarkovSwitching):
    """
    Markov-Switching Autoregression for regime-dependent forecasting.

    Unlike standard HMM (which provides regime labels), MS-AR provides:
    - Direct return forecasts
    - Uncertainty quantification
    - Regime-weighted predictions
    - Different AR models per regime

    Mathematical Foundation
    -----------------------
    Regime: s_t ~ Markov chain with transition matrix P
    Observations: mean-form AR(p) with regime-dependent mean, coefficients and variance

        y_t − μ^{(s_t)} = φ₁^{(s_t)} (y_{t-1} − μ^{(s_{t-1})}) + ... + φ_p^{(s_t)} (y_{t-p} − μ^{(s_{t-p})}) + ε_t
        ε_t ~ N(0, σ²^{(s_t)})

    μ^{(s)} is the regime mean. Within a persistent regime the equivalent intercept is μ^{(s)} (1 − Σ φ^{(s)}).

    Parameters
    ----------
    n_regimes : int, default=2
        Number of discrete regimes.
        - 2: Bull/Bear, Trending/Reverting
        - 3: Bull/Neutral/Bear, Trending/Ranging/Reverting
        - 4+: Usually overfitting (not recommended)
    order : int, default=2
        AR order (number of lags).
        - 1: AR(1) — simple momentum
        - 2: AR(2) — momentum + acceleration (recommended)
        - 5+: Usually overfitting
    switching_variance : bool, default=True
        If True, variance switches across regimes (recommended).

    Attributes
    ----------
    fitted_model_ : statsmodels.MarkovAutoregression result
    regime_probabilities_ : np.ndarray, shape (n_samples, n_regimes)
        Smoothed: P(s_t | y_{1:T}) — look-ahead bias.
    filtered_probabilities_ : np.ndarray, shape (n_samples, n_regimes)
        Filtered: P(s_t | y_{1:t}) — causal, safe for live trading.
    transition_matrix_ : np.ndarray, shape (n_regimes, n_regimes)
    is_fitted : bool
    aic, bic : float

    Examples
    --------
    >>> ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    >>> ms_ar.fit(returns)
    >>> forecast = ms_ar.forecast(steps=10, return_variance=True)
    >>> print(forecast['mean'])
    >>> interp = ms_ar.interpret_regimes()

    References
    ----------
    Hamilton, J. D. (1989). Econometrica, 57(2), 357-384.

    Notes
    -----
    Requires statsmodels>=0.13.0. EM may not converge for T < 200.
    """

    def __init__(self, n_regimes: int = 2, order: int = 2, switching_variance: bool = True, random_state: int = 42):
        super().__init__(n_regimes=n_regimes, random_state=random_state)

        if order < 1:
            raise ValueError("order must be at least 1")

        self.order = order
        self.switching_variance = switching_variance

        self.fitted_model_ = None
        self._exog_used = False

    # ─── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, y: np.ndarray, exog: np.ndarray | None = None, num_restarts: int = 3) -> MarkovSwitchingAR:
        """
        Fit Markov-Switching AR using EM algorithm.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Time series to model (e.g., log-returns)
        exog : np.ndarray, shape (n_samples, n_exog), optional
            Exogenous variables
        num_restarts : int, default=3
            Number of random start-parameter searches (statsmodels ``search_reps``). The search draws from a
            generator seeded with ``random_state``, so fits are reproducible; the global NumPy RNG is left as found.

        Returns
        -------
        self
        """
        y = np.asarray(y).flatten()
        if len(y) < self.order + 10:
            raise ValueError(
                f"Time series too short. Need at least {self.order + 10} samples, "
                f"got {len(y)}"
            )

        self.data_ = y
        self._exog_used = exog is not None

        # statsmodels draws its start-parameter search from the global RNG — seed it locally, then restore.
        global_rng_state = np.random.get_state()
        np.random.seed(self.random_state)
        try:
            sm_model = _HamiltonMarkovAutoregression(endog=y, k_regimes=self.n_regimes, order=self.order,
                                                     switching_ar=True, switching_variance=self.switching_variance,
                                                     exog=exog)
            best_model = sm_model.fit(search_reps=max(1, num_restarts), search_iter=20, maxiter=500, disp=False)
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {e}. Try reducing n_regimes or order.") from e
        finally:
            np.random.set_state(global_rng_state)

        self.fitted_model_ = best_model
        self.regime_probabilities_ = best_model.smoothed_marginal_probabilities
        self.filtered_probabilities_ = best_model.filtered_marginal_probabilities

        # statsmodels: regime_transition[j, i] = P(s_t=j | s_{t-1}=i) — columns sum to 1.
        # Transpose so rows sum to 1: [i, j] = P(to j | from i).
        self.transition_matrix_ = np.squeeze(best_model.regime_transition).T

        self.aic = best_model.aic
        self.bic = best_model.bic
        self._is_fitted = True

        # Streaming state (Online Learning L1): joint P(s_T, …, s_{T−p} | y_{1:T}), flattened (s_T most significant)
        self._last_forward_joint_ = np.ascontiguousarray(
            best_model.filtered_joint_probabilities[..., -1].reshape(-1), dtype=np.float64)
        self._last_forward_alpha_ = np.asarray(self.filtered_probabilities_[-1], dtype=np.float64)
        self._history_buffer_ = np.asarray(self.data_[-self.order:], dtype=np.float64)
        self._n_updates_ = 0
        self._cached_ar_coeffs_, self._cached_means_, self._cached_sigma2s_ = (
            self._extract_all_regime_params_arrays()
        )

        self._validate_convergence()
        return self

    def _validate_convergence(self):
        if not self.fitted_model_.mle_retvals["converged"]:
            warnings.warn(
                "EM did not converge. Results may be unreliable. "
                "Try increasing maxiter or using different initialization."
            )
        mean_probs = self.regime_probabilities_.mean(axis=0)
        for regime in range(self.n_regimes):
            if mean_probs[regime] < 0.05:
                warnings.warn(
                    f"Regime {regime} rarely occurs (<5% of time). "
                    "Consider reducing n_regimes."
                )

    # ─── Inference ──────────────────────────────────────────────────────────────

    def predict_regime(self, causal: bool = True) -> np.ndarray:
        """
        Return most likely regime at each time step.

        Parameters
        ----------
        causal : bool, default=True
            True  → filtered P(s_t | y_{1:t}), no look-ahead — use for ML labels / live trading.
            False → smoothed P(s_t | y_{1:T}), look-ahead  — use for diagnostics only.

        Returns
        -------
        regimes : np.ndarray, shape (n_samples,)
        """
        self._validate_fitted()
        probs = self.filtered_probabilities_ if causal else self.regime_probabilities_
        return np.argmax(probs, axis=1)

    def predict_regime_proba(self, causal: bool = True) -> np.ndarray:
        """
        Return regime probabilities for the training series.

        Parameters
        ----------
        causal : bool, default=True
            True  → filtered probabilities (causal).
            False → smoothed probabilities (look-ahead).

        Returns
        -------
        probabilities : np.ndarray, shape (n_samples, n_regimes)
        """
        self._validate_fitted()
        return self.filtered_probabilities_ if causal else self.regime_probabilities_

    # ─── Forecasting ────────────────────────────────────────────────────────────

    def forecast(self, steps: int = 1, regime: int | None = None, return_variance: bool = False,
                 causal: bool = True) -> np.ndarray | dict[str, np.ndarray]:
        """
        Generate multi-step forecasts accounting for regime uncertainty.

        Parameters
        ----------
        steps : int, default=1
            Forecast horizon
        regime : int, optional
            Condition on the path staying in ``regime`` for all steps. If None, the exact regime-mixed forecast.
        return_variance : bool, default=False
            If True, return a dict with mean, variance, and regime details.
        causal : bool, default=True
            Kept for API symmetry. The forecast origin is always the latest filtered joint regime state, which at the
            last observed bar equals the smoothed one, so this flag does not change the result.

        Returns
        -------
        np.ndarray, shape (steps,)  — if return_variance=False
        dict with keys:
            'mean'               : np.ndarray, shape (steps,)
            'variance'           : np.ndarray, shape (steps,)
            'regime_forecasts'   : dict[int, np.ndarray]  — E[y_{T+h} | s_{T+h} = r, y_{1:T}]
            'regime_probabilities': np.ndarray, shape (steps, n_regimes)

        Notes
        -----
        Mean and variance are the exact predictive moments of the mean-form MS-AR(p) (see
        ``_ms_ar_forecast_moments``), not a mixture of stay-in-regime forecasts. ``regime_forecasts`` are conditional
        on the regime at the forecast horizon, so ``mean == Σ_r regime_probabilities[:, r] · regime_forecasts[r]``.
        """
        self._validate_fitted()
        self._require_no_exog("forecast()")

        if regime is not None:
            return self._forecast_single_regime(regime, steps, return_variance)

        ar_coeffs, means, sigma2s = self._cached_ar_coeffs_, self._cached_means_, self._cached_sigma2s_
        weight0, first0, second0 = _ms_ar_deviation_moments(means, self._last_forward_joint_, self._history_buffer_)
        P = np.ascontiguousarray(self.transition_matrix_, dtype=np.float64)
        probs, first, second = _ms_ar_forecast_moments(ar_coeffs, sigma2s, P, weight0, first0, second0, steps)

        # Moments conditional on the terminal regime s_{T+h} = r, then law of total expectation / variance
        safe = np.where(probs > 0.0, probs, 1.0)
        cond_dev = first / safe                                            # E[z | s = r]
        cond_mean = means[None, :] + cond_dev                              # E[y | s = r]
        cond_var = np.maximum(second / safe - cond_dev ** 2, 0.0)          # Var[y | s = r]
        mean = np.sum(probs * cond_mean, axis=1)
        if not return_variance:
            return mean

        variance = np.sum(probs * cond_var, axis=1) + np.sum(probs * (cond_mean - mean[:, None]) ** 2, axis=1)
        return {
            "mean": mean,
            "variance": variance,
            "regime_probabilities": probs,
            "regime_forecasts": {r: cond_mean[:, r] for r in range(self.n_regimes)},
        }

    def _forecast_single_regime(self, regime: int, steps: int,
                                return_variance: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        """Forecast conditional on s_{T+1} = … = s_{T+h} = ``regime``.

        The uncertain past path is re-weighted by P(s_T → regime). With Z_T the deviation state and F the regime's
        companion matrix:
            E[y_{T+h}]   = μ + [F^h E(Z_T)]₀
            Var[y_{T+h}] = [F^h Cov(Z_T) F^h']₀₀ + σ² Σ_{j=0}^{h-1} ψ_j²
        """
        if not 0 <= regime < self.n_regimes:
            raise ValueError(f"regime must be in [0, {self.n_regimes}), got {regime}")
        ar_coeffs, means, sigma2s = self._cached_ar_coeffs_, self._cached_means_, self._cached_sigma2s_
        joint = self._last_forward_joint_
        s_now = np.arange(len(joint)) // (self.n_regimes ** self.order)
        conditioned = joint * np.asarray(self.transition_matrix_)[s_now, regime]
        conditioned = conditioned / conditioned.sum()
        _, first, second = _ms_ar_deviation_moments(means, conditioned, self._history_buffer_)
        ez = first.sum(axis=0)
        cov_z = second.sum(axis=0) - np.outer(ez, ez)

        F = _companion(ar_coeffs[regime])
        Fh = np.eye(self.order)
        mean = np.empty(steps)
        path_var = np.empty(steps)
        for h in range(steps):
            Fh = F @ Fh
            mean[h] = means[regime] + (Fh @ ez)[0]
            path_var[h] = max((Fh @ cov_z @ Fh.T)[0, 0], 0.0)
        if not return_variance:
            return mean
        return {"mean": mean, "variance": _ar_variance_kernel(ar_coeffs[regime], sigma2s[regime], steps) + path_var}

    def _require_no_exog(self, caller: str) -> None:
        if self._exog_used:
            raise NotImplementedError(
                f"{caller} does not support exogenous variables: the model was fitted with exog, and the "
                "mean-form conditional mean needs the exog values of every new bar."
            )

    # ─── Online Learning L1: streaming filter ───────────────────────────────────

    def update(self, y_new: float | np.ndarray) -> MarkovSwitchingAR:
        """
        Process one or more new observations without refitting parameters.

        Runs the causal Hamilton filter for each new bar on the joint regime path (s_t, …, s_{t−p}), exactly as
        statsmodels filters the training sample, so streaming a series bar by bar reproduces the batch filtered
        probabilities. Parameters (means, AR coefficients, variances, transition matrix) remain fixed at their
        fit-time values.

        This is O(K^(p+2)) per observation — suitable for tick/bar-level streaming.

        Parameters
        ----------
        y_new : float or array-like
            New observation(s) in arrival order.  A scalar updates by one
            step; a 1-D array updates sequentially.

        Returns
        -------
        self

        Notes
        -----
        After calling ``update()``:
        - ``current_regime_proba()`` returns the updated filtered probability.
        - ``current_regime()`` returns the most likely current regime.
        - ``forecast()`` and ``forecast_regime_probabilities(causal=True)``
          project forward from the updated belief state automatically.
        """
        self._validate_fitted()
        self._require_no_exog("update()")
        observations = np.atleast_1d(np.asarray(y_new, dtype=np.float64)).flatten()
        P = np.ascontiguousarray(self.transition_matrix_, dtype=np.float64)

        for y in observations:
            self._last_forward_joint_ = _ms_ar_joint_forward_step(self._cached_ar_coeffs_, self._cached_means_,
                                                                  self._cached_sigma2s_, P, self._last_forward_joint_,
                                                                  self._history_buffer_, float(y))
            self._last_forward_alpha_ = self._last_forward_joint_.reshape(self.n_regimes, -1).sum(axis=1)
            # Roll history buffer: drop oldest, append new observation
            self._history_buffer_ = np.roll(self._history_buffer_, -1)
            self._history_buffer_[-1] = y
            self._n_updates_ += 1

        return self

    # ─── Online Learning L2: sliding-window refit ───────────────────────────────

    def refit_window(self, y_new: float | np.ndarray, window: int, refit_every: int = 1, num_restarts: int = 3) -> MarkovSwitchingAR:
        """
        Process new observations with periodic sliding-window parameter updates.

        Maintains a rolling buffer of the last ``window`` observations. Every
        ``refit_every`` new bars the full EM is re-run on the buffer, refreshing
        all parameters and the belief state. Between refits, a causal L1
        forward step (``update()``) keeps the belief state current using the
        most recently fitted parameters.

        Parameters
        ----------
        y_new : float or array-like
            New observation(s) in arrival order.
        window : int
            Number of bars in the rolling fit window. Must be > ``order + 10``.
        refit_every : int, default=1
            Refit the model after every this many new observations.
            ``refit_every=1`` refits on every bar (most adaptive, slowest).
            ``refit_every=N`` amortises the EM cost over N bars.
        num_restarts : int, default=3
            Passed to ``fit()`` on each refit.

        Returns
        -------
        self

        Notes
        -----
        After each refit, regime labels may permute (label switching). Downstream
        code that caches regime indices should use ``current_regime_proba()``
        rather than hard-coded regime numbers.

        ``n_refits_`` tracks how many refits have occurred since the first
        ``refit_window()`` call.
        """
        self._validate_fitted()

        from collections import deque

        # Initialise (or re-initialise if window size changed)
        if not hasattr(self, "_window_buffer_") or self._window_buffer_.maxlen != window:
            seed = self.data_[-window:] if len(self.data_) >= window else self.data_.copy()
            self._window_buffer_ = deque(seed, maxlen=window)
            self._bars_since_refit_ = 0
            self._n_refits_ = 0

        observations = np.atleast_1d(np.asarray(y_new, dtype=np.float64)).flatten()

        for y in observations:
            self._window_buffer_.append(float(y))
            self._bars_since_refit_ += 1

            if (self._bars_since_refit_ >= refit_every
                    and len(self._window_buffer_) >= self.order + 10):
                window_data = np.array(self._window_buffer_, dtype=np.float64)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.fit(window_data, num_restarts=num_restarts)
                    self._bars_since_refit_ = 0
                    self._n_refits_ += 1
                except RuntimeError:
                    # Refit failed — fall back to L1 update with current params
                    self.update(float(y))
            else:
                self.update(float(y))

        return self

    def _extract_all_regime_params_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cache regime params as contiguous arrays for the Numba filter and forecast kernels."""
        ar_coeffs = np.zeros((self.n_regimes, self.order), dtype=np.float64)
        means = np.zeros(self.n_regimes, dtype=np.float64)
        sigma2s = np.zeros(self.n_regimes, dtype=np.float64)
        for r in range(self.n_regimes):
            c, mu, s2 = self._get_regime_params(r)
            ar_coeffs[r] = c
            means[r] = mu
            sigma2s[r] = s2
        return ar_coeffs, means, sigma2s

    # ─── Parameter extraction ───────────────────────────────────────────────────

    def _get_param_dict(self) -> dict[str, float]:
        return dict(zip(self.fitted_model_.data.param_names, self.fitted_model_.params))

    def _get_regime_params(self, regime: int) -> tuple[np.ndarray, float, float]:
        """Return (ar_coeffs, mean, sigma2) for `regime` in a single pass.

        statsmodels' ``const[k]`` is the regime MEAN μ_k of the mean-form model, not an intercept.
        """
        params = self._get_param_dict()

        ar_coeffs = np.array([params.get(f"ar.L{lag}[{regime}]", 0.0) for lag in range(1, self.order + 1)])
        mean = 0.0
        for name in (f"const[{regime}]", f"intercept[{regime}]", f"mu[{regime}]"):
            if name in params:
                mean = params[name]
                break

        if self.switching_variance:
            sigma2 = params.get(f"sigma2[{regime}]", params.get("sigma2", 1.0))
        else:
            sigma2 = params.get("sigma2", 1.0)

        return ar_coeffs, float(mean), float(sigma2)

    # ─── Diagnostics ────────────────────────────────────────────────────────────

    def get_regime_parameters(self, regime: int | None = None) -> pd.DataFrame:
        """
        Regime-specific parameters in a tidy DataFrame.

        Parameters
        ----------
        regime : int, optional
            Specific regime. If None, returns all regimes.

        Returns
        -------
        pd.DataFrame with columns: regime, mean, intercept, ar_L1, …, ar_Lp,
            variance, persistence, half_life

            ``mean`` is the fitted regime mean μ; ``intercept`` is the equivalent AR intercept μ (1 − Σφ) that applies
            while the process stays in the regime.
        """
        self._validate_fitted()

        regimes = [regime] if regime is not None else range(self.n_regimes)
        rows = []
        for r in regimes:
            ar_coeffs, mean, variance = self._get_regime_params(r)
            persistence = float(np.sum(ar_coeffs))
            if 0 < persistence < 1:
                half_life = np.log(0.5) / np.log(persistence)
            elif persistence >= 1:
                half_life = np.inf
            else:
                half_life = -np.inf

            rows.append({
                "regime": r,
                "mean": mean,
                "intercept": mean * (1.0 - persistence),
                **{f"ar_L{i + 1}": ar_coeffs[i] for i in range(self.order)},
                "variance": variance,
                "persistence": persistence,
                "half_life": half_life,
            })
        return pd.DataFrame(rows)

    def interpret_regimes(self) -> dict[int, str]:
        """
        Automatic regime interpretation based on persistence and variance.

        Returns
        -------
        dict[int, str]  e.g. {0: 'Trending (low_volatility)', 1: 'Mean-Reverting (high_volatility)'}
        """
        self._validate_fitted()

        params = self.get_regime_parameters()
        median_var = params["variance"].median()
        interpretations = {}
        for _, row in params.iterrows():
            p = row["persistence"]
            if p > 0.5:
                label = "Trending"
            elif p > 0:
                label = "Weak Trend"
            elif p > -0.3:
                label = "Mean-Reverting"
            else:
                label = "Strong Mean-Reversion"

            vol = "low_volatility" if row["variance"] < median_var else "high_volatility"
            interpretations[int(row["regime"])] = f"{label} ({vol})"
        return interpretations

    # ─── Model selection ────────────────────────────────────────────────────────

    def train(self, y: np.ndarray, order_range: tuple[int, int] | None = None,
              n_regimes_range: tuple[int, int] | None = None, criterion: str = "bic",
              exog: np.ndarray | None = None) -> MarkovSwitchingAR:
        """
        Fit with automatic hyperparameter selection (grid search over order × n_regimes).

        Parameters
        ----------
        y : np.ndarray
        order_range : (min, max), default=(1, 5)
        n_regimes_range : (min, max), default=(2, 3)
        criterion : 'aic' or 'bic', default='bic'
        exog : np.ndarray, optional

        Returns
        -------
        self — configured and fitted with the optimal hyperparameters
        """
        order_range = order_range or (1, 5)
        n_regimes_range = n_regimes_range or (2, 3)

        best_model = None
        best_score = np.inf

        for n_reg in range(n_regimes_range[0], n_regimes_range[1] + 1):
            for ord_ in range(order_range[0], order_range[1] + 1):
                try:
                    model = MarkovSwitchingAR(
                        n_regimes=n_reg,
                        order=ord_,
                        switching_variance=self.switching_variance,
                        random_state=self.random_state,
                    )
                    model.fit(y, exog=exog)
                    score = model.bic if criterion == "bic" else model.aic
                    if score < best_score:
                        best_score = score
                        best_model = model
                except Exception as e:
                    warnings.warn(f"Failed n_regimes={n_reg}, order={ord_}: {e}")

        if best_model is None:
            raise RuntimeError("All model configurations failed")

        # Adopt the winner's full fitted state, including the streaming filter state and cached params that
        # update() and forecast() read. ``order`` goes through its setter (MarkovSwitchingOU validates it).
        self.__dict__.update({k: v for k, v in vars(best_model).items() if k != "order"})
        self.order = best_model.order
        return self

    # ─── Plotting ───────────────────────────────────────────────────────────────

    def plot_regimes(self, y: np.ndarray | None = None, figsize: tuple[int, int] = (14, 8)) -> Figure:
        """Two-panel plot: time series with regime background + regime probability curves."""
        self._validate_fitted()

        if y is None:
            y = self.data_

        probs = self.regime_probabilities_
        y_plot = y[-len(probs):] if len(y) > len(probs) else y

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        interpretations = self.interpret_regimes()
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_regimes))

        ax1.plot(y_plot, "k-", linewidth=0.8, alpha=0.7, label="Returns")
        regimes = np.argmax(probs, axis=1)
        for regime in range(self.n_regimes):
            indices = np.where(regimes == regime)[0]
            if len(indices):
                for idx in indices:
                    ax1.axvspan(
                        idx - 0.5, idx + 0.5,
                        alpha=0.3, color=colors[regime],
                        label=interpretations[regime] if idx == indices[0] else "",
                    )
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax1.set_ylabel("Returns", fontsize=10)
        ax1.set_title("Time Series with Regime Classification", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(
            dict(zip(labels, handles)).values(),
            dict(zip(labels, handles)).keys(),
            loc="upper left", fontsize=9,
        )

        for regime in range(self.n_regimes):
            ax2.plot(probs[:, regime], label=f"Regime {regime}: {interpretations[regime]}",
                     color=colors[regime], linewidth=1.5)
            ax2.fill_between(range(len(probs)), 0, probs[:, regime], alpha=0.3, color=colors[regime])
        ax2.set_xlabel("Time", fontsize=10)
        ax2.set_ylabel("Probability", fontsize=10)
        ax2.set_title("Regime Probabilities Over Time", fontsize=12, fontweight="bold")
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left", fontsize=9)

        plt.tight_layout()
        return fig

    def plot_forecast(self, steps: int = 20, confidence_intervals: bool = True,
                      figsize: tuple[int, int] = (12, 6)) -> Figure:
        """Forecast plot with 68% and 95% confidence bands."""
        self._validate_fitted()

        result = self.forecast(steps=steps, return_variance=True)
        mean = result["mean"]
        std = np.sqrt(result["variance"])

        n_hist = min(100, len(self.data_))
        hist_idx = np.arange(-n_hist, 0)
        fc_idx = np.arange(0, steps)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(hist_idx, self.data_[-n_hist:], "k-", linewidth=1, label="Historical", alpha=0.7)
        ax.plot(fc_idx, mean, "b-", linewidth=2, label="Forecast")

        if confidence_intervals:
            ax.fill_between(fc_idx, mean - 2 * std, mean + 2 * std,
                            alpha=0.2, color="blue", label="95% CI")
            ax.fill_between(fc_idx, mean - std, mean + std,
                            alpha=0.3, color="blue", label="68% CI")

        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Time Steps", fontsize=10)
        ax.set_ylabel("Returns", fontsize=10)
        ax.set_title(f"MS-AR Forecast ({steps} steps ahead)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        return fig

    def plot_regime_parameters(self, figsize: tuple[int, int] = (12, 8)) -> Figure:
        """Bar-chart visualization of AR coefficients, regime mean, variance, and persistence."""
        self._validate_fitted()

        params = self.get_regime_parameters()
        interpretations = self.interpret_regimes()
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_regimes))
        x = np.arange(len(params))
        x_labels = [f"R{r}\n{interpretations[r]}" for r in params["regime"]]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # AR coefficients
        ax = axes[0, 0]
        ar_cols = [c for c in params.columns if c.startswith("ar_L")]
        width = 0.8 / len(ar_cols)
        for i, col in enumerate(ar_cols):
            offset = (i - len(ar_cols) / 2) * width + width / 2
            ax.bar(x + offset, params[col], width, label=col.replace("ar_L", "Lag "), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Coefficient", fontsize=10)
        ax.set_title("AR Coefficients by Regime", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # Regime mean
        ax = axes[0, 1]
        ax.bar(x, params["mean"], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel("Mean (μ)", fontsize=10)
        ax.set_title("Mean by Regime", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Variance
        ax = axes[1, 0]
        ax.bar(x, params["variance"], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("Variance (σ²)", fontsize=10)
        ax.set_title("Variance by Regime", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Persistence
        ax = axes[1, 1]
        ax.bar(x, params["persistence"], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axhline(y=1, color="red", linestyle="--", linewidth=0.5, alpha=0.5, label="Unit root")
        ax.set_ylabel("Persistence (Σ AR coefs)", fontsize=10)
        ax.set_title("Persistence by Regime", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle("Regime-Specific Parameters", fontsize=13, fontweight="bold", y=1.00)
        plt.tight_layout()
        return fig