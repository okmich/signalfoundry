"""
Base class for Markov-switching time series models.
"""

from __future__ import annotations

import numpy as np
import joblib

from .kernels import _predict_regime_probs, _propagate_regime_probs, _scaled_update


class BaseMarkovSwitching:
    """
    Abstract base class for Markov-switching time series models.

    Provides shared infrastructure: fitted-state tracking, transition-matrix access, regime-probability
    propagation, ``predict_regime()`` / ``predict_regime_proba()`` (default ``causal=True``: filtered, no
    look-ahead), the streaming belief state (``current_regime_proba()`` / ``current_regime()``), AIC/BIC retrieval
    and serialization.  Concrete subclasses must implement ``fit()``, ``update()`` and ``forecast()``.

    Parameters
    ----------
    n_regimes : int, default=2
        Number of discrete regimes (must be >= 2).
    random_state : int, default=42
        Seed for any random initialization.

    Attributes
    ----------
    transition_matrix_ : np.ndarray, shape (n_regimes, n_regimes)
        Row-stochastic: P[i,j] = P(s_t=j | s_{t-1}=i). Set by subclass fit().
    regime_probabilities_ : np.ndarray, shape (n_samples, n_regimes)
        Smoothed: P(s_t | y_{1:T}) — look-ahead. Set by subclass fit().
    filtered_probabilities_ : np.ndarray, shape (n_samples, n_regimes)
        Filtered: P(s_t | y_{1:t}) — causal. Set by subclass fit().
    data_ : np.ndarray
        Training series stored by subclass fit().
    aic, bic : float
        Information criteria. Set by subclass fit().
    is_fitted : bool
    """

    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")

        self.n_regimes = n_regimes
        self.random_state = random_state

        # Populated by subclass fit()
        self.transition_matrix_: np.ndarray | None = None
        self.regime_probabilities_: np.ndarray | None = None    # smoothed — look-ahead
        self.filtered_probabilities_: np.ndarray | None = None  # filtered — causal
        self.data_: np.ndarray | None = None
        self.aic: float | None = None
        self.bic: float | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _validate_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet")

    # ─── Shared inference ───────────────────────────────────────────────────────

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return regime transition matrix.

        Returns
        -------
        transition_matrix : np.ndarray, shape (n_regimes, n_regimes)
            P[i, j] = P(regime_t = j | regime_{t-1} = i)
        """
        self._validate_fitted()
        return self.transition_matrix_

    def get_aic_bic(self) -> tuple[float, float]:
        """Return (AIC, BIC)."""
        self._validate_fitted()
        return self.aic, self.bic

    def predict_regime(self, causal: bool = True) -> np.ndarray:
        """
        Most likely regime at each fitted time step.

        Parameters
        ----------
        causal : bool, default=True
            True  → filtered P(s_t | y_{1:t}), no look-ahead — use for ML labels / live trading.
            False → smoothed P(s_t | y_{1:T}), look-ahead  — use for diagnostics only.

        Returns
        -------
        regimes : np.ndarray, shape (n_fitted,)
        """
        return np.argmax(self.predict_regime_proba(causal=causal), axis=1)

    def predict_regime_proba(self, causal: bool = True) -> np.ndarray:
        """
        Regime probabilities over the fitted series.

        Parameters
        ----------
        causal : bool, default=True
            True  → filtered probabilities (causal). False → smoothed probabilities (look-ahead).

        Returns
        -------
        probabilities : np.ndarray, shape (n_fitted, n_regimes)

        Notes
        -----
        Filtered is causal but still in-sample: the parameters were fitted on these bars. For out-of-sample labels,
        fit on a training window and ``update()`` through the test window.
        """
        self._validate_fitted()
        return self.filtered_probabilities_ if causal else self.regime_probabilities_

    def forecast_regime_probabilities(self, steps: int = 10, causal: bool = True) -> np.ndarray:
        """
        Propagate regime probabilities forward through the transition matrix.

        Parameters
        ----------
        steps : int
        causal : bool, default=True
            True  → start from filtered probability at T (causal).
            False → start from smoothed probability at T. At the last observed bar the two coincide, so this only
            matters for custom subclasses whose sequences differ there.

        Returns
        -------
        regime_probs : np.ndarray, shape (steps, n_regimes)
            P(s_{T+h} = k | y_{1:T}) for h = 1, …, steps
        """
        self._validate_fitted()
        if getattr(self, "_n_updates_", 0) > 0:
            # Streaming updates have advanced the belief state past the training data; at the newest bar the
            # filtered and smoothed beliefs are the same object, so both modes start here.
            initial = np.asarray(self._last_forward_alpha_, dtype=np.float64)
        else:
            src = self.filtered_probabilities_ if causal else self.regime_probabilities_
            initial = np.asarray(src[-1], dtype=np.float64)
        P = np.ascontiguousarray(self.transition_matrix_, dtype=np.float64)
        return _propagate_regime_probs(P, initial, steps)

    # ─── Streaming belief state ─────────────────────────────────────────────────

    def current_regime_proba(self) -> np.ndarray:
        """
        Return filtered regime probabilities at the most recent bar.

        Returns the training-time filtered probability if no ``update()``
        calls have been made, or the streaming belief state otherwise.

        Returns
        -------
        proba : np.ndarray, shape (n_regimes,)
        """
        self._validate_fitted()
        return self._last_forward_alpha_.copy()

    def current_regime(self) -> int:
        """
        Return the most likely regime at the most recent bar.

        Returns
        -------
        regime : int
        """
        self._validate_fitted()
        return int(np.argmax(self._last_forward_alpha_))

    @staticmethod
    def _require_finite(observations: np.ndarray, caller: str) -> None:
        """Reject the whole batch before any state changes: one NaN/inf bar would make the belief state NaN for good."""
        if not np.all(np.isfinite(observations)):
            raise ValueError(f"{caller}: observations must be finite (NaN/inf would corrupt the belief state)")

    @staticmethod
    def _forward_update(alpha_prev: np.ndarray, transition_matrix: np.ndarray, log_liks: np.ndarray) -> np.ndarray:
        """One forward step α_t ∝ (α_{t-1} P) · exp(log p(y_t | s_t)).

        Uses the same kernels as ``_scaled_forward_filter`` in the E-steps, so streaming matches batch bit for bit.
        """
        prior = _predict_regime_probs(np.ascontiguousarray(alpha_prev, dtype=np.float64),
                                      np.ascontiguousarray(transition_matrix, dtype=np.float64))
        return _scaled_update(prior, np.ascontiguousarray(log_liks, dtype=np.float64))[0]

    # ─── Persistence ────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save model to disk using joblib."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> BaseMarkovSwitching:
        """Load a previously saved model."""
        return joblib.load(filepath)
