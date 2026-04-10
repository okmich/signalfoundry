"""
Base class for Markov-switching time series models.
"""

from __future__ import annotations

import numpy as np
import joblib

from .kernels import _propagate_regime_probs


class BaseMarkovSwitching:
    """
    Abstract base class for Markov-switching time series models.

    Provides shared infrastructure: fitted-state tracking, transition-matrix
    access, regime-probability propagation, AIC/BIC retrieval, and
    serialization.  Concrete subclasses must implement ``fit()``,
    ``predict_regime()``, ``predict_regime_proba()``, and ``forecast()``.

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

    def forecast_regime_probabilities(self, steps: int = 10, causal: bool = True) -> np.ndarray:
        """
        Propagate regime probabilities forward through the transition matrix.

        Parameters
        ----------
        steps : int
        causal : bool, default=True
            True  → start from filtered probability at T (causal).
            False → start from smoothed probability (not valid for live use).

        Returns
        -------
        regime_probs : np.ndarray, shape (steps, n_regimes)
            P(s_{T+h} = k | y_{1:T}) for h = 1, …, steps
        """
        self._validate_fitted()
        if causal and getattr(self, "_n_updates_", 0) > 0:
            # Streaming updates have advanced the belief state past training data
            initial = np.asarray(self._last_forward_alpha_, dtype=np.float64)
        else:
            src = self.filtered_probabilities_ if causal else self.regime_probabilities_
            initial = np.asarray(src[-1], dtype=np.float64)
        P = np.asarray(self.transition_matrix_, dtype=np.float64)
        return _propagate_regime_probs(P, initial, steps)

    # ─── Persistence ────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save model to disk using joblib."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> BaseMarkovSwitching:
        """Load a previously saved model."""
        return joblib.load(filepath)
