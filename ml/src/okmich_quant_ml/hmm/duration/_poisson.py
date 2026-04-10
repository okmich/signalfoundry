from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from ._base import BaseDuration


class PoissonDuration(BaseDuration):
    """Shifted Poisson duration model: P(D = u) = Poisson(u - 1 | lambda), u = 1 .. M.

    Single parameter ``lambda`` per state.  The shift ensures minimum
    duration = 1.  The PMF is truncated at ``max_duration`` and
    renormalised.
    """

    def __init__(self, n_states: int, max_duration: int, lambdas: np.ndarray | None = None):
        super().__init__(n_states, max_duration)
        if lambdas is not None:
            self._lambdas = np.asarray(lambdas, dtype=np.float64).copy()
        else:
            # Default: expected duration ≈ max_duration / 4 per state
            self._lambdas = np.full(n_states, max_duration / 4.0, dtype=np.float64)
        self._log_pmf_cache: dict[int, np.ndarray] = {}
        self._rebuild_cache()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @property
    def is_exact_mle(self) -> bool:
        return True

    def log_pmf(self, state: int) -> np.ndarray:
        return self._log_pmf_cache[state]

    def update(self, state: int, eta: np.ndarray) -> None:
        if self._skip_if_unused(state, eta):
            return
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        # Weighted mean of (u - 1) gives lambda_hat
        lambda_hat = np.dot(eta, u - 1) / eta.sum()
        self._lambdas[state] = max(lambda_hat, 0.5)
        self._rebuild_cache_for(state)
        self._check_pmf_normalised(state)
        self._check_survivor_consistency(state)

    def expected_duration(self, state: int) -> float:
        # Expectation under the *truncated* PMF actually used by inference.
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        return float(np.dot(np.exp(self.log_pmf(state)), u))

    def n_parameters(self) -> int:
        return self.n_states  # one lambda per state

    def get_params(self) -> dict:
        return {"lambdas": self._lambdas.copy(), "n_states": self.n_states, "max_duration": self.max_duration}

    def set_params(self, params: dict) -> None:
        self._lambdas = np.asarray(params["lambdas"], dtype=np.float64).copy()
        self._rebuild_cache()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _rebuild_cache(self) -> None:
        self._log_pmf_cache.clear()
        for j in range(self.n_states):
            self._rebuild_cache_for(j)

    def _rebuild_cache_for(self, state: int) -> None:
        lam = self._lambdas[state]
        M = self.max_duration
        u = np.arange(0, M, dtype=np.float64)  # shifted: u-1 = 0 .. M-1
        log_p = poisson.logpmf(u, lam)
        # Truncate and renormalise in log-domain
        log_norm = np.logaddexp.reduce(log_p)
        self._log_pmf_cache[state] = log_p - log_norm
