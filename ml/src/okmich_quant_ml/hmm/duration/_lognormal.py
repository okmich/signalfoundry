from __future__ import annotations

import logging

import numpy as np
from scipy.stats import lognorm

from ._base import BaseDuration

logger = logging.getLogger(__name__)


class LogNormalDuration(BaseDuration):
    """Discretised Log-Normal duration model.

    Parameters per state: ``mu`` and ``sigma`` of the underlying normal
    in log-space.  Discretisation: ``d[u] = CDF(u) - CDF(u-1)`` for
    ``u = 1 .. M``, renormalised.

    The truncation at ``max_duration`` and renormalization does not add a
    free parameter — the underlying continuous distribution has 2 params
    per state and discretisation is deterministic given ``max_duration``.

    M-step uses weighted log-moments (approximate, not exact MLE).
    """

    def __init__(self, n_states: int, max_duration: int, mus: np.ndarray | None = None,
                 sigmas: np.ndarray | None = None):
        super().__init__(n_states, max_duration)
        if mus is not None and sigmas is not None:
            self._mus = np.asarray(mus, dtype=np.float64).copy()
            self._sigmas = np.asarray(sigmas, dtype=np.float64).copy()
        else:
            self._mus = np.full(n_states, np.log(max_duration / 4.0), dtype=np.float64)
            self._sigmas = np.full(n_states, 0.5, dtype=np.float64)
        self._log_pmf_cache: dict[int, np.ndarray] = {}
        self._rebuild_cache()

    @property
    def is_exact_mle(self) -> bool:
        return False

    def log_pmf(self, state: int) -> np.ndarray:
        return self._log_pmf_cache[state]

    def update(self, state: int, eta: np.ndarray) -> None:
        if self._skip_if_unused(state, eta):
            return
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        total = eta.sum()
        log_u = np.log(u)
        mu_log = np.dot(eta, log_u) / total
        var_log = np.dot(eta, log_u ** 2) / total - mu_log ** 2

        if var_log <= 0:
            logger.warning("LogNormal M-step state %d: var_log=%.4f <= 0 — retaining previous params.", state, var_log)
            return

        self._mus[state] = mu_log
        self._sigmas[state] = max(np.sqrt(var_log), 0.01)

        self._rebuild_cache_for(state)
        self._check_pmf_normalised(state)
        self._check_survivor_consistency(state)

    def expected_duration(self, state: int) -> float:
        # Expectation under the *truncated* PMF actually used by inference.
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        return float(np.dot(np.exp(self.log_pmf(state)), u))

    def n_parameters(self) -> int:
        return 2 * self.n_states

    def get_params(self) -> dict:
        return {"mus": self._mus.copy(), "sigmas": self._sigmas.copy(),
                "n_states": self.n_states, "max_duration": self.max_duration}

    def set_params(self, params: dict) -> None:
        self._mus = np.asarray(params["mus"], dtype=np.float64).copy()
        self._sigmas = np.asarray(params["sigmas"], dtype=np.float64).copy()
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._log_pmf_cache.clear()
        for j in range(self.n_states):
            self._rebuild_cache_for(j)

    def _rebuild_cache_for(self, state: int) -> None:
        mu, sigma = self._mus[state], self._sigmas[state]
        M = self.max_duration
        edges = np.arange(0, M + 1, dtype=np.float64)
        # scipy lognorm: shape=sigma, scale=exp(mu)
        cdf_vals = lognorm.cdf(edges, sigma, scale=np.exp(mu))
        pmf = np.diff(cdf_vals)
        pmf = np.maximum(pmf, 1e-300)
        pmf /= pmf.sum()
        self._log_pmf_cache[state] = np.log(pmf)
