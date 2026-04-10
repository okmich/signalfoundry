from __future__ import annotations

import logging

import numpy as np
from scipy.stats import nbinom

from ._base import BaseDuration

logger = logging.getLogger(__name__)


class NegBinDuration(BaseDuration):
    """Negative Binomial duration model: overdispersed, heavy-tailed.

    Parameters per state: ``r`` (number of successes) and ``p`` (success
    probability).  The PMF is ``P(D = u) = NegBin(u - 1 | r, p)`` for
    ``u = 1 .. M`` (shifted so minimum duration = 1), truncated and
    renormalised.

    M-step uses moment matching.  When variance <= mean (underdispersed),
    falls back to a Poisson-equivalent parameterisation and logs a
    warning.
    """

    def __init__(self, n_states: int, max_duration: int, rs: np.ndarray | None = None, ps: np.ndarray | None = None):
        super().__init__(n_states, max_duration)
        if rs is not None and ps is not None:
            self._rs = np.asarray(rs, dtype=np.float64).copy()
            self._ps = np.asarray(ps, dtype=np.float64).copy()
        else:
            self._rs = np.full(n_states, 2.0, dtype=np.float64)
            self._ps = np.full(n_states, 0.5, dtype=np.float64)
        self._log_pmf_cache: dict[int, np.ndarray] = {}
        self._rebuild_cache()

    @property
    def is_exact_mle(self) -> bool:
        return False  # moment matching is approximate

    def log_pmf(self, state: int) -> np.ndarray:
        return self._log_pmf_cache[state]

    def update(self, state: int, eta: np.ndarray) -> None:
        if self._skip_if_unused(state, eta):
            return
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        total = eta.sum()
        m = np.dot(eta, u - 1) / total  # mean of shifted variable
        v = np.dot(eta, (u - 1) ** 2) / total - m ** 2  # variance

        if v <= m or m <= 0:
            # Underdispersed: fall back to Poisson-equivalent
            logger.warning("NegBin M-step state %d: variance (%.4f) <= mean (%.4f) — using Poisson fallback.", state, v, m)
            lam = max(m, 0.5)
            # Approximate NegBin with large r and p close to 1
            self._rs[state] = max(lam * 10, 0.5)
            self._ps[state] = np.clip(self._rs[state] / (self._rs[state] + lam), 0.01, 0.99)
        else:
            # Standard NegBin moment matching: mean = r(1-p)/p, var = r(1-p)/p^2
            p_hat = m / v
            r_hat = m * p_hat / (1 - p_hat)
            self._rs[state] = max(r_hat, 0.5)
            self._ps[state] = np.clip(p_hat, 0.01, 0.99)

        self._rebuild_cache_for(state)
        self._check_pmf_normalised(state)
        self._check_survivor_consistency(state)

    def expected_duration(self, state: int) -> float:
        # Expectation under the *truncated* PMF actually used by inference.
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        return float(np.dot(np.exp(self.log_pmf(state)), u))

    def n_parameters(self) -> int:
        return 2 * self.n_states  # r, p per state

    def get_params(self) -> dict:
        return {"rs": self._rs.copy(), "ps": self._ps.copy(), "n_states": self.n_states, "max_duration": self.max_duration}

    def set_params(self, params: dict) -> None:
        self._rs = np.asarray(params["rs"], dtype=np.float64).copy()
        self._ps = np.asarray(params["ps"], dtype=np.float64).copy()
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._log_pmf_cache.clear()
        for j in range(self.n_states):
            self._rebuild_cache_for(j)

    def _rebuild_cache_for(self, state: int) -> None:
        r, p = self._rs[state], self._ps[state]
        M = self.max_duration
        u = np.arange(0, M, dtype=np.float64)  # shifted: u-1 = 0..M-1
        log_p = nbinom.logpmf(u, r, p)
        log_norm = np.logaddexp.reduce(log_p)
        self._log_pmf_cache[state] = log_p - log_norm
