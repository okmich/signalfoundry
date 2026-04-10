from __future__ import annotations

import logging

import numpy as np
from scipy.stats import gamma as gamma_dist

from ._base import BaseDuration

logger = logging.getLogger(__name__)


class GammaDuration(BaseDuration):
    """Discretised Gamma duration model.

    Parameters per state: ``shape`` (α) and ``scale`` (β).
    Discretisation: ``d[u] = CDF(u) - CDF(u-1)`` for ``u = 1 .. M``,
    renormalised so the truncated PMF sums to 1.

    The truncation at ``max_duration`` and renormalization does not add a
    free parameter — the underlying continuous distribution has 2 params
    per state and discretisation is deterministic given ``max_duration``.

    M-step uses moment matching (approximate, not exact MLE).
    """

    def __init__(self, n_states: int, max_duration: int, shapes: np.ndarray | None = None,
                 scales: np.ndarray | None = None):
        super().__init__(n_states, max_duration)
        if shapes is not None and scales is not None:
            self._shapes = np.asarray(shapes, dtype=np.float64).copy()
            self._scales = np.asarray(scales, dtype=np.float64).copy()
        else:
            self._shapes = np.full(n_states, 4.0, dtype=np.float64)
            self._scales = np.full(n_states, max_duration / 16.0, dtype=np.float64)
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
        m = np.dot(eta, u) / total
        v = np.dot(eta, u ** 2) / total - m ** 2

        if v <= 0 or m <= 0:
            logger.warning("Gamma M-step state %d: invalid moments (mean=%.4f, var=%.4f) — retaining previous params.", state, m, v)
            return

        shape_hat = m ** 2 / v
        scale_hat = v / m
        self._shapes[state] = max(shape_hat, 0.1)
        self._scales[state] = max(scale_hat, 0.1)

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
        return {"shapes": self._shapes.copy(), "scales": self._scales.copy(),
                "n_states": self.n_states, "max_duration": self.max_duration}

    def set_params(self, params: dict) -> None:
        self._shapes = np.asarray(params["shapes"], dtype=np.float64).copy()
        self._scales = np.asarray(params["scales"], dtype=np.float64).copy()
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._log_pmf_cache.clear()
        for j in range(self.n_states):
            self._rebuild_cache_for(j)

    def _rebuild_cache_for(self, state: int) -> None:
        a, s = self._shapes[state], self._scales[state]
        M = self.max_duration
        edges = np.arange(0, M + 1, dtype=np.float64)  # 0, 1, ..., M
        cdf_vals = gamma_dist.cdf(edges, a, scale=s)
        pmf = np.diff(cdf_vals)  # CDF(u) - CDF(u-1) for u=1..M
        pmf = np.maximum(pmf, 1e-300)  # prevent log(0)
        pmf /= pmf.sum()
        self._log_pmf_cache[state] = np.log(pmf)
