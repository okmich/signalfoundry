from __future__ import annotations

import logging

import numpy as np

from ._base import BaseDuration

logger = logging.getLogger(__name__)


class NonparametricDuration(BaseDuration):
    """Histogram-based (nonparametric) duration model.

    Stores a full probability table ``d[state, u]`` for ``u = 1 .. M``.
    Most flexible duration family but has ``M - 1`` free parameters per
    state, which inflates BIC for large ``M``.
    """

    def __init__(self, n_states: int, max_duration: int, log_pmfs: np.ndarray | None = None):
        super().__init__(n_states, max_duration)
        if log_pmfs is not None:
            self._log_pmfs = np.asarray(log_pmfs, dtype=np.float64).copy()
        else:
            # Initialise uniform
            self._log_pmfs = np.full((n_states, max_duration), -np.log(max_duration), dtype=np.float64)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @property
    def is_exact_mle(self) -> bool:
        return True

    def log_pmf(self, state: int) -> np.ndarray:
        return self._log_pmfs[state]

    def update(self, state: int, eta: np.ndarray) -> None:
        if self._skip_if_unused(state, eta):
            return
        M = self.max_duration
        # Floor to prevent zero-mass bins producing -inf
        floor = 1e-10 / M
        d = np.maximum(eta[:M], floor)
        d = d / d.sum()
        self._log_pmfs[state] = np.log(d)
        self._check_pmf_normalised(state)
        self._check_survivor_consistency(state)

    def expected_duration(self, state: int) -> float:
        u = np.arange(1, self.max_duration + 1, dtype=np.float64)
        return float(np.dot(np.exp(self._log_pmfs[state]), u))

    def n_parameters(self) -> int:
        # M - 1 free params per state (histogram sums to 1)
        return self.n_states * (self.max_duration - 1)

    def get_params(self) -> dict:
        return {"log_pmfs": self._log_pmfs.copy(), "n_states": self.n_states, "max_duration": self.max_duration}

    def set_params(self, params: dict) -> None:
        self._log_pmfs = np.asarray(params["log_pmfs"], dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def _renormalise_pmf(self, state: int) -> None:
        lp = self._log_pmfs[state]
        log_norm = np.logaddexp.reduce(lp)
        self._log_pmfs[state] = lp - log_norm
