from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)

_NEG_INF = -np.inf


class BaseDuration(ABC):
    """Abstract base class for HSMM duration distributions.

    Each concrete subclass parameterises P(duration = u | state j) for
    u = 1 .. max_duration and provides the M-step update from expected
    duration counts.

    All computations are in **log-domain** to avoid underflow.
    """

    def __init__(self, n_states: int, max_duration: int):
        if n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")
        if max_duration < 2:
            raise ValueError(f"max_duration must be >= 2, got {max_duration}")
        self.n_states = n_states
        self.max_duration = max_duration  # M

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    @abstractmethod
    def log_pmf(self, state: int) -> np.ndarray:
        """Log P(duration = u | state) for u = 1 .. M.  Shape ``(M,)``."""

    def log_survivor(self, state: int) -> np.ndarray:
        """Log P(duration >= u | state) for u = 1 .. M.  Shape ``(M,)``.

        Default implementation: reverse cumulative logsumexp of ``log_pmf``.
        ``log_survivor[0] = log(1) = 0`` (duration >= 1 is certain),
        ``log_survivor[u-1] = logsumexp(log_pmf[u-1 : M])``.
        """
        lp = self.log_pmf(state)
        M = len(lp)
        log_surv = np.empty(M, dtype=np.float64)
        # Reverse cumulative logsumexp: log_surv[u] = logsumexp(lp[u:])
        log_surv[M - 1] = lp[M - 1]
        for u in range(M - 2, -1, -1):
            log_surv[u] = np.logaddexp(lp[u], log_surv[u + 1])
        return log_surv

    @abstractmethod
    def update(self, state: int, eta: np.ndarray) -> None:
        """M-step: update parameters from expected duration counts.

        Parameters
        ----------
        state : int
            State index.
        eta : np.ndarray, shape ``(M,)``
            Expected number of segments with duration u = 1 .. M.
        """

    @abstractmethod
    def expected_duration(self, state: int) -> float:
        """Analytic expected duration E[D | state]."""

    @abstractmethod
    def n_parameters(self) -> int:
        """Total free parameter count across all states (for AIC/BIC)."""

    @property
    def is_exact_mle(self) -> bool:
        """Whether the M-step is exact MLE (True) or approximate (False).

        Exact-MLE duration models (Poisson, Nonparametric) guarantee EM
        monotonicity.  Approximate models (NegBin, Gamma, LogNormal) use
        moment matching and may cause small LL decreases.
        """
        return False  # subclasses override

    # ------------------------------------------------------------------
    # Param I/O
    # ------------------------------------------------------------------
    @abstractmethod
    def get_params(self) -> dict:
        """Return a serialisable dict of all parameters."""

    @abstractmethod
    def set_params(self, params: dict) -> None:
        """Restore parameters from a dict produced by ``get_params``."""

    # ------------------------------------------------------------------
    # Validation helpers (called by subclasses after update)
    # ------------------------------------------------------------------
    def _check_pmf_normalised(self, state: int, tol: float = 1e-6) -> None:
        """Verify PMF sums to 1; renormalise and warn if not."""
        lp = self.log_pmf(state)
        total = np.exp(logsumexp(lp))
        if abs(total - 1.0) > tol:
            logger.warning("Duration PMF for state %d sums to %.8f — renormalising.", state, total)
            self._renormalise_pmf(state)

    def _renormalise_pmf(self, state: int) -> None:  # pragma: no cover — subclass hook
        """Subclasses override if they store raw PMF tables (e.g. Nonparametric)."""
        pass

    def _check_survivor_consistency(self, state: int, tol: float = 1e-6) -> None:
        """Verify survivor function starts at 1 and is monotonically non-increasing."""
        ls = self.log_survivor(state)
        surv_0 = np.exp(ls[0])
        if abs(surv_0 - 1.0) > tol:
            logger.warning("Survivor[%d][0] = %.8f, expected 1.0.", state, surv_0)
        if len(ls) > 1 and np.any(np.diff(ls) > tol):
            logger.warning("Survivor function for state %d is not monotonically non-increasing.", state)

    def _skip_if_unused(self, state: int, eta: np.ndarray, eps: float = 1e-12) -> bool:
        """Return True (and log) if the state received negligible posterior mass."""
        if eta.sum() < eps:
            logger.debug("State %d unused in this EM iteration — retaining previous duration params.", state)
            return True
        return False
