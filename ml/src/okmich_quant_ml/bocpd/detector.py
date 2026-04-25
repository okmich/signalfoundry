from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .protocols import ObservationModel


def _validate_scalar_observation(x: float, context: str) -> float:
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"{context}: observation must be finite, got {x}")
    return value


class BayesianOnlineChangepointDetector:
    """Adams-MacKay Bayesian online changepoint detection with constant hazard."""

    def __init__(self, observation_model: ObservationModel, hazard_rate: float, r_max: int) -> None:
        self.observation_model = observation_model
        self.hazard_rate = float(hazard_rate)
        self.r_max = int(r_max)
        if not 0.0 < self.hazard_rate < 1.0:
            raise ValueError(f"hazard_rate must be in (0, 1), got {hazard_rate}")
        if self.r_max < 2:
            raise ValueError(f"r_max must be >= 2, got {r_max}")

        self._log_hazard = float(np.log(self.hazard_rate))
        self._log_growth = float(np.log1p(-self.hazard_rate))
        # Reusable buffers for the hot update path — preallocating avoids
        # (r_max,) allocations per observation on long streams.
        self._buf_log_prev = np.empty(self.r_max, dtype=np.float64)
        self._buf_log_joint = np.empty(self.r_max, dtype=np.float64)
        self._buf_next_log = np.empty(self.r_max, dtype=np.float64)
        self.reset()

    def update(self, x: float) -> NDArray:
        """Consume one observation and return the updated run-length posterior."""
        value = _validate_scalar_observation(x, "BayesianOnlineChangepointDetector.update")
        log_pred = np.asarray(self.observation_model.log_pred_probs(value), dtype=np.float64)
        if log_pred.shape != (self.r_max,):
            expected = (self.r_max,)
            raise ValueError(f"observation_model.log_pred_probs returned shape {log_pred.shape}, expected {expected}")
        if not np.all(np.isfinite(log_pred)):
            raise ValueError("observation_model.log_pred_probs returned NaN or Inf values.")

        self._buf_log_prev[:] = -np.inf
        positive = self.run_length_posterior_ > 0.0
        np.log(self.run_length_posterior_, out=self._buf_log_prev, where=positive)
        np.add(self._buf_log_prev, log_pred, out=self._buf_log_joint)

        # logsumexp(x + c) = logsumexp(x) + c for scalar c — skip the temporary array.
        self._buf_next_log[0] = logsumexp(self._buf_log_joint) + self._log_hazard
        np.add(self._buf_log_joint[:-1], self._log_growth, out=self._buf_next_log[1:])
        self._buf_next_log[-1] = np.logaddexp(
            self._buf_next_log[-1], self._buf_log_joint[-1] + self._log_growth
        )

        log_norm = logsumexp(self._buf_next_log)
        if not np.isfinite(log_norm):
            raise ValueError("Unable to normalise run-length posterior; all paths have zero or invalid probability.")

        # Subtract-then-exp is self-normalising up to float precision because
        # log_norm == logsumexp(_buf_next_log); an explicit renormalise would
        # only remove ~r_max * 1e-16 drift, which does not accumulate across
        # steps because each step reseeds from logsumexp.
        np.subtract(self._buf_next_log, log_norm, out=self.run_length_posterior_)
        np.exp(self.run_length_posterior_, out=self.run_length_posterior_)
        self.observation_model.update(value)
        return self.run_length_posterior_.copy()

    def batch(self, xs: NDArray) -> NDArray:
        """Consume a one-dimensional float sequence without resetting state."""
        values = np.asarray(xs, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"xs must be one-dimensional, got shape {values.shape}")
        if values.size > 0 and not np.all(np.isfinite(values)):
            raise ValueError("xs contains NaN or Inf values.")
        if len(values) == 0:
            return np.empty((0, self.r_max), dtype=np.float64)

        batch_update = getattr(self.observation_model, "batch_update_posterior", None)
        if callable(batch_update):
            out = batch_update(values, self.run_length_posterior_, self._log_hazard, self._log_growth)
            self.run_length_posterior_ = out[-1].copy()
            return out

        out = np.empty((len(values), self.r_max), dtype=np.float64)
        for i, value in enumerate(values):
            out[i] = self.update(float(value))
        return out

    def reset(self) -> None:
        """Restore detector and observation model to their prior state."""
        self.run_length_posterior_ = np.zeros(self.r_max, dtype=np.float64)
        self.run_length_posterior_[0] = 1.0
        self.observation_model.reset(self.r_max)

    @property
    def changepoint_prob(self) -> float:
        """Probability that the latest observation starts a new run."""
        return float(self.run_length_posterior_[0])

    @property
    def map_run_length(self) -> int:
        """MAP run-length index for the latest posterior."""
        return int(np.argmax(self.run_length_posterior_))
