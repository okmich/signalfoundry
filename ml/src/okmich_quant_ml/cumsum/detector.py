from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocols import ReferenceModel


def _validate_scalar_observation(x: float, context: str) -> float:
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"{context}: observation must be finite, got {x}")
    return value


def _broadcast_slack(slack: float | NDArray, k: int) -> NDArray:
    arr = np.asarray(slack, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(k, float(arr), dtype=np.float64)
    elif arr.shape == (k,):
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(f"slack shape {arr.shape} incompatible with K={k}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"slack contains non-finite values: {slack}")
    if not np.all(out > 0.0):
        raise ValueError(f"slack must be > 0 per direction, got {slack}")
    return out


class CusumDetector:
    """Sequential univariate CUSUM detector with pluggable reference model.

    See ``signalfoundry-lab/docs/cusum.md`` §3.2 for the full contract.
    """

    def __init__(self, reference_model: ReferenceModel, slack: float | NDArray, reset_to_zero: bool = True) -> None:
        self.reference_model = reference_model
        k = int(reference_model.n_directions)
        if k not in (1, 2):
            raise ValueError(f"reference_model.n_directions must be 1 or 2, got {k}")
        self._k = k
        self.slack = _broadcast_slack(slack, k)
        self.reset_to_zero = bool(reset_to_zero)
        self.cusum_statistic_ = np.zeros(k, dtype=np.float64)

    def update(self, x: float) -> NDArray:
        value = _validate_scalar_observation(x, "CusumDetector.update")
        scores = np.asarray(self.reference_model.score(value), dtype=np.float64)
        if scores.shape != (self._k,):
            raise ValueError(f"reference_model.score returned shape {scores.shape}, expected ({self._k},)")
        if not np.all(np.isfinite(scores)):
            raise ValueError("reference_model.score returned NaN or Inf values.")

        increments = scores - self.slack
        new_state = self.cusum_statistic_ + increments
        if self.reset_to_zero:
            np.maximum(new_state, 0.0, out=new_state)
        self.cusum_statistic_[:] = new_state
        self.reference_model.update(value)
        return self.cusum_statistic_.copy()

    def batch(self, xs: NDArray) -> NDArray:
        values = np.asarray(xs, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"xs must be one-dimensional, got shape {values.shape}")
        if values.size > 0 and not np.all(np.isfinite(values)):
            raise ValueError("xs contains NaN or Inf values.")
        out = np.empty((len(values), self._k), dtype=np.float64)
        for i, value in enumerate(values):
            out[i] = self.update(float(value))
        return out

    def reset(self) -> None:
        self.cusum_statistic_[:] = 0.0
        self.reference_model.reset()

    @property
    def map_direction(self) -> int:
        return int(np.argmax(self.cusum_statistic_))

    def is_above_threshold(self, threshold: float | NDArray) -> bool:
        thr = np.asarray(threshold, dtype=np.float64)
        if thr.ndim == 0:
            return bool(np.any(self.cusum_statistic_ > float(thr)))
        if thr.shape != (self._k,):
            raise ValueError(f"threshold shape {thr.shape} incompatible with K={self._k}")
        return bool(np.any(self.cusum_statistic_ > thr))
