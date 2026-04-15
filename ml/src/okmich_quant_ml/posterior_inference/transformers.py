from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_probability_matrix(probs: NDArray, eps: float) -> NDArray:
    array = np.asarray(probs, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected posterior matrix with shape (T, K), got {array.shape}")
    if array.shape[1] < 2:
        raise ValueError(f"Expected at least 2 classes (K >= 2), got K={array.shape[1]}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Posterior matrix contains NaN or Inf values.")

    clipped = np.clip(array, eps, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Posterior rows must have strictly positive sums.")
    return clipped / row_sums


def _scalar_nll(probs: NDArray, y_idx: NDArray, eps: float) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_idx, dtype=np.int64)
    chosen = np.clip(p[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(chosen)))


class EmaPosteriorTransformer:
    """Exponential moving average smoothing on posterior rows."""

    def __init__(self, alpha: float = 0.20, eps: float = 1e-12) -> None:
        self.alpha = float(alpha)
        self.eps = float(eps)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, eps=self.eps)
        out = np.empty_like(p, dtype=float)
        out[0] = p[0]
        for i in range(1, len(p)):
            out[i] = self.alpha * p[i] + (1.0 - self.alpha) * out[i - 1]
        out = np.clip(out, self.eps, None)
        return out / out.sum(axis=1, keepdims=True)


class TemperatureScalingTransformer:
    """Post-hoc temperature scaling on posterior probabilities."""

    def __init__(self, temperature: float = 1.0, eps: float = 1e-12,
                 search_min: float = 0.25, search_max: float = 20.0,
                 search_points: int = 81, refinement_steps: int = 2) -> None:
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.search_min = float(search_min)
        self.search_max = float(search_max)
        self.search_points = int(search_points)
        self.refinement_steps = int(refinement_steps)
        self._validate_configuration()

    def fit(self, probs: NDArray, y_idx: NDArray) -> TemperatureScalingTransformer:
        p = _as_probability_matrix(probs, eps=self.eps)
        y = np.asarray(y_idx, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError(f"y_idx must be 1D, got shape {y.shape}")
        if len(y) != len(p):
            raise ValueError(f"y_idx length must equal number of rows in probs. Got {len(y)} vs {len(p)}.")
        if np.any(y < 0) or np.any(y >= p.shape[1]):
            raise ValueError(f"y_idx must be in [0, K-1] where K={p.shape[1]}.")

        best_temp = float(np.clip(self.temperature, self.search_min, self.search_max))
        best_nll = _scalar_nll(self._apply_temperature(p, best_temp), y, eps=self.eps)

        lower, upper = self.search_min, self.search_max
        for _ in range(self.refinement_steps + 1):
            grid = np.geomspace(lower, upper, self.search_points)
            nlls = np.array([_scalar_nll(self._apply_temperature(p, t), y, eps=self.eps) for t in grid], dtype=float)
            idx = int(np.argmin(nlls))
            candidate_temp = float(grid[idx])
            candidate_nll = float(nlls[idx])
            if candidate_nll < best_nll:
                best_nll = candidate_nll
                best_temp = candidate_temp

            left_idx = max(0, idx - 1)
            right_idx = min(len(grid) - 1, idx + 1)
            lower = max(self.search_min, float(grid[left_idx]))
            upper = min(self.search_max, float(grid[right_idx]))
            if np.isclose(lower, upper):
                break

        self.temperature = best_temp
        return self

    def transform(self, probs: NDArray) -> NDArray:
        p = _as_probability_matrix(probs, eps=self.eps)
        return self._apply_temperature(p, self.temperature)

    def _apply_temperature(self, probs: NDArray, temperature: float) -> NDArray:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        logits = np.log(np.clip(probs, self.eps, 1.0)) / temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        exp_logits = np.clip(exp_logits, self.eps, None)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _validate_configuration(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.search_min <= 0.0 or self.search_max <= 0.0:
            raise ValueError("search_min and search_max must be > 0.")
        if self.search_min >= self.search_max:
            raise ValueError(f"search_min must be < search_max, got {self.search_min} >= {self.search_max}")
        if self.search_points < 3:
            raise ValueError(f"search_points must be >= 3, got {self.search_points}")
        if self.refinement_steps < 0:
            raise ValueError(f"refinement_steps must be >= 0, got {self.refinement_steps}")
