from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .features import margin


class AbstainMode(StrEnum):
    FLAT = "flat"
    HOLD_LAST = "hold_last"


class ArgmaxInferer:
    """Bridge inferer that collapses posterior probabilities to hard labels."""

    def infer(self, probs: NDArray) -> NDArray:
        return np.argmax(probs, axis=-1)

    def get_metadata(self) -> dict:
        return {}


class MarginGateInferer:
    """Gate label emission using posterior confidence thresholds."""

    def __init__(self, theta_top: float = 0.50, theta_margin: float = 0.10,
                 abstain_mode: AbstainMode | str = AbstainMode.HOLD_LAST, abstain_label: int = 0) -> None:
        self.theta_top = float(theta_top)
        self.theta_margin = float(theta_margin)
        self.abstain_mode = AbstainMode(abstain_mode)
        self.abstain_label = int(abstain_label)
        self._metadata: dict = {}
        self._validate_thresholds()

    def infer(self, probs: NDArray) -> NDArray:
        p = np.asarray(probs, dtype=float)
        if p.ndim != 2:
            raise ValueError(f"Expected posterior matrix with shape (T, K), got {p.shape}")
        if p.shape[1] < 2:
            raise ValueError(f"Expected at least 2 classes (K >= 2), got K={p.shape[1]}")
        if not np.all(np.isfinite(p)):
            raise ValueError("Posterior matrix contains NaN or Inf values.")

        hard_labels = np.argmax(p, axis=1).astype(np.int64)
        top_probs = np.max(p, axis=1)
        margins = margin(p)
        gate_open = (top_probs >= self.theta_top) & (margins >= self.theta_margin)

        labels = self._apply_abstain_mode(hard_labels=hard_labels, gate_open=gate_open)
        self._metadata = {
            "inferer": "MarginGateInferer",
            "theta_top": self.theta_top,
            "theta_margin": self.theta_margin,
            "abstain_mode": self.abstain_mode.value,
            "abstain_label": self.abstain_label,
            "gate_open_rate": float(np.mean(gate_open)) if len(gate_open) > 0 else float("nan"),
            "gate_open_count": int(np.sum(gate_open)),
            "abstained_count": int(len(gate_open) - np.sum(gate_open)),
        }
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)

    def _apply_abstain_mode(self, hard_labels: NDArray, gate_open: NDArray) -> NDArray:
        if len(hard_labels) == 0:
            return hard_labels

        if self.abstain_mode == AbstainMode.FLAT:
            return np.where(gate_open, hard_labels, self.abstain_label).astype(np.int64)

        output = np.empty_like(hard_labels, dtype=np.int64)
        last_label = hard_labels[0] if gate_open[0] else self.abstain_label
        output[0] = int(last_label)
        for i in range(1, len(hard_labels)):
            if gate_open[i]:
                last_label = hard_labels[i]
            output[i] = int(last_label)
        return output

    def _validate_thresholds(self) -> None:
        if not 0.0 <= self.theta_top <= 1.0:
            raise ValueError(f"theta_top must be in [0, 1], got {self.theta_top}")
        if not 0.0 <= self.theta_margin <= 1.0:
            raise ValueError(f"theta_margin must be in [0, 1], got {self.theta_margin}")
