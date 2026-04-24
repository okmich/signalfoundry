from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .features import _validate_posterior_matrix, entropy, margin, rolling_flip_rate


class AbstainMode(StrEnum):
    FLAT = "flat"
    HOLD_LAST = "hold_last"


def _check_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _apply_abstain_mode(hard_labels: NDArray, gate_open: NDArray, mode: AbstainMode, abstain_label: int) -> NDArray:
    if len(hard_labels) == 0:
        return hard_labels.astype(np.int64)

    if mode == AbstainMode.FLAT:
        return np.where(gate_open, hard_labels, abstain_label).astype(np.int64)

    n_rows = len(hard_labels)
    indexed = np.where(gate_open, np.arange(n_rows), -1)
    last_open_idx = np.maximum.accumulate(indexed)
    safe_idx = np.maximum(last_open_idx, 0)
    return np.where(last_open_idx >= 0, hard_labels[safe_idx], abstain_label).astype(np.int64)


def _build_gate_metadata(name: str, gate_open: NDArray, params: dict,
                         abstain_mode: AbstainMode, abstain_label: int) -> dict:
    total = len(gate_open)
    open_count = int(np.sum(gate_open))
    return {
        "inferer": name,
        **params,
        "abstain_mode": abstain_mode.value,
        "abstain_label": abstain_label,
        "gate_open_rate": float(np.mean(gate_open)) if total > 0 else float("nan"),
        "gate_open_count": open_count,
        "abstained_count": total - open_count,
    }


class ArgmaxInferer:
    """Bridge inferer that collapses posterior probabilities to hard labels.

    Validates input shape and NaN/Inf just like the gating inferers, returns ``int64``
    labels with axis-1 reduction. Use this when the downstream consumer needs a hard
    label and no confidence gating is desired.
    """

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "ArgmaxInferer")
        return np.argmax(p, axis=1).astype(np.int64)

    def get_metadata(self) -> dict:
        return {}


class MarginGateInferer:
    """Gate label emission using posterior top-probability and margin."""

    def __init__(self, theta_top: float = 0.50, theta_margin: float = 0.10,
                 abstain_mode: AbstainMode | str = AbstainMode.HOLD_LAST, abstain_label: int = 0) -> None:
        self.theta_top = float(theta_top)
        self.theta_margin = float(theta_margin)
        self.abstain_mode = AbstainMode(abstain_mode)
        self.abstain_label = int(abstain_label)
        self._metadata: dict = {}
        _check_unit_interval(self.theta_top, "theta_top")
        _check_unit_interval(self.theta_margin, "theta_margin")

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "MarginGateInferer")
        hard_labels = np.argmax(p, axis=1).astype(np.int64)
        # Single partition pass yields both the top probability and the top-two gap,
        # avoiding a separate np.max + np.partition sweep.
        partitioned = np.partition(p, -2, axis=1)
        top_probs = partitioned[:, -1]
        margins = partitioned[:, -1] - partitioned[:, -2]
        gate_open = (top_probs >= self.theta_top) & (margins >= self.theta_margin)

        labels = _apply_abstain_mode(hard_labels, gate_open, self.abstain_mode, self.abstain_label)
        self._metadata = _build_gate_metadata(
            "MarginGateInferer",
            gate_open,
            params={"theta_top": self.theta_top, "theta_margin": self.theta_margin},
            abstain_mode=self.abstain_mode,
            abstain_label=self.abstain_label,
        )
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)


class EntropyGateInferer:
    """Gate label emission using normalised entropy ``H(p) / log(K)``.

    Opens the gate when ``norm_entropy <= theta_entropy``. Lower normalised entropy means higher concentration,
    i.e. the posterior is more decisive.
    """

    def __init__(self, theta_entropy: float = 0.50,
                 abstain_mode: AbstainMode | str = AbstainMode.HOLD_LAST, abstain_label: int = 0) -> None:
        self.theta_entropy = float(theta_entropy)
        self.abstain_mode = AbstainMode(abstain_mode)
        self.abstain_label = int(abstain_label)
        self._metadata: dict = {}
        _check_unit_interval(self.theta_entropy, "theta_entropy")

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "EntropyGateInferer")
        hard_labels = np.argmax(p, axis=1).astype(np.int64)
        norm_entropy = entropy(p) / np.log(p.shape[1])
        gate_open = norm_entropy <= self.theta_entropy

        labels = _apply_abstain_mode(hard_labels, gate_open, self.abstain_mode, self.abstain_label)
        self._metadata = _build_gate_metadata(
            "EntropyGateInferer",
            gate_open,
            params={"theta_entropy": self.theta_entropy},
            abstain_mode=self.abstain_mode,
            abstain_label=self.abstain_label,
        )
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)


class StabilityGateInferer:
    """Gate label emission using trailing rolling flip-rate of the argmax label.

    Opens the gate when the fraction of argmax changes over the trailing ``window`` rows
    is ``<= theta_flip_rate``. Intuition: if the hard label has been thrashing recently,
    the current argmax is untrustworthy even if the instantaneous posterior is decisive.
    The first ``window - 1`` rows use an expanding denominator (matches what a live stream
    would have available).
    """

    def __init__(self, theta_flip_rate: float = 0.20, window: int = 5,
                 abstain_mode: AbstainMode | str = AbstainMode.HOLD_LAST, abstain_label: int = 0) -> None:
        self.theta_flip_rate = float(theta_flip_rate)
        self.window = int(window)
        self.abstain_mode = AbstainMode(abstain_mode)
        self.abstain_label = int(abstain_label)
        self._metadata: dict = {}
        _check_unit_interval(self.theta_flip_rate, "theta_flip_rate")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "StabilityGateInferer")
        hard_labels = np.argmax(p, axis=1).astype(np.int64)
        flip_rates = rolling_flip_rate(p, self.window)
        gate_open = flip_rates <= self.theta_flip_rate

        labels = _apply_abstain_mode(hard_labels, gate_open, self.abstain_mode, self.abstain_label)
        self._metadata = _build_gate_metadata(
            "StabilityGateInferer",
            gate_open,
            params={"theta_flip_rate": self.theta_flip_rate, "window": self.window},
            abstain_mode=self.abstain_mode,
            abstain_label=self.abstain_label,
        )
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)


class CompositeGateInferer:
    """Gate label emission on top-prob AND margin AND normalised-entropy.

    The §14 composite variant: most selective of the three gating forms. Opens when all three conditions hold simultaneously.
    """

    def __init__(self, theta_top: float = 0.50, theta_margin: float = 0.10, theta_entropy: float = 0.50,
                 abstain_mode: AbstainMode | str = AbstainMode.HOLD_LAST, abstain_label: int = 0) -> None:
        self.theta_top = float(theta_top)
        self.theta_margin = float(theta_margin)
        self.theta_entropy = float(theta_entropy)
        self.abstain_mode = AbstainMode(abstain_mode)
        self.abstain_label = int(abstain_label)
        self._metadata: dict = {}
        _check_unit_interval(self.theta_top, "theta_top")
        _check_unit_interval(self.theta_margin, "theta_margin")
        _check_unit_interval(self.theta_entropy, "theta_entropy")

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "CompositeGateInferer")
        hard_labels = np.argmax(p, axis=1).astype(np.int64)
        # Fused top-two extraction: single partition pass serves both top-prob and margin.
        partitioned = np.partition(p, -2, axis=1)
        top_probs = partitioned[:, -1]
        margins = partitioned[:, -1] - partitioned[:, -2]
        norm_entropy = entropy(p) / np.log(p.shape[1])
        gate_open = (top_probs >= self.theta_top) & (margins >= self.theta_margin) & (norm_entropy <= self.theta_entropy)

        labels = _apply_abstain_mode(hard_labels, gate_open, self.abstain_mode, self.abstain_label)
        self._metadata = _build_gate_metadata(
            "CompositeGateInferer",
            gate_open,
            params={
                "theta_top": self.theta_top,
                "theta_margin": self.theta_margin,
                "theta_entropy": self.theta_entropy,
            },
            abstain_mode=self.abstain_mode,
            abstain_label=self.abstain_label,
        )
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)
