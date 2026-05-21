from __future__ import annotations

from enum import StrEnum

import numpy as np
from numba import njit
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

    Validates input shape and NaN/Inf just like the gating inferers, returns ``int64`` labels with axis-1 reduction.
    Use this when the downstream consumer needs a hard label and no confidence gating is desired.
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

    Opens the gate when ``norm_entropy <= theta_entropy``. Lower normalised entropy means higher concentration, i.e. the posterior is more decisive.
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

    Opens the gate when the fraction of argmax changes over the trailing ``window`` rows is ``<= theta_flip_rate``.
    Intuition: if the hard label has been thrashing recently, the current argmax is untrustworthy even if the instantaneous posterior is decisive.
    The first ``window - 1`` rows use an expanding denominator (matches what a live stream would have available).
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


@njit(cache=True)
def _viterbi_decode(unary_cost: np.ndarray, transition_cost: np.ndarray) -> np.ndarray:
    """Find the minimum-cost label sequence given unary and pairwise costs.

    ``unary_cost`` is ``(T, K)``; ``transition_cost`` is ``(K, K)`` with entry ``[i, j]`` the cost of
    moving from label ``i`` at bar ``t`` to label ``j`` at bar ``t+1``. Lower cost is preferred. Standard
    Viterbi: forward pass with backpointers, then backward trace. ``O(T·K²)`` time, ``O(T·K)`` memory.
    """
    T, K = unary_cost.shape
    dp = np.empty((T, K), dtype=np.float64)
    backpointer = np.empty((T, K), dtype=np.int64)

    dp[0] = unary_cost[0]
    backpointer[0] = 0

    for t in range(1, T):
        for j in range(K):
            best_cost = np.inf
            best_prev = 0
            for i in range(K):
                cost = dp[t - 1, i] + transition_cost[i, j]
                if cost < best_cost:
                    best_cost = cost
                    best_prev = i
            dp[t, j] = best_cost + unary_cost[t, j]
            backpointer[t, j] = best_prev

    path = np.empty(T, dtype=np.int64)
    path[T - 1] = np.argmin(dp[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]
    return path


class ViterbiInferer:
    """MAP label-sequence inferer over fixed unary (posterior) and pairwise (transition) costs.

    Unlike ``ArgmaxInferer`` which is row-local, this inferer finds the globally minimum-cost label
    sequence via the Viterbi algorithm. Per-bar evidence is converted to unary cost as
    ``-observation_weight * log(p + eps)``; the supplied ``transition_cost`` matrix penalises label
    changes between consecutive bars.

    **Non-causal.** Viterbi requires the full sequence (backward pass after forward). For live trading
    use ``ArgmaxInferer`` or a gate inferer instead; this inferer is for offline analysis, label
    refinement, or backtests that explicitly accept a one-shot batch decode.

    Construction shortcuts:

    * ``with_uniform_smoothing(K, off_diagonal_cost)`` — uniform penalty on any label change.
    * ``from_transition_probabilities(transmat)`` — convert a row-stochastic matrix to cost via
      ``-log(transmat + eps)``. Higher probability ⇒ lower cost.
    """

    def __init__(self, transition_cost: NDArray, observation_weight: float = 1.0,
                 eps: float = 1e-12) -> None:
        cost = np.asarray(transition_cost, dtype=float)
        if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] < 2:
            raise ValueError(f"transition_cost must be a (K, K) matrix with K >= 2, got shape={cost.shape}")
        if not np.isfinite(cost).all():
            raise ValueError("transition_cost contains NaN or Inf values.")
        if float(observation_weight) <= 0.0:
            raise ValueError(f"observation_weight must be > 0, got {observation_weight}")
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.transition_cost = cost
        self.observation_weight = float(observation_weight)
        self.eps = float(eps)
        self._metadata: dict = {}

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "ViterbiInferer")
        T, K = p.shape
        if K != self.transition_cost.shape[0]:
            raise ValueError(
                f"ViterbiInferer: posterior K={K} does not match transition_cost K={self.transition_cost.shape[0]}."
            )
        if T == 0:
            self._metadata = {
                "inferer": "ViterbiInferer", "n_states": int(K), "n_bars": 0,
                "observation_weight": self.observation_weight, "total_path_cost": 0.0,
            }
            return np.zeros(0, dtype=np.int64)
        unary = -self.observation_weight * np.log(np.clip(p, self.eps, None))
        path = _viterbi_decode(unary, self.transition_cost)
        rows = np.arange(T)
        total_unary = float(unary[rows, path].sum())
        total_trans = float(self.transition_cost[path[:-1], path[1:]].sum()) if T > 1 else 0.0
        self._metadata = {
            "inferer": "ViterbiInferer", "n_states": int(K), "n_bars": int(T),
            "observation_weight": self.observation_weight,
            "total_unary_cost": total_unary, "total_transition_cost": total_trans,
            "total_path_cost": total_unary + total_trans,
        }
        return path

    def get_metadata(self) -> dict:
        return dict(self._metadata)

    @classmethod
    def with_uniform_smoothing(cls, n_states: int, off_diagonal_cost: float,
                               observation_weight: float = 1.0, eps: float = 1e-12) -> ViterbiInferer:
        """Build with a uniform off-diagonal transition cost. Diagonal (no change) costs 0."""
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if off_diagonal_cost < 0.0:
            raise ValueError(f"off_diagonal_cost must be >= 0, got {off_diagonal_cost}")
        cost = np.full((n_states, n_states), float(off_diagonal_cost), dtype=float)
        np.fill_diagonal(cost, 0.0)
        return cls(cost, observation_weight=observation_weight, eps=eps)

    @classmethod
    def from_transition_probabilities(cls, transmat: NDArray, observation_weight: float = 1.0,
                                      eps: float = 1e-12, row_sum_tol: float = 1e-6) -> ViterbiInferer:
        """Build by converting a row-stochastic transmat to cost via ``-log(transmat + eps)``.

        Validates that ``transmat`` is finite, non-negative, and that each row sums to 1 within
        ``row_sum_tol``. Non-stochastic rows silently producing valid-looking transition costs would
        bias the decoded path, so this method is strict — pass an explicit ``transition_cost`` to
        the constructor if you need arbitrary cost matrices.
        """
        a = np.asarray(transmat, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
            raise ValueError(f"transmat must be a (K, K) matrix with K >= 2, got shape={a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("transmat contains NaN or Inf values.")
        if (a < 0.0).any():
            raise ValueError(f"transmat contains negative values (min={a.min():.3e}).")
        row_sums = a.sum(axis=1)
        max_dev = float(np.abs(row_sums - 1.0).max())
        if max_dev > row_sum_tol:
            raise ValueError(
                f"transmat rows must sum to 1 within {row_sum_tol} (max deviation {max_dev:.3e}); "
                f"non-stochastic rows would silently produce biased transition costs."
            )
        cost = -np.log(np.clip(a, eps, None))
        return cls(cost, observation_weight=observation_weight, eps=eps)


@njit(cache=True)
def _confidence_weighted_mode_scores(probs: np.ndarray, argmax: np.ndarray, window: int,
                                     confidence_weight: float) -> np.ndarray:
    """Per-bar, per-state confidence-weighted vote scores over the trailing window.

    Score for state ``k`` at bar ``t`` =  count(argmax == k in window) ×
    mean(probs[t', k] for t' in window where argmax[t'] == k) ^ ``confidence_weight``.

    Causal: the window is ``[max(0, t - window + 1), t]``; warmup uses an expanding effective window.
    """
    T, K = probs.shape
    scores = np.zeros((T, K), dtype=np.float64)
    for t in range(T):
        start = t - window + 1
        if start < 0:
            start = 0
        for k in range(K):
            count = 0
            prob_sum = 0.0
            for u in range(start, t + 1):
                if argmax[u] == k:
                    count += 1
                    prob_sum += probs[u, k]
            if count > 0:
                if confidence_weight == 0.0:
                    scores[t, k] = float(count)
                else:
                    mean_prob = prob_sum / count
                    scores[t, k] = float(count) * (mean_prob ** confidence_weight)
    return scores


class ConfidenceWeightedModeInferer:
    """Window mode-voting where votes are weighted by posterior confidence.

    For each bar ``t``, looks at the trailing ``window`` rows, counts how often each label is the
    row-argmax, and weights each count by the mean posterior probability of that label among the
    rows where it was the argmax. The state with the highest score that clears
    ``min_score_threshold`` wins; if no state clears the threshold, the configured abstain mode
    fires (``HOLD_LAST`` carries the last accepted label, ``FLAT`` emits ``abstain_label``).

    Causal: trailing window with expanding denominator for the first ``window − 1`` rows. Replaces
    ``regime_filters.ModeFilterWithConfidence`` with two changes worth noting:

    1. Posterior-only input — the bar argmax is derived internally rather than passed as ``states``.
    2. Abstain semantics on threshold failure replace the original's "fall back to bar argmax"
       behavior, for consistency with the gate-inferer family in this package.
    """

    def __init__(self, window: int = 7, confidence_weight: float = 1.0, min_score_threshold: float = 0.0,
                 abstain_mode: AbstainMode | str = AbstainMode.HOLD_LAST, abstain_label: int = 0) -> None:
        self.window = int(window)
        self.confidence_weight = float(confidence_weight)
        self.min_score_threshold = float(min_score_threshold)
        self.abstain_mode = AbstainMode(abstain_mode)
        self.abstain_label = int(abstain_label)
        self._metadata: dict = {}
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.confidence_weight < 0.0:
            raise ValueError(f"confidence_weight must be >= 0, got {self.confidence_weight}")
        if self.min_score_threshold < 0.0:
            raise ValueError(f"min_score_threshold must be >= 0, got {self.min_score_threshold}")

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "ConfidenceWeightedModeInferer")
        T = p.shape[0]
        argmax = np.argmax(p, axis=1).astype(np.int64)
        if T == 0:
            self._metadata = _build_gate_metadata(
                "ConfidenceWeightedModeInferer", np.zeros(0, dtype=bool),
                params={"window": self.window, "confidence_weight": self.confidence_weight,
                        "min_score_threshold": self.min_score_threshold},
                abstain_mode=self.abstain_mode, abstain_label=self.abstain_label,
            )
            return argmax
        scores = _confidence_weighted_mode_scores(p, argmax, self.window, self.confidence_weight)
        candidate_labels = np.argmax(scores, axis=1).astype(np.int64)
        best_scores = scores[np.arange(T), candidate_labels]
        gate_open = best_scores >= self.min_score_threshold
        labels = _apply_abstain_mode(candidate_labels, gate_open, self.abstain_mode, self.abstain_label)
        self._metadata = _build_gate_metadata(
            "ConfidenceWeightedModeInferer", gate_open,
            params={"window": self.window, "confidence_weight": self.confidence_weight,
                    "min_score_threshold": self.min_score_threshold},
            abstain_mode=self.abstain_mode, abstain_label=self.abstain_label,
        )
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)


@njit(cache=True)
def _confidence_hysteresis_core(probs: np.ndarray, argmax: np.ndarray, entry_threshold: float,
                                exit_threshold: float) -> np.ndarray:
    """State-machine inferer with cumulative-confidence entry / exit thresholds.

    Tracks a current regime. Each bar accumulates ``(1 - p_t[current_regime])`` into an exit-confidence
    counter until ``exit_threshold`` is reached; on exit, the candidate new regime is the bar argmax,
    confirmed only if the running back-trace of consecutive same-argmax posteriors clears
    ``entry_threshold``. Output is the running regime label per bar.
    """
    T = len(argmax)
    out = np.empty(T, dtype=np.int64)
    if T == 0:
        return out
    current_regime = argmax[0]
    exit_confidence = 0.0
    out[0] = current_regime
    for t in range(1, T):
        regime_prob = probs[t, current_regime]
        bar_label = argmax[t]
        if bar_label == current_regime:
            exit_confidence = 0.0
        else:
            exit_confidence += 1.0 - regime_prob
            if exit_confidence >= exit_threshold:
                new_regime = bar_label
                entry_conf = 0.0
                for u in range(t, -1, -1):
                    if argmax[u] == new_regime:
                        entry_conf += probs[u, new_regime]
                        if entry_conf >= entry_threshold:
                            current_regime = new_regime
                            exit_confidence = 0.0
                            break
                    else:
                        break
        out[t] = current_regime
    return out


class ConfidenceHysteresisInferer:
    """State-machine inferer with asymmetric entry / exit thresholds, gated by posterior confidence.

    Models "stairs up, elevator down" regime behavior. Once a regime is current, leaving requires
    accumulating ``exit_threshold`` worth of ``(1 - p_t[current_regime])`` evidence; entering a new
    regime requires accumulating ``entry_threshold`` worth of ``p_t[candidate_regime]`` over a
    consecutive run of same-argmax bars (the back-trace stops at the first disagreement).

    Causal: every decision uses only the running state and the trailing posterior. Replaces the
    confidence-based path of ``regime_filters.HysteresisProcessor`` (the count-based path stays in
    ``regime_filters`` as a pure label-sequence operator).

    Set ``entry_threshold`` and ``exit_threshold`` asymmetrically to encode regime-specific
    persistence:

    * ``entry_threshold > exit_threshold`` ⇒ enter slowly, exit quickly. Conservative.
    * ``entry_threshold < exit_threshold`` ⇒ enter quickly, exit slowly. Aggressive / sticky.
    """

    def __init__(self, entry_threshold: float = 1.0, exit_threshold: float = 1.0) -> None:
        self.entry_threshold = float(entry_threshold)
        self.exit_threshold = float(exit_threshold)
        if self.entry_threshold <= 0.0:
            raise ValueError(f"entry_threshold must be > 0, got {self.entry_threshold}")
        if self.exit_threshold <= 0.0:
            raise ValueError(f"exit_threshold must be > 0, got {self.exit_threshold}")
        self._metadata: dict = {}

    def infer(self, probs: NDArray) -> NDArray:
        p = _validate_posterior_matrix(probs, "ConfidenceHysteresisInferer")
        T, K = p.shape
        argmax = np.argmax(p, axis=1).astype(np.int64)
        labels = _confidence_hysteresis_core(p, argmax, self.entry_threshold, self.exit_threshold)
        if T > 0:
            transitions = int((labels[1:] != labels[:-1]).sum())
            argmax_transitions = int((argmax[1:] != argmax[:-1]).sum())
        else:
            transitions = 0
            argmax_transitions = 0
        self._metadata = {
            "inferer": "ConfidenceHysteresisInferer",
            "entry_threshold": self.entry_threshold, "exit_threshold": self.exit_threshold,
            "n_bars": int(T), "n_states": int(K),
            "n_label_transitions": transitions,
            "n_argmax_transitions": argmax_transitions,
            "smoothing_ratio": (1.0 - transitions / argmax_transitions) if argmax_transitions > 0 else 0.0,
        }
        return labels

    def get_metadata(self) -> dict:
        return dict(self._metadata)
