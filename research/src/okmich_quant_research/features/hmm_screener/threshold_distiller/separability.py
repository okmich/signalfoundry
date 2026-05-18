"""Regime separability metrics for univariate HMM threshold distillation."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import PairwiseSeparability, StateSummary


def build_state_summaries(x: NDArray, ordered_labels: NDArray, original_labels: NDArray, state_order: tuple[int, ...],
                          top_probs: NDArray) -> tuple[StateSummary, ...]:
    """Summarize each ordered state on the raw feature scale."""
    summaries = []
    n = len(x)
    for ordered_state, original_state in enumerate(state_order):
        mask = ordered_labels == ordered_state
        values = x[mask]
        if len(values) == 0:
            summaries.append(
                StateSummary(
                    ordered_state=ordered_state,
                    original_state=original_state,
                    count=0,
                    fraction=0.0,
                    feature_mean=float("nan"),
                    feature_median=float("nan"),
                    feature_std=float("nan"),
                    feature_iqr=float("nan"),
                    mean_top_prob=float("nan"),
                )
            )
            continue
        q25, q75 = np.quantile(values, [0.25, 0.75])
        summaries.append(
            StateSummary(
                ordered_state=ordered_state,
                original_state=int(original_state),
                count=int(mask.sum()),
                fraction=float(mask.mean()) if n else 0.0,
                feature_mean=float(values.mean()),
                feature_median=float(np.median(values)),
                feature_std=float(values.std(ddof=0)),
                feature_iqr=float(q75 - q25),
                mean_top_prob=float(top_probs[mask].mean()),
            )
        )
    return tuple(summaries)


def empirical_overlap_coefficient(left: NDArray, right: NDArray, bins: int = 80) -> float:
    """Approximate the overlap coefficient between two empirical distributions."""
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    lo = float(min(np.min(left), np.min(right)))
    hi = float(max(np.max(left), np.max(right)))
    if not np.isfinite(lo + hi) or hi <= lo:
        return 1.0
    left_hist, edges = np.histogram(left, bins=bins, range=(lo, hi), density=True)
    right_hist, _ = np.histogram(right, bins=edges, density=True)
    width = float(edges[1] - edges[0])
    return float(np.sum(np.minimum(left_hist, right_hist)) * width)


def build_pairwise_separability(x: NDArray, ordered_labels: NDArray, thresholds: tuple[float, ...],
                                eps: float = 1e-12) -> tuple[PairwiseSeparability, ...]:
    """Compute adjacent-state separability metrics."""
    if len(thresholds) == 0:
        return ()

    rows = []
    for lower_state, threshold in enumerate(thresholds):
        upper_state = lower_state + 1
        lower_values = x[ordered_labels == lower_state]
        upper_values = x[ordered_labels == upper_state]
        if len(lower_values) == 0 or len(upper_values) == 0:
            rows.append(
                PairwiseSeparability(
                    lower_ordered_state=lower_state,
                    upper_ordered_state=upper_state,
                    center_distance_over_pooled_iqr=float("nan"),
                    overlap_coefficient=float("nan"),
                    boundary_ambiguity=float("nan"),
                )
            )
            continue

        lower_q25, lower_q75 = np.quantile(lower_values, [0.25, 0.75])
        upper_q25, upper_q75 = np.quantile(upper_values, [0.25, 0.75])
        pooled_iqr = 0.5 * ((lower_q75 - lower_q25) + (upper_q75 - upper_q25))
        center_distance = abs(float(np.median(upper_values) - np.median(lower_values)))
        spread = float(np.std(np.concatenate([lower_values, upper_values]), ddof=0))
        ambiguity_width = max(spread * 0.10, eps)
        near_boundary = np.abs(x - threshold) <= ambiguity_width
        adjacent = (ordered_labels == lower_state) | (ordered_labels == upper_state)
        boundary_ambiguity = float((near_boundary & adjacent).mean())

        rows.append(
            PairwiseSeparability(
                lower_ordered_state=lower_state,
                upper_ordered_state=upper_state,
                center_distance_over_pooled_iqr=float(center_distance / max(pooled_iqr, eps)),
                overlap_coefficient=empirical_overlap_coefficient(lower_values, upper_values),
                boundary_ambiguity=boundary_ambiguity,
            )
        )
    return tuple(rows)
