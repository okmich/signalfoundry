"""Result types for univariate HMM threshold distillation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from .config import EmissionFamily, ModelSelectionMetric, ThresholdMethod


@dataclass(frozen=True)
class StateSummary:
    """Per-state raw-feature and posterior summary after state ordering."""

    ordered_state: int
    original_state: int
    count: int
    fraction: float
    feature_mean: float
    feature_median: float
    feature_std: float
    feature_iqr: float
    mean_top_prob: float


@dataclass(frozen=True)
class ThresholdBoundary:
    """Boundary between two adjacent ordered states."""

    lower_ordered_state: int
    upper_ordered_state: int
    value: float
    empirical_quantile: float
    method: ThresholdMethod


@dataclass(frozen=True)
class PairwiseSeparability:
    """Adjacent-state separability metrics in ordered-state space."""

    lower_ordered_state: int
    upper_ordered_state: int
    center_distance_over_pooled_iqr: float
    overlap_coefficient: float
    boundary_ambiguity: float


@dataclass(frozen=True)
class CandidateFit:
    """One fitted HMM candidate from the tuning grid."""

    n_states: int
    emission_family: EmissionFamily
    random_state: int | None
    log_likelihood: float
    aic: float
    bic: float
    selected_metric: ModelSelectionMetric
    selected_score: float
    mean_top_prob: float
    state_balance_ratio: float
    min_state_fraction: float
    missing_states: int
    valid: bool
    error: str | None = None


@dataclass(frozen=True)
class UnivariateHmmThresholdResult:
    """Output of ``UnivariateHmmThresholdDistiller.fit_distill``."""

    model: Any
    x: NDArray
    gamma: NDArray
    original_labels: NDArray
    ordered_labels: NDArray
    ordered_state_to_original_state: tuple[int, ...]
    thresholds: tuple[ThresholdBoundary, ...]
    state_summaries: tuple[StateSummary, ...]
    separability: tuple[PairwiseSeparability, ...]
    selected_candidate: CandidateFit
    candidates: tuple[CandidateFit, ...]
    threshold_labels: NDArray
    threshold_fidelity: float
    adjusted_rand_index: float
    # Number of boundaries where the raw extraction produced a non-monotonic value
    # and ``_monotonic_thresholds`` had to coerce it upward to maintain ordering.
    # A non-zero value is a real diagnostic: the fitted HMM is producing regimes
    # whose feature-value ranges overlap, not cleanly threshold-separable.
    non_monotonic_count: int = 0
    posterior_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def threshold_values(self) -> tuple[float, ...]:
        """Raw threshold values in ascending state order."""
        return tuple(boundary.value for boundary in self.thresholds)

    @property
    def state_summary_frame(self) -> pd.DataFrame:
        """Per-state summaries as a DataFrame."""
        return pd.DataFrame([summary.__dict__ for summary in self.state_summaries])

    @property
    def threshold_frame(self) -> pd.DataFrame:
        """Threshold boundaries as a DataFrame."""
        return pd.DataFrame([boundary.__dict__ for boundary in self.thresholds])

    @property
    def separability_frame(self) -> pd.DataFrame:
        """Adjacent-state separability metrics as a DataFrame."""
        return pd.DataFrame([item.__dict__ for item in self.separability])

    @property
    def candidate_frame(self) -> pd.DataFrame:
        """Fitted candidate grid as a DataFrame."""
        return pd.DataFrame([candidate.__dict__ for candidate in self.candidates])
