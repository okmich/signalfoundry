"""Configuration for univariate HMM threshold distillation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from numpy.typing import NDArray

from enum import StrEnum


class AxisType(StrEnum):
    """Semantic axis the univariate HMM is intended to describe."""

    DIRECTION = "direction"
    VOLATILITY = "volatility"
    EFFICIENCY = "efficiency"
    MOMENTUM = "momentum"
    LIQUIDITY = "liquidity"


class StateOrdering(StrEnum):
    """How fitted HMM states are ordered before threshold extraction."""

    FEATURE_MEDIAN = "feature_median"
    FEATURE_MEAN = "feature_mean"
    EMISSION_LOCATION = "emission_location"


class ThresholdMethod(StrEnum):
    """Static threshold extraction method."""

    POSTERIOR_MAP_SWITCH = "posterior_map_switch"
    EMISSION_CROSSING = "emission_crossing"
    EMPIRICAL_SWITCH_QUANTILE = "empirical_switch_quantile"


@dataclass(frozen=True)
class UnivariateHmmThresholdConfig:
    """Settings for distilling a fitted univariate HMM into static raw-feature thresholds.

    Model training is the caller's responsibility — pass a pre-fitted HMM to the distiller.
    The distiller orders the fitted states, extracts adjacent-state thresholds on the raw feature,
    and reports fidelity and separability diagnostics.
    """

    axis_type: AxisType
    state_ordering: StateOrdering = StateOrdering.FEATURE_MEDIAN
    threshold_method: ThresholdMethod = ThresholdMethod.POSTERIOR_MAP_SWITCH
    posterior_confidence_thresholds: tuple[float, ...] = (0.95, 0.99)
    emission_grid_size: int = 2048
    eps: float = 1e-12
    # Input-scale contract: the distiller is only valid on stationary, scale-stable
    # features. The caller must apply a StandardScaler / RobustScaler / rolling-z /
    # rolling-rank or equivalent first. Enforced at fit time via a quantile-spread
    # check on x: defaults reject raw log-returns (~0.05 spread) and raw prices/
    # volumes (huge spread) while passing standard / robust / rank-scaled inputs.
    input_scale_quantiles: tuple[float, float] = (0.01, 0.99)
    input_scale_min_spread: float = 0.2
    input_scale_max_spread: float = 50.0

    def __post_init__(self) -> None:
        if any((q < 0.0 or q > 1.0) for q in self.posterior_confidence_thresholds):
            raise ValueError("posterior_confidence_thresholds values must be in [0, 1]")
        if self.emission_grid_size < 128:
            raise ValueError(f"emission_grid_size must be >= 128, got {self.emission_grid_size}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
        q_lo, q_hi = self.input_scale_quantiles
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError(
                f"input_scale_quantiles must satisfy 0 <= lo < hi <= 1, got {self.input_scale_quantiles}"
            )
        if self.input_scale_min_spread <= 0:
            raise ValueError(f"input_scale_min_spread must be positive, got {self.input_scale_min_spread}")
        if self.input_scale_max_spread <= self.input_scale_min_spread:
            raise ValueError(
                f"input_scale_max_spread ({self.input_scale_max_spread}) must exceed "
                f"input_scale_min_spread ({self.input_scale_min_spread})"
            )


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
class UnivariateHmmThresholdResult:
    """Output of ``UnivariateHmmThresholdDistiller.distill``."""

    model: Any
    x: NDArray
    gamma: NDArray
    original_labels: NDArray
    ordered_labels: NDArray
    ordered_state_to_original_state: tuple[int, ...]
    thresholds: tuple[ThresholdBoundary, ...]
    state_summaries: tuple[StateSummary, ...]
    separability: tuple[PairwiseSeparability, ...]
    threshold_labels: NDArray
    threshold_fidelity: float
    adjusted_rand_index: float
    # Number of boundaries where the raw extraction produced a non-monotonic value
    # and ``_monotonic_thresholds`` had to coerce it upward to maintain ordering.
    # A non-zero value is a real diagnostic: the fitted HMM is producing regimes
    # whose feature-value ranges overlap, not cleanly threshold-separable.
    non_monotonic_count: int = 0
    posterior_metrics: dict[str, float] = field(default_factory=dict)
    # Bar-pair agreement on whether a regime switch occurred (rule vs HMM).
    # Parallel to ``threshold_fidelity``, but for transitions rather than labels.
    # Catches the flicker failure mode where bar-by-bar agreement is high but
    # the rule switches regimes far more often than the HMM (a stateless rule
    # approximating a temporal model). See ``transition_metrics`` for the
    # supporting counts and rates that decompose this scalar.
    transition_fidelity: float = 0.0
    transition_metrics: dict[str, float] = field(default_factory=dict)

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
