"""Configuration for univariate HMM threshold distillation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AxisType(StrEnum):
    """Semantic axis the univariate HMM is intended to describe."""

    DIRECTION = "direction"
    VOLATILITY = "volatility"
    EFFICIENCY = "efficiency"
    MOMENTUM = "momentum"
    LIQUIDITY = "liquidity"


class EmissionFamily(StrEnum):
    """Emission families supported by the univariate distiller."""

    LAMBDA = "lambda"
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"


class ModelSelectionMetric(StrEnum):
    """Metric used to pick the best fitted HMM candidate."""

    BIC = "bic"
    AIC = "aic"
    LOG_LIKELIHOOD = "log_likelihood"


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
    """Settings for fitting and distilling a univariate HMM.

    The distiller is exploratory by design: it inspects which univariate feature
    geometry produces coherent latent regimes, then emits static raw-feature
    thresholds that approximate the fitted HMM's state assignments.
    """

    axis_type: AxisType
    n_states_grid: tuple[int, ...] = (2, 3, 4)
    emission_families: tuple[EmissionFamily, ...] = (EmissionFamily.LAMBDA,)
    selection_metric: ModelSelectionMetric = ModelSelectionMetric.BIC
    state_ordering: StateOrdering = StateOrdering.FEATURE_MEDIAN
    threshold_method: ThresholdMethod = ThresholdMethod.POSTERIOR_MAP_SWITCH
    random_states: tuple[int | None, ...] = (100,)
    min_state_fraction: float = 0.01
    posterior_confidence_thresholds: tuple[float, ...] = (0.95, 0.99)
    emission_grid_size: int = 2048
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not self.n_states_grid:
            raise ValueError("n_states_grid must not be empty")
        if any(k < 2 for k in self.n_states_grid):
            raise ValueError(f"all n_states_grid values must be >= 2, got {self.n_states_grid}")
        if not self.emission_families:
            raise ValueError("emission_families must not be empty")
        if not self.random_states:
            raise ValueError("random_states must not be empty")
        if not 0.0 <= self.min_state_fraction < 1.0:
            raise ValueError(f"min_state_fraction must be in [0, 1), got {self.min_state_fraction}")
        if any((q < 0.0 or q > 1.0) for q in self.posterior_confidence_thresholds):
            raise ValueError("posterior_confidence_thresholds values must be in [0, 1]")
        if self.emission_grid_size < 128:
            raise ValueError(f"emission_grid_size must be >= 128, got {self.emission_grid_size}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")
