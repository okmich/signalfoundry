"""Univariate HMM threshold distillation."""
from .config import (
    AxisType,
    EmissionFamily,
    ModelSelectionMetric,
    StateOrdering,
    ThresholdMethod,
    UnivariateHmmThresholdConfig,
)
from .distiller import UnivariateHmmThresholdDistiller
from .result import (
    CandidateFit,
    PairwiseSeparability,
    StateSummary,
    ThresholdBoundary,
    UnivariateHmmThresholdResult,
)
from .separability import build_pairwise_separability, build_state_summaries, empirical_overlap_coefficient

__all__ = [
    "AxisType",
    "EmissionFamily",
    "ModelSelectionMetric",
    "StateOrdering",
    "ThresholdMethod",
    "UnivariateHmmThresholdConfig",
    "UnivariateHmmThresholdDistiller",
    "CandidateFit",
    "PairwiseSeparability",
    "StateSummary",
    "ThresholdBoundary",
    "UnivariateHmmThresholdResult",
    "build_pairwise_separability",
    "build_state_summaries",
    "empirical_overlap_coefficient",
]
