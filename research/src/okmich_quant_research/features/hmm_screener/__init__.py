"""
HMM Feature Screener
====================
Research-time tool for selecting feature subsets for an axis-specific HMM.

Sibling to ``okmich_quant_research.features.screener.FeatureScreener`` — same two-layer pattern (registry filters by
domain knowledge; screener filters empirically), specialised for HMMs whose features *define* the latent state structure
rather than predict a known target.

The axis is a ``registry.Axis``, not a ``signal_type``. ``Axis.DIRECTIONAL`` replaces the former
``trend`` and ``momentum`` keys: measured on FXPIG-Server M5, momentum is trend at half the lookback
(23 of 38 candidates within the ``MAX_VIF = 3.0`` near-duplicate line of some trend candidate, and the
redundancy tracks informativeness), so it was never a separate axis. Eligibility is decided by measured
invariance rather than by namespace -- see ``registry._axis``.

Quick start
-----------
    >>> from okmich_quant_research.features.registry import FeatureRegistry
    >>> from okmich_quant_research.features.hmm_screener import (
    ...     HmmFeatureScreener, HmmScreenerConfig, ScreenStrategy,
    ... )
    >>> reg = FeatureRegistry()
    >>> candidates = reg.candidates_for("regime", min_relevance="HIGH").names()
    >>> config = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=3)
    >>> screener = HmmFeatureScreener(config, raw_data, feature_engineering_fn)
    >>> result = screener.screen(candidates, strategy=ScreenStrategy.ABLATION)
    >>> result.results_       # ranked DataFrame
    >>> result.asymmetry_candidates  # Pareto-optimal non-trap subsets (Stage-1 candidates)
"""
from ..registry import Axis
from ._config import (HMM_ALGO_REGISTRY, HmmScreenerConfig, OneSidedPolicy, ScreenStrategy, build_hmm)
from ._evaluators import (
    AXIS_EVALUATORS,
    AxisEvaluator,
    evaluate_direction,
    evaluate_path_structure,
    evaluate_volatility,
    evaluate_liquidity,
    get_evaluator,
    primary_horizon_for,
)
from ._collinearity import nearest_duplicate_vif, stage0c_collinearity_filter
from ._pareto import ParetoStatus, classify_pareto
from ._persistence import adjacent_pair_count, persistence_score, stage0b_persistence_filter
from ._result import (AxisEvaluation, BaselinePrior, BaselineRole, GreedyStep, GreedyStopReason,
                      HmmScreenerResult, SubsetEvaluation, WinnerPool)
from .screener import HmmFeatureScreener

__all__ = [
    "HmmFeatureScreener",
    "HmmScreenerConfig",
    "Axis",
    "OneSidedPolicy",
    "primary_horizon_for",
    "HmmScreenerResult",
    "ScreenStrategy",
    "SubsetEvaluation",
    "AxisEvaluation",
    "BaselinePrior",
    "BaselineRole",
    "WinnerPool",
    "GreedyStep",
    "GreedyStopReason",
    "ParetoStatus",
    "classify_pareto",
    "AxisEvaluator",
    "AXIS_EVALUATORS",
    "get_evaluator",
    "evaluate_direction",
    "evaluate_volatility",
    "evaluate_path_structure",
    "evaluate_liquidity",
    "build_hmm",
    "HMM_ALGO_REGISTRY",
    "stage0b_persistence_filter",
    "persistence_score",
    "adjacent_pair_count",
    "stage0c_collinearity_filter",
    "nearest_duplicate_vif",
]
