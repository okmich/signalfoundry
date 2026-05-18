"""
HMM Feature Screener
====================
Research-time tool for selecting feature subsets for an axis-specific HMM.

Sibling to ``okmich_quant_research.features.screener.FeatureScreener`` — same two-layer pattern (registry filters by
domain knowledge; screener filters empirically), specialised for HMMs whose features *define* the latent state structure
rather than predict a known target.

Quick start
-----------
    >>> from okmich_quant_research.features.registry import FeatureRegistry
    >>> from okmich_quant_research.features.hmm_screener import (
    ...     HmmFeatureScreener, HmmScreenerConfig, ScreenStrategy,
    ... )
    >>> reg = FeatureRegistry()
    >>> candidates = reg.candidates_for("regime", min_relevance="HIGH").names()
    >>> config = HmmScreenerConfig(signal_type="trend", algo="hmm_lambda", n_states=4)
    >>> screener = HmmFeatureScreener(config, raw_data, feature_engineering_fn)
    >>> result = screener.screen(candidates, strategy=ScreenStrategy.ABLATION)
    >>> result.results_       # ranked DataFrame
    >>> result.keepers        # Pareto-optimal non-trap subsets
"""
from ._config import HMM_ALGO_REGISTRY, HmmScreenerConfig, ScreenStrategy, build_hmm
from ._evaluators import (
    AXIS_EVALUATORS,
    AxisEvaluator,
    evaluate_direction,
    evaluate_momentum,
    evaluate_path_structure,
    evaluate_volatility,
    evaluate_liquidity,
    get_evaluator,
)
from ._pareto import ParetoStatus, classify_pareto
from ._result import AxisEvaluation, HmmScreenerResult, SubsetEvaluation
from .screener import HmmFeatureScreener

__all__ = [
    "HmmFeatureScreener",
    "HmmScreenerConfig",
    "HmmScreenerResult",
    "ScreenStrategy",
    "SubsetEvaluation",
    "AxisEvaluation",
    "ParetoStatus",
    "classify_pareto",
    "AxisEvaluator",
    "AXIS_EVALUATORS",
    "get_evaluator",
    "evaluate_direction",
    "evaluate_momentum",
    "evaluate_volatility",
    "evaluate_path_structure",
    "evaluate_liquidity",
    "build_hmm",
    "HMM_ALGO_REGISTRY"
]
