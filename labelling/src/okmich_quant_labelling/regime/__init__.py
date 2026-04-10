"""
Approach 1 — Causal Regime Detection

Data-driven regime detection using causal features only.
All labelers/optimizers in this module are valid for live trading (leaks_future=False).
"""

from .threshold_optimizer import (
    CausalLabelThresholdOptimizer,
    MarketPropertyType,
    SeparationMetric,
    OptimizationObjective,
)
from .causal_regime_labeler import CausalRegimeLabeler
from .label_generator import (
    RegimeLabelGenerator,
    HmmDirectStrategy,
    OracleDistillationStrategy,
    HmmViterbiDistillationStrategy,
    CausalStrategy,
    create_label_generator,
)

__all__ = [
    "CausalLabelThresholdOptimizer",
    "MarketPropertyType",
    "SeparationMetric",
    "OptimizationObjective",
    "CausalRegimeLabeler",
    # label_generator sub-package
    "RegimeLabelGenerator",
    "HmmDirectStrategy",
    "OracleDistillationStrategy",
    "HmmViterbiDistillationStrategy",
    "CausalStrategy",
    "create_label_generator",
]

