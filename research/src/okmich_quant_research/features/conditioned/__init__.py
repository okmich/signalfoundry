"""
Conditional feature analysis — answers "where does this feature work?"

Computes regime-stratified and time-stratified Spearman IC to build a feature-condition map: which features are
predictive under which conditions.
"""

from ._enums import ConditionPass, FeatureBucket, FeatureStatus
from ._result import FeatureConditionMap
from .analyzer import ConditionalFeatureAnalyzer
from .plots import ConditionedFeatureVisualizer

__all__ = [
    "ConditionalFeatureAnalyzer",
    "ConditionedFeatureVisualizer",
    "ConditionPass",
    "FeatureBucket",
    "FeatureConditionMap",
    "FeatureStatus",
]