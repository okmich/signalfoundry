from __future__ import annotations

import enum


class FeatureStatus(enum.StrEnum):
    """Status of a feature within a single condition cell."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    NEGATIVE = "negative"
    INSUFFICIENT_DATA = "insufficient_data"


class FeatureBucket(enum.StrEnum):
    """Partition bucket a feature is assigned to."""

    GLOBAL_STABLE = "global_stable"
    CONDITION_SPECIFIC = "condition_specific"
    CONDITIONAL_ENSEMBLE = "conditional_ensemble"
    UNCLASSIFIED = "unclassified"


class ConditionPass(enum.StrEnum):
    """Which conditioning dimension produced a FeatureConditionMap."""

    REGIME = "regime"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"