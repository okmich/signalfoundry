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


class NullMethod(enum.StrEnum):
    """How the permutation null breaks the feature-target link when estimating the IC noise floor."""

    CIRCULAR_SHIFT = "circular_shift"  # roll the target in time: keeps its autocorrelation, so persistence is priced in
    IID_SHUFFLE = "iid_shuffle"  # shuffle the target bar by bar: too narrow when feature and target are persistent
