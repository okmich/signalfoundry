"""
Feature Registry
================
Queryable catalog of all feature-computing functions in okmich_quant_features.

    >>> from okmich_quant_research.features.registry import FeatureRegistry
    >>> reg = FeatureRegistry()
    >>> reg.candidates_for("regime", min_relevance="HIGH").names()
"""
from ._schema import (
    FeatureEntry,
    SIGNAL_TYPES,
    RELEVANCE_LEVELS,
    HORIZONS,
    MARKET_REGIMES,
    CRITICAL,
    HIGH,
    MEDIUM,
    LOW,
    NONE,
    H_INTRADAY,
    H_SHORT,
    H_MEDIUM,
    H_LONG,
    H_ANY,
    R_TRENDING,
    R_RANGING,
    R_VOLATILE,
    R_LOW_VOL,
    R_CRISIS,
)
from ._catalog import CATALOG
from .registry import FeatureRegistry

__all__ = [
    "FeatureRegistry",
    "FeatureEntry",
    "CATALOG",
    "SIGNAL_TYPES",
    "RELEVANCE_LEVELS",
    "HORIZONS",
    "MARKET_REGIMES",
    "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE",
    "H_INTRADAY", "H_SHORT", "H_MEDIUM", "H_LONG", "H_ANY",
    "R_TRENDING", "R_RANGING", "R_VOLATILE", "R_LOW_VOL", "R_CRISIS",
]