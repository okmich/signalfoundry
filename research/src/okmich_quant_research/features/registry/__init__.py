"""
Feature Registry
================
Queryable catalog of all feature-computing functions in okmich_quant_features.

    >>> from okmich_quant_research.features.registry import FeatureRegistry
    >>> reg = FeatureRegistry()
    >>> reg.candidates_for("regime", min_relevance="HIGH").names()

Two taxonomies live here and they are NOT the same thing:

  * ``SIGNAL_TYPES`` — a 12-value hand-applied FEATURE TAG ("what family is this indicator from").
  * ``Axis``         — the 4-value SCREENING AXIS a latent-state model is asked to separate, whose
                       membership is decided by MEASURED invariance (``Parity`` / ``ScaleClass``)
                       rather than by a name. See ``_axis``.
"""
from ._axis import (
    AXIS_PRIMARY_HORIZON,
    AXIS_SIGNAL_TYPES,
    Axis,
    PRICE_PATH_AXES,
    is_eligible,
)
from ._invariance import INVARIANCE_STAMPS, load_invariance_stamps
from ._schema import (
    FeatureEntry,
    FeatureInvariance,
    Parity,
    ScaleClass,
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
from ._catalog import CATALOG, UNSTAMPED_MEASUREMENTS
from .registry import FeatureRegistry

__all__ = [
    "FeatureRegistry",
    "FeatureEntry",
    "CATALOG",
    "SIGNAL_TYPES",
    # ── axis layer ────────────────────────────────────────────────────────────
    "Axis",
    "AXIS_SIGNAL_TYPES",
    "AXIS_PRIMARY_HORIZON",
    "PRICE_PATH_AXES",
    "is_eligible",
    # ── measured invariance ───────────────────────────────────────────────────
    "Parity",
    "ScaleClass",
    "FeatureInvariance",
    "INVARIANCE_STAMPS",
    "UNSTAMPED_MEASUREMENTS",
    "load_invariance_stamps",
    "RELEVANCE_LEVELS",
    "HORIZONS",
    "MARKET_REGIMES",
    "CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE",
    "H_INTRADAY", "H_SHORT", "H_MEDIUM", "H_LONG", "H_ANY",
    "R_TRENDING", "R_RANGING", "R_VOLATILE", "R_LOW_VOL", "R_CRISIS",
]