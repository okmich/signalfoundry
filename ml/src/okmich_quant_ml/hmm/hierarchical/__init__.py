"""
Hierarchical hidden Markov models.

- ``engine``: a general HHMM. Any state tree (entry, horizontal and exit parameters per internal node, structural
  zeros anywhere), exact inference by flattening onto the leaves, EM on the tree's own parameters, exit schedules
  (forbid / force exits per depth and bar), categorical / Gaussian / Student-t emissions tied at any level, and
  causal (filter, online Viterbi) and hindsight (smoother, fixed-lag, Viterbi) decoders.
- ``inputs``: observation builders. Zigzag legs of a configurable size (``LegThreshold``) and per-leg features
  (Tayal's structure and volume, leg size, BOCPD break intensity, signed volume) combined by a ``SymbolEncoder``.
- ``flavours``: paper presets on the engine. ``TayalHHMM`` (Tayal 2009), ``joint_multiscale`` and ``embedded_hhmm``
  (Adam et al. 2019).

Every model here duck-types ``BasePomegranateHMM`` (``predict_proba``, ``predict_proba_fixed_lag``,
``compute_per_bar_predictive_loglik``, ``save`` / ``load`` ...) with the leaves as its states, so it plugs into
``HmmModelWrapper`` and the posterior-inference stack; ``*_level`` methods read any tree level.
"""
from __future__ import annotations

from .engine import (
    HHMM,
    CategoricalEmission,
    CovarianceType,
    ExitMode,
    ExitSchedule,
    FitReport,
    GaussianEmission,
    LeafEmission,
    StateNode,
    StudentTEmission,
    Topology,
)
from .flavours import (
    BarLayout,
    BlockUnit,
    EmissionFamily,
    FlowInput,
    RegimeLabel,
    TayalHHMM,
    TrendInput,
    block_ids,
    block_schedule,
    embedded_hhmm,
    joint_multiscale,
    tayal_tree,
)
from .inputs import (
    LegThreshold,
    MarketData,
    SymbolEncoder,
    ThresholdMode,
    Zigzag,
    ZigzagDirection,
    ZigzagLegBuilder,
    aggregate_zigzags,
    calibrate_k,
    events_per_hour,
    leg_feature_frame,
    legs_confirmed_from,
    legs_to_bars,
    observed_hours,
    realized_vol,
    vol_scaled_threshold,
)

__all__ = [
    # engine
    "HHMM",
    "CategoricalEmission",
    "CovarianceType",
    "ExitMode",
    "ExitSchedule",
    "FitReport",
    "GaussianEmission",
    "LeafEmission",
    "StateNode",
    "StudentTEmission",
    "Topology",
    # inputs
    "LegThreshold",
    "MarketData",
    "SymbolEncoder",
    "ThresholdMode",
    "Zigzag",
    "ZigzagDirection",
    "ZigzagLegBuilder",
    "aggregate_zigzags",
    "calibrate_k",
    "events_per_hour",
    "leg_feature_frame",
    "legs_confirmed_from",
    "legs_to_bars",
    "observed_hours",
    "realized_vol",
    "vol_scaled_threshold",
    # flavours
    "BarLayout",
    "BlockUnit",
    "EmissionFamily",
    "FlowInput",
    "RegimeLabel",
    "TayalHHMM",
    "TrendInput",
    "block_ids",
    "block_schedule",
    "embedded_hhmm",
    "joint_multiscale",
    "tayal_tree",
]
