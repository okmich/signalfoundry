"""
Hierarchical HMM — macro-regime (Run / Reversal) x event-time microstructure.

Native pomegranate implementation: the 3-level Tayal/Wisebourt HHMM is reduced (Murphy 2002 DBN
flattening) to a flat 4-state ``DenseHMM`` with a masked transition matrix. EM preserves the hard
structural zeros, so all of ``BasePomegranateHMM`` (fixed-lag, forward-backward, drift log-lik,
serialisation) is reused and FILTER / FIXED_LAG / SMOOTHER, mixtures and pluggable emission
distributions come for free.
"""
from __future__ import annotations

from .config import (
    ALPHABET_SIZE,
    N_STATES,
    AssetGroup,
    AssetGroupConfig,
    FlowBucket,
    FlowFeatureKind,
    HHMMLevel,
    MacroRegime,
    PosteriorMode,
    RefitCadenceUnit,
    SessionPolicy,
    TrendStrength,
    ZigzagDirection,
    all_asset_group_configs,
    decode_symbol,
    encode_symbol,
    get_asset_group_config,
)
from .observations import (
    BOCPDFlowFeature,
    FlowFeature,
    MarketData,
    SignedVolumeFlowFeature,
    Zigzag,
    ZigzagObservationPipeline,
    ZigzagObservations,
    aggregate_zigzags,
    build_flow_feature,
    calibrate_k,
    events_per_hour,
    realized_vol,
    vol_scaled_threshold,
)
from .hhmm import (
    HierarchicalHMM,
    build_transition_mask,
    state_block,
    state_direction,
)
from .persistence import (
    HHMM_SCHEMA_VERSION,
    from_param_dict,
    load_model,
    load_param_dict,
    save_model,
    save_param_dict,
    to_param_dict,
)

__all__ = [
    # config
    "ALPHABET_SIZE",
    "N_STATES",
    "AssetGroup",
    "AssetGroupConfig",
    "FlowBucket",
    "FlowFeatureKind",
    "HHMMLevel",
    "MacroRegime",
    "PosteriorMode",
    "RefitCadenceUnit",
    "SessionPolicy",
    "TrendStrength",
    "ZigzagDirection",
    "all_asset_group_configs",
    "decode_symbol",
    "encode_symbol",
    "get_asset_group_config",
    # observations
    "BOCPDFlowFeature",
    "FlowFeature",
    "MarketData",
    "SignedVolumeFlowFeature",
    "Zigzag",
    "ZigzagObservationPipeline",
    "ZigzagObservations",
    "aggregate_zigzags",
    "build_flow_feature",
    "calibrate_k",
    "events_per_hour",
    "realized_vol",
    "vol_scaled_threshold",
    # hhmm
    "HierarchicalHMM",
    "build_transition_mask",
    "state_block",
    "state_direction",
    # persistence
    "HHMM_SCHEMA_VERSION",
    "from_param_dict",
    "to_param_dict",
    "save_param_dict",
    "load_param_dict",
    "save_model",
    "load_model",
]
