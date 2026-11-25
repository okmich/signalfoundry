"""Paper presets ("flavours") built on the general HHMM engine.

- ``TayalHHMM`` (Tayal 2009): bull / bear regimes over zigzag legs from direction x structure (x flow) symbols.
- ``joint_multiscale``: a slow backdrop state over a fast state on bar features, learned jointly.
- ``embedded_hhmm`` + ``block_schedule`` (Adam et al. 2019): a coarse state per day owning an intraday HMM.
"""
from .bars import BlockUnit, EmissionFamily, block_ids, block_schedule, embedded_hhmm, joint_multiscale
from .tayal import BarLayout, FlowInput, RegimeLabel, TayalHHMM, TrendInput, tayal_features, tayal_tree

__all__ = [
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
    "tayal_features",
    "tayal_tree",
]
