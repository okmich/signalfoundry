"""
Configuration contract for the Hierarchical HMM (macro-regime x event-time microstructure).

This module holds the *shared vocabulary* of the HHMM: the enums that name every fixed
set of values, the 18-symbol discrete observation alphabet (with its encode/decode helpers
and structural layout constants), and the per-asset-group configuration dataclass with the
four published group defaults.

Design notes
------------
- Topology is fixed (not parameterised): 2 macro blocks x 2 production states = 4 flat
  states. See ``hhmm.py`` for the masked-transition DBN reduction. The constants here are
  the single source of truth for that layout so ``observations.py``, ``hhmm.py`` and
  ``persistence.py`` never disagree about symbol/state indexing.
- The alphabet is ``ZigzagDirection (2) x TrendStrength (3) x FlowBucket (3) = 18`` symbols.
  Encoding is ``direction * 9 + strength * 3 + flow`` so the low 9 symbols (0..8) are the
  down sub-alphabet and the high 9 (9..17) are the up sub-alphabet. Production state P+
  emits over the up sub-alphabet; P- over the down sub-alphabet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from enum import IntEnum, StrEnum

import pandas as pd

# ----------------------------------------------------------------------------------------
# Fixed structural layout (do not parameterise — see plan "Topology (fixed)")
# ----------------------------------------------------------------------------------------
N_MACRO_STATES: int = 2                 # Run, Reversal (labels assigned post-hoc)
N_PRODUCTION_PER_MACRO: int = 2         # P+, P- within each macro block
N_STATES: int = N_MACRO_STATES * N_PRODUCTION_PER_MACRO  # 4 flat production states

N_DIRECTIONS: int = 2                   # up, down
N_STRENGTH_BINS: int = 3                # weak, normal, strong
N_FLOW_BINS: int = 3                    # against, neutral, with
SUB_ALPHABET_SIZE: int = N_STRENGTH_BINS * N_FLOW_BINS  # 9 symbols per direction
ALPHABET_SIZE: int = N_DIRECTIONS * SUB_ALPHABET_SIZE   # 18 discrete symbols

TOPOLOGY_NAME: str = "wisebourt_run_reversal_v1"


# ----------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------
class HHMMLevel(StrEnum):
    """Level at which a posterior is requested."""

    MACRO = "macro"            # Run / Reversal, shape (T, 2)
    PRODUCTION = "production"  # per production state, shape (T, 4)


class PosteriorMode(StrEnum):
    """
    Inference recursion for posteriors. Causal-first (2026-07-15 alignment):

    - FILTER    : lag-0 causal forward filtering. The ONLY mode a live strategy may consume.
    - FIXED_LAG : condition on ``lag`` future zigzags. Look-ahead -> teacher-label / maturation
                  diagnostic only, never live.
    - SMOOTHER  : full forward-backward over the whole sequence. Look-ahead -> teacher /
                  diagnostic only.
    """

    FILTER = "filter"
    FIXED_LAG = "fixed_lag"
    SMOOTHER = "smoother"


class MacroRegime(StrEnum):
    """Macro regime labels assigned post-hoc to the two learned macro blocks."""

    RUN = "run"            # trending
    REVERSAL = "reversal"  # mean-reverting


class ZigzagDirection(IntEnum):
    """Direction component of a zigzag symbol. Value doubles as the alphabet sub-index."""

    DOWN = 0
    UP = 1


class TrendStrength(IntEnum):
    """Trend-strength bucket: ``|zigzag magnitude| / sigma_realized`` bucketed."""

    WEAK = 0
    NORMAL = 1
    STRONG = 2


class FlowBucket(IntEnum):
    """Flow-support bucket relative to the zigzag's price direction."""

    AGAINST = 0
    NEUTRAL = 1
    WITH = 2


class FlowFeatureKind(StrEnum):
    """Which ``FlowFeature`` implementation supplies the flow-support component."""

    BOCPD = "bocpd"                  # default everywhere: accumulated changepoint posterior
    SIGNED_VOLUME = "signed_volume"  # opt-in where volume is trustworthy


class SessionPolicy(StrEnum):
    """Per-asset session handling, applied as an inference-time gate (not a feature)."""

    HARD = "hard"              # mandatory session breaks; no inference across breaks
    SOFT = "soft"             # downweight confidence in low-liquidity windows
    CONTINUOUS = "continuous"  # no breaks; optional weekend-mode wider threshold


class RefitCadenceUnit(StrEnum):
    """Unit in which a group's rolling-refit cadence is expressed."""

    ZIGZAG_DAYS = "zigzag_days"
    SESSIONS = "sessions"


class AssetGroup(StrEnum):
    """Per-asset-group fitting units. Per-group fit, not per-instrument."""

    FX_MAJORS = "fx_majors"
    CRYPTO_CFD = "crypto_cfd"
    INDEXES = "indexes"
    XAU = "xau"


# ----------------------------------------------------------------------------------------
# Symbol alphabet encode / decode
# ----------------------------------------------------------------------------------------
def encode_symbol(direction: ZigzagDirection, strength: TrendStrength, flow: FlowBucket) -> int:
    """Encode the 3 discrete components into a single symbol index in ``[0, 18)``."""
    return int(direction) * SUB_ALPHABET_SIZE + int(strength) * N_FLOW_BINS + int(flow)


def decode_symbol(symbol: int) -> tuple[ZigzagDirection, TrendStrength, FlowBucket]:
    """Inverse of :func:`encode_symbol`."""
    if not 0 <= symbol < ALPHABET_SIZE:
        raise ValueError(f"symbol must be in [0, {ALPHABET_SIZE}), got {symbol}")
    direction, rem = divmod(int(symbol), SUB_ALPHABET_SIZE)
    strength, flow = divmod(rem, N_FLOW_BINS)
    return ZigzagDirection(direction), TrendStrength(strength), FlowBucket(flow)


def direction_of_symbol(symbol: int) -> ZigzagDirection:
    """Direction sub-index of a symbol (cheap; avoids full decode)."""
    if not 0 <= symbol < ALPHABET_SIZE:
        raise ValueError(f"symbol must be in [0, {ALPHABET_SIZE}), got {symbol}")
    return ZigzagDirection(int(symbol) // SUB_ALPHABET_SIZE)


def symbols_for_direction(direction: ZigzagDirection) -> range:
    """The 9-symbol sub-alphabet a production state of the given direction may emit over."""
    lo = int(direction) * SUB_ALPHABET_SIZE
    return range(lo, lo + SUB_ALPHABET_SIZE)


# ----------------------------------------------------------------------------------------
# Asset-group configuration
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AssetGroupConfig:
    """
    Per-asset-group HHMM configuration.

    One fitted model is shared across the instruments in a group (EUR/USD and GBP/USD share;
    SPX-CFD and NDX-CFD share; XAU stands alone in its group).

    Parameters
    ----------
    group
        Which asset group this config parameterises.
    k
        Initial directional-change threshold multiplier: ``threshold = k * sigma_realized``.
        Calibrated per group to hit ``target_events_per_hour`` on representative data via
        :func:`observations.calibrate_k`.
    target_events_per_hour
        ``(low, high)`` band of zigzag events/hour the k-calibration targets.
    realized_vol_window
        Lookback for the slow rolling realised-vol estimate ``sigma_realized`` (~24h for
        FX/crypto; session-length for indexes/XAU).
    flow_feature
        Default flow-support implementation. BOCPD everywhere; signed-volume opt-in.
    session_policy
        Session handling mode (HARD / SOFT / CONTINUOUS).
    refit_cadence
        ``(low, high)`` rolling-refit cadence band, in units of ``refit_cadence_unit``.
    refit_cadence_unit
        Whether the cadence band counts zigzag-days or sessions.
    low_liquidity_windows
        SOFT policy only: GMT ``(start, end)`` windows whose production-state posterior
        confidence is downweighted. Empty for non-SOFT groups.
    soft_downweight
        SOFT policy only: multiplicative confidence factor applied inside low-liquidity
        windows (``0 < f <= 1``).
    weekend_mode
        CONTINUOUS policy only: if True, widen the zigzag threshold on weekends to reduce
        thin-market noise.
    weekend_threshold_multiplier
        CONTINUOUS policy only: factor applied to ``k`` during weekend mode (``>= 1``).
    """

    group: AssetGroup
    k: float
    target_events_per_hour: tuple[float, float]
    realized_vol_window: pd.Timedelta
    flow_feature: FlowFeatureKind
    session_policy: SessionPolicy
    refit_cadence: tuple[int, int]
    refit_cadence_unit: RefitCadenceUnit
    low_liquidity_windows: tuple[tuple[time, time], ...] = ()
    soft_downweight: float = 1.0
    weekend_mode: bool = False
    weekend_threshold_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        lo, hi = self.target_events_per_hour
        if not 0 < lo <= hi:
            raise ValueError(f"target_events_per_hour must satisfy 0 < low <= high, got {self.target_events_per_hour}")
        if self.realized_vol_window <= pd.Timedelta(0):
            raise ValueError(f"realized_vol_window must be positive, got {self.realized_vol_window}")
        r_lo, r_hi = self.refit_cadence
        if not 0 < r_lo <= r_hi:
            raise ValueError(f"refit_cadence must satisfy 0 < low <= high, got {self.refit_cadence}")
        if not 0.0 < self.soft_downweight <= 1.0:
            raise ValueError(f"soft_downweight must be in (0, 1], got {self.soft_downweight}")
        if self.weekend_threshold_multiplier < 1.0:
            raise ValueError(f"weekend_threshold_multiplier must be >= 1, got {self.weekend_threshold_multiplier}")
        if self.session_policy is not SessionPolicy.SOFT and self.low_liquidity_windows:
            raise ValueError("low_liquidity_windows is only meaningful for SessionPolicy.SOFT")


# The four published group defaults (plan "Asset-group configuration").
# k is an initial value; calibrate_k refines it per representative data.
_FX_MAJORS = AssetGroupConfig(
    group=AssetGroup.FX_MAJORS,
    k=1.5,
    target_events_per_hour=(3.0, 8.0),
    realized_vol_window=pd.Timedelta("24h"),
    flow_feature=FlowFeatureKind.BOCPD,
    session_policy=SessionPolicy.SOFT,
    refit_cadence=(5, 10),
    refit_cadence_unit=RefitCadenceUnit.ZIGZAG_DAYS,
    low_liquidity_windows=((time(22, 0), time(6, 0)),),  # ~22:00-06:00 GMT thin window
    soft_downweight=0.5,
)

_CRYPTO_CFD = AssetGroupConfig(
    group=AssetGroup.CRYPTO_CFD,
    k=1.5,
    target_events_per_hour=(3.0, 8.0),
    realized_vol_window=pd.Timedelta("24h"),
    flow_feature=FlowFeatureKind.BOCPD,
    session_policy=SessionPolicy.CONTINUOUS,
    refit_cadence=(5, 10),
    refit_cadence_unit=RefitCadenceUnit.ZIGZAG_DAYS,
    weekend_mode=True,
    weekend_threshold_multiplier=1.5,
)

_INDEXES = AssetGroupConfig(
    group=AssetGroup.INDEXES,
    k=1.5,
    target_events_per_hour=(3.0, 8.0),
    realized_vol_window=pd.Timedelta("8h"),  # session-length
    flow_feature=FlowFeatureKind.BOCPD,
    session_policy=SessionPolicy.HARD,
    refit_cadence=(5, 10),
    refit_cadence_unit=RefitCadenceUnit.SESSIONS,
)

_XAU = AssetGroupConfig(
    group=AssetGroup.XAU,
    k=1.5,
    target_events_per_hour=(3.0, 8.0),
    realized_vol_window=pd.Timedelta("8h"),  # session-length
    flow_feature=FlowFeatureKind.BOCPD,
    session_policy=SessionPolicy.HARD,
    refit_cadence=(5, 10),
    refit_cadence_unit=RefitCadenceUnit.SESSIONS,
)

_ASSET_GROUP_CONFIGS: dict[AssetGroup, AssetGroupConfig] = {
    AssetGroup.FX_MAJORS: _FX_MAJORS,
    AssetGroup.CRYPTO_CFD: _CRYPTO_CFD,
    AssetGroup.INDEXES: _INDEXES,
    AssetGroup.XAU: _XAU,
}


def get_asset_group_config(group: AssetGroup) -> AssetGroupConfig:
    """Return the published default config for an asset group."""
    group = AssetGroup(group)
    return _ASSET_GROUP_CONFIGS[group]


def all_asset_group_configs() -> dict[AssetGroup, AssetGroupConfig]:
    """Return a copy of the full asset-group config registry."""
    return dict(_ASSET_GROUP_CONFIGS)
