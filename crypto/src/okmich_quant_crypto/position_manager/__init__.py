"""Factory for crypto position managers, keyed by ``PositionManagerType`` (every MT5 type is supported)."""
from okmich_quant_core import PositionManagerType, StrategyConfig

from .atr_based_position_manager import AtrBasedPositionManager, wilder_atr
from .base import (
    BaseCryptoPositionManager, DistanceBasedPositionManager, MaxLossAmountPositionManager,
    MaxLossStopLossPositionManager,
)
from .percent_change_position_manager import PercentBasedPositionManager
from .point_position_manager import PointBasedPositionManager

_POINT = {PositionManagerType.FIXED_POINT, PositionManagerType.FIXED_POINT_WITH_TRAILING,
          PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN, PositionManagerType.DYNAMIC_POINT}
_PERCENT = {PositionManagerType.FIXED_PERCENT, PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
            PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN, PositionManagerType.DYNAMIC_PERCENT}
_ATR = {PositionManagerType.FIXED_ATR, PositionManagerType.FIXED_ATR_WITH_TRAILING,
        PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN, PositionManagerType.DYNAMIC_ATR}


def get_position_manager(strategy_config: StrategyConfig, **kwargs) -> BaseCryptoPositionManager | None:
    """Build the configured manager, or ``None`` when the strategy has none. ``price_buffer`` is passed by the
    strategy at bootstrap (ATR managers need it)."""
    cfg = strategy_config.position_manager
    if cfg is None:
        return None
    if cfg.type in _POINT:
        return PointBasedPositionManager(strategy_config, **kwargs)
    if cfg.type in _PERCENT:
        return PercentBasedPositionManager(strategy_config, **kwargs)
    if cfg.type in _ATR:
        return AtrBasedPositionManager(strategy_config, **kwargs)
    if cfg.type is PositionManagerType.MAX_LOSS_AMOUNT:
        return MaxLossAmountPositionManager(strategy_config, **kwargs)
    if cfg.type is PositionManagerType.MAX_LOSS_STOP_LOSS:
        return MaxLossStopLossPositionManager(strategy_config, **kwargs)
    raise ValueError(f"unsupported position manager type for crypto: {cfg.type.value}")


__all__ = ["BaseCryptoPositionManager", "DistanceBasedPositionManager", "PointBasedPositionManager",
           "PercentBasedPositionManager", "AtrBasedPositionManager", "MaxLossAmountPositionManager",
           "MaxLossStopLossPositionManager", "get_position_manager", "wilder_atr"]
