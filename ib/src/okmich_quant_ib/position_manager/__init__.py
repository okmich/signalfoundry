"""Factory for IB position managers, keyed by ``PositionManagerType``."""
import logging

from ib_async import IB, Contract

from okmich_quant_core import PositionManagerType, StrategyConfig

from .atr_based_position_manager import ATRBasedIBPositionManager
from .base import BaseIBPositionManager
from .max_loss_amount_position_manager import MaxLossAmountIBPositionManager
from .percent_change_position_manager import PercentChangeIBPositionManager

logger = logging.getLogger(__name__)


_ATR_TYPES = {
    PositionManagerType.FIXED_ATR.value,
    PositionManagerType.FIXED_ATR_WITH_TRAILING.value,
    PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN.value,
    PositionManagerType.DYNAMIC_ATR.value,
}

_PERCENT_TYPES = {
    PositionManagerType.FIXED_PERCENT.value,
    PositionManagerType.FIXED_PERCENT_WITH_TRAILING.value,
    PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN.value,
    PositionManagerType.DYNAMIC_PERCENT.value,
}


def get_position_manager(ib: IB, contract: Contract, strategy_config: StrategyConfig,
                         **kwargs) -> BaseIBPositionManager:
    """Build the configured IB position manager.

    Concrete IB managers expect optional kwargs (e.g. ``price_buffer``,
    ``contract_info``); the strategy passes them through at bootstrap.
    """
    if strategy_config.position_manager is None:
        raise ValueError(
            f"StrategyConfig '{strategy_config.name}' has no position_manager configured."
        )
    ptype = strategy_config.position_manager.type.value

    if ptype in _ATR_TYPES:
        return ATRBasedIBPositionManager(ib, contract, strategy_config, **kwargs)
    if ptype in _PERCENT_TYPES:
        return PercentChangeIBPositionManager(ib, contract, strategy_config, **kwargs)
    if ptype == PositionManagerType.MAX_LOSS_AMOUNT.value:
        return MaxLossAmountIBPositionManager(ib, contract, strategy_config, **kwargs)
    raise ValueError(
        f"Unknown / unsupported position_manager type for IB: {ptype}. "
        f"Supported: ATR-* , PERCENT-*, MAX_LOSS_AMOUNT."
    )


__all__ = [
    "BaseIBPositionManager",
    "ATRBasedIBPositionManager",
    "PercentChangeIBPositionManager",
    "MaxLossAmountIBPositionManager",
    "get_position_manager",
]
