import logging

from .atr_based_position_manager import (
    DynamicAtrBasedPositionManager,
    FixedAtrSlTpWithSingleBreakEvenAtrBasedPositionManager,
    FixedAtrSlTpWithTrailingAtrBasedPositionManager,
    FixedAtrBasedPositionManager,
)
from .base import MaxLossAmountPositionManager, MaxLossStopLossPositionManager
from .percent_change_position_manager import (
    DynamicPercentBasedPositionManager,
    FixedPercentBasedPositionManager,
    FixedPercentSlTpWithTrailingPercentBasedPositionManager,
    FixedPercentSlTpWithSingleBreakEvenPercentBasedPositionManager,
)
from .point_position_manager import (
    FixedPointBasedPositionManager,
    FixedSlTpWithTrailingPointBasedPositionManager,
    FixedSlTpWithSingleBreakEvenPointBasedPositionManager,
    DynamicPointBasedPositionManager,
)
from okmich_quant_core import PositionManagerType, StrategyConfig


def get_position_manager(strategy_config: StrategyConfig, **kwargs):
    # Mapping of manager types to their classes
    _manager_classes = {
        # Point-Based Managers
        PositionManagerType.FIXED_POINT.value: FixedPointBasedPositionManager,
        PositionManagerType.FIXED_POINT_WITH_TRAILING.value: FixedSlTpWithTrailingPointBasedPositionManager,
        PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN.value: FixedSlTpWithSingleBreakEvenPointBasedPositionManager,
        PositionManagerType.DYNAMIC_POINT.value: DynamicPointBasedPositionManager,
        # Percent-Based Managers
        PositionManagerType.FIXED_PERCENT.value: FixedPercentBasedPositionManager,
        PositionManagerType.FIXED_PERCENT_WITH_TRAILING.value: FixedPercentSlTpWithTrailingPercentBasedPositionManager,
        PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN.value: FixedPercentSlTpWithSingleBreakEvenPercentBasedPositionManager,
        PositionManagerType.DYNAMIC_PERCENT.value: DynamicPercentBasedPositionManager,
        # ATR-Based Managers
        PositionManagerType.FIXED_ATR.value: FixedAtrBasedPositionManager,
        PositionManagerType.FIXED_ATR_WITH_TRAILING.value: FixedAtrSlTpWithTrailingAtrBasedPositionManager,
        PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN.value: FixedAtrSlTpWithSingleBreakEvenAtrBasedPositionManager,
        PositionManagerType.DYNAMIC_ATR.value: DynamicAtrBasedPositionManager,
        # Loss amount Managers
        PositionManagerType.MAX_LOSS_AMOUNT.value: MaxLossAmountPositionManager,
        PositionManagerType.MAX_LOSS_STOP_LOSS.value: MaxLossStopLossPositionManager,
    }

    """
    Factory method to create a specific type of position manager

    Args:
        manager_type (PositionManagerType): The type of manager to create
        position_manager_config (dict): Configuration for the manager
        **kwargs: Additional arguments (symbol, magic, system_name, etc.)

    Returns:
        BaseMt5PositionManager: Instance of the requested manager type

    Raises:
        ValueError: If manager_type is not supported
        Exception: If manager creation fails
    """
    position_manager_config = strategy_config.position_manager
    if position_manager_config is None:
        return None

    if position_manager_config.type.value not in _manager_classes:
        raise ValueError(f"Unsupported position manager type: {position_manager_config.type}")
    try:
        manager_class = _manager_classes[position_manager_config.type.value]
        return manager_class(strategy_config, **kwargs)
    except Exception as e:
        logging.error(f"Failed to create position manager of type {position_manager_config.type}: {e}")
        raise
