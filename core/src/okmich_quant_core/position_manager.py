import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime

from .config import StrategyConfig

logger = logging.getLogger(__name__)


class BasePositionManager(ABC):
    """
    Abstract base class for position managers.

    Provides a template for managing open positions across different brokers.
    Subclasses must implement broker-specific operations.
    """

    def __init__(self, strategy_config: StrategyConfig):
        if strategy_config.position_manager is None:
            raise ValueError(
                f"StrategyConfig for '{strategy_config.name}' has no position_manager configured. "
                "Set strategy_config.position_manager to a PositionManagerConfig before using "
                "a BasePositionManager subclass."
            )
        self.symbol = strategy_config.symbol
        self.magic = strategy_config.magic
        self.timeframe = strategy_config.timeframe
        self.strategy_name = strategy_config.name
        self.config = strategy_config.position_manager

        logger.info(
            f"Position manager initialized for {self.symbol} (magic: {self.magic}, type: {self.config.type.value})"
        )

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def close_position(self, position: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def modify_position(self, position: Dict[str, Any], sl: float = None, tp: float = None) -> bool:
        pass

    @abstractmethod
    def manage_long_position(self, position: Dict[str, Any], flag: bool):
        pass

    @abstractmethod
    def manage_short_position(self, position: Dict[str, Any], flag: bool):
        pass

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> int:
        try:
            positions = self.get_open_positions()

            if positions is None:
                logger.error(f"Failed to get positions for {self.symbol} ({self.magic})")
                return -1

            pos_count = len(positions)
            if pos_count == 0:
                return 0

            # Manage each position
            for position in positions:
                position_type = position.get("type")
                if position_type == 0:  # Long position
                    try:
                        self.manage_long_position(position, flag)
                    except Exception as e:
                        logger.error(f"Error managing long position {position.get('ticket')}: {e}", exc_info=True)

                elif position_type == 1:  # Short position
                    try:
                        self.manage_short_position(position, flag)
                    except Exception as e:
                        logger.error(f"Error managing short position {position.get('ticket')}: {e}", exc_info=True)
            # Return updated position count
            final_positions = self.get_open_positions()
            return len(final_positions) if final_positions is not None else pos_count
        except Exception as e:
            logger.error(f"Unexpected error in manage_positions for {self.symbol}: {e}", exc_info=True)
            return -1
