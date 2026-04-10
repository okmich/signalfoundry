import logging
import math
from typing import List, Dict, Any

import MetaTrader5 as mt5

from ..functions import close_position as mt5_close_position, get_positions, modify_position as mt5_modify_position
from okmich_quant_core import StrategyConfig, BasePositionManager

logger = logging.getLogger(__name__)


class BaseMt5PositionManager(BasePositionManager):

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config)
        # MT5-specific attributes can be added here if needed
        self.kwargs = kwargs

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions from MT5.

        Returns:
            List of position dictionaries, None on error
        """
        try:
            positions = get_positions(self.symbol, self.magic)
            if positions is None:
                error = mt5.last_error()
                logger.error(f"MT5 error getting positions for {self.symbol} ({self.magic}): {error}")
                return None
            return positions
        except Exception as e:
            logger.error(f"Error getting positions for {self.symbol} ({self.magic}): {e}")
            return None

    def close_position(self, position: Dict[str, Any]) -> bool:
        """
        Close position via MT5.

        Args:
            position: Position dictionary with 'ticket' key

        Returns:
            True if successful, False otherwise
        """
        ticket = position.get("ticket")
        if ticket is None:
            logger.error(f"Position has no ticket: {position}")
            return False

        try:
            if mt5_close_position(ticket):
                logger.info(f"Closed position {ticket} for {self.symbol} ({self.magic})")
                return True
            else:
                logger.error(f"Failed to close position {ticket} for {self.symbol} ({self.magic})")
                return False
        except Exception as e:
            logger.error(f"Error closing position {ticket} for {self.symbol} ({self.magic}): {e}")
            return False

    def modify_position(self, position: Dict[str, Any], sl: float = None, tp: float = None) -> bool:
        """
        Modify position SL/TP via MT5. All subclasses must use this method
        to set or update SL/TP — never call mt5_modify_position directly.

        Args:
            position: Position dictionary with 'ticket' key
            sl: New stop loss price (None or 0 = keep current)
            tp: New take profit price (None or 0 = keep current)

        Returns:
            True if successful, False otherwise
        """
        ticket = position.get("ticket")
        if ticket is None:
            logger.error(f"Position has no ticket: {position}")
            return False

        try:
            mt5_modify_position(ticket, sl=sl if sl is not None else 0.0, tp=tp if tp is not None else 0.0)
            return True
        except Exception as e:
            logger.error(f"Failed to modify position {ticket} for {self.symbol} ({self.magic}): {e}")
            return False



class MaxLossAmountPositionManager(BaseMt5PositionManager):
    """
    Position manager that closes positions when loss exceeds a threshold.
    """

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.max_loss_amount = strategy_config.position_manager.max_loss_amount

    def manage_long_position(self, position: Dict[str, Any], flag: bool):
        """Manage long position - close if loss exceeds threshold."""
        self._close_on_max_amount(position)

    def manage_short_position(self, position: Dict[str, Any], flag: bool):
        """Manage short position - close if loss exceeds threshold."""
        self._close_on_max_amount(position)

    def _close_on_max_amount(self, position: Dict[str, Any]):
        """Close position if profit loss exceeds max_loss_amount."""
        ticket = position.get("ticket")
        profit = position.get("profit", 0)

        if profit < 0 and math.fabs(profit) >= self.max_loss_amount:
            if self.close_position(position):
                logger.info(f"Closed position {ticket} due to max loss: ${profit:.2f} >= ${self.max_loss_amount:.2f}")
            else:
                logger.error(f"Failed to close position {ticket} despite max loss: ${profit:.2f}")


class MaxLossStopLossPositionManager(BaseMt5PositionManager):
    """
    Position manager that reverse-engineers a stop loss price from a maximum loss amount in account currency.

    Instead of specifying SL in pips/points, you specify how much you are willing to lose (e.g. $50).
    On the first management cycle after a position opens (when no SL is set), the SL price is calculated and applied once.
    The broker then handles the rest natively.

    Calculation:
        MT5 tick_value is the monetary value of one tick (tick_size price move) for 1 lot.
        contract_size is already embedded in tick_value, so it is not multiplied separately.

        loss_per_point = volume * (tick_value / tick_size)   # $ per 1.0 price unit move
        price_distance = max_loss_amount / loss_per_point
        sl = open_price - price_distance   (BUY)
        sl = open_price + price_distance   (SELL)
    """

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.max_loss_amount = strategy_config.position_manager.max_loss_amount

    def manage_long_position(self, position: Dict[str, Any], flag: bool):
        self._set_sl_from_max_loss(position, is_long=True)

    def manage_short_position(self, position: Dict[str, Any], flag: bool):
        self._set_sl_from_max_loss(position, is_long=False)

    def _set_sl_from_max_loss(self, position: Dict[str, Any], is_long: bool):
        """Calculate and set SL price based on max_loss_amount. Only sets SL if none exists."""
        ticket = position.get("ticket")
        current_sl = position.get("sl", 0.0)

        # Only act when no SL is set
        if current_sl != 0.0:
            return

        symbol = position.get("symbol")
        open_price = position.get("price_open")
        volume = position.get("volume")

        if not symbol or open_price is None or volume is None:
            logger.error(f"Position {ticket} missing required fields to calculate SL")
            return

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to fetch symbol info for {symbol}")
            return

        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value

        if tick_size == 0:
            logger.error(f"tick_size is 0 for {symbol}, cannot calculate SL")
            return

        # MT5 tick_value is the monetary value of one tick (tick_size price move) for 1 lot.
        # contract_size is already embedded in tick_value, so we don't multiply by it.
        # Value of a 1.0 price unit move in account currency:
        loss_per_point = volume * (tick_value / tick_size)

        if loss_per_point == 0:
            logger.error(f"loss_per_point is 0 for {symbol}, cannot calculate SL")
            return

        price_distance = self.max_loss_amount / loss_per_point
        sl = open_price - price_distance if is_long else open_price + price_distance

        # Round SL to the symbol's digit precision
        sl = round(sl, symbol_info.digits)

        if sl <= 0:
            logger.error(f"Computed SL <= 0 for position {ticket} ({symbol}): {sl}. max_loss_amount may be too large for this instrument.")
            return

        direction = "BUY" if is_long else "SELL"
        if self.modify_position(position, sl=sl):
            logger.info(
                f"Set SL for {direction} position {ticket} ({symbol}): "
                f"{sl:.{symbol_info.digits}f} "
                f"(max loss ${self.max_loss_amount:.2f}, distance {price_distance:.{symbol_info.digits}f})"
            )
        else:
            logger.error(f"Failed to set SL for {direction} position {ticket} ({symbol})")