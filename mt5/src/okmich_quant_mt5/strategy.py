import json
import logging
import traceback
from abc import abstractmethod
from datetime import datetime
from typing import Union

from . import number_of_minutes_in_timeframe, is_timeframe_match, timeframe_minutes_dict
from .functions import (
    get_positions,
    close_position,
    fetch_recent_data,
    fetch_data_date_range,
    fetch_symbol_info,
    fetch_symbol_tick_info,
    open_position,
    fetch_data_from_position,
    place_pending_order,
    modify_pending_order,
    cancel_pending_order,
    get_pending_orders,
    reconnect_mt5,
)
from .position_manager import get_position_manager
from .filters import create_filter
from .resilience import (
    ConnectionMonitor,
    MT5TransientError,
    MT5PermanentError,
    MT5ConnectionError,
)
from okmich_quant_core import StrategyConfig, BaseSignal, BaseStrategy, OrderType
from okmich_quant_core.price_buffer import PriceBuffer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseMt5Strategy(BaseStrategy):
    def __init__(self, config: StrategyConfig, signal: BaseSignal, *args, **kwargs):
        super().__init__(config, signal, *args, **kwargs)

        self.position_manager = get_position_manager(config) if config.position_manager else None
        self.max_number_of_mins_in_tf = number_of_minutes_in_timeframe(self.strategy_config.timeframe)
        self.max_number_of_open_positions = config.max_number_of_open_positions

        # load symbol information
        self.symbol_info_dict = fetch_symbol_info(self.strategy_config.symbol)
        if self.symbol_info_dict is None:
            raise ValueError(f"Failed to fetch symbol info for {self.strategy_config.symbol}.")
        logger.info(
            "---- Symbol information ----\n{}".format(
                json.dumps(self.symbol_info_dict, indent=4)
            )
        )

        # Initialize PriceBuffer for efficient data fetching
        # You can customize exclude_columns in subclasses if needed
        self.price_buffer = PriceBuffer(
            symbol=self.strategy_config.symbol,
            timeframe=self.strategy_config.timeframe,
            buffer_size=self.strategy_config.bars_to_copy,
            exclude_columns=getattr(
                config, "exclude_columns", None
            ),  # Optional config parameter
            timeframe_minutes=timeframe_minutes_dict[self.strategy_config.timeframe],
        )
        logger.info(f"PriceBuffer initialized for {self.strategy_config.symbol}")

        # Initialize filter chain from configuration
        self.filter_chain = create_filter(self.strategy_config)
        num_filters = len(self.filter_chain.filters)
        logger.info(
            f"Filter chain initialized for {self.strategy_config.symbol}: "
            f"{num_filters} filter(s) active"
        )

        # Initialize connection monitor
        self.connection_monitor = ConnectionMonitor(
            check_interval=60.0,  # Check every 60 seconds
            reconnect_callback=reconnect_mt5
        )
        logger.info("Connection monitor initialized")

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> int:
        """
        Search and runs position management based on the instance's position manager implementation and returns the number of open positions

        :param run_dt:   - datetime this call was initiated
        :param flag:bool - indicates this was called on a new bar
        """
        # try to tell if the run_dt is at most 5 second from the previous_run_dt
        if (
            self.prev_position_chk_dt
            and abs((run_dt - self.prev_position_chk_dt).total_seconds()) <= 5
        ):
            return 0

        res = 0
        if self.position_manager:
            self.position_manager.manage_positions(run_dt, flag)

        self.prev_position_chk_dt = run_dt
        return res

    def is_new_bar(self, run_dt: datetime) -> bool:
        """
        Check if the given datetime represents a new bar for MT5.

        Args:
            run_dt: The datetime to check

        Returns:
            True if this matches the strategy's timeframe, False otherwise
        """
        return is_timeframe_match(self.strategy_config.timeframe, run_dt)

    @abstractmethod
    def on_new_bar(self):
        """
        Runs the complete strategy defined by the implementation. This at minimum should include
        - fetching data
        - generate signals
        - manage positions or possibly exiting positions based on signals
        - open new positions based on signals
        """
        pass

    def fetch_ohlcv(self):
        """
        Fetch OHLCV data using simple position-based fetching.

        Always fetches from position 0 (most recent complete bar).
        This is more reliable than time-based fetching which can have
        timing issues at bar boundaries.
        """
        try:
            # Simple approach: always fetch using position-based method
            # Position 0 = most recent COMPLETE bar (MT5 excludes forming bars)
            new_data = fetch_data_from_position(
                self.strategy_config.symbol,
                self.strategy_config.timeframe,
                start_position=0,
                count=self.strategy_config.bars_to_copy,
            )

            if new_data is None or len(new_data) == 0:
                logger.warning(
                    f"No candle data retrieved for {self.strategy_config.symbol}"
                )
                return None

            # Validate we have enough data
            if len(new_data) < self.strategy_config.bars_to_copy:
                logger.warning(
                    f"Insufficient data: {len(new_data)} < {self.strategy_config.bars_to_copy}"
                )
                return None

            return new_data
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return None

    def fetch_latest_tick_info(self):
        tick_info = fetch_symbol_tick_info(self.strategy_config.symbol)
        if tick_info is None:
            return None
        return tick_info

    def open_position(self, direction, price):
        """
        Open a market order position.

        Deprecated: Use place_order() for more flexibility with order types.

        Returns:
            True if position opened successfully, False otherwise
        """
        custom_dict = {"filling_mode": self.symbol_info_dict["filling_mode"]}
        try:
            open_position(
                symbol=self.strategy_config.symbol,
                order_type=direction,
                volume=self.calculate_lot_size(),
                price=price,
                magic=self.strategy_config.magic,
                **custom_dict,
            )
            return True
        except (MT5TransientError, MT5ConnectionError) as e:
            # Transient errors already retried by decorator - log and fail
            logger.error(
                f"Failed to open position after retries: {e}"
            )
            return False
        except MT5PermanentError as e:
            # Permanent errors - log and fail immediately
            logger.error(
                f"Failed to open position (permanent error): {e}"
            )
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(
                f"Unexpected error opening position: {e}"
            )
            return False

    def place_order(
        self,
        order_type: Union[str, OrderType],
        price: float,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
        expiration: int = None
    ) -> bool:
        """
        Place an order (market or pending).

        Args:
            order_type: OrderType enum or string ('buy', 'sell', 'buy_stop', 'sell_stop', 'buy_limit', 'sell_limit')
            price: Order price (for pending orders) or market price (for market orders)
            sl: Stop loss price (0 = no SL)
            tp: Take profit price (0 = no TP)
            comment: Order comment
            expiration: Order expiration timestamp (None = GTC, only for pending orders)

        Returns:
            True if order placed successfully, False otherwise
        """
        # Convert OrderType enum to string if needed
        if isinstance(order_type, OrderType):
            order_type_str = order_type.value
        else:
            order_type_str = order_type

        order_type_lower = order_type_str.lower()
        volume = self.calculate_lot_size()

        try:
            # Market orders (buy/sell)
            if order_type_lower in ['buy', 'sell']:
                custom_dict = {"filling_mode": self.symbol_info_dict["filling_mode"]}
                result = open_position(
                    symbol=self.strategy_config.symbol,
                    order_type=order_type_lower,
                    volume=volume,
                    price=price,
                    sl=sl,
                    tp=tp,
                    magic=self.strategy_config.magic,
                    comment=comment,
                    **custom_dict,
                )
                logger.info(
                    f"Market order placed: {order_type} {volume} lots @ {price} "
                    f"(SL={sl}, TP={tp})"
                )
                return True

            # Pending orders (buy_stop, sell_stop, buy_limit, sell_limit)
            elif order_type_lower in ['buy_stop', 'sell_stop', 'buy_limit', 'sell_limit']:
                custom_dict = {"filling_mode": self.symbol_info_dict["filling_mode"]}
                result = place_pending_order(
                    symbol=self.strategy_config.symbol,
                    order_type=order_type_lower,
                    volume=volume,
                    price=price,
                    sl=sl,
                    tp=tp,
                    magic=self.strategy_config.magic,
                    comment=comment,
                    expiration=expiration,
                    **custom_dict,
                )
                logger.info(
                    f"Pending order placed: {order_type} {volume} lots @ {price} "
                    f"(SL={sl}, TP={tp}, Expiration={expiration})"
                )
                return True

            else:
                logger.error(f"Invalid order type: {order_type}")
                return False

        except (MT5TransientError, MT5ConnectionError) as e:
            # Transient errors already retried by decorator - log and fail
            logger.error(f"Failed to place {order_type} order after retries: {e}")
            return False
        except MT5PermanentError as e:
            # Permanent errors - log and fail immediately
            logger.error(f"Failed to place {order_type} order (permanent error): {e}")
            return False
        except ValueError as e:
            # Validation errors (invalid parameters)
            logger.error(f"Invalid parameters for {order_type} order: {e}")
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error placing {order_type} order: {e}")
            return False

    def get_pending_orders_for_strategy(self):
        """
        Get all pending orders for this strategy (filtered by symbol and magic).

        Returns:
            List of pending order dictionaries
        """
        return get_pending_orders(
            symbol=self.strategy_config.symbol,
            magic=self.strategy_config.magic
        )

    def close_position(self, ticket):
        """
        Close a position by ticket number.

        Args:
            ticket: Position ticket number

        Returns:
            True if position closed successfully, False otherwise
        """
        try:
            logger.info(
                f"Closing position {ticket} for {self.strategy_config.symbol} ({self.strategy_config.magic})..."
            )
            close_position(
                ticket, **{"filling_mode": self.symbol_info_dict["filling_mode"]}
            )
            return True
        except (MT5TransientError, MT5ConnectionError) as e:
            # Transient errors already retried by decorator - log and fail
            logger.error(f"Failed to close position {ticket} after retries: {e}")
            return False
        except MT5PermanentError as e:
            # Permanent errors - log and fail immediately
            logger.error(f"Failed to close position {ticket} (permanent error): {e}")
            return False
        except ValueError as e:
            # Position not found
            logger.error(f"Position {ticket} not found: {e}")
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error closing position {ticket}: {e}")
            return False

    def calculate_lot_size(self) -> float:
        if self.strategy_config.risk_per_trade:
            # calculate lot size based on free margin of the account
            return 0.01
        elif self.strategy_config.fixed_lot_size_per_trade:
            return self.strategy_config.fixed_lot_size_per_trade
        else:
            raise ValueError(
                f"Error calculating lot size for {self.strategy_config.symbol}. "
                f"Reason: Unknown lot size strategy"
            )


class GenericBasicStrategy(BaseMt5Strategy):

    def __init__(self, config: StrategyConfig, signal: BaseSignal, *args, **kwargs):
        super().__init__(config, signal, *args, **kwargs)

    def on_new_bar(self):
        _symbol = self.strategy_config.symbol
        _magic = self.strategy_config.magic
        logging.info(f"GenericBasicStrategy ({self.strategy_config}) on_new_bar: {self.latest_run_dt}. EXECUTING... ")
        try:
            price_bars = self.fetch_ohlcv()

            # Check if we got valid data
            if price_bars is None:
                logger.warning(f"Skipping strategy execution - no valid price data")
                return None

            entries_long, exits_long, entries_short, exits_short = self.signal_generator.generate(price_bars)
            entries_long, exits_long, entries_short, exits_short = (
                entries_long[-1].item(), exits_long[-1].item(),
                entries_short[-1].item(), exits_short[-1].item(),
            )
            logger.info(
                f"Signal generated for {_symbol} ({self.strategy_config.magic}): "
                f"{(entries_long, exits_long, entries_short, exits_short)}"
            )

            positions = get_positions(_symbol, _magic)
            if len(positions) > 0 and (exits_long != 0 or exits_short != 0):
                for pos in positions:
                    if (exits_long != 0 and pos["type"] == 0) or (exits_short != 0 and pos["type"] == 1):
                        closed = self.close_position(pos["ticket"])
                        if closed and self.notifier:
                            self.notifier.on_trade_closed(
                                symbol=_symbol,
                                ticket=pos["ticket"],
                                profit=pos.get("profit", 0.0),
                            )

            if entries_long != 0 or entries_short != 0:
                positions = get_positions(_symbol, _magic)  # call again incase things changed while closing positions
                if len(positions) >= self.max_number_of_open_positions:
                    logger.info(f"Got an entry signal but an open position for {_symbol} ({self.strategy_config.magic}) already exist.")
                else:
                    # Fetch tick info and determine direction
                    direction = "buy" if entries_long > 0  else ("sell" if entries_short > 0 else "hold")
                    tick = self.fetch_latest_tick_info()

                    # Check filters before opening position
                    filter_context = {
                        "datetime": self.latest_run_dt,
                        "symbol_info": self.symbol_info_dict,
                        "tick_info": tick,
                        "open_positions": len(positions),
                        "spread": tick.get("spread", 0),
                        "signal_type": "long" if entries_long != 0 else "short",
                    }

                    if not self.filter_chain(filter_context):
                        logger.info(f"Filter chain blocked entry signal for {_symbol} ({self.strategy_config.magic})")
                        return None

                    # Filters passed - proceed with opening position
                    price = tick["ask"] if direction == "buy" else tick["bid"]
                    opened = self.open_position(direction, price=price)
                    if opened and self.notifier:
                        self.notifier.on_trade_opened(
                            symbol=_symbol,
                            direction=direction,
                            volume=self.calculate_lot_size(),
                            price=price,
                            sl=0.0,
                            tp=0.0,
                            magic=_magic,
                            ticket=0,
                        )
        except Exception as e:
            traceback.print_exc()
            if self.notifier:
                self.notifier.on_error(self.strategy_config.name, str(e))
        return None
