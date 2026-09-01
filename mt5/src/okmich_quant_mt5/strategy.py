import json
import logging
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Optional, Union

from . import number_of_minutes_in_timeframe, is_timeframe_match, timeframe_minutes_dict
from .functions import (
    get_positions,
    fetch_closed_deals,
    select_history_window,
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
from okmich_quant_core import (StrategyConfig, BaseSignal, BaseStrategy, ClosedTrade, CloseReason, OrderType,
                                PositionSizingType)
from okmich_quant_core.price_buffer import PriceBuffer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#: MT5 deal reason -> the broker-neutral cause. "client" and "expert" both mean "something asked for this close";
#: which of the two it was does not survive as intent, so the strategy's own recorded reason wins over them.
_MT5_CLOSE_REASONS = {
    "take_profit": CloseReason.TAKE_PROFIT, "stop_loss": CloseReason.STOP_LOSS, "stop_out": CloseReason.STOP_OUT,
    "expert": CloseReason.STRATEGY, "client": CloseReason.MANUAL, "mobile": CloseReason.MANUAL,
    "web": CloseReason.MANUAL, "rollover": CloseReason.EXPIRED, "split": CloseReason.EXPIRED,
}


class BaseMt5Strategy(BaseStrategy):
    #: Minimum spacing between position sweeps. Strictly less-than, so a runner polling exactly at this
    #: cadence is NOT swallowed: with ``<=`` a chk_position_interval equal to this value silently dropped
    #: every sweep, and the symptom (no position management at all) looks identical to a quiet market.
    _MIN_POSITION_CHK_SECONDS = 5.0

    def __init__(self, config: StrategyConfig, signal: BaseSignal, *args, **kwargs):
        # Supply the contract envelope's integer-minute timeframe from the MT5 timeframe constant
        # (a raw MT5 constant is NOT minutes — e.g. H1 == 16385) (LOGGING_CONTRACT §6).
        tf_min = number_of_minutes_in_timeframe(config.timeframe)
        if tf_min <= 0:
            # number_of_minutes_in_timeframe returns -1 for an unrecognised timeframe. A bad value
            # would silently corrupt the inference-log path segment (.../<tf>/...) and every
            # asof_bar_ts in the envelope, so fail fast here rather than emit garbage (LOGGING_CONTRACT §6).
            raise ValueError(
                f"Unsupported MT5 timeframe {config.timeframe!r}: cannot resolve integer minutes "
                f"for the inference-log envelope. Supported: {list(timeframe_minutes_dict.keys())}"
            )
        super().__init__(config, signal, *args, timeframe_minutes=tf_min, **kwargs)

        self.position_manager = get_position_manager(config) if config.position_manager else None
        self.max_number_of_mins_in_tf = tf_min
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

        # Deal history must be SELECTED before history_deals_get can serve it. Done once here so a
        # position that closed while this runner was down is still resolvable on the first sweep.
        select_history_window()

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> Optional[list[dict]]:
        """Run position management and return the open positions for this (symbol, magic).

        Returns ``None`` — never ``[]`` — when the book was not actually observed, so the caller's close
        detection cannot mistake an unobserved book for an empty one and report the whole book as closed.

        :param run_dt:   - datetime this call was initiated
        :param flag:bool - indicates this was called on a new bar
        """
        if (
            self.prev_position_chk_dt
            and abs((run_dt - self.prev_position_chk_dt).total_seconds()) < self._MIN_POSITION_CHK_SECONDS
        ):
            return None                      # debounced: the book was NOT observed, not observed-and-empty

        if self.position_manager:
            self.position_manager.manage_positions(run_dt, flag)

        self.prev_position_chk_dt = run_dt
        try:
            return get_positions(self.strategy_config.symbol, self.strategy_config.magic)
        except Exception as e:
            # A failed query is reported as "unobserved", NOT re-raised. get_positions raises so that an entry
            # gate cannot trade on a false flat, and on_new_bar's own call still enforces that. This sweep is
            # not a gate: position management above has already run, so re-raising would add no protection and
            # would newly count toward the circuit breaker on every intra-bar tick — turning a brief terminal
            # hiccup into a disabled strategy. None already means exactly "I could not look".
            logger.error(f"{self.strategy_config.symbol}: position sweep could not read the book: {e}")
            return None

    def resolve_closed_trade(self, key: str, last_seen: dict) -> Optional[ClosedTrade]:
        """Describe a position that left the book, from the broker's own deal history.

        Reports the REALISED close: MT5 books profit, swap and commission on the closing deal, and a position
        that has already gone cannot be asked what it made. Partial closes yield several OUT deals; they are
        summed and the exit price is volume-weighted, so a scaled-out position reports one honest average
        rather than whichever leg happened to be last.
        """
        deals = fetch_closed_deals(int(key))
        if not deals:
            return None
        volume = sum(float(d.get("volume") or 0.0) for d in deals)
        notional = sum(float(d.get("volume") or 0.0) * float(d.get("price") or 0.0) for d in deals)
        last = deals[-1]
        closed_at = datetime.fromtimestamp(last["time"], tz=timezone.utc) if last.get("time") else None
        return ClosedTrade(
            key=key, symbol=self.strategy_config.symbol, magic=self.strategy_config.magic,
            reason=_MT5_CLOSE_REASONS.get(last.get("reason_name"), CloseReason.UNKNOWN),
            volume=volume,
            entry_price=float(last_seen.get("price_open") or 0.0),
            exit_price=(notional / volume) if volume else float(last.get("price") or 0.0),
            profit=sum(float(d.get("profit") or 0.0) for d in deals),
            commission=sum(float(d.get("commission") or 0.0) for d in deals),
            swap=sum(float(d.get("swap") or 0.0) for d in deals),
            closed_at=closed_at, last_seen=last_seen)

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

    def current_spread_points(self, tick_info=None) -> Optional[float]:
        """Live spread in POINTS, derived from the quote.

        MT5's tick struct carries no ``spread`` field — ``symbol_info_tick()._asdict()`` yields
        ``{time, bid, ask, last, volume, time_msc, flags, volume_real}`` — so a ``tick.get("spread", 0)``
        silently evaluates to 0 and every SpreadFilter threshold passes unconditionally. Compute it from
        the quote instead: (ask - bid) / point. Returns None when the quote or the point size is
        unavailable, which SpreadFilter treats as "block" rather than "allow".
        """
        tick_info = tick_info or self.fetch_latest_tick_info()
        point = float(self.symbol_info_dict.get("point", 0.0) or 0.0)
        if not tick_info or point <= 0:
            return None
        bid, ask = tick_info.get("bid"), tick_info.get("ask")
        if not bid or not ask:
            return None
        return (ask - bid) / point

    def _notify_trade_failed(self, direction: str, reason: str, retcode: int = None) -> None:
        if not self.notifier:
            return
        self.notifier.on_trade_failed(
            symbol=self.strategy_config.symbol, direction=direction, reason=reason,
            retcode=retcode, context={"strategy_name": self.strategy_config.name},
        )

    def track_open_positions(self) -> None:
        """Start tracking every open position for this (symbol, magic) right now.

        Called immediately after a successful open. Waiting for the next sweep to notice a new position leaves a
        window in which a position can open AND close unseen, and an exit nobody observed is an exit nobody can
        report. Never raises: the order already went through, and a bookkeeping failure must not be reported to
        the caller as a failed trade.
        """
        try:
            for position in get_positions(self.strategy_config.symbol, self.strategy_config.magic):
                self.register_open_position(position)
        except Exception as e:
            logger.error(f"{self.strategy_config.symbol}: could not track open positions after entry: {e}")

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
            self.track_open_positions()
            return True
        except (MT5TransientError, MT5ConnectionError) as e:
            # Transient errors already retried by decorator - log and fail
            logger.error(f"Failed to open position after retries: {e}")
            self._notify_trade_failed(direction, str(e), getattr(e, "retcode", None))
            return False
        except MT5PermanentError as e:
            # Permanent errors - log and fail immediately
            logger.error(f"Failed to open position (permanent error): {e}")
            self._notify_trade_failed(direction, str(e), e.retcode)
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error opening position: {e}")
            self._notify_trade_failed(direction, f"unexpected: {e}")
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
                self.track_open_positions()
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
            self._notify_trade_failed(order_type_str, str(e), getattr(e, "retcode", None))
            return False
        except MT5PermanentError as e:
            # Permanent errors - log and fail immediately
            logger.error(f"Failed to place {order_type} order (permanent error): {e}")
            self._notify_trade_failed(order_type_str, str(e), e.retcode)
            return False
        except ValueError as e:
            # Validation errors (invalid parameters)
            logger.error(f"Invalid parameters for {order_type} order: {e}")
            self._notify_trade_failed(order_type_str, f"invalid params: {e}")
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error placing {order_type} order: {e}")
            self._notify_trade_failed(order_type_str, f"unexpected: {e}")
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

    def close_position(self, ticket, reason: str = "strategy_close"):
        """
        Close a position by ticket number.

        Args:
            ticket: Position ticket number
            reason: why this system is closing it, recorded as intent for the reconciler to attribute the
                close with. NOT announced here - see BaseStrategy.note_close_intent for why the announcement
                belongs to the reconciler.

        Returns:
            True if position closed successfully, False otherwise
        """
        try:
            logger.info(
                f"Closing position {ticket} for {self.strategy_config.symbol} ({self.strategy_config.magic})..."
            )
            self.note_close_intent(ticket, reason)
            close_position(
                ticket, **{"filling_mode": self.symbol_info_dict["filling_mode"]}
            )
            return True
        except (MT5TransientError, MT5ConnectionError) as e:
            # Transient errors already retried by decorator - log and fail
            logger.error(f"Failed to close position {ticket} after retries: {e}")
            self._notify_trade_failed("CLOSE", f"ticket={ticket}: {e}", getattr(e, "retcode", None))
            return False
        except MT5PermanentError as e:
            # Permanent errors - log and fail immediately
            logger.error(f"Failed to close position {ticket} (permanent error): {e}")
            self._notify_trade_failed("CLOSE", f"ticket={ticket}: {e}", e.retcode)
            return False
        except ValueError as e:
            # Position not found
            logger.error(f"Position {ticket} not found: {e}")
            self._notify_trade_failed("CLOSE", f"ticket={ticket} not found: {e}")
            return False
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error closing position {ticket}: {e}")
            self._notify_trade_failed("CLOSE", f"ticket={ticket} unexpected: {e}")
            return False

    def calculate_lot_size(self) -> float:
        sizing = self.strategy_config.position_sizing
        if sizing.type is PositionSizingType.FIXED:
            assert sizing.units is not None
            return sizing.units
        raise NotImplementedError(
            f"Position sizing '{sizing.type.value}' is declared but not implemented "
            f"for MT5. Override calculate_lot_size in a subclass."
        )


class GenericBasicStrategy(BaseMt5Strategy):

    def __init__(self, config: StrategyConfig, signal: BaseSignal, *args, **kwargs):
        super().__init__(config, signal, *args, **kwargs)

    def on_new_bar(self):
        # Exceptions propagate to the sealed BaseStrategy.run() template, which records the
        # outcome=error heartbeat and re-raises for MultiTrader/StrategyHealth — so this body must
        # NOT swallow them.
        _symbol = self.strategy_config.symbol
        _magic = self.strategy_config.magic
        logging.info(f"GenericBasicStrategy ({self.strategy_config}) on_new_bar: {self.latest_run_dt}. EXECUTING... ")
        price_bars = self.fetch_ohlcv()

        # Check if we got valid data
        if price_bars is None:
            logger.warning(f"Skipping strategy execution - no valid price data")
            return

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
                    # No on_trade_closed here on purpose: pos["profit"] is the UNREALISED figure read
                    # before the close request, and announcing from both here and the reconciler would put
                    # two emitters on one event. close_position records the intent; the next position sweep
                    # resolves the realised fill from broker history and announces it once.
                    self.close_position(pos["ticket"], reason="exit_signal")

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
                    "spread": self.current_spread_points(tick),
                    "signal_type": "long" if entries_long != 0 else "short",
                }

                if not self.filter_chain(filter_context):
                    logger.info(f"Filter chain blocked entry signal for {_symbol} ({self.strategy_config.magic})")
                    return

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
