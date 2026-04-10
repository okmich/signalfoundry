import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Union, Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import talib

from ..timeframe_utils import timeframe_minutes_dict
from okmich_quant_core import OrderType
from ..resilience import (
    with_retry,
    MT5TransientError,
    MT5PermanentError,
    classify_mt5_error,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataFetchError(Exception):
    pass


def initialize_mt5() -> bool:
    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    return True


def reconnect_mt5(login=None, password=None, server=None, sleep_seconds=5) -> bool:
    if mt5.terminal_info():
        return True

    print("MT5 connection lost. Attempting to reconnect in 5 seconds...")
    time.sleep(sleep_seconds)

    if mt5.initialize():
        if login and password and server:
            if mt5.login(login=login, password=password, server=server):
                print("MT5 reconnected and logged in successfully")
                return True
            else:
                print(f"MT5 login failed: {mt5.last_error()}")
                return False
        print("MT5 reconnected successfully (no login credentials provided)")
        return True
    else:
        print(f"MT5 reconnection failed: {mt5.last_error()}")
        return False


def _handle_rates(rates):
    df = pd.DataFrame(rates)
    # MT5 returns timestamps in UTC - convert to timezone-aware datetime in UTC
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # Convert from UTC to local timezone
    local_tz = datetime.now(timezone.utc).astimezone().tzinfo
    df["time"] = df["time"].dt.tz_convert(local_tz)
    df.rename(columns={"time": "date"}, inplace=True)
    df.set_index("date", inplace=True)
    # change type for tick_volume from uint64 to int64
    df["tick_volume"] = df["tick_volume"].astype(np.int64)
    return df[["open", "high", "low", "close", "tick_volume", "spread"]]


def _handle_ticks(rates):
    df = pd.DataFrame(rates)
    print(df.info())
    # MT5 returns timestamps in UTC - convert to timezone-aware datetime in UTC
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # Convert from UTC to local timezone
    local_tz = datetime.now(timezone.utc).astimezone().tzinfo
    df["time"] = df["time"].dt.tz_convert(local_tz)
    df.rename(columns={"time": "date"}, inplace=True)
    df.set_index("date", inplace=True)
    # change type for tick_volume from uint64 to int64
    df["tick_volume"] = df["tick_volume"].astype(np.int64)
    return df[["bid", "ask", "last", "flags"]]


def is_market_open(symbol: str) -> bool:
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        return False
    return symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED


def fetch_symbol_info(symbol_name) -> Union[None, Dict[str, Any]]:
    symbol_info = mt5.symbol_info(symbol_name)
    if symbol_info is not None:
        return symbol_info._asdict()
    else:
        msg = "Failed to load symbol info for {}".format(symbol_name)
        logging.error(msg)
        raise DataFetchError(msg)


def fetch_symbol_tick_info(symbol_name) -> Union[None, Dict[str, Any]]:
    symbol_tick_info = mt5.symbol_info_tick(symbol_name)
    if symbol_tick_info is not None:
        return symbol_tick_info._asdict()
    else:
        msg = "Failed to load symbol tick info for {}".format(symbol_name)
        logging.error(msg)
        raise DataFetchError(msg)


def fetch_data_from_position(
    symbol: str,
    timeframe: Any,
    start_position: int = 0,
    count: int = 0,
    as_dataframe: bool = True,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, start_position, count)
    if rates is None or len(rates) == 0:
        logging.error(
            f"Failed to retrieve data for {symbol}. Cause: {mt5.last_error()}"
        )
        return []
    if as_dataframe:
        return _handle_rates(rates)
    else:
        return rates


def fetch_data_date_range(
    symbol: str, timeframe: Any, start_date: datetime, end_date: datetime
) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        msg = f"Failed to retrieve data for {symbol} from server. Cause: {mt5.last_error()}"
        logging.error(msg)
        raise DataFetchError(msg)
    return _handle_rates(rates)


def fetch_recent_data(
    symbol: str, timeframe: int, now_dt: datetime, count: int
) -> Optional[pd.DataFrame]:
    if count <= 0:
        raise ValueError("Count must be positive")
    if timeframe not in timeframe_minutes_dict:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rates = mt5.copy_rates_from(symbol, timeframe, now_dt, count)
    if rates is None or len(rates) == 0:
        msg = f"Failed to retrieve data for {symbol} from server. Cause: {mt5.last_error()}"
        logging.error(msg)
        raise DataFetchError(msg)
    return _handle_rates(rates)


def fetch_recent_data(
    symbol: str, timeframe: int, now_dt: datetime, count: int
) -> Optional[pd.DataFrame]:
    if count <= 0:
        raise ValueError("Count must be positive")
    if timeframe not in timeframe_minutes_dict:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rates = mt5.copy_rates_from(symbol, timeframe, now_dt, count)
    if rates is None or len(rates) == 0:
        msg = f"Failed to retrieve data for {symbol} from server. Cause: {mt5.last_error()}"
        logging.error(msg)
        raise DataFetchError(msg)
    return _handle_rates(rates)


def fetch_tick_data_date_range(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    include_info_ticks: bool = False,
) -> Optional[pd.DataFrame]:
    copy_flag = mt5.COPY_TICKS_ALL if include_info_ticks else mt5.COPY_TICKS_TRADE
    # ticks = mt5.copy_ticks_range(symbol, start_date, end_date, copy_flag)
    ticks = mt5.copy_ticks_from(symbol, start_date, 10000, copy_flag)
    print(ticks)
    if ticks is None or len(ticks) == 0:
        msg = f"Failed to retrieve tick data for {symbol} from server. Cause: {mt5.last_error()}"
        logging.error(msg)
        raise DataFetchError(msg)
    return _handle_ticks(ticks)


def get_positions(symbol, magic) -> List[Dict[str, Any]]:
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return []

    # Filter by magic number
    positions = [pos for pos in positions if pos.magic == magic]
    if not positions:
        return []

    return [p._asdict() for p in positions]


@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def open_position(symbol: str, order_type: Union[str, OrderType], volume: float, price: float = 0.0,
                  sl: float = 0.0, tp: float = 0.0, magic: int = 0, comment: str = "", **kwargs) -> Dict[str, Any]:
    """
    Open a market position (buy or sell) with retry logic.

    Automatically retries on transient errors (connection issues, requotes, etc.)
    Raises exception on permanent errors (invalid parameters, insufficient funds, etc.)

    Args:
        symbol: Trading symbol
        order_type: OrderType enum or string ('buy', 'sell')
        volume: Order volume (lots)
        price: Order price (0 = use current market price)
        sl: Stop loss price (0 = no SL)
        tp: Take profit price (0 = no TP)
        magic: Magic number
        comment: Order comment
        **kwargs: Additional parameters (e.g., filling_mode)

    Returns:
        Dict with order result

    Raises:
        MT5TransientError: On transient errors (will be retried by decorator)
        MT5PermanentError: On permanent errors (won't be retried)
    """
    # Convert OrderType enum to string if needed
    if isinstance(order_type, OrderType):
        order_type_str = order_type.value
    else:
        order_type_str = order_type

    order_types = {"buy": mt5.ORDER_TYPE_BUY, "sell": mt5.ORDER_TYPE_SELL}
    if order_type_str.lower() not in order_types:
        raise ValueError(f"Invalid order type: {order_type_str}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_types[order_type_str.lower()],
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": comment,
    }
    request.update(kwargs)

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_type = classify_mt5_error(result.retcode)
        error_msg = f"Failed to open position: {result.comment} (retcode: {result.retcode})"

        if error_type == 'transient':
            raise MT5TransientError(error_msg, result.retcode)
        else:
            raise MT5PermanentError(error_msg, result.retcode)

    logging.info(f"Position opened: Ticket={result.order}, Symbol={symbol}, Volume={volume}")
    return result._asdict()


@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def modify_position(position_id, sl=0.0, tp=0.0) -> Dict[str, Any]:
    """
    Modify position stop loss and/or take profit with retry logic.

    Automatically retries on transient errors.

    Args:
        position_id: Position ticket number
        sl: New stop loss price (0 = keep current)
        tp: New take profit price (0 = keep current)

    Returns:
        Dict with modification result

    Raises:
        MT5TransientError: On transient errors (will be retried by decorator)
        MT5PermanentError: On permanent errors (won't be retried)
        ValueError: If position not found
    """
    position = mt5.positions_get(ticket=position_id)
    if not position:
        raise ValueError(f"Position {position_id} not found")
    position = position[0]

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position_id,
        "symbol": position.symbol,
        "sl": sl if sl > 0 else position.sl,
        "tp": tp if tp > 0 else position.tp,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_type = classify_mt5_error(result.retcode)
        error_msg = f"Failed to modify position {position_id}: {result.comment} (retcode: {result.retcode})"

        if error_type == 'transient':
            raise MT5TransientError(error_msg, result.retcode)
        else:
            raise MT5PermanentError(error_msg, result.retcode)

    logging.info(f"Position {position_id} modified: SL={sl}, TP={tp}")
    return result._asdict()


@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def close_position(ticket_id, **kwargs) -> bool:
    """
    Close an open position with retry logic.

    Automatically retries on transient errors.

    Args:
        ticket_id: Position ticket number
        **kwargs: Additional parameters (e.g., filling_mode)

    Returns:
        True if position closed successfully

    Raises:
        MT5TransientError: On transient errors (will be retried by decorator)
        MT5PermanentError: On permanent errors (won't be retried)
        ValueError: If position not found
    """
    position = mt5.positions_get(ticket=ticket_id)
    if not position:
        raise ValueError(f"Position {ticket_id} not found")
    position = position[0]

    order_type = (
        mt5.ORDER_TYPE_SELL
        if position.type == mt5.POSITION_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )
    price = (
        mt5.symbol_info_tick(position.symbol).bid
        if position.type == mt5.POSITION_TYPE_BUY
        else mt5.symbol_info_tick(position.symbol).ask
    )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket_id,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "price": price,
    }
    request.update(kwargs)

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_type = classify_mt5_error(result.retcode)
        error_msg = f"Failed to close position {ticket_id}: {result.comment} (retcode: {result.retcode})"

        if error_type == 'transient':
            raise MT5TransientError(error_msg, result.retcode)
        else:
            raise MT5PermanentError(error_msg, result.retcode)

    logging.info(f"Position {ticket_id} closed successfully")
    return True


def close_all_positions(symbol=None):
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if not positions:
        return None

    results = []
    for pos in positions:
        result = close_position(pos.ticket)
        if result:
            results.append(result)

    print(f"Closed {len(results)} positions for {symbol}")
    return results


@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def place_pending_order(symbol: str, order_type: Union[str, OrderType], volume: float, price: float,
                        sl: float = 0.0, tp: float = 0.0, magic: int = 0, comment: str = "",
                        expiration: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    Place a pending order (buy_stop, sell_stop, buy_limit, sell_limit) with retry logic.

    Automatically retries on transient errors.

    Args:
        symbol: Trading symbol
        order_type: OrderType enum or string ('buy_stop', 'sell_stop', 'buy_limit', 'sell_limit')
        volume: Order volume (lots)
        price: Order price
        sl: Stop loss price (0 = no SL)
        tp: Take profit price (0 = no TP)
        magic: Magic number for order identification
        comment: Order comment
        expiration: Order expiration timestamp (None = GTC - Good Till Cancelled)
        **kwargs: Additional parameters (e.g., filling_mode)

    Returns:
        Dict with order result

    Raises:
        MT5TransientError: On transient errors (will be retried by decorator)
        MT5PermanentError: On permanent errors (won't be retried)
        ValueError: On invalid order type
    """
    # Convert OrderType enum to string if needed
    if isinstance(order_type, OrderType):
        order_type_str = order_type.value
    else:
        order_type_str = order_type

    # Map order types to MT5 constants
    pending_order_types = {
        "buy_stop": mt5.ORDER_TYPE_BUY_STOP,
        "sell_stop": mt5.ORDER_TYPE_SELL_STOP,
        "buy_limit": mt5.ORDER_TYPE_BUY_LIMIT,
        "sell_limit": mt5.ORDER_TYPE_SELL_LIMIT,
    }

    order_type_lower = order_type_str.lower()
    if order_type_lower not in pending_order_types:
        raise ValueError(
            f"Invalid pending order type: {order_type}. Must be one of: {list(pending_order_types.keys())}"
        )

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": pending_order_types[order_type_lower],
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC if expiration is None else mt5.ORDER_TIME_SPECIFIED,
    }

    # Add expiration if specified
    if expiration is not None:
        request["expiration"] = expiration

    # Add any additional parameters (like filling_mode)
    request.update(kwargs)

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_type = classify_mt5_error(result.retcode)
        error_msg = f"Failed to place pending order: {result.comment} (retcode: {result.retcode})"

        if error_type == 'transient':
            raise MT5TransientError(error_msg, result.retcode)
        else:
            raise MT5PermanentError(error_msg, result.retcode)

    logging.info(
        f"Pending order placed: Ticket={result.order}, Type={order_type}, "
        f"Price={price}, Symbol={symbol}, Volume={volume}"
    )
    return result._asdict()


@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def modify_pending_order(ticket: int, price: Optional[float] = None,
                         sl: Optional[float] = None, tp: Optional[float] = None,
                         expiration: Optional[int] = None) -> Dict[str, Any]:
    """
    Modify a pending order with retry logic.

    Automatically retries on transient errors.

    Args:
        ticket: Order ticket number
        price: New order price (None = keep current)
        sl: New stop loss (None = keep current)
        tp: New take profit (None = keep current)
        expiration: New expiration time (None = keep current)

    Returns:
        Dict with modification result

    Raises:
        MT5TransientError: On transient errors (will be retried by decorator)
        MT5PermanentError: On permanent errors (won't be retried)
        ValueError: If pending order not found
    """
    # Get current order info
    orders = mt5.orders_get(ticket=ticket)
    if not orders:
        raise ValueError(f"Pending order {ticket} not found")

    order = orders[0]

    request = {
        "action": mt5.TRADE_ACTION_MODIFY,
        "order": ticket,
        "symbol": order.symbol,
        "price": price if price is not None else order.price_open,
        "sl": sl if sl is not None else order.sl,
        "tp": tp if tp is not None else order.tp,
    }

    # Handle expiration
    if expiration is not None:
        request["type_time"] = mt5.ORDER_TIME_SPECIFIED
        request["expiration"] = expiration
    else:
        request["type_time"] = order.type_time
        if order.type_time == mt5.ORDER_TIME_SPECIFIED:
            request["expiration"] = order.time_expiration

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_type = classify_mt5_error(result.retcode)
        error_msg = f"Failed to modify pending order {ticket}: {result.comment} (retcode: {result.retcode})"

        if error_type == 'transient':
            raise MT5TransientError(error_msg, result.retcode)
        else:
            raise MT5PermanentError(error_msg, result.retcode)

    logging.info(f"Pending order {ticket} modified successfully")
    return result._asdict()


@with_retry(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def cancel_pending_order(ticket: int) -> bool:
    """
    Cancel a pending order with retry logic.

    Automatically retries on transient errors.

    Args:
        ticket: Order ticket number

    Returns:
        True if cancelled successfully

    Raises:
        MT5TransientError: On transient errors (will be retried by decorator)
        MT5PermanentError: On permanent errors (won't be retried)
        ValueError: If pending order not found
    """
    # Get order info to get symbol
    orders = mt5.orders_get(ticket=ticket)
    if not orders:
        raise ValueError(f"Pending order {ticket} not found")

    order = orders[0]

    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": ticket,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        error_type = classify_mt5_error(result.retcode)
        error_msg = f"Failed to cancel pending order {ticket}: {result.comment} (retcode: {result.retcode})"

        if error_type == 'transient':
            raise MT5TransientError(error_msg, result.retcode)
        else:
            raise MT5PermanentError(error_msg, result.retcode)

    logging.info(f"Pending order {ticket} cancelled successfully")
    return True


def get_pending_orders(symbol: Optional[str] = None, magic: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get list of pending orders.

    Args:
        symbol: Filter by symbol (None = all symbols)
        magic: Filter by magic number (None = all magic numbers)

    Returns:
        List of pending orders as dictionaries
    """
    if symbol and magic is not None:
        orders = mt5.orders_get(symbol=symbol, magic=magic)
    elif symbol:
        orders = mt5.orders_get(symbol=symbol)
    elif magic is not None:
        # MT5 doesn't support filtering by magic only, so get all and filter
        orders = mt5.orders_get()
        if orders:
            orders = [o for o in orders if o.magic == magic]
    else:
        orders = mt5.orders_get()

    if not orders:
        return []

    return [o._asdict() for o in orders]


def cancel_all_pending_orders(symbol: Optional[str] = None, magic: Optional[int] = None) -> List[int]:
    """
    Cancel all pending orders matching criteria.

    Args:
        symbol: Filter by symbol (None = all symbols)
        magic: Filter by magic number (None = all magic numbers)

    Returns:
        List of cancelled order tickets
    """
    orders = get_pending_orders(symbol=symbol, magic=magic)
    if not orders:
        print(f"No pending orders found for symbol={symbol}, magic={magic}")
        return []

    cancelled = []
    for order in orders:
        if cancel_pending_order(order['ticket']):
            cancelled.append(order['ticket'])

    print(f"Cancelled {len(cancelled)} pending orders")
    return cancelled


def get_atr(symbol, timeframe, period) -> float:
    bars_count = period + 10
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars_count)

    if rates is None or len(rates) < period:
        raise ValueError(f"Insufficient data for ATR calculation for {symbol}")

    # Convert to numpy arrays
    high = np.array([rate["high"] for rate in rates])
    low = np.array([rate["low"] for rate in rates])
    close = np.array([rate["close"] for rate in rates])

    atr_values = talib.ATR(high, low, close, timeperiod=period)
    valid_atr_values = atr_values[~np.isnan(atr_values)]

    if len(valid_atr_values) == 0:
        raise ValueError("ATR calculation failed - no valid values")

    return float(valid_atr_values[-1])
