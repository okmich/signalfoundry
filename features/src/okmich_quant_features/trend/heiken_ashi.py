import numpy as np
import pandas as pd
from numba import njit

from ..utils import ensure_numpy_types


@njit(fastmath=True, cache=True)
def _heiken_ashi_core(open_prices, high_prices, low_prices, close_prices):
    n = len(open_prices)

    if n == 0:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int32)
        return empty_float, empty_float, empty_float, empty_float, empty_int

    ha_open = np.zeros(n, dtype=np.float64)
    ha_high = np.zeros(n, dtype=np.float64)
    ha_low = np.zeros(n, dtype=np.float64)
    ha_flag = np.ones(n, dtype=np.int32)
    ha_close = (open_prices + high_prices + low_prices + close_prices) / 4

    # Calculate Heiken Ashi Open
    ha_open[0] = (open_prices[0] + close_prices[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    for i in range(n):
        ha_high[i] = max(high_prices[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low_prices[i], ha_open[i], ha_close[i])
        ha_flag[i] = 1 if ha_close[i] > ha_open[i] else -1

    return ha_open, ha_high, ha_low, ha_close, ha_flag


@njit(fastmath=True, cache=True)
def _heiken_ashi_momentum(
    open_prices, high_prices, low_prices, close_prices, lookback_period=3
):
    """
    Calculate momentum from Heiken Ashi candle body sizes.

    Parameters:
    open_prices, high_prices, low_prices, close_prices: array-like
    lookback_period (int): Number of periods to look back for momentum calculation

    Returns:
    numpy.ndarray: Momentum values (positive = bullish momentum, negative = bearish momentum)
    """
    # get the heiken ashi candles
    ha_open, ha_high, ha_low, ha_close, _ = _heiken_ashi_core(
        open_prices, high_prices, low_prices, close_prices
    )

    # Calculate body sizes (absolute values)
    body_sizes = np.abs(ha_close - ha_open)

    # Calculate momentum based on body size trends
    n = len(open_prices)
    momentum = np.zeros(n, dtype=np.float64)

    for i in range(lookback_period, n):
        # Calculate average body size over lookback period
        avg_body_size = np.mean(body_sizes[i - lookback_period : i])

        # Current body size
        current_body_size = body_sizes[i]

        # Rate of change in body size
        if avg_body_size > 0:
            momentum[i] = (current_body_size - avg_body_size) / avg_body_size
        else:
            momentum[i] = 0

    return momentum


@njit(fastmath=True, cache=True)
def _heiken_ashi_momentum_core(
    open_prices, high_prices, low_prices, close_prices, lookback_period
):
    """Numba-optimized core momentum calculation"""
    n = len(open_prices)

    if n == 0:
        return np.empty(0, dtype=np.float64)

    # get the heiken ashi candles
    ha_open, ha_high, ha_low, ha_close, _ = _heiken_ashi_core(
        open_prices, high_prices, low_prices, close_prices
    )

    # Calculate body sizes
    body_sizes = np.abs(ha_close - ha_open)

    # Calculate momentum
    momentum = np.zeros(n, dtype=np.float64)

    for i in range(lookback_period, n):
        if i >= lookback_period:
            # Linear regression slope of body sizes for momentum direction
            x = np.arange(lookback_period)
            y = body_sizes[i - lookback_period : i]

            # Simple slope calculation
            if len(y) > 1:
                slope = (y[-1] - y[0]) / (len(y) - 1)
                momentum[i] = slope
            else:
                momentum[i] = 0
    return momentum


@njit(fastmath=True, cache=True)
def _smooth_array(arr, period):
    """Simple moving average smoothing"""
    n = len(arr)
    smoothed = np.zeros(n, dtype=np.float64)

    for i in range(n):
        start_idx = max(0, i - period + 1)
        smoothed[i] = np.mean(arr[start_idx : i + 1])

    return smoothed


def heiken_ashi_momentum(
    open_prices, high_prices, low_prices, close_prices, lookback_period=3
):
    """
    Calculate momentum from Heiken Ashi candle body sizes.

    Parameters:
    open_prices, high_prices, low_prices, close_prices: array-like
    lookback_period (int): Number of periods to look back for momentum calculation

    Returns:
    numpy.ndarray: Momentum values (positive = bullish momentum, negative = bearish momentum)
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    momentum_arr = _heiken_ashi_momentum(
        open_prices, high_prices, low_prices, close_prices, lookback_period
    )
    return (
        momentum_arr
        if index is None
        else pd.Series(momentum_arr, index=index, dtype=np.float64)
    )


def heiken_ashi_momentum_advanced(
    open_prices,
    high_prices,
    low_prices,
    close_prices,
    lookback_period=3,
    smoothing_period=2,
):
    """
    Advanced momentum calculation with additional features.
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    # Calculate Heiken Ashi
    ha_open, ha_high, ha_low, ha_close, _ = _heiken_ashi_core(
        open_prices, high_prices, low_prices, close_prices
    )

    # Calculate basic momentum
    momentum = _heiken_ashi_momentum_core(
        open_prices, high_prices, low_prices, close_prices, lookback_period
    )
    body_direction = np.where(
        ha_close > ha_open, 1, np.where(ha_close < ha_open, -1, 0)
    )

    # Combine momentum magnitude with direction
    directional_momentum = momentum * body_direction

    # Apply smoothing
    if smoothing_period > 1:
        directional_momentum = _smooth_array(directional_momentum, smoothing_period)

    if index is not None:
        return pd.Series(directional_momentum, index=index, dtype=np.float64)
    else:
        return directional_momentum


def heiken_ashi(open_prices, high_prices, low_prices, close_prices):
    """
    Convert regular candlestick data to Heiken Ashi candlestick data.

    Parameters:
    open_prices (numpy.ndarray|pd.Series): Array of open prices
    high_prices (numpy.ndarray|pd.Series): Array of high prices
    low_prices (numpy.ndarray|pd.Series): Array of low prices
    close_prices (numpy.ndarray|pd.Series): Array of close prices
    ha_flag (numpy.ndarray|pd.Series): Bullish (1) or Bearish (-1) indicator

    Returns:
    tuple: (ha_open, ha_high, ha_low, ha_close) - Heiken Ashi OHLC arrays or series

    Note: All input type should have the same length and represent consecutive time periods in chronological order.
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    # Call the numba-optimized core function
    ha_open, ha_high, ha_low, ha_close, ha_flag = _heiken_ashi_core(
        open_prices, high_prices, low_prices, close_prices
    )

    # Return appropriate type
    if index is not None:
        return (
            pd.Series(ha_open, index=index, dtype=np.float64),
            pd.Series(ha_high, index=index, dtype=np.float64),
            pd.Series(ha_low, index=index, dtype=np.float64),
            pd.Series(ha_close, index=index, dtype=np.float64),
            pd.Series(ha_flag, index=index, dtype=np.int32),
        )
    else:
        return ha_open, ha_high, ha_low, ha_close, ha_flag
