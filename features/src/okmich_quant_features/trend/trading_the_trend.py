import numpy as np
import pandas as pd
from numba import njit

from ..utils import ensure_numpy_types


@njit(fastmath=True, cache=True)
def _trading_the_trend_core(open_prices, high_prices, low_prices, close_prices, inp_period=120,
                            inp_multiplier=3, inp_channel_shift=0):
    """
    Trading the Trend indicator that accepts both numpy arrays.
    Returns only the type indicator for compatibility with your request.
    """
    # Validate input shapes
    if not (
        open_prices.shape == high_prices.shape == low_prices.shape == close_prices.shape
    ):
        raise ValueError("All input arrays must have the same shape")

    n = len(open_prices)

    if n == 0:
        # Fix: Return three separate empty arrays with explicit dtypes
        empty_int = np.empty(0, dtype=np.int32)
        empty_float = np.empty(0, dtype=np.float64)
        return empty_int, empty_float, empty_int

    # Initialize output arrays
    tt_type = np.zeros(n, dtype=np.int32)  # 0 = neutral, 1 = bullish, -1 = bearish
    tt_line = np.zeros(n, dtype=np.float64)  # Trend line values
    tt_line_type = np.zeros(n, dtype=np.int32)  # 0 = neutral, 1 = bullish, -1 = bearish

    # Initialize LWMA calculation arrays
    lwma_values = np.zeros(n, dtype=np.float64)
    lwma_wsumm = np.zeros(n, dtype=np.float64)
    lwma_vsumm = np.zeros(n, dtype=np.float64)
    lwma_weight = 0

    # Variables for tracking previous max/min
    prev_max = 0
    prev_min = 0
    prev_time = -1  # Simplified time tracking

    # Process each bar
    for i in range(n):
        # Update previous max/min (simplified without actual time handling)
        if prev_time != i:  # Simulate time change
            prev_time = i
            start_idx = max(0, i - inp_period - inp_channel_shift + 1)
            end_idx = min(start_idx + max(inp_period - 1, 0), n)

            if end_idx > start_idx:
                prev_max = np.max(close_prices[start_idx:end_idx])
                prev_min = np.min(close_prices[start_idx:end_idx])
            else:
                prev_max = close_prices[i] if i < n else 0
                prev_min = close_prices[i] if i < n else 0

        # Calculate range high and low
        hi = close_prices[i] if close_prices[i] > prev_max else prev_max
        lo = close_prices[i] if close_prices[i] < prev_min else prev_min

        # Calculate true range components
        if i > 0:
            rhigh = (
                high_prices[i]
                if high_prices[i] > close_prices[i - 1]
                else close_prices[i - 1]
            )
            rlow = (
                low_prices[i]
                if low_prices[i] < close_prices[i - 1]
                else close_prices[i - 1]
            )
        else:
            rhigh = high_prices[i]
            rlow = low_prices[i]

        tr_value = rhigh - rlow

        # Store current TR before LWMA computation so the warmup loop can read it
        # at k=0 (lwma_values[i]) correctly instead of getting 0.
        lwma_values[i] = tr_value

        # Calculate LWMA of true range
        if i > inp_period:
            lwma_wsumm[i] = (
                lwma_wsumm[i - 1] + tr_value * inp_period - lwma_vsumm[i - 1]
            )
            lwma_vsumm[i] = lwma_vsumm[i - 1] + tr_value - lwma_values[i - inp_period]
        else:
            lwma_weight = 0
            lwma_wsumm[i] = 0
            lwma_vsumm[i] = 0

            for k in range(min(inp_period, i + 1)):
                w = inp_period - k
                lwma_weight += w
                lwma_wsumm[i] += lwma_values[i - k] * w
                lwma_vsumm[i] += lwma_values[i - k]

        # Calculate LWMA
        if lwma_weight != 0:
            tr = lwma_wsumm[i] / lwma_weight
        else:
            tr = tr_value

        # Calculate limits
        hi_limit = hi - tr * inp_multiplier
        lo_limit = lo + tr * inp_multiplier

        # Calculate trend line
        if i > 0:
            if close_prices[i] > lo_limit and close_prices[i] > hi_limit:
                tt_line[i] = hi_limit
            elif close_prices[i] < lo_limit and close_prices[i] < hi_limit:
                tt_line[i] = lo_limit
            else:
                tt_line[i] = tt_line[i - 1]
        else:
            tt_line[i] = close_prices[i]

        # Determine colors/types
        if close_prices[i] > tt_line[i]:
            tt_line_type[i] = 1
            tt_type[i] = 1
        elif close_prices[i] < tt_line[i]:
            tt_line_type[i] = -1
            tt_type[i] = -1
        else:
            if i > 0:
                tt_line_type[i] = tt_line_type[i - 1]
                tt_type[i] = tt_type[i - 1]
            else:
                tt_line_type[i] = 0
                tt_type[i] = 0

    return tt_type, tt_line, tt_line_type


def trading_the_trend(
    open_prices,
    high_prices,
    low_prices,
    close_prices,
    inp_period=120,
    inp_multiplier=3,
    inp_channel_shift=0):
    """
    Convert MQL5 Trading the Trend indicator to Python function.

    Parameters:
    open_prices (numpy.ndarray|pd.Series): Array of open prices
    high_prices (numpy.ndarray|pd.Series): Array of high prices
    low_prices (numpy.ndarray|pd.Series): Array of low prices
    close_prices (numpy.ndarray|pd.Series): Array of close prices
    inp_period (int): Look back period (default: 120)
    inp_multiplier (float): Multiplier for threshold calculation (default: 3)
    inp_channel_shift (int): Channel shift (default: 0)

    Returns:
    tuple: (tt_type, tt_line, tt_line_type)
        - tt_type: Type indicator for candles/bars (1 for bullish, -1 for bearish, 0 for neutral)
        - tt_line: Trend line values
        - tt_line_type: Type indicator for trend line (1 for bullish, -1 for bearish, 0 for neutral)

    Note: All input arrays should have the same length and represent consecutive time periods in chronological order.
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    tt_type, tt_line, tt_line_type = _trading_the_trend_core(
        open_prices,
        high_prices,
        low_prices,
        close_prices,
        inp_period,
        inp_multiplier,
        inp_channel_shift,
    )

    if index is not None:
        return (
            pd.Series(tt_type, index=index, dtype=np.int32),
            pd.Series(tt_line, index=index, dtype=np.float64),
            pd.Series(tt_line_type, index=index, dtype=np.int32),
        )
    else:
        return tt_type, tt_line, tt_line_type
