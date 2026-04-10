import numpy as np
import pandas as pd
from numba import njit

from ..utils import ensure_numpy_types


@njit(fastmath=True, cache=True)
def _ttm_trend_core(open_prices, high_prices, low_prices, close_prices, inp_period):
    """Numba-optimized TTM Trend core computation"""
    n = len(open_prices)

    if n == 0:
        return np.empty(0, dtype=np.int32)

    ttm_type = np.zeros(n, dtype=np.int32)
    ha_open = np.zeros(n, dtype=np.float64)
    ha_close = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            ha_open[i] = (open_prices[i] + close_prices[i]) / 2.0
        else:
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

        ha_close[i] = (
            open_prices[i] + high_prices[i] + low_prices[i] + close_prices[i]
        ) / 4.0

        if ha_close[i] > ha_open[i]:
            ttm_type[i] = 1
        elif ha_close[i] < ha_open[i]:
            ttm_type[i] = -1
        else:
            if i > 0:
                ttm_type[i] = ttm_type[i - 1]
            else:
                ttm_type[i] = 0

        for k in range(1, min(inp_period + 1, i + 1)):
            prev_high = max(ha_open[i - k], ha_close[i - k])
            prev_low = min(ha_open[i - k], ha_close[i - k])

            if (
                ha_open[i] <= prev_high
                and ha_open[i] >= prev_low
                and ha_close[i] <= prev_high
                and ha_close[i] >= prev_low
            ):
                ttm_type[i] = ttm_type[i - k]
                break

    return ttm_type


def ttm_trend(open_prices, high_prices, low_prices, close_prices, inp_period=10):
    """
    Convert MQL5 TTM Trend indicator to Python function.

    Parameters:
    open_prices (numpy.ndarray): Array of open prices
    high_prices (numpy.ndarray): Array of high prices
    low_prices (numpy.ndarray): Array of low prices
    close_prices (numpy.ndarray): Array of close prices
    inp_period (int): Look back period (default: 10)

    Returns:
    ttm_type: Type indicator (1 for bullish, -1 for bearish, 0 for empty)

    Note: All input arrays should have the same length and represent consecutive time periods in chronological order.
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    ttm_type = _ttm_trend_core(
        open_prices, high_prices, low_prices, close_prices, inp_period
    )

    if index is not None:
        return pd.Series(ttm_type, index=index, dtype=np.int32)
    else:
        return ttm_type
