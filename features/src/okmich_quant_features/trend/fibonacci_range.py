import numpy as np
import pandas as pd
from numba import njit

from ..utils import ensure_numpy_types


@njit(fastmath=True, cache=True)
def _fibonacci_range_core(open_prices, high_prices, low_prices, close_prices):
    """Numba-optimized Fibonacci Range core computation"""
    n = len(open_prices)
    plot_begins = 13

    if n < plot_begins:
        return np.zeros(n, dtype=np.int32)

    fr_type = np.zeros(n, dtype=np.int32)
    for i in range(plot_begins, n):
        bullish_condition = (
            i >= 13
            and high_prices[i] > low_prices[i - 2]
            and high_prices[i] > low_prices[i - 3]
            and high_prices[i] > low_prices[i - 5]
            and high_prices[i] > low_prices[i - 8]
            and high_prices[i] > low_prices[i - 13]
        )

        bearish_condition = (
            i >= 13
            and low_prices[i] < high_prices[i - 2]
            and low_prices[i] < high_prices[i - 3]
            and low_prices[i] < high_prices[i - 5]
            and low_prices[i] < high_prices[i - 8]
            and low_prices[i] < high_prices[i - 13]
        )

        if bullish_condition:
            fr_type[i] = 1
        elif bearish_condition:
            fr_type[i] = -1
        else:
            fr_type[i] = 0

    return fr_type


def fibonacci_range(open_prices, high_prices, low_prices, close_prices):
    """
    Convert MQL5 Fibonacci Range indicator to Python function.

    Parameters:
    open_prices (numpy.ndarray|pd.Series): Array or Series of open prices
    high_prices (numpy.ndarray|pd.Series): Array or Series of high prices
    low_prices (numpy.ndarray|pd.Series): Array or Series of low prices
    close_prices (numpy.ndarray|pd.Series): Array or Series of close prices

    Returns:
    fr_type: Type indicator (1 for bullish, -1 for bearish, 0 for empty)

    Note: All input arrays should have the same length and represent consecutive time periods in chronological order.
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    fr_type = _fibonacci_range_core(open_prices, high_prices, low_prices, close_prices)
    if index is not None:
        return pd.Series(fr_type, index=index, dtype=np.int32)
    else:
        return fr_type
