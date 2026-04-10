import numpy as np
import pandas as pd
from numba import njit

from ..utils import ensure_numpy_types


@njit(fastmath=True, cache=True)
def _precision_trend_core(
    open_prices, high_prices, low_prices, close_prices, ptr_period, ptr_sensitivity
):
    """
    Numba-optimized core computation (works with numpy arrays only)
    """
    n = len(open_prices)

    if n == 0:
        return np.empty(0, dtype=np.int32)

    # Working arrays
    instance_size = 7
    pt_work = np.zeros((n, instance_size), dtype=np.float64)

    # Constants for array indices
    RANGE = 0
    TREND = 1
    AVGR = 2
    AVGD = 3
    AVGU = 4
    MINC = 5
    MAXC = 6

    # Initialize result array
    result = np.zeros(n, dtype=np.int32)

    # Process each bar
    for i in range(n):
        # Calculate range
        pt_work[i, RANGE] = high_prices[i] - low_prices[i]

        # Calculate average range
        avgr_sum = 0.0
        k_count = 0
        for k in range(min(ptr_period, i + 1)):
            avgr_sum += pt_work[i - k, RANGE]
            k_count += 1
        pt_work[i, AVGR] = (avgr_sum / k_count) * ptr_sensitivity

        if i == 0:
            pt_work[i, TREND] = 0
            pt_work[i, AVGD] = close_prices[i] - pt_work[i, AVGR]
            pt_work[i, AVGU] = close_prices[i] + pt_work[i, AVGR]
            pt_work[i, MINC] = close_prices[i]
            pt_work[i, MAXC] = close_prices[i]
        else:
            # Copy previous values
            pt_work[i, TREND] = pt_work[i - 1, TREND]
            pt_work[i, AVGD] = pt_work[i - 1, AVGD]
            pt_work[i, AVGU] = pt_work[i - 1, AVGU]
            pt_work[i, MINC] = pt_work[i - 1, MINC]
            pt_work[i, MAXC] = pt_work[i - 1, MAXC]

            # State machine logic
            prev_trend = int(pt_work[i - 1, TREND])

            if prev_trend == 0:
                if close_prices[i] > pt_work[i - 1, AVGU]:
                    pt_work[i, MINC] = close_prices[i]
                    pt_work[i, AVGD] = close_prices[i] - pt_work[i, AVGR]
                    pt_work[i, TREND] = 1
                if close_prices[i] < pt_work[i - 1, AVGD]:
                    pt_work[i, MAXC] = close_prices[i]
                    pt_work[i, AVGU] = close_prices[i] + pt_work[i, AVGR]
                    pt_work[i, TREND] = 2
            elif prev_trend == 1:
                pt_work[i, AVGD] = pt_work[i - 1, MINC] - pt_work[i, AVGR]
                if close_prices[i] > pt_work[i - 1, MINC]:
                    pt_work[i, MINC] = close_prices[i]
                if close_prices[i] < pt_work[i - 1, AVGD]:
                    pt_work[i, MAXC] = close_prices[i]
                    pt_work[i, AVGU] = close_prices[i] + pt_work[i, AVGR]
                    pt_work[i, TREND] = 2
            elif prev_trend == 2:
                pt_work[i, AVGU] = pt_work[i - 1, MAXC] + pt_work[i, AVGR]
                if close_prices[i] < pt_work[i - 1, MAXC]:
                    pt_work[i, MAXC] = close_prices[i]
                if close_prices[i] > pt_work[i - 1, AVGU]:
                    pt_work[i, MINC] = close_prices[i]
                    pt_work[i, AVGD] = close_prices[i] - pt_work[i, AVGR]
                    pt_work[i, TREND] = 1

        trend_value = int(pt_work[i, TREND])
        if trend_value == 1:
            result[i] = 1
        elif trend_value == 2:
            result[i] = -1
        else:
            result[i] = 0

    return result


def precision_trend(
    open_prices, high_prices, low_prices, close_prices, ptr_period=14, ptr_sensitivity=3
):
    """
    Precision Trend indicator that accepts both numpy arrays and pandas Series.

    Parameters:
    open_prices, high_prices, low_prices, close_prices: array-like
        Can be numpy arrays, pandas Series, or any array-like structure
    ptr_period (int): Precision trend period
    ptr_sensitivity (float): Precision trend sensitivity

    Returns:
    numpy.ndarray or pandas.Series: Array with values 1 (bullish), -1 (bearish)
        Returns pandas Series with same index as input if inputs are Series,
        otherwise returns numpy array
    """
    index, open_prices, high_prices, low_prices, close_prices = ensure_numpy_types(
        open_prices, high_prices, low_prices, close_prices
    )

    result = _precision_trend_core(
        open_prices, high_prices, low_prices, close_prices, ptr_period, ptr_sensitivity
    )
    if index is not None:
        return pd.Series(result, index=index, dtype=np.int32)
    else:
        return result
