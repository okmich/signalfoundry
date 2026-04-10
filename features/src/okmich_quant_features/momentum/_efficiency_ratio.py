from typing import Union

import numpy as np
import pandas as pd


def efficiency_ratio(prices: Union[np.ndarray, pd.Series], window: int = 10) -> Union[np.ndarray, pd.Series]:
    """
    Calculate the Efficiency Ratio (Kaufman's Efficiency Ratio).

    The Efficiency Ratio measures directional movement relative to total price movement.
    Range: 0 to 1, where 1 = perfectly trending, 0 = choppy/sideways.

    Formula: ER = |price[t] - price[t-period]| / sum(|price[t] - price[t-1]|)

    Args:
        prices: Input price series (numpy array or pandas Series)
        window: Lookback period for calculation (default: 10). Must be >= 1.

    Returns:
        Same type as input containing efficiency ratios. First 'period' values are NaN.

    Raises:
        ValueError: If period < 1 or prices has fewer than period + 1 elements
        TypeError: If prices is not numpy array or pandas Series
    """
    # Input validation
    if not isinstance(prices, (np.ndarray, pd.Series)):
        raise TypeError("prices must be a numpy array or pandas Series")

    if window < 1:
        raise ValueError("period must be >= 1")

    if len(prices) < window + 1:
        raise ValueError(f"prices must have at least {window + 1} elements, got {len(prices)}")

    # Handle pandas Series
    if isinstance(prices, pd.Series):
        original_index = prices.index
        original_name = prices.name

        # Calculate absolute price changes
        abs_changes = np.abs(prices.diff())

        # Rolling sum of absolute changes (volatility)
        volatility = abs_changes.rolling(window=window).sum()

        # Directional change over the period
        directional_change = np.abs(prices.diff(periods=window))

        # Calculate efficiency ratio, handling division by zero
        result = directional_change / volatility

        # Ensure NaN for invalid values (when volatility is 0)
        result = result.where(volatility != 0, np.nan)

        return result

    # Handle numpy array
    else:
        prices_array = np.asarray(prices, dtype=np.float64)

        # Calculate price changes once (vectorized)
        price_changes = np.diff(prices_array)
        abs_changes = np.abs(price_changes)

        # Pre-allocate output array
        result = np.full(len(prices_array), np.nan, dtype=np.float64)

        # Use cumulative sum for better performance on large arrays
        # cumsum_abs[i] = sum of abs_changes from index 0 to i
        cumsum_abs = np.concatenate([[0], np.cumsum(abs_changes)])

        # Vectorized calculation for period onwards
        # volatility = sum of abs changes over [i-period, i)
        volatility = cumsum_abs[window:] - cumsum_abs[:-window]

        # Directional change = absolute change over the full period
        directional_change = np.abs(prices_array[window:] - prices_array[:-window])

        # Avoid division by zero by using np.divide with where parameter
        result[window:] = np.divide(
            directional_change,
            volatility,
            where=(volatility != 0),
            out=np.zeros(len(directional_change), dtype=np.float64),
        )
        return result
