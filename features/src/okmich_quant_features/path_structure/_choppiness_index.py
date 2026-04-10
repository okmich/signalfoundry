from typing import Union

import numpy as np
import pandas as pd
import talib


def choppiness_index(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray], window=14):
    """
    Calculate the Choppiness Index (CHOP) indicator for market trend analysis.

    The Choppiness Index measures the degree of volatility in a market and helps
    identify whether the price is trending or moving sideways. It uses the True Range
    and the highest/lowest prices over a specified period to quantify market activity.

    Parameters
    ----------
    high : pd.Series or np.ndarray
        High prices for each period.
    low : pd.Series or np.ndarray
        Low prices for each period.
    close : pd.Series or np.ndarray
        Close prices for each period.
    window : int, optional
        The lookback period for calculating the Choppiness Index. Default is 14.

    Returns
    -------
    pd.Series or np.ndarray
        Choppiness Index values. Return type matches the input type.

    Market Behavior Segmentation Rules
    ==================================
    The Choppiness Index ranges from 0 to 100 and can be interpreted as follows:

    1. CHOP < 38.2 (Lower Third):
       - Indicates a STRONG DOWNTREND or consolidation at lows
       - Market is directional with strong sell-off momentum
       - Prices are falling with conviction; volatility is being crushed
       - Traders should look for short opportunities or avoid long entries

    2. CHOP between 38.2 and 61.8 (Middle Third):
       - Indicates a SIDEWAYS/CHOPPY market (ranging behavior)
       - Market lacks clear direction; prices are oscillating
       - This is a consolidation or accumulation/distribution phase
       - Breakouts are more likely from this zone
       - Mean-reversion strategies work better; trend-following may produce whipsaws

    3. CHOP > 61.8 (Upper Third):
       - Indicates a STRONG UPTREND or volatility expansion
       - Market is directional with strong buying momentum
       - Prices are rising; volatility is expanding naturally
       - Traders should look for long opportunities or avoid short entries

    Key Insights
    ============
    - Fibonacci levels (38.2 and 61.8) often act as natural transition zones
    - CHOP oscillates between extremes; extreme readings often revert
    - During strong trends, CHOP tends to stay compressed (< 38.2 for downtrends,
      or expands > 61.8 for uptrends depending on volatility)
    - Use CHOP with other indicators (e.g., RSI, MACD) to confirm trend direction
    - CHOP > 80 or CHOP < 20 indicates extremely choppy or extremely trending markets

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> prices = pd.read_csv('price_data.csv')
    >>> chop = create_choppiness_index(prices['High'], prices['Low'],
    ...                                prices['Close'], period=14)
    >>> print(chop.head())
    """

    # Determine input type (Series or ndarray)
    is_series = isinstance(high, pd.Series)
    index = high.index if is_series else None

    # Convert to numpy arrays for calculation if needed
    high_arr = high.values if is_series else np.asarray(high, dtype=np.float64)
    low_arr = low.values if is_series else np.asarray(low, dtype=np.float64)
    close_arr = close.values if is_series else np.asarray(close, dtype=np.float64)

    # Ensure inputs are 1-dimensional
    high_arr = np.atleast_1d(high_arr).flatten()
    low_arr = np.atleast_1d(low_arr).flatten()
    close_arr = np.atleast_1d(close_arr).flatten()

    # Validate inputs
    if not (len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("high, low, and close must have the same length")

    if not isinstance(window, int) or window < 1:
        raise ValueError("period must be a positive integer")

    if window > len(high_arr):
        raise ValueError(
            f"period ({window}) cannot exceed data length ({len(high_arr)})"
        )

    n = len(high_arr)

    # Calculate True Range using talib
    tr = talib.TRANGE(high_arr, low_arr, close_arr)

    # Calculate rolling sum of True Range efficiently using numpy
    tr_sum = np.full(n, np.nan)
    for i in range(window - 1, n):
        tr_sum[i] = np.sum(tr[i - window + 1 : i + 1])

    # Calculate highest high and lowest low using rolling window
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_high = high_arr[i - window + 1 : i + 1]
        window_low = low_arr[i - window + 1 : i + 1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)

    # Calculate Choppiness Index
    # CHOP = 100 * LOG10(SUM(TR, n) / (MAX(H, n) - MIN(L, n))) / LOG10(n)
    chop = np.full(n, np.nan)
    log10_period = np.log10(window)

    valid_idx = window - 1
    range_diff = highest_high[valid_idx:] - lowest_low[valid_idx:]

    # Vectorized calculation for valid indices
    valid_mask = (range_diff > 0) & (tr_sum[valid_idx:] > 0)
    calc_idx = np.arange(valid_idx, n)[valid_mask]

    if len(calc_idx) > 0:
        chop[calc_idx] = 100 * (
            np.log10(tr_sum[calc_idx] / range_diff[valid_mask]) / log10_period
        )

    # Return in the same format as input
    if is_series:
        return pd.Series(chop, index=index, name="CHOP")
    else:
        return chop
