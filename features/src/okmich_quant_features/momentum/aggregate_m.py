import numpy as np
import pandas as pd


def _percent_rank_hlc(close_series, high_series, low_series, period):
    """
    Calculate percent rank of current close vs HLC values over period.

    For each bar, compares current close against all High, Low, Close values
    from the last 'period' bars and returns the percentile rank.

    Optimized using vectorized NumPy operations for better performance.
    """
    # Convert to numpy arrays for faster access
    close_arr = close_series.values
    high_arr = high_series.values
    low_arr = low_series.values

    n = len(close_arr)
    ranks = np.zeros(n)

    # Stack HLC into a single array for efficient sliding window operations
    # Shape: (n, 3) where columns are [high, low, close]
    hlc_stacked = np.column_stack([high_arr, low_arr, close_arr])

    for i in range(period, n):
        current_close = close_arr[i]

        # Extract lookback window (period rows)
        window = hlc_stacked[i - period + 1: i + 1]

        # Flatten to get all HLC values
        hlc_values = window.ravel()

        # Remove current close from comparison (last element)
        hlc_values = hlc_values[:-1]

        # Vectorized comparison - count values less than current close
        count = np.sum(hlc_values < current_close)

        # Calculate percentile rank
        ranks[i] = 100.0 * count / len(hlc_values)

    return ranks


def aggregate_m(ohlcv: pd.DataFrame, slow_period: int = 252, fast_period: int = 10, current_bar_weight: int = 60,
                trend_weight: int = 50, high_column: str = "high", low_column: str = "low",
                close_column: str = "close") -> pd.Series:
    """
    Calculate David Varadi's Aggregate M++ Mean Reversion Oscillator

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV dataframe with columns: 'open', 'high', 'low', 'close', 'volume'
    slow_period : int, default=252
        Larger lookback period (typically 252 for one trading year)
    fast_period : int, default=10
        Smaller lookback period for short-term signals
    current_bar_weight : int, default=60
        Weight for current bar vs previous bar (0-100%)
        Higher values = more responsive to current price action
    trend_weight : int, default=50
        Weight for trend component (0-100%)
        Higher values = more trend-following emphasis
        Lower values = more mean-reversion emphasis

    Returns:
    --------
    pd.Series
        Aggregate M++ values (0-100 scale)
        Values > 50: Bullish / Trend following favorable
        Values < 50: Bearish / Mean reversion favorable
    """
    df_cols = (
        ohlcv.columns.str.lower() if hasattr(ohlcv.columns, "str") else ohlcv.columns
    )
    if not all(col in df_cols for col in [high_column, low_column, close_column]):
        raise ValueError(
            f"DataFrame must contain '{high_column}', '{low_column}', and '{close_column}' columns"
        )

    # Validate parameters
    current_bar_weight = max(0, min(100, current_bar_weight))
    trend_weight = max(0, min(100, trend_weight))

    # Access columns directly without copying - use original casing if needed
    # Check if columns need case conversion
    if high_column in ohlcv.columns:
        high_col = ohlcv[high_column]
        low_col = ohlcv[low_column]
        close_col = ohlcv[close_column]
        index = ohlcv.index
    else:
        # Try lowercase
        high_col = ohlcv[ohlcv.columns[ohlcv.columns.str.lower() == high_column][0]]
        low_col = ohlcv[ohlcv.columns[ohlcv.columns.str.lower() == low_column][0]]
        close_col = ohlcv[ohlcv.columns[ohlcv.columns.str.lower() == close_column][0]]
        index = ohlcv.index

    # Calculate percent ranks for both periods
    hlc_slow = _percent_rank_hlc(close_col, high_col, low_col, slow_period)
    hlc_fast = _percent_rank_hlc(close_col, high_col, low_col, fast_period)

    # Combine fast and slow with trend weighting
    # Original formula: (hlc_slow + hlc_fast) * 0.5
    # Modified: weighted combination based on trend_weight
    m = (hlc_slow * trend_weight + hlc_fast * (100 - trend_weight)) / 100

    # Apply smoothing with current bar weighting (exponential smoothing)
    # Optimized using iterative calculation without explicit loop conditions
    n = len(close_col)
    agg_m = np.zeros(n)
    max_period = max(slow_period, fast_period)

    # Calculate weights once
    back_weight = (100 - current_bar_weight) / 100.0
    current_weight = current_bar_weight / 100.0

    # Initialize first valid value
    if max_period < n:
        agg_m[max_period] = m[max_period]

        # Vectorized-style iterative smoothing
        for i in range(max_period + 1, n):
            agg_m[i] = back_weight * agg_m[i - 1] + current_weight * m[i]

    # Create result series with NaN for initial bars
    result = pd.Series(agg_m, index=index)
    result.iloc[:max_period] = np.nan

    return result
