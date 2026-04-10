"""
Feature Window Centering Utilities

This module provides utilities for transforming lagging features into forward-looking labels
by shifting them forward in time with weighted past/future windows.

The primary use case is label generation for supervised learning:
- Takes ANY lagging feature (from okmich_quant_features.*, talib, pandas-ta, etc.)
- Shifts it forward to create a forward-looking label
- No need to create separate forward-looking versions of vast feature libraries

Example:
    # Lagging feature (uses past 120 bars)
    ma = moving_average(close, window=120)

    # Shift it forward by 60 bars to create centered label
    ma_label = center_forward_feature_window(ma, window_size=120, weight=0.5)
    # At time t, ma_label contains the value that will exist at t+60

This creates a training paradigm where models learn to predict t+n/2 (centered) instead
of t+n (fully forward), which is more realistic while still maintaining regime stability through long windows.
"""

from typing import Union, Callable, Any

import pandas as pd


def center_forward_feature_window(
        feature_or_callable: Union[pd.Series, pd.DataFrame, Callable],
        window_size: int,
        weight: float = 0.5,
        **kwargs: Any,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Transform a lagging feature into a forward-looking label by shifting it forward in time.

    This function takes ANY lagging feature (from your feature libraries, talib, pandas-ta, etc.)
    and shifts it forward to create a label with weighted past/future window characteristics.
    This eliminates the need to create separate forward-looking versions of vast feature libraries.

    Parameters
    ----------
    feature_or_callable : pd.Series, pd.DataFrame, or Callable
        - If Series/DataFrame: Pre-calculated lagging feature(s)
        - If Callable: Function that calculates a lagging feature (will be called with **kwargs)
    window_size : int
        Total window size representing the temporal context
    weight : float, default=0.5
        Future weight ranging from 0.0 to 1.0:
        - 0.0: No shift (feature remains fully lagging)
        - 0.5: Centered shift (shift forward by window_size/2)
        - 1.0: Full forward shift (shift forward by window_size)
    **kwargs : Any
        Additional arguments passed to callable (if feature_or_callable is a function)

    Returns
    -------
    pd.Series or pd.DataFrame
        The shifted feature(s) that now "look ahead" in time

    Raises
    ------
    ValueError
        If weight is not in [0, 1] range or window_size is not positive
    TypeError
        If feature_or_callable is not a Series, DataFrame, or Callable

    Notes
    -----
    Shift Calculation:
        - future_bars = int(window_size * weight)
        - Positive shift moves data forward in time
        - At time t, the shifted value represents the feature value at t + future_bars

    NaN Behavior:
        The function creates NaNs at the end due to forward shifting:
        - End: future_bars NaNs (cannot know future values)
        - For weight=0.0: No additional NaNs
        - For weight=0.5: Last 60 bars are NaN (if window_size=120)
        - For weight=1.0: Last 120 bars are NaN (if window_size=120)

    Examples
    --------
    >>> # Example 1: Transform moving average into centered label
    >>> ma = moving_average(close, window=120)  # Lagging feature
    >>> ma_label = center_forward_feature_window(ma, window_size=120, weight=0.5)
    >>> # At time t, ma_label contains the MA value that will exist at t+60

    >>> # Example 2: Transform RSI into forward-looking label
    >>> rsi = talib.RSI(close, timeperiod=14)  # Lagging feature
    >>> rsi_label = center_forward_feature_window(rsi, window_size=120, weight=0.3)
    >>> # At time t, rsi_label contains the RSI value that will exist at t+36

    >>> # Example 3: No shift (weight=0.0) - feature remains lagging
    >>> atr = talib.ATR(high, low, close, timeperiod=20)
    >>> atr_no_shift = center_forward_feature_window(atr, window_size=120, weight=0.0)
    >>> # atr_no_shift is identical to atr (no forward shift)

    >>> # Example 4: Full forward shift (weight=1.0)
    >>> bb_upper = talib.BBANDS(close, timeperiod=20)[0]
    >>> bb_forward = center_forward_feature_window(bb_upper, window_size=120, weight=1.0)
    >>> # At time t, bb_forward contains the BB value that will exist at t+120

    >>> # Example 5: Using with a callable
    >>> ma_label = center_forward_feature_window(
    ...     moving_average,
    ...     window_size=120,
    ...     weight=0.5,
    ...     close=close,
    ...     window=120
    ... )

    >>> # Example 6: Multiple features (DataFrame)
    >>> features = pd.DataFrame({
    ...     'ma': moving_average(close, window=120),
    ...     'rsi': talib.RSI(close, timeperiod=14)
    ... })
    >>> features_labels = center_forward_feature_window(features, window_size=120, weight=0.5)

    See Also
    --------
    pandas.Series.shift : The underlying shift operation
    """
    # Validate weight
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0.0 and 1.0, got {weight}")

    # Validate window_size
    if window_size < 1 or not isinstance(window_size, int):
        raise ValueError(f"window_size must be a positive integer, got {window_size}")

    # If callable, execute it to get the feature
    if callable(feature_or_callable):
        feature = feature_or_callable(**kwargs)
    elif isinstance(feature_or_callable, (pd.Series, pd.DataFrame)):
        feature = feature_or_callable
    else:
        raise TypeError(
            f"feature_or_callable must be a pd.Series, pd.DataFrame, or Callable, "
            f"got {type(feature_or_callable)}"
        )

    # Calculate how many bars to shift forward
    future_bars = int(window_size * weight)

    # Shift to create forward-looking label
    # Negative shift moves values UP (to earlier indices), making them "look ahead"
    # At time t, we get the value that will exist at t + future_bars
    # Example: shift(-60) means result[40] = original[100], so at time 40 we see time 100's value
    forward_feature = feature.shift(-future_bars)

    # Update name to reflect the transformation
    if isinstance(forward_feature, pd.Series):
        original_name = forward_feature.name or "feature"
        forward_feature.name = f"{original_name}_forward_w{weight:.2f}"

    return forward_feature
