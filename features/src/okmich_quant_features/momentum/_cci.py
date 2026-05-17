"""Commodity Channel Index — normalized mean-distance oscillator.

CCI measures how far the typical price has wandered from its rolling mean, scaled by the mean absolute deviation.
Treat as a momentum / mean-reversion oscillator (neighbor of RSI, Stochastic, Williams %R).
"""
from typing import Union

import numpy as np
import pandas as pd
import talib

SeriesOrArray = Union[pd.Series, np.ndarray]


def cci(high_prices: SeriesOrArray, low_prices: SeriesOrArray, close_prices: SeriesOrArray, window: int = 20) -> SeriesOrArray:
    """Commodity Channel Index via talib.

    CCI = (typical_price - SMA(typical_price, window)) / (0.015 * mean_abs_deviation), where
    typical_price = (H + L + C) / 3. Vol-normalised (by MAD) — values are roughly bounded to
    +/-100 in calm regimes, with extremes around +/-200. Type-preserving.
    """
    if isinstance(close_prices, pd.Series):
        high_values = high_prices.values
        low_values = low_prices.values
        close_values = close_prices.values
        is_series = True
    else:
        high_values = high_prices
        low_values = low_prices
        close_values = close_prices
        is_series = False

    cci_values = talib.CCI(high_values, low_values, close_values, timeperiod=window)
    if is_series:
        return pd.Series(cci_values, index=close_prices.index, name="cci")
    else:
        return cci_values
