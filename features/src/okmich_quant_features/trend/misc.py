from typing import Union, Tuple

import numpy as np
import pandas as pd
import talib

SeriesOrArray = Union[pd.Series, np.ndarray]


def bollinger_band(close_prices: Union[pd.Series, np.ndarray], window: int = 20, deviation_up: float = 2.0,
                   deviation_down: float = 2.0) -> Tuple[SeriesOrArray, SeriesOrArray, SeriesOrArray, SeriesOrArray, SeriesOrArray]:
    if isinstance(close_prices, pd.Series):
        values = close_prices.values
        is_series = True
    else:
        values = close_prices
        is_series = False

    upper_band, middle_band, lower_band = talib.BBANDS(values, timeperiod=window, nbdevup=deviation_up, nbdevdn=deviation_down)

    bb_width = (upper_band - lower_band) / middle_band
    percent_b = (values - lower_band) / (upper_band - lower_band + 1.0e-8)
    if is_series:
        return (
            pd.Series(upper_band, index=close_prices.index, name="upper_band"),
            pd.Series(middle_band, index=close_prices.index, name="middle_band"),
            pd.Series(lower_band, index=close_prices.index, name="lower_band"),
            pd.Series(percent_b, index=close_prices.index, name="percent_b"),
            pd.Series(bb_width, index=close_prices.index, name="bb_width"),
        )
    else:
        return upper_band, middle_band, lower_band, percent_b, bb_width


def cci(high_prices: SeriesOrArray, low_prices: SeriesOrArray, close_prices: SeriesOrArray, window: int = 20) -> SeriesOrArray:
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
