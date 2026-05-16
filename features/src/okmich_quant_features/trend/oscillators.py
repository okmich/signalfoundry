from typing import Union, Tuple

import numpy as np
import pandas as pd
import talib

SeriesOrArray = Union[pd.Series, np.ndarray]


def bollinger_band(close_prices: Union[pd.Series, np.ndarray], window: int = 20, deviation_up: float = 2.0,
                   deviation_down: float = 2.0) -> Tuple[SeriesOrArray, SeriesOrArray, SeriesOrArray, SeriesOrArray, SeriesOrArray]:
    """Bollinger Bands via talib: SMA(close, window) +/- (deviation * stdev(close, window)).

    Returns (upper, middle, lower, percent_b, bb_width):
        percent_b = (close - lower) / (upper - lower + eps)   — position within band, 0..1 inside
        bb_width  = (upper - lower) / middle                  — width relative to mid

    Type-preserving: pd.Series in -> 5 pd.Series out (named); np.ndarray in -> 5 np.ndarray out.
    """
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


def envelope(close_prices: SeriesOrArray, high_prices: SeriesOrArray, low_prices: SeriesOrArray, ma_period: int = 20,
             atr_period: int = 14, k_atr: float = 2.0) -> Tuple[SeriesOrArray, SeriesOrArray, SeriesOrArray, SeriesOrArray, SeriesOrArray]:
    """ATR envelope: middle = SMA(close, ma_period); bands = middle +/- k_atr * ATR(atr_period).

    The ATR-based analog of Bollinger Bands. Where Bollinger uses standard deviation of close for the band width, this
    uses Wilder ATR (talib), making the bands scale-equivariant across instruments and more robust to gaps / fat tails.

    Returns (upper_envelope, middle_envelope, lower_envelope, percent_e, env_width):
        upper_envelope  = middle + k_atr * ATR
        middle_envelope = SMA(close, ma_period)
        lower_envelope  = middle - k_atr * ATR
        percent_e       = (close - lower) / (upper - lower + eps)   — analog of %B
        env_width       = (upper - lower) / middle                  — width relative to mid
    """
    if isinstance(close_prices, pd.Series):
        close_values = close_prices.values
        high_values = high_prices.values
        low_values = low_prices.values
        is_series = True
    else:
        close_values = close_prices
        high_values = high_prices
        low_values = low_prices
        is_series = False

    middle_band = talib.SMA(close_values, timeperiod=ma_period)
    atr_values = talib.ATR(high_values, low_values, close_values, timeperiod=atr_period)
    upper_band = middle_band + k_atr * atr_values
    lower_band = middle_band - k_atr * atr_values

    env_width = (upper_band - lower_band) / middle_band
    percent_e = (close_values - lower_band) / (upper_band - lower_band + 1.0e-8)

    if is_series:
        return (
            pd.Series(upper_band, index=close_prices.index, name="upper_envelope"),
            pd.Series(middle_band, index=close_prices.index, name="middle_envelope"),
            pd.Series(lower_band, index=close_prices.index, name="lower_envelope"),
            pd.Series(percent_e, index=close_prices.index, name="percent_e"),
            pd.Series(env_width, index=close_prices.index, name="env_width"),
        )
    else:
        return upper_band, middle_band, lower_band, percent_e, env_width


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
