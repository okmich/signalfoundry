from enum import Enum
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import talib


class MovingAverageType(Enum):
    SIMPLE = "simple"
    EXPONENTIAL = "exponential"
    LINEAR_WEIGHTED = "linear_weighted"
    DOUBLE_EXPONENTIAL = "double_exponential"
    TRIPLE_EXPONENTIAL = "triple_exponential"
    SMOOTHED = "smoothed"
    VWAP = "vwap"


def _to_array(data: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
    if isinstance(data, pd.Series):
        return data.values
    return np.array(data)


def _to_request_type(values: np.ndarray, reference: Union[List, np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    if isinstance(reference, pd.Series):
        return pd.Series(values, index=reference.index, name=reference.name)
    return values


def sma(values: Union[List, np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
    original_data = values
    data = _to_array(values)
    result = talib.SMA(data, timeperiod=period)
    return _to_request_type(result, original_data)


def ema(values: Union[List, np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
    original_data = values
    data = _to_array(values)
    result = talib.EMA(data, timeperiod=period)
    return _to_request_type(result, original_data)


def lwma(values: Union[List, np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
    original_data = values
    data = _to_array(values)
    result = talib.WMA(data, timeperiod=period)
    return _to_request_type(result, original_data)


def dema(values: Union[List, np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
    original_data = values
    data = _to_array(values)
    result = talib.DEMA(data, timeperiod=period)
    return _to_request_type(result, original_data)


def tema(values: Union[List, np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
    original_data = values
    data = _to_array(values)
    result = talib.TEMA(data, timeperiod=period)
    return _to_request_type(result, original_data)


def smma(values: Union[List, np.ndarray, pd.Series], period: int) -> Union[np.ndarray, pd.Series]:
    original_data = values
    data = _to_array(values)
    result = np.full(len(data), np.nan, dtype=np.float64)

    if len(data) >= period:
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
    return _to_request_type(result, original_data)


def vwap(values: Union[List, np.ndarray, pd.Series], volumes: Union[List, np.ndarray, pd.Series],
         period: int) -> Union[np.ndarray, pd.Series]:
    """Volume Weighted Moving Average (VWAP)"""
    original_prices = values
    prices = _to_array(values)
    volumes = _to_array(volumes)

    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have the same length")

    result = np.full(len(prices), np.nan, dtype=np.float64)
    for i in range(period - 1, len(prices)):
        price_slice = prices[i - period + 1 : i + 1]
        volume_slice = volumes[i - period + 1 : i + 1]

        if np.sum(volume_slice) != 0:
            result[i] = np.sum(price_slice * volume_slice) / np.sum(volume_slice)

    return _to_request_type(result, original_prices)


def moving_average(ma_type: MovingAverageType, data: Union[List, np.ndarray, pd.Series], period: int,
                   **kwargs) -> Union[np.ndarray, pd.Series]:
    """
    Generic function to return moving average based on type and period
    """

    if ma_type == MovingAverageType.SIMPLE:
        return sma(data, period)
    elif ma_type == MovingAverageType.EXPONENTIAL:
        return ema(data, period)
    elif ma_type == MovingAverageType.LINEAR_WEIGHTED:
        return lwma(data, period)
    elif ma_type == MovingAverageType.DOUBLE_EXPONENTIAL:
        return dema(data, period)
    elif ma_type == MovingAverageType.TRIPLE_EXPONENTIAL:
        return tema(data, period)
    elif ma_type == MovingAverageType.SMOOTHED:
        return smma(data, period)
    elif ma_type == MovingAverageType.VWAP:
        volumes = kwargs.get("volumes")
        if volumes is None:
            raise ValueError("Volumes required for VWAP calculation")
        return vwap(data, volumes, period)
    else:
        raise ValueError(f"Unsupported moving average type: {ma_type}")


def keltner_channels(high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series],
                     close: Union[np.ndarray, pd.Series], ma_type: MovingAverageType,
                     ma_period: int, atr_period: int, atr_multiplier: float = 2.0, **kwargs) -> Tuple[
    Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series], Union[np.ndarray, pd.Series]]:
    # Store original data types for index preservation
    original_close = close

    # Convert to numpy arrays for calculation
    high_arr = _to_array(high)
    low_arr = _to_array(low)
    close_arr = _to_array(close)

    # Validate input lengths
    if not (len(high_arr) == len(low_arr) == len(close_arr)):
        raise ValueError("High, low, and close arrays must have the same length")

    # Calculate the middle line using the specified moving average
    middle_line = moving_average(ma_type, close, ma_period, **kwargs)

    # Calculate ATR
    atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=atr_period)
    atr_series = _to_request_type(atr, original_close)

    # Calculate upper and lower bands
    if isinstance(middle_line, pd.Series):
        upper_band = middle_line + (atr_series * atr_multiplier)
        lower_band = middle_line - (atr_series * atr_multiplier)
    else:
        upper_band = middle_line + (atr * atr_multiplier)
        lower_band = middle_line - (atr * atr_multiplier)

    return upper_band, middle_line, lower_band
