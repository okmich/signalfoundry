"""Normalized MA-based trend features.

Each `norm_<type>` primitive returns ``log(close / MA(close, period))`` — the scale-free signed distance of price
from its moving average. Positive when price is above the MA (above trend), negative when below. Comparable across
instruments; additive across horizons (log returns compose).

For raw smoothing of any series, use filters.py (smooth_ema, smooth_sma, smooth_wma, smooth_kalman, ...).
This module is for *trend feature* MA usage; filters.py is for *smoothing*.
"""
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd
import talib

SeriesOrArray = Union[np.ndarray, pd.Series]


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


def _to_request_type(values: np.ndarray, reference: Union[List, np.ndarray, pd.Series]) -> SeriesOrArray:
    if isinstance(reference, pd.Series):
        return pd.Series(values, index=reference.index, name=reference.name)
    return values


def norm_sma(values: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / SMA(close, period)) — scale-free signed distance from SMA. Causal."""
    original_data = values
    data = _to_array(values)
    result = talib.SMA(data, timeperiod=period)
    return _to_request_type(np.log(values / result), original_data)


def norm_ema(values: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / EMA(close, period)) — scale-free signed distance from EMA. Causal."""
    original_data = values
    data = _to_array(values)
    result = talib.EMA(data, timeperiod=period)
    return _to_request_type(np.log(values / result), original_data)


def norm_lwma(values: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / WMA(close, period)) — scale-free signed distance from linear-weighted MA. Causal."""
    original_data = values
    data = _to_array(values)
    result = talib.WMA(data, timeperiod=period)
    return _to_request_type(np.log(values / result), original_data)


def norm_dema(values: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / DEMA(close, period)) — scale-free signed distance from double-EMA. Causal."""
    original_data = values
    data = _to_array(values)
    result = talib.DEMA(data, timeperiod=period)
    return _to_request_type(np.log(values / result), original_data)


def norm_tema(values: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / TEMA(close, period)) — scale-free signed distance from triple-EMA. Causal."""
    original_data = values
    data = _to_array(values)
    result = talib.TEMA(data, timeperiod=period)
    return _to_request_type(np.log(values / result), original_data)


def norm_smma(values: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / SMMA(close, period)) — scale-free signed distance from Wilder-smoothed MA. Causal."""
    original_data = values
    data = _to_array(values)
    result = np.full(len(data), np.nan, dtype=np.float64)

    if len(data) >= period:
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
    return _to_request_type(np.log(values / result), original_data)


def norm_vwap(values: SeriesOrArray, volumes: SeriesOrArray, period: int) -> SeriesOrArray:
    """log(close / VWAP(close, volume, period)) — scale-free signed distance from rolling VWAP. Causal."""
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

    return _to_request_type(np.log(values / result), values)


def norm_moving_average(ma_type: MovingAverageType, data: SeriesOrArray, period: int, **kwargs) -> SeriesOrArray:
    """Dispatcher returning log(close / MA(close, period)) for the configured MA type."""
    if ma_type == MovingAverageType.SIMPLE:
        return norm_sma(data, period)
    elif ma_type == MovingAverageType.EXPONENTIAL:
        return norm_ema(data, period)
    elif ma_type == MovingAverageType.LINEAR_WEIGHTED:
        return norm_lwma(data, period)
    elif ma_type == MovingAverageType.DOUBLE_EXPONENTIAL:
        return norm_dema(data, period)
    elif ma_type == MovingAverageType.TRIPLE_EXPONENTIAL:
        return norm_tema(data, period)
    elif ma_type == MovingAverageType.SMOOTHED:
        return norm_smma(data, period)
    elif ma_type == MovingAverageType.VWAP:
        volumes = kwargs.get("volumes")
        if volumes is None:
            raise ValueError("Volumes required for VWAP calculation")
        return norm_vwap(data, volumes, period)
    else:
        raise ValueError(f"Unsupported moving average type: {ma_type}")


_TALIB_MA_FN = {
    MovingAverageType.SIMPLE: talib.SMA,
    MovingAverageType.EXPONENTIAL: talib.EMA,
    MovingAverageType.LINEAR_WEIGHTED: talib.WMA,
    MovingAverageType.DOUBLE_EXPONENTIAL: talib.DEMA,
    MovingAverageType.TRIPLE_EXPONENTIAL: talib.TEMA,
}


def ma_slope_norm(close: SeriesOrArray, period: int, ma_type: MovingAverageType = MovingAverageType.EXPONENTIAL,
                  atr_period: int = 14, slope_window: int = 5,
                  high: SeriesOrArray = None, low: SeriesOrArray = None) -> SeriesOrArray:
    """Per-bar slope of the raw MA, normalized by ATR — trend strength comparable across instruments.

        slope_per_bar = (MA[t] - MA[t-slope_window]) / slope_window
        ma_slope_norm = slope_per_bar / ATR(atr_period)

    Reads in units of "ATRs per bar". The raw MA is computed via talib directly (not via norm_moving_average,
    which returns log-distance). Restricted to talib-native MA flavors: SIMPLE, EXPONENTIAL, LINEAR_WEIGHTED,
    DOUBLE_EXPONENTIAL, TRIPLE_EXPONENTIAL.

    ATR is computed from high/low/close when supplied; otherwise falls back to a close-only proxy
    (rolling std of close first differences over atr_period).
    """
    if ma_type not in _TALIB_MA_FN:
        raise ValueError(f"ma_slope_norm supports only talib-native MA types {list(_TALIB_MA_FN)}; got {ma_type}")

    close_arr = _to_array(close).astype(np.float64)
    ma_arr = _TALIB_MA_FN[ma_type](close_arr, timeperiod=period)

    slope = np.full_like(close_arr, np.nan, dtype=np.float64)
    slope[slope_window:] = (ma_arr[slope_window:] - ma_arr[:-slope_window]) / slope_window

    if high is not None and low is not None:
        atr = talib.ATR(_to_array(high), _to_array(low), close_arr, timeperiod=atr_period)
    else:
        diffs = np.diff(close_arr, prepend=close_arr[0])
        atr = pd.Series(diffs).rolling(atr_period).std().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        result = slope / atr
    return _to_request_type(result, close)
