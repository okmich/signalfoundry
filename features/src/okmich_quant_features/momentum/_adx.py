from typing import Union, Tuple

import numpy as np
import pandas as pd
import talib

from ..utils import ensure_numpy_types_for_series

ArrayLike = Union[np.ndarray, pd.Series]
np.ndarray = np.ndarray
PandasArray = pd.Series


def adx(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> ArrayLike:
    # Convert inputs to numpy
    index, high_np = ensure_numpy_types_for_series(high)
    _, low_np = ensure_numpy_types_for_series(low)
    _, close_np = ensure_numpy_types_for_series(close)

    adx_values = talib.ADX(high_np, low_np, close_np, timeperiod=period)

    if index is not None:
        return pd.Series(index=index, data=adx_values, name="adx")
    else:
        return adx_values


def plus_di(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> ArrayLike:
    index, high_np = ensure_numpy_types_for_series(high)
    _, low_np = ensure_numpy_types_for_series(low)
    _, close_np = ensure_numpy_types_for_series(close)
    values = talib.PLUS_DI(high_np, low_np, close_np, timeperiod=period)
    if index is not None:
        return pd.Series(index=index, data=values, name="plus_di")
    return values


def minus_di(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> ArrayLike:
    index, high_np = ensure_numpy_types_for_series(high)
    _, low_np = ensure_numpy_types_for_series(low)
    _, close_np = ensure_numpy_types_for_series(close)
    values = talib.MINUS_DI(high_np, low_np, close_np, timeperiod=period)
    if index is not None:
        return pd.Series(index=index, data=values, name="minus_di")
    return values
