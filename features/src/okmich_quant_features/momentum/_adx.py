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


def di_spread(high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14) -> ArrayLike:
    """Directional-movement spread: ``plus_di - minus_di``.

    The canonical ODD combination of the DI pair. ``plus_di`` and ``minus_di`` are each ONE-SIDED:
    under a reflected price path (every log-return negated) ``minus_di`` maps onto ``plus_di``, not
    onto its own negation (measured refl_corr -0.687, conjugate_corr 1.000). A one-sided feature's
    high state means "strong move THIS way" while its low state pools "the other way" WITH "no move
    at all", so a K=2 split on one alone is NOT an up/down partition. The spread is odd by
    construction and is what a directional label actually needs.

    Mirrors ``timothymasters.trend.aroon_diff``, the equivalent combination for the Aroon pair.
    """
    plus = plus_di(high, low, close, period=period)
    minus = minus_di(high, low, close, period=period)
    values = np.asarray(plus, dtype=float) - np.asarray(minus, dtype=float)
    if isinstance(plus, pd.Series):
        return pd.Series(index=plus.index, data=values, name="di_spread")
    return values
