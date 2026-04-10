from typing import Union

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _auto_corr_kernel(values: np.ndarray, window: int, lag: int) -> np.ndarray:
    """
    Numba kernel for rolling lag-k autocorrelation.

    Computes Pearson correlation between values[t-window+1..t] and the same
    window shifted back by `lag` bars, for every bar t.
    """
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(window + lag - 1, n):
        # current window: [i-window+1 .. i]
        # lagged window:  [i-window+1-lag .. i-lag]
        mu_c = 0.0
        mu_l = 0.0
        for j in range(window):
            mu_c += values[i - window + 1 + j]
            mu_l += values[i - window + 1 + j - lag]
        mu_c /= window
        mu_l /= window
        num = 0.0
        ss_c = 0.0
        ss_l = 0.0
        for j in range(window):
            dc = values[i - window + 1 + j] - mu_c
            dl = values[i - window + 1 + j - lag] - mu_l
            num += dc * dl
            ss_c += dc * dc
            ss_l += dl * dl
        denom = (ss_c * ss_l) ** 0.5
        if denom > 0.0:
            out[i] = num / denom
    return out


def auto_corr(series: Union[pd.Series, np.ndarray], window: int, lag: int = 1) -> Union[pd.Series, np.ndarray]:
    """
    Calculate the rolling auto-correlation of a series.

    Computes Pearson correlation coefficient between a series and its lagged version over a rolling window.
    Returns NaN for windows with insufficient data or constant values (zero variance).

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input time series data. Must be numeric.
    window : int
        Size of the rolling window. Must be > lag.
    lag : int, default 1
        Number of periods to lag (1 = correlation with previous value).
        Must be positive and < window.

    Returns
    -------
    pd.Series or np.ndarray
        Rolling auto-correlation values. Same type as input.
        - Values in [-1, 1] for valid correlations
        - NaN for insufficient data or zero-variance windows

    Raises
    ------
    ValueError
        If parameters are invalid or incompatible.
    TypeError
        If series is not numeric.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> auto_corr(s, window=3, lag=1)
    """
    if not isinstance(window, int) or window < 2:
        raise ValueError(f"window must be an integer >= 2, got {window}")

    if not isinstance(lag, int) or lag < 1:
        raise ValueError(f"lag must be a positive integer, got {lag}")

    if lag >= window:
        raise ValueError(f"lag ({lag}) must be < window ({window})")

    is_series = isinstance(series, pd.Series)
    values = series.to_numpy(dtype=np.float64) if is_series else np.asarray(series, dtype=np.float64)

    result = _auto_corr_kernel(values, window, lag)

    return pd.Series(result, index=series.index) if is_series else result