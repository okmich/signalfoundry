"""
Core helper functions shared across all Timothy Masters indicators.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source files: COMP_VAR.CPP, STATS.CPP.

Functions
---------
atr_kernel          Rolling ATR — Numba JIT (absolute or log mode)
variance_kernel     Rolling historical variance — Numba JIT
normal_cdf_compress Normal-CDF outlier compression (100*Φ(k*x) − 50)
f_cdf_compress      F-distribution CDF compression
"""

import numpy as np
from numba import njit
from scipy.stats import norm as _sp_norm, f as _sp_f


# --------------------------------------------------------------------------- #
# ATR  (COMP_VAR.CPP:29–67)                                                   #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def atr_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int, use_log: bool) -> np.ndarray:
    """
    Rolling Average True Range.
    Replicates Masters' ``atr()`` function exactly, including the two modes:

    * ``use_log=False`` — absolute price differences (standard ATR)
    * ``use_log=True``  — log-ratio true range, appropriate for normalising
      indicators that operate in log(price) space

    Special case: ``length == 0`` returns the current bar's high-low range
    (or log ratio) rather than a rolling average.

    Parameters
    ----------
    high, low, close : 1-D float64 arrays  (same length n)
    length : int
        Lookback period.  ``length == 0`` → single-bar range.
    use_log : bool
        False for absolute mode, True for log-ratio mode.

    Returns
    -------
    out : 1-D float64 array
        Rolling ATR.  The first ``length`` values are NaN (warmup).
    """
    n = len(close)
    out = np.full(n, np.nan)

    for i in range(1, n):
        if length == 0:
            if use_log:
                out[i] = np.log(high[i] / low[i])
            else:
                out[i] = high[i] - low[i]
            continue

        start = i - length + 1
        if start < 1:            # not enough bars yet
            continue

        total = 0.0
        for k in range(start, i + 1):
            if use_log:
                tr = np.log(high[k] / low[k])
                v1 = abs(np.log(high[k] / close[k - 1]))
                v2 = abs(np.log(close[k - 1] / low[k]))
            else:
                tr = high[k] - low[k]
                v1 = abs(high[k] - close[k - 1])
                v2 = abs(close[k - 1] - low[k])

            if v1 > tr:
                tr = v1
            if v2 > tr:
                tr = v2
            total += tr

        out[i] = total / length

    return out


# --------------------------------------------------------------------------- #
# Variance  (COMP_VAR.CPP:77–108)                                             #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def variance_kernel(close: np.ndarray, length: int, use_change: bool) -> np.ndarray:
    """
    Rolling historical variance.
    Replicates Masters' ``variance()`` function.
    * ``use_change=False`` — variance of log(close)
    * ``use_change=True``  — variance of log(close[i] / close[i-1])

    Parameters
    ----------
    close : 1-D float64 array
    length : int    Lookback period.
    use_change : bool

    Returns
    -------
    out : 1-D float64 array
        Rolling variance.  Warmup values are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan)

    for i in range(n):
        if use_change:
            start = i - length + 1
            if start < 1:
                continue
            # Build the change series over [start, i]
            mean_val = 0.0
            for k in range(start, i + 1):
                mean_val += np.log(close[k] / close[k - 1])
            mean_val /= length

            var = 0.0
            for k in range(start, i + 1):
                d = np.log(close[k] / close[k - 1]) - mean_val
                var += d * d
            out[i] = var / length
        else:
            start = i - length + 1
            if start < 0:
                continue
            mean_val = 0.0
            for k in range(start, i + 1):
                mean_val += np.log(close[k])
            mean_val /= length

            var = 0.0
            for k in range(start, i + 1):
                d = np.log(close[k]) - mean_val
                var += d * d
            out[i] = var / length

    return out


# --------------------------------------------------------------------------- #
# Normal-CDF outlier compression  (used throughout COMP_VAR.CPP)             #
# --------------------------------------------------------------------------- #

def normal_cdf_compress(values: np.ndarray, multiplier: float) -> np.ndarray:
    """
    Compress outliers using the normal CDF.

    Masters uses this transformation throughout to map unbounded indicator
    values into a roughly [-50, 50] range:

        output = 100 * Φ(multiplier * value) - 50

    where Φ is the standard normal CDF.  Different indicators use different
    multiplier values (e.g. 1.0, 1.5, 2.0, 4.0, 8.0).

    Parameters
    ----------
    values : array-like
    multiplier : float
        Scaling factor applied before the CDF transform.

    Returns
    -------
    np.ndarray  in approximately [-50, 50]
    """
    return 100.0 * _sp_norm.cdf(multiplier * np.asarray(values, dtype=np.float64)) - 50.0


# --------------------------------------------------------------------------- #
# F-distribution CDF compression  (COMP_VAR.CPP:1018–1051)                  #
# --------------------------------------------------------------------------- #

def f_cdf_compress(values: np.ndarray, df1: float, df2: float, scale: float = 1.0) -> np.ndarray:
    """
    Compress a ratio using the F-distribution CDF.
    Used by the variance-ratio indicators:

        output = 100 * F_CDF(df1, df2, scale * value) - 50

    Parameters
    ----------
    values : array-like
    df1, df2 : float
        Numerator and denominator degrees of freedom.
    scale : float, default 1.0
        Pre-multiplier applied to ``values`` before the CDF (Masters uses
        ``mult`` for the price-variance version).

    Returns
    -------
    np.ndarray  in approximately [-50, 50]
    """
    v = np.asarray(values, dtype=np.float64)
    return 100.0 * _sp_f.cdf(scale * v, df1, df2) - 50.0
