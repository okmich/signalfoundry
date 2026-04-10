"""
Volatility / variance ratio indicators #22–23.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source file: COMP_VAR.CPP:1018–1051.

Indicators
----------
22. price_variance_ratio   Ratio of short to long log-price variance
23. change_variance_ratio  Ratio of short to long log-return variance

Both are passed through the F-distribution CDF and compressed to [-50, 50].
"""

import numpy as np

from ._helpers import variance_kernel, f_cdf_compress


def price_variance_ratio(close: np.ndarray, short_period: int = 10, multiplier: float = 4.0) -> np.ndarray:
    """
    Ratio of short to long log-price variance, F-CDF compressed.

    Computes ``variance(log_close, short)`` / ``variance(log_close, long)`` where ``long = short * multiplier``.
    Passed through the F-distribution CDF with ``df1=2, df2=2*mult, scale=mult``:

        output = 100 * F_CDF(2, 2*mult, mult * ratio) - 50

    Parameters
    ----------
    close        : array-like  Close prices.
    short_period : int         Short variance window (default 10).
    multiplier   : float       Long = short * multiplier (default 4.0).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    long_period = int(short_period * multiplier)

    short_var = variance_kernel(close, short_period, False)  # log-price variance
    long_var = variance_kernel(close, long_period, False)

    ratio = short_var / (long_var + 1e-60)

    # NaN where either variance is NaN (warmup)
    ratio[np.isnan(short_var) | np.isnan(long_var)] = np.nan

    out = f_cdf_compress(ratio, df1=2.0, df2=2.0 * multiplier, scale=multiplier)
    out[np.isnan(ratio)] = np.nan
    return out


def change_variance_ratio(close: np.ndarray, short_period: int = 10, multiplier: float = 4.0) -> np.ndarray:
    """
    Ratio of short to long log-return variance, F-CDF compressed.

    Computes ``variance(log_returns, short)`` / ``variance(log_returns, long)``
    where ``long = short * multiplier``.  Passed through the F-distribution CDF with ``df1=4, df2=4*mult``:

        output = 100 * F_CDF(4, 4*mult, ratio) - 50

    Parameters
    ----------
    close        : array-like  Close prices.
    short_period : int         Short variance window (default 10).
    multiplier   : float       Long = short * multiplier (default 4.0).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    long_period = int(short_period * multiplier)

    short_var = variance_kernel(close, short_period, True)   # log-return variance
    long_var = variance_kernel(close, long_period, True)

    ratio = short_var / (long_var + 1e-60)
    ratio[np.isnan(short_var) | np.isnan(long_var)] = np.nan

    out = f_cdf_compress(ratio, df1=4.0, df2=4.0 * multiplier, scale=1.0)
    out[np.isnan(ratio)] = np.nan
    return out
