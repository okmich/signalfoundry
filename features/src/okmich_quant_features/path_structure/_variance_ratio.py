from typing import Union, Optional

import numpy as np
import pandas as pd


def variance_ratio(series: pd.Series, window: Optional[int], q: int = 2, min_periods: Union[int, None] = None) -> Union[pd.Series, float]:
    """
    Compute rolling variance ratio for time-varying regime detection or the variance ratio for the series if window
    is not specified.

    This function calculates the variance ratio over a rolling window, allowing
    detection of regime changes between trending and mean-reverting behavior.

    Parameters
    ----------
    series : pd.Series
        Return series (not price levels).
    window : Optional int
        Rolling window size for variance calculation. If note, the variance ratio is calculated for the entire series
    q : int, default=2
        Aggregation lag (q-period).
    min_periods : int, optional
        Minimum observations in window. Defaults to window.

    Returns
    -------
    Union[pd.Series, float]
        Rolling variance ratio series with same index as input if window is specified.
        float if window is None
            Variance ratio statistic:
            - VR > 1: positive serial correlation (trending)
            - VR < 1: negative serial correlation (mean reversion)
            - VR = 1: random walk
            Returns np.nan if insufficient data or zero variance.

    Notes
    -----
    Formula: VR(q) = Var_t(r_{t-q+1:t}) / (q * Var_t(r_t))
    where Var_t denotes variance computed over rolling window.

    Use Cases
    ---------
    - Regime detection
    - Microstructure detection: identify stale prices or liquidity issues
    - Intraday patterns: 5-min returns with q=12 for hourly patterns

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(500),
    ...                     index=pd.date_range('2020-01-01', periods=500))
    >>> rolling_vr = variance_ratio(returns, window=252, q=5)
    >>> # Values > 1.2 might indicate trending regime
    >>> # Values < 0.8 might indicate mean-reversion regime
    """
    if min_periods is None:
        min_periods = window

    if window is None:
        return _variance_ratio_single(series, lag=q, min_periods=min_periods)

    # Validate inputs
    if window < q + 1:
        raise ValueError(f"window ({window}) must be >= q + 1 ({q + 1})")

    if len(series) < min_periods:
        return pd.Series(np.nan, index=series.index)

    # Single-period variance (denominator)
    var1 = series.rolling(window=window, min_periods=min_periods).var(ddof=1)

    # q-period aggregated returns
    agg_returns = series.rolling(window=q, min_periods=q).sum()

    # Variance of q-period returns (numerator)
    varq = agg_returns.rolling(window=window, min_periods=min_periods).var(ddof=1)

    # Compute variance ratio, handling division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = varq / (q * var1)

    return vr


def _variance_ratio_single(
    series: pd.Series, lag: int = 2, min_periods: Union[int, None] = None
) -> float:
    if min_periods is None:
        min_periods = lag + 1

    if len(series) < min_periods:
        return np.nan

    # Use ddof=1 for unbiased variance estimator
    one_var = series.var(ddof=1)

    if one_var <= 0 or np.isnan(one_var):
        return np.nan

    # Compute q-period overlapping returns
    lagged_returns = series.rolling(window=lag, min_periods=lag).sum()

    # Drop NaN values introduced by rolling
    lagged_returns = lagged_returns.dropna()

    if len(lagged_returns) < 2:
        return np.nan

    lag_var = lagged_returns.var(ddof=1)

    # Variance ratio: should be ~1 under random walk
    vr = lag_var / (lag * one_var)
    return vr


def variance_ratio_test(
    series: pd.Series, lags: list[int] = [2, 5, 10], return_stats: bool = False
) -> Union[pd.DataFrame, tuple[pd.DataFrame, dict]]:
    """
    Compute variance ratios for multiple lags with z-statistics.

    Parameters
    ----------
    series : pd.Series
        Return series.
    lags : list of int
        List of lags to test.
    return_stats : bool, default=False
        If True, return additional statistics.

    Returns
    -------
    pd.DataFrame or tuple
        DataFrame with VR and z-statistics for each lag.
        If return_stats=True, also returns dict with additional info.

    Notes
    -----
    Under the random walk null hypothesis with i.i.d. returns:
    - VR(q) ~ N(1, 2(2q-1)(q-1)/(3qT)) for large T
    - z-stat = (VR - 1) / sqrt(variance)

    Examples
    --------
    >>> returns = pd.Series(np.random.randn(1000))
    >>> results = variance_ratio_test(returns, lags=[2, 5, 10, 20])
    >>> print(results)
    """
    n = len(series)
    results = []

    for q in lags:
        vr = _variance_ratio_single(series, lag=q)

        # Asymptotic variance under homoskedastic null
        # Var(VR) = 2(2q-1)(q-1) / (3qT)
        vr_variance = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
        vr_std = np.sqrt(vr_variance)

        # Z-statistic
        z_stat = (vr - 1) / vr_std if vr_std > 0 else np.nan

        # Two-tailed p-value
        from scipy import stats

        p_value = (
            2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan
        )

        results.append({"lag": q, "VR": vr, "z_stat": z_stat, "p_value": p_value})

    df = pd.DataFrame(results)

    if return_stats:
        stats_dict = {
            "n_obs": n,
            "mean": series.mean(),
            "std": series.std(),
            "skew": series.skew(),
            "kurt": series.kurtosis(),
        }
        return df, stats_dict

    return df
