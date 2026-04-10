import numpy as np
import pandas as pd


def calculate_log_returns(prices):
    """
    Compute log returns from prices.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series

    Returns
    -------
    pd.Series or np.ndarray
        Log returns (same type as input)
    """
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1))
    else:
        log_rets = np.zeros_like(prices, dtype=float)
        log_rets[1:] = np.log(prices[1:] / prices[:-1])
        log_rets[0] = np.nan
        return log_rets


def calculate_volatility(prices, use_log_returns=True, window=None):
    """
    Calculate volatility (standard deviation of returns).

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    use_log_returns : bool, default=True
        Whether to use log returns (True) or simple returns (False)
    window : int, optional
        Rolling window size. If None, calculates global volatility.

    Returns
    -------
    float or pd.Series
        If window is None: returns single float (global volatility)
        If window is int: returns pd.Series (rolling volatility)
    """
    if use_log_returns:
        returns = calculate_log_returns(prices)
    else:
        if isinstance(prices, pd.Series):
            returns = prices.pct_change()
        else:
            returns = np.zeros_like(prices, dtype=float)
            returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
            returns[0] = np.nan

    if window is None:
        # Global volatility
        if isinstance(returns, pd.Series):
            return returns.std()
        else:
            return np.nanstd(returns)
    else:
        # Rolling volatility
        if isinstance(returns, pd.Series):
            return returns.rolling(window=window).std()
        else:
            # Convert to Series for rolling computation
            returns_series = pd.Series(returns)
            return returns_series.rolling(window=window).std().values


def normalize_by_volatility(targets, price_series, use_log_returns=True):
    """
    Normalize targets by dividing by global volatility.

    Converts targets to "units of volatility" (similar to Sharpe ratio scaling).

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets to normalize
    price_series : pd.Series or np.ndarray
        Price series used to calculate volatility
    use_log_returns : bool, default=True
        Whether to use log returns for volatility calculation

    Returns
    -------
    pd.Series or np.ndarray
        Normalized targets (same type as input)
    """
    volatility = calculate_volatility(price_series, use_log_returns=use_log_returns, window=None)

    if volatility == 0 or np.isnan(volatility):
        # If volatility is zero or NaN, return targets unchanged
        return targets

    if isinstance(targets, pd.Series):
        return targets / volatility
    else:
        return targets / volatility


def clip_by_percentile(targets, percentile=95.0):
    """
    Clip extreme values to specified percentile bounds.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets to clip
    percentile : float, default=95.0
        Percentile for clipping (e.g., 95.0 clips to [5th, 95th] percentiles)

    Returns
    -------
    pd.Series or np.ndarray
        Clipped targets (same type as input)
    """
    if isinstance(targets, pd.Series):
        lower_bound = targets.quantile((100 - percentile) / 100)
        upper_bound = targets.quantile(percentile / 100)
        return targets.clip(lower=lower_bound, upper=upper_bound)
    else:
        lower_bound = np.nanpercentile(targets, 100 - percentile)
        upper_bound = np.nanpercentile(targets, percentile)
        return np.clip(targets, lower_bound, upper_bound)


def clip_by_std(targets, n_std=3.0):
    """
    Clip extreme values to N standard deviations from mean.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Regression targets to clip
    n_std : float, default=3.0
        Number of standard deviations for clipping

    Returns
    -------
    pd.Series or np.ndarray
        Clipped targets (same type as input)
    """
    if isinstance(targets, pd.Series):
        mean = targets.mean()
        std = targets.std()
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        return targets.clip(lower=lower_bound, upper=upper_bound)
    else:
        mean = np.nanmean(targets)
        std = np.nanstd(targets)
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        return np.clip(targets, lower_bound, upper_bound)
