from itertools import product
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd


def quantile_based_volatility_labeling(price_series: pd.Series, window: int = 20, upper_q: float = 0.7,
                                       lower_q: float = 0.3, name: str = "vol_label") -> pd.Series:
    """
    This function identifies high, low, and normal volatility periods by comparing current volatility to recent historical
    volatility distribution. It calculates rolling volatility (standard deviation of log returns) and then determines
    dynamic thresholds using rolling quantiles to classify each period.

    Parameters
    ----------
    price_series : pd.Series
        Series of price data (typically closing prices) with datetime index.
    window : int, default 20
        Lookback window for calculating volatility and quantiles. Determines
        the number of periods to consider for the historical distribution.
    upper_q : float, default 0.7
        Upper quantile threshold (0-1). Volatility above this quantile of recent
        history is classified as high volatility.
    lower_q : float, default 0.3
        Lower quantile threshold (0-1). Volatility below this quantile of recent
        history is classified as low volatility.
    name : str, default 'vol_label'
        Name for the output Series.

    Returns
    -------
    pd.Series
        Series with volatility regime trend:
        - 1.0: High volatility (current volatility ≥ upper_q quantile of recent window)
        - -1.0: Low volatility (current volatility ≤ lower_q quantile of recent window)
        - 0.0: Normal volatility (between the thresholds)
        The Series has the same index as price_series and specified name.

    Raises
    ------
    ValueError
        If upper_q is not greater than lower_q, or if quantile thresholds
        are not between 0 and 1.

    Notes
    -----
    The algorithm follows these steps:
    1. Calculate log returns from the price series
    2. Compute rolling standard deviation (volatility) over the specified window
    3. Calculate rolling quantiles of volatility for upper and lower thresholds
    4. Compare current volatility to these dynamic thresholds to assign trend

    The rolling quantile approach creates adaptive thresholds that respond to
    changing market conditions, making it more responsive than fixed thresholds.

    Examples
    --------
    >>> price_data = pd.Series([100, 101, 103, 102, 105, ...])
    >>> trend = quantile_based_volatility_labeling(price_data,
    ...                                                     window=30,
    ...                                                     upper_q=0.75,
    ...                                                     lower_q=0.25)
    >>> trend.value_counts()
    -1.0    0.25  # 25% low volatility periods
     0.0    0.50  # 50% normal volatility periods
     1.0    0.25  # 25% high volatility periods
    """
    if upper_q <= lower_q:
        raise ValueError("upper_q must be greater than lower_q")
    if not (0 <= lower_q <= 1 and 0 <= upper_q <= 1):
        raise ValueError("Quantile thresholds must be between 0 and 1")

    ret = np.log(price_series / price_series.shift())
    vol = ret.rolling(window).std()

    uq = vol.rolling(window=window).quantile(upper_q)
    lq = vol.rolling(window=window).quantile(lower_q)
    conditions = [
        (vol >= uq),  # High volatility
        (vol <= lq),  # Low volatility
    ]
    choices = [1.0, -1.0]
    lab = pd.Series(
        np.select(conditions, choices, default=0.0),
        index=price_series.index,
        name=name,
        dtype=np.float64,
    )
    return lab


def optimize_quantile_based_volatility_labels(price_series: pd.Series, window_range: List[int] = None,
                                              upper_q_range: List[float] = None,
                                              lower_q_range: List[float] = None, target_ratio: float = 0.2,
                                              verbose: bool = False) \
        -> Tuple[Dict[str, Any], pd.Series]:
    """
    Optimizes hyperparameters for the volatility trend function by finding parameters that produce a
    desired distribution of trend.

    The goal of optimizing is to find the parameter set that produce a reasonable distribution of 1, 0, and -1
    volatility trend.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with price data
    price_col : str
        Name of the price column
    window_range : List[int]
        List of window values to test
    upper_q_range : List[float]
        List of upper quantile values to test
    lower_q_range : List[float]
        List of lower quantile values to test
    target_ratio : float
        Target ratio of non-zero trend (high + low volatility) to total periods.
        Helps avoid over-fitting to extreme parameter sets.

    Returns:
    --------
    Tuple[Dict[str, Any], pd.Series]
        Best parameters and the corresponding label series
    """
    if window_range is None:
        window_range = [10, 20, 30, 50]
    if upper_q_range is None:
        upper_q_range = [0.6, 0.7, 0.8]
    if lower_q_range is None:
        lower_q_range = [0.2, 0.3, 0.4]

    ret = np.log(price_series / price_series.shift())

    best_params = None
    best_labels = None
    best_score = float("inf")
    best_distribution = None

    param_combinations = list(product(window_range, upper_q_range, lower_q_range))

    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations...")

    for i, (window, upper_q, lower_q) in enumerate(param_combinations):
        if upper_q <= lower_q:
            continue

        # Use the same vol definition as the labeling function: return-vol, not vol-of-vol.
        vol = ret.rolling(window).std()
        uq = vol.rolling(window=window).quantile(upper_q)
        lq = vol.rolling(window=window).quantile(lower_q)

        # Create trend
        conditions = [(vol >= uq), (vol <= lq)]
        choices = [1.0, -1.0]
        labels = np.select(conditions, choices, default=0.0)
        labels_series = pd.Series(labels, index=price_series.index, dtype=np.float64)

        unique, counts = np.unique(labels, return_counts=True)
        dist_dict = dict(zip(unique, counts / len(labels[~np.isnan(labels)])))

        high_pct = dist_dict.get(1.0, 0)
        low_pct = dist_dict.get(-1.0, 0)
        neutral_pct = dist_dict.get(0.0, 0)
        non_zero_pct = high_pct + low_pct

        ratio_penalty = abs(non_zero_pct - target_ratio) * 10  # Weighted penalty
        imbalance_penalty = abs(high_pct - low_pct) * 5  # Penalize imbalance

        score = ratio_penalty + imbalance_penalty

        if score < best_score:
            best_score = score
            best_params = {
                "window": window,
                "upper_q": upper_q,
                "lower_q": lower_q,
                "score": score,
                "distribution": {
                    "high_vol_pct": high_pct,
                    "low_vol_pct": low_pct,
                    "neutral_pct": neutral_pct,
                    "non_zero_pct": non_zero_pct,
                },
            }
            best_labels = labels_series

        if verbose and (i + 1) % 50 == 0:
            print(f"Tested {i + 1}/{len(param_combinations)} combinations")

    if verbose:
        print("Optimization complete!")
    return best_params, best_labels
