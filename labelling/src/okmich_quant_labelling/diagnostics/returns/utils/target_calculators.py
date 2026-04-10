import numpy as np
import pandas as pd


def calculate_slope(prices, cumr, start_idx, end_idx, direction=1):
    """
    Calculate OLS regression slope (beta) for a trend segment.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    cumr : pd.Series or np.ndarray
        Cumulative returns in basis points
    start_idx : int
        Segment start index
    end_idx : int
        Segment end index
    direction : int, default=1
        Trend direction (+1 for uptrend, -1 for downtrend)

    Returns
    -------
    float
        Slope value (rate of change per bar)
    """
    if end_idx <= start_idx:
        return 0.0

    # Extract segment
    segment_cumr = cumr[start_idx:end_idx + 1]

    if len(segment_cumr) < 2:
        return 0.0

    # Create time indices (0, 1, 2, ...)
    X = np.arange(len(segment_cumr))
    Y = segment_cumr.values if isinstance(segment_cumr, pd.Series) else segment_cumr

    # Calculate OLS beta: cov(X,Y) / var(X)
    X_mean = X.mean()
    Y_mean = Y.mean()

    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sum((X - X_mean) ** 2)

    if denominator == 0:
        return 0.0

    beta = numerator / denominator
    return direction * beta


def calculate_momentum(prices, start_idx, current_idx, direction=1, use_log_returns=True):
    """
    Calculate time-normalized return (momentum).

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    start_idx : int
        Segment start index
    current_idx : int
        Current index
    direction : int, default=1
        Trend direction (+1 for uptrend, -1 for downtrend)
    use_log_returns : bool, default=True
        Whether to use log returns

    Returns
    -------
    float
        Momentum value (return per bar)
    """
    if current_idx <= start_idx:
        return 0.0

    start_price = prices[start_idx]
    current_price = prices[current_idx]

    if start_price == 0 or np.isnan(start_price) or np.isnan(current_price):
        return 0.0

    # Calculate cumulative return
    if use_log_returns:
        cumulative_return = np.log(current_price / start_price)
    else:
        cumulative_return = (current_price - start_price) / start_price

    # Normalize by time
    elapsed_bars = current_idx - start_idx + 1
    momentum = cumulative_return / elapsed_bars

    return direction * momentum


def calculate_cumulative_return(prices, start_idx, current_idx, direction=1, use_log_returns=True):
    """
    Calculate total return from segment start to current point.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    start_idx : int
        Segment start index
    current_idx : int
        Current index
    direction : int, default=1
        Trend direction (+1 for uptrend, -1 for downtrend)
    use_log_returns : bool, default=True
        Whether to use log returns

    Returns
    -------
    float
        Cumulative return
    """
    if current_idx < start_idx:
        return 0.0

    start_price = prices[start_idx]
    current_price = prices[current_idx]

    if start_price == 0 or np.isnan(start_price) or np.isnan(current_price):
        return 0.0

    if use_log_returns:
        cumulative_return = np.log(current_price / start_price)
    else:
        cumulative_return = (current_price - start_price) / start_price

    return direction * cumulative_return


def calculate_amplitude_per_bar(cumr, start_idx, end_idx, duration=None):
    """
    Calculate average amplitude over segment duration.

    Parameters
    ----------
    cumr : pd.Series or np.ndarray
        Cumulative returns in basis points
    start_idx : int
        Segment start index
    end_idx : int
        Segment end index
    duration : int, optional
        Segment duration. If None, calculated as end_idx - start_idx + 1

    Returns
    -------
    float
        Amplitude per bar
    """
    if end_idx <= start_idx:
        return 0.0

    if duration is None:
        duration = end_idx - start_idx + 1

    if duration == 0:
        return 0.0

    total_amplitude = cumr[end_idx] - cumr[start_idx]

    return total_amplitude / duration


def calculate_return_to_extreme(prices, current_idx, extreme_price, direction=1, use_log_returns=True):
    """
    Calculate remaining return from current point to segment extreme.

    This is a FORWARD-LOOKING target (uses future information).

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    current_idx : int
        Current index
    extreme_price : float
        Segment high (for uptrend) or low (for downtrend)
    direction : int, default=1
        Trend direction (+1 for uptrend, -1 for downtrend)
    use_log_returns : bool, default=True
        Whether to use log returns

    Returns
    -------
    float
        Remaining return to extreme
    """
    current_price = prices[current_idx]

    if current_price == 0 or extreme_price == 0 or np.isnan(current_price) or np.isnan(extreme_price):
        return 0.0

    if use_log_returns:
        if direction > 0:  # Uptrend: remaining upside
            remaining_return = np.log(extreme_price / current_price)
        else:  # Downtrend: remaining downside
            remaining_return = -np.log(current_price / extreme_price)
    else:
        if direction > 0:
            remaining_return = (extreme_price - current_price) / current_price
        else:
            remaining_return = -(current_price - extreme_price) / extreme_price

    return remaining_return


def calculate_forward_return(prices, current_idx, end_idx, direction=1, use_log_returns=True):
    """
    Calculate total return from current point to segment end.

    This is a FORWARD-LOOKING target (uses future information).

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    current_idx : int
        Current index
    end_idx : int
        Segment end index
    direction : int, default=1
        Trend direction (+1 for uptrend, -1 for downtrend)
    use_log_returns : bool, default=True
        Whether to use log returns

    Returns
    -------
    float
        Forward return
    """
    if end_idx <= current_idx:
        return 0.0

    current_price = prices[current_idx]
    end_price = prices[end_idx]

    if current_price == 0 or np.isnan(current_price) or np.isnan(end_price):
        return 0.0

    if use_log_returns:
        forward_return = np.log(end_price / current_price)
    else:
        forward_return = (end_price - current_price) / current_price

    return direction * forward_return


def calculate_percentage_from_extreme(prices, current_idx, start_price, direction=1):
    """
    Calculate distance from segment start as percentage.

    Used primarily in AutoLabel framework.

    Parameters
    ----------
    prices : pd.Series or np.ndarray
        Price series
    current_idx : int
        Current index
    start_price : float
        Segment start price
    direction : int, default=1
        Trend direction (+1 for uptrend, -1 for downtrend)

    Returns
    -------
    float
        Percentage distance from start
    """
    current_price = prices[current_idx]

    if start_price == 0 or np.isnan(current_price) or np.isnan(start_price):
        return 0.0

    percentage = (current_price - start_price) / start_price

    return direction * percentage
