import numpy as np


def extract_segments(labels: np.ndarray, prices: np.ndarray) -> list[dict]:
    """
    Extract contiguous trend segments from the raw label array.

    Parameters
    ----------
    labels : np.ndarray
        Label array with values -1, 0, +1 (or -1.0, 0.0, +1.0)
    prices : np.ndarray
        Price array

    Returns
    -------
    list[dict]
        List of segment dictionaries with keys:
        - start_idx: first bar index of the segment
        - end_idx: last bar index of the segment
        - direction: +1 (uptrend) or -1 (downtrend)
        - start_price: price at start_idx
        - extreme_price: max price (uptrend) or min price (downtrend) in segment
    """
    segments = []
    n = len(labels)
    i = 0

    while i < n:
        lbl = labels[i]
        if lbl == 0 or lbl == 0.0:  # Handle both int and float zeros
            i += 1
            continue

        # Walk to end of this contiguous block
        start = i
        while i < n and labels[i] == lbl:
            i += 1
        end = i - 1

        seg_prices = prices[start:end + 1]
        extreme = seg_prices.max() if lbl > 0 else seg_prices.min()
        segments.append(
            {
                "start_idx": start,
                "end_idx": end,
                "direction": int(lbl),
                "start_price": prices[start],
                "extreme_price": extreme,
            }
        )

    return segments


def ols_slope(prices: np.ndarray, start: int, current: int, direction: int) -> float:
    """
    Compute OLS slope of log-price over time indices [0, …, current-start].

    Parameters
    ----------
    prices : np.ndarray
        Price array
    start : int
        Start index of the segment
    current : int
        Current index
    direction : int
        Direction of the trend (+1 or -1)

    Returns
    -------
    float
        OLS slope multiplied by direction
    """
    n = current - start + 1
    if n < 2:
        return 0.0

    X = np.arange(n, dtype=float)
    Y = np.log(prices[start:current + 1])

    X_mean = X.mean()
    Y_mean = Y.mean()
    dX = X - X_mean

    denom = np.dot(dX, dX)
    if denom == 0:
        return 0.0

    beta = np.dot(dX, Y - Y_mean) / denom
    return direction * beta
