"""Non-directional path-shape statistics derived from the return series.

These features measure the *shape* of the recent return path rather than its direction:

    velocity_consistency     — std(returns) / |mean(returns)|, coefficient of variation
    velocity_magnitude       — |mean(log_returns)|, direction-agnostic momentum strength
    returns_sign_persistence — fraction of returns matching the first return's sign in the window

Neighbors of `efficiency_ratio` and `choppiness_index` — all answer "how trendy / consistent / noisy
is this path?" without caring about up/down.
"""
import numpy as np
import pandas as pd


def velocity_magnitude(close: pd.Series, lookback: int) -> pd.Series:
    """|mean(log_returns over lookback)| — direction-agnostic momentum strength.

    High values indicate sustained movement in some direction (trending). Low values indicate
    no net movement (ranging or mean-reverting).
    """
    log_returns = np.log(close / close.shift(1))
    magnitude = log_returns.rolling(window=lookback).mean().abs()
    magnitude.name = f"velocity_magnitude_back_{lookback}"
    return magnitude


def velocity_consistency(close: pd.Series, lookback: int) -> pd.Series:
    """Coefficient of variation of log returns: std(returns) / |mean(returns)|.

    Low values indicate smooth, consistent momentum (high-quality trends). High values indicate
    choppy, noisy returns. Inf results (when mean is near zero) are replaced with NaN.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2 for consistency calculation")

    log_returns = np.log(close / close.shift(1))
    rolling_mean = log_returns.rolling(window=lookback).mean()
    rolling_std = log_returns.rolling(window=lookback).std()

    consistency = rolling_std / rolling_mean.abs()
    consistency = consistency.replace([np.inf, -np.inf], np.nan)
    consistency.name = f"velocity_consistency_back_{lookback}"
    return consistency


def returns_sign_persistence(returns_series: pd.Series, window: int) -> pd.Series:
    """Rolling fraction of return signs matching the first sign in the window — directional persistence.

    Output in [0, 1]: 1.0 means every return in the window has the same sign as the first one (perfect
    persistence); 0.5 means random; near-0 means the first sign is repeatedly flipped against.
    """
    signs = np.sign(returns_series)
    return signs.rolling(window).apply(lambda x: np.mean(x == x[0]), raw=True)
