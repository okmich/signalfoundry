import numpy as np
import pandas as pd

import talib

from sklearn.linear_model import LinearRegression


def momentum_slope(series: pd.Series, window: int = 20):
    slopes = []
    lr = LinearRegression()
    for i in range(len(series)):
        if i + window < len(series):
            y = series.iloc[i: i + window].values.reshape(-1, 1)
            x = np.arange(window).reshape(-1, 1)
            if np.any(np.isnan(y)):
                slopes.append(np.nan)
            else:
                lr.fit(x, y)
                slopes.append(lr.coef_[0][0])
        else:
            slopes.append(np.nan)
    return slopes


def returns_sign_persistence(returns_series: pd.Series, window):
    signs = np.sign(returns_series)
    return signs.rolling(window).apply(lambda x: np.mean(x == x[0]), raw=True)


def high_proximity(close: pd.Series, lookback: int = 200) -> pd.Series:
    """
    Long-term High Proximity: how close price is to 52-bar high.

    Parameters
    ----------
    close : pd.Series
    lookback : int
        Period to define high (default ~1 year).

    Returns
    -------
    pd.Series
        Ratio of (close - min) / (max - min).

    High-Impact Use Case
    --------------------
    - Identifying structural strength / breakout readiness.
    - Equities and commodities (where yearly highs matter).
    """
    rolling_high = talib.MAX(close.values, timeperiod=lookback)
    rolling_low = talib.MIN(close.values, timeperiod=lookback)
    return (close - rolling_low) / (rolling_high - rolling_low + 1e-9)


def rolling_high_proximity(close: pd.Series, lookback: int = 60) -> pd.Series:
    """
    Short-term High Proximity: intraday or short-horizon extension.

    Parameters
    ----------
    close : pd.Series
    lookback : int
        Shorter lookback period.

    Returns
    -------
    pd.Series
        Proximity to recent high.

    High-Impact Use Case
    --------------------
    - Intraday breakout/mean-reversion decisions.
    - FX and index futures for short-term bias.
    """
    high_ = close.rolling(lookback).max()
    low_ = close.rolling(lookback).min()
    return (close - low_) / (high_ - low_ + 1e-9)


def session_high_low_pct(
        high: pd.Series, low: pd.Series, close: pd.Series, session_window: int = 60
) -> pd.Series:
    """
    Session High-Low Percent: position of price in current session range.

    Parameters
    ----------
    high : pd.Series
    low : pd.Series
    close : pd.Series
    session_window : int
        Number of bars per session (e.g., 60 for 1h).

    Returns
    -------
    pd.Series
        Ratio of (price - low) / (high - low).

    High-Impact Use Case
    --------------------
    - Intraday momentum scalping or VWAP-based bias detection.
    - Equities, FX, or futures (session-bound behavior).
    """
    session_high = high.rolling(session_window).max()
    session_low = low.rolling(session_window).min()
    return (close - session_low) / (session_high - session_low + 1e-9)


def lagged_return_skip(df: pd.DataFrame, lag: int = 12, skip: int = 1) -> pd.Series:
    """
    Implements the '12-1' momentum skip logic to calculate lagged returns while skipping recent periods.

    This approach avoids microstructure noise and short-term mean reversion effects by excluding
    the most recent price movements from the momentum calculation.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'close' prices
    lag : int, default=12
        Number of periods to look back for price comparison
    skip : int, default=1
        Number of recent periods to skip to avoid noise

    Returns:
    --------
    pd.Series
        Lagged percentage returns skipping recent periods

    High-Impact Use Cases:
    ----------------------
    - **Trend Continuation Signals**: Strong positive returns with skip logic indicate persistent trends
    - **Regime Filtering**: Combine with other indicators to filter high-probability momentum regimes
    - **Overnight Gap Analysis**: Use with daily data to capture momentum skipping overnight gaps
    - **Mean Reversion Avoidance**: Skip recent mean-reverting periods in momentum strategies

    Example:
    --------
    >>> # Detect persistent uptrends avoiding recent noise
    >>> momentum = lagged_return_skip(df, lag=12, skip=1)
    >>> strong_trend = (momentum > 0.02) & (momentum.shift(1) > 0.015)
    """
    return df["close"].pct_change(lag + skip).shift(skip)


def pr_skip(df: pd.DataFrame, lag: int = 12, skip: int = 1) -> pd.Series:
    """
    Alternative price-return skip logic for robust momentum measurement.

    Provides a different calculation method for skip-period momentum that can be more stable
    than simple percentage changes, especially during high volatility periods.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'close' prices
    lag : int, default=12
        Lookback period for price comparison
    skip : int, default=1
        Recent periods to skip for noise reduction

    Returns:
    --------
    pd.Series
        Price ratio-based returns with skip logic

    High-Impact Use Cases:
    ----------------------
    - **Volatility-Adjusted Momentum**: More stable than pct_change during high volatility
    - **Multi-Timeframe Confirmation**: Use different lags to confirm momentum across timeframes
    - **Breakout Validation**: Validate price breakouts with skip-period momentum confirmation
    - **Portfolio Sorting**: Rank securities by robust momentum for long-short strategies

    Example:
    --------
    >>> # Confirm breakout with multi-timeframe momentum
    >>> short_term = pr_skip(df, lag=5, skip=1)
    >>> long_term = pr_skip(df, lag=20, skip=2)
    >>> breakout_confirmed = (short_term > 0) & (long_term > 0) & (df['close'] > resistance)
    """
    return (df["close"] / df["close"].shift(lag + skip)) - 1


def lagged_delta_returns(series: pd.Series, lag: int = 1) -> pd.Series:
    """
    Change in returns — detects momentum reversal or acceleration patterns.

    Measures how returns are changing over time, providing insights into momentum persistence,
    reversal points, and the rate of change in market dynamics.

    Parameters:
    -----------
    series : pd.Series
        DataFrame containing 'close' prices
    lag : int, default=1
        Lag period for calculating the difference in returns

    Returns:
    --------
    pd.Series
        Change in returns over the specified lag period

    High-Impact Use Cases:
    ----------------------
    - **Momentum Reversal Signals**: Large negative changes indicate potential momentum breakdown
    - **Acceleration Confirmation**: Positive changes confirm strengthening momentum
    - **Volatility Regime Detection**: High absolute values indicate momentum instability
    - **Stop-Loss Triggers**: Use breakdowns in delta returns as dynamic stop signals

    Example:
    --------
    >>> # Momentum breakdown detection for exit signals
    >>> delta_ret = lagged_delta_returns(df, lag=2)
    >>> momentum_breakdown = (delta_ret < -0.03) & (series.pct_change(5) > 0.05)
    >>> # Exit long positions when strong momentum suddenly breaks down
    """
    ret = series.pct_change()
    return ret.diff(lag)


# ============================================================================
# VELOCITY-BASED MOMENTUM FEATURES
# ============================================================================
# These features measure momentum as average velocity (mean of returns) rather than price displacement.
# They are designed for both predictive feature engineering and as building blocks for forward-looking labels.


def sustained_velocity(close: pd.Series, lookback: int) -> pd.Series:
    """
    Calculate backward-looking sustained velocity (average of past returns).

    This measures the average log return over the past N bars, providing a
    smoothed view of recent directional momentum. Unlike price displacement
    (MOM, ROC), this captures average velocity.

    Formula:
        V̄_back(t,N) = (1/N) * Σ ln(Close[t-i] / Close[t-i-1]) for i in [0, N-1]

    Parameters
    ----------
    close : pd.Series
        Close prices for each period

    lookback : int
        Number of past bars to look back (N)

    Returns
    -------
    pd.Series
        Sustained backward velocity values.
        The first `lookback` bars will be NaN due to insufficient history.

    High-Impact Use Cases
    ---------------------
    - **Predictive Feature**: Zero lookahead bias - safe for ML features
    - **Velocity Measurement**: Smoothed momentum less noisy than ROC
    - **Multi-Timeframe Analysis**: Use different lookbacks for regime detection
    - **Label Generation**: Can be shifted forward for supervised learning labels

    Best for
    --------
    - FX, Equities, Crypto (any liquid market)
    - Feature engineering for ML models
    - Momentum regime classification

    See Also
    --------
    momentum_acceleration : Rate of change of velocity
    exponential_velocity : Exponentially weighted velocity
    """
    # Calculate log returns
    log_returns = np.log(close / close.shift(1))

    # Use rolling mean for backward velocity
    velocity = log_returns.rolling(window=lookback).mean()

    velocity.name = f"sustained_velocity_back_{lookback}"
    return velocity


def momentum_acceleration(
        close: pd.Series, lookback: int, short_window: int = None
) -> pd.Series:
    """
    Calculate backward-looking momentum acceleration (rate of change of velocity).

    Measures how the average return has changed over the lookback window by
    comparing recent vs older velocity. This detects momentum acceleration or
    deceleration patterns.

    Formula:
        Acceleration_back(t,N) = V̄_back(t, 0 to N/2) - V̄_back(t, N/2 to N)

    Parameters
    ----------
    close : pd.Series
        Close prices for each period

    lookback : int
        Number of past bars to look back (N)

    short_window : int, optional
        Window for first half comparison. Default is lookback // 2

    Returns
    -------
    pd.Series
        Backward momentum acceleration values.
        First `lookback` bars will be NaN.

    High-Impact Use Cases
    ---------------------
    - **Momentum Quality**: Positive acceleration indicates strengthening momentum
    - **Early Reversal Detection**: Negative acceleration warns of momentum breakdown
    - **Entry Timing**: Enter during acceleration phases for better risk-reward
    - **ML Feature**: Zero lookahead - safe for predictive modeling

    Best for
    --------
    - FX, Crypto (rapid momentum shifts)
    - Intraday trading (detecting momentum buildup/exhaustion)
    - Feature engineering for trend-following models

    See Also
    --------
    sustained_velocity : The underlying velocity metric

    Notes
    -----
    Performance: Vectorized implementation using rolling windows.
    Significantly faster than loop-based approach for large datasets.
    """
    # Default short window is half of lookback
    if short_window is None:
        short_window = lookback // 2

    if short_window < 2:
        raise ValueError("short_window must be at least 2")

    if lookback < 4:
        raise ValueError("lookback must be at least 4 for acceleration calculation")

    # Calculate log returns once
    log_returns = np.log(close / close.shift(1))

    # Vectorized approach using cumulative sums
    # At time t in the original loop:
    #   - Recent half: mean of log_returns[t-short_window : t] (iloc slicing, exclusive end)
    #   - Older half: mean of log_returns[t-lookback : t-short_window]
    #
    # Using rolling:
    #   - rolling(N).mean() at position t = mean of [t-N+1 : t+1] (inclusive both ends)
    #
    # To match the original loop behavior:
    #   - Recent velocity at t should be mean of [t-short_window : t]
    #     This is rolling(short_window).mean().shift(1) at position t
    #   - Older velocity at t should be mean of [t-lookback : t-short_window]

    # Recent velocity: mean over previous short_window bars (ending at t-1)
    recent_velocity = log_returns.rolling(window=short_window).mean().shift(1)

    # Older velocity: We need the mean of a window ending at t-short_window
    # This is (lookback - short_window) bars ending at t-short_window
    older_window_size = lookback - short_window

    # Shift back by short_window to get the window ending at t-short_window
    older_velocity = (
        log_returns.rolling(window=older_window_size).mean().shift(short_window + 1)
    )

    # Acceleration = recent - older
    acceleration = recent_velocity - older_velocity

    acceleration.name = f"momentum_accel_back_{lookback}"
    return acceleration


def exponential_velocity(
        close: pd.Series, lookback: int, alpha: float = 0.3
) -> pd.Series:
    """
    Calculate exponentially weighted backward velocity (recent past weighted more).

    Computes an exponentially weighted moving average of past returns,
    giving more weight to recent price movements. More responsive than
    simple average velocity.

    Formula:
        EWMA_Velocity_back(t,N) = EWM(returns, alpha)

    Parameters
    ----------
    close : pd.Series
        Close prices for each period

    lookback : int
        Reference window size (N) - used for naming/context

    alpha : float, default=0.3
        Smoothing parameter for exponential weighting.
        - Higher alpha (e.g., 0.5): More weight on recent past (responsive)
        - Lower alpha (e.g., 0.1): More balanced weighting (smooth)

    Returns
    -------
    pd.Series
        Exponentially weighted backward velocity.
        First `lookback` bars will have reduced significance.

    High-Impact Use Cases
    ---------------------
    - **Adaptive Momentum**: Responds faster to trend changes than SMA
    - **Trend Following**: Captures momentum with appropriate lag reduction
    - **ML Feature**: Zero lookahead - safe for predictive modeling
    - **Multi-Speed Momentum**: Use different alphas for fast/slow signals

    Best for
    --------
    - FX, Crypto (need responsiveness to rapid changes)
    - Intraday strategies (reduce lag while maintaining smoothness)
    - Adaptive trend following systems

    See Also
    --------
    sustained_velocity : Simple average version
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1")

    # Calculate log returns
    log_returns = np.log(close / close.shift(1))

    # Use pandas ewm for efficient calculation
    # span = (2/alpha) - 1, solving for span from alpha
    span = (2 / alpha) - 1
    ewma_velocity = log_returns.ewm(span=span, adjust=False).mean()

    ewma_velocity.name = f"ewma_velocity_back_{lookback}_a{int(alpha * 100)}"
    return ewma_velocity


def velocity_magnitude(close: pd.Series, lookback: int) -> pd.Series:
    """
    Calculate the absolute magnitude of backward velocity (direction-agnostic).

    Measures the magnitude of sustained movement regardless of direction.
    Useful for detecting high-momentum periods without caring about up/down.

    Formula:
        |V̄_back(t,N)| = |sustained_velocity(t,N)|

    Parameters
    ----------
    close : pd.Series
        Close prices for each period

    lookback : int
        Number of past bars to look back (N)

    Returns
    -------
    pd.Series
        Absolute magnitude of backward velocity.
        First `lookback` bars will be NaN.

    High-Impact Use Cases
    ---------------------
    - **Momentum Filter**: Filter for high-magnitude periods (trending)
    - **Regime Detection**: High magnitude = trending, low = ranging
    - **2D Momentum Space**: Combine with signed velocity for direction+strength
    - **ML Feature**: Direction-agnostic momentum strength

    Best for
    --------
    - All markets (universal momentum strength measure)
    - Regime classification (trending vs ranging)
    - Volatility-momentum analysis

    See Also
    --------
    sustained_velocity : Signed velocity (direction matters)
    """
    velocity = sustained_velocity(close, lookback)
    magnitude = np.abs(velocity)
    magnitude.name = f"velocity_magnitude_back_{lookback}"
    return magnitude


def velocity_consistency(close: pd.Series, lookback: int) -> pd.Series:
    """
    Calculate consistency of backward velocity (momentum smoothness).

    Measures how consistent the returns are over the lookback window.
    Low values indicate smooth, consistent momentum. High values indicate  choppy, noisy momentum.

    Formula:
        Consistency(t,N) = StdDev(Returns[t-N:t]) / |Mean(Returns[t-N:t])|

    This is the coefficient of variation - a normalized measure of dispersion.

    Parameters
    ----------
    close : pd.Series
        Close prices for each period

    lookback : int
        Number of past bars to look back (N)

    Returns
    -------
    pd.Series
        Velocity consistency metric (coefficient of variation).
        First `lookback` bars will be NaN.

    High-Impact Use Cases
    ---------------------
    - **Trend Quality**: Low consistency = smooth trends (high quality)
    - **Entry Filter**: Only trade during consistent momentum periods
    - **Risk Assessment**: High consistency = choppy/risky momentum
    - **ML Feature**: Momentum quality indicator

    Best for
    --------
    - FX, Equities (distinguishing smooth trends from choppy moves)
    - Trend-following systems (quality filter)
    - Risk management (avoid choppy momentum)

    See Also
    --------
    sustained_velocity : The velocity being measured
    velocity_magnitude : Direction-agnostic magnitude
    """
    if lookback < 2:
        raise ValueError("lookback must be at least 2 for consistency calculation")

    # Calculate log returns
    log_returns = np.log(close / close.shift(1))

    # Calculate rolling mean and std
    rolling_mean = log_returns.rolling(window=lookback).mean()
    rolling_std = log_returns.rolling(window=lookback).std()

    # Coefficient of variation
    consistency = rolling_std / rolling_mean.abs()

    # Replace inf values (when mean is near zero) with NaN
    consistency = consistency.replace([np.inf, -np.inf], np.nan)

    consistency.name = f"velocity_consistency_back_{lookback}"
    return consistency
