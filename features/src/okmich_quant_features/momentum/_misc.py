import numpy as np
import pandas as pd

import talib


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
    exponential_velocity : Exponentially weighted velocity
    """
    # Calculate log returns
    log_returns = np.log(close / close.shift(1))

    # Use rolling mean for backward velocity
    velocity = log_returns.rolling(window=lookback).mean()

    velocity.name = f"sustained_velocity_back_{lookback}"
    return velocity


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
