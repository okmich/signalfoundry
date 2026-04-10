import numpy as np
import pandas as pd
import talib


def log_returns(close: pd.Series, period: int = 1) -> pd.Series:
    """
    Compute log returns of a price series.

    High Impact Use:
        - Foundational feature for momentum or volatility estimation.
        - Often used as base input to all other features.

    Best for:
        - FX, Equities, Crypto (any liquid market).
    """
    return np.log(close / close.shift(period))


def rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute rolling Sharpe ratio (mean/std) of returns.

    High Impact Use:
        - Identifies high signal-to-noise momentum periods.
        - Aids regime filtering and trend quality assessment.

    Best for:
        - FX, Indices, Commodities.
    """
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return mean / (std + 1e-8)


def roc(price_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Rate of Change (ROC) — percentage change over n periods.

    Description:
        Uses TA-Lib ROC to measure the percentage change between current
        and n-period-ago closing prices — a direct measure of price velocity.

    High-Impact Use Case:
        - Detecting directional momentum bursts
        - Baseline feature for momentum phase classification

    High-Impact Asset Classes:
        - FX (trend continuation)
        - Equity indices (momentum rotation)
        - Commodities (breakout acceleration)
    """
    return pd.Series(talib.ROC(price_series.values, timeperiod=window), index=price_series.index)


def roc_smoothed(price_series: pd.Series, roc_window: int = 21, signal_window: int = 9) -> tuple[
    pd.Series, pd.Series, pd.Series]:
    """
    Smoothed Rate of Change with signal line and divergence using EMA.

    Description:
        Calculates ROC, its EMA-smoothed signal line, and the divergence between them.
        Similar conceptually to MACD but applied to rate of change momentum.
        Uses EMA for optimal responsiveness with minimal lag.

    Parameters:
    -----------
    price_series : pd.Series
        Price series (typically close prices)
    roc_window : int, default=21
        Period for the ROC calculation (longer captures trend momentum)
    signal_window : int, default=9
        Period for EMA smoothing of the ROC line

    Returns:
    --------
    tuple[pd.Series, pd.Series, pd.Series]
        - roc_raw: Raw ROC line
        - roc_signal: EMA-smoothed ROC (signal line)
        - roc_divergence: Difference between raw ROC and signal (roc_raw - roc_signal)

    High-Impact Use Cases:
    ----------------------
    - **Momentum Quality Assessment**: Divergence shows if momentum is clean or noisy
    - **Trend Direction**: Signal line provides filtered trend direction
    - **Early Warning**: When raw ROC diverges from signal, momentum may be weakening
    - **ML Feature Engineering**: Provides signal + noise characteristics in single feature set

    High-Impact Asset Classes:
    --------------------------
    - FX (trend continuation with quality filter)
    - Equity indices (momentum rotation detection)
    - Crypto (separating signal from noise in volatile moves)

    Example:
    --------
    >>> roc_raw, roc_signal, roc_div = roc_smoothed(close, roc_window=21, signal_window=9)
    >>> # Strong clean momentum: high signal with low divergence
    >>> clean_momentum = (roc_signal.abs() > threshold) & (roc_div.abs() < noise_threshold)
    >>>
    >>> # Multi-timeframe momentum analysis
    >>> roc_fast, sig_fast, div_fast = roc_smoothed(close, 10, 5)   # Fast momentum
    >>> roc_slow, sig_slow, div_slow = roc_smoothed(close, 50, 20)  # Trend momentum

    Notes:
    ------
    - EMA chosen over SMA/other methods based on empirical analysis showing high correlation
      (>0.7) between different smoothing methods - EMA provides best lag/responsiveness balance
    - Focus on varying window lengths rather than smoothing methods for feature diversity
    """
    # Calculate raw ROC
    roc_raw = pd.Series(talib.ROC(price_series.values, timeperiod=roc_window), index=price_series.index)

    # Calculate signal line using EMA (optimal responsiveness)
    roc_signal = pd.Series(
        talib.EMA(roc_raw.values, timeperiod=signal_window), index=price_series.index
    )

    return roc_raw, roc_signal, roc_raw - roc_signal


def roc_velocity(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Rate of change of ROC — measures momentum acceleration/deceleration.

    This 'momentum of momentum' indicator is a powerful leading signal that identifies
    when trends are strengthening or weakening before price action becomes obvious.

    Parameters:
    -----------
    series : pd.Series
    window : int, default=10
        Period for the initial ROC calculation

    Returns:
    --------
    pd.Series
        Acceleration of momentum (positive = accelerating, negative = decelerating)

    High-Impact Use Cases:
    ----------------------
    - **Early Reversal Detection**: Negative acceleration often precedes price reversals
    - **Trend Strength Assessment**: Strong positive acceleration confirms healthy trends
    - **Entry Timing**: Enter positions during acceleration phases for better risk-reward
    - **Divergence Detection**: Price making new highs with decelerating momentum = bearish divergence

    Example:
    --------
    >>> # Detect potential trend exhaustion
    >>> acceleration = roc_velocity(series, window=10)
    >>> price_highs = series > series.rolling(20).max()
    >>> bearish_divergence = price_highs & (acceleration < 0)
    """
    roc_series = roc(series, window=window)
    return roc_series.diff()


def mean_adjusted_ratio(price_series: pd.Series, n: int = 20) -> pd.Series:
    """
    Mean-Adjusted Ratio (MAR) — price vs. moving average ratio.

    Description:
        Ratio of current price to its moving average minus one,
        expressing normalized momentum relative to trend baseline.

    High-Impact Use Case:
        - Detecting overextension above/below moving average
        - Momentum normalization across instruments

    High-Impact Asset Classes:
        - Equities, FX, and crypto (persistent mean drift)
    """
    ma = talib.SMA(price_series.values, timeperiod=n)
    return price_series / ma - 1


def rsi(price_series: pd.Series, period: int = 14) -> pd.Series:
    return pd.Series(talib.RSI(price_series.values, timeperiod=period), index=price_series.index)


def macd(price_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> (pd.Series, pd.Series, pd.Series):
    """
    MACD based on PPO (Percentage Price Oscillator) for cross-asset comparability.

    Description:
        Uses PPO underneath to normalize MACD by price, making it comparable across assets
        with different price scales. PPO = ((EMA_fast - EMA_slow) / EMA_slow) * 100

        Unlike traditional MACD which uses absolute price differences, PPO expresses
        the difference as a percentage, enabling:
        - Cross-asset comparison (compare momentum of BTC vs ETH vs stocks)
        - Cross-timeframe analysis without rescaling
        - More stable signals across different price regimes

    Formula:
        PPO_line = ((EMA(fast) - EMA(slow)) / EMA(slow)) * 100
        Signal_line = EMA(PPO_line, signal_period)
        Histogram = PPO_line - Signal_line

    Returns:
        tuple: (ppo_line, signal_line, histogram)
        - ppo_line: Percentage difference between fast and slow EMAs
        - signal_line: EMA smoothing of PPO line
        - histogram: Difference between PPO and signal (momentum divergence)

    High-Impact Use Case:
        - Early trend acceleration or deceleration detection
        - Confirmation for volume or VWAP-based momentum
        - Cross-asset momentum comparison and ranking
        - Portfolio rotation strategies

    High-Impact Asset Classes:
        - FX (trend tracking across pairs)
        - Equities (cross-sector momentum comparison)
        - Crypto (BTC/ETH/ALT momentum ranking)
        - Multi-asset portfolios (compare momentum across asset classes)

    """
    # Use PPO for percentage-based, scale-independent momentum
    ppo_line = talib.PPO(price_series.values, fastperiod=fast, slowperiod=slow, matype=1)  # matype=1 = EMA

    # Signal line is EMA of PPO
    signal_line = talib.EMA(ppo_line, timeperiod=signal)

    # Histogram is difference between PPO and signal
    histogram = ppo_line - signal_line

    return (
        pd.Series(ppo_line, index=price_series.index),
        pd.Series(signal_line, index=price_series.index),
        pd.Series(histogram, index=price_series.index),
    )


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    return pd.Series(talib.MOM(close.values, timeperiod=period), index=close.index)


def momentum_acceleration(close: pd.Series, short: int = 5, long: int = 10) -> pd.Series:
    """
    Compute momentum acceleration = short-term momentum - long-term momentum.

    High Impact Use:
        - Captures rate-of-change-of-momentum.
        - Useful for detecting emerging momentum or exhaustion.

    Best for:
        - Crypto, FX.
    """
    mom_short = talib.MOM(close.values, timeperiod=short)
    mom_long = talib.MOM(close.values, timeperiod=long)
    return pd.Series(mom_short - mom_long, index=close.index)


def jerk(close: pd.Series, window: int = 3) -> pd.Series:
    """
    Compute jerk: 3rd derivative of price over window.

    High Impact Use:
        - Detects sharp accelerations or reversals.
        - Can mark regime shifts or blow-offs.

    Best for:
        - Intraday FX, Crypto.
    """
    mom = pd.Series(talib.MOM(close.values, timeperiod=window), index=close.index)
    accel = mom.diff()
    return accel.diff(window)


def rolling_slope(series: pd.Series, window: int = 20) -> pd.Series:
    """OLS slope over a rolling window, normalised by local mean price.

    Output is dimensionless (slope per bar / mean price level), making it
    comparable across assets and price levels.
    """
    x = np.arange(window)
    raw_slope = series.rolling(window).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
    mean_price = series.rolling(window).mean()
    return raw_slope / mean_price


def stochastic(close: pd.Series, high: pd.Series, low: pd.Series, fastk_period: int = 14, slowk_period: int = 3) \
        -> (pd.Series, pd.Series):
    k, signal = talib.STOCH(high.values, low.values, close.values,
                            fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=3)
    return pd.Series(k, index=close.index), pd.Series(signal, index=close.index)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=period), index=close.index)


def detrended_price(close: pd.Series, period: int = 20) -> pd.Series:
    sma = talib.SMA(close.values, timeperiod=period)

    # Shift the SMA back by (period/2 + 1) to center it and remove trend
    shift_amount = period // 2 + 1
    sma_shifted = pd.Series(sma, index=close.index).shift(shift_amount)
    return close - sma_shifted


def ema_residual(series: pd.Series, period: int = 20) -> pd.Series:
    ema = talib.EMA(series.values, timeperiod=period)
    return series - ema


def momentum_volatility_ratio(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute ratio of momentum to volatility.

    High Impact Use:
        - Detects high “signal per unit noise” momentum bursts.
        - Key input for quality-adjusted momentum states.

    Best for:
        - FX, Indices.
    """
    mom = talib.MOM(close.values, timeperiod=period)
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(mom / (atr + 1e-8), index=close.index)


def trend_quality(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute trend quality = slope / volatility.

    High Impact Use:
        - High values signal smooth directional persistence.
        - Excellent for momentum classifier training or filtering.

    Best for:
        - FX, Equities.
    """
    slope_ = rolling_slope(close, period)
    atr_ = talib.ATR(high.values, low.values, close.values, timeperiod=period)
    return slope_ / (atr_ + 1e-8)


def vol_adj_mom_osc(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    Volatility-Adjusted Momentum Oscillator.

    Description:
        Normalizes momentum (ROC) by average true range (ATR) using TA-Lib functions.

    High-Impact Use Case:
        - Momentum comparison across volatility regimes
        - Regime-adaptive scaling for signals

    High-Impact Asset Classes:
        - FX and crypto (volatility clustering prominent)
    """
    atr = talib.ATR(df["high"].values, df["low"].values, df["close"].values, timeperiod=n)
    roc = talib.ROC(df["close"].values, timeperiod=n)
    return pd.Series(roc / np.where(atr == 0, np.nan, atr), index=df.index)
