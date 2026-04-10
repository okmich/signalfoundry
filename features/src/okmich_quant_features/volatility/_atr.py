import numpy as np
import talib


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> (np.ndarray, np.ndarray):
    """
    Compute Average True Range (ATR) using TA-Lib, then normalizes it and returns both values.

    High Impact Use:
        - Volatility proxy for position sizing or state conditioning.
        - Common in volatility filters for momentum systems.

    Best for:
        - FX, Commodities, Indices.
    """
    atr = talib.ATR(high, low, close, timeperiod=period)
    return atr, np.where(close > 0, atr / close, np.nan)


def atr_sma_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, sma_period: int = 20) -> np.ndarray:
    """
    Calculate the ratio of ATR to its Simple Moving Average.
    This ratio helps identify when volatility is above or below its recent average, which can be useful for volatility
    breakout strategies or risk management.
    """
    atr = talib.ATR(high, low, close, timeperiod=period)
    atr_sma = talib.SMA(atr, timeperiod=sma_period)
    return np.divide(atr, atr_sma, out=np.full_like(atr, np.nan), where=atr_sma != 0)


def atr_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray, short_period: int = 7, long_period: int = 21) -> np.ndarray:
    """
    Calculate the ratio of short-term ATR to long-term ATR.
    This ratio compares recent volatility to longer-term volatility, which can help identify periods of increasing or
    decreasing volatility momentum.

    Note:
    -----
    Useful for identifying volatility regime changes and momentum shifts.
    """
    # Calculate short and long ATR
    atr_short = talib.ATR(high, low, close, timeperiod=short_period)
    atr_long = talib.ATR(high, low, close, timeperiod=long_period)
    return np.divide(atr_short, atr_long, out=np.full_like(atr_short, np.nan), where=atr_long != 0)


def ttr_ema_ratio(high: np.ndarray, low: np.ndarray, close: np.ndarray, ema_period: int = 20) -> np.ndarray:
    """
    Calculate the ratio of True Range to its EMA (TTR vs EMA of TTR).

    TTR (True Total Range) is calculated as the maximum of:
    - Current high minus current low
    - Absolute value of current high minus previous close
    - Absolute value of current low minus previous close

    This ratio shows how current volatility compares to its exponential moving average, providing a smoothed volatility momentum indicator.

    Note:
    -----
    This function uses the raw True Range calculation rather than ATR, providing a more responsive volatility measure.
    """
    tr = talib.TRANGE(high, low, close)
    tr_ema = talib.EMA(tr, timeperiod=ema_period)
    return np.divide(tr, tr_ema, out=np.full_like(tr, np.nan), where=tr_ema != 0)
