"""
Price Structure Features for Market Microstructure Analysis.
These features capture the geometric and statistical properties of price bars to detect market efficiency, information content, and regime characteristics.

Module Structure
----------------
intrabar_efficiency_ratio  : Body/range ratio — measures trend vs indecision
realized_skewness          : Rolling skewness of log returns (tail direction)
realized_kurtosis          : Rolling excess kurtosis of log returns (tail weight)
distance_to_extremes       : Signed distance of close from rolling range midpoint
range_compression_ratio    : Short-term vs long-term range EMA (coiling detection)
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Union

from ..volatility._volatility import _parkinson_volatility_nb

ArrayLike = Union[pd.Series, np.ndarray]


# --------------------------------------------------------------------------- #
# Intrabar Efficiency Ratio (IER)                                             #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ier_kernel(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean(|close-open|) / mean(high-low) over `window` bars."""
    n = len(close)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        net_sum = 0.0
        range_sum = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            rng = high[j] - low[j]
            if rng > 0.0:
                net_sum += abs(close[j] - open_[j])
                range_sum += rng
                count += 1
        if range_sum > 0.0 and count >= window // 2:
            result[i] = net_sum / range_sum

    return result


def intrabar_efficiency_ratio(open_: ArrayLike, high: ArrayLike, low: ArrayLike,
                               close: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Rolling Intrabar Efficiency Ratio (IER).

    Measures how efficiently price moves within each bar. High IER = price moved
    cleanly from open to close (trending bar). Low IER = large wick relative to body
    (indecision, rejection, mean-reversion pressure).

    Formula:
        IER = mean(|close - open|) / mean(high - low) over window

    Returns values in [0, 1]:
        - ≈ 1.0: All bars are full-range trend bars (strong directional move)
        - ≈ 0.5: Typical mixed market
        - ≈ 0.0: All bars are doji/spinning top (no conviction, just noise)

    Parameters
    ----------
    open_, high, low, close : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series, same length)
    window : int, default=20
        Rolling window length.

    Returns
    -------
    ArrayLike
        IER series; first (window - 1) values are NaN. Same type as `close`.

    Interpretation
    --------------
    - IER > 0.7 → Strong trend bars; momentum strategies preferred
    - IER < 0.3 → Choppy/indecisive bars; mean-reversion strategies preferred
    - Falling IER → Trend exhaustion or regime shift
    """
    is_series = isinstance(close, pd.Series)
    index = close.index if is_series else None

    open_arr = np.asarray(open_.values if isinstance(open_, pd.Series) else open_, dtype=np.float64)
    high_arr = np.asarray(high.values if isinstance(high, pd.Series) else high, dtype=np.float64)
    low_arr = np.asarray(low.values if isinstance(low, pd.Series) else low, dtype=np.float64)
    close_arr = np.asarray(close.values if isinstance(close, pd.Series) else close, dtype=np.float64)

    result = _ier_kernel(open_arr, high_arr, low_arr, close_arr, window)

    if is_series:
        return pd.Series(result, index=index, name=f'ier_{window}')
    return result


# --------------------------------------------------------------------------- #
# Realized Skewness                                                           #
# --------------------------------------------------------------------------- #

def realized_skewness(close: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Rolling realized skewness of log returns.

    Captures the asymmetry of the return distribution over a rolling window.
    Negative skew (typical in equities) = occasional large crashes. Positive
    skew = occasional large upside moves (common in commodities, crypto).

    Formula:
        log_ret = log(close[t] / close[t-1])
        skewness = rolling(window).skew(log_ret)

    Parameters
    ----------
    close : ArrayLike
        Close prices (1-D float64 array or pd.Series)
    window : int, default=20
        Rolling window (minimum 3 observations required by pandas).

    Returns
    -------
    ArrayLike
        Rolling skewness; first (window) values are NaN. Same type as `close`.

    Interpretation
    --------------
    - Positive skew → Right tail: occasional large gains
    - Negative skew → Left tail: occasional large losses (more common in equities)
    - Skew near 0   → Roughly symmetric return distribution
    """
    is_series = isinstance(close, pd.Series)
    index = close.index if is_series else None

    close_s = close if is_series else pd.Series(close)
    log_ret = np.log(close_s / close_s.shift(1))
    skew = log_ret.rolling(window).skew()

    if is_series:
        return pd.Series(skew.values, index=index, name=f'skew_{window}')
    return skew.values


# --------------------------------------------------------------------------- #
# Realized Kurtosis                                                           #
# --------------------------------------------------------------------------- #

def realized_kurtosis(close: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Rolling realized excess kurtosis of log returns.

    Captures fat-tail behavior of the return distribution. High kurtosis indicates
    a heavy-tailed regime (frequent extreme moves), often associated with
    regime switching, news events, or liquidity crises.

    Formula:
        log_ret = log(close[t] / close[t-1])
        kurtosis = rolling(window).kurt(log_ret)  # excess kurtosis (normal = 0)

    Parameters
    ----------
    close : ArrayLike
        Close prices (1-D float64 array or pd.Series)
    window : int, default=20
        Rolling window (minimum 4 observations required by pandas).

    Returns
    -------
    ArrayLike
        Rolling excess kurtosis; first (window) values are NaN. Same type as `close`.

    Interpretation
    --------------
    - Excess kurtosis ≈ 0 → Normal (Gaussian) distribution
    - Excess kurtosis > 3 → Heavy tails (frequent extreme moves, regime switching)
    - Excess kurtosis < 0 → Thin tails (returns more concentrated around mean)
    """
    is_series = isinstance(close, pd.Series)
    index = close.index if is_series else None

    close_s = close if is_series else pd.Series(close)
    log_ret = np.log(close_s / close_s.shift(1))
    kurt = log_ret.rolling(window).kurt()

    if is_series:
        return pd.Series(kurt.values, index=index, name=f'kurt_{window}')
    return kurt.values


# --------------------------------------------------------------------------- #
# Distance to Extremes                                                        #
# --------------------------------------------------------------------------- #

def distance_to_extremes(high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Signed distance of close from the rolling range midpoint, normalized by Parkinson vol.

    Measures where the current close sits relative to its recent price extremes, in units of
    annualized Parkinson volatility × price. Positive = close is above the rolling midpoint (bullish position). Negative = below midpoint.

    Formula:
        roll_high = max(high[t-window+1..t])
        roll_low  = min(low[t-window+1..t])
        roll_mid  = (roll_high + roll_low) / 2
        park_vol  = Parkinson(high, low, window)              [annualized]
        daily_vol = park_vol / sqrt(252) * close              [price units]
        DTE = (close - roll_mid) / (daily_vol + ε)

    Parameters
    ----------
    high, low, close : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series, same length)
    window : int, default=20
        Rolling window for both range extremes and Parkinson vol.

    Returns
    -------
    ArrayLike
        Signed distance; first (window - 1) values are NaN. Same type as `close`.

    Interpretation
    --------------
    - DTE > +2 → Close near the top of rolling range (resistance zone, overbought)
    - DTE ≈  0 → Close at mid-range (balanced, neutral)
    - DTE < -2 → Close near the bottom of rolling range (support zone, oversold)
    """
    is_series = isinstance(close, pd.Series)
    index = close.index if is_series else None

    high_s = high if isinstance(high, pd.Series) else pd.Series(high)
    low_s = low if isinstance(low, pd.Series) else pd.Series(low)
    close_s = close if is_series else pd.Series(close)

    high_arr = high_s.values.astype(np.float64)
    low_arr = low_s.values.astype(np.float64)

    roll_high = high_s.rolling(window).max()
    roll_low = low_s.rolling(window).min()
    roll_mid = (roll_high + roll_low) / 2.0

    # Parkinson vol (annualized), convert to per-bar price units
    park_vol = pd.Series(_parkinson_volatility_nb(high_arr, low_arr, window),
                         index=close_s.index)
    daily_vol = (park_vol / np.sqrt(252)) * close_s

    dist = (close_s - roll_mid) / (daily_vol + 1e-10)

    if is_series:
        return pd.Series(dist.values, index=index, name=f'dte_{window}')
    return dist.values


# --------------------------------------------------------------------------- #
# Range Compression Ratio                                                     #
# --------------------------------------------------------------------------- #

def range_compression_ratio(high: ArrayLike, low: ArrayLike, short_span: int = 5, long_span: int = 20) -> ArrayLike:
    """
    Range Compression Ratio: short-term EMA of H-L range / long-term EMA.

    Detects whether the price range is compressing (coiling before a breakout) or expanding (trending, high volatility).
    A falling ratio indicates the market is "coiling" — often a precursor to a volatility expansion.

    Formula:
        range = high - low
        RCR = EMA(range, short_span) / EMA(range, long_span)

    Parameters
    ----------
    high, low : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series)
    short_span : int, default=5
        EMA span for short-term range smoothing.
    long_span : int, default=20
        EMA span for long-term range smoothing.

    Returns
    -------
    ArrayLike
        RCR series; first few values are NaN. Same type as `high`.

    Interpretation
    --------------
    - RCR < 0.5 → Strong range compression (high breakout potential)
    - RCR ≈ 1.0 → Short/long range balanced (no compression)
    - RCR > 1.5 → Range expansion (trending or volatile regime)
    """
    is_series = isinstance(high, pd.Series)
    index = high.index if is_series else None

    high_s = high if is_series else pd.Series(high)
    low_s = low if isinstance(low, pd.Series) else pd.Series(low)

    hl_range = high_s - low_s
    short_ema = hl_range.ewm(span=short_span, adjust=False).mean()
    long_ema = hl_range.ewm(span=long_span, adjust=False).mean()

    with np.errstate(invalid='ignore', divide='ignore'):
        rcr = short_ema / (long_ema + 1e-10)

    if is_series:
        return pd.Series(rcr.values, index=index, name=f'rcr_{short_span}_{long_span}')
    return rcr.values


# --------------------------------------------------------------------------- #
# Price Path Fractal Dimension (OHLC proxy)                                   #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _fractal_dimension_kernel(high: np.ndarray, low: np.ndarray, open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Bar-level fractal dimension proxy.
        FD = log2[2(H−L) / (|O−C| + ε)]
    """
    n = len(high)
    fd = np.empty(n, dtype=np.float64)
    for i in range(n):
        rng = high[i] - low[i]
        body = abs(close[i] - open_[i])
        if body < 1e-10:
            body = 1e-10
        ratio = 2.0 * rng / body
        fd[i] = np.log(ratio) / np.log(2.0) if ratio > 0 else 1.0

    return fd


def price_path_fractal_dimension(open_: pd.Series, high: pd.Series,
                                  low: pd.Series, close: pd.Series,
                                  window: int = 20) -> pd.DataFrame:
    """
    Price Path Fractal Dimension (OHLC Proxy). Estimates complexity of intrabar price movement using the ratio of the
    bar range to the body size — a proxy for the fractal dimension of the intrabar price path.

    Formula
    -------
        FD       = log₂[2(H−L) / (|O−C| + ε)]
        FD_ema   = EMA(FD, window)
        delta_FD = FD_ema.diff(window)

    Parameters
    ----------
    open_, high, low, close : pd.Series
        OHLC prices.
    window : int, default=20
        EMA span for smoothing; also used for delta lookback.

    Returns
    -------
    pd.DataFrame
        Columns: ``FD``, ``FD_ema``, ``delta_FD``.

        - FD ≈ 1 : Straight-line (trending)
        - FD ≈ 2 : Area-filling (noisy/ranging)
        - Rising FD : Increasing complexity → regime transition
    """
    fd = _fractal_dimension_kernel(high.values, low.values,
                                   open_.values, close.values)
    fd_s = pd.Series(fd, index=open_.index)
    fd_ema = fd_s.ewm(span=window, adjust=False).mean()
    delta_fd = fd_ema.diff(window)

    return pd.DataFrame({'FD': fd_s, 'FD_ema': fd_ema, 'delta_FD': delta_fd},
                        index=open_.index)


# --------------------------------------------------------------------------- #
# Close-Open Gap Analysis                                                      #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _gap_fill_ratio_kernel(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Fraction of overnight gap filled within the bar."""
    n = len(open_)
    gfr = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        gap = open_[i] - close[i - 1]

        if gap < -1e-10:  # Down gap
            gap_size = close[i - 1] - open_[i]
            if gap_size > 1e-10:
                gfr[i] = (close[i] - open_[i]) / gap_size
            else:
                gfr[i] = 0.0
        elif gap > 1e-10:  # Up gap
            gap_size = open_[i] - close[i - 1]
            if gap_size > 1e-10:
                gfr[i] = (open_[i] - close[i]) / gap_size
            else:
                gfr[i] = 0.0
        else:
            gfr[i] = 0.0

    return gfr


def close_open_gap_analysis(open_: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Close-Open Gap Analysis.

    Measures gap size and how much of the gap is filled within the bar.

    Formula
    -------
        gap(t)          = O(t) − C(t−1)
        gap_fill_ratio  = fraction of gap filled before close

    Returns
    -------
    pd.DataFrame
        Columns: ``gap``, ``gap_fill_ratio``.

        - GFR > 1  : Gap overfilled / reversed → mean-reversion
        - GFR ≈ 0  : Gap not filled → continuation
        - GFR < 0  : Gap extended → strong directional flow
    """
    gap = open_ - close.shift(1)
    gfr = _gap_fill_ratio_kernel(open_.values, close.values)
    return pd.DataFrame({'gap': gap, 'gap_fill_ratio': gfr}, index=open_.index)


# --------------------------------------------------------------------------- #
# Return-Spread Cross-Correlation                                              #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _return_spread_xcorr_kernel(returns: np.ndarray, delta_spread: np.ndarray, lag: int, window: int) -> np.ndarray:
    """Rolling cross-correlation between returns and delta_spread at given lag."""
    n = len(returns)
    xcorr = np.full(n, np.nan, dtype=np.float64)

    abs_lag = abs(lag)

    for i in range(window + abs_lag, n):
        if lag > 0:
            # Spread leads returns: lagged spread correlated with current returns
            r = returns[i - window + 1: i + 1]
            s = delta_spread[i - window + 1 - lag: i + 1 - lag]
        else:
            # Returns lead spread: lagged returns correlated with current spread
            r = returns[i - window + 1 - abs_lag: i + 1 - abs_lag]
            s = delta_spread[i - window + 1: i + 1]

        valid = (~np.isnan(r)) & (~np.isnan(s))
        if np.sum(valid) < window // 2:
            continue

        rv = r[valid]
        sv = s[valid]
        rm = np.mean(rv)
        sm = np.mean(sv)
        cov = np.mean((rv - rm) * (sv - sm))
        rs = np.std(rv)
        ss = np.std(sv)
        if rs > 1e-10 and ss > 1e-10:
            xcorr[i] = cov / (rs * ss)
        else:
            xcorr[i] = 0.0
    return xcorr


def return_spread_cross_correlation(close: pd.Series, spread: pd.Series, mid_price: pd.Series, window: int = 20,
                                    lags: list = None) -> pd.DataFrame:
    """
    Return-Spread Cross-Correlation Structure. Measures the lead-lag relationship between normalised spread changes and
    log-returns at multiple lags.

    Formula
    -------
        r(t)          = log[C(t)/C(t−1)]
        ΔS_norm(t)    = [S(t)/M(t)] − [S(t−1)/M(t−1)]
        xcorr(lag)    = Corr[r(t), ΔS_norm(t−lag)]  (positive lag = spread leads)

    Parameters
    ----------
    close, spread, mid_price : pd.Series
        Price and spread data.
    window : int, default=20
        Rolling correlation window.
    lags : list, default=[1, 2, 3]
        Lags to compute.  Positive = spread leads returns.

    Returns
    -------
    pd.DataFrame
        One column per lag: ``xcorr_lag+1``, ``xcorr_lag+2``, ...

        - ρ(lag=+1) > 0 : Spread widening precedes price drops (liquidity leads)
        - ρ(lag=−1) > 0 : Returns lead spread (price-driven regime)
    """
    if lags is None:
        lags = [1, 2, 3]

    ret = np.log(close / close.shift(1)).values
    s_norm = (spread / (mid_price + 1e-10)).values
    ds = np.concatenate([[np.nan], np.diff(s_norm)])

    result = {}
    for lag in lags:
        xcorr = _return_spread_xcorr_kernel(ret, ds, lag, window)
        result[f'xcorr_lag{lag:+d}'] = xcorr

    return pd.DataFrame(result, index=close.index)
