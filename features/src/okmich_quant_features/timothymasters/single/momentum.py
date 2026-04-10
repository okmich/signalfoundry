"""
Momentum & oscillator indicators #1–11.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source file: COMP_VAR.CPP.

Indicators
----------
1.  rsi                  Relative Strength Index (Wilder's EMA): COMP_VAR.CPP:144–178
2.  detrended_rsi        RSI residual after regressing on a longer RSI: COMP_VAR.CPP:184–287
3.  stochastic           Raw / fast-K / slow-D stochastic: COMP_VAR.CPP:294–342
4.  stoch_rsi            Stochastic applied to RSI values: COMP_VAR.CPP:349–413
5.  ma_difference        Normalised short-minus-long moving-average gap: COMP_VAR.CPP:420–451
6.  macd                 MACD line or histogram, ATR-normalised: COMP_VAR.CPP:458–502
7.  ppo                  Percentage Price Oscillator: COMP_VAR.CPP:509–546
8.  price_change_osc     Average absolute log-return ratio, short vs long: COMP_VAR.CPP:971–1011
9.  close_minus_ma       Current log-price minus rolling log mean, ATR-scaled: COMP_VAR.CPP:860–882
10. price_intensity      (Close – Open) / True Range, optionally smoothed: COMP_VAR.CPP:628–667
11. reactivity           Gietzen's aspect-ratio indicator: COMP_VAR.CPP:1130–1189

Output convention
-----------------
All public functions return a 1-D ``float64`` numpy array of the same length as the input ``close`` array.
Warmup bars that do not yet have a valid value are filled with ``np.nan``.
After the normal-CDF compression the valid range is approximately [-50, 50].

Numba kernels
-------------
Every non-trivial inner loop is compiled by Numba (``@njit(cache=True)``).
The shared helper ``_ncdf(x, k)`` = 100 * Phi(k*x) - 50 is inlined using ``math.erf`` so that scipy is never needed inside Numba context.
"""

import math

import numpy as np
from numba import njit

from ._helpers import atr_kernel, normal_cdf_compress


# --------------------------------------------------------------------------- #
# Private Numba helpers                                                        #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ncdf(x: float, k: float) -> float:
    """
    100 * Phi(k * x) - 50, where Phi is the standard normal CDF.

    Uses the identity  Phi(z) = 0.5 * erfc(-z/sqrt(2))
    => 100*Phi(kx) - 50 = 50 * erf(kx / sqrt(2))
    """
    return 50.0 * math.erf(k * x * 0.7071067811865476)  # 0.707... = 1/sqrt(2)


# --------------------------------------------------------------------------- #
# 1. RSI  (COMP_VAR.CPP:144–178)                                              #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _rsi_kernel(close: np.ndarray, lookback: int) -> np.ndarray:
    """
    Wilder's Relative Strength Index.

    Initialisation: simple average of abs up/down moves over bars 1..(lookback-1).
    EMA update from bar ``lookback`` onward: alpha = 1 / lookback.
    Neutral fill = 50.0 for warmup bars.
    """
    n = len(close)
    out = np.full(n, np.nan)

    if n <= lookback:
        return out

    # --- seed: simple average over bars 1..lookback-1 (lookback-1 diffs) ---
    upsum = 0.0
    dnsum = 0.0
    for i in range(1, lookback):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            upsum += diff
        else:
            dnsum -= diff           # keep positive

    if lookback > 1:
        upsum /= (lookback - 1.0)
        dnsum /= (lookback - 1.0)

    # --- Wilder's EMA from bar lookback onward ---
    decay = (lookback - 1.0) / lookback
    for i in range(lookback, n):
        diff = close[i] - close[i - 1]
        if diff > 0.0:
            upsum = decay * upsum + diff / lookback
            dnsum *= decay
        else:
            upsum *= decay
            dnsum = decay * dnsum - diff / lookback  # diff negative, subtract flips sign

        out[i] = 100.0 * upsum / (upsum + dnsum + 1e-60)

    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (Wilder's EMA).

    Parameters
    ----------
    close : array-like  Close prices.
    period : int        Lookback (default 14).

    Returns
    -------
    np.ndarray  RSI in [0, 100].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    return _rsi_kernel(close, period)


# --------------------------------------------------------------------------- #
# 2. DETRENDED RSI  (COMP_VAR.CPP:184–287)                                   #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _detrended_rsi_kernel(close: np.ndarray, short_len: int, long_len: int, reg_len: int, transform_short: bool) -> np.ndarray:
    """
    Detrended RSI: residual of short RSI after regressing on long RSI.

    First compute both RSI series (using Wilder's EMA), then for each bar fit a rolling OLS of ``short_rsi ~ long_rsi``
    over ``reg_len`` bars and return the residual at the current bar.

    If ``transform_short`` is True (short lookback == 2), the short RSI is transformed: ``-10 * log(2 / (1 + 0.00999 * (2*RSI - 100)) - 1)`` before the regression.

    front_bad = long_len + reg_len - 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    front_bad = long_len + reg_len - 1
    if n <= front_bad:
        return out

    # --- compute both RSI series ---
    short_rsi = _rsi_kernel(close, short_len)
    long_rsi = _rsi_kernel(close, long_len)

    # --- optional transform on short RSI (short_len == 2) ---
    if transform_short:
        for i in range(n):
            if not math.isnan(short_rsi[i]):
                v = 1.0 + 0.00999 * (2.0 * short_rsi[i] - 100.0)
                if v > 1e-15 and v < 2.0:
                    inner = 2.0 / v - 1.0
                    if inner > 1e-15:
                        short_rsi[i] = -10.0 * math.log(inner)
                    else:
                        short_rsi[i] = np.nan
                else:
                    short_rsi[i] = np.nan

    # --- rolling OLS residual ---
    for i in range(front_bad, n):
        # window: [i - reg_len + 1, i]
        start = i - reg_len + 1

        # check both series valid in window
        ok = True
        for k in range(start, i + 1):
            if math.isnan(short_rsi[k]) or math.isnan(long_rsi[k]):
                ok = False
                break
        if not ok:
            continue

        xmean = 0.0   # long RSI mean
        ymean = 0.0   # short RSI mean
        for k in range(start, i + 1):
            xmean += long_rsi[k]
            ymean += short_rsi[k]
        xmean /= reg_len
        ymean /= reg_len

        xss = 0.0
        xy = 0.0
        for k in range(start, i + 1):
            dx = long_rsi[k] - xmean
            xss += dx * dx
            xy += dx * (short_rsi[k] - ymean)

        if xss < 1e-30:
            out[i] = 0.0
        else:
            coef = xy / xss
            # residual at current bar relative to regression through means
            out[i] = (short_rsi[i] - ymean) - coef * (long_rsi[i] - xmean)

    return out


def detrended_rsi(close: np.ndarray, short_period: int = 7, long_period: int = 14, reg_len: int = 32) -> np.ndarray:
    """
    Detrended RSI.

    Computes two RSI series (short and long) then returns the residual of a rolling OLS regression of ``short_rsi`` on
    ``long_rsi`` over ``reg_len`` bars.

    Parameters
    ----------
    close       : array-like  Close prices.
    short_period: int         Short RSI lookback.
    long_period : int         Long RSI lookback (must be > short_period).
    reg_len     : int         Rolling regression window.

    Returns
    -------
    np.ndarray  Residual in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    if long_period <= short_period:
        raise ValueError("long_period must be > short_period")
    return _detrended_rsi_kernel(close, short_period, long_period, reg_len, transform_short=(short_period == 2))


# --------------------------------------------------------------------------- #
# 3. STOCHASTIC  (COMP_VAR.CPP:294–342)                                      #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _stochastic_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int, smoothing: int) -> np.ndarray:
    """
    Stochastic oscillator.

    smoothing: 0 = raw, 1 = fast-K (1/3 EMA), 2 = slow-D (1/3 EMA on K).
    front_bad = lookback - 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    if n < lookback:
        return out

    k_prev = 50.0   # neutral seed for EMA
    d_prev = 50.0

    for i in range(lookback - 1, n):
        start = i - lookback + 1
        hi = high[start]
        lo = low[start]
        for k in range(start + 1, i + 1):
            if high[k] > hi:
                hi = high[k]
            if low[k] < lo:
                lo = low[k]

        rng = hi - lo
        if rng < 1e-60:
            raw = 50.0
        else:
            raw = 100.0 * (close[i] - lo) / rng

        if smoothing == 0:
            out[i] = raw
        elif smoothing == 1:
            k_val = 0.33333333 * raw + 0.66666667 * k_prev
            k_prev = k_val
            out[i] = k_val
        else:                               # smoothing == 2
            k_val = 0.33333333 * raw + 0.66666667 * k_prev
            k_prev = k_val
            d_val = 0.33333333 * k_val + 0.66666667 * d_prev
            d_prev = d_val
            out[i] = d_val
    return out


def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, smoothing: int = 1) -> np.ndarray:
    """
    Stochastic oscillator.

    Parameters
    ----------
    high, low, close : array-like  OHLC data (same length).
    period   : int   Lookback for high/low range (default 14).
    smoothing: int   0=raw, 1=fast-K (1/3 EMA), 2=slow-D (1/3 EMA of K).

    Returns
    -------
    np.ndarray  in [0, 100].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return _stochastic_kernel(high, low, close, period, smoothing)


# --------------------------------------------------------------------------- #
# 4. STOCHASTIC RSI  (COMP_VAR.CPP:349–413)                                  #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _stoch_rsi_kernel(close: np.ndarray, rsi_len: int, stoch_len: int, smooth_len: int) -> np.ndarray:
    """
    Stochastic applied to RSI values.

    1. Compute RSI(close, rsi_len).
    2. Apply stochastic(RSI, stoch_len) → raw StochRSI in [0, 100].
    3. If smooth_len > 1: apply standard EMA alpha=2/(smooth_len+1).
    front_bad = rsi_len + stoch_len - 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    front_bad = rsi_len + stoch_len - 1
    if n <= front_bad:
        return out

    rsi_vals = _rsi_kernel(close, rsi_len)

    smooth_alpha = 2.0 / (smooth_len + 1.0) if smooth_len > 1 else 0.0
    smoothed = 50.0

    for i in range(front_bad, n):
        start = i - stoch_len + 1

        # check all RSI values valid
        ok = True
        for k in range(start, i + 1):
            if math.isnan(rsi_vals[k]):
                ok = False
                break
        if not ok:
            continue

        min_val = rsi_vals[start]
        max_val = rsi_vals[start]
        for k in range(start + 1, i + 1):
            if rsi_vals[k] < min_val:
                min_val = rsi_vals[k]
            if rsi_vals[k] > max_val:
                max_val = rsi_vals[k]

        rng = max_val - min_val
        if rng < 1e-60:
            raw = 50.0
        else:
            raw = 100.0 * (rsi_vals[i] - min_val) / rng

        if smooth_len > 1:
            smoothed = smooth_alpha * raw + (1.0 - smooth_alpha) * smoothed
            out[i] = smoothed
        else:
            out[i] = raw
    return out


def stoch_rsi(close: np.ndarray, rsi_period: int = 14, stoch_period: int = 14, smooth_period: int = 1) -> np.ndarray:
    """
    Stochastic RSI.

    Parameters
    ----------
    close        : array-like  Close prices.
    rsi_period   : int  Lookback for RSI (default 14).
    stoch_period : int  Lookback for stochastic applied to RSI (default 14).
    smooth_period: int  Final EMA smoothing period; <=1 means no smoothing.

    Returns
    -------
    np.ndarray  in [0, 100].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    return _stoch_rsi_kernel(close, rsi_period, stoch_period, smooth_period)


# --------------------------------------------------------------------------- #
# 5. MA DIFFERENCE  (COMP_VAR.CPP:420–451)                                   #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ma_difference_kernel(close: np.ndarray, atr_abs: np.ndarray, short_len: int, long_len: int, lag: int) -> np.ndarray:
    """
    Normalised difference between a short and lagged long simple MA.

    value = (short_MA - long_MA_lagged) / (sqrt(|center_diff|) * ATR + 1e-60)
    output = 100 * Phi(1.5 * value) - 50

    center_diff = 0.5*(long_len - 1) + lag - 0.5*(short_len - 1)
                = 0.5*(long_len - short_len) + lag

    front_bad = long_len + lag - 1.
    atr_abs: pre-computed ATR(absolute, length=long_len+lag) array.
    """
    n = len(close)
    out = np.full(n, np.nan)

    front_bad = long_len + lag - 1
    if n <= front_bad:
        return out

    center_diff = 0.5 * (long_len - short_len) + lag
    sqrt_center = math.sqrt(math.fabs(center_diff)) if center_diff != 0.0 else 1.0

    for i in range(front_bad, n):
        # short MA at bar i
        s_sum = 0.0
        for k in range(i - short_len + 1, i + 1):
            s_sum += close[k]
        short_ma = s_sum / short_len

        # long MA at bar (i - lag)
        j = i - lag
        l_sum = 0.0
        for k in range(j - long_len + 1, j + 1):
            l_sum += close[k]
        long_ma = l_sum / long_len

        denom = sqrt_center * atr_abs[i] + 1e-60
        val = (short_ma - long_ma) / denom
        out[i] = _ncdf(val, 1.5)
    return out


def ma_difference(high: np.ndarray, low: np.ndarray, close: np.ndarray, short_period: int = 10, long_period: int = 40,
                  lag: int = 0) -> np.ndarray:
    """
    Moving-Average Difference, ATR-normalised and CDF-compressed.

    Parameters
    ----------
    high, low, close : array-like
    short_period : int  Short SMA length.
    long_period  : int  Long SMA length.
    lag          : int  Bars to lag the long MA (default 0).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    atr_len = long_period + lag
    atr_abs = atr_kernel(high, low, close, atr_len, False)

    return _ma_difference_kernel(close, atr_abs, short_period, long_period, lag)


# --------------------------------------------------------------------------- #
# 6. MACD  (COMP_VAR.CPP:458–502)                                            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _macd_kernel(close: np.ndarray, atr_abs: np.ndarray, short_len: int, long_len: int, signal_len: int) -> np.ndarray:
    """
    ATR-normalised MACD line or histogram.

    EMA alpha = 2 / (length + 1).
    normalization denom = sqrt(0.5*(long_len - short_len)) * ATR + 1e-60.
    Compression: 100 * Phi(1.0 * value) - 50.
    If signal_len > 1: smooth ALL bars (including warmup), then subtract (histogram).
    front_bad = long_len + signal_len - 1 (C++ marks as warmup).
    """
    n = len(close)
    out = np.full(n, 0.0)  # Initialize to 0 like C++

    if n < long_len:
        out[:] = np.nan
        return out

    short_alpha = 2.0 / (short_len + 1.0)
    long_alpha = 2.0 / (long_len + 1.0)

    sqrt_center = math.sqrt(0.5 * (long_len - short_len))

    short_ema = close[0]
    long_ema = close[0]

    # First pass: compute compressed MACD line for all bars (C++ style)
    for i in range(1, n):
        short_ema = short_alpha * close[i] + (1.0 - short_alpha) * short_ema
        long_ema = long_alpha * close[i] + (1.0 - long_alpha) * long_ema

        # Only compute if ATR is available (C++ uses variable-length ATR for early bars)
        if not np.isnan(atr_abs[i]):
            denom = sqrt_center * atr_abs[i] + 1e-60
            macd_normalized = (short_ema - long_ema) / denom
            out[i] = _ncdf(macd_normalized, 1.0)

    # Second pass: if signal > 1, smooth ALL compressed values and subtract
    if signal_len > 1:
        sig_alpha = 2.0 / (signal_len + 1.0)
        smoothed = out[0]  # C++ initializes with output[0]

        for i in range(1, n):
            smoothed = sig_alpha * out[i] + (1.0 - sig_alpha) * smoothed
            out[i] -= smoothed

    # Mark warmup bars as NaN (C++ front_bad = long_len + signal_len)
    warmup = long_len + signal_len if signal_len > 1 else long_len
    for i in range(min(warmup, n)):
        out[i] = np.nan

    return out


def macd(high: np.ndarray, low: np.ndarray, close: np.ndarray, short_period: int = 12, long_period: int = 26,
         signal_period: int = 9) -> np.ndarray:
    """
    MACD (line or histogram), ATR-normalised, CDF-compressed.

    Parameters
    ----------
    high, low, close : array-like
    short_period  : int  Short EMA length (default 12).
    long_period   : int  Long EMA length (default 26).
    signal_period : int  Signal EMA length (default 9).  Use 1 to get
                        the raw normalised MACD line instead of histogram.

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    if short_period < 1:
        raise ValueError(f"short_period must be >= 1, got {short_period}")
    if signal_period < 1:
        raise ValueError(f"signal_period must be >= 1, got {signal_period}")
    if long_period <= short_period:
        raise ValueError(f"long_period must be > short_period, got long={long_period} short={short_period}")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    # C++ uses ATR with length = long_period + signal_period for normalization
    atr_length = long_period + signal_period
    atr_abs = atr_kernel(high, low, close, atr_length, False)
    return _macd_kernel(close, atr_abs, short_period, long_period, signal_period)


# --------------------------------------------------------------------------- #
# 7. PPO  (COMP_VAR.CPP:509–546)                                             #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ppo_kernel(close: np.ndarray, short_len: int, long_len: int, signal_len: int) -> np.ndarray:
    """
    Percentage Price Oscillator.

    raw_ppo = 100 * (short_EMA - long_EMA) / (long_EMA + 1e-15)
    Compressed: 100 * Phi(0.2 * raw_ppo) - 50.
    If signal_len > 1: subtract EMA of compressed PPO.
    front_bad = long_len - 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    if n < long_len:
        return out

    short_alpha = 2.0 / (short_len + 1.0)
    long_alpha = 2.0 / (long_len + 1.0)
    sig_alpha = 2.0 / (signal_len + 1.0) if signal_len > 1 else 0.0

    short_ema = close[0]
    long_ema = close[0]
    signal_ema = 0.0
    sig_initialized = False

    for i in range(1, n):
        short_ema = short_alpha * close[i] + (1.0 - short_alpha) * short_ema
        long_ema = long_alpha * close[i] + (1.0 - long_alpha) * long_ema

        if i < long_len - 1:
            continue

        raw_ppo = 100.0 * (short_ema - long_ema) / (long_ema + 1e-15)
        compressed = _ncdf(raw_ppo, 0.2)

        if signal_len > 1:
            if not sig_initialized:
                signal_ema = compressed
                sig_initialized = True
            else:
                signal_ema = sig_alpha * compressed + (1.0 - sig_alpha) * signal_ema
            out[i] = compressed - signal_ema
        else:
            out[i] = compressed
    return out


def ppo(close: np.ndarray, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> np.ndarray:
    """
    Percentage Price Oscillator.

    Parameters
    ----------
    close        : array-like
    short_period : int  Short EMA length (default 12).
    long_period  : int  Long EMA length (default 26).
    signal_period: int  Signal EMA length (default 9).  Use 1 for raw PPO.

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    return _ppo_kernel(close, short_period, long_period, signal_period)


# --------------------------------------------------------------------------- #
# 8. PRICE CHANGE OSCILLATOR  (COMP_VAR.CPP:971–1011)                       #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _price_change_osc_kernel(close: np.ndarray, atr_log: np.ndarray, short_len: int, mult: float) -> np.ndarray:
    """
    Ratio of short to long average absolute log-returns, ATR-normalised.

    long_len = short_len * mult (rounded to int).
    denom = (0.36 + 1/short_len + 0.7 * log(0.5*mult)/1.609) * ATR_log + 1e-60
    Compression: 100 * Phi(4.0 * value) - 50.
    front_bad = long_len (approximately).
    """
    long_len = int(short_len * mult)
    n = len(close)
    out = np.full(n, np.nan)

    front_bad = long_len
    if n <= front_bad:
        return out

    # empirical denom scale
    v = math.log(0.5 * mult) / 1.609
    denom_scale = 0.36 + 1.0 / short_len + 0.7 * v

    for i in range(front_bad, n):
        # short avg abs log-return
        s_sum = 0.0
        for k in range(i - short_len + 1, i + 1):
            s_sum += math.fabs(math.log(close[k] / close[k - 1]))
        s_val = s_sum / short_len

        # long avg abs log-return
        l_sum = 0.0
        for k in range(i - long_len + 1, i + 1):
            l_sum += math.fabs(math.log(close[k] / close[k - 1]))
        l_val = l_sum / long_len

        denom = denom_scale * atr_log[i] + 1e-60
        val = (s_val - l_val) / denom
        out[i] = _ncdf(val, 4.0)
    return out


def price_change_osc(high: np.ndarray, low: np.ndarray, close: np.ndarray, short_period: int = 10,
                     multiplier: float = 4.0) -> np.ndarray:
    """
    Price Change Oscillator.

    Compares average absolute log-returns over a short vs long window.
    The long window length = ``short_period * multiplier``.

    Parameters
    ----------
    high, low, close : array-like
    short_period : int    Short window length (default 10).
    multiplier   : float  Long/short ratio (default 4.0).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    long_len = int(short_period * multiplier)
    atr_log = atr_kernel(high, low, close, long_len, True)
    return _price_change_osc_kernel(close, atr_log, short_period, multiplier)


# --------------------------------------------------------------------------- #
# 9. CLOSE MINUS MOVING AVERAGE  (COMP_VAR.CPP:860–882)                     #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _close_minus_ma_kernel(log_close: np.ndarray, atr_log: np.ndarray, lookback: int) -> np.ndarray:
    """
    Distance from current log-price to rolling log-price mean, ATR-scaled.

    value = (log_close[i] - mean(log_close[i-lookback..i-1])) /
            (ATR_log[i] * sqrt(lookback + 1) + 1e-60)
    Note: C++ excludes current bar from mean calculation.
    Compression: 100 * Phi(1.0 * value) - 50.
    front_bad = lookback (to have lookback bars before current).
    """
    n = len(log_close)
    out = np.full(n, np.nan)

    sqrt_lkp1 = math.sqrt(lookback + 1.0)
    for i in range(lookback, n):
        lmean = 0.0
        # C++ uses [i-lookback .. i-1] (excludes current bar)
        for k in range(i - lookback, i):
            lmean += log_close[k]
        lmean /= lookback

        denom = atr_log[i] * sqrt_lkp1 + 1e-60
        val = (log_close[i] - lmean) / denom
        out[i] = _ncdf(val, 1.0)
    return out


def close_minus_ma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, atr_period: int = 60) -> np.ndarray:
    """
    Close Minus Moving Average, log-ATR-normalised.

    Parameters
    ----------
    high, low, close : array-like
    period     : int  MA lookback (default 20).
    atr_period : int  ATR lookback for normalisation (default 60).
                     Masters uses the same as the MA or a longer period.

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    log_close = np.log(close)
    atr_log = atr_kernel(high, low, close, atr_period, True)

    return _close_minus_ma_kernel(log_close, atr_log, period)


# --------------------------------------------------------------------------- #
# 10. PRICE INTENSITY  (COMP_VAR.CPP:628–667)                               #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _price_intensity_kernel(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, smooth_len: int) -> np.ndarray:
    """
    (Close – Open) / True Range, optionally EMA-smoothed.

    True range includes the prior close:
      TR = max(high-low, |high-close_prev|, |close_prev-low|)
    Raw PI = (close - open) / (TR + 1e-60).
    If smooth_len > 1: EMA with alpha = 2/(smooth_len+1).
    Compression multiplier: 0.8 * sqrt(max(1, smooth_len)).
    front_bad = 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    if n < 2:
        return out

    mult = 0.8 * math.sqrt(max(1.0, float(smooth_len)))
    smooth_alpha = 2.0 / (smooth_len + 1.0) if smooth_len > 1 else 0.0
    ema = 0.0
    initialized = False

    for i in range(1, n):
        tr = high[i] - low[i]
        v1 = math.fabs(high[i] - close[i - 1])
        v2 = math.fabs(close[i - 1] - low[i])
        if v1 > tr:
            tr = v1
        if v2 > tr:
            tr = v2

        raw = (close[i] - open_[i]) / (tr + 1e-60)
        if smooth_len > 1:
            if not initialized:
                ema = raw
                initialized = True
            else:
                ema = smooth_alpha * raw + (1.0 - smooth_alpha) * ema
            out[i] = _ncdf(ema, mult)
        else:
            out[i] = _ncdf(raw, mult)
    return out


def price_intensity(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, smooth_period: int = 1) -> np.ndarray:
    """
    Price Intensity: (Close – Open) / True Range.

    Parameters
    ----------
    open_, high, low, close : array-like  OHLC data.
    smooth_period : int  EMA smoothing period (<=1 means no smoothing).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return _price_intensity_kernel(open_, high, low, close, smooth_period)


# --------------------------------------------------------------------------- #
# 11. REACTIVITY  (COMP_VAR.CPP:1130–1189)                                  #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _reactivity_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, lookback: int, multiplier: int) -> np.ndarray:
    """
    Gietzen's Reactivity indicator.

    alpha = 2 / (lookback * multiplier + 1)
    range = max(high[i-lookback..i]) - min(low[i-lookback..i])
    smoothed_range, smoothed_vol: EMA seeded at icase = lookback.
    aspect_ratio = (range / smoothed_range) / (volume / smoothed_vol)
    raw = aspect_ratio * (close[i] - close[i-lookback]) / (smoothed_range + 1e-60)
    Compression: 100 * Phi(0.6 * raw) - 50.
    front_bad = lookback.
    """
    n = len(close)
    out = np.full(n, np.nan)

    if n <= lookback:
        return out

    alpha = 2.0 / (lookback * multiplier + 1.0)
    one_minus = 1.0 - alpha

    # seed at bar `lookback`
    hi0 = high[0]
    lo0 = low[0]
    for k in range(1, lookback + 1):
        if high[k] > hi0:
            hi0 = high[k]
        if low[k] < lo0:
            lo0 = low[k]

    smoothed_range = hi0 - lo0
    if smoothed_range < 1e-60:
        smoothed_range = 1e-60
    smoothed_vol = volume[lookback]
    if smoothed_vol < 1e-60:
        smoothed_vol = 1e-60

    for i in range(lookback, n):
        # rolling high/low over [i-lookback, i]
        hi = high[i - lookback]
        lo = low[i - lookback]
        for k in range(i - lookback + 1, i + 1):
            if high[k] > hi:
                hi = high[k]
            if low[k] < lo:
                lo = low[k]

        rng = hi - lo

        # update EMA (use current bar's range and volume)
        smoothed_range = alpha * rng + one_minus * smoothed_range
        vol = volume[i]
        if vol < 1e-60:
            vol = 1e-60
        smoothed_vol = alpha * vol + one_minus * smoothed_vol

        aspect = (rng / (smoothed_range + 1e-60)) / (vol / smoothed_vol)
        raw = aspect * (close[i] - close[i - lookback]) / (smoothed_range + 1e-60)
        out[i] = _ncdf(raw, 0.6)
    return out


def reactivity(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 10, multiplier: int = 4) -> np.ndarray:
    """
    Reactivity (Gietzen).

    Measures whether a bar's price change is large relative to its trading range and whether that range is large relative
    to smoothed range, adjusted for volume.

    Parameters
    ----------
    high, low, close, volume : array-like  OHLCV data.
    period     : int  Lookback for range and price-change (default 10).
    multiplier : int  EMA smoothing multiplier (1–32, default 4).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return _reactivity_kernel(high, low, close, volume, period, multiplier)
