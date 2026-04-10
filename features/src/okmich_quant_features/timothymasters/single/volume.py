"""
Volume-based indicators #24–32.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source file: COMP_VAR.CPP.

Indicators
----------
24. intraday_intensity    (2*close - high - low) / (high - low) * volume, MA: COMP_VAR.CPP:1058–1123
25. money_flow            Intraday intensity divided by mean volume (Chaikin): COMP_VAR.CPP:1058–1123
26. price_volume_fit      OLS slope of log(close) on log(volume+1), CDF-compressed: COMP_VAR.CPP:1196–1237
27. vwma_ratio            Log ratio of VWAP to simple average price, sqrt-scaled: COMP_VAR.CPP:1244–1277
28. normalized_obv        Signed-volume ratio normalised by total volume: COMP_VAR.CPP:1284–1333
29. delta_obv             Difference of normalised OBV over a lag period: COMP_VAR.CPP:1284–1333
30. normalized_pvi        Sum of log-returns on rising-volume bars, volatility-scaled: COMP_VAR.CPP:1340–1386
31. normalized_nvi        Sum of log-returns on falling-volume bars, volatility-scaled: COMP_VAR.CPP:1340–1386
32. volume_momentum       Log ratio of short to long average volume, cube-root-scaled: COMP_VAR.CPP:1393–1437
"""

import math

import numpy as np
from numba import njit

from ._helpers import variance_kernel


# --------------------------------------------------------------------------- #
# Shared Numba helper                                                          #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ncdf(x: float, k: float) -> float:
    """100 * Phi(k * x) - 50."""
    return 50.0 * math.erf(k * x * 0.7071067811865476)


# --------------------------------------------------------------------------- #
# 24–25. INTRADAY INTENSITY / MONEY FLOW  (COMP_VAR.CPP:1058–1123)          #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _intraday_intensity_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                               lookback: int, smooth_len: int, is_money_flow: bool) -> np.ndarray:
    """
    Intraday intensity (or money flow) with optional EMA volume scaling.

    raw[i] = 100 * (2*close - high - low) / (high - low) * volume   (if hl > 0)
    MA over ``lookback`` bars.
    If ``smooth_len > 1``: divide by EMA(volume, smooth_len).
    If ``is_money_flow``:  divide by MA(volume, lookback) instead.
    front_bad = lookback - 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    # pre-compute per-bar raw intensity
    raw = np.zeros(n)
    for i in range(n):
        hl = high[i] - low[i]
        if hl < 1e-60:
            raw[i] = 0.0
        else:
            raw[i] = 100.0 * (2.0 * close[i] - high[i] - low[i]) / hl * volume[i]

    # EMA of volume (for optional smooth_len scaling)
    ema_vol = volume[0]
    ema_alpha = 2.0 / (smooth_len + 1.0) if smooth_len > 1 else 0.0

    # Pre-warm EMA for bars 1..(lookback-2) so the outer loop can do a single
    # incremental step per bar instead of recomputing from bar 1 every time.
    if smooth_len > 1:
        for k in range(1, lookback - 1):
            ema_vol = ema_alpha * volume[k] + (1.0 - ema_alpha) * ema_vol

    for i in range(lookback - 1, n):
        # moving average of intensity over lookback
        s = 0.0
        for k in range(i - lookback + 1, i + 1):
            s += raw[k]
        ma_intensity = s / lookback

        if is_money_flow:
            # divide by mean volume over the lookback window
            vol_sum = 0.0
            for k in range(i - lookback + 1, i + 1):
                vol_sum += volume[k]
            mean_vol = vol_sum / lookback
            if mean_vol > 0.0:
                out[i] = ma_intensity / mean_vol
            else:
                out[i] = 0.0
        elif smooth_len > 1:
            # Single incremental EMA step (O(1) per bar, not O(n))
            ema_vol = ema_alpha * volume[i] + (1.0 - ema_alpha) * ema_vol
            if ema_vol > 0.0:
                out[i] = ma_intensity / ema_vol
            else:
                out[i] = 0.0
        else:
            out[i] = ma_intensity

    return out


def intraday_intensity(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
                       period: int = 14, smooth_period: int = 0) -> np.ndarray:
    """
    Intraday Intensity.

    MA of ``(2*close - high - low) / (high - low) * volume``.
    If ``smooth_period > 1``: divide by EMA-smoothed volume.

    Parameters
    ----------
    high, low, close, volume : array-like  OHLCV data.
    period        : int  MA lookback (default 14).
    smooth_period : int  Volume EMA period (0 = no volume scaling).

    Returns
    -------
    np.ndarray  Raw intraday intensity.  Warmup bars are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if smooth_period < 0:
        raise ValueError(f"smooth_period must be >= 0, got {smooth_period}")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return _intraday_intensity_kernel(high, low, close, volume, period, smooth_period, False)


def money_flow(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray,
               period: int = 14) -> np.ndarray:
    """
    Chaikin Money Flow: intraday intensity divided by mean volume.

    Parameters
    ----------
    high, low, close, volume : array-like  OHLCV data.
    period : int  MA lookback (default 14).

    Returns
    -------
    np.ndarray  in approximately [-100, 100].  Warmup bars are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return _intraday_intensity_kernel(high, low, close, volume, period, 0, True)


# --------------------------------------------------------------------------- #
# 26. PRICE VOLUME FIT  (COMP_VAR.CPP:1196–1237)                            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _price_volume_fit_kernel(log_close: np.ndarray, log_vol1: np.ndarray, lookback: int) -> np.ndarray:
    """
    OLS slope of log(close) on log(volume+1) over a rolling window.

    coef = Cov(log_vol, log_close) / Var(log_vol)
    output = 100 * Phi(9.0 * coef) - 50
    front_bad = lookback - 1.
    """
    n = len(log_close)
    out = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        start = i - lookback + 1

        xmean = 0.0
        ymean = 0.0
        for k in range(start, i + 1):
            xmean += log_vol1[k]
            ymean += log_close[k]
        xmean /= lookback
        ymean /= lookback

        xss = 0.0
        xy = 0.0
        for k in range(start, i + 1):
            dx = log_vol1[k] - xmean
            xss += dx * dx
            xy += dx * (log_close[k] - ymean)

        coef = xy / (xss + 1e-30)
        out[i] = _ncdf(coef, 9.0)

    return out


def price_volume_fit(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Price-Volume Fit: OLS regression slope of log(close) on log(volume+1).

    A positive slope indicates that higher-volume bars tend to close higher.

    Parameters
    ----------
    close, volume : array-like
    period : int  Rolling regression window (default 20).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    log_close = np.log(np.maximum(close, 1e-60))
    log_vol1 = np.log(volume + 1.0)

    return _price_volume_fit_kernel(log_close, log_vol1, period)


# --------------------------------------------------------------------------- #
# 27. VOLUME WEIGHTED MA RATIO  (COMP_VAR.CPP:1244–1277)                   #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _vwma_ratio_kernel(close: np.ndarray, volume: np.ndarray, lookback: int) -> np.ndarray:
    """
    Log ratio of VWAP to simple average price, sqrt-scaled.

    vwap   = Σ(volume * close) / Σ(volume)
    sma    = Σ(close) / lookback
    value  = 1000 * log(lookback * Σ(v*c) / (Σ(v) * Σ(c))) / sqrt(lookback)
    output = 100 * Phi(1.0 * value) - 50
    front_bad = lookback - 1.
    """
    n = len(close)
    out = np.full(n, np.nan)

    sqrt_lkb = math.sqrt(lookback)

    for i in range(lookback - 1, n):
        start = i - lookback + 1

        numer = 0.0   # Σ(volume * close)
        sum_v = 0.0   # Σ(volume)
        sum_c = 0.0   # Σ(close)
        for k in range(start, i + 1):
            numer += volume[k] * close[k]
            sum_v += volume[k]
            sum_c += close[k]

        if sum_v < 1e-60 or sum_c < 1e-60:
            out[i] = 0.0
        else:
            ratio = lookback * numer / (sum_v * sum_c)
            if ratio > 1e-60:
                val = 1000.0 * math.log(ratio) / sqrt_lkb
                out[i] = _ncdf(val, 1.0)
            else:
                out[i] = -50.0

    return out


def vwma_ratio(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Volume-Weighted MA Ratio: log(VWAP / simple average), sqrt-scaled.

    Parameters
    ----------
    close, volume : array-like
    period : int  Rolling window (default 20).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return _vwma_ratio_kernel(close, volume, period)


# --------------------------------------------------------------------------- #
# 28–29. NORMALIZED/DELTA OBV  (COMP_VAR.CPP:1284–1333)                    #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _obv_kernel(close: np.ndarray, volume: np.ndarray, lookback: int) -> np.ndarray:
    """
    Normalised On-Balance Volume.

    For each bar i (from lookback onward):
      signed_vol = Σ ±volume[j] over j in [i-lookback+1, i]
                   (+ if close[j] > close[j-1], - if close[j] < close[j-1])
      total_vol  = Σ volume[j]
      value      = signed_vol / total_vol * sqrt(lookback)
      output     = 100 * Phi(0.6 * value) - 50

    front_bad = lookback (needs one prior close for direction).
    """
    n = len(close)
    out = np.full(n, np.nan)

    sqrt_lkb = math.sqrt(lookback)

    for i in range(lookback, n):
        signed = 0.0
        total = 0.0
        for k in range(i - lookback + 1, i + 1):
            total += volume[k]
            if close[k] > close[k - 1]:
                signed += volume[k]
            elif close[k] < close[k - 1]:
                signed -= volume[k]

        if total < 1e-60:
            out[i] = 0.0
        else:
            val = signed / total * sqrt_lkb
            out[i] = _ncdf(val, 0.6)

    return out


def normalized_obv(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Normalised On-Balance Volume.

    Parameters
    ----------
    close, volume : array-like
    period : int  Rolling window (default 20).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    return _obv_kernel(close, volume, period)


def delta_obv(close: np.ndarray, volume: np.ndarray, period: int = 20, delta_period: int = 5) -> np.ndarray:
    """
    Delta (difference) of Normalised OBV over ``delta_period`` bars.

    Parameters
    ----------
    close, volume  : array-like
    period         : int  OBV rolling window (default 20).
    delta_period   : int  Lag for differencing (default 5).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    obv = _obv_kernel(close, volume, period)

    # difference: out[i] = obv[i] - obv[i - delta_period]
    n = len(obv)
    out = np.full(n, np.nan)
    for i in range(delta_period, n):
        if not math.isnan(obv[i]) and not math.isnan(obv[i - delta_period]):
            out[i] = obv[i] - obv[i - delta_period]
    return out


# --------------------------------------------------------------------------- #
# 30–31. NORMALIZED PVI / NVI  (COMP_VAR.CPP:1340–1386)                    #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _pvi_nvi_kernel(close: np.ndarray, volume: np.ndarray, lookback: int, vol_length: int, is_pvi: bool) -> np.ndarray:
    """
    Normalised Positive (or Negative) Volume Index.

    PVI: sum log-returns when volume > prior volume.
    NVI: sum log-returns when volume < prior volume.
    Normalised: divide by sqrt(lookback) then by sqrt(volatility).
    output = 100 * Phi(0.5 * value) - 50.
    front_bad = max(lookback, vol_length).
    """
    n = len(close)
    out = np.full(n, np.nan)

    front_bad = lookback if lookback > vol_length else vol_length

    for i in range(front_bad, n):
        s = 0.0
        for k in range(i - lookback + 1, i + 1):
            if is_pvi:
                cond = volume[k] > volume[k - 1]
            else:
                cond = volume[k] < volume[k - 1]
            if cond:
                s += math.log(close[k] / close[k - 1])

        val = s / math.sqrt(lookback)

        # normalise by volatility (square root of change variance)
        vol_start = i - vol_length + 1
        if vol_start < 1:
            out[i] = np.nan
            continue

        mean_r = 0.0
        for k in range(vol_start, i + 1):
            mean_r += math.log(close[k] / close[k - 1])
        mean_r /= vol_length

        var_r = 0.0
        for k in range(vol_start, i + 1):
            d = math.log(close[k] / close[k - 1]) - mean_r
            var_r += d * d
        var_r /= vol_length

        vol_denom = math.sqrt(var_r) + 1e-60
        val /= vol_denom

        out[i] = _ncdf(val, 0.5)

    return out


def normalized_pvi(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Normalised Positive Volume Index.

    Sums log-returns when volume is increasing, normalised by volatility.

    Parameters
    ----------
    close, volume : array-like
    period : int  Rolling window (default 20).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    vol_length = max(2 * period, 250)
    return _pvi_nvi_kernel(close, volume, period, vol_length, True)


def normalized_nvi(close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Normalised Negative Volume Index.

    Sums log-returns when volume is decreasing, normalised by volatility.

    Parameters
    ----------
    close, volume : array-like
    period : int  Rolling window (default 20).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    vol_length = max(2 * period, 250)
    return _pvi_nvi_kernel(close, volume, period, vol_length, False)


# --------------------------------------------------------------------------- #
# 32. VOLUME MOMENTUM  (COMP_VAR.CPP:1393–1437)                             #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _volume_momentum_kernel(volume: np.ndarray, short_len: int, mult: float) -> np.ndarray:
    """
    Log ratio of short-term to long-term average volume, cube-root-scaled.

    long_len = int(short_len * mult)
    short_vol = MA(volume, short_len)
    long_vol  = MA(volume, long_len)
    denom     = exp(log(mult) / 3)   (cube root of mult)
    value     = log(short_vol / long_vol) / denom
    output    = 100 * Phi(3.0 * value) - 50
    front_bad = long_len - 1.
    """
    long_len = int(short_len * mult)
    n = len(volume)
    out = np.full(n, np.nan)

    denom = math.exp(math.log(mult) / 3.0)

    for i in range(long_len - 1, n):
        s_sum = 0.0
        for k in range(i - short_len + 1, i + 1):
            s_sum += volume[k]
        short_vol = s_sum / short_len

        l_sum = 0.0
        for k in range(i - long_len + 1, i + 1):
            l_sum += volume[k]
        long_vol = l_sum / long_len

        if short_vol < 1e-60 or long_vol < 1e-60:
            out[i] = 0.0
        else:
            val = math.log(short_vol / long_vol) / denom
            out[i] = _ncdf(val, 3.0)

    return out


def volume_momentum(volume: np.ndarray, short_period: int = 10, multiplier: float = 4.0) -> np.ndarray:
    """
    Volume Momentum: log ratio of short to long average volume.

    Parameters
    ----------
    volume       : array-like
    short_period : int    Short average window (default 10).
    multiplier   : float  Long/short ratio (default 4.0).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    volume = np.asarray(volume, dtype=np.float64)
    return _volume_momentum_kernel(volume, short_period, multiplier)