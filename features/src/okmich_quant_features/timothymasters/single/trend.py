"""
Trend strength indicators #12–21.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source file: COMP_VAR.CPP.

Indicators
----------
12. linear_trend         Linear Legendre regression coefficient, ATR-scaled: COMP_VAR.CPP:553–621
13. quadratic_trend      Quadratic Legendre regression coefficient, ATR-scaled: COMP_VAR.CPP:553–621
14. cubic_trend          Cubic Legendre regression coefficient, ATR-scaled: COMP_VAR.CPP:553–621
15. linear_deviation     Residual of current bar from linear Legendre fit: COMP_VAR.CPP:889–964
16. quadratic_deviation  Residual from quadratic Legendre fit: COMP_VAR.CPP:889–964
17. cubic_deviation      Residual from cubic Legendre fit: COMP_VAR.CPP:889–964
18. adx                  Average Directional Index (Wilder's 3-phase): COMP_VAR.CPP:674–794
19. aroon_up             Bars since highest high, normalised: COMP_VAR.CPP:801–853
20. aroon_down           Bars since lowest low, normalised: COMP_VAR.CPP:801–853
21. aroon_diff           Aroon Up minus Aroon Down: COMP_VAR.CPP:801–853

Design notes
------------
* Linear/Quadratic/Cubic Trend use pre-computed Legendre weights from ``_legendre.legendre_weights(n)`` (cached Python-side)
    then pass the weight arrays into Numba kernels together with the log-price series.
* Trend output is degraded by R², so poor polynomial fits produce smaller values (Masters' quality-weighting).
* ADX follows Wilder's exact three-phase initialisation — Phase 1 (simple sums), Phase 2 (EMA + simple ADX average), Phase 3 (EMA everything).
* All outputs use the normal-CDF compression  100·Φ(k·x)−50  except ADX which is already in [0, 100].
"""

import math

import numpy as np
from numba import njit

from ._helpers import atr_kernel
from ._legendre import legendre_weights, legendre_dot


# --------------------------------------------------------------------------- #
# Shared Numba helper                                                          #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ncdf(x: float, k: float) -> float:
    """100 * Phi(k * x) - 50."""
    return 50.0 * math.erf(k * x * 0.7071067811865476)


# --------------------------------------------------------------------------- #
# 12–14. TREND INDICATORS  (COMP_VAR.CPP:553–621)                            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _trend_kernel(log_prices: np.ndarray, atr_log: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray,
                  lookback: int, poly_index: int) -> np.ndarray:
    """
    Legendre trend coefficient for one polynomial degree.

    poly_index: 0 = linear (c1 / w1), 1 = quadratic (c2 / w2), 2 = cubic (c3 / w3).

    For each bar i:
      c0 = mean(log_price window)           (constant component)
      c1 = dot(log_price window, w1)        (linear component)
      c2 = dot(log_price window, w2)        (quadratic; only when poly_index >= 1)
      c3 = dot(log_price window, w3)        (cubic; only when poly_index == 2)

    Because Legendre weights are orthonormal, the total variance explained
    by the first n_poly components = c1² + c2² + ... + cn_poly².

      yss    = Σ (log_price - c0)²           (total SS around mean)
      ss_exp = c1² [+ c2² [+ c3²]]           (explained SS)
      rsq    = max(0, ss_exp / (yss + 1e-60))
      raw    = c_{poly_index+1} * 2 / (atr * k_len + 1e-60)
      output = 100 * Phi(rsq * raw) - 50

    k_len = lookback - 1  (or 2 if lookback == 2).
    front_bad = lookback - 1.
    """
    n = len(log_prices)
    out = np.full(n, np.nan)

    k_len = 2 if lookback == 2 else lookback - 1

    for i in range(lookback - 1, n):
        start = i - lookback + 1

        # --- mean (c0) ---
        c0 = 0.0
        for k in range(lookback):
            c0 += log_prices[start + k]
        c0 /= lookback

        # --- polynomial dot products ---
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        for k in range(lookback):
            lp = log_prices[start + k]
            c1 += lp * w1[k]
            if poly_index >= 1:
                c2 += lp * w2[k]
            if poly_index >= 2:
                c3 += lp * w3[k]

        # --- total SS around mean ---
        yss = 0.0
        for k in range(lookback):
            d = log_prices[start + k] - c0
            yss += d * d

        # --- target coefficient ---
        if poly_index == 0:
            coef = c1
            w_target = w1
        elif poly_index == 1:
            coef = c2
            w_target = w2
        else:
            coef = c3
            w_target = w3

        # --- R²: goodness of fit for THIS polynomial component only ---
        # C++ computes R² = 1 - Σ(residual²) / Σ(total²)
        # where residual = (actual - mean) - (predicted_from_this_component)
        rss = 0.0
        for k in range(lookback):
            lp = log_prices[start + k]
            actual_dev = lp - c0                    # actual deviation from mean
            predicted_dev = coef * w_target[k]      # predicted deviation using this polynomial
            residual = actual_dev - predicted_dev   # residual error
            rss += residual * residual

        rsq = 1.0 - rss / (yss + 1e-60)
        if rsq < 0.0:
            rsq = 0.0

        denom = atr_log[i] * k_len + 1e-60
        raw = coef * 2.0 / denom

        out[i] = _ncdf(rsq * raw, 1.0)

    return out


def _compute_trend(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int, atr_period: int,
                   poly_index: int,   # 0=linear(w1), 1=quadratic(w2), 2=cubic(w3)
                   ) -> np.ndarray:
    """Shared implementation for linear/quadratic/cubic trend."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    log_prices = np.log(close)
    atr_log = atr_kernel(high, low, close, atr_period, True)

    w1, w2, w3 = legendre_weights(lookback)

    return _trend_kernel(log_prices, atr_log, w1, w2, w3, lookback, poly_index)


def linear_trend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, atr_period: int = 60) -> np.ndarray:
    """
    Linear Legendre trend coefficient, R²-weighted, ATR-normalised.

    Parameters
    ----------
    high, low, close : array-like
    period     : int  Regression lookback (default 20).
    atr_period : int  ATR lookback for normalisation (default 60).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    return _compute_trend(high, low, close, period, atr_period, 0)


def quadratic_trend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, atr_period: int = 60) -> np.ndarray:
    """
    Quadratic Legendre trend coefficient, R²-weighted, ATR-normalised.

    Parameters
    ----------
    high, low, close : array-like
    period     : int  Regression lookback (default 20).
    atr_period : int  ATR lookback for normalisation (default 60).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    return _compute_trend(high, low, close, period, atr_period, 1)


def cubic_trend(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, atr_period: int = 60) -> np.ndarray:
    """
    Cubic Legendre trend coefficient, R²-weighted, ATR-normalised.

    Parameters
    ----------
    high, low, close : array-like
    period     : int  Regression lookback (default 20).
    atr_period : int  ATR lookback for normalisation (default 60).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    return _compute_trend(high, low, close, period, atr_period, 2)


# --------------------------------------------------------------------------- #
# 15–17. DEVIATION INDICATORS  (COMP_VAR.CPP:889–964)                       #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _deviation_kernel(log_prices: np.ndarray, w1: np.ndarray, w2: np.ndarray, w3: np.ndarray, lookback: int,
                      n_polys: int) -> np.ndarray:
    """
    Deviation of current bar from a 1-, 2-, or 3-polynomial Legendre fit.

    Fits (c0 + c1*w1 + c2*w2 + c3*w3) to the lookback window, computes RMS residual over the window, then returns:
        output = (log_price[i] - predicted[i]) / (rms_error + 1e-60)

    Compressed: 100 * Phi(0.6 * output) - 50.
    n_polys: 1=linear, 2=quadratic, 3=cubic.
    front_bad = lookback - 1.
    """
    n = len(log_prices)
    out = np.full(n, np.nan)

    for i in range(lookback - 1, n):
        start = i - lookback + 1

        # constant component c0 = mean
        c0 = 0.0
        for k in range(lookback):
            c0 += log_prices[start + k]
        c0 /= lookback

        # polynomial coefficients via dot products (weights orthonormal)
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        for k in range(lookback):
            lp = log_prices[start + k]
            c1 += lp * w1[k]
            if n_polys >= 2:
                c2 += lp * w2[k]
            if n_polys >= 3:
                c3 += lp * w3[k]

        # RMS residual over the window
        ss = 0.0
        for k in range(lookback):
            pred = c0 + c1 * w1[k]
            if n_polys >= 2:
                pred += c2 * w2[k]
            if n_polys >= 3:
                pred += c3 * w3[k]
            d = log_prices[start + k] - pred
            ss += d * d
        rms = math.sqrt(ss / lookback)

        # prediction at current bar (last Legendre weight, index lookback-1)
        last = lookback - 1
        pred_current = c0 + c1 * w1[last]
        if n_polys >= 2:
            pred_current += c2 * w2[last]
        if n_polys >= 3:
            pred_current += c3 * w3[last]

        dev = (log_prices[i] - pred_current) / (rms + 1e-60)
        out[i] = _ncdf(dev, 0.6)

    return out


def _compute_deviation(close: np.ndarray, lookback: int, n_polys: int) -> np.ndarray:
    """Shared implementation for deviation indicators."""
    close = np.asarray(close, dtype=np.float64)

    min_lookback = n_polys + 2    # need more points than poly terms
    if lookback < min_lookback:
        raise ValueError(f"lookback must be >= {min_lookback} for {n_polys}-poly deviation")

    log_prices = np.log(close)
    w1, w2, w3 = legendre_weights(lookback)

    return _deviation_kernel(log_prices, w1, w2, w3, lookback, n_polys)


def linear_deviation(close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Deviation of current log-price from a linear Legendre fit.

    Parameters
    ----------
    close  : array-like  Close prices.
    period : int         Regression lookback (default 20, minimum 3).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    return _compute_deviation(close, period, 1)


def quadratic_deviation(close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Deviation of current log-price from a quadratic Legendre fit.

    Parameters
    ----------
    close  : array-like  Close prices.
    period : int         Regression lookback (default 20, minimum 4).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    return _compute_deviation(close, period, 2)


def cubic_deviation(close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Deviation of current log-price from a cubic Legendre fit.

    Parameters
    ----------
    close  : array-like  Close prices.
    period : int         Regression lookback (default 20, minimum 5).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    return _compute_deviation(close, period, 3)


# --------------------------------------------------------------------------- #
# 18. ADX  (COMP_VAR.CPP:674–794)                                            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _adx_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray, lookback: int) -> np.ndarray:
    """
    Wilder's Average Directional Index (three-phase initialisation).

    Phase 1 (bars 1..lookback):    simple cumulative sums of DM+, DM-, ATR.
    Phase 2 (bars lookback+1..2*lookback-1): EMA update; simple-average ADX.
    Phase 3 (bars 2*lookback onward): EMA update of DM+, DM-, ATR and ADX.

    EMA alpha = (lookback-1) / lookback   (Wilder's smoothing).
    Output range: [0, 100].
    front_bad = 2 * lookback - 1.
    """
    n = len(high)
    out = np.full(n, np.nan)

    if n < 2 * lookback:
        return out

    decay = (lookback - 1.0) / lookback

    # --- Phase 1: bars 1..lookback ---
    dm_plus_sum = 0.0
    dm_minus_sum = 0.0
    atr_sum = 0.0

    for i in range(1, lookback + 1):
        dm_up = high[i] - high[i - 1]
        dm_dn = low[i - 1] - low[i]

        if dm_up < 0.0:
            dm_up = 0.0
        if dm_dn < 0.0:
            dm_dn = 0.0

        if dm_up >= dm_dn:
            dm_dn = 0.0
        else:
            dm_up = 0.0

        dm_plus_sum += dm_up
        dm_minus_sum += dm_dn

        tr = high[i] - low[i]
        v1 = math.fabs(high[i] - close[i - 1])
        v2 = math.fabs(close[i - 1] - low[i])
        if v1 > tr:
            tr = v1
        if v2 > tr:
            tr = v2
        atr_sum += tr

    # --- Phase 2: bars lookback+1..2*lookback-1 ---
    adx_sum = 0.0

    for i in range(lookback + 1, 2 * lookback):
        dm_up = high[i] - high[i - 1]
        dm_dn = low[i - 1] - low[i]
        if dm_up < 0.0:
            dm_up = 0.0
        if dm_dn < 0.0:
            dm_dn = 0.0
        if dm_up >= dm_dn:
            dm_dn = 0.0
        else:
            dm_up = 0.0

        dm_plus_sum = decay * dm_plus_sum + dm_up
        dm_minus_sum = decay * dm_minus_sum + dm_dn

        tr = high[i] - low[i]
        v1 = math.fabs(high[i] - close[i - 1])
        v2 = math.fabs(close[i - 1] - low[i])
        if v1 > tr:
            tr = v1
        if v2 > tr:
            tr = v2
        atr_sum = decay * atr_sum + tr

        di_plus = dm_plus_sum / (atr_sum + 1e-60)
        di_minus = dm_minus_sum / (atr_sum + 1e-60)
        di_sum = di_plus + di_minus
        if di_sum < 1e-60:
            adx_sum += 0.0
        else:
            adx_sum += math.fabs(di_plus - di_minus) / di_sum

    adx = adx_sum / (lookback - 1.0) if lookback > 1 else 0.0

    # --- Phase 3: bars 2*lookback-1 onward ---
    for i in range(2 * lookback - 1, n):
        dm_up = high[i] - high[i - 1]
        dm_dn = low[i - 1] - low[i]
        if dm_up < 0.0:
            dm_up = 0.0
        if dm_dn < 0.0:
            dm_dn = 0.0
        if dm_up >= dm_dn:
            dm_dn = 0.0
        else:
            dm_up = 0.0

        dm_plus_sum = decay * dm_plus_sum + dm_up
        dm_minus_sum = decay * dm_minus_sum + dm_dn

        tr = high[i] - low[i]
        v1 = math.fabs(high[i] - close[i - 1])
        v2 = math.fabs(close[i - 1] - low[i])
        if v1 > tr:
            tr = v1
        if v2 > tr:
            tr = v2
        atr_sum = decay * atr_sum + tr

        di_plus = dm_plus_sum / (atr_sum + 1e-60)
        di_minus = dm_minus_sum / (atr_sum + 1e-60)
        di_sum = di_plus + di_minus
        if di_sum < 1e-60:
            term = 0.0
        else:
            term = math.fabs(di_plus - di_minus) / di_sum

        adx = decay * adx + term / lookback
        out[i] = 100.0 * adx

    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Average Directional Index (Wilder's original three-phase algorithm).

    Parameters
    ----------
    high, low, close : array-like  OHLC data.
    period : int  Lookback (default 14).  front_bad = 2*period - 1.

    Returns
    -------
    np.ndarray  in [0, 100].  Warmup bars are NaN.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    return _adx_kernel(high, low, close, period)


# --------------------------------------------------------------------------- #
# 19–21. AROON  (COMP_VAR.CPP:801–853)                                       #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _aroon_kernel(high: np.ndarray, low: np.ndarray, lookback: int) -> tuple:
    """
    Aroon Up, Down, and Diff.

    For each bar i (from lookback onward):
      - Search [i-lookback, i] for highest high → bars_since_high
      - Search [i-lookback, i] for lowest  low  → bars_since_low
      aroon_up   = 100 * (lookback - bars_since_high) / lookback
      aroon_down = 100 * (lookback - bars_since_low)  / lookback
      aroon_diff = aroon_up - aroon_down

    front_bad = lookback.
    Returns three float64 arrays (up, down, diff).
    """
    n = len(high)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    diff = np.full(n, np.nan)

    for i in range(lookback, n):
        # highest high in [i-lookback, i]
        imax = i
        xmax = high[i]
        for j in range(i - 1, i - lookback - 1, -1):
            if high[j] > xmax:
                xmax = high[j]
                imax = j

        # lowest low in [i-lookback, i]
        imin = i
        xmin = low[i]
        for j in range(i - 1, i - lookback - 1, -1):
            if low[j] < xmin:
                xmin = low[j]
                imin = j

        a_up = 100.0 * (lookback - (i - imax)) / lookback
        a_dn = 100.0 * (lookback - (i - imin)) / lookback

        up[i] = a_up
        dn[i] = a_dn
        diff[i] = a_up - a_dn

    return up, dn, diff


def aroon_up(high: np.ndarray, low: np.ndarray, period: int = 25) -> np.ndarray:
    """
    Aroon Up: 100 * (period - bars_since_period_high) / period.

    Parameters
    ----------
    high, low : array-like
    period    : int  Lookback (default 25).

    Returns
    -------
    np.ndarray  in [0, 100].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    up, _, _ = _aroon_kernel(high, low, period)
    return up


def aroon_down(high: np.ndarray, low: np.ndarray, period: int = 25) -> np.ndarray:
    """
    Aroon Down: 100 * (period - bars_since_period_low) / period.

    Parameters
    ----------
    high, low : array-like
    period    : int  Lookback (default 25).

    Returns
    -------
    np.ndarray  in [0, 100].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    _, dn, _ = _aroon_kernel(high, low, period)
    return dn


def aroon_diff(high: np.ndarray, low: np.ndarray, period: int = 25) -> np.ndarray:
    """
    Aroon Diff: Aroon Up minus Aroon Down.

    Parameters
    ----------
    high, low : array-like
    period    : int  Lookback (default 25).

    Returns
    -------
    np.ndarray  in [-100, 100].  Warmup bars are NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    _, _, df = _aroon_kernel(high, low, period)
    return df
