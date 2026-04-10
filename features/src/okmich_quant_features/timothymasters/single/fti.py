"""
Frequency Trend Intensity indicators #35--38.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source files: FTI.CPP, COMP_VAR.CPP.

Indicators
----------
35. fti_lowpass      Lowpass-filtered price at the dominant period: COMP_VAR.CPP:1548--1595, FTI.CPP
36. fti_best_width   Half-width of the lowpass channel at the dominant period: COMP_VAR.CPP:1548--1595, FTI.CPP
37. fti_best_period  Dominant period selected by FTI local-maximum sort: COMP_VAR.CPP:1548--1595, FTI.CPP
38. fti_best_fti     FTI value at the dominant period, gamma-CDF compressed: COMP_VAR.CPP:1548--1595, FTI.CPP
"""

import math

import numpy as np
from numba import njit
from scipy.special import gammainc


# --------------------------------------------------------------------------- #
# FIR lowpass filter coefficient computation  (FTI.CPP:find_coefs)            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _find_coefs(period: int, half_length: int) -> np.ndarray:
    """
    Compute FIR lowpass filter coefficients for a given period.

    Parameters
    ----------
    period      : int  Filter period.
    half_length : int  Half-length of the symmetric filter.

    Returns
    -------
    c : 1-D float64 array of length half_length + 1
        Symmetric filter coefficients (c[0] is the centre weight).
    """
    PI = 3.14159265358979
    d = np.array([0.35577019, 0.2436983, 0.07211497, 0.00630165])

    c = np.zeros(half_length + 1, dtype=np.float64)

    fact = 2.0 / period
    c[0] = fact
    fact_pi = fact * PI

    for i in range(1, half_length + 1):
        c[i] = math.sin(i * fact_pi) / (i * PI)

    # Taper end point
    c[half_length] *= 0.5

    # Apply Blackman-Harris window and accumulate normalisation sum
    sumg = c[0]
    for i in range(1, half_length + 1):
        s = d[0]
        fact_i = i * PI / half_length
        for j in range(1, 4):
            s += 2.0 * d[j] * math.cos(j * fact_i)
        c[i] *= s
        sumg += 2.0 * c[i]

    # Normalise
    for i in range(half_length + 1):
        c[i] /= sumg

    return c


# --------------------------------------------------------------------------- #
# Core per-bar FTI computation  (FTI.CPP:process)                            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _fti_process_one_bar(close_window: np.ndarray, all_coefs: np.ndarray, min_period: int, max_period: int,
                         half_length: int, lookback: int, beta: float, noise_cut: float) -> tuple:
    """
    Process one bar: compute filtered values, widths, FTI values, and
    sort local maxima to find the dominant period.

    Parameters
    ----------
    close_window : 1-D float64 array of length lookback (chronological, log prices).
    all_coefs    : 2-D float64 array shape (n_periods, half_length+1).
    min_period   : int
    max_period   : int
    half_length  : int
    lookback     : int
    beta         : float  Fractile for channel width (e.g. 0.95).
    noise_cut    : float  Noise threshold fraction (e.g. 0.20).

    Returns
    -------
    filtered     : 1-D float64 array (n_periods,) -- filtered log-price per period.
    width        : 1-D float64 array (n_periods,) -- channel half-width per period.
    fti_vals     : 1-D float64 array (n_periods,) -- FTI value per period.
    best_idx     : int -- index (0-based, relative to min_period) of best period.
    """
    n_periods = max_period - min_period + 1
    channel_len = lookback - half_length

    # Extend y by LS projection
    y_len = lookback + half_length
    y = np.zeros(y_len, dtype=np.float64)
    for j in range(lookback):
        y[j] = close_window[j]

    # Fit LS line to last half_length+1 points, project half_length more
    hl1 = half_length + 1
    xmean = -0.5 * half_length
    ymean = 0.0
    for i in range(hl1):
        ymean += y[lookback - 1 - i]
    ymean /= hl1

    xsq = 0.0
    xy = 0.0
    for i in range(hl1):
        xdiff = -float(i) - xmean
        ydiff = y[lookback - 1 - i] - ymean
        xsq += xdiff * xdiff
        xy += xdiff * ydiff

    slope = xy / (xsq + 1e-30)
    for i in range(half_length):
        y[lookback + i] = (float(i) + 1.0 - xmean) * slope + ymean

    filtered = np.zeros(n_periods, dtype=np.float64)
    width = np.zeros(n_periods, dtype=np.float64)
    fti_vals = np.zeros(n_periods, dtype=np.float64)

    # Work arrays
    diff_work = np.zeros(channel_len, dtype=np.float64)
    leg_work = np.zeros(channel_len, dtype=np.float64)

    for ip in range(n_periods):
        cptr = all_coefs[ip]

        # Convolution + leg tracking
        extreme_type = 0
        extreme_value = 0.0
        n_legs = 0
        longest_leg = 0.0
        prior = 0.0

        for iy in range(half_length, lookback):
            # Symmetric FIR convolution
            s = cptr[0] * y[iy]
            for ii in range(1, half_length + 1):
                s += cptr[ii] * (y[iy + ii] + y[iy - ii])

            if iy == lookback - 1:
                filtered[ip] = s

            diff_work[iy - half_length] = abs(y[iy] - s)

            # Leg tracking (zigzag on filtered signal)
            if iy == half_length:
                extreme_type = 0
                extreme_value = s
                n_legs = 0
                longest_leg = 0.0
            elif extreme_type == 0:
                if s > extreme_value:
                    extreme_type = -1
                elif s < extreme_value:
                    extreme_type = 1
            elif iy == lookback - 1:
                leg_work[n_legs] = abs(extreme_value - s)
                if leg_work[n_legs] > longest_leg:
                    longest_leg = leg_work[n_legs]
                n_legs += 1
            else:
                if extreme_type == 1 and s > prior:
                    # Was going down, now going up
                    leg_work[n_legs] = extreme_value - prior
                    if leg_work[n_legs] > longest_leg:
                        longest_leg = leg_work[n_legs]
                    n_legs += 1
                    extreme_type = -1
                    extreme_value = prior
                elif extreme_type == -1 and s < prior:
                    # Was going up, now going down
                    leg_work[n_legs] = prior - extreme_value
                    if leg_work[n_legs] > longest_leg:
                        longest_leg = leg_work[n_legs]
                    n_legs += 1
                    extreme_type = 1
                    extreme_value = prior

            prior = s

        # Width: sort diff_work, take beta fractile
        sorted_diff = np.sort(diff_work[:channel_len])
        idx = int(beta * (channel_len + 1)) - 1
        if idx < 0:
            idx = 0
        if idx >= channel_len:
            idx = channel_len - 1
        width[ip] = sorted_diff[idx]

        # FTI: mean of non-noise legs / width
        if n_legs > 0 and longest_leg > 0.0:
            noise_level = noise_cut * longest_leg
            leg_sum = 0.0
            leg_count = 0
            for il in range(n_legs):
                if leg_work[il] > noise_level:
                    leg_sum += leg_work[il]
                    leg_count += 1
            if leg_count > 0:
                leg_sum /= leg_count
            fti_vals[ip] = leg_sum / (width[ip] + 1e-5)
        else:
            fti_vals[ip] = 0.0

    # Sort FTI local maxima descending by FTI value
    # Collect local maxima (including endpoints)
    max_count = 0
    max_fti = np.zeros(n_periods, dtype=np.float64)
    max_idx = np.zeros(n_periods, dtype=np.int64)

    for i in range(n_periods):
        is_peak = False
        if i == 0 or i == n_periods - 1:
            is_peak = True
        elif fti_vals[i] >= fti_vals[i - 1] and fti_vals[i] >= fti_vals[i + 1]:
            is_peak = True
        if is_peak:
            max_fti[max_count] = fti_vals[i]
            max_idx[max_count] = i
            max_count += 1

    # Simple insertion sort descending by fti value
    for i in range(1, max_count):
        key_fti = max_fti[i]
        key_idx = max_idx[i]
        j = i - 1
        while j >= 0 and max_fti[j] < key_fti:
            max_fti[j + 1] = max_fti[j]
            max_idx[j + 1] = max_idx[j]
            j -= 1
        max_fti[j + 1] = key_fti
        max_idx[j + 1] = key_idx

    best_idx = max_idx[0] if max_count > 0 else 0
    return filtered, width, fti_vals, best_idx


# --------------------------------------------------------------------------- #
# Full-series processing (all 4 outputs in one pass)                          #
# --------------------------------------------------------------------------- #

def _fti_process_all(close: np.ndarray, lookback: int, half_length: int, min_period: int, max_period: int) -> tuple:
    """
    Compute all four FTI outputs over the entire series.

    Parameters
    ----------
    close       : 1-D float64 array of close prices.
    lookback    : int  Number of bars in the analysis window.
    half_length : int  Half-length of the FIR filter.
    min_period  : int  Minimum filter period.
    max_period  : int  Maximum filter period.

    Returns
    -------
    out_lowpass     : np.ndarray  exp(filtered log-price) at best period.
    out_best_width  : np.ndarray  Half-width of channel at best period.
    out_best_period : np.ndarray  Best period (integer).
    out_best_fti    : np.ndarray  FTI value at best period, gamma-CDF compressed.
    """
    beta = 0.95
    noise_cut = 0.20

    n = len(close)
    n_periods = max_period - min_period + 1

    # Pre-compute filter coefficients for all periods
    all_coefs = np.zeros((n_periods, half_length + 1), dtype=np.float64)
    for ip in range(n_periods):
        period = min_period + ip
        all_coefs[ip] = _find_coefs(period, half_length)

    front_bad = lookback - 1

    out_lowpass = np.full(n, np.nan)
    out_best_width = np.full(n, np.nan)
    out_best_period = np.full(n, np.nan)
    out_best_fti = np.full(n, np.nan)

    for icase in range(front_bad, n):
        # Build log-price window: chronological order, length = lookback
        window = np.zeros(lookback, dtype=np.float64)
        for j in range(lookback):
            window[j] = math.log(close[icase - lookback + 1 + j])

        filtered, width, fti_vals, best_idx = _fti_process_one_bar(
            window, all_coefs,
            min_period, max_period, half_length, lookback,
            beta, noise_cut,
        )

        k = best_idx  # 0-based index into period arrays

        # Lowpass: exp(filtered log-price)
        out_lowpass[icase] = math.exp(filtered[k])

        # Best period
        out_best_period[icase] = float(min_period + k)

        # Best width: half the channel width in price space
        fval = filtered[k]
        w = width[k]
        out_best_width[icase] = 0.5 * (math.exp(fval + w) - math.exp(fval - w))

        # Best FTI: gamma-CDF compressed
        fti_val = fti_vals[k]
        out_best_fti[icase] = 100.0 * gammainc(2.0, fti_val / 3.0) - 50.0
    return out_lowpass, out_best_width, out_best_period, out_best_fti


# --------------------------------------------------------------------------- #
# 35. FTI LOWPASS  (COMP_VAR.CPP:1548--1595)                                #
# --------------------------------------------------------------------------- #

def fti_lowpass(close: np.ndarray, lookback: int = 60, half_length: int = 40, min_period: int = 8,
                max_period: int = 40) -> np.ndarray:
    """
    Lowpass-filtered price at the FTI-dominant period.

    Applies a bank of symmetric FIR lowpass filters across a range of periods, selects the period with the highest
    Frequency Trend Intensity, and returns the filtered price at that period.

    Parameters
    ----------
    close       : array-like  Close prices.
    lookback    : int  Analysis window length (default 60).
    half_length : int  FIR filter half-length (default 40).
    min_period  : int  Minimum trial period (default 8).
    max_period  : int  Maximum trial period (default 40).

    Returns
    -------
    np.ndarray  Filtered price (in original price units).  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    out_lowpass, _, _, _ = _fti_process_all(close, lookback, half_length, min_period, max_period)
    return out_lowpass


# --------------------------------------------------------------------------- #
# 36. FTI BEST WIDTH  (COMP_VAR.CPP:1548--1595)                             #
# --------------------------------------------------------------------------- #

def fti_best_width(close: np.ndarray, lookback: int = 60, half_length: int = 40, min_period: int = 8,
                   max_period: int = 40) -> np.ndarray:
    """
    Half-width of the lowpass channel at the FTI-dominant period.

    Parameters
    ----------
    close       : array-like  Close prices.
    lookback    : int  Analysis window length (default 60).
    half_length : int  FIR filter half-length (default 40).
    min_period  : int  Minimum trial period (default 8).
    max_period  : int  Maximum trial period (default 40).

    Returns
    -------
    np.ndarray  Channel half-width (in price units).  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    _, out_width, _, _ = _fti_process_all(close, lookback, half_length, min_period, max_period)
    return out_width


# --------------------------------------------------------------------------- #
# 37. FTI BEST PERIOD  (COMP_VAR.CPP:1548--1595)                            #
# --------------------------------------------------------------------------- #

def fti_best_period(close: np.ndarray, lookback: int = 60, half_length: int = 40, min_period: int = 8,
                    max_period: int = 40) -> np.ndarray:
    """
    Dominant period selected by the FTI local-maximum sort.

    Parameters
    ----------
    close       : array-like  Close prices.
    lookback    : int  Analysis window length (default 60).
    half_length : int  FIR filter half-length (default 40).
    min_period  : int  Minimum trial period (default 8).
    max_period  : int  Maximum trial period (default 40).

    Returns
    -------
    np.ndarray  Best period (integer cast to float).  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    _, _, out_period, _ = _fti_process_all(close, lookback, half_length, min_period, max_period)
    return out_period


# --------------------------------------------------------------------------- #
# 38. FTI BEST FTI  (COMP_VAR.CPP:1548--1595)                               #
# --------------------------------------------------------------------------- #

def fti_best_fti(close: np.ndarray, lookback: int = 60, half_length: int = 40, min_period: int = 8,
                 max_period: int = 40) -> np.ndarray:
    """
    FTI value at the dominant period, compressed via the incomplete gamma CDF.

    ``output = 100 * gammainc(2, fti/3) - 50``

    Parameters
    ----------
    close       : array-like  Close prices.
    lookback    : int  Analysis window length (default 60).
    half_length : int  FIR filter half-length (default 40).
    min_period  : int  Minimum trial period (default 8).
    max_period  : int  Maximum trial period (default 40).

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    _, _, _, out_fti = _fti_process_all(close, lookback, half_length, min_period, max_period)
    return out_fti
