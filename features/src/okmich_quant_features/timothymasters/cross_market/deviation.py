"""
Paired-market deviation indicator #3.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.
C++ source file: Paired/COMP_VAR.CPP:168–240.

Indicator
---------
3. deviation   Rolling OLS spread deviation, RMS-normalised and CDF-compressed.

Algorithm
---------
For each bar i with a lookback window ending at i:

1. Compute means of log(close1) and log(close2) over the window.
2. Fit linear regression of log(close1) on log(close2) via OLS (in deviations from mean — no explicit intercept needed).
3. Walk the window backwards so the *last* residual computed is the one at the current bar.
4. RMS error = sqrt(sum_of_squared_residuals / lookback).
5. current_dev = residual_at_current_bar / RMS_error.
6. factor = lookback^(−1/6).
7. output = 100 * Phi(factor * current_dev) − 50.
8. Optional EMA smoothing (matches C++ exactly — first valid bar is unsmoothed, EMA applied from the second valid bar onward).

Output convention
-----------------
Returns a 1-D ``float64`` numpy array, same length as input.
Warmup bars (0 … lookback−2) are ``np.nan``.
Valid range approximately [−50, 50].
"""

import math

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _deviation_kernel(close1: np.ndarray, close2: np.ndarray, lookback: int) -> np.ndarray:
    """
    Core deviation computation.  Exact port of COMP_VAR.CPP:VAR_DEVIATION.

    The backwards inner loop ensures that after the loop `diff` holds the residual at the *current* bar (k = icase, i = 0), matching the C++:
        for (i = lookback-1; i >= 0; i--) { k = icase - i; ... }
    """
    n = len(close1)
    out = np.full(n, np.nan)

    for icase in range(lookback - 1, n):

        # Pass 1: log-space means
        xmean = 0.0
        ymean = 0.0
        for i in range(lookback):
            k = icase - i
            xmean += math.log(close2[k])
            ymean += math.log(close1[k])
        xmean /= lookback
        ymean /= lookback

        # Pass 2: OLS regression coefficient (no intercept — deviations from mean)
        xss = 0.0
        xy = 0.0
        for i in range(lookback):
            k = icase - i
            xd = math.log(close2[k]) - xmean
            yd = math.log(close1[k]) - ymean
            xss += xd * xd
            xy += xd * yd

        coef = xy / xss if xss > 0.0 else 1.0

        # Pass 3: backwards — residual at current bar + sum of squared residuals
        ss = 0.0
        diff = 0.0
        for i in range(lookback - 1, -1, -1):  # i = lookback-1 … 0
            k = icase - i                       # k goes oldest → newest
            xd = math.log(close2[k]) - xmean
            yd = math.log(close1[k]) - ymean
            diff = yd - coef * xd               # after loop: diff = residual at icase
            ss += diff * diff

        denom = math.sqrt(ss / lookback)

        if denom > 0.0:
            factor = math.exp(-math.log(float(lookback)) / 6.0)   # lookback^(-1/6)
            # 100 * Phi(x) - 50  =  50 * erf(x / sqrt(2))
            out[icase] = 50.0 * math.erf(
                factor * diff / denom * 0.7071067811865476
            )
        else:
            out[icase] = 0.0

    return out


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def deviation(close1: np.ndarray, close2: np.ndarray, period: int = 20, smooth_period: int = 0) -> np.ndarray:
    """
    Paired-market spread deviation.

    Measures how far market 1's current log-price deviates from the linear relationship with market 2, normalised by the
    RMS of the regression fit. This is the core mean-reversion signal for pairs trading.

    Parameters
    ----------
    close1       : array-like  Date-aligned close prices for market 1 (predicted).
    close2       : array-like  Date-aligned close prices for market 2 (predictor).
    period       : int         Lookback window for regression (default 20).
    smooth_period: int         EMA smoothing period (0 or 1 = no smoothing).

    Returns
    -------
    np.ndarray  in approximately [−50, 50].  Warmup bars are NaN.
    """
    close1 = np.asarray(close1, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)
    if len(close1) != len(close2):
        raise ValueError(f"close1 and close2 must have the same length; got {len(close1)} and {len(close2)}")

    out = _deviation_kernel(close1, close2, period)

    # EMA smoothing — exact port of C++ post-processing:
    # smoothed starts at first valid bar (unsmoothed), EMA from second onward.
    if smooth_period > 1:
        alpha = 2.0 / (smooth_period + 1.0)
        front_bad = period - 1
        n = len(out)
        smoothed = out[front_bad]           # first valid bar stays unsmoothed
        for i in range(front_bad + 1, n):
            smoothed = alpha * out[i] + (1.0 - alpha) * smoothed
            out[i] = smoothed
    return out
