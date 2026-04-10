"""
Paired-market PURIFY / LOG_PURIFY indicators #4–5.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.
C++ source files: Paired/COMP_VAR.CPP:242–279, Paired/PURIFY.CPP, Paired/LEGENDRE.CPP.

Indicators
----------
4. purify       SVD-based spread purification (raw prices)
5. log_purify   SVD-based spread purification (log prices)

Algorithm
---------
For each bar, build a small design matrix with up to 3 predictors (Legendre-trend, Legendre-acceleration, volatility)
plus a constant, and solve for the predicted market 1 value via SVD least-squares.
The residual (prediction error) at the current bar is normalised by the RMS error across the lookback window and
compressed through the normal CDF.

Output convention
-----------------
Returns a 1-D ``float64`` numpy array, same length as input.
Warmup bars are ``np.nan``.  Valid range approximately [−50, 50].
"""

import math

import numpy as np
from numba import njit

from ..single._legendre import _legendre_3_kernel


# ---------------------------------------------------------------------------
# Legendre-2: reuse the first two vectors from legendre_3
# ---------------------------------------------------------------------------

def _legendre_2_weights(n: int) -> tuple:
    """Return (w1, w2) Legendre weights for a window of *n* points.

    The C++ ``legendre_2(n, ...)`` uses the same algorithm as ``legendre_3``
    but only the linear (w1) and quadratic (w2) vectors.  We reuse the
    existing ``_legendre_3_kernel`` and discard w3.
    """
    w1, w2, _w3 = _legendre_3_kernel(n)
    return w1, w2


# ---------------------------------------------------------------------------
# Numba kernel: Legendre dot product over a sub-window
# ---------------------------------------------------------------------------

@njit(cache=True)
def _leg_dot(log_prices, weights, start, length):
    """Dot product of log_prices[start : start+length] with weights[0:length]."""
    s = 0.0
    for j in range(length):
        s += log_prices[start + j] * weights[j]
    return s


# ---------------------------------------------------------------------------
# Numba kernel: volatility predictor
# ---------------------------------------------------------------------------

@njit(cache=True)
def _vol_predictor(log_prices, start, vol_length):
    """Mean absolute log-backward-ratio over vol_length bars.

    C++ computes: mean(|log(dptr[i]/dptr[i+1])|) for i in [0, vol_length-2]
    where dptr points into the sub-window of length vol_length ending at
    the current position.  This gives vol_length-1 differences.
    """
    total = 0.0
    for j in range(vol_length - 1):
        total += abs(log_prices[start + j] - log_prices[start + j + 1])
    return total / (vol_length - 1) if vol_length > 1 else 0.0


# ---------------------------------------------------------------------------
# Rolling purify loop (Python — uses np.linalg.lstsq per bar)
# ---------------------------------------------------------------------------

def _purify_loop(close1: np.ndarray, close2: np.ndarray, lookback: int, trend_length: int, accel_length: int,
                 vol_length: int, use_log: bool) -> np.ndarray:
    """Full rolling PURIFY computation.  Returns NaN-padded output."""
    n = len(close1)
    out = np.full(n, np.nan)

    # Pre-compute log prices for predictor (always log) and predicted
    log2 = np.log(close1)  # will be unused if not use_log, but cheap
    log_predictor = np.log(close2)

    # Legendre weights generated separately for each sub-window size.
    # Using lookback-length weights on shorter sub-windows is a basis mismatch —
    # the k-th Legendre polynomial over n points differs from the same polynomial
    # evaluated over m != n points.
    leg1_trend = np.zeros(max(trend_length, 1))
    leg2_accel = np.zeros(max(accel_length, 1))
    if trend_length >= 2:
        leg1_trend, _ = _legendre_2_weights(trend_length)
    if accel_length >= 2:
        _, leg2_accel = _legendre_2_weights(accel_length)

    # Count active predictors
    npred = 0
    if trend_length > 0:
        npred += 1
    if accel_length > 0:
        npred += 1
    if vol_length > 0:
        npred += 1
    ncol = npred + 1  # +1 for constant

    # front_bad: need lookback bars + enough history for the longest predictor
    max_pred_len = max(trend_length, accel_length, vol_length, 0)
    front_bad = lookback + max_pred_len - 1

    for icase in range(front_bad, n):
        # Build design matrix A[lookback, ncol] and target b[lookback]
        A = np.empty((lookback, ncol), dtype=np.float64)
        b = np.empty(lookback, dtype=np.float64)

        for row in range(lookback):
            # ptr = index in the original array for this row
            # row 0 = current bar (icase), row lookback-1 = oldest
            ptr = icase - row

            col = 0

            # Trend predictor: Legendre-1 dot product over trend_length bars
            if trend_length > 0:
                # sub-window of trend_length ending at ptr
                start = ptr - trend_length + 1
                A[row, col] = _leg_dot(log_predictor, leg1_trend, start, trend_length)
                col += 1

            # Acceleration predictor: Legendre-2 dot product over accel_length bars
            if accel_length > 0:
                start = ptr - accel_length + 1
                A[row, col] = _leg_dot(log_predictor, leg2_accel, start, accel_length)
                col += 1

            # Volatility predictor
            if vol_length > 0:
                start = ptr - vol_length + 1
                A[row, col] = _vol_predictor(log_predictor, start, vol_length)
                col += 1

            # Constant column
            A[row, col] = 1.0

            # Target
            if use_log:
                b[row] = log2[ptr]
            else:
                b[row] = close1[ptr]

        # SVD solve with rcond threshold matching C++ backsub(1e-7, ...)
        coefs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=1e-7)

        # Compute predictions for all rows and MSE
        pred = A @ coefs
        mse = 0.0
        for row in range(lookback):
            diff = b[row] - pred[row]
            mse += diff * diff
        mse = math.sqrt(mse / lookback)

        # Current bar residual (row 0)
        current_diff = b[0] - pred[0]

        # Normalise and CDF-compress
        raw = current_diff / (mse + 1e-6)
        # 100 * Phi(0.5 * raw) - 50 = 50 * erf(0.5 * raw / sqrt(2))
        out[icase] = 50.0 * math.erf(0.5 * raw * 0.7071067811865476)

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def purify(close1: np.ndarray, close2: np.ndarray, lookback: int = 60, trend_length: int = 20, accel_length: int = 20,
           vol_length: int = 20) -> np.ndarray:
    """
    PURIFY indicator (#4) — SVD-based spread purification using raw prices.

    Decomposes the relationship between two markets by regressing market 1 on trend, acceleration, and volatility
    components of market 2.  The residual measures how far market 1 deviates from the predicted relationship, normalised
    by RMS error and CDF-compressed.

    Parameters
    ----------
    close1 : array-like  Date-aligned close prices for market 1 (predicted).
    close2 : array-like  Date-aligned close prices for market 2 (predictor).
    lookback      : int  Regression lookback window (default 60).
    trend_length  : int  Legendre linear trend sub-window (default 20, 0 to disable).
    accel_length  : int  Legendre quadratic acceleration sub-window (default 20, 0 to disable).
    vol_length    : int  Volatility sub-window (default 20, 0 to disable).

    Returns
    -------
    np.ndarray  in approximately [−50, 50].  Warmup bars are NaN.
    """
    close1 = np.asarray(close1, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)
    if len(close1) != len(close2):
        raise ValueError(
            f"close1 and close2 must have the same length; "
            f"got {len(close1)} and {len(close2)}"
        )
    return _purify_loop(close1, close2, lookback, trend_length, accel_length, vol_length, use_log=False)


def log_purify(close1: np.ndarray, close2: np.ndarray, lookback: int = 60, trend_length: int = 20,
               accel_length: int = 20, vol_length: int = 20) -> np.ndarray:
    """
    LOG PURIFY indicator (#5) — SVD-based spread purification using log prices.

    Same algorithm as ``purify`` but the target (market 1) is log-transformed, making it more suitable for assets with
    large price level differences.

    Parameters
    ----------
    close1 : array-like  Date-aligned close prices for market 1 (predicted).
    close2 : array-like  Date-aligned close prices for market 2 (predictor).
    lookback      : int  Regression lookback window (default 60).
    trend_length  : int  Legendre linear trend sub-window (default 20, 0 to disable).
    accel_length  : int  Legendre quadratic acceleration sub-window (default 20, 0 to disable).
    vol_length    : int  Volatility sub-window (default 20, 0 to disable).

    Returns
    -------
    np.ndarray  in approximately [−50, 50].  Warmup bars are NaN.
    """
    close1 = np.asarray(close1, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)
    if len(close1) != len(close2):
        raise ValueError(f"close1 and close2 must have the same length; got {len(close1)} and {len(close2)}")
    return _purify_loop(close1, close2, lookback, trend_length, accel_length, vol_length, use_log=True)
