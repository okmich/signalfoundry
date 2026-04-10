"""
Paired-market correlation indicators #1–2.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.
C++ source files: Paired/COMP_VAR.CPP:134–164, Paired/SPEARMAN.CPP.

Indicators
----------
1. correlation        Rolling Spearman rank correlation × 50
2. delta_correlation  Change in rolling Spearman correlation over a lag period

Output convention
-----------------
All public functions return a 1-D ``float64`` numpy array of the same length
as the input arrays.  Warmup bars are ``np.nan``.

``correlation`` range: approximately [−50, 50].
``delta_correlation`` range: approximately [−100, 100].
"""

import math

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Numba kernel: Spearman rank correlation (exact port of SPEARMAN.CPP)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman rank correlation with tie correction.

    Exact port of SPEARMAN.CPP.  Uses the rank-based formula:
        rho = 0.5 * (SSx + SSy - sum_rank_diff²) / sqrt(SSx * SSy)
    where SSx, SSy include the standard tie-correction term.

    Parameters
    ----------
    x, y : 1-D float64 arrays, same length (the lookback window).

    Returns
    -------
    float  rho in [−1, 1].
    """
    n = len(x)
    dn = float(n)

    # --- rank x with tie correction ---
    rx = np.empty(n, dtype=np.float64)
    idx = np.argsort(x)
    x_tie_corr = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[idx[j]] == x[idx[i]]:
            j += 1
        ntied = float(j - i)
        x_tie_corr += ntied * ntied * ntied - ntied
        mid_rank = 0.5 * (float(i) + float(j) + 1.0)
        for k in range(i, j):
            rx[idx[k]] = mid_rank
        i = j

    # --- rank y with tie correction ---
    ry = np.empty(n, dtype=np.float64)
    idx = np.argsort(y)
    y_tie_corr = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and y[idx[j]] == y[idx[i]]:
            j += 1
        ntied = float(j - i)
        y_tie_corr += ntied * ntied * ntied - ntied
        mid_rank = 0.5 * (float(i) + float(j) + 1.0)
        for k in range(i, j):
            ry[idx[k]] = mid_rank
        i = j

    # --- Spearman formula ---
    ssx = (dn * dn * dn - dn - x_tie_corr) / 12.0
    ssy = (dn * dn * dn - dn - y_tie_corr) / 12.0
    rankerr = 0.0
    for i in range(n):
        diff = rx[i] - ry[i]
        rankerr += diff * diff

    return 0.5 * (ssx + ssy - rankerr) / math.sqrt(ssx * ssy + 1e-20)


# ---------------------------------------------------------------------------
# Numba kernel: rolling correlation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _correlation_kernel(close1: np.ndarray, close2: np.ndarray, lookback: int) -> np.ndarray:
    """
    Rolling Spearman correlation × 50.

    C++ note: x = close2 (predictor), y = close1 (predicted).
    Spearman is symmetric so the order does not affect the result.
    front_bad = lookback - 1.
    """
    n = len(close1)
    out = np.full(n, np.nan)
    for icase in range(lookback - 1, n):
        x = close2[icase - lookback + 1: icase + 1]
        y = close1[icase - lookback + 1: icase + 1]
        out[icase] = 50.0 * _spearman_rho(x, y)
    return out


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def correlation(close1: np.ndarray, close2: np.ndarray, period: int = 63) -> np.ndarray:
    """
    Rolling Spearman rank correlation between two date-aligned markets.

    Uses rank-based (Spearman) rather than Pearson correlation for robustness to outliers and non-linear price relationships.
    Output is scaled to [−50, 50].

    Parameters
    ----------
    close1 : array-like  Close prices for market 1 (predicted).
    close2 : array-like  Close prices for market 2 (predictor / reference).
    period : int         Lookback window (default 63).

    Returns
    -------
    np.ndarray  in [−50, 50].  Warmup bars are NaN.
    """
    close1 = np.asarray(close1, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)
    if len(close1) != len(close2):
        raise ValueError(
            f"close1 and close2 must have the same length; "
            f"got {len(close1)} and {len(close2)}"
        )
    return _correlation_kernel(close1, close2, period)


def delta_correlation(close1: np.ndarray, close2: np.ndarray, period: int = 63, delta_period: int = 63) -> np.ndarray:
    """
    Change in rolling Spearman correlation over ``delta_period`` bars.

    Detects when the relationship between two markets is strengthening or weakening.
    Computed as ``correlation[i] − correlation[i − delta_period]``.

    Parameters
    ----------
    close1, close2 : array-like  Date-aligned close prices.
    period         : int  Correlation lookback (default 63).
    delta_period   : int  Differencing lag (default 63).

    Returns
    -------
    np.ndarray  Warmup bars are NaN.
    """
    close1 = np.asarray(close1, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)
    if len(close1) != len(close2):
        raise ValueError(f"close1 and close2 must have the same length; got {len(close1)} and {len(close2)}")

    corr = _correlation_kernel(close1, close2, period)

    n = len(corr)
    out = np.full(n, np.nan)
    front_bad = period - 1 + delta_period
    for i in range(front_bad, n):
        if not math.isnan(corr[i]) and not math.isnan(corr[i - delta_period]):
            out[i] = corr[i] - corr[i - delta_period]
    return out
