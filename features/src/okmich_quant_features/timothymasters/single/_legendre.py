"""
Legendre polynomial weight generation for Timothy Masters' trend indicators.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source file: LEGENDRE.CPP.

The ``legendre_3`` function generates three sets of orthogonal polynomial weights (linear, quadratic, cubic) over a window
of ``n`` points.  These weights are used as dot-product kernels by the six trend/deviation indicators:

    LINEAR TREND, QUADRATIC TREND, CUBIC TREND  (COMP_VAR.CPP:553–621)
    LINEAR DEVIATION, QUADRATIC DEVIATION, CUBIC DEVIATION  (COMP_VAR.CPP:889–964)

Design notes
------------
Masters uses *discrete* Legendre polynomials, not the classical continuous ones.  The weights are normalised so that
their sum-of-squares equals one, enabling direct comparison of regression coefficients across window sizes.

The Numba kernel ``_legendre_3_kernel`` pre-computes all three weight vectors for a given window length.
Results are cached by the Python wrapper ``legendre_weights`` so each unique window length is compiled and stored only once.
"""

from functools import lru_cache

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
# Numba kernel                                                                #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _legendre_3_kernel(n: int) -> tuple:
    """
    Compute discrete orthonormal Legendre weights for a window of ``n`` points.

    Replicates Masters' ``legendre_3(n, w1, w2, w3)`` function.

    The three weight vectors correspond to the linear (w1), quadratic (w2), and cubic (w3) Legendre polynomials evaluated
    at equally-spaced points {0, 1, ..., n-1} and then orthonormalised via Gram-Schmidt.

    Parameters
    ----------
    n : int  Window length (must be >= 2).

    Returns
    -------
    (w1, w2, w3) : three float64 arrays of length n
    """
    w1 = np.empty(n, dtype=np.float64)
    w2 = np.empty(n, dtype=np.float64)
    w3 = np.empty(n, dtype=np.float64)

    # Normalised x in [-1, +1]
    # Masters maps index k -> x[k] = (2k - (n-1)) / (n-1)  (for n > 1)
    denom = n - 1.0 if n > 1 else 1.0

    for k in range(n):
        x = (2.0 * k - (n - 1.0)) / denom

        # Raw polynomial values
        p1 = x
        p2 = 0.5 * (3.0 * x * x - 1.0)
        p3 = 0.5 * (5.0 * x * x * x - 3.0 * x)

        w1[k] = p1
        w2[k] = p2
        w3[k] = p3

    # Orthonormalise: subtract projections and normalise each vector.
    # Full Gram-Schmidt includes orthogonalisation against the constant vector
    # {1, 1, ..., 1} / sqrt(n) so that polynomial coefficients measure
    # deviations from the mean rather than absolute price level.
    #
    # w1 and w3 are odd functions over the symmetric grid, so their sum is
    # already zero.  w2 (even, P2 basis) has a non-zero sum for finite n and
    # must be centred explicitly before the remaining steps.

    # --- w2: remove DC component (project out the constant vector) ---
    sum2 = 0.0
    for k in range(n):
        sum2 += w2[k]
    mean2 = sum2 / n
    for k in range(n):
        w2[k] -= mean2

    # --- w2: subtract projection onto w1 ---
    dot12 = 0.0
    dot11 = 0.0
    for k in range(n):
        dot12 += w1[k] * w2[k]
        dot11 += w1[k] * w1[k]
    if dot11 > 0.0:
        proj = dot12 / dot11
        for k in range(n):
            w2[k] -= proj * w1[k]

    # --- w3: subtract projections onto w1 and w2 ---
    dot13 = 0.0
    dot23 = 0.0
    dot22 = 0.0
    for k in range(n):
        dot13 += w1[k] * w3[k]
        dot23 += w2[k] * w3[k]
        dot22 += w2[k] * w2[k]
    if dot11 > 0.0:
        proj1 = dot13 / dot11
        for k in range(n):
            w3[k] -= proj1 * w1[k]
    if dot22 > 0.0:
        proj2 = dot23 / dot22
        for k in range(n):
            w3[k] -= proj2 * w2[k]

    # --- Normalise each vector to unit length ---
    ss1 = 0.0
    ss2 = 0.0
    ss3 = 0.0
    for k in range(n):
        ss1 += w1[k] * w1[k]
        ss2 += w2[k] * w2[k]
        ss3 += w3[k] * w3[k]

    norm1 = ss1 ** 0.5 if ss1 > 0.0 else 1.0
    norm2 = ss2 ** 0.5 if ss2 > 0.0 else 1.0
    norm3 = ss3 ** 0.5 if ss3 > 0.0 else 1.0

    for k in range(n):
        w1[k] /= norm1
        w2[k] /= norm2
        w3[k] /= norm3

    return w1, w2, w3


# --------------------------------------------------------------------------- #
# Python wrapper with LRU cache                                                #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def legendre_weights(n: int) -> tuple:
    """
    Return the three orthonormal Legendre weight vectors for a window of ``n`` points.
    Results are cached — each unique window length is computed only once.

    Parameters
    ----------
    n : int  Window length.  Must be >= 2 for w1, >= 3 for w2, >= 4 for w3.

    Returns
    -------
    (w1, w2, w3) : three read-only numpy float64 arrays of length n

    Notes
    -----
    The returned arrays are **read-only** (``flags.writeable = False``) so they are safe to share across calls.  Callers
    that need a mutable copy should call ``.copy()``.
    """
    if n < 2:
        raise ValueError(f"legendre_weights requires n >= 2, got {n}.")

    w1, w2, w3 = _legendre_3_kernel(n)
    # Make immutable so the cache is safe
    for arr in (w1, w2, w3):
        arr.flags.writeable = False

    return w1, w2, w3


# --------------------------------------------------------------------------- #
# Vectorised dot-product helper                                                #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def legendre_dot(log_prices: np.ndarray, weights: np.ndarray, i: int, n: int) -> float:
    """
    Dot product of a rolling window of log-prices with a Legendre weight vector.

    Parameters
    ----------
    log_prices : 1-D float64 array  (full series, already log-transformed)
    weights    : 1-D float64 array  (length n, from ``legendre_weights``)
    i          : int  Current bar index (end of window, inclusive)
    n          : int  Window length

    Returns
    -------
    float  Dot product = Σ log_prices[i-n+1+k] * weights[k]
    """
    result = 0.0
    start = i - n + 1
    for k in range(n):
        result += log_prices[start + k] * weights[k]
    return result
