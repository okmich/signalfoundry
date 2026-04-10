"""
Information-theoretic indicators #33--34.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.  C++ source files: INFORM.CPP, COMP_VAR.CPP.

Indicators
----------
33. entropy              Normalised word-entropy of up/down sequences
                         COMP_VAR.CPP:1444--1488, INFORM.CPP:entropy()
34. mutual_information   Mutual information between next-bar direction
                         and prior word, COMP_VAR.CPP:1495--1541, INFORM.CPP:mut_inf()
"""

import math

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
# Shared Numba helper                                                          #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ncdf(x: float, k: float) -> float:
    """100 * Phi(k * x) - 50."""
    return 50.0 * math.erf(k * x * 0.7071067811865476)


# --------------------------------------------------------------------------- #
# Numba kernels                                                                #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _entropy_kernel(x_rev: np.ndarray, wordlen: int) -> float:
    """
    Compute normalised entropy of binary up/down words.

    Parameters
    ----------
    x_rev : 1-D float64 array in reverse chronological order
            (x_rev[0] = most recent bar).
    wordlen : int  Word length (number of consecutive up/down bits).

    Returns
    -------
    float  Entropy normalised to [0, 1].
    """
    nx = len(x_rev)
    nb = 1 << wordlen  # 2^wordlen bins

    bins = np.zeros(nb, dtype=np.float64)

    for i in range(wordlen, nx):
        k = 1 if x_rev[i - 1] > x_rev[i] else 0
        for j in range(1, wordlen):
            k *= 2
            if x_rev[i - j - 1] > x_rev[i - j]:
                k += 1
        bins[k] += 1.0

    count = nx - wordlen
    if count <= 0:
        return 0.0

    ent = 0.0
    for i in range(nb):
        p = bins[i] / count
        if p > 0.0:
            ent -= p * math.log(p)

    return ent / math.log(nb)


@njit(cache=True)
def _mutinf_kernel(x_rev: np.ndarray, wordlen: int) -> float:
    """
    Compute mutual information between next-bar direction and prior word.

    Parameters
    ----------
    x_rev : 1-D float64 array in reverse chronological order.
    wordlen : int  Word length.

    Returns
    -------
    float  Mutual information (nats).
    """
    nx = len(x_rev)
    nb = 2 << wordlen  # 2^(wordlen+1)
    n = nx - wordlen - 1
    m = nb // 2

    if n <= 0:
        return 0.0

    bins = np.zeros(nb, dtype=np.float64)
    dep_marg = np.zeros(2, dtype=np.float64)

    for i in range(n):
        k = 1 if x_rev[i] > x_rev[i + 1] else 0
        dep_marg[k] += 1.0
        for j in range(1, wordlen + 1):
            k *= 2
            if x_rev[i + j] > x_rev[i + j + 1]:
                k += 1
        bins[k] += 1.0

    dep_marg[0] /= n
    dep_marg[1] /= n

    mi = 0.0
    for i in range(m):
        hist_marg = (bins[i] + bins[i + m]) / n
        p0 = bins[i] / n
        if p0 > 0.0:
            denom0 = hist_marg * dep_marg[0]
            if denom0 > 0.0:
                mi += p0 * math.log(p0 / denom0)
        p1 = bins[i + m] / n
        if p1 > 0.0:
            denom1 = hist_marg * dep_marg[1]
            if denom1 > 0.0:
                mi += p1 * math.log(p1 / denom1)

    return mi


# --------------------------------------------------------------------------- #
# 33. ENTROPY  (COMP_VAR.CPP:1444--1488)                                     #
# --------------------------------------------------------------------------- #

def entropy(
    close: np.ndarray,
    word_length: int = 3,
    mult: int = 10,
) -> np.ndarray:
    """
    Normalised word-entropy of up/down price sequences.

    Measures how uniformly distributed the binary up/down word patterns are
    over a rolling window.  High entropy means the sequence is close to random;
    low entropy means the sequence has detectable structure.

    Parameters
    ----------
    close       : array-like  Close prices.
    word_length : int  Bit-word length (default 3).
    mult        : int  Multiplier for window size (default 10).
                  Window = 2^word_length * mult + 1 bars.

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    needed = (1 << word_length) * mult + 1
    front_bad = needed - 1

    out = np.full(n, np.nan)

    for icase in range(front_bad, n):
        # Build reverse-chronological window: x_rev[0]=close[icase], x_rev[1]=close[icase-1], ...
        x_rev = close[icase - front_bad: icase + 1][::-1]

        value = _entropy_kernel(x_rev, word_length)

        if word_length == 1:
            value = 1.0 - math.exp(math.log(1.00000001 - value) / 5.0)
            mean = 0.6
        else:
            value = 1.0 - math.exp(math.log(1.0 - value) / word_length)
            mean = 1.0 / word_length + 0.35

        out[icase] = 100.0 * _normal_cdf(8.0 * (value - mean)) - 50.0

    return out


# --------------------------------------------------------------------------- #
# 34. MUTUAL INFORMATION  (COMP_VAR.CPP:1495--1541)                          #
# --------------------------------------------------------------------------- #

def mutual_information(
    close: np.ndarray,
    word_length: int = 3,
    mult: int = 10,
) -> np.ndarray:
    """
    Mutual information between next-bar direction and prior binary word.

    Measures the statistical dependence between the current bar's up/down
    direction and the preceding ``word_length``-bit pattern.  High values
    indicate that prior patterns are predictive of direction.

    Parameters
    ----------
    close       : array-like  Close prices.
    word_length : int  Bit-word length (default 3).
    mult        : int  Multiplier for window size (default 10).
                  Window = 2 * 2^word_length * mult + 1 bars.

    Returns
    -------
    np.ndarray  in approximately [-50, 50].  Warmup bars are NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    needed = (2 << word_length) * mult + 1
    front_bad = needed - 1

    out = np.full(n, np.nan)

    for icase in range(front_bad, n):
        # Build reverse-chronological window
        x_rev = close[icase - front_bad: icase + 1][::-1]

        value = _mutinf_kernel(x_rev, word_length)
        value = value * mult * math.sqrt(word_length) - 0.12 * word_length - 0.04

        out[icase] = 100.0 * _normal_cdf(3.0 * value) - 50.0

    return out


# --------------------------------------------------------------------------- #
# Python-level normal CDF (used in post-processing, not inside Numba)         #
# --------------------------------------------------------------------------- #

def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x * 0.7071067811865476))