"""
Multi-market risk / dimensionality indicators #11–15.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.
C++ source files: Multi/COMP_VAR.CPP (MAHAL, ABS_RATIO, ABS_SHIFT, COHERENCE).

Indicators
----------
11. mahal            Mahalanobis distance (log-odds of F-statistic)
12. abs_ratio        Fraction of variance in top-k eigenvalues (ABS RATIO)
13. abs_shift        Z-score shift in abs_ratio short vs long window
14. coherence        Eigenvalue-weighted correlation coherence
15. delta_coherence  Change in coherence over delta_length bars

Input convention
----------------
``closes`` is a list of N date-aligned close arrays of equal length.
Warmup bars are ``np.nan``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_closes(closes: list[np.ndarray], min_markets: int = 2) -> tuple[int, int]:
    """Validate closes list; return (n_bars, n_markets)."""
    n_markets = len(closes)
    if n_markets < min_markets:
        raise ValueError(
            f"At least {min_markets} markets required; got {n_markets}."
        )
    n_bars = len(closes[0])
    for i, c in enumerate(closes):
        if len(c) != n_bars:
            raise ValueError(
                f"All close arrays must have the same length; market {i} has length {len(c)}, expected {n_bars}."
            )
    return n_bars, n_markets


def _rolling_log_returns(closes_arr: np.ndarray, icase: int, lookback: int) -> np.ndarray:
    """
    Compute log returns for a window ending BEFORE icase (excludes current bar).

    Returns (lookback-1, n_markets) float64 array.
    """
    window = closes_arr[icase - lookback: icase]  # shape (lookback, n_markets)
    return np.diff(np.log(window), axis=0)         # (lookback-1, n_markets)


def _rolling_log_returns_incl(closes_arr: np.ndarray, icase: int, lookback: int) -> np.ndarray:
    """
    Compute log returns for a window ending AT icase (includes current bar).

    Returns (lookback-1, n_markets) float64 array.
    """
    window = closes_arr[icase - lookback + 1: icase + 1]  # shape (lookback, n_markets)
    return np.diff(np.log(window), axis=0)                 # (lookback-1, n_markets)


# ---------------------------------------------------------------------------
# 11. MAHAL
# ---------------------------------------------------------------------------

def mahal(closes: list[np.ndarray], lookback: int = 120, smoothing: int = 0) -> np.ndarray:
    """
    Mahalanobis distance expressed as log-odds of the F-statistic CDF.

    Measures how unusual the current bar's log return vector is relative to the recent multivariate distribution.
    High positive values indicate anomalous / extreme joint moves.

    Parameters
    ----------
    closes : list of np.ndarray
        Date-aligned close arrays for N ≥ 2 markets.
    lookback : int
        Rolling window size (default 120).  Current bar is excluded.
    smoothing : int
        EMA smoothing period (0 = no smoothing, default 0).

    Returns
    -------
    np.ndarray, float64.  First ``lookback`` bars are NaN.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if smoothing < 0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}")
    n_bars, n_markets = _validate_closes(closes)
    closes_arr = np.column_stack([np.asarray(c, dtype=np.float64) for c in closes])

    result = np.full(n_bars, np.nan, dtype=np.float64)
    alpha = 2.0 / (smoothing + 1) if smoothing > 0 else 0.0
    ema_val = np.nan

    for icase in range(lookback, n_bars):
        # Log returns in window BEFORE current bar (excludes icase)
        ret = _rolling_log_returns(closes_arr, icase, lookback)  # (lookback-1, n_markets)
        L = lookback
        M = n_markets

        if L - 1 - M <= 0:
            # Not enough degrees of freedom
            result[icase] = 0.0
            continue

        mean_ret = ret.mean(axis=0)                    # (n_markets,)
        cov = np.cov(ret, rowvar=False)                # (n_markets, n_markets)

        # Current bar's log return
        cur_log_ret = np.log(closes_arr[icase]) - np.log(closes_arr[icase - 1])
        diff = cur_log_ret - mean_ret                  # (n_markets,)

        try:
            inv_diff = np.linalg.solve(cov, diff)
        except np.linalg.LinAlgError:
            result[icase] = 0.0
            continue

        d2 = float(diff @ inv_diff)

        # F-statistic (Hotelling T² → F)
        f_stat = d2 * (L - 1) * (L - 1 - M) / (M * (L - 2) * L)
        if f_stat < 0:
            f_stat = 0.0

        p = float(f_dist.cdf(f_stat, M, L - 1 - M))
        p = max(0.5, min(p, 0.99999))
        raw = float(np.log(p / (1.0 - p)))

        if smoothing > 0:
            if np.isnan(ema_val):
                ema_val = raw
            else:
                ema_val = alpha * raw + (1.0 - alpha) * ema_val
            result[icase] = ema_val
        else:
            result[icase] = raw

    return result


# ---------------------------------------------------------------------------
# 12. ABS RATIO
# ---------------------------------------------------------------------------

def abs_ratio(closes: list[np.ndarray], lookback: int = 120, fraction: float = 0.2) -> np.ndarray:
    """
    Fraction of total variance captured by the top-k eigenvalues (ABS RATIO).

    High values indicate that markets are moving together (systematic risk).
    Low values indicate diversification.

    Parameters
    ----------
    closes : list of np.ndarray
        Date-aligned close arrays for N ≥ 2 markets.
    lookback : int
        Rolling covariance window size (includes current bar, default 120).
    fraction : float
        Fraction of markets to include in the numerator (default 0.2).

    Returns
    -------
    np.ndarray, float64, range [0, 100].  First ``lookback - 1`` bars are NaN.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n_bars, n_markets = _validate_closes(closes)
    closes_arr = np.column_stack([np.asarray(c, dtype=np.float64) for c in closes])

    k = max(1, int(fraction * n_markets + 0.5))
    alpha = 2.0 / (lookback / 2.0 + 1.0)

    result = np.full(n_bars, np.nan, dtype=np.float64)
    smoothed_numer = np.nan
    smoothed_denom = np.nan

    for icase in range(lookback - 1, n_bars):
        ret = _rolling_log_returns_incl(closes_arr, icase, lookback)  # (lookback-1, n_markets)
        cov = np.cov(ret, rowvar=False)                                # (n_markets, n_markets)

        eigenvalues = np.linalg.eigh(cov)[0]
        eigenvalues = eigenvalues[::-1]  # descending order
        eigenvalues = np.maximum(eigenvalues, 0.0)  # numerical safety

        total_sum = float(np.sum(eigenvalues))
        top_k_sum = float(np.sum(eigenvalues[:k]))

        if np.isnan(smoothed_numer):
            smoothed_numer = top_k_sum
            smoothed_denom = total_sum
        else:
            smoothed_numer = alpha * top_k_sum + (1.0 - alpha) * smoothed_numer
            smoothed_denom = alpha * total_sum + (1.0 - alpha) * smoothed_denom

        result[icase] = 100.0 * smoothed_numer / (smoothed_denom + 1e-30)

    return result


# ---------------------------------------------------------------------------
# 13. ABS SHIFT
# ---------------------------------------------------------------------------

def abs_shift(closes: list[np.ndarray], lookback: int = 120, fraction: float = 0.2, long_lookback: int = 60,
              short_lookback: int = 10) -> np.ndarray:
    """
    Z-score shift: how many standard deviations the short-term mean of abs_ratio has moved from its long-term mean.

    Parameters
    ----------
    closes : list of np.ndarray
        Date-aligned close arrays for N ≥ 2 markets.
    lookback : int
        abs_ratio rolling window (default 120).
    fraction : float
        abs_ratio top-k fraction (default 0.2).
    long_lookback : int
        Long rolling window for mean/variance (default 60).
    short_lookback : int
        Short rolling window for mean (default 10).

    Returns
    -------
    np.ndarray, float64.  Warmup bars are NaN.
    """
    if long_lookback < short_lookback + 1:
        long_lookback = short_lookback + 1

    base = abs_ratio(closes, lookback=lookback, fraction=fraction)

    series = pd.Series(base)
    long_mean = series.rolling(long_lookback).mean()
    long_var = series.rolling(long_lookback).var()
    short_mean = series.rolling(short_lookback).mean()

    denom = np.sqrt(long_var.to_numpy(dtype=np.float64))
    numerator = short_mean.to_numpy(dtype=np.float64) - long_mean.to_numpy(dtype=np.float64)

    result = np.full(len(base), np.nan, dtype=np.float64)
    valid = denom > 1e-30
    result[valid] = numerator[valid] / denom[valid]
    return result


# ---------------------------------------------------------------------------
# 14. COHERENCE
# ---------------------------------------------------------------------------

def coherence(closes: list[np.ndarray], lookback: int = 120) -> np.ndarray:
    """
    Eigenvalue-weighted correlation coherence.

    Measures how structured (non-random) the correlation matrix is.
    High positive → markets strongly co-move.
    Near zero → markets roughly independent.
    High negative → markets strongly diverge (unusual).

    Range: approximately [-100, 100].

    Parameters
    ----------
    closes : list of np.ndarray
        Date-aligned close arrays for N ≥ 2 markets.
    lookback : int
        Rolling covariance window (includes current bar, default 120).

    Returns
    -------
    np.ndarray, float64, range [-100, 100].  First ``lookback - 1`` bars are NaN.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    n_bars, n_markets = _validate_closes(closes)
    closes_arr = np.column_stack([np.asarray(c, dtype=np.float64) for c in closes])

    result = np.full(n_bars, np.nan, dtype=np.float64)
    factor = 0.5 * (n_markets - 1) if n_markets > 1 else 1.0

    for icase in range(lookback - 1, n_bars):
        ret = _rolling_log_returns_incl(closes_arr, icase, lookback)  # (lookback-1, n_markets)
        cov = np.cov(ret, rowvar=False)

        # Normalise to correlation matrix
        stds = np.sqrt(np.diag(cov))
        outer_stds = np.outer(stds, stds)
        outer_stds = np.where(outer_stds < 1e-30, 1.0, outer_stds)
        corr = cov / outer_stds
        np.fill_diagonal(corr, 1.0)

        eigenvalues = np.linalg.eigh(corr)[0]
        eigenvalues = eigenvalues[::-1]  # descending order

        total = float(np.sum(eigenvalues))
        if total < 1e-30:
            result[icase] = 0.0
            continue

        weighted = sum(
            (factor - i) * float(eigenvalues[i]) / factor
            for i in range(n_markets)
        )
        result[icase] = 200.0 * (weighted / total - 0.5)

    return result


# ---------------------------------------------------------------------------
# 15. DELTA COHERENCE
# ---------------------------------------------------------------------------

def delta_coherence(closes: list[np.ndarray], lookback: int = 120, delta_length: int = 20) -> np.ndarray:
    """
    Change in coherence over ``delta_length`` bars.

    Parameters
    ----------
    closes : list of np.ndarray
        Date-aligned close arrays for N ≥ 2 markets.
    lookback : int
        Coherence lookback (default 120).
    delta_length : int
        Lag for differencing coherence (default 20).

    Returns
    -------
    np.ndarray, float64.  First ``lookback - 1 + delta_length`` bars are NaN.
    """
    if delta_length < 1:
        raise ValueError(f"delta_length must be >= 1, got {delta_length}")
    coh = coherence(closes, lookback=lookback)
    result = np.full(len(coh), np.nan, dtype=np.float64)
    for i in range(delta_length, len(coh)):
        if not np.isnan(coh[i]) and not np.isnan(coh[i - delta_length]):
            result[i] = coh[i] - coh[i - delta_length]
    return result
