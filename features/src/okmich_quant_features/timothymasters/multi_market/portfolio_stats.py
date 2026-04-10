"""
Multi-market portfolio statistical indicators #1–10.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.
C++ source files: Multi/COMP_VAR.CPP, Multi/TREND_CMMA.CPP.

Indicators
----------
1.  trend_rank    Rank of target market linear_trend among all markets
2.  trend_median  Median of linear_trend across all markets
3.  trend_range   Max − min of linear_trend across all markets
4.  trend_iqr     IQR (P75 − P25) of linear_trend across all markets
5.  trend_clump   40th/60th fractile clump measure of linear_trend
6.  cmma_rank     Rank of target market close_minus_ma among all markets
7.  cmma_median   Median of close_minus_ma across all markets
8.  cmma_range    Max − min of close_minus_ma across all markets
9.  cmma_iqr      IQR (P75 − P25) of close_minus_ma across all markets
10. cmma_clump    40th/60th fractile clump measure of close_minus_ma

Input convention
----------------
All lists must contain N date-aligned arrays of equal length.
``closes[0]``, ``highs[0]``, ``lows[0]`` are the **target market**.
Warmup bars are ``np.nan``.
"""

from __future__ import annotations

import numpy as np

from ..single.trend import linear_trend
from ..single.momentum import close_minus_ma


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_markets(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray]) -> int:
    """Validate market lists and return n_markets."""
    n_markets = len(closes)
    if n_markets < 2:
        raise ValueError(
            f"At least 2 markets required; got {n_markets}."
        )
    if len(highs) != n_markets or len(lows) != n_markets:
        raise ValueError(
            "highs, lows, and closes must all have the same number of markets."
        )
    n_bars = len(closes[0])
    for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
        if len(h) != n_bars or len(l) != n_bars or len(c) != n_bars:
            raise ValueError(
                f"All arrays must have the same length; market {i} has inconsistent lengths."
            )
    return n_markets


def _compute_base_matrix(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], fn, period: int,
                         atr_period: int) -> np.ndarray:
    """Compute fn for each market and stack into (n_bars, n_markets) matrix."""
    n_markets = len(closes)
    n_bars = len(closes[0])
    matrix = np.full((n_bars, n_markets), np.nan, dtype=np.float64)
    for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
        matrix[:, i] = fn(h, l, c, period=period, atr_period=atr_period)
    return matrix


def _clump(row: np.ndarray) -> float:
    """
    40th/60th fractile clump statistic (C++ reference).

    Returns the 40th-percentile value if > 0, the 60th-percentile value if < 0,
    or 0 otherwise.
    """
    n = len(row)
    k = max(0, int(0.4 * (n + 1)) - 1)
    m = n - k - 1
    sorted_row = np.sort(row)
    if sorted_row[k] > 0.0:
        return sorted_row[k]
    elif sorted_row[m] < 0.0:
        return sorted_row[m]
    else:
        return 0.0


def _apply_row_stat(matrix: np.ndarray, stat_fn) -> np.ndarray:
    """Apply stat_fn to each row, respecting NaN warmup."""
    n_bars = matrix.shape[0]
    result = np.full(n_bars, np.nan, dtype=np.float64)
    for i in range(n_bars):
        row = matrix[i]
        if np.all(np.isnan(row)):
            continue
        if np.any(np.isnan(row)):
            # partial NaN — skip row (shouldn't happen with aligned warmup)
            continue
        result[i] = stat_fn(row)
    return result


def _rank_stat(row: np.ndarray) -> float:
    """Rank of element [0] among all elements, scaled to [-50, 50]."""
    n = len(row)
    target = row[0]
    count_le = np.sum(row <= target)
    return 100.0 * (count_le - 1) / (n - 1) - 50.0


# ---------------------------------------------------------------------------
# Public functions — TREND variants
# ---------------------------------------------------------------------------

def trend_rank(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
               atr_period: int = 60) -> np.ndarray:
    """
    Rank of the target market's linear_trend among all markets.

    Positive values → target market is trending more strongly upward than most.
    Negative values → target market is lagging.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets. Index 0 is the target market.
    period : int
        Legendre regression lookback (default 20).
    atr_period : int
        ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64, range approximately [-50, 50].  Warmup bars are NaN.
    """
    n_markets = _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, linear_trend, period, atr_period)
    return _apply_row_stat(matrix, _rank_stat)


def trend_median(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
                 atr_period: int = 60) -> np.ndarray:
    """
    Median of linear_trend across all markets.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  Legendre regression lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, linear_trend, period, atr_period)
    return _apply_row_stat(matrix, np.median)


def trend_range(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
                atr_period: int = 60) -> np.ndarray:
    """
    Range (max − min) of linear_trend across all markets.

    A large range indicates diverging market trends.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  Legendre regression lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64, non-negative.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, linear_trend, period, atr_period)
    return _apply_row_stat(matrix, lambda row: float(np.max(row) - np.min(row)))


def trend_iqr(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
              atr_period: int = 60) -> np.ndarray:
    """
    Interquartile range (P75 − P25) of linear_trend across all markets.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  Legendre regression lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64, non-negative.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, linear_trend, period, atr_period)
    return _apply_row_stat(
        matrix,
        lambda row: float(np.percentile(row, 75) - np.percentile(row, 25)),
    )


def trend_clump(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
                atr_period: int = 60) -> np.ndarray:
    """
    40th/60th fractile clump measure of linear_trend across all markets.

    Returns the 40th-percentile value if it is positive (most markets trending up), the 60th-percentile if it is negative
    (most markets trending down), or 0 when markets are mixed.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  Legendre regression lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, linear_trend, period, atr_period)
    return _apply_row_stat(matrix, _clump)


# ---------------------------------------------------------------------------
# Public functions — CMMA variants
# ---------------------------------------------------------------------------

def cmma_rank(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
              atr_period: int = 60) -> np.ndarray:
    """
    Rank of the target market's close_minus_ma among all markets.

    Positive → target market's close is above its MA more than most peers.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets. Index 0 is the target market.
    period : int  MA lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64, range approximately [-50, 50].  Warmup bars are NaN.
    """
    n_markets = _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, close_minus_ma, period, atr_period)
    return _apply_row_stat(matrix, _rank_stat)


def cmma_median(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
                atr_period: int = 60) -> np.ndarray:
    """
    Median of close_minus_ma across all markets.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  MA lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, close_minus_ma, period, atr_period)
    return _apply_row_stat(matrix, np.median)


def cmma_range(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
               atr_period: int = 60) -> np.ndarray:
    """
    Range (max − min) of close_minus_ma across all markets.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  MA lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64, non-negative.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, close_minus_ma, period, atr_period)
    return _apply_row_stat(matrix, lambda row: float(np.max(row) - np.min(row)))


def cmma_iqr(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
             atr_period: int = 60) -> np.ndarray:
    """
    Interquartile range (P75 − P25) of close_minus_ma across all markets.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  MA lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64, non-negative.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, close_minus_ma, period, atr_period)
    return _apply_row_stat(matrix, lambda row: float(np.percentile(row, 75) - np.percentile(row, 25)))


def cmma_clump(highs: list[np.ndarray], lows: list[np.ndarray], closes: list[np.ndarray], period: int = 20,
               atr_period: int = 60) -> np.ndarray:
    """
    40th/60th fractile clump measure of close_minus_ma across all markets.

    Parameters
    ----------
    highs, lows, closes : list of np.ndarray
        Date-aligned OHLC arrays for N markets.
    period : int  MA lookback (default 20).
    atr_period : int  ATR normalisation lookback (default 60).

    Returns
    -------
    np.ndarray, float64.  Warmup bars are NaN.
    """
    _validate_markets(highs, lows, closes)
    highs = [np.asarray(h, dtype=np.float64) for h in highs]
    lows = [np.asarray(l, dtype=np.float64) for l in lows]
    closes = [np.asarray(c, dtype=np.float64) for c in closes]
    matrix = _compute_base_matrix(highs, lows, closes, close_minus_ma, period, atr_period)
    return _apply_row_stat(matrix, _clump)
