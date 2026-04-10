"""
Paired-market trend comparison indicators #6–7.

Source / Attribution
--------------------
Ported from Timothy Masters, "Statistically Sound Indicators For Financial
Market Prediction", Apress, 2013.
C++ source files: Paired/COMP_VAR.CPP:281–307, Paired/TREND_CMMA.CPP.

Indicators
----------
6. trend_diff   Difference of linear Legendre trend between two markets.
7. cmma_diff    Difference of Close-Minus-MA between two markets.

Both reuse the corresponding single-market implementations from
``okmich_quant_features.timothymasters``.  The C++ ``trend()`` and
``cmma()`` helpers in TREND_CMMA.CPP are identical to the single-market
``linear_trend`` and ``close_minus_ma`` implementations.

Output convention
-----------------
Returns a 1-D ``float64`` numpy array, same length as input.
Warmup bars are ``np.nan`` (propagated from the single-market functions).
Valid range approximately [−100, 100].
"""

import numpy as np

from ..single.trend import linear_trend
from ..single.momentum import close_minus_ma


def trend_diff(high1: np.ndarray, low1: np.ndarray, close1: np.ndarray, high2: np.ndarray, low2: np.ndarray,
               close2: np.ndarray, period: int = 20, atr_period: int = 60) -> np.ndarray:
    """
    Difference of linear Legendre trend between two date-aligned markets.

    Identifies when one market's trend is stronger than the other's.
    Positive values indicate market 1 is trending more strongly upward.

    Computed as ``linear_trend(market1) − linear_trend(market2)``.

    Parameters
    ----------
    high1, low1, close1 : array-like  OHLC data for market 1 (predicted).
    high2, low2, close2 : array-like  OHLC data for market 2 (predictor).
    period     : int  Legendre regression lookback (default 20).
    atr_period : int  ATR lookback for normalisation (default 60).

    Returns
    -------
    np.ndarray  in approximately [−100, 100].  Warmup bars are NaN.
    """
    high1 = np.asarray(high1, dtype=np.float64)
    low1 = np.asarray(low1, dtype=np.float64)
    close1 = np.asarray(close1, dtype=np.float64)
    high2 = np.asarray(high2, dtype=np.float64)
    low2 = np.asarray(low2, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)

    n = len(close1)
    if len(high1) != n or len(low1) != n:
        raise ValueError(
            f"high1, low1, close1 must have the same length; "
            f"got {len(high1)}, {len(low1)}, {n}"
        )
    if len(high2) != len(close2) or len(low2) != len(close2):
        raise ValueError(
            f"high2, low2, close2 must have the same length; "
            f"got {len(high2)}, {len(low2)}, {len(close2)}"
        )
    if len(close2) != n:
        raise ValueError(
            f"market-1 and market-2 arrays must have the same length; "
            f"got {n} and {len(close2)}"
        )

    t1 = linear_trend(high1, low1, close1, period=period, atr_period=atr_period)
    t2 = linear_trend(high2, low2, close2, period=period, atr_period=atr_period)
    return t1 - t2


def cmma_diff(high1: np.ndarray, low1: np.ndarray, close1: np.ndarray, high2: np.ndarray, low2: np.ndarray,
              close2: np.ndarray, period: int = 20, atr_period: int = 60) -> np.ndarray:
    """
    Difference of Close-Minus-MA between two date-aligned markets.

    Compares how far each market's close is from its own moving average, normalised by ATR.  Positive values indicate
    market 1 is further above its MA than market 2.

    Computed as ``close_minus_ma(market1) − close_minus_ma(market2)``.

    Parameters
    ----------
    high1, low1, close1 : array-like  OHLC data for market 1 (predicted).
    high2, low2, close2 : array-like  OHLC data for market 2 (predictor).
    period     : int  MA lookback (default 20).
    atr_period : int  ATR lookback for normalisation (default 60).

    Returns
    -------
    np.ndarray  in approximately [−100, 100].  Warmup bars are NaN.
    """
    high1 = np.asarray(high1, dtype=np.float64)
    low1 = np.asarray(low1, dtype=np.float64)
    close1 = np.asarray(close1, dtype=np.float64)
    high2 = np.asarray(high2, dtype=np.float64)
    low2 = np.asarray(low2, dtype=np.float64)
    close2 = np.asarray(close2, dtype=np.float64)

    n = len(close1)
    if len(high1) != n or len(low1) != n:
        raise ValueError(
            f"high1, low1, close1 must have the same length; "
            f"got {len(high1)}, {len(low1)}, {n}"
        )
    if len(high2) != len(close2) or len(low2) != len(close2):
        raise ValueError(
            f"high2, low2, close2 must have the same length; "
            f"got {len(high2)}, {len(low2)}, {len(close2)}"
        )
    if len(close2) != n:
        raise ValueError(
            f"market-1 and market-2 arrays must have the same length; "
            f"got {n} and {len(close2)}"
        )

    c1 = close_minus_ma(high1, low1, close1, period=period, atr_period=atr_period)
    c2 = close_minus_ma(high2, low2, close2, period=period, atr_period=atr_period)
    return c1 - c2
