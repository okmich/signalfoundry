"""A synthetic OHLCV frame that exercises the failure modes a smooth random walk hides.

Look-ahead bugs live in the edge cases: doji bars (|C-O| = 0) trip division guards and caps, tied volumes trip
rank/quantile binning, session gaps trip resampling and forward fills. So prices sit on a tick grid with a
realistic share of dojis, volumes are integers with many ties, spreads are integer points, and the index has a
weekend gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(n: int = 2400, seed: int = 7, tick: float = 0.0001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.round(rng.standard_t(4, n) * 3.0) * tick                     # fat tails, many zero steps
    close = 1.1 + np.cumsum(steps)
    open_ = np.r_[close[0], close[:-1]]
    wick_hi = np.round(rng.exponential(2.0, n)) * tick
    wick_lo = np.round(rng.exponential(2.0, n)) * tick
    high = np.maximum(open_, close) + wick_hi
    low = np.minimum(open_, close) - wick_lo
    tick_volume = rng.integers(20, 300, n).astype(float)
    tick_volume[rng.random(n) < 0.02] *= 8                                   # volume bursts
    spread = rng.integers(0, 12, n).astype(float)
    start = pd.Timestamp("2024-01-01 00:00")                                 # a Monday
    first_week = pd.date_range(start, periods=n // 2, freq="5min")
    second = pd.date_range(first_week[-1] + pd.Timedelta(days=2), periods=n - n // 2, freq="5min")
    idx = first_week.append(second)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "tick_volume": tick_volume,
                         "volume": tick_volume, "spread": spread}, index=idx)
