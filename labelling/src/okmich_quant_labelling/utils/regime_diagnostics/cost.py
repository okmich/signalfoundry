"""Broker-agnostic cost helper: turn any per-bar spread series into a seasonal round-trip cost profile."""
from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from .clock import build_clock


class CostBucket(StrEnum):
    HOUR_OF_DAY = "hour_of_day"
    HOUR_OF_WEEK = "hour_of_week"


def seasonal_cost(spread_frac: pd.Series, bucket: CostBucket = CostBucket.HOUR_OF_DAY, stat: str = "median",
                  fit_until=None) -> pd.Series:
    """Per-bar round-trip cost = the ``stat`` of ``spread_frac`` in each bar's clock bucket.

    ``spread_frac`` is the quoted spread as a fraction of price (≈ log-return units), from whatever source the caller
    trusts. Buckets live on the New York 17:00 roll clock. With ``fit_until``, the profile is estimated only from bars
    before that timestamp and then applied everywhere, so it carries no out-of-sample information.
    """
    clock = build_clock(spread_frac.index, exclude_windows=())
    buckets = clock.hour_of_week if CostBucket(bucket) == CostBucket.HOUR_OF_WEEK else clock.hour_of_week % 24
    fit = np.ones(len(spread_frac), dtype=bool)
    if fit_until is not None:
        stamp = pd.Timestamp(fit_until)
        tz = spread_frac.index.tz
        stamp = stamp.tz_localize(tz) if stamp.tz is None else stamp.tz_convert(tz)
        fit = spread_frac.index < stamp
    profile = pd.Series(spread_frac.to_numpy(dtype=float)[fit]).groupby(buckets[fit]).agg(stat)
    return pd.Series(profile.reindex(buckets).to_numpy(), index=spread_frac.index, name="cost")
