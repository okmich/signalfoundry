"""Bar clock on the New York 17:00 roll: trading day, intraday slot, hour-of-week buckets, data gaps, excluded windows.

Every bar is placed on a (trading day, slot) lattice. The trading day ends at 17:00 New York, the FX convention, so
New York wall time shifted by +7h gives a calendar whose midnight is the roll. Buckets are therefore DST-robust:
a London-open bar falls in the same bucket in summer and winter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NEW_YORK = "America/New_York"
_ROLL_SHIFT = pd.Timedelta(hours=7)  # 17:00 New York -> 00:00 of the next trading day
_NS_PER_DAY = 86_400_000_000_000

# The FX 17:00 New York roll; covers the 20:00-24:00 UTC bid-bar spread artefact window in both DST seasons
DEFAULT_EXCLUDE_WINDOWS: tuple[tuple[str, str, str], ...] = ((NEW_YORK, "16:00", "20:00"),)


@dataclass(frozen=True)
class BarClock:
    index: pd.DatetimeIndex
    spacing: pd.Timedelta
    trading_day: np.ndarray  # day number of the trading day each bar belongs to
    slot: np.ndarray  # bar position within its trading day
    hour_of_week: np.ndarray  # 0..167, Monday 00:00 = Sunday 17:00 New York
    gap_before: np.ndarray  # the step from bar i-1 to bar i is a data gap
    excluded: np.ndarray  # bar i lies inside an excluded clock window

    @property
    def bars_per_day(self) -> int:
        return int(round(pd.Timedelta(days=1) / self.spacing))

    def bars(self, duration: str | pd.Timedelta) -> int:
        """Number of bars in a duration such as "1h", rounded, at least 1."""
        return max(1, int(round(pd.Timedelta(duration) / self.spacing)))


def build_clock(index: pd.DatetimeIndex, exclude_windows=DEFAULT_EXCLUDE_WINDOWS) -> BarClock:
    """Place every bar on the trading-day lattice. ``index`` must be tz-aware and on the true clock."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("index must be a tz-aware DatetimeIndex on the true clock; convert broker stamps first.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing.")
    if len(index) < 3:
        raise ValueError("index needs at least 3 bars.")

    index = index.as_unit("ns")
    steps = np.diff(index.asi8)
    spacing = pd.Timedelta(int(pd.Series(steps).mode().iloc[0]), "ns")

    wall = index.tz_convert(NEW_YORK).tz_localize(None) + _ROLL_SHIFT
    day_start = wall.normalize()
    trading_day = day_start.asi8 // _NS_PER_DAY
    slot = (wall.asi8 - day_start.asi8) // spacing.value
    hour_of_week = wall.dayofweek.to_numpy() * 24 + wall.hour.to_numpy()

    gap_before = np.r_[False, steps > 1.5 * spacing.value]
    excluded = np.zeros(len(index), dtype=bool)
    for tz, start, end in exclude_windows:
        local = index.tz_convert(tz)
        minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
        lo, hi = _minutes(start), _minutes(end)
        excluded |= ((minutes >= lo) & (minutes < hi)) if lo < hi else ((minutes >= lo) | (minutes < hi))

    return BarClock(index, spacing, trading_day, slot.astype(np.int64), hour_of_week.astype(np.int64), gap_before,
                    excluded)


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)
