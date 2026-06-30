"""Calendar parquet reader + ``is_event_window`` helper.

Used to mask out event-window bars (e.g. when computing entropy stats / per-class ECE excluding
events) and — the intended pipeline consumer — to feed the macro event channel via an
``ExplicitRelease`` availability policy. The mask radius defaults to ±3 bars on a 5m grid
(±15 minutes), tunable via arguments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical on-disk location of the materialized calendar (mirrors the macro store root).
DEFAULT_CALENDAR_PATH = Path(r"E:\data_dump\calendars\high_impact_news.parquet")

# Default mask radius for 5m bars: ±3 bars ≈ ±15 minutes around each event.
_DEFAULT_WINDOW_BARS = 3
_DEFAULT_BAR_SECONDS = 300


def load_calendar(path: Path = DEFAULT_CALENDAR_PATH) -> pd.DataFrame:
    """Load the high-impact news calendar parquet.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp_utc`` (datetime64[ns, UTC]), ``event_name`` (str),
        ``source`` (str), ``impact_tier`` (int). Sorted by ``timestamp_utc``.
    """
    return pd.read_parquet(path)


def is_event_window(bar_timestamps: pd.DatetimeIndex, calendar: pd.DataFrame, window_bars: int = _DEFAULT_WINDOW_BARS,
                    bar_seconds: int = _DEFAULT_BAR_SECONDS) -> pd.Series:
    """Return a boolean mask, ``True`` where the bar is within ``window_bars`` of any high-impact event.

    Parameters
    ----------
    bar_timestamps
        UTC-aware DatetimeIndex (the index of the 5m XAUUSD bar frame).
    calendar
        Result of ``load_calendar()``.
    window_bars
        Half-window in bars. Default ``3`` ≈ ±15 minutes on 5m.
    bar_seconds
        Seconds per bar. Default ``300`` (5m).
    """
    if calendar.empty:
        return pd.Series(False, index=bar_timestamps)
    radius_td = np.timedelta64(pd.Timedelta(seconds=window_bars * bar_seconds))
    # Normalize both sides to tz-naive UTC datetime64[ns] arrays. A tz-aware ``.to_numpy()``
    # returns an *object* array of Timestamps, on which searchsorted/subtraction are slow and
    # version-fragile; dropping the tz (after converting to UTC) keeps true datetime64 arithmetic.
    events = (
        pd.to_datetime(calendar["timestamp_utc"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        .sort_values().to_numpy()
    )
    if bar_timestamps.tz is None:
        bars = bar_timestamps.to_numpy()  # already naive — interpreted as UTC
    else:
        bars = bar_timestamps.tz_convert("UTC").tz_localize(None).to_numpy()
    # For each bar, find the nearest event via binary search. searchsorted
    # gives the insertion index; the nearest event is either at that index or
    # one before. Take min(|bar - events[idx-1]|, |bar - events[idx]|).
    idx = np.searchsorted(events, bars)
    left = np.clip(idx - 1, 0, len(events) - 1)
    right = np.clip(idx, 0, len(events) - 1)
    dist_left = np.abs(bars - events[left])
    dist_right = np.abs(bars - events[right])
    nearest = np.minimum(dist_left, dist_right)
    mask = nearest <= radius_td
    return pd.Series(mask, index=bar_timestamps)
