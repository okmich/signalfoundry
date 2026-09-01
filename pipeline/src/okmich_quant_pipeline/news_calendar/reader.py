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

from okmich_quant_pipeline.news_calendar.features import _event_distances_seconds, _events_ns, _to_ns_utc

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
    # Nearest event = min(seconds to next, seconds since last); within radius ⇒ in an event window.
    # Shared with compute_event_features so the two never diverge on the binary-search / tz handling.
    to_next, since = _event_distances_seconds(_to_ns_utc(bar_timestamps), _events_ns(calendar["timestamp_utc"]))
    mask = np.minimum(to_next, since) <= window_bars * bar_seconds
    return pd.Series(mask, index=bar_timestamps)
