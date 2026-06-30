"""Per-bar event-timing features from the news calendar.

These are NOT macro-series features: ``minutes_to_next`` and ``blackout`` are forward/symmetric in
time, so they can't come from the backward asof-merge that the daily macro series use — they are
computed directly against each bar's timestamp (the same family as ``reader.is_event_window``). The
backward asof-merge / ``ExplicitRelease`` path is for the *surprise* feature (deferred: needs
released actuals + consensus), not for these.

No-lookahead: ``minutes_to_next`` legitimately reads *future* event timestamps because scheduled
high-impact releases are exogenously known months in advance (FOMC ~18mo, BLS ~12mo) — the schedule
was public well before any intraday bar. The nearest future event is therefore always
already-scheduled. (We don't track schedule-publication timestamps, so this is an assumption of the
data asset, valid for the high-impact scheduled set; ``minutes_since_last`` / ``blackout`` are
backward/symmetric and unconditionally safe.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from okmich_quant_pipeline.news_calendar._types import ImpactTier

# Defaults: ±3 bars on a 5m grid (±15 min) for the blackout window; saturate the minute-distance
# features at one day (beyond a day there is no "imminent event" signal left to resolve).
_DEFAULT_BLACKOUT_BARS = 3
_DEFAULT_BAR_SECONDS = 300
_DEFAULT_HORIZON_MINUTES = 24 * 60


def _to_ns_utc(index: pd.DatetimeIndex) -> np.ndarray:
    """Bar index → tz-naive UTC ``datetime64[ns]`` (real datetime arithmetic, never object arrays)."""
    if index.tz is None:
        return index.to_numpy()  # already naive — interpreted as UTC
    return index.tz_convert("UTC").tz_localize(None).to_numpy()


def _events_ns(timestamps: pd.Series) -> np.ndarray:
    """Event timestamp column → sorted tz-naive UTC ``datetime64[ns]`` array."""
    return (
        pd.to_datetime(timestamps, utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        .sort_values().to_numpy()
    )


def _event_distances_seconds(bars_ns: np.ndarray, events_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar seconds to the next event (ts >= bar) and since the last event (ts <= bar).

    ``np.inf`` where there is no event on that side. ``events_ns`` must be sorted ascending.
    """
    n = len(events_ns)
    if n == 0:
        inf = np.full(len(bars_ns), np.inf)
        return inf, inf.copy()
    one_sec = np.timedelta64(1, "s")

    idx_next = np.searchsorted(events_ns, bars_ns, side="left")  # first event >= bar
    has_next = idx_next < n
    to_next = np.where(has_next, (events_ns[np.clip(idx_next, 0, n - 1)] - bars_ns) / one_sec, np.inf)

    idx_prev = np.searchsorted(events_ns, bars_ns, side="right") - 1  # last event <= bar
    has_prev = idx_prev >= 0
    since = np.where(has_prev, (bars_ns - events_ns[np.clip(idx_prev, 0, n - 1)]) / one_sec, np.inf)
    return to_next, since


def compute_event_features(bar_index: pd.DatetimeIndex, calendar: pd.DataFrame, *,
                           tiers: tuple[ImpactTier, ...] = (ImpactTier.HIGH,),
                           blackout_bars: int = _DEFAULT_BLACKOUT_BARS, bar_seconds: int = _DEFAULT_BAR_SECONDS,
                           horizon_minutes: int = _DEFAULT_HORIZON_MINUTES) -> pd.DataFrame:
    """Per-bar event-timing features, indexed by ``bar_index``.

    Columns (no ``macro_`` prefix — the attach adds it):
    - ``minutes_to_next``    — minutes to the next event in ``tiers``, saturated at ``horizon_minutes``.
    - ``minutes_since_last`` — minutes since the most recent such event, saturated at ``horizon_minutes``.
    - ``blackout``           — 1.0 if within ±(``blackout_bars`` × ``bar_seconds``) of such an event.

    The minute features saturate (never NaN) so the modeling frame stays dense; the blackout test
    uses the raw, un-saturated distances. ``bar_seconds`` should match the dataset's timeframe (5m
    → 300) so the blackout window is the intended wall-clock radius.
    """
    if not isinstance(bar_index, pd.DatetimeIndex):
        raise ValueError("bar_index must be a pandas DatetimeIndex")

    tier_vals = {int(t) for t in tiers}
    cal = calendar[calendar["impact_tier"].isin(tier_vals)] if "impact_tier" in calendar.columns else calendar
    events_ns = _events_ns(cal["timestamp_utc"]) if len(cal) else np.array([], dtype="datetime64[ns]")

    bars_ns = _to_ns_utc(bar_index)
    to_next_s, since_s = _event_distances_seconds(bars_ns, events_ns)

    horizon_s = horizon_minutes * 60
    radius_s = blackout_bars * bar_seconds
    return pd.DataFrame({
        "minutes_to_next": np.minimum(to_next_s, horizon_s) / 60.0,
        "minutes_since_last": np.minimum(since_s, horizon_s) / 60.0,
        "blackout": (np.minimum(to_next_s, since_s) <= radius_s).astype(float),
    }, index=bar_index)
