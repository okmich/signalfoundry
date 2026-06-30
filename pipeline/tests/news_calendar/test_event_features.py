"""Per-bar event-timing feature tests (compute_event_features + attach_events_to_dataset).

All offline: a small hand-built calendar (two HIGH events + one MEDIUM) is probed on a 5m grid so
every minutes_to_next / minutes_since_last / blackout value is checkable by hand.
"""
from __future__ import annotations

import pandas as pd
import pytest

from okmich_quant_pipeline._io import atomic_write_parquet
from okmich_quant_pipeline.macro.attach import attach_events_to_dataset
from okmich_quant_pipeline.news_calendar._types import ImpactTier
from okmich_quant_pipeline.news_calendar.features import compute_event_features


def _cal() -> pd.DataFrame:
    # Two HIGH events (18:00, 18:30) and one MEDIUM (12:00) on 2024-03-20.
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2024-03-20 18:00", "2024-03-20 18:30", "2024-03-20 12:00"], utc=True),
        "impact_tier": [3, 3, 2],
    })


def _bars(*stamps: str, tz: str | None = "UTC") -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(stamps))).tz_localize(tz) if tz else pd.DatetimeIndex(pd.to_datetime(list(stamps)))


def test_minutes_to_next_and_since_last() -> None:
    f = compute_event_features(_bars("2024-03-20 17:45", "2024-03-20 18:15"), _cal())
    assert f["minutes_to_next"].tolist() == [15.0, 15.0]      # 17:45->18:00, 18:15->18:30
    assert f["minutes_since_last"].iloc[1] == 15.0            # 18:15 since 18:00


def test_exact_release_instant_is_zero_both_ways() -> None:
    f = compute_event_features(_bars("2024-03-20 18:00"), _cal())
    assert f["minutes_to_next"].iloc[0] == 0.0 and f["minutes_since_last"].iloc[0] == 0.0


def test_minutes_saturate_at_horizon_no_nan() -> None:
    f = compute_event_features(_bars("2024-03-20 17:45"), _cal(), horizon_minutes=60)
    assert f["minutes_since_last"].iloc[0] == 60.0           # nothing before -> saturates, not NaN
    assert f["minutes_to_next"].notna().all()


def test_blackout_window_is_inclusive_at_radius() -> None:
    # radius = 3 bars * 300s = 900s = 15 min.
    f = compute_event_features(_bars("2024-03-20 17:45", "2024-03-20 17:40"), _cal(), blackout_bars=3, bar_seconds=300)
    assert f["blackout"].iloc[0] == 1.0                      # exactly 15 min -> inside
    assert f["blackout"].iloc[1] == 0.0                      # 20 min -> outside


def test_tier_filter_ignores_lower_impact() -> None:
    f = compute_event_features(_bars("2024-03-20 12:05"), _cal(), tiers=(ImpactTier.HIGH,), horizon_minutes=1440)
    assert f["minutes_since_last"].iloc[0] == 1440.0         # the 12:00 MEDIUM is not counted
    assert f["minutes_to_next"].iloc[0] == 355.0             # 12:05 -> 18:00
    assert f["blackout"].iloc[0] == 0.0


def test_no_high_tier_events_saturates_and_no_blackout() -> None:
    medium_only = pd.DataFrame({"timestamp_utc": pd.to_datetime(["2024-03-20 12:00"], utc=True), "impact_tier": [2]})
    f = compute_event_features(_bars("2024-03-20 12:05"), medium_only, horizon_minutes=1440)
    assert f["minutes_to_next"].iloc[0] == 1440.0 and f["minutes_since_last"].iloc[0] == 1440.0
    assert f["blackout"].iloc[0] == 0.0


def test_empty_calendar() -> None:
    empty = pd.DataFrame({"timestamp_utc": pd.to_datetime([], utc=True), "impact_tier": pd.Series([], dtype=int)})
    f = compute_event_features(_bars("2024-03-20 12:05"), empty, horizon_minutes=1440)
    assert f["minutes_to_next"].iloc[0] == 1440.0 and f["blackout"].iloc[0] == 0.0


def test_tz_naive_bars_treated_as_utc() -> None:
    naive = compute_event_features(_bars("2024-03-20 17:45", tz=None), _cal())
    aware = compute_event_features(_bars("2024-03-20 17:45"), _cal())
    assert naive["minutes_to_next"].iloc[0] == aware["minutes_to_next"].iloc[0] == 15.0


def test_no_lookahead_bar_before_first_event() -> None:
    # A bar strictly before the first HIGH event sees it only as "to_next", never as "since".
    f = compute_event_features(_bars("2024-03-20 17:00"), _cal(), horizon_minutes=1440)
    assert f["minutes_to_next"].iloc[0] == 60.0              # 17:00 -> 18:00
    assert f["minutes_since_last"].iloc[0] == 1440.0         # nothing before -> saturated


def test_compute_requires_datetimeindex() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        compute_event_features(pd.RangeIndex(3), _cal())


def test_us_resolution_bar_index_matches_ns() -> None:
    # Real parquet bar frames are often datetime64[us] while event stamps are [ns]; numpy must align
    # the units (the macro asof-merge had to normalize this explicitly — here searchsorted handles it).
    ns = compute_event_features(_bars("2024-03-20 17:45"), _cal())
    us = compute_event_features(_bars("2024-03-20 17:45").as_unit("us"), _cal())
    assert us["minutes_to_next"].iloc[0] == ns["minutes_to_next"].iloc[0] == 15.0
    assert us["blackout"].iloc[0] == ns["blackout"].iloc[0] == 1.0


def test_non_utc_tz_index_is_converted() -> None:
    # 14:00 America/New_York == 18:00 UTC == the first HIGH event.
    ny = pd.DatetimeIndex(pd.to_datetime(["2024-03-20 14:00"])).tz_localize("America/New_York")
    f = compute_event_features(ny, _cal())
    assert f["minutes_to_next"].iloc[0] == 0.0 and f["minutes_since_last"].iloc[0] == 0.0


def test_attach_events_to_dataset(tmp_path) -> None:
    cal = _cal()
    cal["event_name"], cal["source"] = "X", "y"
    cal["release_date"] = cal["timestamp_utc"].dt.date
    path = tmp_path / "cal.parquet"
    atomic_write_parquet(cal, path)

    idx = pd.date_range("2024-03-20 17:00", "2024-03-20 19:00", freq="5min", tz="UTC")
    ds = pd.DataFrame({"feat_x": range(len(idx))}, index=idx)
    out = attach_events_to_dataset(ds, path, blackout_bars=3, bar_seconds=300)

    assert list(out.columns) == ["feat_x", "macro_event_minutes_to_next", "macro_event_minutes_since_last", "macro_event_blackout"]
    assert len(out) == len(ds)                               # nothing dropped (features saturate)
    assert out[[c for c in out.columns if c.startswith("macro_event_")]].notna().all().all()
    assert out.loc[pd.Timestamp("2024-03-20 18:00", tz="UTC"), "macro_event_blackout"] == 1.0
