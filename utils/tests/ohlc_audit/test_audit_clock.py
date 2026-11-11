import datetime as dt

import numpy as np
import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit.clock import ClockEra, ClockOverride, DstSwitch, EraShiftClock, wall_clock
from okmich_quant_utils.ohlc_audit.enums import OverrideStatus

T = pd.Timestamp
NY = "America/New_York"


def _clock(**kw) -> EraShiftClock:
    eras = kw.pop("eras", (ClockEra(T("2000-01-01"), 7.0), ClockEra(T("2024-06-05"), 9.0), ClockEra(T("2026-08-30"), 6.0)))
    return EraShiftClock(NY, eras, **kw)


def _utc(clock, raw: str, symbol: str = "X", **kw) -> pd.Timestamp:
    return clock.to_utc(pd.DatetimeIndex([T(raw)]), symbol, **kw)[0]


def test_ny_plus_shift_tracks_us_dst():
    c = _clock()
    assert _utc(c, "2024-01-10 12:00") == T("2024-01-10 10:00", tz="UTC")     # NY+7 in EST = UTC+2
    assert _utc(c, "2024-05-10 12:00") == T("2024-05-10 09:00", tz="UTC")     # NY+7 in EDT = UTC+3
    assert _utc(c, "2025-01-10 12:00") == T("2025-01-10 08:00", tz="UTC")     # NY+9 in EST = UTC+4


def test_forward_seam_is_continuous_in_utc():
    c = _clock()
    a, b = _utc(c, "2024-06-04 23:55"), _utc(c, "2024-06-05 02:00")         # 16:55 and 17:00 NY across the +2 h seam
    assert b - a == pd.Timedelta(minutes=5)


def test_tz_label_is_ignored():
    idx = pd.DatetimeIndex([T("2024-01-10 12:00", tz="Etc/GMT-2")])
    assert wall_clock(idx)[0] == T("2024-01-10 12:00")
    assert _clock().to_utc(idx, "X")[0] == T("2024-01-10 10:00", tz="UTC")


def test_default_dst_switch_is_the_ny_moment_on_the_raw_clock():
    c = _clock()
    off = c.offset_hours(pd.DatetimeIndex([T("2024-03-10 08:55"), T("2024-03-10 09:00"), T("2024-03-10 10:00")]), "X")
    assert list(off) == [2.0, 3.0, 3.0]                                         # 02:00 EST = 07:00 UTC = 09:00 raw


def test_measured_dst_switch_applies_to_its_symbols_only():
    c = _clock(dst_switches=(DstSwitch(frozenset({"BTC"}), dt.date(2024, 3, 10), T("2024-03-10 06:40")),))
    raws = pd.DatetimeIndex([T("2024-03-10 06:35"), T("2024-03-10 06:40")])
    assert list(c.offset_hours(raws, "BTC")) == [2.0, 3.0]
    assert list(c.offset_hours(raws, "EURUSD")) == [2.0, 2.0]


def test_fall_back_fold_is_unrepresentable():
    c = _clock(dst_switches=(DstSwitch(frozenset({"BTC"}), dt.date(2025, 11, 2), T("2025-11-01 20:00")),))
    folds = c.unrepresentable_utc("BTC", T("2025-10-25", tz="UTC"), T("2025-11-10", tz="UTC"))
    assert (T("2025-11-01 15:00", tz="UTC"), T("2025-11-01 16:00", tz="UTC")) in folds


def test_backward_seam_is_unrepresentable():
    c = _clock()
    gaps = c.unrepresentable_utc("X", T("2026-08-25", tz="UTC"), T("2026-09-05", tz="UTC"))
    assert (T("2026-08-29 19:00", tz="UTC"), T("2026-08-29 22:00", tz="UTC")) in gaps     # offset 5 -> 2


def test_proposed_overrides_apply_only_on_request_confirmed_always():
    prop = ClockOverride(frozenset({"AUDUSD"}), T("2026-09-13"), T("2026-09-19"), 9.0, OverrideStatus.PROPOSED)
    conf = ClockOverride(frozenset({"BTC"}), T("2026-08-29 18:05"), T("2026-08-30"), 6.0, OverrideStatus.CONFIRMED)
    c = _clock(overrides=(prop, conf))
    raw = "2026-09-15 12:00"
    assert _utc(c, raw, "AUDUSD") == _utc(c, raw, "EURUSD")
    assert _utc(c, raw, "AUDUSD", include_proposed=True) == _utc(c, raw, "EURUSD") - pd.Timedelta(hours=3)
    assert _utc(c, "2026-08-29 20:00", "BTC") == T("2026-08-29 18:00", tz="UTC")       # NY+6 in EDT = UTC+2


def test_to_raw_inverts_to_utc():
    c = _clock()
    rng = np.random.default_rng(0)
    raw = pd.DatetimeIndex(T("2023-06-01") + pd.to_timedelta(rng.integers(0, 3 * 365 * 288, 500) * 5, unit="min"))
    utc = c.to_utc(raw, "X")
    back = c.to_raw(utc, "X")
    ok = back.notna()
    assert ok.all()
    assert (c.to_utc(back, "X") == utc).all()                       # always consistent
    # raw stamps in the forward-seam overlap (00:00-02:00 on the +2 h seam) have an older twin; the twin wins
    overlap = (raw >= T("2024-06-05 00:00")) & (raw < T("2024-06-05 02:00"))
    assert (back[~overlap] == raw[~overlap]).all()


def test_eras_must_increase():
    with pytest.raises(ValueError):
        EraShiftClock(NY, (ClockEra(T("2024-06-05"), 9.0), ClockEra(T("2000-01-01"), 7.0)))
