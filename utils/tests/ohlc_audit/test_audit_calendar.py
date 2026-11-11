import datetime as dt

import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit.calendar import Anchor, eve_of_first, expected_grid, explicit, minus, union
from okmich_quant_utils.ohlc_audit.profiles.fxpig import SESSIONS
from okmich_quant_utils.ohlc_audit.enums import SlotState

NY = "America/New_York"


def _state(template: str, ny_time: str, tz: str = NY) -> SlotState:
    t = pd.Timestamp(ny_time, tz=tz)
    grid = expected_grid(SESSIONS[template], t - pd.Timedelta(days=3), t + pd.Timedelta(days=3))
    return SlotState(grid.loc[t.tz_convert("UTC")])


@pytest.mark.parametrize("when, state", [
    ("2024-01-07 16:55", SlotState.CLOSED), ("2024-01-07 17:00", SlotState.THIN), ("2024-01-07 18:00", SlotState.OPEN),
    ("2024-01-12 16:55", SlotState.OPEN), ("2024-01-12 17:00", SlotState.CLOSED),
    ("2024-07-12 16:55", SlotState.OPEN), ("2024-07-12 17:00", SlotState.CLOSED),        # NY-anchored through DST
    ("2024-01-10 17:30", SlotState.THIN),                                                   # rollover hour
])
def test_fx_week(when, state):
    assert _state("fx", when) is state


def test_fx_christmas_trading_day_is_holiday():
    assert _state("fx", "2024-12-24 16:55") is SlotState.OPEN
    assert _state("fx", "2024-12-24 18:00") is SlotState.HOLIDAY
    assert _state("fx", "2024-12-25 16:55") is SlotState.HOLIDAY
    assert _state("fx", "2024-12-25 18:00") is SlotState.OPEN


def test_metal_daily_break_and_friday_close():
    assert _state("metal", "2024-01-08 16:55") is SlotState.OPEN
    assert _state("metal", "2024-01-08 17:00") is SlotState.CLOSED
    assert _state("metal", "2024-01-08 18:00") is SlotState.OPEN
    assert _state("metal", "2024-01-12 16:45") is SlotState.OPEN
    assert _state("metal", "2024-01-12 16:50") is SlotState.CLOSED


def test_us_index_holiday_rules():
    assert _state("us_index", "2024-01-15 12:40") is SlotState.OPEN            # MLK
    assert _state("us_index", "2024-01-15 12:45") is SlotState.EARLY_CLOSE
    assert _state("us_index", "2024-01-15 17:00") is SlotState.CLOSED          # holiday never relabels the break
    assert _state("us_index", "2024-01-15 18:00") is SlotState.OPEN
    assert _state("us_index", "2024-03-29 10:00") is SlotState.HOLIDAY         # Good Friday
    assert _state("us_index", "2025-01-09 09:30") is SlotState.EARLY_CLOSE     # national day of mourning
    assert _state("us_index", "2025-01-09 09:25") is SlotState.OPEN
    assert _state("us_index", "2025-11-28 03:00") is SlotState.HOLIDAY         # CME Globex halt
    assert _state("us_index", "2024-11-29 13:00") is SlotState.EARLY_CLOSE     # day after Thanksgiving


def test_de30_xetra_closure_runs_berlin_to_utc():
    assert _state("de30", "2023-12-22 15:55") is SlotState.OPEN
    assert _state("de30", "2023-12-22 16:00") is SlotState.HOLIDAY             # 22:00 CET
    assert _state("de30", "2023-12-26 19:10") is SlotState.HOLIDAY
    assert _state("de30", "2023-12-26 19:15") is SlotState.OPEN                # 00:15 UTC Wed 27th
    assert _state("de30", "2023-12-25 17:30") is SlotState.CLOSED


def test_hkg50_monday_holiday_starts_friday_night():
    assert _state("hkg50", "2023-09-29 14:55") is SlotState.OPEN
    assert _state("hkg50", "2023-09-29 15:00") is SlotState.HOLIDAY            # Sat 03:00 HKT, end of Fri night session
    assert _state("hkg50", "2023-10-02 21:10") is SlotState.HOLIDAY
    assert _state("hkg50", "2023-10-02 21:15") is SlotState.OPEN               # Tue 09:15 HKT


def test_crypto_is_always_open():
    assert _state("crypto", "2024-06-08 03:00", tz="UTC") is SlotState.OPEN


def test_business_anchor_steps_over_weekends():
    monday = dt.date(2023, 10, 2)
    assert Anchor(-1, dt.time(3), "Asia/Hong_Kong", business=True, then_days=1).day(monday) == dt.date(2023, 9, 30)
    assert Anchor(1, dt.time(9, 15), "Asia/Hong_Kong", business=True).day(dt.date(2023, 9, 29)) == dt.date(2023, 10, 2)


def test_date_source_helpers():
    a, b = explicit("2024-02-10", "2024-02-12", "2024-02-13"), explicit("2024-02-12")
    assert minus(a, b)(2024, 2024) == [dt.date(2024, 2, 10), dt.date(2024, 2, 13)]
    assert union(a, b)(2024, 2024) == [dt.date(2024, 2, 10), dt.date(2024, 2, 12), dt.date(2024, 2, 13)]
    assert eve_of_first(explicit("2024-02-12", "2024-02-13"))(2024, 2024) == [dt.date(2024, 2, 9)]   # Friday before
