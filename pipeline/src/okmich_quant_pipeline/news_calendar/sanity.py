"""Sanity checks for the high-impact news calendar.

Run via:

    python -m okmich_quant_pipeline.news_calendar.sanity

What this validates
-------------------
1. Parquet loads cleanly with expected columns and dtypes.
2. Event counts per source / event_name match the published schedules
   (FOMC = 8 meetings/year, NFP = ~12/year, ECB = 4/year on FF, ...).
3. **Known-date spot checks** -- UTC timestamps for a handful of well-known
   events match the agency's published time convention with correct DST.
4. ``is_event_window`` produces the expected mask radius around a sample
   set of events on a synthetic 5m bar grid.
5. Event density per month is roughly uniform (no quarter-long gaps that
   would signal a parser dropping data).

Prints a PASS/FAIL line per check so the report can cite this script.
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import Counter

import pandas as pd

from okmich_quant_pipeline.news_calendar.reader import is_event_window, load_calendar


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def check_load() -> pd.DataFrame:
    print("\n[1] Load calendar parquet")
    cal = load_calendar()
    expected_cols = {"timestamp_utc", "event_name", "source", "impact_tier", "release_date"}
    missing = expected_cols - set(cal.columns)
    if missing:
        _fail(f"missing columns: {missing}")
        sys.exit(1)
    if cal["timestamp_utc"].dt.tz is None:
        _fail("timestamp_utc is tz-naive; expected UTC-aware")
        sys.exit(1)
    _ok(f"loaded {len(cal)} rows, all expected columns present, timestamp_utc is tz-aware UTC")
    return cal


def check_counts(cal: pd.DataFrame) -> None:
    print("\n[2] Event counts vs published schedules")
    by_event = Counter(cal["event_name"])
    # FOMC: 8 meetings/year x 3 years = 24. Each meeting => 1 statement + 1 press conf = 48 total.
    if by_event.get("US_FOMC_STATEMENT", 0) == 24 and by_event.get("US_FOMC_PRESS_CONFERENCE", 0) == 24:
        _ok("FOMC: 24 statements + 24 press conferences (8 meetings/year x 3)")
    else:
        _fail(f"FOMC count off: statements={by_event.get('US_FOMC_STATEMENT', 0)}, press={by_event.get('US_FOMC_PRESS_CONFERENCE', 0)}")
    # NFP: ~12/year x 3 years = ~36 (give or take benchmark revisions + shutdown shifts).
    nfp = by_event.get("US_NFP", 0)
    if 28 <= nfp <= 40:
        _ok(f"US_NFP: {nfp} rows (expected ~36 incl. benchmark revisions / shutdown shifts)")
    else:
        _fail(f"US_NFP outside expected range: {nfp}")
    # ECB+BoE: FF lists 3-4/year x 3 years = ~9-12 per bank.
    ecb = by_event.get("EU_ECB_RATE_DECISION", 0)
    boe = by_event.get("UK_BOE_RATE_DECISION", 0)
    if 9 <= ecb <= 14 and 9 <= boe <= 14:
        _ok(f"ECB rate decisions: {ecb}, BoE rate decisions: {boe} (FF only flags SEP/MPR meetings)")
    else:
        _fail(f"ECB/BoE counts unexpected: ECB={ecb}, BoE={boe}")


def check_known_dates(cal: pd.DataFrame) -> None:
    """Spot-check published-time-convention UTC timestamps for known events."""
    print("\n[3] Known-date timestamp spot checks (DST verification)")
    cal = cal.copy()
    cal["release_date"] = pd.to_datetime(cal["release_date"]).dt.date
    cases: list[tuple[str, dt.date, str, str]] = [
        # (event_name, release_date, expected_utc, note)
        ("US_FOMC_STATEMENT",        dt.date(2024, 1, 31), "2024-01-31 19:00:00+00:00", "EST: 14:00 ET = 19:00 UTC"),
        ("US_FOMC_STATEMENT",        dt.date(2024, 7, 31), "2024-07-31 18:00:00+00:00", "EDT: 14:00 ET = 18:00 UTC"),
        ("US_NFP",                   dt.date(2024, 1, 5),  "2024-01-05 13:30:00+00:00", "EST: 08:30 ET = 13:30 UTC"),
        ("US_NFP",                   dt.date(2024, 7, 5),  "2024-07-05 12:30:00+00:00", "EDT: 08:30 ET = 12:30 UTC"),
        ("US_CPI",                   dt.date(2024, 1, 11), "2024-01-11 13:30:00+00:00", "EST: 08:30 ET = 13:30 UTC"),
        ("EU_ECB_RATE_DECISION",     dt.date(2024, 1, 25), "2024-01-25 13:15:00+00:00", "CET: 14:15 local = 13:15 UTC"),
        ("EU_ECB_RATE_DECISION",     dt.date(2024, 7, 18), "2024-07-18 12:15:00+00:00", "CEST: 14:15 local = 12:15 UTC"),
        ("UK_BOE_RATE_DECISION",     dt.date(2024, 2, 1),  "2024-02-01 12:00:00+00:00", "GMT: 12:00 UK = 12:00 UTC"),
        ("UK_BOE_RATE_DECISION",     dt.date(2024, 8, 1),  "2024-08-01 11:00:00+00:00", "BST: 12:00 UK = 11:00 UTC"),
    ]
    for event_name, rd, expected, note in cases:
        sub = cal[(cal["event_name"] == event_name) & (cal["release_date"] == rd)]
        if sub.empty:
            _fail(f"{event_name} {rd} -- row not found in calendar  ({note})")
            continue
        got = str(sub.iloc[0]["timestamp_utc"])
        if got == expected:
            _ok(f"{event_name} {rd}  ->  {got}   ({note})")
        else:
            _fail(f"{event_name} {rd}  expected {expected}, got {got}   ({note})")


def check_is_event_window(cal: pd.DataFrame) -> None:
    print("\n[4] is_event_window mask radius on 5m bar grid")
    # FOMC Mar 20 2024 statement at 18:00 UTC, press conf at 18:30 UTC.
    # With +/-3 bars (+/-15 min) on 5m bars, the two windows merge into a
    # continuous span from 17:45 to 18:45 UTC inclusive.
    bars = pd.date_range("2024-03-20 17:00", "2024-03-20 19:00", freq="5min", tz="UTC")
    mask = is_event_window(bars, cal, window_bars=3, bar_seconds=300)
    expected_start = pd.Timestamp("2024-03-20 17:45", tz="UTC")
    expected_end = pd.Timestamp("2024-03-20 18:45", tz="UTC")
    masked_bars = bars[mask]
    if len(masked_bars) and masked_bars.min() == expected_start and masked_bars.max() == expected_end:
        _ok(f"FOMC Mar 20 2024 mask spans {masked_bars.min()} -> {masked_bars.max()} (statement + press conf merged correctly)")
    else:
        _fail(f"Mask span unexpected: {masked_bars.min() if len(masked_bars) else 'empty'} -> {masked_bars.max() if len(masked_bars) else 'empty'}")
    # Mask coverage on a year of XAUUSD-trading bars (rough estimate).
    year_bars = pd.date_range("2024-01-01", "2024-12-31 23:55", freq="5min", tz="UTC")
    year_mask = is_event_window(year_bars, cal, window_bars=3, bar_seconds=300)
    pct = year_mask.mean() * 100
    print(f"          +/-3-bar mask coverage on 2024 5m bars: {pct:.2f}%  ({year_mask.sum()} of {len(year_bars)})")


def check_monthly_density(cal: pd.DataFrame) -> None:
    print("\n[5] Monthly event density (historical months only -- agencies publish ~6 mo ahead)")
    # Only audit months that are already in the past. BLS/BEA/Census publish
    # schedules 6-12 months ahead; future months legitimately carry only the
    # FOMC dates (Fed publishes 18+ months ahead).
    today = pd.Timestamp.now(tz="UTC").normalize()
    cutoff = today - pd.DateOffset(months=1)
    monthly = cal.set_index("timestamp_utc").resample("MS").size()
    historical = monthly[monthly.index < cutoff]
    suspected_gaps = historical[historical < 3]
    future_months = len(monthly) - len(historical)
    if not len(suspected_gaps):
        _ok(
            f"historical months all have >=3 events "
            f"(min={historical.min()}, median={int(historical.median())}, max={historical.max()}; "
            f"{future_months} future months not audited)"
        )
    else:
        _fail(f"historical months with <3 events (possible parser drops): {list(suspected_gaps.index)}")


def main() -> None:
    cal = check_load()
    check_counts(cal)
    check_known_dates(cal)
    check_is_event_window(cal)
    check_monthly_density(cal)
    print("\nAll sanity checks complete.")


if __name__ == "__main__":
    main()
