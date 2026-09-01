"""Offline unit tests for the migrated news-calendar package.

Covers the pure parsing/logic surface — release-time DST conversion, the three fetchers'
HTML/JSON parsers (on fixtures, no network), the ``is_event_window`` mask, and the shared HTTP
``browser_ua`` swap. Network ``fetch()`` calls stay out of the suite; ``sanity.py`` is the
live integration check.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from okmich_quant_pipeline import http as http_mod
from okmich_quant_pipeline.news_calendar import build as build_mod
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source
from okmich_quant_pipeline.news_calendar.fetchers import fomc, forexfactory, fred
from okmich_quant_pipeline.news_calendar.reader import is_event_window
from okmich_quant_pipeline.news_calendar.release_times import RELEASE_TIMES, to_utc


# --------------------------------------------------------------------------- #
# release_times — DST-aware UTC conversion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("event, date, expected", [
    # 14:00 ET FOMC: 19:00 UTC in winter (EST), 18:00 UTC in summer (EDT).
    (EventName.US_FOMC_STATEMENT, dt.date(2024, 1, 31), "2024-01-31T19:00:00+00:00"),
    (EventName.US_FOMC_STATEMENT, dt.date(2024, 7, 31), "2024-07-31T18:00:00+00:00"),
    # 08:30 ET BLS: 13:30 UTC winter, 12:30 UTC summer.
    (EventName.US_NFP, dt.date(2024, 1, 5), "2024-01-05T13:30:00+00:00"),
    (EventName.US_NFP, dt.date(2024, 7, 5), "2024-07-05T12:30:00+00:00"),
    # 14:15 local ECB: 13:15 UTC winter (CET), 12:15 UTC summer (CEST).
    (EventName.EU_ECB_RATE_DECISION, dt.date(2024, 1, 25), "2024-01-25T13:15:00+00:00"),
    (EventName.EU_ECB_RATE_DECISION, dt.date(2024, 7, 18), "2024-07-18T12:15:00+00:00"),
    # 12:00 local BoE: 12:00 UTC winter (GMT), 11:00 UTC summer (BST).
    (EventName.UK_BOE_RATE_DECISION, dt.date(2024, 2, 1), "2024-02-01T12:00:00+00:00"),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2024, 8, 1), "2024-08-01T11:00:00+00:00"),
])
def test_to_utc_dst(event: EventName, date: dt.date, expected: str) -> None:
    assert to_utc(date, event).isoformat() == expected


def test_every_event_name_has_a_release_time() -> None:
    # Guards the documented failure mode: a new EventName without a release_times entry would
    # make build.py raise KeyError at orchestration. Catch it here instead.
    for e in EventName:
        assert e in RELEASE_TIMES


# --------------------------------------------------------------------------- #
# fred fetcher — ALFRED release-date parsing
# --------------------------------------------------------------------------- #

def test_fred_parse_release_dates_filters_range_and_dedupes() -> None:
    html = """
    <ul>
      <li data-release-date="2023-12-08"></li>
      <li data-release-date="2024-01-05"></li>
      <li data-release-date="2024-01-05"></li>   <!-- dup in a widget -->
      <li data-release-date="2024-02-02"></li>
      <li data-release-date="2027-01-08"></li>   <!-- out of range -->
    </ul>
    """
    out = fred._parse_release_dates(html, year_min=2024, year_max=2026)
    assert out == [dt.date(2024, 1, 5), dt.date(2024, 2, 2)]  # 2023 + 2027 dropped, dup collapsed, sorted


# --------------------------------------------------------------------------- #
# fomc fetcher — meeting-row + calendar-HTML parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("year, month_cell, date_cell, expected", [
    ("2024", "January", "30-31", dt.date(2024, 1, 31)),       # same-month, statement = last day
    ("2024", "April/May", "30-1", dt.date(2024, 5, 1)),       # cross-month -> last day in second month
    ("2024", "September", "17-18*", dt.date(2024, 9, 18)),    # asterisk (SEP marker) stripped
    ("2024", "June", "12", dt.date(2024, 6, 12)),             # single-day meeting
    ("2024", "December/January", "31-1", dt.date(2025, 1, 1)),  # cross-year roll
])
def test_fomc_parse_meeting_row(year: str, month_cell: str, date_cell: str, expected: dt.date) -> None:
    assert fomc._parse_meeting_row(int(year), month_cell, date_cell) == expected


def test_fomc_parse_calendar_html() -> None:
    html = """
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a>2024 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>January</strong></div>
        <div class="fomc-meeting__date">30-31</div>
      </div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>April/May</strong></div>
        <div class="fomc-meeting__date">30-1</div>
      </div>
    </div>
    """
    parsed = fomc._parse_calendar_html(html, year_min=2024, year_max=2026)
    dates = [d for d, _pc, _y in parsed]
    assert dates == [dt.date(2024, 1, 31), dt.date(2024, 5, 1)]
    assert all(pc is True for _d, pc, _y in parsed)  # 2024 >= 2019 -> press conf unconditionally


def test_fomc_parse_raises_on_malformed_meeting_row() -> None:
    # Row passes the meeting-like pre-filter (month + numeric date) but the date cell is malformed;
    # the parser must fail loudly rather than silently drop a high-impact event.
    html = """
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a>2024 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>January</strong></div>
        <div class="fomc-meeting__date">30-31-32</div>
      </div>
    </div>
    """
    with pytest.raises(RuntimeError, match="failed to parse"):
        fomc._parse_calendar_html(html, year_min=2024, year_max=2026)


def test_fomc_parse_skips_annotated_non_meeting_rows() -> None:
    # A "(notation vote)" row is an inter-meeting administrative action, not a scheduled meeting —
    # it must be skipped (no bogus event, no raise) while the real meeting row is kept.
    html = """
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a>2025 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>January</strong></div>
        <div class="fomc-meeting__date">28-29</div>
      </div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>August</strong></div>
        <div class="fomc-meeting__date">22 (notation vote)</div>
      </div>
    </div>
    """
    parsed = fomc._parse_calendar_html(html, year_min=2025, year_max=2027)
    assert [d for d, _pc, _y in parsed] == [dt.date(2025, 1, 29)]  # annotated row dropped, no raise


def test_fomc_fetch_builds_statement_and_press_rows(monkeypatch) -> None:
    html = """
    <div class="panel panel-default">
      <div class="panel-heading"><h4><a>2024 FOMC Meetings</a></h4></div>
      <div class="row fomc-meeting">
        <div class="fomc-meeting__month"><strong>January</strong></div>
        <div class="fomc-meeting__date">30-31</div>
      </div>
    </div>
    """
    monkeypatch.setattr(fomc, "get", lambda url: type("R", (), {"text": html})())
    df = fomc.fetch(2024, 2026)
    assert set(df["event_name"]) == {EventName.US_FOMC_STATEMENT, EventName.US_FOMC_PRESS_CONFERENCE}
    assert (df["source"] == Source.FED).all()


# --------------------------------------------------------------------------- #
# forexfactory fetcher — JS-blob extraction + event mapping/filtering
# --------------------------------------------------------------------------- #

_FF_HTML = """
<html><body><script>
window.calendarComponentStates[1] = {
  days: [
    {"date":"Jan 25","events":[
      {"country":"EZ","name":"Main Refinancing Rate","dateline":1706184900,"impactName":"high"},
      {"country":"EZ","name":"ECB Press Conference","dateline":1706186700,"impactName":"high"},
      {"country":"EZ","name":"Monetary Policy Statement","dateline":1706184900,"impactName":"medium"},
      {"country":"UK","name":"Official Bank Rate","dateline":1706789100,"impactName":"high"},
      {"country":"US","name":"Unmapped Thing","dateline":1706184900,"impactName":"low"}
    ]}
  ],
  more: 1
};
</script></body></html>
"""


def test_ff_extract_days_brace_matches() -> None:
    days = forexfactory._extract_days(_FF_HTML)
    assert len(days) == 1 and len(days[0]["events"]) == 5


def test_ff_parse_quarter_maps_and_filters() -> None:
    rows = forexfactory._parse_quarter(_FF_HTML)
    names = {r["event_name"] for r in rows}
    # high-impact mapped events kept; the medium EZ statement and the unmapped US/low row dropped.
    assert names == {EventName.EU_ECB_RATE_DECISION, EventName.EU_ECB_PRESS_CONFERENCE, EventName.UK_BOE_RATE_DECISION}
    assert all(r["impact_tier"] == ImpactTier.HIGH for r in rows)
    sources = {r["source"] for r in rows}
    assert sources == {Source.ECB, Source.BOE}


def test_ff_range_url_format() -> None:
    url = forexfactory._range_url(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    assert url.endswith("range=jan1.2024-mar31.2024")


def test_ff_quarter_windows_count() -> None:
    assert len(forexfactory._quarter_windows(2024, 2026)) == 12  # 4 quarters x 3 years


# --------------------------------------------------------------------------- #
# reader — is_event_window mask
# --------------------------------------------------------------------------- #

def _calendar(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True)})


def test_is_event_window_merges_adjacent_events() -> None:
    cal = _calendar(["2024-03-20 18:00", "2024-03-20 18:30"])  # statement + press conf
    bars = pd.date_range("2024-03-20 17:00", "2024-03-20 19:00", freq="5min", tz="UTC")
    masked = bars[is_event_window(bars, cal, window_bars=3, bar_seconds=300)]
    assert masked.min() == pd.Timestamp("2024-03-20 17:45", tz="UTC")
    assert masked.max() == pd.Timestamp("2024-03-20 18:45", tz="UTC")


def test_is_event_window_empty_calendar_is_all_false() -> None:
    bars = pd.date_range("2024-03-20 17:00", periods=10, freq="5min", tz="UTC")
    mask = is_event_window(bars, _calendar([]), window_bars=3)
    assert not mask.any() and len(mask) == len(bars)


def test_is_event_window_localizes_tz_naive_bars() -> None:
    cal = _calendar(["2024-03-20 18:00"])
    bars = pd.date_range("2024-03-20 17:45", "2024-03-20 18:15", freq="5min")  # tz-naive -> treated as UTC
    mask = is_event_window(bars, cal, window_bars=3, bar_seconds=300)
    assert mask.all()


# --------------------------------------------------------------------------- #
# shared http — browser_ua header swap (offline, monkeypatched session)
# --------------------------------------------------------------------------- #

def test_http_browser_ua_swaps_only_when_requested(monkeypatch) -> None:
    captured: list[dict | None] = []

    class _Resp:
        def raise_for_status(self) -> None:
            pass

    class _Session:
        def get(self, url, timeout=None, headers=None):  # noqa: ANN001
            captured.append(headers)
            return _Resp()

    monkeypatch.setattr(http_mod, "get_session", lambda: _Session())
    http_mod.get("https://x", browser_ua=True)
    http_mod.get("https://x")
    assert captured[0]["User-Agent"].startswith("Mozilla")  # browser UA on request 1
    assert captured[1] is None                              # default session UA on request 2


# --------------------------------------------------------------------------- #
# build — coverage validation (partial-scrape guard) + dynamic default window
# --------------------------------------------------------------------------- #

def test_build_validate_coverage_raises_on_partial_scrape() -> None:
    final = pd.DataFrame({"event_name": ["US_FOMC_STATEMENT"]})  # 1 row for a 3-year window
    with pytest.raises(ValueError, match="coverage check failed"):
        build_mod._validate_coverage(final, 2024, 2026)


def test_build_validate_coverage_passes_on_healthy_counts() -> None:
    rows = (["US_FOMC_STATEMENT"] * 8 + ["US_NFP"] * 12 + ["US_CPI"] * 12
            + ["EU_ECB_RATE_DECISION"] * 4 + ["UK_BOE_RATE_DECISION"] * 4)
    build_mod._validate_coverage(pd.DataFrame({"event_name": rows}), 2024, 2024)  # 1 year — must not raise


def test_build_validate_coverage_counts_only_completed_years() -> None:
    # 2025-2027 evaluated in 2026: only 2025 is fully completed, so the floor is one year's worth.
    # One year of counts clears it even though a naive 3-year floor (e.g. NFP >= 27) would reject —
    # this is what stops the partial current/future year from false-tripping the guard.
    rows = (["US_FOMC_STATEMENT"] * 8 + ["US_NFP"] * 10 + ["US_CPI"] * 10
            + ["EU_ECB_RATE_DECISION"] * 3 + ["UK_BOE_RATE_DECISION"] * 3)
    build_mod._validate_coverage(pd.DataFrame({"event_name": rows}), 2025, 2027, today_year=2026)


def test_default_years_track_today() -> None:
    y = dt.date.today().year
    assert build_mod._default_years() == (y - 1, y + 1)  # rolls forward with the calendar
