"""Offline unit tests for the migrated news-calendar package.

Covers the pure parsing/logic surface — release-time DST conversion, the three fetchers'
HTML/JSON parsers (on fixtures, no network), the ``is_event_window`` mask, and the shared HTTP
``browser_ua`` swap. Network ``fetch()`` calls stay out of the suite; ``sanity.py`` is the
live integration check.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from okmich_quant_pipeline import http as http_mod
from okmich_quant_pipeline.news_calendar import build as build_mod
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source
from okmich_quant_pipeline.news_calendar.fetchers import fomc, forexfactory, fred
from okmich_quant_pipeline.news_calendar.reader import is_event_window
from okmich_quant_pipeline.news_calendar.release_times import (EXCEPTIONS, RELEASE_TIMES, local_date, release_tz,
                                                               to_utc)


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


@pytest.mark.parametrize("event, date, expected", [
    # ECB: 13:45 / 14:30 Frankfurt through the 2022-06-09 meeting, 14:15 / 14:45 from 2022-07-21 (both CEST here).
    (EventName.EU_ECB_RATE_DECISION, dt.date(2022, 6, 9), "2022-06-09T11:45:00+00:00"),
    (EventName.EU_ECB_RATE_DECISION, dt.date(2022, 7, 21), "2022-07-21T12:15:00+00:00"),
    (EventName.EU_ECB_PRESS_CONFERENCE, dt.date(2022, 6, 9), "2022-06-09T12:30:00+00:00"),
    (EventName.EU_ECB_PRESS_CONFERENCE, dt.date(2022, 7, 21), "2022-07-21T12:45:00+00:00"),
    # Single-release exceptions: BoE 2020 at 07:00 (GMT, then BST) and 14:30; PCE at 10:00 ET on a GDP day.
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 3, 11), "2020-03-11T07:00:00+00:00"),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 3, 19), "2020-03-19T14:30:00+00:00"),
    (EventName.UK_BOE_RATE_DECISION, dt.date(2020, 5, 7), "2020-05-07T06:00:00+00:00"),
    (EventName.US_PCE, dt.date(2019, 11, 27), "2019-11-27T15:00:00+00:00"),
    (EventName.US_PCE, dt.date(2019, 10, 31), "2019-10-31T12:30:00+00:00"),
])
def test_to_utc_follows_eras_and_exceptions(event: EventName, date: dt.date, expected: str) -> None:
    assert to_utc(date, event).isoformat() == expected


@pytest.mark.parametrize("event, date", [
    (EventName.US_NFP, dt.date(2010, 12, 3)),
    (EventName.US_FOMC_STATEMENT, dt.date(2013, 1, 30)),     # statements came at 12:30 or 14:15 then
    (EventName.EU_ECB_RATE_DECISION, dt.date(2010, 12, 2)),
])
def test_to_utc_refuses_dates_before_the_first_checked_time(event: EventName, date: dt.date) -> None:
    with pytest.raises(ValueError, match="no checked release time before"):
        to_utc(date, event)


def test_release_time_tables_are_ordered_single_zone_and_exceptions_differ() -> None:
    for event, eras in RELEASE_TIMES.items():
        starts = [d for d, _ in eras]
        assert starts == sorted(set(starts))
        assert {t.tz for _, t in eras} == {release_tz(event)}
    for (event, day), t in EXCEPTIONS.items():
        assert t.tz == release_tz(event) and day >= RELEASE_TIMES[event][0][0]
        assert t != next(rt for start, rt in reversed(RELEASE_TIMES[event]) if day >= start)


def test_local_date_is_the_date_in_the_event_zone() -> None:
    instant = dt.datetime(2024, 1, 5, 2, tzinfo=dt.timezone.utc)             # 2024-01-04 21:00 in New York
    assert local_date(instant, EventName.US_NFP) == dt.date(2024, 1, 4)
    assert local_date(instant, EventName.UK_BOE_RATE_DECISION) == dt.date(2024, 1, 5)


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


# Fixtures shaped like the live pages (verified 2026-11-26): the current calendar is one panel per year with
# div.row.fomc-meeting rows; a historical page is one div.panel per meeting or action, headed by an h5.

def _cur_row(month: str, date: str, links: str = "") -> str:
    return (f'<div class="row fomc-meeting"><div class="fomc-meeting__month"><strong>{month}</strong></div>'
            f'<div class="fomc-meeting__date">{date}</div>{links}</div>')


def _cur_page(panels: dict[int, list[str]]) -> str:
    head = '<div class="panel panel-default"><div class="panel-heading"><h4><a>{} FOMC Meetings</a></h4></div>'
    body = "".join(f'{head.format(y)}{"".join(rows)}</div>' for y, rows in panels.items())
    return f"<html><body>{body}</body></html>"


def _hist_page(year: int, panels: list[tuple[str, str]]) -> str:
    body = "".join(f'<div class="panel panel-default panel-padded"><h5>{h}</h5>{links}</div>' for h, links in panels)
    return f"<html><body><h3>{year}</h3>{body}</body></html>"


def _stmt(yyyymmdd: str, title: str = "Statement") -> str:
    return f'<a href="/newsevents/pressreleases/monetary{yyyymmdd}a.htm">{title}</a>'


_PRESS = '<a href="/monetarypolicy/fomcpresconf.htm">Press Conference</a>'
_FRAMEWORK = "Statement on Longer-Run Goals and Monetary Policy Strategy"

_HIST_2020 = _hist_page(2020, [
    ("January 28-29 Meeting - 2020", _stmt("20200129")),
    ("March 2 (unscheduled) Meeting - 2020", _stmt("20200303") + _PRESS),
    ("March 17-18 (cancelled) Meeting - 2020", ""),
    ("March 19 (notation vote) - 2020", '<a href="/newsevents/pressreleases/monetary20200319b.htm">Press Release</a>'),
    ("August 27 (notation vote) - 2020", _stmt("20200827", _FRAMEWORK)),
    ("October 4 (unscheduled) - 2020", _stmt("20201011")),
    ("December 15-16 Meeting - 2020", _stmt("20201216")),
])

_HIST_2018 = _hist_page(2018, [
    ("January 31-February  1 Meeting - 2018", _stmt("20180201")),   # month named inside the range, double space
    ("June 12-13 Meeting - 2018", _stmt("20180613") + _PRESS),
    ("Jul/Aug 31-1 Meeting - 2018", _stmt("20180801")),
    ("August 9 Conference Call - 2018", _stmt("20180810")),
    ("March  4 (unscheduled) - 2018", ""),                             # call with no public statement
])


def test_fomc_historical_page_meetings_and_actions() -> None:
    meetings, actions = fomc._parse_historical_html(_HIST_2020, 2020)
    assert meetings == [(dt.date(2020, 1, 29), True, 2020), (dt.date(2020, 12, 16), True, 2020)]   # cancelled skipped
    assert [(a.action_date, a.release_date, a.action, a.has_press_conference) for a in actions] == [
        (dt.date(2020, 3, 2), dt.date(2020, 3, 3), fomc.FomcAction.UNSCHEDULED_MEETING, True),
        (dt.date(2020, 8, 27), dt.date(2020, 8, 27), fomc.FomcAction.NOTATION_VOTE, False),
        (dt.date(2020, 10, 4), dt.date(2020, 10, 11), fomc.FomcAction.UNSCHEDULED_MEETING, False),   # released later
    ]  # the 03-19 notation vote issued only a press release -> skipped
    assert actions[1].statement == _FRAMEWORK


def test_fomc_historical_page_heading_forms_and_pre_2019_press_conferences() -> None:
    meetings, actions = fomc._parse_historical_html(_HIST_2018, 2018)
    assert meetings == [(dt.date(2018, 2, 1), False, 2018), (dt.date(2018, 6, 13), True, 2018),
                        (dt.date(2018, 8, 1), False, 2018)]   # before 2019 the press-conference anchor decides
    assert [(a.action_date, a.release_date, a.action) for a in actions] == [
        (dt.date(2018, 8, 9), dt.date(2018, 8, 10), fomc.FomcAction.CONFERENCE_CALL)]


@pytest.mark.parametrize("heading, match", [
    ("March 3rd Meeting - 2020", "not recognised"),          # a meeting-like heading that does not parse
    ("March 3 (telephone poll) - 2020", "annotation"),       # an annotation we do not know
    ("March 3 - 2020", "neither"),                           # neither 'Meeting' nor an annotation
    ("March 17-18 Meeting - 2019", "on the 2020 page"),      # heading year differs from the page year
])
def test_fomc_historical_page_raises_on_unknown_heading(heading: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fomc._parse_historical_html(_hist_page(2020, [(heading, _stmt("20200303"))]), 2020)


def test_fomc_statement_link_without_dated_url_raises() -> None:
    page = _hist_page(2020, [("March 2 (unscheduled) Meeting - 2020", '<a href="/statement.htm">Statement</a>')])
    with pytest.raises(ValueError, match="dated URL"):
        fomc._parse_historical_html(page, 2020)


def test_fomc_calendar_actions_from_annotated_rows() -> None:
    framework = _stmt("20250822", _FRAMEWORK)
    html = _cur_page({2025: [
        _cur_row("January", "28-29"),
        _cur_row("August", "22 (notation vote)", framework),
        _cur_row("October", "3 (conference call)"),        # no statement -> skipped
        _cur_row("November", "4 (cancelled)"),
    ]})
    actions = fomc._parse_calendar_actions(html, 2025, 2027)
    assert [(a.action_date, a.release_date, a.action) for a in actions] == [
        (dt.date(2025, 8, 22), dt.date(2025, 8, 22), fomc.FomcAction.NOTATION_VOTE)]
    assert [d for d, _pc, _y in fomc._parse_calendar_html(html, 2025, 2027)] == [dt.date(2025, 1, 29)]


def _route(monkeypatch, pages: dict[str, str]) -> list[str]:
    seen: list[str] = []

    def fake_get(url: str):
        seen.append(url)
        return type("R", (), {"text": pages[url]})()
    monkeypatch.setattr(fomc, "get", fake_get)
    return seen


def _hist_url(year: int) -> str:
    return fomc._HISTORICAL_URL.format(year=year)


def test_fomc_fetch_reads_historical_pages_for_years_before_the_current_page(monkeypatch) -> None:
    current = _cur_page({2021: [_cur_row("January", "26-27")], 2022: [_cur_row("March", "15-16*")]})
    seen = _route(monkeypatch, {fomc._CALENDAR_URL: current, _hist_url(2020): _HIST_2020})
    df = fomc.fetch(2020, 2022)
    stmts = df.loc[df["event_name"] == EventName.US_FOMC_STATEMENT, "release_date"].tolist()
    assert stmts == [dt.date(2020, 1, 29), dt.date(2020, 12, 16), dt.date(2021, 1, 27), dt.date(2022, 3, 16)]
    assert seen == [fomc._CALENDAR_URL, _hist_url(2020)]        # 2021-22 come from the current page only


def test_fomc_fetch_raises_on_a_historical_page_without_meetings(monkeypatch) -> None:
    current = _cur_page({2021: [_cur_row("January", "26-27")]})
    _route(monkeypatch, {fomc._CALENDAR_URL: current, _hist_url(2020): _hist_page(2020, [])})
    with pytest.raises(RuntimeError, match="zero meetings"):
        fomc.fetch(2020, 2021)


def test_fomc_fetch_raises_on_a_year_missing_inside_the_current_page(monkeypatch) -> None:
    current = _cur_page({2021: [_cur_row("January", "26-27")], 2023: [_cur_row("January", "31-1")]})
    _route(monkeypatch, {fomc._CALENDAR_URL: current})
    with pytest.raises(RuntimeError, match=r"not \[2022\]"):
        fomc.fetch(2021, 2023)


@pytest.mark.parametrize("fn", [fomc.fetch, fomc.fetch_unscheduled])
def test_fomc_rejects_years_before_the_release_time_convention(monkeypatch, fn) -> None:
    monkeypatch.setattr(fomc, "get", lambda url: pytest.fail("must raise before any request"))
    with pytest.raises(ValueError, match=f">= {fomc.EARLIEST_YEAR}"):
        fn(fomc.EARLIEST_YEAR - 1, 2026)


def test_fomc_fetch_unscheduled_merges_pages_and_keeps_columns_when_empty(monkeypatch) -> None:
    panels = {y: [_cur_row("January", "26-27")] for y in range(2021, 2025)}
    panels[2025] = [_cur_row("August", "22 (notation vote)", _stmt("20250822", "Statement on Longer-Run Goals"))]
    current = _cur_page(panels)
    _route(monkeypatch, {fomc._CALENDAR_URL: current, _hist_url(2020): _HIST_2020})
    df = fomc.fetch_unscheduled(2020, 2025)
    assert list(df.columns) == fomc.UNSCHEDULED_COLUMNS
    assert df["release_date"].tolist() == [dt.date(2020, 3, 3), dt.date(2020, 8, 27), dt.date(2020, 10, 11),
                                           dt.date(2025, 8, 22)]
    assert (df["source"] == Source.FED).all()
    empty = fomc.fetch_unscheduled(2021, 2021)
    assert empty.empty and list(empty.columns) == fomc.UNSCHEDULED_COLUMNS


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


def test_ff_parse_days_maps_and_filters() -> None:
    rows = forexfactory._parse_days(forexfactory._extract_days(_FF_HTML))
    names = {r["event_name"] for r in rows}
    # The three dating names kept; the EZ statement (a checked-only name) and the unmapped US row dropped.
    assert names == {EventName.EU_ECB_RATE_DECISION, EventName.EU_ECB_PRESS_CONFERENCE, EventName.UK_BOE_RATE_DECISION}
    assert all(r["impact_tier"] == ImpactTier.HIGH for r in rows)
    sources = {r["source"] for r in rows}
    assert sources == {Source.ECB, Source.BOE}


def test_ff_parse_days_dates_decisions_by_the_rate_name_only() -> None:
    # Before August 2015 the BoE minutes (09:30) and Inflation Report (10:30) came out on their own days, filed under
    # decision-day names; neither is a decision. A rate name rated "medium" (ECB 2021) still is one.
    days = [{"events": [
        {"country": "UK", "name": "MPC Official Bank Rate Votes", "dateline": 1296034200, "impactName": "high"},
        {"country": "UK", "name": "BOE Monetary Policy Report", "dateline": 1297852200, "impactName": "high"},
        {"country": "EZ", "name": "Main Refinancing Rate", "dateline": 1615466700, "impactName": "medium"},
    ]}]
    rows = forexfactory._parse_days(days)
    assert [(r["event_name"], r["release_date"], r["impact_tier"]) for r in rows] == [
        (EventName.EU_ECB_RATE_DECISION, dt.date(2021, 3, 11), ImpactTier.HIGH)]


def test_ff_range_url_format() -> None:
    url = forexfactory._range_url(dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    assert url.endswith("range=jan1.2024-mar31.2024")


def test_ff_month_windows_cover_every_day_once() -> None:
    windows = forexfactory._month_windows(2023, 2024)
    assert len(windows) == 24
    assert windows[1] == (dt.date(2023, 2, 1), dt.date(2023, 2, 28))
    assert windows[13] == (dt.date(2024, 2, 1), dt.date(2024, 2, 29))      # leap year
    assert windows[-1] == (dt.date(2024, 12, 1), dt.date(2024, 12, 31))
    assert sum((e - s).days + 1 for s, e in windows) == 365 + 366     # contiguous, no day twice


def _ff_page(start: dt.date, end: dt.date, events: dict[dt.date, list[dict]] | None = None) -> str:
    """A ForexFactory-shaped page with one ``days`` entry per calendar day of [start, end]."""
    days = [{"date": f"{d:%b} {d.day}", "events": (events or {}).get(d, [])} for d in pd.date_range(start, end).date]
    return f"<script>window.calendarComponentStates[1] = {{\n  days: {json.dumps(days)},\n  more: 1\n}};</script>"


def test_ff_fetch_requests_one_month_each(monkeypatch) -> None:
    ecb = {"country": "EZ", "name": "Main Refinancing Rate", "dateline": 1709213400, "impactName": "high"}  # 02-29
    seen: list[str] = []

    def fake_get(url: str):
        seen.append(url)
        start, end = next(w for w in forexfactory._month_windows(2024, 2024) if forexfactory._range_url(*w) == url)
        return type("R", (), {"text": _ff_page(start, end, {dt.date(2024, 2, 29): [ecb]})})()
    monkeypatch.setattr(forexfactory, "get", fake_get)
    monkeypatch.setattr(forexfactory, "_PAUSE_SECONDS", 0.0)
    df = forexfactory.fetch(2024, 2024)
    assert len(seen) == 12 and seen[2].endswith("range=mar1.2024-mar31.2024")
    assert df["release_date"].tolist() == [dt.date(2024, 2, 29)]


def test_ff_fetch_raises_on_a_truncated_month(monkeypatch) -> None:
    # The live failure: a range answered with fewer days than asked for, and no error.
    short = _ff_page(dt.date(2024, 1, 1), dt.date(2024, 1, 30))
    monkeypatch.setattr(forexfactory, "get", lambda url: type("R", (), {"text": short})())
    with pytest.raises(RuntimeError, match="30 days returned for 2024-01-01 -> 2024-01-31, expected 31"):
        forexfactory.fetch(2024, 2024)


def test_ff_fetch_with_recorded_returns_the_us_instants_too(monkeypatch) -> None:
    nfp = {"country": "US", "name": "Non-Farm Employment Change", "dateline": 1704461400, "impactName": "high"}
    boe = {"country": "UK", "name": "Official Bank Rate", "dateline": 1706788800, "impactName": "high"}
    events = {dt.date(2024, 1, 5): [nfp], dt.date(2024, 2, 1): [boe]}

    def fake_get(url: str):
        start, end = next(w for w in forexfactory._month_windows(2024, 2024) if forexfactory._range_url(*w) == url)
        return type("R", (), {"text": _ff_page(start, end, events)})()
    monkeypatch.setattr(forexfactory, "get", fake_get)
    monkeypatch.setattr(forexfactory, "_PAUSE_SECONDS", 0.0)
    rows, recorded = forexfactory.fetch_with_recorded(2024, 2024)
    assert rows["event_name"].tolist() == [EventName.UK_BOE_RATE_DECISION]          # calendar rows: ECB/BoE only
    assert recorded.to_dict("records") == [
        {"event_name": "US_NFP", "release_date": dt.date(2024, 1, 5),
         "recorded_utc": pd.Timestamp("2024-01-05 13:30", tz="UTC")},
        {"event_name": "UK_BOE_RATE_DECISION", "release_date": dt.date(2024, 2, 1),
         "recorded_utc": pd.Timestamp("2024-02-01 12:00", tz="UTC")},
    ]


def _pairs(rows: list[tuple[str, dt.date, str]]) -> pd.DataFrame:
    """``(event, release_date, recorded UTC)`` -> ``check_recorded_times`` input, stamped by ``release_times``."""
    return pd.DataFrame([{"event_name": e, "release_date": d, "timestamp_utc": pd.Timestamp(to_utc(d, EventName(e))),
                          "recorded_utc": pd.Timestamp(r, tz="UTC")} for e, d, r in rows])


_NOW = pd.Timestamp("2026-11-26", tz="UTC")


def test_ff_check_recorded_times_accepts_agreement_and_minute_offsets() -> None:
    forexfactory.check_recorded_times(_pairs([
        ("US_NFP", dt.date(2025, 12, 16), "2025-12-16 13:29"),               # older of two same-instant releases
        ("US_NFP", dt.date(2025, 12, 16), "2025-12-16 13:30"),
        ("EU_ECB_RATE_DECISION", dt.date(2021, 6, 10), "2021-06-10 11:45"),  # 13:45 CEST, before the 2022 change
        ("US_PCE", dt.date(2019, 11, 27), "2019-11-27 15:00"),               # a 10:00 ET exception
    ]), now=_NOW)


def test_ff_check_recorded_times_raises_on_a_disagreement() -> None:
    late = _pairs([("US_PCE", dt.date(2024, 6, 28), "2024-06-28 14:00")])
    with pytest.raises(ValueError, match=r"US_PCE 2024-06-28: stamped 08:30, ForexFactory recorded 10:00 "
                                         r"America/New_York \(\+90 min\)"):
        forexfactory.check_recorded_times(late, now=_NOW)


def test_ff_check_recorded_times_allows_the_scheduled_dst_gap_only() -> None:
    # 2027-03-18: New York is on summer time, London not yet. ForexFactory lists the scheduled BoE decision an hour
    # early until it has happened.
    early = _pairs([("UK_BOE_RATE_DECISION", dt.date(2027, 3, 18), "2027-03-18 11:00")])
    forexfactory.check_recorded_times(early, now=_NOW)
    with pytest.raises(ValueError):                                          # released: the instant must be right
        forexfactory.check_recorded_times(early, now=pd.Timestamp("2027-03-19", tz="UTC"))
    with pytest.raises(ValueError):                                          # both on summer time: no gap
        forexfactory.check_recorded_times(_pairs([("UK_BOE_RATE_DECISION", dt.date(2027, 4, 29), "2027-04-29 10:00")]),
                                          now=_NOW)
    with pytest.raises(ValueError):                                          # a US event never has the gap
        forexfactory.check_recorded_times(_pairs([("US_CPI", dt.date(2027, 3, 10), "2027-03-10 12:30")]), now=_NOW)


def test_ff_check_recorded_times_skips_a_listed_forexfactory_error() -> None:
    forexfactory.check_recorded_times(_pairs([("US_PPI", dt.date(2021, 4, 9), "2021-04-09 12:56")]), now=_NOW)


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

def _calendar_rows(counts: dict[str, int], years: list[int]) -> pd.DataFrame:
    """``counts[event]`` rows of each event in each year, dated mid-year."""
    rows = [(name, dt.date(y, 6, 1)) for y in years for name, n in counts.items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["event_name", "release_date"])


_HEALTHY = {"US_FOMC_STATEMENT": 8, "US_NFP": 12, "US_CPI": 12, "EU_ECB_RATE_DECISION": 8, "UK_BOE_RATE_DECISION": 8}


def test_build_validate_coverage_raises_on_partial_scrape() -> None:
    final = _calendar_rows({"US_FOMC_STATEMENT": 1}, [2024])  # 1 row for a 3-year window
    with pytest.raises(ValueError, match="coverage check failed"):
        build_mod._validate_coverage(final, 2024, 2026)


def test_build_validate_coverage_passes_on_healthy_counts() -> None:
    build_mod._validate_coverage(_calendar_rows(_HEALTHY, [2024]), 2024, 2024)  # 1 year — must not raise


def test_build_validate_coverage_counts_only_completed_years() -> None:
    # 2025-2027 evaluated in 2026: only 2025 is fully completed, so the floor is one year's worth.
    # One year of counts clears it even though a naive 3-year floor (e.g. NFP >= 27) would reject —
    # this is what stops the partial current/future year from false-tripping the guard.
    counts = {"US_FOMC_STATEMENT": 8, "US_NFP": 10, "US_CPI": 10, "EU_ECB_RATE_DECISION": 6, "UK_BOE_RATE_DECISION": 6}
    build_mod._validate_coverage(_calendar_rows(counts, [2025]), 2025, 2027, today_year=2026)


def test_build_validate_coverage_raises_on_a_missing_year_hidden_by_the_window_total() -> None:
    # 2019-2026 evaluated in 2026: FOMC present only for 2021-2026 gives 48 statements, above the old total floor of
    # 6 x 7 completed years = 42. The per-year check must still name the two missing years.
    others = {k: v for k, v in _HEALTHY.items() if k != "US_FOMC_STATEMENT"}
    fomc_rows = _calendar_rows({"US_FOMC_STATEMENT": 8}, list(range(2021, 2027)))
    final = pd.concat([_calendar_rows(others, list(range(2019, 2027))), fomc_rows], ignore_index=True)
    with pytest.raises(ValueError, match=r"US_FOMC_STATEMENT 2019: 0 < 6[\s\S]*US_FOMC_STATEMENT 2020: 0 < 6"):
        build_mod._validate_coverage(final, 2019, 2026, today_year=2026)
    build_mod._validate_coverage(_calendar_rows(_HEALTHY, list(range(2019, 2027))), 2019, 2026, today_year=2026)


def _stub_fetchers(monkeypatch, rows: pd.DataFrame, recorded: pd.DataFrame) -> None:
    """``build`` on ``rows`` (served by the FRED fetcher) and ForexFactory instants ``recorded``, coverage unchecked."""
    monkeypatch.setattr(build_mod.fomc, "fetch", lambda a, b: rows.iloc[:0])
    monkeypatch.setattr(build_mod.fred, "fetch", lambda a, b: rows)
    monkeypatch.setattr(build_mod.forexfactory, "fetch_with_recorded", lambda a, b: (rows.iloc[:0], recorded))
    monkeypatch.setattr(build_mod, "_validate_coverage", lambda *a, **k: None)


def _pce_rows(*days: dt.date) -> pd.DataFrame:
    return pd.DataFrame([{"release_date": d, "event_name": EventName.US_PCE, "source": Source.BEA,
                          "impact_tier": ImpactTier.HIGH} for d in days])


def _recorded(*rows: tuple[str, dt.date, str]) -> pd.DataFrame:
    return pd.DataFrame([{"event_name": e, "release_date": d, "recorded_utc": pd.Timestamp(r, tz="UTC")}
                         for e, d, r in rows])


def test_build_stamps_exceptions_and_checks_them(monkeypatch, tmp_path) -> None:
    days = (dt.date(2019, 11, 27), dt.date(2019, 12, 20), dt.date(2024, 6, 28))
    _stub_fetchers(monkeypatch, _pce_rows(*days), _recorded(("US_PCE", days[0], "2019-11-27 15:00"),
                                                            ("US_PCE", days[2], "2024-06-28 12:30")))
    out = build_mod.build(2019, 2024, tmp_path / "cal.parquet")
    assert out["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M").tolist() == ["2019-11-27 15:00", "2019-12-20 15:00",
                                                                             "2024-06-28 12:30"]
    assert (tmp_path / "cal.parquet").exists()


def test_build_refuses_to_write_when_a_release_time_disagrees(monkeypatch, tmp_path) -> None:
    day = dt.date(2024, 6, 28)
    _stub_fetchers(monkeypatch, _pce_rows(day), _recorded(("US_PCE", day, "2024-06-28 14:00")))
    with pytest.raises(ValueError, match="US_PCE 2024-06-28: stamped 08:30"):
        build_mod.build(2024, 2024, tmp_path / "cal.parquet")
    assert not (tmp_path / "cal.parquet").exists()


def test_default_years_track_today() -> None:
    y = dt.date.today().year
    assert build_mod._default_years() == (y - 1, y + 1)  # rolls forward with the calendar
