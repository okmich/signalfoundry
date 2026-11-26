"""ForexFactory fetcher — ECB + BoE rate decisions.

Why ForexFactory for these two
------------------------------
The official ECB and BoE calendar pages are heavily JavaScript-rendered: direct HTML scraping yields only news-feed
dates, not the meeting calendar. FRED tracks the rates as daily series, not as meeting-keyed releases. ForexFactory
embeds a clean JSON event blob in the page HTML, no JS required, and supports range queries like
``?range=jan1.2025-jan31.2025``.

One month per request
---------------------
ForexFactory answers a range query with at most about two months and raises no error. ``jan1.2024-mar31.2024``
returns Jan 1 -> Mar 2 (62 days), checked 2026-11-26. Quarter-sized requests therefore dropped every March, June,
September and December. ``_month_days`` asks for one calendar month per request and raises unless the reply holds one
entry per calendar day. Requests are paced by ``_PAUSE_SECONDS``.

Coverage
--------
Requested a month at a time, ForexFactory lists every ECB and BoE rate decision (checked 2011-2027 on 2026-11-26):
- ECB: 12 a year through 2014, 8 a year from 2015.
- BoE: 12 a year through 2015, 11 in 2016, 8 a year from 2017, and 10 in 2020 with the two March emergency cuts.

Each calendar event is dated by one name (``_NAME_MAP``): "Main Refinancing Rate", "ECB Press Conference" and
"Official Bank Rate". The BoE's other decision-day names ("MPC Official Bank Rate Votes", "BOE Monetary Policy Report")
fell on other days before August 2015, when the minutes came out two weeks after the decision at 09:30 and the
Inflation Report on its own day at 10:30. Dating by them added 73 false BoE decision days in 2011-2015. Those names
and the ECB's "Monetary Policy Statement" are only checked against the decision's time (``_RECORDED_NAME_MAP``).

ForexFactory's impact label is not used. It changes from year to year (it rated 7 of the 8 ECB decisions in 2021
"medium"), and a rate decision is a high-impact event by definition, as every other calendar event is.

Timestamps
----------
``dateline`` is the instant ForexFactory recorded. Both builders stamp rows from ``release_times.py`` and use
``dateline`` only to check that table (``check_recorded_times``). Its faults, checked 2011-2026 on 2026-11-26:
- Two releases of one event at one instant (a month delayed by a US government shutdown) are listed a minute or two
  apart, the older first: NFP 2025-12-16 at 08:29 and 08:30 ET. A few other rows are a minute or two off.
- A scheduled ECB or BoE event that falls while New York and Europe disagree on daylight saving (the US changes clocks
  2-3 weeks before Europe in March and one week after it in autumn) reads an hour early until it has happened: the
  2027-03-18 decisions read 11:00 / 12:15 UTC against the true 12:00 / 13:15 (fetched 2026-11-26). Released events
  carry the corrected instant.
- ``_RECORDED_ERRORS`` lists the recorded instants that are wrong.

Data shape
----------
FF embeds the calendar as a JS object on the page:

    window.calendarComponentStates[1] = {
        days: [{"date": "...", "dateline": 1709244000, "events": [...]}],
        ...
    }

The outer object uses unquoted JS keys (``days:``) but ``days`` itself is a valid JSON array. We brace-match the array
and ``json.loads`` it.

Each event carries:
- ``country``    : two-letter code (FF uses "EZ" for Eurozone, "UK" for United Kingdom — not "EUR"/"GBP" or "DE"/"GB")
- ``name``       : event name (e.g. "Main Refinancing Rate", "Official Bank Rate", "ECB Press Conference")
- ``dateline``   : UNIX seconds UTC
- ``impactName`` : "high" | "medium" | "low"
"""
from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Iterator
from zoneinfo import ZoneInfo

import pandas as pd

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source
from okmich_quant_pipeline.news_calendar.release_times import local_date, release_tz

_BASE_URL = "https://www.forexfactory.com/calendar?range={start}-{end}"
_PAUSE_SECONDS = 1.0     # between consecutive month requests

# (country, event_name) -> our canonical EventName: the one name that dates each calendar event (module docstring).
# The country code is FF's (EZ for eurozone, UK for united kingdom).
_NAME_MAP: dict[tuple[str, str], EventName] = {
    ("EZ", "Main Refinancing Rate"): EventName.EU_ECB_RATE_DECISION,
    ("EZ", "ECB Press Conference"): EventName.EU_ECB_PRESS_CONFERENCE,
    ("UK", "Official Bank Rate"): EventName.UK_BOE_RATE_DECISION,
}

# Names ForexFactory lists with the decision on decision days. They never date a row: before August 2015 the BoE ones
# fell on minutes and Inflation Report days.
_DECISION_DAY_NAMES: dict[tuple[str, str], EventName] = {
    ("EZ", "Monetary Policy Statement"): EventName.EU_ECB_RATE_DECISION,
    ("UK", "Monetary Policy Summary"): EventName.UK_BOE_RATE_DECISION,
    ("UK", "MPC Official Bank Rate Votes"): EventName.UK_BOE_RATE_DECISION,
    ("UK", "BOE Monetary Policy Report"): EventName.UK_BOE_RATE_DECISION,
}

# Every name whose recorded instant ``check_recorded_times`` compares with ``release_times``, joined on the calendar's
# (event, date). The ECB/BoE calendar rows themselves come from ``_NAME_MAP``.
_RECORDED_NAME_MAP: dict[tuple[str, str], EventName] = {
    **_NAME_MAP,
    **_DECISION_DAY_NAMES,
    ("US", "FOMC Statement"): EventName.US_FOMC_STATEMENT,
    ("US", "Federal Funds Rate"): EventName.US_FOMC_STATEMENT,
    ("US", "FOMC Press Conference"): EventName.US_FOMC_PRESS_CONFERENCE,
    ("US", "Non-Farm Employment Change"): EventName.US_NFP,
    ("US", "CPI m/m"): EventName.US_CPI,
    ("US", "PPI m/m"): EventName.US_PPI,
    ("US", "Advance GDP q/q"): EventName.US_GDP,
    ("US", "Prelim GDP q/q"): EventName.US_GDP,
    ("US", "Final GDP q/q"): EventName.US_GDP,
    ("US", "Core PCE Price Index m/m"): EventName.US_PCE,
    ("US", "Retail Sales m/m"): EventName.US_RETAIL_SALES,
}

# Recorded instants that are wrong, with the checked release time.
_RECORDED_ERRORS: dict[tuple[EventName, dt.date], str] = {
    (EventName.US_PPI, dt.date(2021, 4, 9)): "recorded 08:56 ET; BLS released it at 08:30 (its release schedule)",
}

# A recorded instant this close to the stamped one agrees with it (the minute offsets in the module docstring).
_TOLERANCE = pd.Timedelta(minutes=5)
_NEW_YORK = "America/New_York"

# Country -> publishing source. FF aggregates from the central banks themselves.
_SOURCE_MAP: dict[str, Source] = {
    "EZ": Source.ECB,
    "UK": Source.BOE,
}

# Month abbreviations used by FF's URL convention (e.g. "jan1.2024").
_FF_MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _extract_days(text: str) -> list[dict]:
    """Extract the ``days`` array from the JS-wrapped calendar blob.

    The outer ``window.calendarComponentStates[1]`` object uses unquoted JS
    keys, but ``days`` itself is valid JSON. We brace-match the array and
    parse it directly.
    """
    needle = "window.calendarComponentStates[1] = {"
    anchor = text.find(needle)
    if anchor < 0:
        raise RuntimeError("FF fetcher: calendar blob not found in page HTML")
    days_marker = text.find("days:", anchor)
    if days_marker < 0:
        raise RuntimeError("FF fetcher: 'days:' key not found in calendar blob")
    start = text.find("[", days_marker)
    if start < 0:
        raise RuntimeError("FF fetcher: opening '[' for days array not found")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise RuntimeError("FF fetcher: unterminated days array — page HTML truncated?")


def _range_url(start: dt.date, end: dt.date) -> str:
    """Build the ?range= URL for a [start, end] window (FF format: 'jan1.2024')."""
    def _fmt(d: dt.date) -> str:
        return f"{_FF_MONTH_ABBR[d.month - 1]}{d.day}.{d.year}"
    return _BASE_URL.format(start=_fmt(start), end=_fmt(end))


def _month_windows(year_min: int, year_max: int) -> list[tuple[dt.date, dt.date]]:
    """(first day, last day) of every calendar month from ``year_min`` January to ``year_max`` December."""
    windows: list[tuple[dt.date, dt.date]] = []
    for y in range(year_min, year_max + 1):
        for m in range(1, 13):
            start = dt.date(y, m, 1)
            end = (dt.date(y + 1, 1, 1) if m == 12 else dt.date(y, m + 1, 1)) - dt.timedelta(days=1)
            windows.append((start, end))
    return windows


def _check_full_window(days: list[dict], start: dt.date, end: dt.date) -> None:
    """Raise unless ForexFactory returned one entry per calendar day of ``[start, end]`` (a truncated reply carries no
    error of its own)."""
    expected = (end - start).days + 1
    if len(days) != expected:
        raise RuntimeError(f"FF fetcher: {len(days)} days returned for {start} -> {end}, expected {expected}. "
                           f"ForexFactory truncated the range or changed its page.")


def _month_days(year_min: int, year_max: int) -> Iterator[tuple[dt.date, list[dict]]]:
    """(month start, verified ``days`` array) for every month of ``[year_min, year_max]``, one paced request each."""
    for i, (start, end) in enumerate(_month_windows(year_min, year_max)):
        if i:
            time.sleep(_PAUSE_SECONDS)
        days = _extract_days(get(_range_url(start, end)).text)
        _check_full_window(days, start, end)
        yield start, days


def _parse_days(days: list[dict]) -> list[dict]:
    """ECB/BoE calendar rows (``_NAME_MAP`` names, any ForexFactory impact label) from one window's ``days`` array."""
    rows: list[dict] = []
    for day in days:
        for ev in day.get("events", []):
            country = ev.get("country")
            key = (country, ev.get("name"))
            ts = ev.get("dateline")
            if key not in _NAME_MAP or ts is None:
                continue
            event = _NAME_MAP[key]
            rows.append({
                "release_date": local_date(dt.datetime.fromtimestamp(ts, dt.timezone.utc), event),
                "event_name": event,
                "source": _SOURCE_MAP[country],
                "impact_tier": ImpactTier.HIGH,
            })
    return rows


def _recorded_days(days: list[dict]) -> list[dict]:
    """``(event_name, release_date, recorded_utc)`` for every ``_RECORDED_NAME_MAP`` event in one window."""
    rows: list[dict] = []
    for day in days:
        for ev in day.get("events", []):
            event = _RECORDED_NAME_MAP.get((ev.get("country"), ev.get("name")))
            ts = ev.get("dateline")
            if event is None or ts is None:
                continue
            recorded = pd.Timestamp(ts, unit="s", tz="UTC")
            rows.append({"event_name": event.value, "release_date": local_date(recorded, event),
                         "recorded_utc": recorded})
    return rows


def _dst_disagrees(day: dt.date, tz: str) -> bool:
    """True when ``tz`` and New York differ on daylight saving at noon on ``day``."""
    noon = dt.datetime(day.year, day.month, day.day, 12)
    return bool(noon.replace(tzinfo=ZoneInfo(tz)).dst()) != bool(noon.replace(tzinfo=ZoneInfo(_NEW_YORK)).dst())


def _known_fault(event: EventName, release_date: dt.date, stamped: pd.Timestamp, gap: pd.Timedelta,
                 now: pd.Timestamp) -> bool:
    """True when a disagreement is one of ForexFactory's faults in the module docstring."""
    if (event, release_date) in _RECORDED_ERRORS:
        return True
    tz = release_tz(event)
    hour_early = abs(gap + pd.Timedelta(hours=1)) <= _TOLERANCE
    return stamped > now and hour_early and tz != _NEW_YORK and _dst_disagrees(release_date, tz)


def check_recorded_times(pairs: pd.DataFrame, *, now: pd.Timestamp | None = None) -> None:
    """Raise unless every stamped row agrees with the instant ForexFactory recorded for it.

    ``pairs`` holds ``event_name``, ``release_date``, ``timestamp_utc`` (stamped by ``release_times``) and
    ``recorded_utc`` (``dateline``), one row per recorded instant. A row agrees when the two are within ``_TOLERANCE``
    or the gap is a known ForexFactory fault. ``now`` (default: the current time) separates released events from
    scheduled ones.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    bad: list[str] = []
    for r in pairs.itertuples(index=False):
        gap = r.recorded_utc - r.timestamp_utc
        event = EventName(r.event_name)
        if abs(gap) <= _TOLERANCE or _known_fault(event, r.release_date, r.timestamp_utc, gap, now):
            continue
        tz = ZoneInfo(release_tz(event))
        bad.append(f"{event.value} {r.release_date}: stamped {r.timestamp_utc.astimezone(tz):%H:%M}, ForexFactory "
                   f"recorded {r.recorded_utc.astimezone(tz):%H:%M} {tz.key} ({gap.total_seconds() / 60:+.0f} min)")
    if bad:
        raise ValueError("release times disagree with ForexFactory's recorded instants. Check each release with its "
                         "publisher, then add it to release_times.EXCEPTIONS (or RELEASE_TIMES for a lasting change), "
                         "or to forexfactory._RECORDED_ERRORS if ForexFactory is wrong:\n  " + "\n  ".join(bad))


def fetch_with_recorded(year_min: int, year_max: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ECB + BoE high-impact meeting dates in ``[year_min, year_max]``, and the instants ForexFactory recorded.

    The first frame is ``fetch``'s. The second holds ``event_name`` (str), ``release_date`` and ``recorded_utc`` for
    every ``_RECORDED_NAME_MAP`` event, US ones included, for ``check_recorded_times``. Both come from the same
    month-by-month requests.
    """
    rows: list[dict] = []
    recorded: list[dict] = []
    for _start, days in _month_days(year_min, year_max):
        rows.extend(_parse_days(days))
        recorded.extend(_recorded_days(days))
    if not rows:
        raise RuntimeError(
            f"FF fetcher: parsed zero matching events in [{year_min}, {year_max}]. "
            f"Either FF coverage changed or _NAME_MAP needs an update."
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["release_date", "event_name"])
    df = df.sort_values(["release_date", "event_name"]).reset_index(drop=True)
    instants = pd.DataFrame(recorded, columns=["event_name", "release_date", "recorded_utc"]).drop_duplicates()
    return df, instants.reset_index(drop=True)


def fetch(year_min: int, year_max: int) -> pd.DataFrame:
    """Return a DataFrame of ECB + BoE high-impact meeting dates in
    ``[year_min, year_max]``.

    Polls ForexFactory one month at a time (see module docstring); one row per (release_date, event_name).
    """
    return fetch_with_recorded(year_min, year_max)[0]


if __name__ == "__main__":
    df = fetch(2024, 2026)
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows; by event_name:")
    print(df.event_name.value_counts())
