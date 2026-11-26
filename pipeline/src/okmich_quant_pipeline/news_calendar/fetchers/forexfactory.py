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
Requested a month at a time, ForexFactory lists every ECB and BoE rate decision as high impact (checked 2011-2026 on
2026-11-26):
- ECB: 12 a year through 2014, 8 a year from 2015.
- BoE: 12 a year through 2015, 8 a year from 2017, and 10 in 2020 with the two March emergency cuts.

Timestamps
----------
``dateline`` is the real release instant for past events:
- ECB decisions: 13:45 Frankfurt through the 2022-06-09 meeting, 14:15 from 2022-07-21.
- BoE decisions: 12:00 London, except 2020-03-11, 05-07, 08-06 and 11-05 at 07:00 and 2020-03-19 at 14:30.

A *future* event fetched before a DST change can be an hour off: the ECB 2026-10-29 decision, fetched in June, read
12:15 UTC against the true 13:15. ``build.py`` stamps events by ``release_times.py`` and ignores ``dateline``.

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

import pandas as pd

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source

_BASE_URL = "https://www.forexfactory.com/calendar?range={start}-{end}"
_PAUSE_SECONDS = 1.0     # between consecutive month requests

# (country, event_name) -> our canonical EventName. The country code is FF's
# (EZ for eurozone, UK for united kingdom).
_NAME_MAP: dict[tuple[str, str], EventName] = {
    ("EZ", "Main Refinancing Rate"): EventName.EU_ECB_RATE_DECISION,
    ("EZ", "Monetary Policy Statement"): EventName.EU_ECB_RATE_DECISION,
    ("EZ", "ECB Press Conference"): EventName.EU_ECB_PRESS_CONFERENCE,
    ("UK", "Official Bank Rate"): EventName.UK_BOE_RATE_DECISION,
    ("UK", "Monetary Policy Summary"): EventName.UK_BOE_RATE_DECISION,
    ("UK", "MPC Official Bank Rate Votes"): EventName.UK_BOE_RATE_DECISION,
    ("UK", "BOE Monetary Policy Report"): EventName.UK_BOE_RATE_DECISION,
}

# Country -> publishing source. FF aggregates from the central banks themselves.
_SOURCE_MAP: dict[str, Source] = {
    "EZ": Source.ECB,
    "UK": Source.BOE,
}

# FF's impact strings -> our ImpactTier. We only act on "high" (mask events),
# but parse all three so future consumers can widen the filter.
_IMPACT_MAP: dict[str, ImpactTier] = {
    "high": ImpactTier.HIGH,
    "medium": ImpactTier.MEDIUM,
    "low": ImpactTier.LOW,
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
    """Matched high-impact ECB/BoE rows from one window's ``days`` array."""
    rows: list[dict] = []
    for day in days:
        for ev in day.get("events", []):
            country = ev.get("country")
            name = ev.get("name")
            key = (country, name)
            if key not in _NAME_MAP:
                continue
            ts = ev.get("dateline")
            if ts is None:
                continue
            # Filter to high-impact (FF marks every captured ECB/BoE rate
            # decision as high; non-high entries in this name set would be
            # something we'd want to inspect manually).
            impact = _IMPACT_MAP.get(ev.get("impactName", ""), ImpactTier.LOW)
            if impact != ImpactTier.HIGH:
                continue
            release_date = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
            rows.append({
                "release_date": release_date,
                "event_name": _NAME_MAP[key],
                "source": _SOURCE_MAP[country],
                "impact_tier": impact,
            })
    return rows


def fetch(year_min: int, year_max: int) -> pd.DataFrame:
    """Return a DataFrame of ECB + BoE high-impact meeting dates in
    ``[year_min, year_max]``.

    Polls ForexFactory one month at a time (see module docstring). Same release_date can appear twice per bank
    (e.g. "Main Refinancing Rate" + "Monetary Policy Statement" both map to EU_ECB_RATE_DECISION) — the orchestrator
    deduplicates on (release_date, event_name).
    """
    rows: list[dict] = []
    for _start, days in _month_days(year_min, year_max):
        rows.extend(_parse_days(days))
    if not rows:
        raise RuntimeError(
            f"FF fetcher: parsed zero matching events in [{year_min}, {year_max}]. "
            f"Either FF coverage changed or _NAME_MAP needs an update."
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["release_date", "event_name"])
    return df.sort_values(["release_date", "event_name"]).reset_index(drop=True)


if __name__ == "__main__":
    df = fetch(2024, 2026)
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows; by event_name:")
    print(df.event_name.value_counts())
