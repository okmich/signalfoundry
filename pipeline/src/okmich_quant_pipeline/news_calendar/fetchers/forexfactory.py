"""ForexFactory fetcher — ECB + BoE rate decisions.

Why ForexFactory for these two
------------------------------
The official ECB and BoE calendar pages are heavily JavaScript-rendered: direct HTML scraping yields only news-feed
dates, not the meeting calendar. FRED tracks the rates as daily series, not as meeting-keyed releases. ForexFactory
embeds a clean JSON event blob in the page HTML, no JS required, and supports range queries like
``?range=jan1.2025-mar31.2025`` so the whole 3-year window fits in a dozen polite HTTP calls.

Coverage gap (documented)
-------------------------
ForexFactory only flags the 3-4 highest-impact ECB/BoE meetings per year (the SEP/MPR projection meetings). T
he remaining 4-5 interim meetings per year per bank are **not** in FF — not even at medium or low impact. For the
Week 1 XAUUSD news mask this gap is ~60 unmasked bars per year against ~75k bars total (well below the noise floor of
the entropy stats Block B computes). If a downstream consumer needs full ECB/BoE coverage, that's a separate fetcher
(Playwright on the official sites) — out of scope here.

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

import pandas as pd

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source

_BASE_URL = "https://www.forexfactory.com/calendar?range={start}-{end}"

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


def _quarter_windows(year_min: int, year_max: int) -> list[tuple[dt.date, dt.date]]:
    """Yield (start, end) windows spanning [year_min Jan 1, year_max Dec 31] in
    quarter-sized chunks. Quarters are large enough to keep the call count
    low (12 per 3 years) and small enough that FF doesn't truncate."""
    windows: list[tuple[dt.date, dt.date]] = []
    for y in range(year_min, year_max + 1):
        for q in range(4):
            start_month = q * 3 + 1
            end_month = start_month + 2
            start = dt.date(y, start_month, 1)
            # Last day of end_month
            if end_month == 12:
                end = dt.date(y, 12, 31)
            else:
                end = dt.date(y, end_month + 1, 1) - dt.timedelta(days=1)
            windows.append((start, end))
    return windows


def _parse_quarter(html: str) -> list[dict]:
    """Parse one quarter's HTML and return matched-event rows."""
    days = _extract_days(html)
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

    Polls ForexFactory one quarter at a time. Same release_date can appear twice per bank
    (e.g. "Main Refinancing Rate" + "Monetary Policy Statement" both map to EU_ECB_RATE_DECISION) — the orchestrator
    deduplicates on (release_date, event_name).
    """
    rows: list[dict] = []
    for start, end in _quarter_windows(year_min, year_max):
        url = _range_url(start, end)
        resp = get(url)
        rows.extend(_parse_quarter(resp.text))
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
