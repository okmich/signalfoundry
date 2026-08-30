"""FOMC meeting fetcher — federalreserve.gov.

Source pages
------------
- Current + future (~5 years): https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- Historical (>5y old):        https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm
                               (one page per year)

Page structure
--------------
The current calendars page renders one ``<div class="panel panel-default">`` per year. Inside each panel a table lists
rows of meeting dates. Each row has columns: Month, Meeting (date range), Statement (PDF link), etc. A two-day meeting
renders as e.g. "31-1" with the month spanning two months in the "Month" cell ("April/May"). One-day or
single-day-displayed meetings render as e.g. "17-18" within a single month.

The fetcher returns one row per meeting, keyed on the **second** (policy statement) day — that's the day the rate
decision and press conference land on the chart.

Press conferences: since 2019, every FOMC meeting has had a press conference (Powell-era policy). The page only adds the
"Press Conference" anchor *after* the meeting concludes, so a presence-check would under-mask future meetings as the
calendar rolls forward. We therefore emit a press-conf row unconditionally for every meeting at/after 2019, and use
anchor-presence as a fallback for older meetings. The asterisk "*" in the date cell marks SEP meetings (quarterly, with
a Summary of Economic Projections) — NOT press-conference meetings.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

import pandas as pd
from bs4 import BeautifulSoup

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source

logger = logging.getLogger(__name__)

_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# Map English month names to integers. We accept three-letter abbreviations
# too (the Fed sometimes uses "Jan/Feb" or "January/February").
_MONTH_LOOKUP: dict[str, int] = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"], start=1)}
for _m, _i in list(_MONTH_LOOKUP.items()):
    _MONTH_LOOKUP[_m[:3]] = _i


def _month_to_int(token: str) -> int:
    key = token.strip().lower()
    if key not in _MONTH_LOOKUP:
        raise ValueError(f"Unrecognised month token: {token!r}")
    return _MONTH_LOOKUP[key]


def _parse_meeting_row(year: int, month_cell: str, date_cell: str) -> dt.date:
    """Parse one meeting row → statement_date.

    ``month_cell`` is like "January", "April/May", or "Apr/May".
    ``date_cell`` is like "30-31" or "30-1" or "30-31*" (asterisk marks SEP
    meetings and is stripped here).

    Statement day is always the *last* day of the meeting. For "30-1" with month "April/May" the statement is May 1; for
    "30-31" the statement is the same month.
    """
    cleaned = date_cell.replace("*", "").strip()
    # The dash can be a hyphen or an en-dash depending on the page.
    parts = re.split(r"[-–]", cleaned)
    if len(parts) == 1:
        # Single-day meeting (rare).
        day_first = day_last = int(parts[0])
    elif len(parts) == 2:
        day_first, day_last = int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Unexpected date cell shape: {date_cell!r}")

    # Decide which month the statement (last) day belongs to.
    month_tokens = [t for t in re.split(r"[\\/]+", month_cell) if t.strip()]
    if len(month_tokens) == 1:
        statement_month = _month_to_int(month_tokens[0])
        statement_year = year
    else:
        # Cross-month meeting like "April/May". The first day belongs to
        # the first token, the last day to the second token.
        statement_month = _month_to_int(month_tokens[-1])
        # Cross-year (Dec/Jan) is theoretically possible; handle it.
        first_month = _month_to_int(month_tokens[0])
        statement_year = year + 1 if statement_month < first_month else year

    return dt.date(statement_year, statement_month, day_last)


def _parse_calendar_html(html: str, year_min: int, year_max: int) -> list[tuple[dt.date, bool, int]]:
    """Parse the Fed calendars HTML → list of (statement_date, has_press_conf, year).

    Structure (verified 2026-05-25):
        div.panel.panel-default
          div.panel-heading
            h4 > a  -- text is e.g. "2026 FOMC Meetings"
          div.row.fomc-meeting              ── one meeting per row (sibling of panel-heading)
            div.fomc-meeting__month         ── "<strong>January</strong>"
            div.fomc-meeting__date          ── "27-28" or "17-18*" or "30-1"
            (other cells: Statement, Press Conf, Minutes)
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[tuple[dt.date, bool, int]] = []
    errors: list[str] = []
    skipped: list[str] = []

    for panel in soup.select("div.panel.panel-default"):
        header = panel.find("h4")
        if header is None:
            continue
        m = re.search(r"(20\d{2})", header.get_text())
        if not m:
            continue
        year = int(m.group(1))
        if year < year_min or year > year_max:
            continue
        # Each meeting is a div whose class list contains both 'row' and 'fomc-meeting'.
        for row in panel.find_all("div", class_=lambda cs: cs is not None and "fomc-meeting" in cs and "row" in cs):
            month_el = row.find("div", class_=lambda cs: cs is not None and "fomc-meeting__month" in cs)
            date_el = row.find("div", class_=lambda cs: cs is not None and "fomc-meeting__date" in cs)
            if month_el is None or date_el is None:
                continue
            month_cell = month_el.get_text(" ", strip=True)
            date_cell = date_el.get_text(" ", strip=True)
            if not month_cell or not date_cell or not re.search(r"\d", date_cell):
                continue
            # A parenthetical annotation (e.g. "22 (notation vote)", "(unscheduled)",
            # "(conference call)") marks a non-standard inter-meeting action, NOT a scheduled
            # meeting with a 14:00 ET statement + press conference. A real meeting cell is only a
            # day range, optionally with a SEP "*". Skip annotated rows (and report the count) —
            # minting a bogus statement/press-conf event would be wrong.
            if "(" in date_cell:
                skipped.append(f"{year} {month_cell} {date_cell!r}")
                continue
            # The row is now meeting-shaped (day range + optional "*"), so a parse failure here
            # means the Fed changed a meeting-row format — fail the fetch rather than silently
            # dropping a high-impact event.
            try:
                stmt_date = _parse_meeting_row(year, month_cell, date_cell)
            except ValueError as exc:
                errors.append(f"{year} month={month_cell!r} date={date_cell!r}: {exc}")
                continue
            # Powell-era policy: every FOMC meeting since 2019 has a press
            # conference. Fall back to anchor detection for pre-2019 meetings.
            if year >= 2019:
                has_press_conf = True
            else:
                has_press_conf = any(
                    "press conference" in a.get_text(" ", strip=True).lower()
                    for a in row.find_all("a")
                )
            out.append((stmt_date, has_press_conf, year))

    if skipped:
        logger.info(f"FOMC fetcher: skipped {len(skipped)} annotated non-meeting row(s): {skipped}")
    if errors:
        raise RuntimeError(
            "FOMC fetcher: meeting-like rows failed to parse (page format may have changed):\n  "
            + "\n  ".join(errors)
        )
    return out


def fetch(year_min: int, year_max: int) -> pd.DataFrame:
    """Return a DataFrame of FOMC meeting statement dates in ``[year_min, year_max]``.

    Each meeting produces two rows: ``US_FOMC_STATEMENT`` (14:00 ET) and ``US_FOMC_PRESS_CONFERENCE``
    (14:30 ET, when the meeting had one).
    """
    resp = get(_CALENDAR_URL)
    parsed = _parse_calendar_html(resp.text, year_min, year_max)
    if not parsed:
        raise RuntimeError(
            f"FOMC fetcher: parsed zero meetings in [{year_min}, {year_max}] from "
            f"{_CALENDAR_URL}. Page structure may have changed."
        )
    rows: list[dict] = []
    for stmt_date, has_pc, _year in parsed:
        rows.append({
            "release_date": stmt_date,
            "event_name": EventName.US_FOMC_STATEMENT,
            "source": Source.FED,
            "impact_tier": ImpactTier.HIGH,
        })
        if has_pc:
            rows.append({
                "release_date": stmt_date,
                "event_name": EventName.US_FOMC_PRESS_CONFERENCE,
                "source": Source.FED,
                "impact_tier": ImpactTier.HIGH,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = fetch(2024, 2026)
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows")
