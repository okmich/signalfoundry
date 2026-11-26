"""FOMC meeting fetcher — federalreserve.gov.

Source pages
------------
- Current calendar: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm. It lists the coming year and
  about five past years (2021-2027 as of 2026-11).
- Historical: https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm, one page per year. A year moves
  here when it drops off the current page.

``fetch`` reads the current page, then the historical page of every requested year older than the current page's first
year. A historical page that yields no scheduled meeting raises, and so does a year missing from the middle of the
current page. Future years not yet posted are simply absent.

Supported years
---------------
``EARLIEST_YEAR`` (2014) onward. The release-time convention in ``release_times.py`` (statement 14:00 ET, press
conference 14:30 ET) applies to every scheduled meeting from March 2013 (Fed announcement of 2013-03-13). Before that,
statements came out at other times (about 14:15 ET). 2014 is the first full year on the current times. Earlier years
raise rather than return rows stamped at the wrong time.

Current page structure (verified 2026-11-26)
--------------------------------------------
One ``<div class="panel panel-default">`` per year, headed by an ``h4`` such as "2026 FOMC Meetings". Each meeting is a
``div.row.fomc-meeting`` holding ``div.fomc-meeting__month`` ("January", "April/May") and ``div.fomc-meeting__date``
("27-28", "17-18*", "30-1"). The asterisk marks SEP meetings, NOT press-conference meetings. An inter-meeting action
carries a note in the date cell: "22 (notation vote)", "(unscheduled)", "(conference call)".

Historical page structure (verified 2026-11-26 for 1990-2020)
-------------------------------------------------------------
One ``div.panel`` per meeting or action, headed by an ``h5``. Examples: "January 28-29 Meeting - 2014",
"Jul/Aug 31-1 Meeting - 2018", "July 31-August 1 Meeting - 2012", "October 4 (unscheduled) - 2019",
"March 17-18 (cancelled) Meeting - 2020", "March 23 (notation vote) - 2020", "August 1 Conference Call - 2011".
A heading that ends in a year and does not parse raises: silently dropping a meeting is the failure this guards against.

Scheduled meetings and press conferences
----------------------------------------
``fetch`` returns one row per scheduled meeting, keyed on the **last** (policy statement) day. That is the day the rate
decision and press conference land on the chart. Since 2019 every meeting has had a press conference (Powell-era
policy). The current page adds the "Press Conference" anchor only *after* a meeting concludes, so a presence check
would under-mask future meetings. A press-conference row is therefore emitted for every meeting from 2019 onward. For
older meetings (quarterly press conferences, 2014-2018) the anchor decides.

Inter-meeting actions
---------------------
``fetch_unscheduled`` returns the unscheduled meetings, notation votes and conference calls that published a statement,
e.g. the 2020-03-03 and 2020-03-15 emergency cuts. Release times vary (2020-03-03 at 10:00 ET, 2020-03-15 on a Sunday
evening), so these rows are **date-only**. They are not part of the news-calendar parquet, which needs a fixed release
time per event. The release date comes from the statement link (``.../monetary20191011a.htm``) and can be later than
the action day: the 2019-10-04 call's statement came out on 2019-10-11. Actions with no statement are skipped:
cancelled meetings, notation votes that only issued a press release, and calls with no public release.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import enum
import logging
import re
from collections.abc import Iterator

import pandas as pd
from bs4 import BeautifulSoup, Tag

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source

logger = logging.getLogger(__name__)

_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

EARLIEST_YEAR = 2014    # first full year on the 14:00 / 14:30 ET release times (see module docstring)
_PRESS_CONF_EVERY_MEETING_FROM = 2019

# Map English month names to integers. We accept three-letter abbreviations
# too (the Fed sometimes uses "Jan/Feb" or "January/February").
_MONTH_LOOKUP: dict[str, int] = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"], start=1)}
for _m, _i in list(_MONTH_LOOKUP.items()):
    _MONTH_LOOKUP[_m[:3]] = _i


class FomcAction(enum.StrEnum):
    """Kind of inter-meeting FOMC action. Such actions have no fixed release time (date-only)."""

    UNSCHEDULED_MEETING = "unscheduled_meeting"
    NOTATION_VOTE = "notation_vote"
    CONFERENCE_CALL = "conference_call"


_ANNOTATIONS: dict[str, FomcAction] = {
    "unscheduled": FomcAction.UNSCHEDULED_MEETING,
    "notation vote": FomcAction.NOTATION_VOTE,
    "conference call": FomcAction.CONFERENCE_CALL,
}

UNSCHEDULED_COLUMNS = ["action_date", "release_date", "action", "statement", "has_press_conference", "source"]


@dc.dataclass(frozen=True)
class _Action:
    """One inter-meeting action that published a statement."""

    action_date: dt.date
    release_date: dt.date
    action: FomcAction
    statement: str
    has_press_conference: bool


_HISTORICAL_HEADING = re.compile(
    r"^(?P<month>[A-Za-z/]+)\s+(?P<first>\d{1,2})(?:\s*[-–]\s*(?:(?P<month2>[A-Za-z]+)\s+)?(?P<last>\d{1,2}))?"
    r"\s*(?:\((?P<note>[^)]*)\))?\s*(?P<kind>Meeting|Conference Call)?\s*[-–]\s*(?P<year>\d{4})$")
_STATEMENT_HREF = re.compile(r"monetary(\d{8})[a-z]*\d*\.htm")


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


def _action_kind(note: str) -> FomcAction | None:
    """Map an annotation ("unscheduled", "notation vote", "conference call") to its action; None for a cancellation."""
    key = note.strip().lower()
    if "cancel" in key:
        return None
    for token, action in _ANNOTATIONS.items():
        if token in key:
            return action
    raise ValueError(f"Unrecognised FOMC annotation: {note!r}")


def _statement_link(container: Tag) -> tuple[str, dt.date] | None:
    """(link title, release date) of the first statement link in ``container``; None when it has none."""
    for a in container.find_all("a"):
        title = a.get_text(" ", strip=True)
        if not title.lower().startswith("statement"):
            continue
        m = _STATEMENT_HREF.search(a.get("href", ""))
        if m is None:
            raise ValueError(f"FOMC statement link without a dated URL: {a.get('href')!r}")
        return title, dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def _has_press_conference_link(container: Tag) -> bool:
    return any("press conference" in a.get_text(" ", strip=True).lower() for a in container.find_all("a"))


def _check_years(year_min: int, year_max: int) -> None:
    if year_min > year_max:
        raise ValueError(f"year_min {year_min} > year_max {year_max}")
    if year_min < EARLIEST_YEAR:
        raise ValueError(f"FOMC fetcher supports years >= {EARLIEST_YEAR}; got year_min={year_min}. Statements before "
                         f"March 2013 were not released at the 14:00 ET convention in release_times.py.")


# ---- current calendar page ----------------------------------------------------------------------------------------

def _panel_year(panel: Tag) -> int | None:
    header = panel.find("h4")
    m = re.search(r"(20\d{2})", header.get_text()) if header is not None else None
    return int(m.group(1)) if m else None


def _calendar_years(html: str) -> set[int]:
    """Years the current calendar page lists, regardless of any requested window."""
    soup = BeautifulSoup(html, "html.parser")
    return {y for y in (_panel_year(p) for p in soup.select("div.panel.panel-default")) if y is not None}


def _calendar_rows(html: str, year_min: int, year_max: int) -> Iterator[tuple[int, Tag, str, str]]:
    """(year, row, month_cell, date_cell) for every dated meeting row of the current page within the window."""
    soup = BeautifulSoup(html, "html.parser")
    for panel in soup.select("div.panel.panel-default"):
        year = _panel_year(panel)
        if year is None or year < year_min or year > year_max:
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
            yield year, row, month_cell, date_cell


def _parse_calendar_html(html: str, year_min: int, year_max: int) -> list[tuple[dt.date, bool, int]]:
    """Parse the current calendar page → list of (statement_date, has_press_conf, year) for scheduled meetings."""
    out: list[tuple[dt.date, bool, int]] = []
    errors: list[str] = []
    skipped: list[str] = []
    for year, row, month_cell, date_cell in _calendar_rows(html, year_min, year_max):
        # A parenthetical annotation ("22 (notation vote)", "(unscheduled)", "(conference call)") marks an inter-meeting
        # action, NOT a scheduled meeting with a 14:00 ET statement + press conference; ``fetch_unscheduled`` reads
        # those. A real meeting cell is only a day range, optionally with a SEP "*".
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
        has_press_conf = year >= _PRESS_CONF_EVERY_MEETING_FROM or _has_press_conference_link(row)
        out.append((stmt_date, has_press_conf, year))

    if skipped:
        logger.info(f"FOMC fetcher: skipped {len(skipped)} annotated inter-meeting row(s) (read by fetch_unscheduled): "
                    f"{skipped}")
    if errors:
        raise RuntimeError(
            "FOMC fetcher: meeting-like rows failed to parse (page format may have changed):\n  "
            + "\n  ".join(errors)
        )
    return out


def _parse_calendar_actions(html: str, year_min: int, year_max: int) -> list[_Action]:
    """Inter-meeting actions with a statement on the current calendar page (the annotated rows)."""
    out: list[_Action] = []
    for year, row, month_cell, date_cell in _calendar_rows(html, year_min, year_max):
        if "(" not in date_cell:
            continue
        day_part, _, rest = date_cell.partition("(")
        action = _action_kind(rest.rstrip(")"))
        if action is None:
            continue
        link = _statement_link(row)
        if link is None:
            continue
        action_date = _parse_meeting_row(year, month_cell, day_part)
        out.append(_Action(action_date, link[1], action, link[0], _has_press_conference_link(row)))
    return out


# ---- historical pages ---------------------------------------------------------------------------------------------

def _historical_headings(html: str) -> Iterator[tuple[str, Tag]]:
    """(heading text, enclosing panel) for every meeting/action heading of a historical page."""
    soup = BeautifulSoup(html, "html.parser")
    for h in soup.find_all(["h3", "h4", "h5"]):
        text = " ".join(h.get_text(" ", strip=True).split())
        # Meeting headings end in "- YYYY" and carry a day before the first dash; the bare year header does not.
        if not re.search(r"[-–]\s*\d{4}$", text) or not re.search(r"\d", re.split(r"[-–]", text)[0]):
            continue
        panel = h.find_parent("div", class_=lambda cs: cs is not None and "panel" in cs)
        if panel is None:
            raise ValueError(f"FOMC historical heading outside a panel: {text!r}")
        yield text, panel


def _parse_historical_html(html: str, year: int) -> tuple[list[tuple[dt.date, bool, int]], list[_Action]]:
    """Parse one historical page → (scheduled meetings as (statement_date, has_press_conf, year), actions)."""
    meetings: list[tuple[dt.date, bool, int]] = []
    actions: list[_Action] = []
    for text, panel in _historical_headings(html):
        m = _HISTORICAL_HEADING.match(text)
        if m is None:
            raise ValueError(f"FOMC historical heading not recognised (page format may have changed): {text!r}")
        if int(m.group("year")) != year:
            raise ValueError(f"FOMC historical heading {text!r} on the {year} page")
        month_cell = m.group("month") if m.group("month2") is None else f"{m.group('month')}/{m.group('month2')}"
        date_cell = m.group("first") if m.group("last") is None else f"{m.group('first')}-{m.group('last')}"
        day = _parse_meeting_row(year, month_cell, date_cell)
        note, kind = m.group("note"), m.group("kind")
        if note is None and kind == "Meeting":
            has_press_conf = year >= _PRESS_CONF_EVERY_MEETING_FROM or _has_press_conference_link(panel)
            meetings.append((day, has_press_conf, year))
            continue
        if note is not None:
            action = _action_kind(note)
        elif kind == "Conference Call":
            action = FomcAction.CONFERENCE_CALL
        else:
            raise ValueError(f"FOMC historical heading has neither 'Meeting' nor an annotation: {text!r}")
        if action is None:
            continue
        link = _statement_link(panel)
        if link is not None:
            actions.append(_Action(day, link[1], action, link[0], _has_press_conference_link(panel)))
    return meetings, actions


# ---- fetch --------------------------------------------------------------------------------------------------------

def _historical_years(listed: set[int], year_min: int, year_max: int) -> list[int]:
    """Requested years that need a historical page; raises when the current page skips a year inside its own span."""
    if not listed:
        raise RuntimeError(f"FOMC fetcher: no year panels on {_CALENDAR_URL}. Page structure may have changed.")
    first, last = min(listed), max(listed)
    gaps = [y for y in range(max(year_min, first), min(year_max, last) + 1) if y not in listed]
    if gaps:
        raise RuntimeError(f"FOMC fetcher: {_CALENDAR_URL} lists {first}-{last} but not {gaps}.")
    return list(range(year_min, min(year_max, first - 1) + 1))


def _fetch_pages(year_min: int, year_max: int) -> tuple[str, dict[int, str]]:
    """The current page's HTML plus {year: HTML} of every historical page the window needs."""
    current = get(_CALENDAR_URL).text
    years = _historical_years(_calendar_years(current), year_min, year_max)
    return current, {y: get(_HISTORICAL_URL.format(year=y)).text for y in years}


def fetch(year_min: int, year_max: int) -> pd.DataFrame:
    """Return a DataFrame of scheduled FOMC meeting statement dates in ``[year_min, year_max]``.

    Each meeting produces two rows: ``US_FOMC_STATEMENT`` (14:00 ET) and ``US_FOMC_PRESS_CONFERENCE``
    (14:30 ET, when the meeting had one). Years before ``EARLIEST_YEAR`` raise ``ValueError``.
    """
    _check_years(year_min, year_max)
    current, historical = _fetch_pages(year_min, year_max)
    parsed = _parse_calendar_html(current, year_min, year_max)
    for year, html in historical.items():
        meetings, _ = _parse_historical_html(html, year)
        if not meetings:
            raise RuntimeError(f"FOMC fetcher: parsed zero meetings from {_HISTORICAL_URL.format(year=year)}. "
                               f"Page structure may have changed.")
        parsed.extend(meetings)
    if not parsed:
        raise RuntimeError(
            f"FOMC fetcher: parsed zero meetings in [{year_min}, {year_max}] from "
            f"{_CALENDAR_URL}. Page structure may have changed."
        )
    rows: list[dict] = []
    for stmt_date, has_pc, _year in sorted(parsed):
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


def fetch_unscheduled(year_min: int, year_max: int) -> pd.DataFrame:
    """Return the inter-meeting FOMC actions that published a statement in ``[year_min, year_max]`` (date-only).

    Columns: ``action_date`` (day of the call, vote or meeting), ``release_date`` (day the statement was published),
    ``action`` (``FomcAction``), ``statement`` (link title, e.g. "Statement" or "Statement on Longer-Run Goals and
    Monetary Policy Strategy"), ``has_press_conference``, ``source``. Years are the action's calendar year. Empty (with
    these columns) when the window had none. Years before ``EARLIEST_YEAR`` raise ``ValueError``.
    """
    _check_years(year_min, year_max)
    current, historical = _fetch_pages(year_min, year_max)
    actions = _parse_calendar_actions(current, year_min, year_max)
    for year, html in historical.items():
        actions.extend(_parse_historical_html(html, year)[1])
    actions.sort(key=lambda a: (a.release_date, a.action_date))
    rows = [{**dc.asdict(a), "source": Source.FED} for a in actions]
    return pd.DataFrame(rows, columns=UNSCHEDULED_COLUMNS)


if __name__ == "__main__":
    df = fetch(2024, 2026)
    print(df.to_string(index=False))
    print(f"\n{len(df)} rows")
    print(fetch_unscheduled(EARLIEST_YEAR, 2026).to_string(index=False))
