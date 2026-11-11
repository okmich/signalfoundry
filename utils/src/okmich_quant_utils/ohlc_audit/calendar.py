"""Session calendars: which bar slots a symbol is expected to fill.

A template has two parts:
- A daily session in one timezone. Trading day D runs from ``open_time`` on D - ``open_day_offset`` to ``close_time``
  on D, for each weekday of D listed.
- Holiday rules. Each date a rule yields becomes a window between two anchors (day offset, time, zone), so a closure
  can start in one zone and end in another (FXPIG DE30: 22:00 Berlin the evening before, to 00:15 UTC the day after).

Slot states:
- OPEN, and THIN (open, but often empty: declared daily ranges such as the FX rollover hour).
- CLOSED: outside the schedule, so a bar there is out of session.
- HOLIDAY / EARLY_CLOSE: absence expected, a bar tolerated, because brokers reopen tens of minutes early in some years.
Holiday windows relabel OPEN and THIN slots only, so CLOSED stays strict.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence

import holidays
import numpy as np
import pandas as pd

from .enums import SlotState

DateSource = Callable[[int, int], Iterable[dt.date]]   # (first_year, last_year) -> dates


@dc.dataclass(frozen=True)
class Anchor:
    """A point relative to holiday date D, at ``time`` local to ``tz``.

    The day is D moved by ``day_offset``: calendar days, or weekdays (Mon-Fri) when ``business`` is set. ``then_days``
    calendar days are then added. HKG50 closes "the morning after the previous trading day": business -1, then +1.
    """

    day_offset: int
    time: dt.time
    tz: str
    business: bool = False
    then_days: int = 0

    def day(self, d: dt.date) -> dt.date:
        if not self.business:
            base = d + dt.timedelta(days=self.day_offset)
        else:
            base, step, left = d, (1 if self.day_offset > 0 else -1), abs(self.day_offset)
            while left:
                base += dt.timedelta(days=step)
                if base.weekday() < 5:
                    left -= 1
        return base + dt.timedelta(days=self.then_days)

    def resolve(self, d: dt.date) -> pd.Timestamp:
        local = pd.Timestamp(dt.datetime.combine(self.day(d), self.time))
        return local.tz_localize(self.tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


@dc.dataclass(frozen=True)
class HolidayRule:
    name: str
    dates: DateSource
    start: Anchor
    end: Anchor
    state: SlotState = SlotState.HOLIDAY

    def windows(self, first_year: int, last_year: int) -> list[tuple[pd.Timestamp, pd.Timestamp, dt.date]]:
        return [(self.start.resolve(d), self.end.resolve(d), d) for d in sorted(set(self.dates(first_year, last_year)))]


@dc.dataclass(frozen=True)
class DailySession:
    """Trading day D: ``open_time`` on D - ``open_day_offset`` to ``close_time`` on D (``None`` = 24:00), in the template tz."""

    open_time: dt.time
    close_time: dt.time | None
    open_day_offset: int
    weekdays: frozenset[int]
    close_overrides: Mapping[int, dt.time] = dc.field(default_factory=dict)   # weekday of D -> earlier close


@dc.dataclass(frozen=True)
class SessionTemplate:
    name: str
    tz: str
    session: DailySession
    holiday_rules: Sequence[HolidayRule] = ()
    thin_ranges: Sequence[tuple[dt.time, dt.time]] = ()    # daily local [start, end) where OPEN slots become THIN

    def thin_mask(self, slots: pd.DatetimeIndex) -> np.ndarray:
        if not self.thin_ranges:
            return np.zeros(len(slots), dtype=bool)
        local = slots.tz_convert(self.tz)
        minutes = local.hour * 60 + local.minute
        hit = np.zeros(len(slots), dtype=bool)
        for a, b in self.thin_ranges:
            hit |= (minutes >= a.hour * 60 + a.minute) & (minutes < b.hour * 60 + b.minute)
        return hit

    def open_intervals(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """UTC [open, close) intervals of every trading day touching [start, end)."""
        s = self.session
        first = (start.tz_convert(self.tz) - pd.Timedelta(days=2)).date()
        last = (end.tz_convert(self.tz) + pd.Timedelta(days=2)).date()
        out = []
        for day in pd.date_range(first, last, freq="D").date:
            if day.weekday() not in s.weekdays:
                continue
            o = pd.Timestamp(dt.datetime.combine(day - dt.timedelta(days=s.open_day_offset), s.open_time))
            close_t = s.close_overrides.get(day.weekday(), s.close_time)
            c = (pd.Timestamp(dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0))) if close_t is None
                 else pd.Timestamp(dt.datetime.combine(day, close_t)))
            o = o.tz_localize(self.tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")
            c = c.tz_localize(self.tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")
            out.append((o, c))
        return out


def expected_grid(template: SessionTemplate, start: pd.Timestamp, end: pd.Timestamp, bar_minutes: int = 5) -> pd.Series:
    """SlotState per bar slot (UTC bar-start stamps) on [start, end)."""
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    slots = pd.date_range(start.ceil(f"{bar_minutes}min"), end, freq=f"{bar_minutes}min", inclusive="left").as_unit("ns")
    ns = slots.asi8
    codes = {s: i for i, s in enumerate(SlotState)}
    state = np.full(len(slots), codes[SlotState.CLOSED], dtype=np.int8)
    is_open = np.zeros(len(slots), dtype=bool)
    for o, c in template.open_intervals(start, end):
        is_open[np.searchsorted(ns, o.value):np.searchsorted(ns, c.value)] = True
    state[is_open] = codes[SlotState.OPEN]
    state[is_open & template.thin_mask(slots)] = codes[SlotState.THIN]
    years = (start.year - 1, end.year + 1)
    priority = {SlotState.EARLY_CLOSE: 1, SlotState.MAINTENANCE: 1, SlotState.HOLIDAY: 2}
    rank = np.zeros(len(slots), dtype=np.int8)
    for rule in template.holiday_rules:
        for w0, w1, _ in rule.windows(*years):
            i0, i1 = np.searchsorted(ns, w0.value), np.searchsorted(ns, w1.value)
            if i1 <= i0:
                continue
            hit = is_open[i0:i1] & (rank[i0:i1] < priority[rule.state])
            state[i0:i1][hit] = codes[rule.state]
            rank[i0:i1][hit] = priority[rule.state]
    cats = [s.value for s in SlotState]
    return pd.Series(pd.Categorical.from_codes(state, categories=cats), index=slots, name="slot_state")


# ---- date sources -------------------------------------------------------------------------------------------------

def financial(calendar: str, include: Sequence[str] | None = None, exclude: Sequence[str] = ()) -> DateSource:
    """Dates of a ``holidays`` financial calendar (English names), optionally filtered by name substring."""
    def dates(y0: int, y1: int) -> list[dt.date]:
        cal = holidays.financial_holidays(calendar, years=range(y0, y1 + 1), language="en_US")
        out = []
        for d, name in cal.items():
            if include is not None and not any(k in name for k in include):
                continue
            if any(k in name for k in exclude):
                continue
            out.append(d)
        return out
    return dates


def fixed(month: int, day: int, weekdays_only: bool = True) -> DateSource:
    def dates(y0: int, y1: int) -> list[dt.date]:
        out = [dt.date(y, month, day) for y in range(y0, y1 + 1)]
        return [d for d in out if not weekdays_only or d.weekday() < 5]
    return dates


def day_after(source: DateSource) -> DateSource:
    def dates(y0: int, y1: int) -> list[dt.date]:
        return [d + dt.timedelta(days=1) for d in source(y0, y1)]
    return dates


def eve_of_first(source: DateSource) -> DateSource:
    """The weekday before the first date of each run of consecutive dates (e.g. Lunar New Year's Eve)."""
    def dates(y0: int, y1: int) -> list[dt.date]:
        ds = sorted(set(source(y0, y1)))
        out = []
        for i, d in enumerate(ds):
            if i == 0 or (d - ds[i - 1]).days > 4:
                eve = d - dt.timedelta(days=1)
                while eve.weekday() >= 5:
                    eve -= dt.timedelta(days=1)
                out.append(eve)
        return out
    return dates


def explicit(*days: str) -> DateSource:
    parsed = [dt.date.fromisoformat(d) for d in days]
    def dates(y0: int, y1: int) -> list[dt.date]:
        return [d for d in parsed if y0 <= d.year <= y1]
    return dates


def union(*sources: DateSource) -> DateSource:
    def dates(y0: int, y1: int) -> list[dt.date]:
        return sorted({d for s in sources for d in s(y0, y1)})
    return dates


def minus(source: DateSource, *others: DateSource) -> DateSource:
    def dates(y0: int, y1: int) -> list[dt.date]:
        drop = {d for o in others for d in o(y0, y1)}
        return [d for d in source(y0, y1) if d not in drop]
    return dates
