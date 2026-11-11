"""Broker clocks: raw bar stamps -> true UTC.

A broker's raw stamp is its server wall-clock. For MT5 "NY close" servers that is New York local time plus a fixed
shift: raw = UTC + shift + NY's UTC offset (-5 h EST / -4 h EDT). The shift changes at seams (eras). The tz label stored
on the parquet index is ignored: only the wall-clock digits matter.

The clock is a piecewise-constant offset schedule keyed on raw time, not a tz localisation. A server changes its DST
offset at its own moment, and for FXPIG crypto (the only symbols trading over the weekend) that is NOT 02:00 New York:
it happened at Sat 04:45, Sat 20:00 and Sun 06:40 raw, measured against Binance. By default the switch is the anchor
zone's own transition, expressed in raw time; ``DstSwitch`` records a measured moment per symbol.

Folds and skips:
- Fall-back: the raw hour before the switch occurs twice, and the server keeps one set of bars. FXPIG keeps the first
  pass, so those stamps map with the old offset and the second hour is missing in UTC.
- Spring-forward: raw stamps jump by an hour, so UTC stays continuous.
"""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
import functools
from typing import Protocol
from collections.abc import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .enums import OverrideStatus

HOUR_NS = 3_600_000_000_000


@dc.dataclass(frozen=True)
class ClockEra:
    """From ``start_raw`` (raw wall-clock, inclusive) until the next era: raw = anchor-local time + ``shift_hours``."""

    start_raw: pd.Timestamp
    shift_hours: float


@dc.dataclass(frozen=True)
class ClockOverride:
    """A per-symbol deviation from the era shift on ``[start_raw, end_raw)`` (raw wall-clock)."""

    symbols: frozenset[str]
    start_raw: pd.Timestamp
    end_raw: pd.Timestamp
    shift_hours: float
    status: OverrideStatus
    note: str = ""


@dc.dataclass(frozen=True)
class DstSwitch:
    """The raw stamp at which the server applied the anchor-zone DST change dated ``transition_date`` (local date)."""

    symbols: frozenset[str]
    transition_date: dt.date
    switch_raw: pd.Timestamp
    note: str = ""


class ClockModel(Protocol):
    overrides: Sequence[ClockOverride]

    def to_utc(self, raw: pd.DatetimeIndex, symbol: str, include_proposed: bool = False) -> pd.DatetimeIndex: ...

    def to_raw(self, utc: pd.DatetimeIndex, symbol: str) -> pd.DatetimeIndex: ...

    def unrepresentable_utc(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]: ...

    def ambiguous_raw(self, symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]: ...


class ZoneClock:
    """Raw stamps are wall-clock time in one real zone (``"UTC"`` for CCXT crypto and IB, whose parquets are UTC).

    No eras, no overrides, no broker-specific DST moments: the zone's own rules apply. With ``"UTC"`` every conversion
    is the identity. For a zone with DST, the hour repeated at fall-back maps to NaT, because wall digits alone can't
    say which pass a bar belongs to. Store such data in UTC.
    """

    eras: tuple = ()
    overrides: tuple = ()
    dst_switches: tuple = ()

    def __init__(self, tz: str = "UTC"):
        self.tz = tz

    def to_utc(self, raw: pd.DatetimeIndex, symbol: str, include_proposed: bool = False) -> pd.DatetimeIndex:
        return wall_clock(raw).tz_localize(self.tz, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")

    def to_raw(self, utc: pd.DatetimeIndex, symbol: str) -> pd.DatetimeIndex:
        return utc.tz_convert(self.tz).tz_localize(None)

    def unrepresentable_utc(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        return []

    def ambiguous_raw(self, symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
        out = []
        for utc, before, after in _transitions(self.tz) if self.tz != "UTC" else ():
            if after < before:
                p = utc + pd.Timedelta(hours=before)                 # local wall time when the clock falls back
                out.append((p - pd.Timedelta(hours=before - after), p, before - after))
        return out


def wall_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Raw wall-clock digits as a naive index, whatever tz label the file carries."""
    return index.tz_localize(None) if index.tz is not None else index


@functools.lru_cache(maxsize=8)
def _transitions(tz_name: str, first_year: int = 1990, last_year: int = 2040) -> tuple[tuple[pd.Timestamp, float, float], ...]:
    """(UTC instant, offset before, offset after) for every UTC-offset change of ``tz_name``, offsets in hours."""
    tz = ZoneInfo(tz_name)
    out = []
    prev = None
    day = dt.datetime(first_year, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(last_year + 1, 1, 1, tzinfo=dt.UTC)
    while day < end:
        off = day.astimezone(tz).utcoffset()
        if prev is not None and off != prev:
            t = day - dt.timedelta(days=1)
            while t.astimezone(tz).utcoffset() == prev:   # refine to the hour (all transitions sit on whole hours)
                t += dt.timedelta(hours=1)
            out.append((pd.Timestamp(t).tz_convert(None), prev.total_seconds() / 3600, off.total_seconds() / 3600))
        prev = off
        day += dt.timedelta(days=1)
    return tuple(out)


class EraShiftClock:
    """raw = UTC + era shift (or per-symbol override) + anchor-zone UTC offset, switched at raw-time DST moments."""

    def __init__(self, anchor_tz: str, eras: Sequence[ClockEra], overrides: Sequence[ClockOverride] = (), dst_switches: Sequence[DstSwitch] = ()):
        if not eras:
            raise ValueError("at least one era is required")
        starts = [e.start_raw for e in eras]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("eras must have strictly increasing start_raw")
        self.anchor_tz = anchor_tz
        self.eras = tuple(eras)
        self.overrides = tuple(overrides)
        self.dst_switches = tuple(dst_switches)

    def shift_hours(self, raw: pd.DatetimeIndex, symbol: str, include_proposed: bool = False) -> np.ndarray:
        """Era shift (hours) for each raw stamp. Before the first era, the first era's shift applies."""
        wall = wall_clock(raw)
        starts = pd.DatetimeIndex([e.start_raw for e in self.eras])
        pos = np.clip(starts.searchsorted(wall, side="right") - 1, 0, len(self.eras) - 1)
        shift = np.array([e.shift_hours for e in self.eras], dtype=float)[pos]
        for ov in self.overrides:
            if symbol not in ov.symbols:
                continue
            if ov.status is not OverrideStatus.CONFIRMED and not include_proposed:
                continue
            shift[(wall >= ov.start_raw) & (wall < ov.end_raw)] = ov.shift_hours
        return shift

    def _switch_table(self, symbol: str) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
        """Raw switch stamps, the anchor offset after each, and the offset before the first one (cached per symbol)."""
        cache = self.__dict__.setdefault("_switch_cache", {})
        if symbol not in cache:
            cache[symbol] = self._build_switch_table(symbol)
        return cache[symbol]

    def _build_switch_table(self, symbol: str) -> tuple[pd.DatetimeIndex, np.ndarray, float]:
        measured = {s.transition_date: s.switch_raw for s in self.dst_switches if symbol in s.symbols}
        trans = _transitions(self.anchor_tz)
        stamps, after = [], []
        for utc, off_before, off_after in trans:
            local_date = (utc + pd.Timedelta(hours=off_before)).date()
            if local_date in measured:
                stamps.append(measured[local_date])
            else:
                # default: the anchor transition instant, read on the old clock; the era shift is taken at that instant
                approx = utc + pd.Timedelta(hours=off_before + self.eras[0].shift_hours)
                shift = self.shift_hours(pd.DatetimeIndex([approx]), symbol)[0]
                stamps.append(utc + pd.Timedelta(hours=off_before + shift))
            after.append(off_after)
        return pd.DatetimeIndex(stamps), np.asarray(after, dtype=float), trans[0][1]

    def anchor_offset_hours(self, raw: pd.DatetimeIndex, symbol: str) -> np.ndarray:
        """Anchor-zone UTC offset (e.g. -5/-4 for New York) in force on the server clock at each raw stamp."""
        stamps, after, first = self._switch_table(symbol)
        pos = stamps.searchsorted(wall_clock(raw), side="right") - 1
        return np.where(pos >= 0, after[np.clip(pos, 0, None)], first)

    def offset_hours(self, raw: pd.DatetimeIndex, symbol: str, include_proposed: bool = False) -> np.ndarray:
        """raw - UTC in hours, per raw stamp."""
        return self.shift_hours(raw, symbol, include_proposed) + self.anchor_offset_hours(raw, symbol)

    def unrepresentable_utc(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """UTC intervals with no raw stamp: where the offset drops (fall-back fold, backward seam).

        Raw time continues while the offset falls by d hours, so the next d hours of UTC are never stamped. A server
        that merges a repeated hour leaves the same hole. The data cannot have bars there, so it is not a gap.
        """
        lo, hi = start.tz_convert("UTC").tz_localize(None), end.tz_convert("UTC").tz_localize(None)
        out = []
        for p, before, at in self._offset_drops(symbol):
            u0 = p - pd.Timedelta(hours=before)
            u1 = u0 + pd.Timedelta(hours=before - at)
            if u1 > lo and u0 < hi:
                out.append((u0.tz_localize("UTC"), u1.tz_localize("UTC")))
        return out

    def ambiguous_raw(self, symbol: str) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
        """Raw intervals the server clock passes through twice: [p - d, p) where the offset drops by d hours at p.

        A server keeps one bar per raw stamp, so a bar there may hold the first pass, the second, or both merged
        (the FXPIG crypto folds of 2024-11-02 and 2025-11-01). Only a reference can tell which.
        """
        return [(p - pd.Timedelta(hours=before - at), p, before - at) for p, before, at in self._offset_drops(symbol)]

    def _offset_drops(self, symbol: str) -> list[tuple[pd.Timestamp, float, float]]:
        stamps, _, _ = self._switch_table(symbol)
        points = list(stamps) + [e.start_raw for e in self.eras]
        for ov in self.overrides:
            if symbol in ov.symbols and ov.status is OverrideStatus.CONFIRMED:
                points += [ov.start_raw, ov.end_raw]
        out = []
        for p in sorted(set(points)):
            before = self.offset_hours(pd.DatetimeIndex([p - pd.Timedelta(microseconds=1)]), symbol)[0]
            at = self.offset_hours(pd.DatetimeIndex([p]), symbol)[0]
            if at < before:
                out.append((p, before, at))
        return out

    def to_utc(self, raw: pd.DatetimeIndex, symbol: str, include_proposed: bool = False) -> pd.DatetimeIndex:
        wall = wall_clock(raw)
        utc = wall - pd.to_timedelta(self.offset_hours(wall, symbol, include_proposed), unit="h")
        return utc.tz_localize("UTC")

    def to_raw(self, utc: pd.DatetimeIndex, symbol: str) -> pd.DatetimeIndex:
        """Raw stamps for UTC instants (inverse of :meth:`to_utc`, confirmed overrides only).

        Candidate raw = UTC + k for each total offset k in use; it is valid when the clock gives offset k there. Where an
        instant has two valid stamps (forward seam) the smaller wins; where it has none (fold or backward seam): NaT.
        """
        u = utc.tz_convert("UTC").tz_localize(None)
        shifts = {e.shift_hours for e in self.eras} | {o.shift_hours for o in self.overrides
                                                     if symbol in o.symbols and o.status is OverrideStatus.CONFIRMED}
        offsets = {t[1] for t in _transitions(self.anchor_tz)} | {t[2] for t in _transitions(self.anchor_tz)}
        best = np.full(len(u), np.datetime64("NaT"), dtype="datetime64[ns]")
        for k in sorted({s + o for s in shifts for o in offsets}):
            cand = u + pd.Timedelta(hours=k)
            ok = np.isclose(self.offset_hours(cand, symbol), k)
            cand_ns = cand.to_numpy(dtype="datetime64[ns]")
            take = ok & (np.isnat(best) | (cand_ns < best))
            best[take] = cand_ns[take]
        return pd.DatetimeIndex(best)
