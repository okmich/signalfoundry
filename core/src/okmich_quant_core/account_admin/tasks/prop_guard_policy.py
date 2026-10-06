"""``prop_guard``'s policy: its config, the trading-day clock, the arithmetic and the calendar (ACCOUNT_ADMIN_SPEC §6).

Everything here is a pure function of the policy and the numbers handed in, so the hand calculations of the policy
replay test (§14.7) can be checked directly against it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .._validate import Problems, boolean, choice, integer, number, section, text, timestamp, timezone_name, unknown_keys
from ..enums import (RESTRICTIVE_DIRECTIVES, POLICY_CONDITIONS, AccountDirective, CalendarKind, Condition, DailyBase,
                     LatchScope, LimitReference, MaxLossMode, TargetMeasure)

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEKLY_AT = re.compile(r"^\s*([A-Za-z]{3})\s+(\d{1,2}):(\d{2})\s*$")
_MINUTES_PER_WEEK = 7 * 24 * 60

_POLICY_KEYS = frozenset({"initial_capital", "account_start_utc", "day_tz", "day_start_hour", "daily_base",
                          "daily_limit_reference", "daily_loss_pct", "daily_warn_fraction", "max_loss_pct",
                          "max_loss_mode", "max_loss_trail_locks_at_initial", "max_warn_fraction", "profit_target_pct",
                          "target_measure", "conditions", "calendar"})
_TASK_KEYS = frozenset({"kind", "policy", "checks", "max_override_s"})


@dataclass(frozen=True)
class ConditionRule:
    directive: AccountDirective
    latch: LatchScope


@dataclass(frozen=True)
class CalendarWindow:
    """A scheduled directive (spec §6.5). Weekly windows are minutes-of-week in their zone and may wrap the weekend;
    dated windows are UTC instants. Active on ``[start, end)``."""

    id: str
    kind: CalendarKind
    directive: AccountDirective
    tz: ZoneInfo | None = None
    start_mow: int | None = None      # weekly: minute of week, Monday 00:00 = 0
    end_mow: int | None = None
    start_utc: datetime | None = None  # dated
    end_utc: datetime | None = None

    def active(self, now: datetime) -> bool:
        if self.kind is CalendarKind.DATED:
            return self.start_utc <= now < self.end_utc
        local = now.astimezone(self.tz)
        mow = local.weekday() * 1440 + local.hour * 60 + local.minute + local.second / 60.0
        if self.start_mow < self.end_mow:
            return self.start_mow <= mow < self.end_mow
        return mow >= self.start_mow or mow < self.end_mow   # wraps the week's end

    def is_past(self, now: datetime) -> bool:
        return self.kind is CalendarKind.DATED and now >= self.end_utc


@dataclass(frozen=True)
class PropPolicy:
    initial_capital: float
    account_start_utc: datetime
    day_tz: ZoneInfo
    day_start_hour: int
    daily_base: DailyBase
    daily_limit_reference: LimitReference
    daily_loss_pct: float
    daily_warn_fraction: float
    max_loss_pct: float
    max_loss_mode: MaxLossMode
    max_loss_trail_locks_at_initial: bool
    max_warn_fraction: float
    profit_target_pct: float | None
    target_measure: TargetMeasure
    conditions: Mapping[Condition, ConditionRule]
    calendar: tuple[CalendarWindow, ...]
    close_grace_s: int
    obey_grace_s: int
    max_override_s: int


def _parse_weekly_at(value: Any, where: str, p: Problems) -> int | None:
    m = _WEEKLY_AT.match(value) if isinstance(value, str) else None
    if not m or m.group(1).lower() not in _DAYS or not (0 <= int(m.group(2)) <= 23) or not (0 <= int(m.group(3)) <= 59):
        p.add(where, f"must be '<Day HH:MM>' like 'Fri 16:30', got {value!r}")
        return None
    return _DAYS.index(m.group(1).lower()) * 1440 + int(m.group(2)) * 60 + int(m.group(3))


def _parse_calendar(raw: Any, p: Problems) -> tuple[CalendarWindow, ...]:
    if raw is None:
        p.add(p.prefix, "missing (use [] for none)")
        return ()
    if not isinstance(raw, list):
        p.add(p.prefix, "must be a list of windows")
        return ()
    windows: list[CalendarWindow] = []
    ids: set[str] = set()
    for i, item in enumerate(raw):
        q = p.child(i)
        if not isinstance(item, Mapping):
            q.add(q.prefix, "must be an object")
            continue
        wid = text(item, "id", q)
        if wid is not None:
            if wid in ids:
                q.add(q.at("id"), f"duplicate window id {wid!r}")
            ids.add(wid)
        kind = choice(item, "kind", CalendarKind, q)
        directive = choice(item, "directive", AccountDirective, q, allowed=RESTRICTIVE_DIRECTIVES)
        if kind is CalendarKind.WEEKLY:
            unknown_keys(item, {"id", "kind", "tz", "start", "end", "directive"}, q)
            tz = timezone_name(item, "tz", q)
            start = _parse_weekly_at(item.get("start"), q.at("start"), q)
            end = _parse_weekly_at(item.get("end"), q.at("end"), q)
            if start is not None and end is not None and start == end:
                q.add(q.prefix, "start and end are the same instant")
            if None not in (wid, directive, tz, start, end) and start != end:
                windows.append(CalendarWindow(wid, kind, directive, tz=tz, start_mow=start, end_mow=end))
        elif kind is CalendarKind.DATED:
            unknown_keys(item, {"id", "kind", "start_utc", "end_utc", "directive"}, q)
            start_utc = timestamp(item, "start_utc", q)
            end_utc = timestamp(item, "end_utc", q)
            if start_utc is not None and end_utc is not None and not start_utc < end_utc:
                q.add(q.prefix, "start_utc must be before end_utc")
            elif None not in (wid, directive, start_utc, end_utc):
                windows.append(CalendarWindow(wid, kind, directive, start_utc=start_utc, end_utc=end_utc))
    return tuple(windows)


def parse_prop_guard(entry: Mapping[str, Any], prefix: str) -> tuple[PropPolicy | None, list[str]]:
    """Validate a ``prop_guard`` entry completely; the policy, or ``None`` and every problem found (spec §6.7)."""
    p = Problems(prefix)
    unknown_keys(entry, _TASK_KEYS, p)
    max_override_s = integer(entry, "max_override_s", p, minimum=60)
    checks = section(entry, "checks", p)
    close_grace = obey_grace = None
    if checks is not None:
        c = p.child("checks")
        unknown_keys(checks, {"close_grace_s", "obey_grace_s"}, c)
        close_grace = integer(checks, "close_grace_s", c, minimum=0)
        obey_grace = integer(checks, "obey_grace_s", c, minimum=0)
    raw = section(entry, "policy", p)
    if raw is None:
        return None, p.items
    q = p.child("policy")
    unknown_keys(raw, _POLICY_KEYS, q)
    capital = number(raw, "initial_capital", q, gt=0)
    start = timestamp(raw, "account_start_utc", q)
    tz = timezone_name(raw, "day_tz", q)
    hour = integer(raw, "day_start_hour", q, minimum=0, maximum=23)
    daily_base = choice(raw, "daily_base", DailyBase, q)
    reference = choice(raw, "daily_limit_reference", LimitReference, q)
    daily_pct = number(raw, "daily_loss_pct", q, gt=0, lt=100)
    daily_warn = number(raw, "daily_warn_fraction", q, gt=0, lt=1)
    max_pct = number(raw, "max_loss_pct", q, gt=0, lt=100)
    mode = choice(raw, "max_loss_mode", MaxLossMode, q)
    locks = boolean(raw, "max_loss_trail_locks_at_initial", q)
    max_warn = number(raw, "max_warn_fraction", q, gt=0, lt=1)
    target_pct = number(raw, "profit_target_pct", q, gt=0, nullable=True)
    measure = choice(raw, "target_measure", TargetMeasure, q)
    rules: dict[Condition, ConditionRule] = {}
    conditions = section(raw, "conditions", q)
    if conditions is not None:
        cq = q.child("conditions")
        unknown_keys(conditions, {c.value for c in POLICY_CONDITIONS}, cq)
        for cond in POLICY_CONDITIONS:
            if cond is Condition.TARGET and target_pct is None and cond.value not in conditions:
                continue   # no target configured: its condition may be omitted
            rule = section(conditions, cond.value, cq)
            if rule is None:
                continue
            rq = cq.child(cond.value)
            unknown_keys(rule, {"directive", "latch"}, rq)
            directive = choice(rule, "directive", AccountDirective, rq, allowed=RESTRICTIVE_DIRECTIVES)
            latch = choice(rule, "latch", LatchScope, rq)
            if directive is not None and latch is not None:
                rules[cond] = ConditionRule(directive, latch)
    calendar = _parse_calendar(raw.get("calendar"), q.child("calendar"))
    if p:
        return None, p.items
    return PropPolicy(initial_capital=capital, account_start_utc=start, day_tz=tz, day_start_hour=hour,
                      daily_base=daily_base, daily_limit_reference=reference, daily_loss_pct=daily_pct,
                      daily_warn_fraction=daily_warn, max_loss_pct=max_pct, max_loss_mode=mode,
                      max_loss_trail_locks_at_initial=locks, max_warn_fraction=max_warn, profit_target_pct=target_pct,
                      target_measure=measure, conditions=rules, calendar=calendar, close_grace_s=close_grace,
                      obey_grace_s=obey_grace, max_override_s=max_override_s), []


def boundary_at_or_before(zone: ZoneInfo, hour: int, now: datetime) -> datetime:
    """The latest instant at or before ``now`` that is ``hour``:00 local time in ``zone``. Built from the local date so
    a DST change moves the boundary with the zone, as the account's rules do."""
    local = now.astimezone(zone)
    candidate = datetime.combine(local.date(), time(hour), tzinfo=zone)
    if candidate > local:
        candidate = datetime.combine(local.date() - timedelta(days=1), time(hour), tzinfo=zone)
    return candidate.astimezone(now.tzinfo)


def next_boundary_after(zone: ZoneInfo, hour: int, start: datetime) -> datetime:
    """The first boundary strictly after a boundary ``start`` (23, 24 or 25 h later across DST)."""
    return boundary_at_or_before(zone, hour, start + timedelta(hours=26))


def day_boundary(policy: PropPolicy, now: datetime) -> datetime:
    """``T_d``: the latest instant at or before ``now`` that is ``day_start_hour``:00 in ``day_tz`` (spec §6.3)."""
    return boundary_at_or_before(policy.day_tz, policy.day_start_hour, now)


def boundary_key(policy: PropPolicy) -> dict:
    """What defines the day boundary, stored with the trading day so a policy edit is recognised as one."""
    return {"day_tz": policy.day_tz.key, "day_start_hour": policy.day_start_hour}


@dataclass(frozen=True)
class Levels:
    """The spec §6.3 quantities for one cycle, in the account currency."""

    daily_base: float
    daily_allowance: float
    daily_floor: float
    daily_warn: float
    max_allowance: float
    anchor: float
    max_floor: float
    max_warn: float
    target: float | None

    def to_dict(self) -> dict[str, float | None]:
        r = lambda v: None if v is None else round(v, 2)
        return {"daily_floor": r(self.daily_floor), "daily_warn": r(self.daily_warn), "max_floor": r(self.max_floor),
                "max_warn": r(self.max_warn), "target": r(self.target), "daily_base": r(self.daily_base),
                "anchor": r(self.anchor)}


def compute_levels(policy: PropPolicy, daily_base: float, balance_high: float, equity_high: float) -> Levels:
    """Spec §6.3. ``balance_high``/``equity_high`` are the high-water marks since ``account_start_utc``."""
    c = policy.initial_capital
    reference = c if policy.daily_limit_reference is LimitReference.INITIAL_CAPITAL else daily_base
    l_d = policy.daily_loss_pct / 100.0 * reference
    f_d = daily_base - l_d
    w_d = f_d + (1.0 - policy.daily_warn_fraction) * l_d
    l_m = policy.max_loss_pct / 100.0 * c
    if policy.max_loss_mode is MaxLossMode.STATIC:
        anchor = c
    elif policy.max_loss_mode is MaxLossMode.TRAILING_BALANCE:
        anchor = max(c, balance_high)
    else:
        anchor = max(c, equity_high)
    f_m = anchor - l_m
    if policy.max_loss_trail_locks_at_initial:
        f_m = min(f_m, c)
    w_m = f_m + (1.0 - policy.max_warn_fraction) * l_m
    target = c + c * policy.profit_target_pct / 100.0 if policy.profit_target_pct is not None else None
    # Money is compared in cents: 100000 * 1.1 is 110000.00000000001 in floating point, and a balance of exactly
    # 110000.00 must meet a 10 % target.
    cents = lambda v: None if v is None else round(v, 2)
    return Levels(daily_base=cents(daily_base), daily_allowance=cents(l_d), daily_floor=cents(f_d), daily_warn=cents(w_d),
                  max_allowance=cents(l_m), anchor=cents(anchor), max_floor=cents(f_m), max_warn=cents(w_m),
                  target=cents(target))
