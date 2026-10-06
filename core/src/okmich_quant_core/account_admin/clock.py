"""The Admin's clock: irregular and off-phase (ACCOUNT_ADMIN_SPEC §7.2).

The trading systems run on fixed ticks: every strategy at second 0 of each minute, position checks at every multiple
of ``chk_position_interval``. They share one terminal whose IPC serves one call at a time. So the Admin never starts a
cycle inside a blackout window, and jitters its period with an unseeded generator so it does not phase-lock with them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import reduce
from typing import Iterable, Mapping, Sequence

from ._validate import Problems, integer

#: The hyperperiod of the blackout windows is scanned second by second; beyond a day the config is surely wrong.
MAX_HYPERPERIOD_S = 86_400

#: The Supervisor judges the Admin's heartbeat wedged after WedgeMultiple (3) x 1 min + WedgeGrace (60 s) = 240 s by
#: default. The heartbeat's asof trails its write by up to 120 s (the open of the last complete minute), so the worst
#: gap between cycles must stay under 240 - 120 = 120 s; 110 s leaves a margin (spec §7.2, §9).
HEARTBEAT_GAP_MAX_S = 110


@dataclass(frozen=True)
class BlackoutWindow:
    """Blocked when ``(t - offset_s) mod every_s < length_s``, ``t`` in wall-clock seconds."""

    every_s: int
    offset_s: int
    length_s: int

    def blocks(self, t: float) -> bool:
        return (t - self.offset_s) % self.every_s < self.length_s

    def end_after(self, t: float) -> float:
        """The end of this window's current block containing ``t`` (only meaningful when ``blocks(t)``)."""
        return t - ((t - self.offset_s) % self.every_s) + self.length_s


@dataclass(frozen=True)
class ClockConfig:
    cycle_s: int
    jitter_s: int
    valid_for_s: int
    blackout: tuple[BlackoutWindow, ...]


def blocked(t: float, windows: Iterable[BlackoutWindow]) -> bool:
    return any(w.blocks(t) for w in windows)


def longest_blackout_run(windows: Sequence[BlackoutWindow]) -> int:
    """``B``: the longest run of merged, contiguous blocked seconds, over one hyperperiod, wrapping around.

    Raises ``ValueError`` when the windows block every second (no cycle could ever start) or their hyperperiod is too
    long to scan. All window parameters are integers, so a scan on the integer grid is exact.
    """
    if not windows:
        return 0
    period = reduce(math.lcm, (w.every_s for w in windows))
    if period > MAX_HYPERPERIOD_S:
        raise ValueError(f"blackout hyperperiod {period}s exceeds {MAX_HYPERPERIOD_S}s")
    marks = [blocked(s, windows) for s in range(period)]
    if all(marks):
        raise ValueError("the blackout windows block every second")
    start = marks.index(False)   # begin scanning at a free second so a run wrapping the period end is counted whole
    best = run = 0
    for i in range(1, period + 1):
        if marks[(start + i) % period]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def gap_max(config: ClockConfig) -> int:
    """The worst gap between two cycle starts: ``cycle_s + jitter_s + B + 1`` (spec §7.2)."""
    return config.cycle_s + config.jitter_s + longest_blackout_run(config.blackout) + 1


def parse_clock(raw: Mapping | None, p: Problems) -> ClockConfig | None:
    """Parse and bound-check the ``clock`` section. Records every problem in ``p``."""
    if raw is None:
        return None
    before = len(p.items)   # a child collector shares its parent's list: judge only this section's own problems
    cycle_s = integer(raw, "cycle_s", p, minimum=1)
    jitter_s = integer(raw, "jitter_s", p, minimum=0)
    valid_for_s = integer(raw, "valid_for_s", p, minimum=1)
    windows: list[BlackoutWindow] = []
    blackout = raw.get("blackout")
    if not isinstance(blackout, list):
        p.add(p.at("blackout"), "must be a list of {every_s, offset_s, length_s}")
    else:
        for i, item in enumerate(blackout):
            q = p.child("blackout").child(i)
            if not isinstance(item, Mapping):
                q.add(q.prefix, "must be an object")
                continue
            every = integer(item, "every_s", q, minimum=1)
            offset = integer(item, "offset_s", q, minimum=0)
            length = integer(item, "length_s", q, minimum=1)
            if None not in (every, offset, length):
                if length >= every:
                    q.add(q.prefix, f"length_s {length} must be shorter than every_s {every}")
                else:
                    windows.append(BlackoutWindow(every, offset % every, length))
    if None in (cycle_s, jitter_s, valid_for_s) or len(p.items) > before:
        return None
    config = ClockConfig(cycle_s, jitter_s, valid_for_s, tuple(windows))
    if not jitter_s < cycle_s / 2:
        p.add(p.at("jitter_s"), f"must be < cycle_s / 2 = {cycle_s / 2}")
    try:
        worst = gap_max(config)
    except ValueError as exc:
        p.add(p.at("blackout"), str(exc))
        return None
    if worst > valid_for_s / 4:
        p.add(p.at("valid_for_s"), f"gap_max {worst}s must be <= valid_for_s / 4 = {valid_for_s / 4:g}s "
                                   f"(a single slow cycle must never make the directive look stale)")
    if worst > HEARTBEAT_GAP_MAX_S:
        p.add(p.at("cycle_s"), f"gap_max {worst}s must be <= {HEARTBEAT_GAP_MAX_S}s so the Supervisor never sees a "
                               f"live Admin's heartbeat as wedged")
    return None if len(p.items) > before else config


class AdminClock:
    """Schedules cycle starts. Times are wall-clock epoch seconds (``time.time()``)."""

    def __init__(self, config: ClockConfig, rng: random.Random | None = None):
        self.config = config
        self._rng = rng if rng is not None else random.SystemRandom()   # unseeded: restarts never phase-lock

    def is_blocked(self, t: float) -> bool:
        return blocked(t, self.config.blackout)

    def release(self, t: float) -> float:
        """``t`` if it is free, else the end of the blackout containing it plus ``U(0, 1)`` s (repeated, since the end
        of one window may fall inside another)."""
        for _ in range(10_000):
            covering = [w for w in self.config.blackout if w.blocks(t)]
            if not covering:
                return t
            t = max(w.end_after(t) for w in covering) + self._rng.uniform(0.0, 1.0)
        raise RuntimeError("blackout windows never release")   # unreachable for a validated config

    def next_start(self, previous_start: float, now: float) -> float:
        """The next cycle start after one that began at ``previous_start``: the jittered period, moved out of any
        blackout; after an overrun (``now`` already past it) the next free instant from ``now``."""
        c = self.config
        target = previous_start + c.cycle_s + self._rng.uniform(-c.jitter_s, c.jitter_s)
        return self.release(max(target, now))
