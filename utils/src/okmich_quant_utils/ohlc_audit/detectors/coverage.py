"""D3 missing bars, D4 out-of-session bars, D5 overfull weeks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding, runs
from ..symbol_data import SymbolData
from ..enums import Action, FindingKind, Severity, SlotState

MAJOR_MINUTES = 30


def missing(sd: SymbolData) -> tuple[list[Finding], dict]:
    """D3: runs of expected-but-absent slots, and one INFO finding per recognised holiday / early-close window."""
    grid, state = sd.grid, sd.grid.to_numpy()
    present = sd.present_mask()
    is_open, is_thin = state == SlotState.OPEN.value, state == SlotState.THIN.value
    bm, out = sd.profile.bar_minutes, []
    for a, b in runs(is_open & ~present):
        n = b - a
        kind = FindingKind.HOLE if n >= sd.bars_per_day else FindingKind.OUTAGE
        sev = Severity.MAJOR if n * bm >= MAJOR_MINUTES else Severity.MINOR
        out.append(sd.from_utc(kind, sev, "D3", grid.index[a], grid.index[b - 1] + sd.bar, n, Action.REFETCH, minutes=n * bm))
    for a, b in runs(is_thin & ~present):
        n = b - a
        if n * bm >= MAJOR_MINUTES:
            out.append(sd.from_utc(FindingKind.THIN_NO_TICK, Severity.MINOR, "D3", grid.index[a], grid.index[b - 1] + sd.bar, n, Action.REFETCH,
                                   minutes=n * bm))
    out += structural(sd, present)
    stats = {"open_slots": int(is_open.sum()), "open_present": int((is_open & present).sum()),
             "thin_slots": int(is_thin.sum()), "thin_present": int((is_thin & present).sum())}
    stats["coverage_pct"] = round(100 * stats["open_present"] / max(stats["open_slots"], 1), 3)
    return out, stats


def structural(sd: SymbolData, present: np.ndarray) -> list[Finding]:
    grid, out = sd.grid, []
    ns, state = grid.index.asi8, grid.to_numpy()
    lo, hi = grid.index[0], grid.index[-1] + sd.bar
    years = (lo.year - 1, hi.year + 1)
    for rule in sd.template.holiday_rules:
        for w0, w1, day in rule.windows(*years):
            if w1 <= lo or w0 >= hi:
                continue
            i0, i1 = np.searchsorted(ns, w0.value), np.searchsorted(ns, w1.value)
            sel = state[i0:i1] == rule.state.value
            if not sel.any():
                continue
            n_missing = int((sel & ~present[i0:i1]).sum())
            out.append(sd.from_utc(FindingKind.STRUCTURAL_CLOSURE, Severity.INFO, "D3", max(w0, lo), min(w1, hi), n_missing, Action.NONE,
                                   rule=rule.name, date=str(day), slots=int(sel.sum()), missing=n_missing,
                                   tolerated_bars=int((sel & present[i0:i1]).sum())))
    return out


def out_of_session(sd: SymbolData) -> list[Finding]:
    """D4: bars whose slot is CLOSED (market shut by the weekly schedule)."""
    closed = sd.bar_states() == SlotState.CLOSED.value
    return [sd.from_positions(FindingKind.OUT_OF_SESSION, Severity.CRITICAL, "D4", a, b, Action.DROP, bars=b - a) for a, b in runs(closed)]


def overfull(sd: SymbolData) -> list[Finding]:
    """D5: weeks holding more bars than slots the calendar allows. A cross-check: D1/D4 name the offending bars."""
    tz = sd.template.tz
    allowed = pd.Series(sd.grid.to_numpy() != SlotState.CLOSED.value, index=sd.grid.index)
    wk_slots = allowed.groupby(allowed.index.tz_convert(tz).tz_localize(None).to_period("W-SAT")).sum()
    bars = pd.Series(1, index=sd.utc)
    wk_bars = bars.groupby(sd.utc.tz_convert(tz).tz_localize(None).to_period("W-SAT")).sum()
    out = []
    for wk, n in wk_bars.items():
        cap = int(wk_slots.get(wk, 0))
        if n > cap:
            u0 = pd.Timestamp(wk.start_time).tz_localize(tz).tz_convert("UTC")
            u1 = pd.Timestamp(wk.end_time).ceil("D").tz_localize(tz).tz_convert("UTC")
            out.append(sd.from_utc(FindingKind.OVERFULL, Severity.CRITICAL, "D5", u0, u1, int(n), Action.NONE, bars=int(n), allowed=cap))
    return out
