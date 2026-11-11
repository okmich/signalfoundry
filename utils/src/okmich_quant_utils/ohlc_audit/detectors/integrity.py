"""D1 timestamp integrity, D2 OHLC integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding, group_positions, runs
from ..symbol_data import SymbolData
from ..enums import Action, FindingKind, Severity

C, D = Severity.CRITICAL, Action.DROP


def timestamps(sd: SymbolData, now: pd.Timestamp | None = None) -> list[Finding]:
    raw, out = sd.raw, []
    nat = sd.missing_stamp_finding()
    if nat is not None:
        out.append(nat)
    dup = raw.duplicated(keep="first")
    for a, b in runs(dup):
        out.append(sd.from_positions(FindingKind.DUPLICATE_STAMP, C, "D1", a, b, D, bars=b - a))
    dec = np.r_[False, np.diff(raw.asi8) < 0]
    for a, b in runs(dec):
        out.append(sd.from_positions(FindingKind.NON_MONOTONIC, C, "D1", a, b, D, bars=b - a))
    off = (raw.asi8 % sd.bar.value) != 0
    for a, b in runs(off):
        out.append(sd.from_positions(FindingKind.OFF_GRID, C, "D1", a, b, D, bars=b - a))
    now = now or pd.Timestamp.now(tz="UTC")
    fut = np.asarray(sd.utc > now + sd.bar)
    for a, b in runs(fut):
        out.append(sd.from_positions(FindingKind.FUTURE_STAMP, C, "D1", a, b, D, bars=b - a))
    # Two raw stamps landing on (or before) an already-used UTC slot: a seam, a DST fold, or a splice. Judged in raw-stamp
    # order, not stored order: rows that are merely out of order (NON_MONOTONIC above) are sorted by remediation, not
    # deleted. An overlap belongs to the timestamp, so the finding names every row at an overlapping stamp (duplicate
    # copies included: dedupe must not keep one of them).
    order = np.argsort(raw.asi8, kind="stable")
    rs, u = raw.asi8[order], sd.utc.asi8[order]
    first = np.r_[True, rs[1:] != rs[:-1]]                                 # first row of each distinct stamp
    uu = u[first]
    over_stamp = np.r_[False, uu[1:] <= np.maximum.accumulate(uu)[:-1]]
    over = over_stamp[np.cumsum(first) - 1]
    for a, b in group_positions(np.where(over)[0], max_gap=12):
        sel = a + np.flatnonzero(over[a:b + 1])
        out.append(sd.from_rows(FindingKind.CLOCK_OVERLAP, C, "D1", order[sel], D, bars=len(sel)))
    # bars in a raw hour the server clock passed through twice (DST fold, backward seam): may be merged
    for r0, r1, d in sd.profile.clock.ambiguous_raw(sd.symbol):
        inside = np.asarray((raw >= r0) & (raw < r1))
        for a, b in runs(inside):
            out.append(sd.from_positions(FindingKind.DST_FOLD, Severity.MAJOR, "D1", a, b, Action.NONE, bars=b - a,
                                         repeated_raw_hours=d, note="raw time repeated; bars may merge two passes"))
    return out


def ohlc(sd: SymbolData) -> list[Finding]:
    df, out = sd.df, []
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    nonfinite = ~(np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c))
    for a, b in runs(nonfinite):
        out.append(sd.from_positions(FindingKind.NON_FINITE, C, "D2", a, b, D, bars=b - a))
    with np.errstate(invalid="ignore"):
        nonpos = ((o <= 0) | (h <= 0) | (lo <= 0) | (c <= 0)) & ~nonfinite & sd.spec.positive_prices
        viol = ((h < np.maximum(o, c)) | (lo > np.minimum(o, c)) | (h < lo)) & ~nonfinite
    # Only MT5 tick_volume is checked. Other columns (IB volume / average / barCount, CCXT volume) are carried untouched:
    # IB reports volume = -1 for "not available" (FX midpoint bars), which is not a defect.
    if "tick_volume" in df:
        viol |= df["tick_volume"].to_numpy(dtype=float) < 0
    for a, b in runs(nonpos):
        out.append(sd.from_positions(FindingKind.NON_POSITIVE, C, "D2", a, b, D, bars=b - a))
    for a, b in runs(viol):
        out.append(sd.from_positions(FindingKind.OHLC_VIOLATION, C, "D2", a, b, D, bars=b - a))
    return out
