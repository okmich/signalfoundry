"""The finding record every detector emits, and small helpers shared by detectors."""
from __future__ import annotations

import dataclasses as dc
from typing import Any

import numpy as np
import pandas as pd

from .enums import Action, FindingKind, Severity


@dc.dataclass
class Finding:
    symbol: str
    kind: FindingKind
    severity: Severity
    detector: str                      # "D1" .. "D13", "registry", "override"
    start_raw: pd.Timestamp            # inclusive, raw wall-clock
    end_raw: pd.Timestamp              # exclusive
    start_utc: pd.Timestamp | None
    end_utc: pd.Timestamp | None
    n_bars: int
    action: Action
    evidence: dict[str, Any] = dc.field(default_factory=dict)
    registry_id: str = ""              # registry window (or override) this finding falls in
    broker_wide: bool = False          # >= 3 symbols share an overlapping finding of this kind
    blind: bool = False                # touches the holdout cutoff: report windows and kinds only (SPEC §8)
    # Row-level findings: positions of the defective rows in the frame as given (as stored). Remediation drops exactly
    # these rows, never everything inside the window: an off-grid 08:20:30 must not take the valid 08:25:00 with it.
    rows: np.ndarray | None = dc.field(default=None, repr=False, compare=False)

    def overlaps(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return self.start_raw < end and self.end_raw > start

    def as_row(self) -> dict[str, Any]:
        row = {f.name: getattr(self, f.name) for f in dc.fields(self) if f.name != "rows"}
        row["kind"], row["severity"], row["action"] = self.kind.value, self.severity.value, self.action.value
        row["evidence"] = "" if self.blind else ";".join(f"{k}={v}" for k, v in self.evidence.items())
        return row


def uncovered(start: pd.Timestamp, end: pd.Timestamp, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Timedelta:
    """How much of [start, end) the union of ``windows`` leaves uncovered."""
    gap, cursor = pd.Timedelta(0), start
    for a, b in sorted(windows):
        if b <= cursor:
            continue
        if a > cursor:
            gap += min(a, end) - cursor
        cursor = max(cursor, b)
        if cursor >= end:
            break
    return gap + max(end - cursor, pd.Timedelta(0))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist(), strict=True))


def group_positions(pos: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Group sorted integer positions into [first, last] clusters whose members are <= ``max_gap`` apart."""
    if len(pos) == 0:
        return []
    out, start, prev = [], int(pos[0]), int(pos[0])
    for p in pos[1:]:
        p = int(p)
        if p - prev > max_gap:
            out.append((start, prev))
            start = p
        prev = p
    out.append((start, prev))
    return out


def merge_windows(windows: list[tuple[pd.Timestamp, pd.Timestamp]], gap: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not windows:
        return []
    ws = sorted(windows)
    out = [list(ws[0])]
    for a, b in ws[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]
