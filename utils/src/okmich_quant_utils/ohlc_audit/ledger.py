"""Hole ledger (SPEC §6.4): one row per refetch target, persistent across runs, in ``<state>/<broker>/<tf>/holes.parquet``.

Counting rule (G3): a target becomes PERMANENT after 3 counted failed attempts.
- EMPTY, REJECTED_VERIFY and (spikes) CONFIRMED count. TRANSIENT never counts, and neither does progress (PARTIAL).
- At most one counted attempt per target per run, and counted attempts are >= 24 h apart, so a broker-side history repair
  has time to land. An attempt inside the cool-down is still made (it may fill the hole), it just doesn't count.
- PERMANENT targets are not refetched unless forced (``retry_permanent``).
A target found again in a later run matches its row by id, or, after a partial fill, by lying inside the row's window.
"""
from __future__ import annotations

import dataclasses as dc
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .enums import COUNTED_RESULTS, AttemptResult, HoleStatus, TargetKind

MAX_ATTEMPTS = 3
COOLDOWN = pd.Timedelta(hours=24)
_PROGRESS = frozenset({AttemptResult.FILLED, AttemptResult.PARTIAL, AttemptResult.REPLACED})
_FAIL_ORDER = (AttemptResult.REJECTED_VERIFY, AttemptResult.CONFIRMED, AttemptResult.EMPTY)


@dc.dataclass(frozen=True)
class Target:
    """A window to ask the broker for again."""

    symbol: str
    kind: TargetKind
    start_raw: pd.Timestamp        # inclusive raw stamp
    end_raw: pd.Timestamp          # exclusive
    expected: int                  # missing OPEN slots (missing / dropped), 1 for a spike
    cause: str                     # "outage", "hole", "registry:<id>", "override:<...>", a finding kind, "D9"

    @property
    def id(self) -> str:
        return f"{self.symbol}|{self.kind.value}|{self.start_raw:%Y-%m-%dT%H:%M}|{self.end_raw:%Y-%m-%dT%H:%M}"


@dc.dataclass
class Outcome:
    """What one attempt on one target gave."""

    result: AttemptResult
    filled: int = 0                # bars inserted (or 1 for a replaced spike)
    shift_minutes: int = 0         # handshake offset k
    evidence: str = ""


@dc.dataclass
class LedgerRow:
    id: str
    symbol: str
    kind: TargetKind
    cause: str
    start_raw: pd.Timestamp
    end_raw: pd.Timestamp
    expected: int
    first_seen: pd.Timestamp
    filled: int = 0
    attempts: int = 0                              # counted failed attempts
    last_attempt: pd.Timestamp | None = None
    last_counted: pd.Timestamp | None = None
    last_result: str = ""
    status: HoleStatus = HoleStatus.OPEN
    shift_minutes: int = 0
    evidence: str = ""

    def contains(self, t: Target) -> bool:
        return self.symbol == t.symbol and self.kind is t.kind and self.start_raw <= t.start_raw and t.end_raw <= self.end_raw


COLUMNS = [f.name for f in dc.fields(LedgerRow)]


def aggregate(outcomes: list[Outcome], all_targets_resolved: bool) -> AttemptResult:
    """One result per ledger row per run (a row may cover several residual targets after a partial fill)."""
    got = {o.result for o in outcomes}
    progress, transient = got & _PROGRESS, AttemptResult.TRANSIENT in got
    failed = [r for r in _FAIL_ORDER if r in got]
    if progress and not failed and not transient and all_targets_resolved:
        return AttemptResult.REPLACED if AttemptResult.REPLACED in got else AttemptResult.FILLED
    if progress:
        return AttemptResult.PARTIAL
    if transient:
        return AttemptResult.TRANSIENT
    return failed[0] if failed else AttemptResult.NO_MARGIN


class HoleLedger:
    def __init__(self, rows: Iterable[LedgerRow] = ()):
        self.rows: dict[str, LedgerRow] = {r.id: r for r in rows}

    # ---- persistence ------------------------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> HoleLedger:
        if not Path(path).is_file():
            return cls()
        df = pd.read_parquet(path)
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path}: not a hole ledger, missing columns {sorted(missing)}")
        rows = []
        for r in df.to_dict("records"):
            for k in ("last_attempt", "last_counted"):
                r[k] = None if pd.isna(r[k]) else pd.Timestamp(r[k])
            r["kind"], r["status"] = TargetKind(r["kind"]), HoleStatus(r["status"])
            r["start_raw"], r["end_raw"], r["first_seen"] = pd.Timestamp(r["start_raw"]), pd.Timestamp(r["end_raw"]), pd.Timestamp(r["first_seen"])
            rows.append(LedgerRow(**{k: r[k] for k in COLUMNS}))
        return cls(rows)

    def frame(self) -> pd.DataFrame:
        recs = []
        for r in sorted(self.rows.values(), key=lambda r: (r.symbol, r.start_raw, r.kind.value)):
            d = dc.asdict(r)
            d["kind"], d["status"] = r.kind.value, r.status.value
            recs.append(d)
        df = pd.DataFrame(recs, columns=COLUMNS)
        for k in ("first_seen", "last_attempt", "last_counted"):
            df[k] = pd.to_datetime(df[k], utc=True)
        for k in ("start_raw", "end_raw"):
            df[k] = pd.to_datetime(df[k]).astype("datetime64[ns]")
        return df

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
        os.close(fd)
        try:
            self.frame().to_parquet(tmp, index=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ---- per run ----------------------------------------------------------------------------------------------------

    def match(self, t: Target) -> LedgerRow | None:
        if t.id in self.rows:
            return self.rows[t.id]
        inside = [r for r in self.rows.values() if r.status is not HoleStatus.RESOLVED and r.contains(t)]
        return min(inside, key=lambda r: r.end_raw - r.start_raw) if inside else None

    def should_fetch(self, t: Target, retry_permanent: bool = False) -> bool:
        row = self.match(t)
        return row is None or row.status is not HoleStatus.PERMANENT or retry_permanent

    def record(self, attempts: list[tuple[Target, Outcome]], now: pd.Timestamp, seen: Iterable[Target], symbols: Iterable[str]) -> None:
        """Apply one run's attempts. ``seen``: every target found this run (fetched or not); open rows of ``symbols``
        that no target matched are RESOLVED (the export has the bars now)."""
        now = pd.Timestamp(now)
        by_row: dict[str, tuple[LedgerRow, list[Target], list[Outcome]]] = {}
        for t, o in attempts:
            row = self.match(t)
            if row is None:
                row = LedgerRow(t.id, t.symbol, t.kind, t.cause, t.start_raw, t.end_raw, t.expected, now)
                self.rows[row.id] = row
            by_row.setdefault(row.id, (row, [], []))
            by_row[row.id][1].append(t)
            by_row[row.id][2].append(o)
        for row, targets, outs in by_row.values():
            done = sum(o.filled for o in outs) >= sum(t.expected for t in targets)
            result = aggregate(outs, done)
            counted = result in COUNTED_RESULTS and (row.last_counted is None or now - row.last_counted >= COOLDOWN)
            if counted:
                row.attempts += 1
                row.last_counted = now
            row.last_attempt, row.last_result = now, result.value
            row.filled += sum(o.filled for o in outs)
            row.shift_minutes = next((o.shift_minutes for o in outs if o.shift_minutes), row.shift_minutes)
            row.evidence = "; ".join(dict.fromkeys(o.evidence for o in outs if o.evidence))
            if result in (AttemptResult.FILLED, AttemptResult.REPLACED):
                row.status = HoleStatus.FILLED
            elif row.attempts >= MAX_ATTEMPTS:
                row.status = HoleStatus.PERMANENT
            elif result is AttemptResult.PARTIAL:
                row.status = HoleStatus.PARTIAL
            elif row.status is not HoleStatus.PERMANENT:
                row.status = HoleStatus.OPEN
        matched = {r.id for t in seen if (r := self.match(t)) is not None}
        names = set(symbols)
        for row in self.rows.values():
            if row.symbol in names and row.id not in matched and row.status in (HoleStatus.OPEN, HoleStatus.PARTIAL, HoleStatus.PERMANENT):
                row.status = HoleStatus.RESOLVED

    def open_rows(self) -> list[LedgerRow]:
        return [r for r in self.rows.values() if r.status in (HoleStatus.OPEN, HoleStatus.PARTIAL)]
