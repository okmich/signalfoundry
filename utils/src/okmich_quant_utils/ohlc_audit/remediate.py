"""P3 remediation into the state folder (SPEC §6.1-6.3, 6.7, 6.8 up to verification; amended by §13).

Per symbol, never touching ``raw/``:
1. Sort by raw stamp (out-of-order bars keep their stamps), then drop duplicate stamps, keeping the first.
2. Drop, with a recorded reason:
   - registry DROP windows;
   - PROPOSED clock-override windows (refetched in P4, not restamped);
   - mechanical detector findings, listed or not (copied blocks, out of session, broken OHLC, off-grid or future
     stamps, clock overlaps).
3. Hold, never drop: unlisted judgment findings (clock misalignment, weekly stamp drift, bad prices vs reference).
   The symbol goes to REVIEW until the registry records a decision.
4. Prices and spread values are never edited; quality flags mark what a consumer may want to mask.
5. Splice in the fill store: verified broker bars won by earlier refetches (fills.py).
6. With a fetch adapter (P4, refetch.py): refetch holes, dropped windows and spike suspects through the overlap
   handshake, verify each splice, store the winners and update the hole ledger.
7. Verify: re-audit the staged copy. 0 CRITICAL detector findings (outside held windows) means VERIFIED.

Every staged row is a raw row or a verified broker row from the fill store, never a value edited by formula. Columns,
dtypes, the index name and its tz label are kept exactly. The same raw files and fill store give byte-identical staged
files (SPEC A10).
"""
from __future__ import annotations

import dataclasses as dc
import json
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .audit import BrokerAuditReport, SymbolReport, audit_broker
from .clock import wall_clock
from .enums import (COUNTED_RESULTS, AttemptResult, FindingKind, HoleStatus, OverrideStatus, QualityFlag, RemediationStatus,
                    Severity, SlotState, TargetKind, Timeframe, WindowAction)
from .fetch import FetchAdapter
from .fills import REPLACE, FillStore, compose, merge_rows
from .findings import Finding
from .handshake import PRICE_COLUMNS
from .ledger import HoleLedger
from .policy import JUDGMENT, MECHANICAL, REVIEW_KINDS
from .profile import BrokerProfile
from .profiles import get_profile
from .refetch import RefetchConfig, RefetchRun, run_refetch
from .registry import BadWindow, load_registry, windows_for
from .state import StateLayout, resolve_state_dir, sha256
from .symbol_data import SymbolData

log = logging.getLogger(__name__)

__all__ = ["remediate_broker", "RemediationRun", "SymbolRemediation", "MECHANICAL", "JUDGMENT", "plan_symbol", "quality_flags"]
NEAR_DROPPED_BARS = 12
_WINDOW_FLAGS = {FindingKind.SPIKE_SUSPECT: QualityFlag.SPIKE_SUSPECT, FindingKind.MARKET_EVENT: QualityFlag.MARKET_EVENT,
                 FindingKind.STALE_RUN: QualityFlag.STALE_RUN, FindingKind.ZERO_VOLUME: QualityFlag.ZERO_VOLUME,
                 FindingKind.DST_FOLD: QualityFlag.DST_FOLD, FindingKind.INTERLEAVE: QualityFlag.INTERLEAVE_SUSPECT}


@dc.dataclass
class SymbolRemediation:
    symbol: str
    file: str
    rows_in: int
    rows_out: int
    reordered: bool
    duplicates_removed: int
    dropped: dict[str, int]                 # reason -> bars ("registry:<id>", "override:<symbol>:<date>", "<finding kind>")
    held: list[dict]                        # unlisted judgment findings kept for review
    flags: dict[str, int]                   # quality flag -> bars
    status: RemediationStatus
    remaining_critical: dict[str, int]      # CRITICAL detector findings left after remediation (outside held windows)
    refetched: int = 0                      # staged rows from the fill store (filled holes + replaced spikes)
    replaced: int = 0                       # of which replaced spikes
    fetch: dict[str, int] = dc.field(default_factory=dict)   # this run's refetch results: result -> targets
    input_sha256: str = ""
    output_sha256: str = ""
    staging_path: str = ""
    quality_path: str = ""


@dc.dataclass
class RemediationRun:
    broker: str
    bar_minutes: int
    run_at: pd.Timestamp
    data_root: str
    state_root: str
    registry: str | None
    blind_after: pd.Timestamp | None
    written: bool
    run_dir: str
    symbols: dict[str, SymbolRemediation]
    before: BrokerAuditReport = dc.field(repr=False)
    after: BrokerAuditReport | None = dc.field(repr=False)        # None when every symbol came out EMPTY
    refetch: RefetchRun | None = dc.field(default=None, repr=False)
    ledger: HoleLedger | None = dc.field(default=None, repr=False)
    fetch_source: str = ""

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.symbols.values():
            top = ", ".join(f"{k}={v}" for k, v in sorted(r.dropped.items(), key=lambda kv: -kv[1])[:3])
            fetched = " ".join(f"{k}={v}" for k, v in sorted(r.fetch.items()))
            rows.append({"symbol": r.symbol, "rows_in": r.rows_in, "rows_out": r.rows_out, "dropped": sum(r.dropped.values()),
                         "duplicates": r.duplicates_removed, "reordered": r.reordered, "refetched": r.refetched, "dropped_top": top,
                         "held": len(r.held), "fetch": fetched, "status": r.status.value})
        return pd.DataFrame(rows)

    def manifest(self) -> dict:
        def sym(r: SymbolRemediation) -> dict:
            d = dc.asdict(r)
            d["status"] = r.status.value
            return d
        return {"tool": "raw-data-audit remediate", "broker": self.broker, "bar_minutes": self.bar_minutes, "run_at": str(self.run_at),
                "data_root": self.data_root, "state_root": self.state_root, "registry": self.registry,
                "registry_sha256": sha256(Path(self.registry)) if self.registry and Path(self.registry).is_file() else None,
                "blind_after": str(self.blind_after) if self.blind_after is not None else None, "written": self.written,
                "symbols": {s: sym(r) for s, r in self.symbols.items()}, "fetch": self.fetch_manifest()}

    def attempts_frame(self) -> pd.DataFrame:
        """This run's refetch attempts, one row per target (evidence blanked after the blind-after cutoff)."""
        if self.refetch is None:
            return pd.DataFrame()
        rows = []
        for t, o in self.refetch.attempts:
            row = self.ledger.match(t) if self.ledger is not None else None
            counted = bool(row is not None and row.last_counted == self.run_at and o.result in COUNTED_RESULTS)
            blind = self.blind_after is not None and t.end_raw > self.blind_after
            rows.append({"id": t.id, "symbol": t.symbol, "kind": t.kind.value, "cause": t.cause, "start_raw": t.start_raw, "end_raw": t.end_raw,
                         "expected": t.expected, "result": o.result.value, "filled": o.filled, "shift_minutes": o.shift_minutes,
                         "counted": counted, "attempts": row.attempts if row else 0, "status": row.status.value if row else "",
                         "evidence": "" if blind else o.evidence})
        return pd.DataFrame(rows)

    def fetch_manifest(self) -> dict | None:
        if self.refetch is None:
            return None
        rf = self.refetch
        reps = []
        for s, rows in rf.won.items():
            for ts, r in rows[rows["_kind"] == REPLACE].iterrows():
                if self.blind_after is None or ts < self.blind_after:
                    reps.append({"symbol": s, "raw": str(ts), "old": [float(r[f"_old_{c}"]) for c in PRICE_COLUMNS],
                                 "new": [float(r[c]) for c in PRICE_COLUMNS]})
        att = self.attempts_frame()
        return {"source": self.fetch_source, "calls": rf.calls, "seconds": round(rf.seconds, 1), "aborted": rf.aborted,
                "targets": len(rf.seen), "skipped_permanent": [t.id for t in rf.skipped], "replacements": reps,
                "attempts": att.astype({"start_raw": str, "end_raw": str}).to_dict("records") if len(att) else []}


# ---- planning ----------------------------------------------------------------------------------------------------------

def _decision_windows(rep: SymbolReport, profile: BrokerProfile, registry: list[BadWindow]) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Time windows dropped by decision: registry DROP rows and PROPOSED clock overrides."""
    out = [(w.from_raw, w.to_raw, f"registry:{w.id}") for w in windows_for(registry, rep.symbol, {WindowAction.DROP})]
    out += [(ov.start_raw, ov.end_raw, f"override:{rep.symbol}:{ov.start_raw:%Y-%m-%d}") for ov in profile.clock.overrides
            if rep.symbol in ov.symbols and ov.status is OverrideStatus.PROPOSED]
    return out


def _drop_windows(rep: SymbolReport, profile: BrokerProfile, registry: list[BadWindow]) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Every dropped stretch as a time window, for NEAR_DROPPED flags and refetch targeting. (What is dropped is decided
    by ``plan_symbol``: mechanical defects by exact row.) Rows without a timestamp have no window."""
    mech = [(f.start_raw, f.end_raw, f.kind.value) for f in rep.detector_findings() if f.kind in MECHANICAL and pd.notna(f.start_raw)]
    return _decision_windows(rep, profile, registry) + mech


def _held(rep: SymbolReport) -> list[Finding]:
    """Unregistered CRITICAL/MAJOR judgment findings (clock, stamp drift, bad prices), whatever their action: remediation
    never resolves these itself, so the bars are kept and the symbol waits for a registry decision."""
    return [f for f in rep.detector_findings() if f.kind in JUDGMENT and f.severity in (Severity.CRITICAL, Severity.MAJOR)
            and not f.registry_id]


def assess(held: list[Finding], after: list[Finding]) -> tuple[RemediationStatus, Counter, list[Finding]]:
    """Status of a staged copy from the held findings and the re-audit's detector findings.

    - FAILED_VERIFY: a CRITICAL non-judgment finding remains (an integrity defect: overlap, overfull, broken OHLC...).
      Neither a registry decision nor a held window excuses it: those record judgments, not broken data.
    - REVIEW: findings were held, or an unregistered judgment finding (CRITICAL/MAJOR) or an unregistered MAJOR review
      finding shows after remediation outside the held windows. Removing mechanically bad rows can expose bars that are
      only judged bad: those need a decision too.
    Returns (status, remaining CRITICAL kinds, every finding awaiting review).
    """
    windows = [(f.start_raw, f.end_raw) for f in held]
    remaining = Counter(f.kind.value for f in after if f.severity is Severity.CRITICAL and f.kind not in JUDGMENT)
    open_ = [f for f in after if not f.registry_id and not any(f.start_raw < b and f.end_raw > a for a, b in windows)]
    unresolved = [f for f in open_ if (f.kind in JUDGMENT and f.severity in (Severity.CRITICAL, Severity.MAJOR))
                  or (f.kind in REVIEW_KINDS and f.severity is Severity.MAJOR)]
    review = held + unresolved
    status = RemediationStatus.FAILED_VERIFY if remaining else RemediationStatus.REVIEW if review else RemediationStatus.VERIFIED
    return status, remaining, review


def plan_symbol(wall: pd.DatetimeIndex, rep: SymbolReport, profile: BrokerProfile, registry: list[BadWindow]):
    """Positions (into the original frame) to keep, in raw-stamp order, plus the bookkeeping.

    - Registry DROP rows and PROPOSED overrides drop every row stamped inside their window.
    - Mechanical defects drop exactly the rows their finding names (rows without a timestamp included).
    - Duplicate stamps keep one copy, the first one that has no defect if there is one.
    """
    n = len(wall)
    ns = wall.asi8
    stamped = ~np.asarray(pd.isna(wall))
    reason = np.full(n, "", dtype=object)
    for a, b, why in _decision_windows(rep, profile, registry):
        hit = stamped & (reason == "") & (ns >= a.value) & (ns < b.value)
        reason[hit] = why
    for f in rep.detector_findings():
        if f.kind in MECHANICAL and f.rows is not None and len(f.rows):
            r = f.rows[reason[f.rows] == ""]
            reason[r] = f.kind.value
    bad = reason != ""
    cand = np.flatnonzero(stamped)
    order = cand[np.lexsort((cand, bad[cand], ns[cand]))]          # by stamp; a clean copy before a defective one
    s = ns[order]
    dup_sorted = np.r_[False, s[1:] == s[:-1]]
    dup = np.zeros(n, dtype=bool)
    dup[order[dup_sorted]] = True
    drop = bad & ~dup
    keep = order[~dup_sorted & ~bad[order]]
    return keep, int(dup.sum()), dict(Counter(reason[drop])), bool((np.diff(ns[stamped]) < 0).any())


# ---- quality flags -----------------------------------------------------------------------------------------------------

def quality_flags(sd: SymbolData, after: SymbolReport, drop_windows: list[tuple[pd.Timestamp, pd.Timestamp, str]],
                  held: list[Finding]) -> np.ndarray:
    raw = sd.raw
    flags = np.zeros(len(raw), dtype=np.int32)

    def mark(a, b, flag):
        i0, i1 = raw.searchsorted(a), raw.searchsorted(b)
        flags[i0:i1] |= int(flag)

    for f in after.detector_findings():
        if f.kind in _WINDOW_FLAGS:
            mark(f.start_raw, f.end_raw, _WINDOW_FLAGS[f.kind])
        elif f.kind is FindingKind.BAD_PRICES and f.severity is Severity.MAJOR:
            mark(f.start_raw, f.end_raw, QualityFlag.REFERENCE_DIVERGENCE)
        elif f.kind in (FindingKind.OUTAGE, FindingKind.HOLE):
            i = raw.searchsorted(f.end_raw)
            if i < len(raw):
                flags[i] |= int(QualityFlag.AFTER_GAP)
    near = pd.Timedelta(minutes=NEAR_DROPPED_BARS * sd.profile.bar_minutes)
    for a, b, _ in drop_windows:
        mark(a - near, a, QualityFlag.NEAR_DROPPED)
        mark(b, b + near, QualityFlag.NEAR_DROPPED)
    for f in held:
        mark(f.start_raw, f.end_raw, QualityFlag.HELD_FOR_REVIEW)
    if "spread" in sd.df:
        s = sd.df["spread"].to_numpy(dtype=float)
        flags[s < 0] |= int(QualityFlag.SPREAD_INVALID)
        flags[s == 0] |= int(QualityFlag.SPREAD_ZERO)
    flags[sd.bar_states() == SlotState.THIN.value] |= int(QualityFlag.THIN_SLOT)
    return flags


def refetch_flags(sd: SymbolData, used: pd.DataFrame, ledger: HoleLedger | None) -> np.ndarray:
    """REFETCHED / RESTAMPED on staged rows that came from the fill store; SPIKE_CONFIRMED on spikes the broker re-served."""
    flags = np.zeros(len(sd.raw), dtype=np.int32)
    if len(used):
        pos = sd.raw.get_indexer(used.index)
        ok = pos >= 0
        flags[pos[ok]] |= int(QualityFlag.REFETCHED)
        moved = ok & (used["_shift_minutes"].to_numpy() != 0)
        flags[pos[moved]] |= int(QualityFlag.RESTAMPED)
    if ledger is not None:
        confirmed = [r.start_raw for r in ledger.rows.values() if r.symbol == sd.symbol and r.kind is TargetKind.SPIKE
                     and r.last_result == AttemptResult.CONFIRMED.value and r.status in (HoleStatus.OPEN, HoleStatus.PERMANENT)]
        if confirmed:
            pos = sd.raw.get_indexer(pd.DatetimeIndex(confirmed))
            flags[pos[pos >= 0]] |= int(QualityFlag.SPIKE_CONFIRMED)
    return flags


def _flag_counts(flags: np.ndarray) -> dict[str, int]:
    out = {}
    for f in QualityFlag:
        n = int(((flags & int(f)) != 0).sum()) if f is not QualityFlag.NONE else 0
        if n:
            out[f.name] = n
    return out


# ---- writing -----------------------------------------------------------------------------------------------------------

def _compression(path: Path) -> str:
    try:
        md = pq.ParquetFile(path).metadata
        return md.row_group(0).column(0).compression.lower() if md.num_row_groups else "snappy"
    except Exception:
        return "snappy"


def _write_atomic(df: pd.DataFrame, path: Path, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        df.to_parquet(tmp, compression=compression)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---- entry point -------------------------------------------------------------------------------------------------------

def remediate_broker(broker: str | None = None, timeframe: Timeframe = Timeframe.M5, symbols: Iterable[str] | None = None,
                     profile: BrokerProfile | None = None, data_dir: str | Path | None = None, state_dir: str | Path | None = None,
                     registry_path: str | Path | None = None, blind_after: pd.Timestamp | None = None, write: bool = True,
                     adapter: FetchAdapter | None = None, refetch: RefetchConfig | None = None, now: pd.Timestamp | None = None
                     ) -> RemediationRun:
    """Remediate into ``<state>/<broker>/<tf>/staging`` (``write=False``: plan, fetch, verify and report; write nothing).

    ``adapter`` (an opened FetchAdapter) turns on the P4 refetch; ``refetch`` tunes it. ``now`` (UTC) stamps the run
    and the ledger (tests pass it to step through the 24 h cool-down).
    """
    if profile is None:
        if not broker:
            raise ValueError("pass a broker name or a profile")
        profile = get_profile(broker, data_dir)
    if profile.bar_minutes != int(timeframe):
        raise ValueError(f"{profile.name} profile is for {profile.bar_minutes}-minute bars, not {int(timeframe)}")
    data_root = profile.data_dir.parent.parent.parent if profile.raw_root is None else profile.raw_root.parent   # raises if unset
    layout = StateLayout(resolve_state_dir(state_dir, data_root), profile.name, profile.bar_minutes)
    registry = load_registry(Path(registry_path)) if registry_path else []
    names = sorted(symbols or profile.symbols)
    run_at = (pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)).floor("s")
    store, ledger = FillStore(layout.fills), HoleLedger.load(layout.ledger)

    log.info("%s: auditing %d symbols before remediation", profile.name, len(names))
    before = audit_broker(profile=profile, symbols=names, registry=registry, registry_path=registry_path, blind_after=blind_after)
    originals, walls, keeps, plans, drops, held, stores, staged, used, spikes, gone = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for s in names:
        orig = pd.read_parquet(profile.path(s))
        wall = wall_clock(pd.DatetimeIndex(orig.index)).as_unit("ns")
        keep, dups, dropped, reordered = plan_symbol(wall, before.symbols[s], profile, registry)
        originals[s], walls[s], keeps[s], plans[s] = orig, wall, keep, (dups, dropped, reordered)
        drops[s] = _drop_windows(before.symbols[s], profile, registry)
        held[s] = _held(before.symbols[s])
        stores[s] = store.load(f"{profile.symbol(s).file_stem}.parquet")
        staged[s], used[s] = compose(orig, wall, keep, stores[s])
        removed = np.ones(len(orig), dtype=bool)
        removed[keep] = False
        removed &= ~wall.isin(wall[keep])                      # duplicate copies leave their stamp present: not dropped
        gone[s] = orig[removed]
        replaced = set(used[s].index[used[s]["_kind"] == REPLACE]) if len(used[s]) else set()
        spikes[s] = [f.start_raw for f in before.symbols[s].detector_findings()
                     if f.kind is FindingKind.SPIKE_SUSPECT and f.start_raw not in replaced]

    live = [s for s in names if len(staged[s])]                     # a symbol with every bar dropped stages nothing (EMPTY)
    fetched = None
    if adapter is not None:
        log.info("%s: refetching from %s", profile.name, adapter.name)
        held_windows = {s: [(f.start_raw, f.end_raw) for f in held[s]] for s in live}
        bar = pd.Timedelta(minutes=profile.bar_minutes)
        spans = {}                                                 # the original span: a dropped head or tail stays a target
        for s in live:                                             # UTC extrema of the observed stamps, then + bar in UTC (a
            u = profile.clock.to_utc(walls[s][~np.asarray(pd.isna(walls[s]))], s)        # raw + bar can cross a clock seam)
            spans[s] = (u.min(), u.max() + bar)
        fetched = run_refetch(profile, {s: staged[s] for s in live}, drops, held_windows, spikes, ledger, adapter, refetch, run_at,
                              dropped=gone, spans=spans)
        for s, rows in fetched.won.items():
            stores[s] = merge_rows(stores[s], rows)
            staged[s], used[s] = compose(originals[s], walls[s], keeps[s], stores[s])
        ledger.record(fetched.attempts, run_at, fetched.seen, live)
    log.info("%s: re-auditing the staged copies", profile.name)
    after = audit_broker(profile=profile, symbols=live, frames={s: staged[s] for s in live}, registry=registry,
                         registry_path=registry_path, blind_after=blind_after) if live else None

    run_dir = layout.runs / f"{run_at:%Y%m%dT%H%M%SZ}"
    if write:
        layout.ensure()
        run_dir.mkdir(parents=True, exist_ok=True)
    per_symbol_fetch: dict[str, Counter] = {s: Counter() for s in names}
    for t, o in (fetched.attempts if fetched else []):
        per_symbol_fetch[t.symbol][o.result.value] += 1
    results = {}
    for s in names:
        spec = profile.symbol(s)
        dups, dropped, reordered = plans[s]
        if s not in live:
            results[s] = SymbolRemediation(s, f"{spec.file_stem}.parquet", len(originals[s]), 0, reordered, dups, dropped,
                                           [], {}, RemediationStatus.EMPTY, {}, input_sha256=sha256(profile.path(s)) if write else "")
            if write:                                              # an earlier run's copy must not outlive this verdict
                for stale in (layout.staging / results[s].file, layout.quality / results[s].file):
                    if stale.exists():
                        stale.unlink()
                        log.warning("%s %s: removed the previous run's %s", profile.name, s, stale)
            log.warning("%s %s: every bar was dropped; nothing staged", profile.name, s)
            continue
        sd = SymbolData.build(profile, s, staged[s], blind_after)
        status, remaining, review = assess(held[s], after.symbols[s].detector_findings())
        flags = quality_flags(sd, after.symbols[s], drops[s], review) | refetch_flags(sd, used[s], ledger)
        n_rep = int((used[s]["_kind"] == REPLACE).sum()) if len(used[s]) else 0
        r = SymbolRemediation(s, f"{spec.file_stem}.parquet", len(originals[s]), len(staged[s]), reordered, dups, dropped,
                              [{"kind": f.kind.value, "detector": f.detector, "start_raw": str(f.start_raw), "end_raw": str(f.end_raw),
                                "evidence": f.as_row()["evidence"], "stage": "before" if f in held[s] else "after"} for f in review],
                              _flag_counts(flags), status, dict(remaining), len(used[s]), n_rep, dict(per_symbol_fetch[s]))
        if write:
            src = profile.path(s)
            stage_path = layout.staging / r.file
            qual_path = layout.quality / r.file
            _write_atomic(staged[s], stage_path, _compression(src))
            _write_atomic(pd.DataFrame({"quality": flags}, index=staged[s].index), qual_path, "snappy")
            if fetched is not None and s in fetched.won:
                store.save(r.file, stores[s])
            r.input_sha256, r.output_sha256 = sha256(src), sha256(stage_path)
            r.staging_path, r.quality_path = str(stage_path), str(qual_path)
        results[s] = r
        log.info("%s %s: %d -> %d bars (%d refetched), %s", profile.name, s, r.rows_in, r.rows_out, r.refetched, status.value)
    run = RemediationRun(profile.name, profile.bar_minutes, run_at, str(data_root), str(layout.root),
                         str(registry_path) if registry_path else None, pd.Timestamp(blind_after) if blind_after is not None else None,
                         write, str(run_dir) if write else "", results, before, after, fetched, ledger, adapter.name if adapter else "")
    if write:
        if fetched is not None:
            ledger.save(layout.ledger)
            run.attempts_frame().to_csv(run_dir / "refetch_attempts.csv", index=False)
        (run_dir / "manifest.json").write_text(json.dumps(run.manifest(), indent=2, default=str), encoding="utf-8")
        before.findings_frame().to_csv(run_dir / "findings_before.csv", index=False)
        (after.findings_frame() if after is not None else pd.DataFrame()).to_csv(run_dir / "findings_after.csv", index=False)
        run.summary_frame().to_csv(run_dir / "summary.csv", index=False)
    return run
