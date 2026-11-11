"""Refetch (SPEC §6.4-6.5): fill holes and dropped windows, and re-ask spike suspects, from the broker itself.

Per symbol, on the staged copy (raw rows after drops, plus earlier verified fills):
1. **Targets.**
   - DROPPED: every run of missing OPEN slots that overlaps a window remediation dropped. Always asked.
   - MISSING: every other run of missing OPEN slots of at least ``min_hole_minutes`` (default 30, the MAJOR line).
     Shorter gaps are no-tick slots (3,514 of 3,634 on FXPIG 2026-11-11 are single bars in metals / JPN225 quiet
     hours); they keep their AFTER_GAP flag.
   - SPIKE: single-symbol spike suspects (market events are never touched).
   Targets the ledger marks PERMANENT are skipped unless ``retry_permanent``.
2. **Fetch** in chunks: targets whose windows (with 12 trusted margin bars a side, padded by the 12 h maximum clock
   shift) lie within a day of each other share one call. Transient errors are retried with backoff; a fetch holding
   under half the stored bars of its range is asked again (the terminal may still be syncing history).
3. **Handshake** per target (handshake.py), then the fetched bars are restamped onto the stored clock.
4. **Unchanged is not a fix.** Fetched bars identical (stamp and OHLC) to bars remediation dropped are refused: the
   broker still holds the defect (a replay, a fold, a mis-clocked week), and serving it again is no new evidence. If
   that leaves nothing, the attempt is CONFIRMED (counted, like a spike served unchanged). On the 2026-11-11 export, a
   broker replaying the raw files had 28 of 153 dropped windows pass verification without this rule.
5. **Splice and verify** per chunk: re-run D1, D2, D4-D8 (and D13 where a reference exists) on the chunk +-8 days. A new
   finding of a VERIFY_REJECT kind within a day of a target rejects that target. A replaced spike must no longer spike.
Nothing is written here: the caller decides (dry run or not).
"""
from __future__ import annotations

import contextlib
import dataclasses as dc
import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from .clock import wall_clock
from .detectors import alignment, content, coverage, integrity, reference
from .enums import SEVERITY_RANK, AssetClass, AttemptResult, FindingKind, TargetKind
from .fetch import FetchAdapter, TransientFetchError
from .fills import FILL, META_COLUMNS, OLD_COLUMNS, REPLACE, conform
from .findings import Finding, uncovered
from .handshake import MAX_SHIFT, MIN_MARGIN_BARS, PRICE_COLUMNS, handshake, restamp
from .ledger import HoleLedger, Outcome, Target
from .policy import VERIFY_REJECT
from .profile import BrokerProfile
from .symbol_data import SymbolData

log = logging.getLogger(__name__)
Window = tuple[pd.Timestamp, pd.Timestamp, str]


@dc.dataclass(frozen=True)
class RefetchConfig:
    min_hole_minutes: int = 30                      # raw-file holes shorter than this are no-tick gaps: flagged, not asked
    retry_permanent: bool = False
    margin_bars: int = MIN_MARGIN_BARS
    chunk_gap: pd.Timedelta = pd.Timedelta(days=1)
    chunk_span: pd.Timedelta = pd.Timedelta(days=14)
    verify_pad: pd.Timedelta = pd.Timedelta(days=8)
    near: pd.Timedelta = pd.Timedelta(days=1)
    retries: int = 3
    backoff: tuple[float, ...] = (2.0, 8.0)
    sync_retries: int = 2
    sync_wait: float = 2.0
    max_transient_chunks: int = 3                   # consecutive failed chunks before the run stops fetching


@dc.dataclass
class RefetchRun:
    attempts: list[tuple[Target, Outcome]]
    seen: list[Target]
    skipped: list[Target]                           # PERMANENT, not asked
    won: dict[str, pd.DataFrame]                    # per symbol: store rows won this run (naive stamps, raw schema + META)
    calls: int
    seconds: float
    aborted: str = ""


@dc.dataclass
class _Plan:
    target: Target
    left: pd.DatetimeIndex
    right: pd.DatetimeIndex
    lo: pd.Timestamp                                # fetch range (margins padded by the maximum clock shift)
    hi: pd.Timestamp


def naive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.DatetimeIndex(wall_clock(pd.DatetimeIndex(df.index)).as_unit("ns"), name="raw")
    return out


# ---- targets -----------------------------------------------------------------------------------------------------------

def find_targets(profile: BrokerProfile, symbol: str, staged: pd.DataFrame, drops: list[Window], spikes: list[pd.Timestamp],
                 cfg: RefetchConfig, span: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> list[Target]:
    """``span`` (UTC): the original file's span, so a dropped head or tail is still a missing run, not off the grid."""
    sd = SymbolData.build(profile, symbol, staged, span=span)
    missing, _ = coverage.missing(sd)
    out = []
    for f in missing:
        if f.kind not in (FindingKind.OUTAGE, FindingKind.HOLE):
            continue
        hit = [why for a, b, why in drops if f.start_raw < b and f.end_raw > a]
        if hit:
            out.append(Target(symbol, TargetKind.DROPPED, f.start_raw, f.end_raw, f.n_bars, hit[0]))
        elif f.n_bars * profile.bar_minutes >= cfg.min_hole_minutes:
            out.append(Target(symbol, TargetKind.MISSING, f.start_raw, f.end_raw, f.n_bars, f.kind.value))
    present = set(sd.raw.asi8.tolist())
    for t in spikes:
        if t.value in present:
            out.append(Target(symbol, TargetKind.SPIKE, t, t + sd.bar, 1, "D9"))
    return sorted(out, key=lambda t: (t.start_raw, t.kind.value))


def _trusted(work: pd.DataFrame, spikes: list[pd.Timestamp], held: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DatetimeIndex:
    """Stored stamps that may serve as handshake margins: not a spike suspect, not inside a held (suspect-clock) window."""
    idx = work.index
    bad = idx.isin(pd.DatetimeIndex(spikes)) if spikes else np.zeros(len(idx), dtype=bool)
    for a, b in held:
        bad |= (idx >= a) & (idx < b)
    return idx[~bad]


def _plan(t: Target, trusted: pd.DatetimeIndex, bar: pd.Timedelta, m: int) -> _Plan:
    i0 = trusted.searchsorted(t.start_raw)
    left = trusted[max(0, i0 - m):i0]
    i1 = trusted.searchsorted(t.end_raw if t.kind is not TargetKind.SPIKE else t.start_raw + bar)
    right = trusted[i1:i1 + m]
    lo = (left[0] if len(left) else t.start_raw) - MAX_SHIFT
    hi = (right[-1] + bar if len(right) else t.end_raw) + MAX_SHIFT
    return _Plan(t, left, right, lo, hi)


def _chunks(plans: list[_Plan], cfg: RefetchConfig) -> list[list[_Plan]]:
    out: list[list[_Plan]] = []
    for p in sorted(plans, key=lambda p: p.lo):
        if out:
            lo, hi = out[-1][0].lo, max(q.hi for q in out[-1])
            if p.lo - hi <= cfg.chunk_gap and max(hi, p.hi) - lo <= cfg.chunk_span:
                out[-1].append(p)
                continue
        out.append([p])
    return out


# ---- fetch with retries --------------------------------------------------------------------------------------------------

def as_raw(bars: pd.DataFrame) -> pd.DataFrame:
    """An adapter's bars on unique, sorted, naive ns raw stamps (a labelled index keeps its digits, the label is dropped)."""
    out = bars.copy()
    out.index = pd.DatetimeIndex(wall_clock(pd.DatetimeIndex(bars.index)).as_unit("ns"), name="raw")
    return out[~out.index.duplicated(keep="first")].sort_index(kind="stable")


def _fetch(adapter: FetchAdapter, symbol: str, lo: pd.Timestamp, hi: pd.Timestamp, stored: int, cfg: RefetchConfig,
           sleep: Callable[[float], None]) -> pd.DataFrame:
    for i in range(cfg.retries):
        try:
            bars = as_raw(adapter.fetch(symbol, lo, hi))
            break
        except TransientFetchError:
            if i == cfg.retries - 1:
                raise
            sleep(cfg.backoff[min(i, len(cfg.backoff) - 1)])
    for _ in range(cfg.sync_retries):
        if len(bars) >= 0.5 * stored:
            break
        sleep(cfg.sync_wait)
        try:
            again = as_raw(adapter.fetch(symbol, lo, hi))
        except TransientFetchError:
            break
        if len(again) > len(bars):
            bars = again
    return bars


# ---- verification --------------------------------------------------------------------------------------------------------

def _peer_names(profile: BrokerProfile, symbol: str, available: list[str]) -> list[str]:
    ac = profile.symbol(symbol).asset_class
    if ac is AssetClass.FX and len(symbol) == 6:
        fx = [s for s in available if profile.symbol(s).asset_class is AssetClass.FX and len(s) == 6]
        return alignment._fx_peers(fx).get(symbol, [])
    if ac in (AssetClass.METAL, AssetClass.US_INDEX):
        return [s for s in available if s != symbol and profile.symbol(s).asset_class is ac]
    if ac is AssetClass.EU_INDEX:
        return [s for s in available if profile.symbol(s).asset_class is AssetClass.US_INDEX]
    return []


def verify_findings(profile: BrokerProfile, symbol: str, df: pd.DataFrame, peers: dict[str, SymbolData]) -> tuple[list[Finding], SymbolData]:
    sd = SymbolData.build(profile, symbol, df)
    out = (integrity.timestamps(sd) + integrity.ohlc(sd) + coverage.out_of_session(sd) + coverage.overfull(sd)
           + content.repeated_blocks(sd) + content.alternation(sd) + alignment.event_stamps(sd))
    ref = profile.references.get(symbol)
    if ref is not None and ref.path.exists():
        out += reference.reference_check(sd, ref)
    if peers:
        out += [f for f in alignment.peer_alignment({symbol: sd, **peers}) if f.symbol == symbol]
    return out, sd


#: The evidence that measures how bad a finding is, per kind (absolute value; larger is worse).
MAGNITUDE = {FindingKind.BAD_PRICES: "max_dev_bp", FindingKind.INTERLEAVE: "x_scale", FindingKind.CLOCK_MISALIGNED: "lag_hours",
             FindingKind.EVENT_STAMP_DRIFT: "shift_hours"}


def _magnitude(f: Finding) -> float:
    key = MAGNITUDE.get(f.kind)
    try:
        return abs(float(f.evidence.get(key, 0.0))) if key else 0.0
    except (TypeError, ValueError):
        return 0.0


def worse_findings(before: list[Finding], after: list[Finding]) -> list[Finding]:
    """Findings of ``after`` that are new or worse than ``before``: no overlapping finding of the same kind, or, against
    every overlapping one, a higher severity, a larger magnitude (MAGNITUDE), more bars, or time outside their windows.
    An existing finding is no licence for a splice to deepen it: 59.8 bp to 953 bp over the same 101 bars is worse."""
    out = []
    for f in after:
        same = [g for g in before if g.kind is f.kind and g.overlaps(f.start_raw, f.end_raw)]
        if (not same or SEVERITY_RANK[f.severity] > max(SEVERITY_RANK[g.severity] for g in same)
                or _magnitude(f) > max(_magnitude(g) for g in same)
                or f.n_bars > sum(g.n_bars for g in same)
                or uncovered(f.start_raw, f.end_raw, [(g.start_raw, g.end_raw) for g in same]) > pd.Timedelta(0)):
            out.append(f)
    return out


def _splice(df: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` (fills, or replacements of existing stamps) spliced into ``df``; both on naive raw stamps."""
    if rows.empty:
        return df
    return pd.concat([df.drop(rows.index, errors="ignore"), rows[list(df.columns)]]).sort_index(kind="stable")


def _verify(profile: BrokerProfile, symbol: str, work: dict[str, pd.DataFrame], cands: dict[Target, pd.DataFrame],
            cfg: RefetchConfig) -> dict[Target, str]:
    lo = min(t.start_raw for t in cands) - cfg.verify_pad
    hi = max(t.end_raw for t in cands) + cfg.verify_pad
    w = work[symbol]
    before = w[(w.index >= lo) & (w.index < hi)]
    after = _splice(before, pd.concat(list(cands.values())))
    peers = {}
    for p in _peer_names(profile, symbol, sorted(work)):
        pw = work[p]
        sl = pw[(pw.index >= lo) & (pw.index < hi)]
        if len(sl) > 1:
            peers[p] = SymbolData.build(profile, p, sl)
    fb, _ = verify_findings(profile, symbol, before, peers) if len(before) > 1 else ([], None)
    fa, sd_after = verify_findings(profile, symbol, after, peers)
    new = worse_findings([g for g in fb if g.kind in VERIFY_REJECT], [f for f in fa if f.kind in VERIFY_REJECT])
    still_spiking = {f.start_raw for f in content.spikes(sd_after)}
    bad = {}
    for t in cands:
        hits = [f for f in new if f.overlaps(t.start_raw - cfg.near, t.end_raw + cfg.near)]
        if hits:
            bad[t] = "verify: splice creates " + ", ".join(sorted({f"{f.kind.value} ({f.detector})" for f in hits}))
        elif t.kind is TargetKind.SPIKE and t.start_raw in still_spiking:
            bad[t] = "verify: the refetched bar still spikes"
    return bad


# ---- one chunk -----------------------------------------------------------------------------------------------------------

def _meta(rows: pd.DataFrame, kind: str, t: Target, shift_minutes: int, now: pd.Timestamp, old: np.ndarray | None = None) -> pd.DataFrame:
    rows = rows.copy()
    rows["_kind"], rows["_target"], rows["_shift_minutes"], rows["_fetched_at"] = kind, t.id, shift_minutes, str(now)
    for c, v in zip(OLD_COLUMNS, old if old is not None else [np.nan] * len(OLD_COLUMNS), strict=True):
        rows[c] = float(v)
    return rows[[*rows.columns.difference(META_COLUMNS, sort=False), *META_COLUMNS]]


def _bar_keys(df: pd.DataFrame) -> list[tuple]:
    return list(zip(pd.DatetimeIndex(df.index).as_unit("ns").asi8.tolist(), *(df[c].to_numpy(dtype=float).tolist() for c in PRICE_COLUMNS),
                    strict=True))


def _process_chunk(profile: BrokerProfile, symbol: str, work: dict[str, pd.DataFrame], like: pd.DataFrame, plans: list[_Plan],
                   bars: pd.DataFrame, served_from: pd.Timestamp | None, cfg: RefetchConfig, now: pd.Timestamp,
                   dropped_keys: frozenset[tuple] = frozenset()) -> list[tuple[Target, Outcome, pd.DataFrame]]:
    bar = pd.Timedelta(minutes=profile.bar_minutes)
    w = work[symbol]
    results: dict[Target, Outcome] = {}
    cands: dict[Target, pd.DataFrame] = {}
    shifts: dict[Target, int] = {}
    refused: dict[Target, int] = {}
    for p in plans:
        t = p.target
        if bars.empty:
            note = f"; the terminal serves {symbol} from {served_from}" if served_from is not None else ""
            results[t] = Outcome(AttemptResult.EMPTY, evidence=f"no bars served in or around the window{note}")
            continue
        hs = handshake(w, bars, p.left, p.right, bar, cfg.margin_bars)
        if not hs.ok:
            results[t] = Outcome(AttemptResult.REJECTED_VERIFY, evidence=f"handshake: {hs.reason}")
            continue
        conf = conform(restamp(bars, hs), like)
        if isinstance(conf, str):
            results[t] = Outcome(AttemptResult.REJECTED_VERIFY, 0, hs.shift_minutes, conf)
            continue
        shifts[t] = hs.shift_minutes
        if t.kind is TargetKind.SPIKE:
            if t.start_raw not in conf.index:
                results[t] = Outcome(AttemptResult.EMPTY, 0, hs.shift_minutes, "the broker has no bar at the spike stamp")
                continue
            old = w.loc[t.start_raw, PRICE_COLUMNS].to_numpy(dtype=float)
            new = conf.loc[t.start_raw, PRICE_COLUMNS].to_numpy(dtype=float)
            if (old == new).all():
                results[t] = Outcome(AttemptResult.CONFIRMED, 0, hs.shift_minutes, "the broker serves the same bar")
            else:
                cands[t] = _meta(conf.loc[[t.start_raw]], REPLACE, t, hs.shift_minutes, now, old)
        else:
            inside = conf[(conf.index >= t.start_raw) & (conf.index < t.end_raw) & ~conf.index.isin(w.index)]
            same = np.array([k in dropped_keys for k in _bar_keys(inside)], dtype=bool) if dropped_keys and len(inside) else None
            if same is not None and same.any():
                refused[t] = int(same.sum())
                inside = inside[~same]
            if inside.empty and t in refused:
                results[t] = Outcome(AttemptResult.CONFIRMED, 0, hs.shift_minutes,
                                     f"the broker serves the {refused[t]} dropped bars unchanged")
            elif inside.empty:
                results[t] = Outcome(AttemptResult.EMPTY, 0, hs.shift_minutes, "the margins match but the broker has no bars inside")
            else:
                cands[t] = _meta(inside, FILL, t, hs.shift_minutes, now)
    while cands:                                    # until a pass is clean; each failing pass removes >= 1 candidate
        bad = _verify(profile, symbol, work, {t: r[list(like.columns)] for t, r in cands.items()}, cfg)
        if not bad:
            break
        for t, why in bad.items():
            results[t] = Outcome(AttemptResult.REJECTED_VERIFY, 0, shifts[t], why)
            del cands[t]
    out = []
    for p in plans:
        t = p.target
        if t in cands:
            rows = cands[t]
            k = shifts[t]
            moved = f", restamped by {k} min" if k else ""
            if t.kind is TargetKind.SPIKE:
                o = Outcome(AttemptResult.REPLACED, 1, k, f"the broker now serves a different bar{moved}")
            else:
                res = AttemptResult.FILLED if len(rows) >= t.expected else AttemptResult.PARTIAL
                also = f"; {refused[t]} re-served dropped bars refused" if t in refused else ""
                o = Outcome(res, len(rows), k, f"{len(rows)}/{t.expected} bars{moved}{also}")
            out.append((t, o, rows))
        else:
            out.append((t, results[t], pd.DataFrame()))
    return out


# ---- entry point ---------------------------------------------------------------------------------------------------------

def run_refetch(profile: BrokerProfile, staged: dict[str, pd.DataFrame], drops: dict[str, list[Window]],
                held: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], spikes: dict[str, list[pd.Timestamp]], ledger: HoleLedger,
                adapter: FetchAdapter, cfg: RefetchConfig | None = None, now: pd.Timestamp | None = None,
                sleep: Callable[[float], None] = time.sleep, dropped: dict[str, pd.DataFrame] | None = None,
                spans: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None) -> RefetchRun:
    """Refetch every target of the staged symbols. ``staged`` frames keep the raw index labelling; ``drops`` / ``held`` /
    ``spikes`` come from the remediation plan; ``dropped`` holds the raw bars remediation removed (any index labelling),
    which a fetch may not put back unchanged; ``spans`` (UTC) are the original files' spans; ``ledger`` decides what is
    skipped (it is not updated here). A target with fewer than ``margin_bars`` trusted bars on a side (the file's head or
    tail) is recorded as NO_MARGIN and not asked: a one-sided handshake can't prove the clock."""
    cfg = cfg or RefetchConfig()
    now = pd.Timestamp.now(tz="UTC").floor("s") if now is None else pd.Timestamp(now)
    t0 = time.perf_counter()
    names = sorted(staged)
    work = {s: naive(staged[s]) for s in names}
    like = {s: staged[s].iloc[:0] for s in names}
    bar = pd.Timedelta(minutes=profile.bar_minutes)
    seen, skipped, attempts = [], [], []
    won: dict[str, list[pd.DataFrame]] = {s: [] for s in names}
    calls_before = getattr(adapter, "calls", 0)
    aborted, failed_in_row = "", 0
    for s in names:
        targets = find_targets(profile, s, staged[s], drops.get(s, []), spikes.get(s, []), cfg, (spans or {}).get(s))
        seen += targets
        ask = [t for t in targets if ledger.should_fetch(t, cfg.retry_permanent)]
        skipped += [t for t in targets if t not in ask]
        if not ask:
            continue
        trusted = _trusted(work[s], spikes.get(s, []), held.get(s, []))
        gone = (dropped or {}).get(s)
        dropped_keys = frozenset(_bar_keys(naive(gone))) if gone is not None and len(gone) else frozenset()
        stem = profile.symbol(s).file_stem
        plans = [_plan(t, trusted, bar, cfg.margin_bars) for t in ask]
        for p in plans:
            short = [side for side, m in (("before", p.left), ("after", p.right)) if len(m) < cfg.margin_bars]
            if short:
                why = f"only {len(p.left)} / {len(p.right)} trusted bars before / after (need {cfg.margin_bars}): not asked"
                attempts.append((p.target, Outcome(AttemptResult.NO_MARGIN, evidence=why)))
        chunks = _chunks([p for p in plans if len(p.left) >= cfg.margin_bars and len(p.right) >= cfg.margin_bars], cfg)
        log.info("%s %s: %d targets in %d fetches", profile.name, s, len(ask), len(chunks))
        for chunk in chunks:
            lo, hi = chunk[0].lo, max(p.hi for p in chunk)
            if aborted:
                attempts += [(p.target, Outcome(AttemptResult.TRANSIENT, evidence=aborted)) for p in chunk]
                continue
            stored = int(((work[s].index >= lo) & (work[s].index < hi)).sum())
            try:
                bars = _fetch(adapter, stem, lo, hi, stored, cfg, sleep)
            except TransientFetchError as e:
                failed_in_row += 1
                attempts += [(p.target, Outcome(AttemptResult.TRANSIENT, evidence=str(e))) for p in chunk]
                if failed_in_row >= cfg.max_transient_chunks:
                    aborted = f"fetching stopped after {failed_in_row} failed fetches in a row: {e}"
                    log.warning("%s: %s", profile.name, aborted)
                continue
            failed_in_row = 0
            served = None
            if bars.empty:
                with contextlib.suppress(TransientFetchError):
                    served = adapter.served_from(stem)
            for t, o, rows in _process_chunk(profile, s, work, like[s], chunk, bars, served, cfg, now, dropped_keys):
                attempts.append((t, o))
                if not rows.empty:
                    won[s].append(rows)
                    work[s] = _splice(work[s], rows[list(like[s].columns)])
    won_frames = {s: pd.concat(v).sort_index(kind="stable") for s, v in won.items() if v}
    return RefetchRun(attempts, seen, skipped, won_frames, getattr(adapter, "calls", 0) - calls_before, time.perf_counter() - t0, aborted)
