"""Audit orchestration: per-symbol detectors, broker-level merges, registry matching, verdicts (SPEC §4-§5)."""
from __future__ import annotations

import dataclasses as dc
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .detectors import alignment, content, coverage, integrity, reference
from .findings import Finding, uncovered
from .policy import MISSING_DATA_KINDS, MISSING_DATA_WINDOWS, REVIEW_KINDS
from .profile import BrokerProfile
from .profiles import get_profile
from .registry import BadWindow, load_registry, windows_for
from .symbol_data import SymbolData
from .enums import SEVERITY_RANK, Action, FindingKind, OverrideStatus, Severity, Timeframe, Verdict, WindowAction, WindowKind

log = logging.getLogger(__name__)

#: Unregistered CRITICAL/MAJOR findings of these kinds need a human before remediation is trusted.
INTERLEAVE_BROKER_WIDE = 5          # D7: alternation on >= 5 symbols at once is a feed event, not news
MARKET_EVENT_MIN_SYMBOLS = 3        # D9: same-class spikes on >= 3 symbols are a market event
BROKER_WIDE_MIN_SYMBOLS = 3
_REGISTRY_SEVERITY = {WindowAction.DROP: Severity.CRITICAL, WindowAction.REFETCH: Severity.MAJOR, WindowAction.INSPECT: Severity.MINOR}
_REGISTRY_ACTION = {WindowAction.DROP: Action.DROP, WindowAction.REFETCH: Action.REFETCH, WindowAction.INSPECT: Action.NONE}


@dc.dataclass
class SymbolReport:
    symbol: str
    asset_class: str
    rows: int
    first_raw: pd.Timestamp
    last_raw: pd.Timestamp
    stats: dict
    findings: list[Finding]
    verdict: Verdict = Verdict.CLEAN

    def detector_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.detector not in ("registry", "override")]


@dc.dataclass
class BrokerAuditReport:
    broker: str
    bar_minutes: int
    run_at: pd.Timestamp
    blind_after_raw: pd.Timestamp | None
    registry_source: str | None
    symbols: dict[str, SymbolReport]
    events: pd.DataFrame

    def findings_frame(self) -> pd.DataFrame:
        rows = [f.as_row() for r in self.symbols.values() for f in r.findings]
        return pd.DataFrame(rows)

    def summary_frame(self) -> pd.DataFrame:
        out = []
        for r in self.symbols.values():
            det = r.detector_findings()
            row = {"symbol": r.symbol, "class": r.asset_class, "bars": r.rows, "first_raw": r.first_raw, "last_raw": r.last_raw,
                   "coverage_pct": r.stats.get("coverage_pct")}
            for sev in Severity:
                row[sev.value] = sum(1 for f in det if f.severity is sev)
            row["unregistered_crit_major"] = sum(1 for f in det if f.severity in (Severity.CRITICAL, Severity.MAJOR) and not f.registry_id)
            row["registry_windows"] = sum(1 for f in r.findings if f.detector == "registry")
            row["verdict"] = r.verdict.value
            out.append(row)
        return pd.DataFrame(out)


# ---- per symbol ------------------------------------------------------------------------------------------------------

def single_symbol_findings(sd: SymbolData) -> tuple[list[Finding], dict]:
    found = integrity.timestamps(sd) + integrity.ohlc(sd)
    missing, stats = coverage.missing(sd)
    found += missing + coverage.out_of_session(sd) + coverage.overfull(sd)
    found += content.repeated_blocks(sd) + content.alternation(sd) + content.spikes(sd) + content.stale_and_volume(sd) + content.spread(sd)
    found += alignment.event_stamps(sd)
    ref = sd.profile.references.get(sd.symbol)
    if ref is not None and ref.path.exists():
        found += reference.reference_check(sd, ref)
    return found, stats


def registry_findings(sd: SymbolData, registry: list[BadWindow]) -> list[Finding]:
    out = []
    for w in windows_for(registry, sd.symbol):
        mask = w.mask(sd.raw)
        u = sd.utc[mask]
        sev, act = _REGISTRY_SEVERITY[w.action], _REGISTRY_ACTION[w.action]
        out.append(Finding(sd.symbol, FindingKind.REGISTERED, sev, "registry", w.from_raw, w.to_raw, u.min() if len(u) else None,
                           u.max() + sd.bar if len(u) else None, int(mask.sum()), act, {"window_kind": w.kind.value, "note": w.evidence},
                           registry_id=w.id, blind=sd._blind(w.from_raw, w.to_raw)))
    for ov in sd.profile.clock.overrides:
        if sd.symbol in ov.symbols and ov.status is OverrideStatus.PROPOSED:
            mask = np.asarray((sd.raw >= ov.start_raw) & (sd.raw < ov.end_raw))
            u = sd.utc[mask]
            out.append(Finding(sd.symbol, FindingKind.PROPOSED_OVERRIDE, Severity.CRITICAL, "override", ov.start_raw, ov.end_raw,
                               u.min() if len(u) else None, u.max() + sd.bar if len(u) else None, int(mask.sum()), Action.DROP,
                               {"shift_hours": ov.shift_hours, "note": ov.note}, registry_id=f"override:{sd.symbol}:{ov.start_raw:%Y-%m-%d}",
                               blind=sd._blind(ov.start_raw, ov.end_raw)))
    return out


def _covers(k: Finding, f: Finding) -> bool:
    """May registry window ``k`` register finding ``f``? A missing-data row (outage / hole) records absent bars only."""
    return f.kind in MISSING_DATA_KINDS or WindowKind(k.evidence.get("window_kind", WindowKind.SUSPECT)) not in MISSING_DATA_WINDOWS


def _match_registry(findings: list[Finding], known: list[Finding], bar: pd.Timedelta) -> None:
    """A finding is registered only when compatible registry / override windows cover all of it but at most one bar.

    A partial overlap is not a decision about the rest: a 5-minute row must not silence a week-long clock finding. Such a
    finding stays unregistered (it is held for review) and names the windows that cover part of it.
    """
    for f in findings:
        if f.detector in ("registry", "override") or f.severity is Severity.INFO or pd.isna(f.start_raw):
            continue
        hits = [k for k in known if _covers(k, f) and f.overlaps(k.start_raw, k.end_raw)]
        if not hits:
            continue
        if uncovered(f.start_raw, f.end_raw, [(k.start_raw, k.end_raw) for k in hits]) <= bar:
            f.registry_id = hits[0].registry_id
        else:
            f.evidence["partly_registered"] = "+".join(sorted({k.registry_id for k in hits}))


def verdict_of(report: SymbolReport) -> Verdict:
    heavy = [f for f in report.findings if f.severity in (Severity.CRITICAL, Severity.MAJOR)]
    if not heavy:
        return Verdict.CLEAN
    if any(not f.registry_id and f.kind in REVIEW_KINDS for f in heavy if f.detector not in ("registry", "override")):
        return Verdict.REVIEW
    return Verdict.USABLE_AFTER_REMEDIATION


def audit_symbol(profile: BrokerProfile, symbol: str, df: pd.DataFrame | None = None, registry: list[BadWindow] | None = None,
                 blind_after: pd.Timestamp | None = None) -> SymbolReport:
    """Single-symbol audit (no peer-based checks: D8a needs the broker's other symbols). ``registry`` defaults to empty."""
    sd = SymbolData.build(profile, symbol, df, blind_after)
    found, stats = single_symbol_findings(sd)
    known = registry_findings(sd, registry or [])
    _match_registry(found, known, sd.bar)
    rep = SymbolReport(symbol, sd.spec.asset_class.value, len(sd.df), sd.raw.min(), sd.raw.max(), stats, found + known)
    rep.verdict = verdict_of(rep)
    return rep


# ---- broker level ----------------------------------------------------------------------------------------------------

def _clusters(items: list[Finding], pad: pd.Timedelta) -> list[list[Finding]]:
    """Group findings whose UTC windows overlap (after padding by ``pad``), across symbols."""
    items = sorted((f for f in items if f.start_utc is not None), key=lambda f: f.start_utc)
    groups: list[list[Finding]] = []
    end = None
    for f in items:
        if groups and f.start_utc - pad <= end:
            groups[-1].append(f)
            end = max(end, f.end_utc + pad)
        else:
            groups.append([f])
            end = f.end_utc + pad
    return groups


def _merge_cross_symbol(all_found: dict[str, list[Finding]], sds: dict[str, SymbolData]) -> None:
    # D7: alternation on many symbols at once is a feed event. Two correlated symbols whipsawing together (BTC + ETH on
    # 2026-08-10, both within 15 bp of Binance) is a market move, so a 2-member class doesn't count as broker-wide.
    class_size: dict[str, int] = {}
    for sd in sds.values():
        class_size[sd.spec.asset_class.value] = class_size.get(sd.spec.asset_class.value, 0) + 1
    inter = [f for fs in all_found.values() for f in fs if f.kind is FindingKind.INTERLEAVE]
    for g in _clusters(inter, pd.Timedelta(minutes=30)):
        if len({f.symbol for f in g}) >= INTERLEAVE_BROKER_WIDE:
            for f in g:
                f.severity, f.broker_wide = Severity.MAJOR, True
    # D9: a spike is a market event when same-class symbols spike within +-1 bar of IT (not of a chain of spikes: three
    # symbols spiking 15 minutes apart are three suspects, not one event).
    spikes = [f for fs in all_found.values() for f in fs if f.kind is FindingKind.SPIKE_SUSPECT]
    if spikes:
        class_of = {s: sd.spec.asset_class.value for s, sd in sds.items()}
        for f in market_event_spikes(spikes, class_of, class_size, next(iter(sds.values())).bar):
            f.kind, f.severity, f.action, f.broker_wide = FindingKind.MARKET_EVENT, Severity.INFO, Action.NONE, True


def market_event_spikes(spikes: list[Finding], class_of: dict[str, str], class_size: dict[str, int], tol: pd.Timedelta) -> list[Finding]:
    """The spikes backed by same-class spikes on enough symbols within ``tol`` of their own time (each spike judged on its
    own neighbourhood, never through a chain). Sorted by time with a two-sided window: O(n log n), not O(n^2)."""
    by_class: dict[str, list[Finding]] = {}
    for f in spikes:
        if f.start_utc is not None:
            by_class.setdefault(class_of[f.symbol], []).append(f)
    out = []
    for c, fs in by_class.items():
        need = min(MARKET_EVENT_MIN_SYMBOLS, class_size[c]) if class_size[c] >= 2 else MARKET_EVENT_MIN_SYMBOLS
        if len({f.symbol for f in fs}) < need:
            continue                                     # not enough symbols in this class ever spike: no event possible
        fs = sorted(fs, key=lambda f: f.start_utc)
        t = np.array([f.start_utc.value for f in fs], dtype=np.int64)
        lo = np.searchsorted(t, t - tol.value, side="left")
        hi = np.searchsorted(t, t + tol.value, side="right")
        for f, a, b in zip(fs, lo, hi, strict=True):
            if b - a >= need and len({g.symbol for g in fs[a:b]}) >= need:
                out.append(f)
    return out


def _broker_wide_events(reports: dict[str, SymbolReport]) -> pd.DataFrame:
    """Clusters of MAJOR/CRITICAL detector findings of one kind on >= 3 symbols (minor no-tick gaps are left out)."""
    rows = []
    det = [f for r in reports.values() for f in r.detector_findings() if f.severity in (Severity.CRITICAL, Severity.MAJOR)]
    by_kind: dict[FindingKind, list[Finding]] = {}
    for f in det:
        by_kind.setdefault(f.kind, []).append(f)
    for kind, fs in by_kind.items():
        for g in _clusters(fs, pd.Timedelta(minutes=30)):
            syms = sorted({f.symbol for f in g})
            if len(syms) < BROKER_WIDE_MIN_SYMBOLS:
                continue
            for f in g:
                f.broker_wide = True
            worst = max(g, key=lambda f: SEVERITY_RANK[f.severity]).severity
            rows.append({"kind": kind.value, "severity": worst.value, "start_raw": min(f.start_raw for f in g), "end_raw": max(f.end_raw for f in g),
                         "start_utc": min(f.start_utc for f in g), "end_utc": max(f.end_utc for f in g), "n_symbols": len(syms),
                         "symbols": ",".join(syms), "registered": all(f.registry_id for f in g), "blind": all(f.blind for f in g)})
    return pd.DataFrame(rows).sort_values("start_utc", ignore_index=True) if rows else pd.DataFrame(rows)


def audit_broker(broker: str | None = None, timeframe: Timeframe = Timeframe.M5, symbols: Iterable[str] | None = None,
                 profile: BrokerProfile | None = None, frames: dict[str, pd.DataFrame] | None = None, data_dir: str | Path | None = None,
                 registry_path: str | Path | None = None, registry: list[BadWindow] | None = None,
                 blind_after: pd.Timestamp | None = None) -> BrokerAuditReport:
    """Audit every (or the listed) symbol of a broker.

    Run inputs, none of them stored in the profile:
    - ``data_dir``: market-data root (argument, else OKMICH_QUANT_MARKET_DATA_DIR, else fail). Not needed when
      ``frames`` supplies the bars in memory.
    - ``registry_path`` / ``registry``: the curated bad-window list (CSV). Without one every finding is unregistered.
    - ``blind_after``: raw stamp after which findings report windows and kinds only (holdout protection).
    """
    if profile is None:
        if not broker:
            raise ValueError("pass a broker name or a profile")
        profile = get_profile(broker, data_dir, require_data=frames is None)
    if profile.bar_minutes != int(timeframe):
        raise ValueError(f"{profile.name} profile is for {profile.bar_minutes}-minute bars, not {int(timeframe)}")
    if registry is None:
        registry = load_registry(Path(registry_path)) if registry_path else []
    names = sorted(symbols or (frames.keys() if frames else profile.symbols))
    sds, found, stats = {}, {}, {}
    t0 = time.perf_counter()
    for i, s in enumerate(names, 1):
        sds[s] = SymbolData.build(profile, s, frames.get(s) if frames else None, blind_after)
        found[s], stats[s] = single_symbol_findings(sds[s])
        log.info("%s [%d/%d] %s: %d bars, %d findings (%.0fs)", profile.name, i, len(names), s, len(sds[s].df), len(found[s]),
                 time.perf_counter() - t0)
    log.info("%s: peer clock alignment over %d symbols", profile.name, len(sds))
    for f in alignment.peer_alignment(sds):
        found[f.symbol].append(f)
    _merge_cross_symbol(found, sds)
    reports = {}
    for s in names:
        known = registry_findings(sds[s], registry)
        _match_registry(found[s], known, sds[s].bar)
        sd = sds[s]
        reports[s] = SymbolReport(s, sd.spec.asset_class.value, len(sd.df), sd.raw.min(), sd.raw.max(), stats[s], found[s] + known)
    events = _broker_wide_events(reports)
    for r in reports.values():
        r.verdict = verdict_of(r)
    blind = pd.Timestamp(blind_after) if blind_after is not None else None
    source = str(registry_path) if registry_path else ("in-memory" if registry else None)
    return BrokerAuditReport(profile.name, profile.bar_minutes, pd.Timestamp.now(tz="UTC").floor("s"), blind, source, reports, events)
