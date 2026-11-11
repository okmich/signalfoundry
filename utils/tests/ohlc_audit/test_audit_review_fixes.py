"""Regression tests for the 2026-11-11 code reviews of the signalfoundry promotion (findings R1-R19, SPEC §17)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit.audit import SymbolReport, audit_broker, market_event_spikes
from okmich_quant_utils.ohlc_audit.clock import ClockEra, EraShiftClock
from okmich_quant_utils.ohlc_audit.enums import Action, AssetClass, AttemptResult, FindingKind, RemediationStatus, Severity, Verdict
from okmich_quant_utils.ohlc_audit.findings import Finding
from okmich_quant_utils.ohlc_audit.profile import BrokerProfile, FetchSource, SymbolSpec
from okmich_quant_utils.ohlc_audit.profiles.fxpig import SESSIONS
from okmich_quant_utils.ohlc_audit.handshake import handshake
from okmich_quant_utils.ohlc_audit.refetch import worse_findings
from okmich_quant_utils.ohlc_audit.registry import COLUMNS
from okmich_quant_utils.ohlc_audit.remediate import _held, assess, remediate_broker
from okmich_quant_utils.ohlc_audit.snapshot import history_diff, snapshot_broker
from okmich_quant_utils.ohlc_audit.state import StateLayout
from okmich_quant_utils.ohlc_audit.symbol_data import SymbolData

from .test_audit_detectors import SYMS, _frames, _inject, _profile
from .test_audit_refetch import CFG, T0, FakeBroker
from .test_audit_remediate import _setup

M5 = pd.Timedelta(minutes=5)


def _registry(tmp_path, *rows):
    p = tmp_path / "reg.csv"
    p.write_text(",".join(COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def _staged(run, s):
    df = pd.read_parquet(run.symbols[s].staging_path)
    df.index = df.index.tz_localize(None).as_unit("ns")
    return df


def test_r1_swapped_rows_are_sorted_not_deleted(tmp_path):
    frames = _frames()
    df = frames["EURUSD"]
    order = np.arange(len(df))
    order[[1000, 1001]] = [1001, 1000]
    frames["EURUSD"] = df.iloc[order]
    rep = audit_broker(profile=_profile(tmp_path), frames=frames)
    assert not [f for f in rep.symbols["EURUSD"].findings if f.kind is FindingKind.CLOCK_OVERLAP]
    profile, _, state = _setup(tmp_path, frames)
    r = remediate_broker(profile=profile, state_dir=state).symbols["EURUSD"]
    assert r.reordered and r.rows_out == r.rows_in and r.dropped == {}


def test_r2_off_grid_stamp_drops_only_its_own_row(tmp_path):
    frames = _frames()
    df = frames["EURUSD"]
    t = df.index[2000]
    df.index = df.index.where(df.index != t, t + pd.Timedelta(seconds=30))
    profile, _, state = _setup(tmp_path, frames)
    run = remediate_broker(profile=profile, state_dir=state)
    r = run.symbols["EURUSD"]
    assert r.dropped == {"off_grid": 1} and r.rows_out == r.rows_in - 1
    assert t + M5 in _staged(run, "EURUSD").index                                    # the next, valid bar survives


def _shifted_week(tmp_path, window_kind, action, start, end):
    profile, _, state = _setup(tmp_path, _inject(_frames()))
    reg = _registry(tmp_path, f"w1,GBPUSD,{start},{end},{window_kind},{action},partial note,2026-11-11,2026-11-11")
    return remediate_broker(profile=profile, state_dir=state, registry_path=reg)


def test_r3_partial_registry_overlap_keeps_review(tmp_path):
    run = _shifted_week(tmp_path, "mixed_clock", "inspect", "2024-04-30 12:00", "2024-04-30 12:05")   # 5 min of a shifted week
    assert run.before.symbols["GBPUSD"].verdict is Verdict.REVIEW
    assert run.symbols["GBPUSD"].status is RemediationStatus.REVIEW


def test_r3_outage_row_cannot_register_a_clock_finding(tmp_path):
    run = _shifted_week(tmp_path, "outage", "refetch", "2024-04-28 00:00", "2024-05-05 00:00")       # covers it, wrong type
    assert run.symbols["GBPUSD"].status is RemediationStatus.REVIEW


def _finding(kind, sev, a, b, n):
    t0 = pd.Timestamp("2024-04-10")
    return Finding("BTCUSDT", kind, sev, "D13", t0 + a * M5, t0 + b * M5, None, None, n, Action.NONE)


def test_r4_refetch_verification_rejects_worsening():
    before = [_finding(FindingKind.BAD_PRICES, Severity.MAJOR, 0, 4, 4)]
    escalated = [_finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 0, 4, 4)]
    grown = [_finding(FindingKind.BAD_PRICES, Severity.MAJOR, 0, 40, 40)]
    same = [_finding(FindingKind.BAD_PRICES, Severity.MAJOR, 0, 4, 4)]
    assert worse_findings(before, escalated) == escalated
    assert worse_findings(before, grown) == grown
    assert worse_findings(before, same) == []


def test_r5_major_bad_prices_are_held():
    f = _finding(FindingKind.BAD_PRICES, Severity.MAJOR, 0, 4, 4)            # D13 MAJOR uses Action.NONE
    rep = SymbolReport("BTCUSDT", "crypto", 100, f.start_raw, f.end_raw, {}, [f], Verdict.REVIEW)
    assert _held(rep) == [f]


def test_r5_status_after_remediation():
    judged = _finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 0, 4, 4)          # bars exposed once overlaps are dropped
    broken = _finding(FindingKind.OHLC_VIOLATION, Severity.CRITICAL, 10, 11, 1)
    kept = _finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 20, 24, 4)
    kept.registry_id = "inspect-1"                                                  # a recorded decision to keep the bars
    assert assess([], [judged])[0] is RemediationStatus.REVIEW
    assert assess([], [broken])[0] is RemediationStatus.FAILED_VERIFY
    assert assess([], [kept])[0] is RemediationStatus.VERIFIED
    assert assess([judged], [judged])[0] is RemediationStatus.REVIEW                # held window: not a failure


def test_r6_findings_crossing_the_cutoff_are_blind(tmp_path):
    df = _frames()["EURUSD"]
    cut = df.index[5000]
    sd = SymbolData.build(_profile(tmp_path), "EURUSD", df, blind_after=cut)
    assert sd.from_positions(FindingKind.STALE_RUN, Severity.MINOR, "D10", 4990, 5010, Action.FLAG).blind
    assert not sd.from_positions(FindingKind.STALE_RUN, Severity.MINOR, "D10", 4980, 4990, Action.FLAG).blind


def test_r7_dropped_prefix_and_suffix_are_targets_without_margin(tmp_path):
    truth = _frames()
    idx = truth["EURUSD"].index
    profile, _, state = _setup(tmp_path, truth)
    reg = _registry(tmp_path, f"head,EURUSD,{idx[0]},{idx[24]},replay,drop,head,2026-11-11,2026-11-11",
                    f"tail,EURUSD,{idx[-24]},{idx[-1] + M5},replay,drop,tail,2026-11-11,2026-11-11")
    broker = FakeBroker(truth)
    run = remediate_broker(profile=profile, state_dir=state, registry_path=reg, adapter=broker, refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert set(att["cause"]) == {"registry:head", "registry:tail"}
    assert set(att["result"]) == {AttemptResult.NO_MARGIN.value} and not att["counted"].any()
    assert broker.calls == 0                                                         # not asked: no margin to verify against


def test_r8_dropping_every_bar_is_a_per_symbol_outcome(tmp_path):
    frames = _frames()
    idx = frames["EURUSD"].index
    profile, _, state = _setup(tmp_path, frames)
    reg = _registry(tmp_path, f"all,EURUSD,{idx[0]},{idx[-1] + M5},replay,drop,everything,2026-11-11,2026-11-11")
    run = remediate_broker(profile=profile, state_dir=state, registry_path=reg)
    assert run.symbols["EURUSD"].status is RemediationStatus.EMPTY and run.symbols["EURUSD"].rows_out == 0
    assert run.symbols["EURUSD"].staging_path == "" and run.symbols["GBPUSD"].status is RemediationStatus.VERIFIED


def test_r9_missing_timestamps_are_quarantined(tmp_path):
    frames = _frames()
    df = frames["EURUSD"]
    df.index = df.index.where(np.arange(len(df)) != 3000, pd.NaT)
    rep = audit_broker(profile=_profile(tmp_path), frames=frames)
    nat = [f for f in rep.symbols["EURUSD"].findings if f.kind is FindingKind.MISSING_STAMP]
    assert len(nat) == 1 and nat[0].severity is Severity.CRITICAL and nat[0].n_bars == 1
    profile, _, state = _setup(tmp_path, frames)
    run = remediate_broker(profile=profile, state_dir=state)
    r = run.symbols["EURUSD"]
    assert r.dropped == {"missing_stamp": 1} and r.rows_out == r.rows_in - 1
    staged = _staged(run, "EURUSD")
    assert not staged.index.hasnans and staged.index.is_monotonic_increasing


# ---- second review (R10-R13) ---------------------------------------------------------------------------------------------

H = pd.Timedelta(hours=1)


def _margins_case():
    """Stored bars with a 3 h hole; 12 margin bars a side (distinct OHLC per bar)."""
    stored = _frames()["EURUSD"]
    t0 = stored.index[3000]
    stored = stored.drop(stored.loc[t0:t0 + 3 * H - M5].index)
    i = stored.index.searchsorted(t0)
    return stored, stored.index[i - 12:i], stored.index[i:i + 12]


def _fetch(stored, copies):
    """A fetch holding, for each (offset, stamps, n_kept), the stored bars at ``stamps`` moved by the offset, with all
    but ``n_kept`` of them altered."""
    parts = []
    for k, stamps, keep in copies:
        part = stored.loc[stamps].copy()
        part.iloc[keep:, part.columns.get_loc("close")] *= 1.01
        part.index = part.index + k
        parts.append(part)
    return pd.concat(parts).sort_index()


def test_r10_handshake_takes_the_offset_that_meets_both_thresholds():
    stored, left, right = _margins_case()
    zero = pd.Timedelta(0)
    # k = 0 reproduces 12/12 + 9/12 (21 votes, right side 75%); k = +1 h reproduces 10/12 + 10/12 (20 votes, both >= 80%)
    fetched = _fetch(stored, [(zero, left, 12), (zero, right, 9), (H, left, 10), (H, right, 10)])
    hs = handshake(stored, fetched, left, right, M5)
    assert hs.ok and hs.shift == H


def test_r10_handshake_prefers_no_shift_and_rejects_ambiguity():
    stored, left, right = _margins_case()
    zero = pd.Timedelta(0)
    both = _fetch(stored, [(zero, left, 12), (zero, right, 12), (H, left, 12), (H, right, 12)])
    assert handshake(stored, both, left, right, M5).shift == zero                   # no evidence the clock moved
    two = _fetch(stored, [(H, left, 12), (H, right, 12), (2 * H, left, 12), (2 * H, right, 12)])
    hs = handshake(stored, two, left, right, M5)
    assert not hs.ok and "ambiguous" in hs.reason


def test_r11_spikes_need_their_own_peers_within_a_bar(tmp_path):
    frames = _frames()
    t = frames["EURUSD"].index[4000]
    for sym, at in zip(SYMS, (t, t + 3 * M5, t + 6 * M5), strict=True):            # 01:00, 01:15, 01:30 on three symbols
        df = frames[sym]
        i = df.index.get_loc(at)
        c = df["close"].iloc[i] * 1.01
        df.iloc[i, df.columns.get_loc("close")] = c
        df.iloc[i, df.columns.get_loc("high")] = max(df["high"].iloc[i], c)
    rep = audit_broker(profile=_profile(tmp_path), frames=frames)
    kinds = [f.kind for r in rep.symbols.values() for f in r.findings if f.detector == "D9"]
    assert kinds == [FindingKind.SPIKE_SUSPECT] * 3


def test_r12_backfilled_older_history_is_not_a_rewrite(tmp_path):
    full = _frames()
    late = {s: df.iloc[2000:] for s, df in full.items()}
    profile, root, state = _setup(tmp_path, late)
    layout = StateLayout(state, profile.name, profile.bar_minutes).ensure()
    snap = snapshot_broker(profile, layout)
    _, _, _ = _setup(tmp_path / "again", full)                                     # the same data with older bars added
    for s in SYMS:
        (tmp_path / "again" / "md" / "raw" / "SYNTH" / "5" / f"{s}.parquet").replace(root / "raw" / "SYNTH" / "5" / f"{s}.parquet")
    assert history_diff(profile, snap) == []


def test_r13_a_snapshot_missing_requested_symbols_fails_loudly(tmp_path):
    profile, _, state = _setup(tmp_path, _frames())
    layout = StateLayout(state, profile.name, profile.bar_minutes).ensure()
    snap = snapshot_broker(profile, layout, ["EURUSD"])
    assert history_diff(profile, snap, ["EURUSD"]) == []
    with pytest.raises(ValueError, match="not in snapshot"):
        history_diff(profile, snap)


# ---- third review (R14-R19) ----------------------------------------------------------------------------------------------

NY = "America/New_York"


def test_r14_registry_decisions_never_excuse_integrity_failures():
    overlap = _finding(FindingKind.CLOCK_OVERLAP, Severity.CRITICAL, 0, 2, 2)
    overfull = _finding(FindingKind.OVERFULL, Severity.CRITICAL, 0, 300, 300)
    judged = _finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 0, 4, 4)
    for f in (overlap, overfull, judged):
        f.registry_id = "inspect-1"
    assert assess([], [overlap])[0] is RemediationStatus.FAILED_VERIFY
    assert assess([], [overfull])[0] is RemediationStatus.FAILED_VERIFY
    assert assess([], [judged])[0] is RemediationStatus.VERIFIED                    # a recorded judgment decision


def test_r15_refetch_verification_rejects_a_deeper_defect():
    before = _finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 0, 101, 101)
    before.evidence["max_dev_bp"] = 59.8
    deeper = _finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 0, 101, 101)
    deeper.evidence["max_dev_bp"] = 953.1
    same = _finding(FindingKind.BAD_PRICES, Severity.CRITICAL, 0, 101, 101)
    same.evidence["max_dev_bp"] = 59.8
    assert worse_findings([before], [deeper]) == [deeper]
    assert worse_findings([before], [same]) == []


def test_r15_magnitude_has_no_tolerance():
    # The user's ruling (2026-11-12): any increase rejects, however small, interleave size included.
    before = _finding(FindingKind.INTERLEAVE, Severity.MAJOR, 0, 50, 50)
    before.evidence["x_scale"] = 3.0
    nudged = _finding(FindingKind.INTERLEAVE, Severity.MAJOR, 0, 50, 50)
    nudged.evidence["x_scale"] = 3.0001
    assert worse_findings([before], [nudged]) == [nudged]


def _seam_setup(tmp_path, frames, seam, shift_after):
    """Like ``_setup``, with a second clock era from raw stamp ``seam`` (the files keep their NY+7 stamps)."""
    profile, root, state = _setup(tmp_path, frames)
    clock = EraShiftClock(NY, (ClockEra(pd.Timestamp("2000-01-01"), 7.0), ClockEra(seam, shift_after)))
    symbols = {s: SymbolSpec(s, s, AssetClass.FX, "fx") for s in SYMS}
    return BrokerProfile("SYNTH", 5, clock, symbols, {"fx": SESSIONS["fx"]}, FetchSource("x", "y"), raw_root=root / "raw"), root, state


def test_r16_every_copy_of_an_overlapping_stamp_is_dropped(tmp_path):
    frames = _frames()
    df = frames["EURUSD"]
    seam = df.index[4000]                                  # from here the clock is NY+8: the next hour overlaps the last
    dup = df.loc[[seam + 2 * M5]].copy()
    dup[["open", "high", "low", "close"]] *= 1.0001         # a second copy of an overlapping stamp
    frames["EURUSD"] = pd.concat([df, dup])                 # the copy is stored last: the planner used to prefer it
    profile, _, state = _seam_setup(tmp_path, frames, seam, 8.0)
    run = remediate_broker(profile=profile, state_dir=state)
    assert seam + 2 * M5 not in _staged(run, "EURUSD").index
    assert not [f for f in run.after.symbols["EURUSD"].findings if f.kind is FindingKind.CLOCK_OVERLAP]


def test_r17_dropped_tail_stays_a_target_at_an_era_boundary(tmp_path):
    truth = _frames()
    idx = truth["EURUSD"].index
    profile, _, state = _seam_setup(tmp_path, truth, idx[-1] + M5, 9.0)       # forward seam right after the last bar
    reg = _registry(tmp_path, f"tail,EURUSD,{idx[-24]},{idx[-1] + M5},replay,drop,tail,2026-11-11,2026-11-11")
    run = remediate_broker(profile=profile, state_dir=state, registry_path=reg, adapter=FakeBroker(truth), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert list(att["cause"]) == ["registry:tail"] and list(att["result"]) == [AttemptResult.NO_MARGIN.value]


def test_r18_market_event_matching_is_not_quadratic():
    t0, n = pd.Timestamp("2024-01-01", tz="UTC"), 20_000
    syms = ("EURUSD", "GBPUSD", "AUDUSD")
    lone = [Finding(syms[i % 3], FindingKind.SPIKE_SUSPECT, Severity.MINOR, "D9", None, None, t0 + i * 3 * M5, t0 + i * 3 * M5 + M5,
                    1, Action.REFETCH) for i in range(n)]                         # one symbol at a time, 15 min apart
    trio = [Finding(s, FindingKind.SPIKE_SUSPECT, Severity.MINOR, "D9", None, None, t0 - H, t0 - H + M5, 1, Action.REFETCH) for s in syms]
    started = time.perf_counter()
    events = market_event_spikes(lone + trio, dict.fromkeys(syms, "fx"), {"fx": 3}, M5)
    assert time.perf_counter() - started < 2.0
    assert sorted(f.symbol for f in events) == sorted(syms) and all(f.start_utc == t0 - H for f in events)


def test_r19_an_empty_run_removes_the_previous_staging(tmp_path):
    frames = _frames()
    idx = frames["EURUSD"].index
    profile, _, state = _setup(tmp_path, frames)
    first = remediate_broker(profile=profile, state_dir=state).symbols["EURUSD"]
    assert Path(first.staging_path).is_file() and Path(first.quality_path).is_file()
    reg = _registry(tmp_path, f"all,EURUSD,{idx[0]},{idx[-1] + M5},replay,drop,everything,2026-11-11,2026-11-11")
    second = remediate_broker(profile=profile, state_dir=state, registry_path=reg).symbols["EURUSD"]
    assert second.status is RemediationStatus.EMPTY
    assert not Path(first.staging_path).exists() and not Path(first.quality_path).exists()

