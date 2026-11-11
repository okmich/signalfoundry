"""P3: remediation into the state folder, quality flags, snapshots and D12 (SPEC A8 staging part, A9, A10)."""
import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from .test_audit_detectors import SYMS, _frames, _inject

from okmich_quant_utils.ohlc_audit.clock import ClockEra, EraShiftClock
from okmich_quant_utils.ohlc_audit.enums import AssetClass, FindingKind, QualityFlag, RemediationStatus
from okmich_quant_utils.ohlc_audit.profile import BrokerProfile, FetchSource, SymbolSpec
from okmich_quant_utils.ohlc_audit.profiles.fxpig import SESSIONS
from okmich_quant_utils.ohlc_audit.registry import COLUMNS
from okmich_quant_utils.ohlc_audit.remediate import remediate_broker
from okmich_quant_utils.ohlc_audit.snapshot import history_diff, snapshot_broker
from okmich_quant_utils.ohlc_audit.state import STATE_DIR_VAR, StateDirNotConfiguredError, StateLayout, resolve_state_dir, sha256

NY = "America/New_York"
LABEL = dt.timezone(dt.timedelta(hours=2))          # FXPIG files carry a fixed +02:00 label


def _setup(tmp_path, frames=None):
    root = tmp_path / "md"
    folder = root / "raw" / "SYNTH" / "5"
    folder.mkdir(parents=True)
    for s, df in (frames or _inject(_frames())).items():
        out = df.copy()
        out.index = pd.DatetimeIndex(out.index, name="date").tz_localize(LABEL)
        out.to_parquet(folder / f"{s}.parquet", compression="gzip")
    state = tmp_path / "state"
    state.mkdir()
    profile = BrokerProfile("SYNTH", 5, EraShiftClock(NY, (ClockEra(pd.Timestamp("2000-01-01"), 7.0),)),
                            {s: SymbolSpec(s, s, AssetClass.FX, "fx") for s in SYMS}, {"fx": SESSIONS["fx"]}, FetchSource("x", "y"),
                            raw_root=root / "raw")
    return profile, root, state


def test_mechanical_dropped_judgment_held(tmp_path):
    profile, _, state = _setup(tmp_path)
    run = remediate_broker(profile=profile, state_dir=state)
    eur, gbp, aud = (run.symbols[s] for s in ("EURUSD", "GBPUSD", "AUDUSD"))
    assert eur.dropped.get("repeated_block", 0) >= 12 and eur.dropped.get("out_of_session", 0) >= 24
    assert eur.status is RemediationStatus.VERIFIED
    assert aud.duplicates_removed == 1 and aud.status is RemediationStatus.VERIFIED
    assert gbp.status is RemediationStatus.REVIEW and gbp.held                       # unlisted clock shift: kept for a decision
    q = pd.read_parquet(state / "SYNTH" / "5" / "quality" / "GBPUSD.parquet")["quality"].to_numpy()
    assert ((q & int(QualityFlag.HELD_FOR_REVIEW)) != 0).sum() > 1000


def test_raw_untouched_and_schema_kept(tmp_path):
    profile, root, state = _setup(tmp_path)
    before = {p.name: sha256(p) for p in (root / "raw" / "SYNTH" / "5").glob("*.parquet")}
    run = remediate_broker(profile=profile, state_dir=state)
    assert {p.name: sha256(p) for p in (root / "raw" / "SYNTH" / "5").glob("*.parquet")} == before
    assert not list(root.rglob("staging")) and not list(root.rglob("*.json"))      # nothing written into the data tree
    for s in SYMS:
        orig = pd.read_parquet(root / "raw" / "SYNTH" / "5" / f"{s}.parquet")
        staged = pd.read_parquet(run.symbols[s].staging_path)
        assert list(staged.columns) == list(orig.columns) and (staged.dtypes == orig.dtypes).all()
        assert staged.index.name == orig.index.name and str(staged.index.tz) == str(orig.index.tz)
        assert staged.index.is_monotonic_increasing and staged.index.is_unique
        first = orig[~orig.index.duplicated(keep="first")]                          # rows are only selected, never edited
        pd.testing.assert_frame_equal(staged, first.loc[staged.index])


def test_idempotent_byte_identical(tmp_path):
    profile, _, state = _setup(tmp_path)
    a = remediate_broker(profile=profile, state_dir=state)
    b = remediate_broker(profile=profile, state_dir=state)
    for s in SYMS:
        assert a.symbols[s].output_sha256 == b.symbols[s].output_sha256
        assert sha256(Path(a.symbols[s].quality_path)) == sha256(Path(b.symbols[s].quality_path))


def test_registry_drop_is_applied_with_its_reason(tmp_path):
    profile, _, state = _setup(tmp_path)
    reg = tmp_path / "reg.csv"
    reg.write_text(",".join(COLUMNS) + "\n" + "w1,GBPUSD,2024-04-28 17:00,2024-05-04 03:00,mixed_clock,drop,shifted week,2026-11-11,2026-11-11\n",
                   encoding="utf-8")
    run = remediate_broker(profile=profile, state_dir=state, registry_path=reg)          # the window covers the whole +3 h week
    assert run.symbols["GBPUSD"].dropped.get("registry:w1", 0) > 1000
    assert run.symbols["GBPUSD"].status is RemediationStatus.VERIFIED                # decision recorded: no longer held


def test_dry_run_writes_nothing(tmp_path):
    profile, _, state = _setup(tmp_path)
    run = remediate_broker(profile=profile, state_dir=state, write=False)
    assert not run.written and not any(state.rglob("*"))


def test_state_dir_rules(tmp_path, monkeypatch):
    root = tmp_path / "md"
    (root / "raw").mkdir(parents=True)
    inside = root / "state"
    inside.mkdir()
    with pytest.raises(StateDirNotConfiguredError, match="inside the market-data tree"):
        resolve_state_dir(inside, root)
    monkeypatch.delenv(STATE_DIR_VAR, raising=False)
    with pytest.raises(StateDirNotConfiguredError, match="no state folder"):
        resolve_state_dir(None, root)
    with pytest.raises(StateDirNotConfiguredError, match="not an existing folder"):
        resolve_state_dir(tmp_path / "missing", root)
    elsewhere = tmp_path / "st"
    elsewhere.mkdir()
    monkeypatch.setenv(STATE_DIR_VAR, str(elsewhere))
    assert resolve_state_dir(None, root) == elsewhere


def test_quality_flags_spread_and_thin(tmp_path):
    frames = _frames()
    frames["EURUSD"].iloc[100:110, frames["EURUSD"].columns.get_loc("spread")] = 0
    profile, _, state = _setup(tmp_path, frames)
    run = remediate_broker(profile=profile, state_dir=state)
    q = pd.read_parquet(run.symbols["EURUSD"].quality_path)["quality"].to_numpy()
    assert ((q & int(QualityFlag.SPREAD_ZERO)) != 0).sum() == 10
    assert ((q & int(QualityFlag.THIN_SLOT)) != 0).sum() == 12 * 5 * 6                  # 17:00-18:00 NY, 5 days x 6 weeks


def test_snapshot_and_history_diff(tmp_path):
    profile, root, state = _setup(tmp_path, _frames())
    layout = StateLayout(state, profile.name, profile.bar_minutes).ensure()
    snap = snapshot_broker(profile, layout)
    assert history_diff(profile, snap) == []
    p = root / "raw" / "SYNTH" / "5" / "EURUSD.parquet"
    df = pd.read_parquet(p)
    df.iloc[500:505, df.columns.get_loc("close")] *= 1.001                              # rewritten bars
    df = df.drop(df.index[2000:2003])                                                   # vanished bars
    new_tail = df.iloc[-3:].copy()
    new_tail.index = new_tail.index + pd.Timedelta(days=30)                             # appended after the snapshot: fine
    pd.concat([df, new_tail]).to_parquet(p)
    found = history_diff(profile, snap)
    assert [f.kind for f in found] == [FindingKind.HISTORY_REWRITE] * 2
    assert found[0].evidence.get("changed") == 5 and found[1].evidence.get("vanished") == 3
