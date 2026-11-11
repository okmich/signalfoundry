"""P4: refetch through the overlap handshake, the hole ledger and spike replacement (SPEC A5, A6, A7), with a fake broker."""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from .test_audit_detectors import SYMS, _frames
from .test_audit_remediate import _setup

from okmich_quant_utils.ohlc_audit.cli import main
from okmich_quant_utils.ohlc_audit.enums import HoleStatus, QualityFlag
from okmich_quant_utils.ohlc_audit.fetch import FetchUnavailableError, MT5FetchAdapter, TransientFetchError
from okmich_quant_utils.ohlc_audit.handshake import handshake
from okmich_quant_utils.ohlc_audit.ledger import HoleLedger
from okmich_quant_utils.ohlc_audit.refetch import RefetchConfig
from okmich_quant_utils.ohlc_audit.registry import COLUMNS
from okmich_quant_utils.ohlc_audit.remediate import remediate_broker

PX = ["open", "high", "low", "close"]
H = pd.Timedelta(hours=1)
NO_SHIFT = pd.Timedelta(0)
CFG = RefetchConfig(backoff=(0.0,), sync_wait=0.0)
T0 = pd.Timestamp("2026-11-12 09:00", tz="UTC")
WED = pd.Timestamp("2024-04-17 10:00") + 7 * H                         # raw stamp of an ordinary Wednesday morning
SPIKE_AT = pd.Timestamp("2024-04-23 12:00") + 7 * H


class FakeBroker:
    """Serves 'true' bars (naive raw stamps) per symbol, optionally distorted: a moved clock, a transform, failures."""

    name = "fake"

    def __init__(self, truth, shift=NO_SHIFT, transform=None, transient=0, empty=False):
        self.truth, self.shift, self.transform, self.transient, self.empty = truth, shift, transform, transient, empty
        self.calls = 0

    def open(self):
        pass

    def close(self):
        pass

    def fetch(self, symbol, start, end):
        self.calls += 1
        if self.transient:
            self.transient -= 1
            raise TransientFetchError("(-10005, 'IPC timeout')")
        df = self.truth[symbol].copy()
        df.index = df.index + self.shift
        if self.transform is not None:
            df = self.transform(df, symbol)
        if self.empty:
            df = df.iloc[:0]
        wall = df.index.tz_localize(None) if df.index.tz is not None else df.index
        return df[(wall >= start) & (wall < end)]

    def served_from(self, symbol):
        return self.truth[symbol].index.min()


def _drop(frames, sym, at, bars):
    out = {s: df.copy() for s, df in frames.items()}
    out[sym] = out[sym].drop(out[sym].loc[at:at + pd.Timedelta(minutes=5 * (bars - 1))].index)
    return out


def _spike(frames, syms):
    out = {s: df.copy() for s, df in frames.items()}
    for s in syms:
        df = out[s]
        i = df.index.get_loc(SPIKE_AT)
        c = df["close"].iloc[i] * 1.01
        df.iloc[i, df.columns.get_loc("close")] = c
        df.iloc[i, df.columns.get_loc("high")] = max(df["high"].iloc[i], c)
    return out


def _staged(run, sym):
    df = pd.read_parquet(run.symbols[sym].staging_path)
    df.index = df.index.tz_localize(None).as_unit("ns")
    return df


def _quality(run, sym, flag):
    q = pd.read_parquet(run.symbols[sym].quality_path)["quality"].to_numpy()
    return int(((q & int(flag)) != 0).sum())


def _ledger(state):
    return HoleLedger.load(state / "SYNTH" / "5" / "holes.parquet")


def _same_bars(staged, truth):
    assert (staged.index == truth.index.as_unit("ns")).all()
    np.testing.assert_array_equal(staged[PX].to_numpy(), truth[PX].to_numpy())


# ---- A5: handshake and verification -------------------------------------------------------------------------------------

def test_hole_filled_and_reapplied_without_fetch(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))
    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att[["kind", "result", "filled", "shift_minutes"]].values.tolist() == [["missing", "filled", 24, 0]]
    _same_bars(_staged(run, "EURUSD"), truth["EURUSD"])
    assert _quality(run, "EURUSD", QualityFlag.REFETCHED) == 24 and _quality(run, "EURUSD", QualityFlag.RESTAMPED) == 0
    assert [r.status for r in _ledger(state).rows.values()] == [HoleStatus.FILLED]
    again = remediate_broker(profile=profile, state_dir=state, now=T0 + H)          # no fetch: the fill store supplies the bars
    assert again.symbols["EURUSD"].refetched == 24
    assert again.symbols["EURUSD"].output_sha256 == run.symbols["EURUSD"].output_sha256


def test_broker_clock_shift_is_absorbed(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))
    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth, shift=H), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att[["result", "filled", "shift_minutes"]].values.tolist() == [["filled", 24, 60]]
    _same_bars(_staged(run, "EURUSD"), truth["EURUSD"])
    assert _quality(run, "EURUSD", QualityFlag.RESTAMPED) == 24


def test_adapter_units_and_labels_are_normalised(tmp_path):
    """Parquet loads stamps in us, and an adapter may hand back a labelled index: the handshake must still match."""
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))

    def as_stored(df, s):
        out = df.copy()
        out.index = pd.DatetimeIndex(out.index).as_unit("us").tz_localize("+02:00")
        return out

    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth, transform=as_stored), refetch=CFG, now=T0)
    assert run.attempts_frame()[["result", "filled", "shift_minutes"]].values.tolist() == [["filled", 24, 0]]
    _same_bars(_staged(run, "EURUSD"), truth["EURUSD"])


def test_price_disagreement_on_margins_is_rejected(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))
    other = FakeBroker(truth, transform=lambda df, s: df.assign(**{c: df[c] * 1.001 for c in PX}))
    run = remediate_broker(profile=profile, state_dir=state, adapter=other, refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att["result"].tolist() == ["rejected_verify"] and "handshake" in att["evidence"].iloc[0]
    assert len(_staged(run, "EURUSD")) == len(truth["EURUSD"]) - 24
    row = next(iter(_ledger(state).rows.values()))
    assert row.attempts == 1 and row.status is HoleStatus.OPEN


def test_interior_on_another_clock_is_rejected(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))

    def later_feed(df, s):                      # margins match, but the bars inside come from one hour later
        if s != "EURUSD":
            return df
        out = df.copy()
        inside = (out.index >= WED) & (out.index < WED + 2 * H)
        out.loc[inside, PX] = truth[s].reindex(out.index[inside] + H)[PX].to_numpy()
        return out

    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth, transform=later_feed), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att["result"].tolist() == ["rejected_verify"] and "repeated_block" in att["evidence"].iloc[0]
    assert run.symbols["EURUSD"].refetched == 0


def test_interleaved_fetch_is_rejected(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))

    def two_feeds(df, s):                       # bars inside alternate between two price levels
        out = df.copy()
        pos = np.where((out.index >= WED) & (out.index < WED + 2 * H))[0][::2]
        out.iloc[pos, [out.columns.get_loc(c) for c in PX]] *= 1.004
        return out

    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth, transform=two_feeds), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att["result"].tolist() == ["rejected_verify"] and "interleave" in att["evidence"].iloc[0]


def _bad_window(tmp_path, truth):
    """EURUSD with 3 h of bad prices that the registry drops."""
    bad = {s: df.copy() for s, df in truth.items()}
    w = (bad["EURUSD"].index >= pd.Timestamp("2024-04-24 10:00")) & (bad["EURUSD"].index < pd.Timestamp("2024-04-24 13:00"))
    bad["EURUSD"].loc[w, PX] *= 1.002
    reg = tmp_path / "reg.csv"
    reg.write_text(",".join(COLUMNS) + "\nw1,EURUSD,2024-04-24 10:00,2024-04-24 13:00,bad_prices,drop,off feed,2026-11-11,2026-11-11\n",
                   encoding="utf-8")
    return bad, reg


def test_dropped_registry_window_is_refetched(tmp_path):
    truth = _frames()
    bad, reg = _bad_window(tmp_path, truth)
    profile, _, state = _setup(tmp_path, bad)
    run = remediate_broker(profile=profile, state_dir=state, registry_path=reg, adapter=FakeBroker(truth), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att[["kind", "cause", "result", "filled"]].values.tolist() == [["dropped", "registry:w1", "filled", 36]]
    _same_bars(_staged(run, "EURUSD"), truth["EURUSD"])


def test_dropped_bars_served_back_unchanged_are_refused(tmp_path):
    """A broker still holding the defect (it serves exactly the bars remediation dropped) is no fix: CONFIRMED, counted."""
    truth = _frames()
    bad, reg = _bad_window(tmp_path, truth)
    profile, _, state = _setup(tmp_path, bad)
    run = remediate_broker(profile=profile, state_dir=state, registry_path=reg, adapter=FakeBroker(bad), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att[["kind", "result", "filled", "counted"]].values.tolist() == [["dropped", "confirmed", 0, True]]
    assert "36 dropped bars unchanged" in att["evidence"].iloc[0]
    assert run.symbols["EURUSD"].refetched == 0 and len(_staged(run, "EURUSD")) == len(truth["EURUSD"]) - 36


def test_handshake_unit_cases():
    truth = _frames()["EURUSD"]
    bar = pd.Timedelta(minutes=5)
    stored = truth.drop(truth.loc[WED:WED + 115 * pd.Timedelta(minutes=1)].index)
    i = stored.index.searchsorted(WED)
    left, right = stored.index[i - 12:i], stored.index[i:i + 12]
    assert handshake(stored, truth, left, right, bar).ok
    moved = truth.copy()
    moved.index = moved.index - 2 * H
    hs = handshake(stored, moved, left, right, bar)
    assert hs.ok and hs.shift == -2 * H
    assert not handshake(stored, truth, left[:5], right, bar).ok                         # too few margin bars
    assert "none of the margin bars" in handshake(stored, truth.iloc[:10], left, right, bar).reason


# ---- A6: hole ledger --------------------------------------------------------------------------------------------------

def test_empty_three_times_a_day_apart_goes_permanent(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))
    broker = FakeBroker(truth, empty=True)
    seen = []
    for t in (T0, T0 + 2 * H, T0 + 25 * H, T0 + 50 * H):
        remediate_broker(profile=profile, state_dir=state, adapter=broker, refetch=CFG, now=t)
        row = next(iter(_ledger(state).rows.values()))
        seen.append((row.attempts, row.status))
    assert seen == [(1, HoleStatus.OPEN), (1, HoleStatus.OPEN), (2, HoleStatus.OPEN), (3, HoleStatus.PERMANENT)]
    assert "serves EURUSD from" in row.evidence
    calls = broker.calls
    run = remediate_broker(profile=profile, state_dir=state, adapter=broker, refetch=CFG, now=T0 + 80 * H)
    assert broker.calls == calls and len(run.refetch.skipped) == 1                        # permanent: not asked again
    healed = FakeBroker(truth)
    remediate_broker(profile=profile, state_dir=state, adapter=healed, refetch=RefetchConfig(backoff=(0.0,), sync_wait=0.0,
                     retry_permanent=True), now=T0 + 81 * H)
    assert [r.status for r in _ledger(state).rows.values()] == [HoleStatus.FILLED]


def test_transient_errors_never_count_and_stop_the_run(tmp_path):
    truth = _frames()
    frames = truth
    for week in range(4):                                                               # four holes, four separate fetches
        frames = _drop(frames, "EURUSD", pd.Timestamp("2024-04-09 03:00") + pd.Timedelta(weeks=week), 12)
    profile, _, state = _setup(tmp_path, frames)
    broker = FakeBroker(truth, transient=10 ** 6)
    run = remediate_broker(profile=profile, state_dir=state, adapter=broker, refetch=CFG, now=T0)
    assert run.attempts_frame()["result"].tolist() == ["transient"] * 4
    assert "stopped" in run.refetch.aborted and broker.calls == 3 * CFG.retries          # the 4th fetch is never made
    assert {(r.attempts, r.status) for r in _ledger(state).rows.values()} == {(0, HoleStatus.OPEN)}


def test_filled_and_resolved_leave_the_open_set(tmp_path):
    truth = _frames()
    profile, root, state = _setup(tmp_path, _drop(_drop(truth, "EURUSD", WED, 24), "GBPUSD", WED, 24))
    remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth, empty=True), refetch=CFG, now=T0)
    assert len(_ledger(state).open_rows()) == 2
    full = truth["GBPUSD"].copy()                                                        # a new export has GBPUSD's bars
    full.index = pd.DatetimeIndex(full.index, name="date").tz_localize(pd.Timestamp("2024-01-01", tz="+02:00").tz)
    full.to_parquet(root / "raw" / "SYNTH" / "5" / "GBPUSD.parquet", compression="gzip")
    remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth), refetch=CFG, now=T0 + 25 * H)
    led = _ledger(state)
    assert {r.symbol: r.status for r in led.rows.values()} == {"EURUSD": HoleStatus.FILLED, "GBPUSD": HoleStatus.RESOLVED}
    assert led.open_rows() == []


def test_dry_run_fetch_writes_nothing(tmp_path):
    truth = _frames()
    profile, _, state = _setup(tmp_path, _drop(truth, "EURUSD", WED, 24))
    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth), refetch=CFG, now=T0, write=False)
    assert run.attempts_frame()["result"].tolist() == ["filled"] and not any(state.rglob("*"))


# ---- A7: spikes ---------------------------------------------------------------------------------------------------------

def test_spike_replaced_when_the_broker_serves_a_clean_bar(tmp_path):
    truth = _frames()
    profile, root, state = _setup(tmp_path, _spike(truth, ["EURUSD"]))
    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(truth), refetch=CFG, now=T0)
    att = run.attempts_frame()
    assert att[["symbol", "kind", "result"]].values.tolist() == [["EURUSD", "spike", "replaced"]]
    _same_bars(_staged(run, "EURUSD"), truth["EURUSD"])
    rep = run.manifest()["fetch"]["replacements"]
    assert len(rep) == 1 and rep[0]["old"][3] == pytest.approx(truth["EURUSD"].loc[SPIKE_AT, "close"] * 1.01)
    assert _quality(run, "EURUSD", QualityFlag.REFETCHED) == 1 and run.symbols["EURUSD"].replaced == 1
    # a later export that changed the bar wins over the stored replacement
    p = root / "raw" / "SYNTH" / "5" / "EURUSD.parquet"
    raw = pd.read_parquet(p)
    raw.loc[raw.index.tz_localize(None) == SPIKE_AT, "close"] *= 0.999
    raw.to_parquet(p, compression="gzip")
    again = remediate_broker(profile=profile, state_dir=state, now=T0 + H)
    assert again.symbols["EURUSD"].replaced == 0


def test_spike_confirmed_when_the_broker_serves_the_same_bar(tmp_path):
    spiked = _spike(_frames(), ["EURUSD"])
    profile, _, state = _setup(tmp_path, spiked)
    run = remediate_broker(profile=profile, state_dir=state, adapter=FakeBroker(spiked), refetch=CFG, now=T0)
    assert run.attempts_frame()["result"].tolist() == ["confirmed"]
    assert _staged(run, "EURUSD").loc[SPIKE_AT, "close"] == spiked["EURUSD"].loc[SPIKE_AT, "close"]
    assert _quality(run, "EURUSD", QualityFlag.SPIKE_CONFIRMED) == 1
    assert next(iter(_ledger(state).rows.values())).attempts == 1


def test_market_event_is_never_refetched(tmp_path):
    spiked = _spike(_frames(), SYMS)
    profile, _, state = _setup(tmp_path, spiked)
    broker = FakeBroker(_frames())
    run = remediate_broker(profile=profile, state_dir=state, adapter=broker, refetch=CFG, now=T0)
    assert broker.calls == 0 and run.attempts_frame().empty


# ---- MT5 adapter: call surface, env handling, error mapping ---------------------------------------------------------------

ALLOWED_MT5_CALLS = {"initialize", "account_info", "last_error", "symbol_info", "symbol_select", "copy_rates_range", "shutdown"}


class FakeMT5:
    """Only the read calls the adapter may use; data calls refuse keyword arguments (the real binding ignores them)."""

    TIMEFRAME_M5 = 5

    def __init__(self, bars, login=123, server="SYNTH", rates_error=None, symbols=("EURUSD",)):
        self.bars, self.login, self.server, self.rates_error, self.symbols = bars, login, server, rates_error, symbols
        self.called, self.error, self.init_kwargs = [], (1, "Success"), None

    def initialize(self, **kwargs):
        self.called.append("initialize")
        self.init_kwargs = kwargs
        return True

    def account_info(self):
        self.called.append("account_info")
        return SimpleNamespace(login=self.login, server=self.server)

    def last_error(self):
        self.called.append("last_error")
        return self.error

    def symbol_info(self, *args, **kwargs):
        assert not kwargs
        self.called.append("symbol_info")
        if args[0] not in self.symbols:
            self.error = (-1, "Terminal: Call failed")
            return None
        return SimpleNamespace(visible=False)

    def symbol_select(self, *args, **kwargs):
        assert not kwargs and args[1] is True
        self.called.append("symbol_select")
        return True

    def copy_rates_range(self, *args, **kwargs):
        assert not kwargs and len(args) == 4
        self.called.append("copy_rates_range")
        _, tf, a, b = args
        assert tf == 5 and a.utcoffset() == pd.Timedelta(0) and b.utcoffset() == pd.Timedelta(0)
        if self.rates_error:
            self.error = self.rates_error
            return None
        t = self.bars.index.asi8 // 10 ** 9
        sel = (t >= int(a.timestamp())) & (t <= int(b.timestamp()))
        df = self.bars[sel]
        return np.rec.fromarrays([t[sel], *(df[c].to_numpy() for c in PX), df["tick_volume"].to_numpy().astype(np.uint64),
                                  df["spread"].to_numpy(), np.zeros(len(df), dtype=np.uint64)],
                                 names=["time", *PX, "tick_volume", "spread", "real_volume"])

    def shutdown(self):
        self.called.append("shutdown")


def _env(tmp_path, server="SYNTH"):
    d = tmp_path / "env"
    d.mkdir(exist_ok=True)
    (d / ".env.fake.demo").write_text(f"TERMINAL_PATH=C:\\fake\\terminal64.exe\nLOGIN_SERVER={server}\nLOGIN_ID=123\nLOGIN_PASSWORD=secret\n",
                                      encoding="utf-8")
    return d


def test_mt5_adapter_reads_only_and_keeps_raw_digits(tmp_path, monkeypatch):
    for k in ("TERMINAL_PATH", "LOGIN_SERVER", "LOGIN_ID", "LOGIN_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LOGIN_SERVER", "untouched")
    bars = _frames()["EURUSD"]
    mt5 = FakeMT5(bars)
    with MT5FetchAdapter("fake", "demo", 5, "SYNTH", env_dir=_env(tmp_path), module=mt5) as ad:
        got = ad.fetch("EURUSD", WED, WED + 2 * H)
        assert ad.calls == 1
    assert mt5.init_kwargs == {"path": "C:\\fake\\terminal64.exe", "server": "SYNTH", "login": 123, "password": "secret"}
    assert set(mt5.called) <= ALLOWED_MT5_CALLS and mt5.called[-1] == "shutdown"
    assert "LOGIN_PASSWORD" not in os.environ and os.environ["LOGIN_SERVER"] == "untouched"     # env file read, environment restored
    want = bars.loc[WED:WED + 2 * H - pd.Timedelta(minutes=5)]
    assert (got.index == want.index.as_unit("ns")).all() and got.index.tz is None
    np.testing.assert_array_equal(got[PX].to_numpy(), want[PX].to_numpy())


def test_mt5_adapter_refuses_another_feed(tmp_path):
    bars = _frames()["EURUSD"]
    mt5 = FakeMT5(bars)
    with pytest.raises(FetchUnavailableError, match="refusing to mix feeds"):
        MT5FetchAdapter("fake", "demo", 5, "SYNTH", env_dir=_env(tmp_path, server="Other-Server"), module=mt5).open()
    assert mt5.called == []
    wrong = FakeMT5(bars, login=999)
    with pytest.raises(FetchUnavailableError, match="logged in as"):
        MT5FetchAdapter("fake", "demo", 5, "SYNTH", env_dir=_env(tmp_path), module=wrong).open()
    assert wrong.called[-1] == "shutdown"


@pytest.mark.parametrize("error, exc", [((-10005, "IPC timeout"), TransientFetchError), ((-10004, "No IPC connection"), TransientFetchError),
                                        ((-2, "Invalid params"), FetchUnavailableError)])
def test_mt5_adapter_error_mapping(tmp_path, error, exc):
    mt5 = FakeMT5(_frames()["EURUSD"], rates_error=error)
    with MT5FetchAdapter("fake", "demo", 5, "SYNTH", env_dir=_env(tmp_path), module=mt5) as ad, pytest.raises(exc):
        ad.fetch("EURUSD", WED, WED + H)


def test_mt5_adapter_unknown_symbol_stops(tmp_path):
    mt5 = FakeMT5(_frames()["EURUSD"])
    with MT5FetchAdapter("fake", "demo", 5, "SYNTH", env_dir=_env(tmp_path), module=mt5) as ad, \
            pytest.raises(FetchUnavailableError, match="not a symbol"):
        ad.fetch("NOPE", WED, WED + H)


def test_package_import_does_not_load_metatrader():
    code = "import sys, okmich_quant_utils.ohlc_audit; print('MetaTrader5' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ---- CLI ------------------------------------------------------------------------------------------------------------------

def test_cli_fetch_flags_and_holes(tmp_path, monkeypatch, capsys):
    root = tmp_path / "md"
    (root / "raw").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    base = ["--broker", "FXPIG-Server", "--data-dir", str(root), "--state-dir", str(state), "--quiet"]
    assert main(["remediate", *base, "--retry-permanent"]) == 2
    assert "only apply with --fetch" in capsys.readouterr().err
    monkeypatch.delenv("OKMICH_QUANT_ENV_DIR", raising=False)
    assert main(["remediate", *base, "--fetch"]) == 2
    assert "OKMICH_QUANT_ENV_DIR" in capsys.readouterr().err
    assert main(["holes", *base]) == 0
    assert "no hole ledger rows" in capsys.readouterr().out
