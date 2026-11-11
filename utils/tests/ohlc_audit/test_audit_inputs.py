"""Run inputs (data root, registry, reports) and edge-case frames."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit import cli
from okmich_quant_utils.ohlc_audit.audit import audit_broker, audit_symbol
from okmich_quant_utils.ohlc_audit.clock import ZoneClock
from okmich_quant_utils.ohlc_audit.enums import AssetClass, FindingKind, Severity
from okmich_quant_utils.ohlc_audit.profile import (MARKET_DATA_DIR_VAR, BrokerProfile, FetchSource, MarketDataDirNotConfiguredError, SymbolSpec,
                                          resolve_market_data_dir)
from okmich_quant_utils.ohlc_audit.profiles import get_profile
from okmich_quant_utils.ohlc_audit.profiles.fxpig import SESSIONS
from okmich_quant_utils.ohlc_audit.symbol_data import NotABarSeries


# ---- data root ------------------------------------------------------------------------------------------------------

def test_argument_wins_over_env(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    monkeypatch.setenv(MARKET_DATA_DIR_VAR, str(b))
    assert resolve_market_data_dir(a) == a
    assert resolve_market_data_dir() == b


@pytest.mark.parametrize("env, arg, msg", [(None, None, "no market-data root"), ("  ", None, "no market-data root"),
                                           ("/nope/x", None, "not an existing folder"), (None, "/nope/y", "not an existing folder")])
def test_missing_or_bad_root_fails(monkeypatch, env, arg, msg):
    if env is None:
        monkeypatch.delenv(MARKET_DATA_DIR_VAR, raising=False)
    else:
        monkeypatch.setenv(MARKET_DATA_DIR_VAR, env)
    with pytest.raises(MarketDataDirNotConfiguredError, match=msg):
        resolve_market_data_dir(arg)


def test_profile_without_root_fails_only_when_reading(monkeypatch):
    monkeypatch.delenv(MARKET_DATA_DIR_VAR, raising=False)
    with pytest.raises(MarketDataDirNotConfiguredError):
        get_profile("FXPIG-Server")
    p = get_profile("FXPIG-Server", require_data=False)
    with pytest.raises(MarketDataDirNotConfiguredError):
        p.load_raw("EURUSD")


def test_profile_paths_hang_off_the_root(tmp_path):
    p = get_profile("FXPIG-Server", tmp_path)
    assert p.path("EURUSD") == tmp_path / "raw" / "FXPIG-Server" / "5" / "EURUSD.r.parquet"
    assert p.references["BTCUSDT"].path == tmp_path / "raw" / "crypto" / "binance" / "ohlcv" / "5m" / "BTC_USDT_USDT.parquet"
    with pytest.raises(FileNotFoundError, match="no bar file"):
        p.load_raw("EURUSD")


# ---- CLI ------------------------------------------------------------------------------------------------------------

def _fx_week(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ny = pd.date_range("2024-04-07 17:00", "2024-04-12 17:00", freq="5min", inclusive="left")
    close = 1.08 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(ny))))
    open_ = np.r_[close[0], close[:-1]]
    w = np.abs(rng.normal(0, 3e-5, (2, len(ny)))) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + w[0], "low": np.minimum(open_, close) - w[1], "close": close,
                         "tick_volume": 100.0, "spread": 7}, index=pd.DatetimeIndex(ny + pd.Timedelta(hours=7), name="date"))


def _data_root(tmp_path):
    folder = tmp_path / "md" / "raw" / "FXPIG-Server" / "5"
    folder.mkdir(parents=True)
    _fx_week().to_parquet(folder / "EURUSD.r.parquet")
    return tmp_path / "md"


def test_cli_audits_and_keeps_the_data_tree_clean(tmp_path, monkeypatch, capsys):
    root = _data_root(tmp_path)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    code = cli.main(["audit", "--broker", "FXPIG-Server", "--data-dir", str(root), "--symbols", "EURUSD", "--quiet"])
    assert code == 0
    assert "OHLC AUDIT" in capsys.readouterr().out
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before            # nothing written into market data
    assert list((tmp_path / "tmp" / "raw-data-audit" / "FXPIG-Server").glob("*/findings.csv"))


def test_cli_usage_errors_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(MARKET_DATA_DIR_VAR, raising=False)
    assert cli.main(["audit", "--broker", "FXPIG-Server", "--no-write"]) == 2
    assert "no market-data root" in capsys.readouterr().err
    root = _data_root(tmp_path)
    assert cli.main(["audit", "--broker", "FXPIG-Server", "--data-dir", str(root), "--registry", str(tmp_path / "missing.csv"), "--no-write"]) == 2


# ---- edge-case frames -----------------------------------------------------------------------------------------------

def _crypto_profile() -> BrokerProfile:
    return BrokerProfile("EDGE", 5, ZoneClock("UTC"), {"X": SymbolSpec("X", "X", AssetClass.CRYPTO, "crypto"),
                                                       "F": SymbolSpec("F", "F", AssetClass.FX, "fx")}, SESSIONS, FetchSource("-", "-"))


def _bars(index, price=100.0) -> pd.DataFrame:
    n = len(index)
    c = price + np.arange(n) * 0.01
    return pd.DataFrame({"open": c, "high": c + 0.02, "low": c - 0.02, "close": c + 0.005}, index=index)


def test_empty_frame_is_rejected_clearly():
    with pytest.raises(NotABarSeries, match="no bars"):
        audit_symbol(_crypto_profile(), "X", _bars(pd.DatetimeIndex([], name="date")))


def test_single_bar_and_tiny_frames_run():
    for n in (1, 2, 50):
        rep = audit_symbol(_crypto_profile(), "X", _bars(pd.date_range("2024-01-01", periods=n, freq="5min")))
        assert rep.rows == n


def test_unsorted_and_tz_aware_input():
    idx = pd.date_range("2024-01-01", periods=600, freq="5min", tz="UTC")
    df = _bars(idx).iloc[np.r_[0:300, 310:600, 300:310]]                   # a block appended out of order
    rep = audit_symbol(_crypto_profile(), "X", df)
    kinds = {f.kind for f in rep.detector_findings()}
    assert FindingKind.NON_MONOTONIC in kinds


def test_nan_rows_are_flagged_not_fatal():
    df = _bars(pd.date_range("2024-01-01", periods=600, freq="5min"))
    df.iloc[100:103, df.columns.get_loc("close")] = np.nan
    rep = audit_symbol(_crypto_profile(), "X", df)
    assert any(f.kind is FindingKind.NON_FINITE and f.severity is Severity.CRITICAL for f in rep.detector_findings())


def test_all_bars_in_closed_slots():
    sat = pd.date_range("2024-01-06 10:00", periods=48, freq="5min")        # Saturday, FX closed (raw = UTC here)
    rep = audit_symbol(_crypto_profile(), "F", _bars(sat, 1.1))
    assert any(f.kind is FindingKind.OUT_OF_SESSION for f in rep.detector_findings())


def test_broker_audit_with_one_symbol_and_no_peers():
    rep = audit_broker(profile=_crypto_profile(), frames={"X": _bars(pd.date_range("2024-01-01", periods=3000, freq="5min"))})
    assert rep.symbols["X"].rows == 3000
