"""Tests for the account level: the log tree mirrors the live account folder a system is deployed in
(okmich_quant_core.account.deployment_account + the <log_base>[/<account>] resolution)."""

from __future__ import annotations

import json

import pytest

from okmich_quant_core import deployment_account, text_log_dir
from okmich_quant_core.account import is_account
from okmich_quant_core.logging import JsonlEventLogger, LogicalSystemIdentity, RunnerIdentity, RunnerStatus, runner_log_dir


@pytest.mark.parametrize("value", ["fxify.demo", "deriv.live", "ib.paper", "ic_markets.demo"])
def test_is_account(value):
    assert is_account(value)


@pytest.mark.parametrize("value", ["fxify", "Fxify.demo", "fxify.demo.x", "ctlpb_raw-multi", "_account_admin", ".archive"])
def test_is_not_account(value):
    assert not is_account(value)


def _script(tmp_path, *parts):
    path = tmp_path.joinpath("live", *parts, "run.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


@pytest.mark.parametrize("parts, want", [
    (("fxify.demo", "ctlpb_raw-multi"), "fxify.demo"),                 # a multi-trader in an account
    (("icmarkets.demo", "s", "EURUSD", "5"), "icmarkets.demo"),        # a single-trader in an account
    (("fxify.demo", "_account_admin"), "fxify.demo"),                  # the Account Admin
    (("ctlpb_raw-multi",), None),                                      # flat, pre-account deployment
])
def test_deployment_account_from_the_script_location(tmp_path, parts, want):
    assert deployment_account(script=_script(tmp_path, *parts), live_base=tmp_path / "live") == want


def test_no_account_outside_the_live_tree_or_without_a_live_base(tmp_path):
    script = tmp_path / "lab" / "systems" / "x" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    assert deployment_account(script=script, live_base=tmp_path / "live") is None
    assert deployment_account(script=_script(tmp_path, "fxify.demo", "x"), live_base="") is None


def test_live_base_comparison_ignores_case_on_windows(tmp_path):
    script = _script(tmp_path, "fxify.demo", "x")
    base = tmp_path / "live"
    if str(base).upper() == str(base):
        pytest.skip("path has no letters to case-fold")
    import os
    if os.path.normcase("A") != os.path.normcase("a"):
        pytest.skip("case-sensitive filesystem")
    assert deployment_account(script=script, live_base=str(base).upper()) == "fxify.demo"


def test_default_script_is_the_running_main(monkeypatch, tmp_path):
    """Under pytest the main script is not in a live tree, so everything logs flat, as before."""
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(tmp_path / "live"))
    assert deployment_account() is None


# --- the three channels follow the deployment -------------------------------------------------

@pytest.fixture
def deployed(monkeypatch, tmp_path):
    """Pretend the running main script is <tmp>/live/fxify.demo/s-multi/run.py."""
    script = _script(tmp_path, "fxify.demo", "s-multi")
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(tmp_path / "live"))
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path / "logs"))
    monkeypatch.setattr("okmich_quant_core.account._main_script", lambda: script)
    return tmp_path


def test_all_channels_mirror_the_account_folder(deployed):
    logical = LogicalSystemIdentity(strategy="s-multi", symbol="EURUSD", timeframe_minutes=5)
    cfg = deployed / "live" / "fxify.demo" / "s-multi" / "config.json"
    cfg.write_text(json.dumps({"strategies": [{"name": "s", "symbol": "EURUSD", "timeframe": 5}]}), encoding="utf-8")
    logger = JsonlEventLogger(logical)
    rs = RunnerStatus(RunnerIdentity.generate(name="r", broker="b", account_id="1"), [logical])
    try:
        root = deployed / "logs" / "fxify.demo" / "s-multi"
        assert text_log_dir(cfg) == root
        assert logger.directory == root / "EURUSD" / "5" / "inference"
        assert rs.status_path == root / "status.json"
        assert runner_log_dir("s-multi") == root
        rs.mark_started()
        assert json.loads(rs.status_path.read_text(encoding="utf-8"))["account"] == "fxify.demo"
    finally:
        logger.close()


def test_flat_deployment_logs_flat(monkeypatch, tmp_path):
    script = _script(tmp_path, "s-multi")
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(tmp_path / "live"))
    monkeypatch.setattr("okmich_quant_core.account._main_script", lambda: script)
    logical = LogicalSystemIdentity(strategy="s-multi", symbol="EURUSD", timeframe_minutes=5)
    rs = RunnerStatus(RunnerIdentity.generate(name="r", broker="b", account_id="1"), [logical], log_base=tmp_path / "logs")
    rs.mark_started()
    assert rs.status_path == tmp_path / "logs" / "s-multi" / "status.json"
    assert json.loads(rs.status_path.read_text(encoding="utf-8"))["account"] is None
