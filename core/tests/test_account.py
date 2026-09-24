"""Tests for the account level (okmich_quant_core.account + the <log_base>/<account> resolution).

The account comes from OKMICH_QUANT_ACCOUNT only (conftest sets it to test.demo), so every channel of one
runner resolves the same folder.
"""

from __future__ import annotations

import json
import os

import pytest

from okmich_quant_core import AccountConfigError, load_account_env, resolve_account, text_log_dir
from okmich_quant_core.account import account_env_path, validate_account
from okmich_quant_core.logging import JsonlEventLogger, LogicalSystemIdentity, RunnerIdentity, RunnerStatus, runner_log_dir


@pytest.fixture
def logical():
    return LogicalSystemIdentity(strategy="s", symbol="EURUSD", timeframe_minutes=5)


# --- the account ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["fxify.demo", "deriv.live", "ib.paper", "ic_markets.demo", " fxify.demo "])
def test_validate_account_accepts_env_stems(value):
    assert validate_account(value) == value.strip()


@pytest.mark.parametrize("value", ["fxify", "Fxify.demo", "fxify.demo.x", "../x", "fxify/demo", ".demo", "fxify.", "a b.demo"])
def test_validate_account_rejects_non_stems(value):
    with pytest.raises(AccountConfigError):
        validate_account(value)


def test_resolve_account_reads_env(monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", "deriv.live")
    assert resolve_account() == "deriv.live"


@pytest.mark.parametrize("value", [None, "", "   ", "fxify"])
def test_resolve_account_required_and_valid(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OKMICH_QUANT_ACCOUNT", raising=False)
    else:
        monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", value)
    with pytest.raises(AccountConfigError):
        resolve_account()


# --- account env file ----------------------------------------------------------------------

def test_account_env_path_uses_env_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_ENV_DIR", str(tmp_path))
    monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", "fxify.demo")
    assert account_env_path() == tmp_path / ".env.fxify.demo"


def test_account_env_path_requires_env_dir(monkeypatch):
    monkeypatch.delenv("OKMICH_QUANT_ENV_DIR", raising=False)
    with pytest.raises(AccountConfigError, match="OKMICH_QUANT_ENV_DIR"):
        account_env_path()


def test_load_account_env_loads_the_accounts_file(tmp_path, monkeypatch):
    (tmp_path / ".env.fxify.demo").write_text("LOGIN_ID=123\n", encoding="utf-8")
    (tmp_path / ".env.icmarkets.demo").write_text("LOGIN_ID=999\n", encoding="utf-8")
    monkeypatch.setenv("OKMICH_QUANT_ENV_DIR", str(tmp_path))
    monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", "fxify.demo")
    monkeypatch.delenv("LOGIN_ID", raising=False)
    assert load_account_env() == tmp_path / ".env.fxify.demo"
    assert os.environ["LOGIN_ID"] == "123"


def test_load_account_env_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_ENV_DIR", str(tmp_path))
    with pytest.raises(AccountConfigError, match="not found"):
        load_account_env()


def test_load_account_env_explicit_file_overrides(tmp_path, monkeypatch):
    override = tmp_path / "custom.env"
    override.write_text("LOGIN_ID=555\n", encoding="utf-8")
    monkeypatch.delenv("LOGIN_ID", raising=False)
    assert load_account_env(env_file=override) == override
    assert os.environ["LOGIN_ID"] == "555"


def test_load_account_env_explicit_missing_file_raises(tmp_path):
    with pytest.raises(AccountConfigError, match="not found"):
        load_account_env(env_file=tmp_path / "nope.env")


def test_load_account_env_requires_the_account_even_with_an_explicit_file(tmp_path, monkeypatch):
    """--env-file without OKMICH_QUANT_ACCOUNT must fail here, before the runner logs in to the broker."""
    override = tmp_path / "custom.env"
    override.write_text("LOGIN_ID=555\n", encoding="utf-8")
    monkeypatch.delenv("OKMICH_QUANT_ACCOUNT", raising=False)
    with pytest.raises(AccountConfigError, match="OKMICH_QUANT_ACCOUNT"):
        load_account_env(env_file=override)


@pytest.mark.parametrize("line", ["OKMICH_QUANT_ACCOUNT=deriv.live", "OKMICH_QUANT_LOG_BASE=D:/elsewhere", "okmich_quant_env_dir=x"])
def test_load_account_env_refuses_ops_keys(tmp_path, monkeypatch, line):
    """A broker env file must not re-point the runner's account or log root after logging has started."""
    (tmp_path / ".env.fxify.demo").write_text(f"LOGIN_ID=1\n{line}\n", encoding="utf-8")
    monkeypatch.setenv("OKMICH_QUANT_ENV_DIR", str(tmp_path))
    monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", "fxify.demo")
    with pytest.raises(AccountConfigError, match="OKMICH_QUANT_"):
        load_account_env()
    assert os.environ["OKMICH_QUANT_ACCOUNT"] == "fxify.demo"


# --- <log_base>/<account> resolution ------------------------------------------------------

def test_explicit_log_base_still_gets_the_account(tmp_path, logical):
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    try:
        assert logger.directory == tmp_path / "test.demo" / "s" / "EURUSD" / "5" / "inference"
    finally:
        logger.close()


def test_logger_without_account_fails_fast(tmp_path, logical, monkeypatch):
    monkeypatch.delenv("OKMICH_QUANT_ACCOUNT", raising=False)
    with pytest.raises(AccountConfigError):
        JsonlEventLogger(logical, log_base=tmp_path)


def test_all_channels_resolve_the_same_account_folder(tmp_path, logical, monkeypatch):
    """Text log, inference JSONL and status.json of one runner land in the same <log_base>/<account>/<root>."""
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path))
    monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", "deriv.live")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"strategy": {"name": "s", "symbol": "EURUSD", "timeframe": 5}}), encoding="utf-8")
    logger = JsonlEventLogger(logical)
    rs = RunnerStatus(RunnerIdentity.generate(name="r", broker="b", account_id="1"), [logical])
    try:
        root = tmp_path / "deriv.live" / "s"
        assert text_log_dir(cfg) == root
        assert logger.directory.parent.parent.parent == root
        assert rs.status_path.parent == root
        rs.mark_started()
        assert json.loads(rs.status_path.read_text(encoding="utf-8"))["account"] == "deriv.live"
    finally:
        logger.close()


def test_runner_log_dir(tmp_path):
    assert runner_log_dir("ctlpb_raw-multi", log_base=tmp_path) == tmp_path / "test.demo" / "ctlpb_raw-multi"


def test_text_log_dir_requires_account_when_log_base_set(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"strategy": {"name": "s", "symbol": "EURUSD", "timeframe": 5}}), encoding="utf-8")
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path / "logs"))
    monkeypatch.delenv("OKMICH_QUANT_ACCOUNT", raising=False)
    with pytest.raises(AccountConfigError):
        text_log_dir(cfg)
