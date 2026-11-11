"""fetch-mt5-data finds broker .env files through OKMICH_QUANT_ENV_DIR and fails loudly when it is unset."""
from __future__ import annotations

import sys

import pytest

from okmich_quant_core.env_loader import ENV_DIR_VAR, EnvDirNotConfiguredError
from okmich_quant_mt5.utils.script import market_data_downloader as mdd

MT5_VARS = ("TERMINAL_PATH", "LOGIN_ID", "LOGIN_PASSWORD", "LOGIN_SERVER")


@pytest.fixture
def env_dir(tmp_path, monkeypatch):
    for var in MT5_VARS:
        monkeypatch.setenv(var, "unset-by-test")  # teardown restores the machine's values after load_dotenv
    body = "TERMINAL_PATH=C:/terminal.exe\nLOGIN_ID=123\nLOGIN_PASSWORD=pw\nLOGIN_SERVER=FXPIG-Server\n"
    (tmp_path / ".env.fxify.demo").write_text(body, encoding="utf-8")
    monkeypatch.setenv(ENV_DIR_VAR, str(tmp_path))
    return tmp_path


def test_downloader_loads_from_env_dir(env_dir):
    downloader = mdd.MarketDataDownloader("fxify", "demo")
    assert downloader.env_loaded


def test_downloader_raises_when_env_dir_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_DIR_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(EnvDirNotConfiguredError):
        mdd.MarketDataDownloader("fxify", "demo")


@pytest.mark.parametrize("argv", [["--list-brokers"], ["--broker", "fxify", "--symbols", "EURUSD.r", "--timeframe", "M5"]])
def test_cli_exits_with_message_when_env_dir_unset(monkeypatch, tmp_path, argv):
    monkeypatch.delenv(ENV_DIR_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fetch-mt5-data", *argv])
    with pytest.raises(SystemExit) as exc:
        mdd.main()
    assert ENV_DIR_VAR in str(exc.value.code)
