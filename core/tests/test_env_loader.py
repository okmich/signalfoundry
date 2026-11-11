"""Env-file folder resolution: ``OKMICH_QUANT_ENV_DIR`` or an explicit folder, never the working directory."""
from __future__ import annotations

import os

import pytest

from okmich_quant_core.env_loader import ENV_DIR_VAR, EnvDirNotConfiguredError, EnvLoader, load_broker_env, resolve_env_dir

MT5_VARS = ("TERMINAL_PATH", "LOGIN_ID", "LOGIN_PASSWORD", "LOGIN_SERVER")


def _write_env(folder, name, server="Test-Server"):
    body = f"TERMINAL_PATH=C:/terminal.exe\nLOGIN_ID=123\nLOGIN_PASSWORD=pw\nLOGIN_SERVER={server}\n"
    (folder / name).write_text(body, encoding="utf-8")


@pytest.fixture
def mt5_vars(monkeypatch):
    """load_dotenv writes os.environ; pre-set the keys through monkeypatch so teardown restores the machine's values."""
    for var in MT5_VARS:
        monkeypatch.setenv(var, "unset-by-test")


def test_env_var_names_the_folder(tmp_path, monkeypatch, mt5_vars):
    _write_env(tmp_path, ".env.fxify.demo", server="FXPIG-Server")
    monkeypatch.setenv(ENV_DIR_VAR, str(tmp_path))
    loader = EnvLoader()
    assert loader.base_dir == tmp_path
    assert loader.load("fxify", "demo")
    assert os.environ["LOGIN_SERVER"] == "FXPIG-Server"


def test_explicit_folder_wins_over_env_var(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(ENV_DIR_VAR, str(tmp_path))
    assert resolve_env_dir(other) == other
    assert EnvLoader(str(other)).base_dir == other


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_or_blank_env_var_fails_loudly(monkeypatch, tmp_path, value):
    monkeypatch.chdir(tmp_path)  # a cwd holding env files must not be used as a fallback
    _write_env(tmp_path, ".env.fxify.demo")
    if value is None:
        monkeypatch.delenv(ENV_DIR_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_DIR_VAR, value)
    with pytest.raises(EnvDirNotConfiguredError, match=ENV_DIR_VAR):
        EnvLoader()
    with pytest.raises(EnvDirNotConfiguredError):
        load_broker_env("fxify", "demo")


def test_env_var_pointing_at_missing_folder_fails(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR_VAR, str(tmp_path / "does-not-exist"))
    with pytest.raises(EnvDirNotConfiguredError, match="not an existing folder"):
        resolve_env_dir()


def test_list_configs_reads_the_env_dir(tmp_path, monkeypatch):
    _write_env(tmp_path, ".env.fxify.demo")
    _write_env(tmp_path, ".env.deriv.live")
    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")  # not a broker file
    monkeypatch.setenv(ENV_DIR_VAR, str(tmp_path))
    assert EnvLoader().list_available_configs() == [("deriv", "live"), ("fxify", "demo")]


def test_load_broker_env_uses_env_dir(tmp_path, monkeypatch, mt5_vars):
    _write_env(tmp_path, ".env.icmarkets.demo", server="ICMarketsSC-Demo")
    monkeypatch.setenv(ENV_DIR_VAR, str(tmp_path))
    assert load_broker_env("icmarkets", "demo")
