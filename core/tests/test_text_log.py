"""Tests for the per-process text-log path resolution (okmich_quant_core.logging.text_log)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from okmich_quant_core import setup_text_logger, text_log_dir


def _write_config(dir_: Path, payload: dict) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    cfg = dir_ / "config.json"
    cfg.write_text(json.dumps(payload), encoding="utf-8")
    return cfg


def test_text_log_dir_uses_log_base_runner_root_for_single(tmp_path, monkeypatch):
    log_base = tmp_path / "logs"
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(log_base))
    cfg = _write_config(tmp_path / "live" / "rsi2" / "EURUSD" / "5",
                        {"strategy": {"name": "rsi2_mean_reversion", "symbol": "EURUSD", "timeframe": 5},
                         "strategies": []})
    # single-trader -> <log_base>/<strategy>
    assert text_log_dir(cfg) == log_base / "rsi2_mean_reversion"


def test_text_log_dir_uses_multi_runner_root(tmp_path, monkeypatch):
    log_base = tmp_path / "logs"
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(log_base))
    cfg = _write_config(tmp_path / "live" / "basket",
                        {"strategies": [{"name": "rsi2_mean_reversion", "symbol": "BTCUSD", "timeframe": 5}]})
    # multi-trader -> <log_base>/<strategy>-multi (statutory suffix)
    assert text_log_dir(cfg) == log_base / "rsi2_mean_reversion-multi"


def test_text_log_dir_falls_back_to_config_dir_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("OKMICH_QUANT_LOG_BASE", raising=False)
    cfg_dir = tmp_path / "live" / "rsi2" / "EURUSD" / "5"
    cfg = _write_config(cfg_dir, {"strategy": {"name": "rsi2", "symbol": "EURUSD", "timeframe": 5}})
    assert text_log_dir(cfg) == cfg_dir


def test_setup_text_logger_writes_under_log_base(tmp_path, monkeypatch):
    log_base = tmp_path / "logs"
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(log_base))
    cfg = _write_config(tmp_path / "live" / "rsi2" / "EURUSD" / "5",
                        {"strategy": {"name": "rsi2", "symbol": "EURUSD", "timeframe": 5}})
    try:
        log_file = setup_text_logger(cfg)
        assert log_file.parent == log_base / "rsi2"
        assert log_file.name.startswith("z_system_log_") and log_file.suffix == ".log"
        logging.getLogger(__name__).info("hello")
        for h in logging.getLogger().handlers:
            h.flush()
        assert log_file.exists()
        assert "hello" in log_file.read_text(encoding="utf-8")
    finally:
        logging.getLogger().handlers.clear()


def test_setup_text_logger_prefix_for_ib(tmp_path, monkeypatch):
    monkeypatch.delenv("OKMICH_QUANT_LOG_BASE", raising=False)
    cfg = _write_config(tmp_path / "live", {"strategy": {"name": "s", "symbol": "X", "timeframe": 5}})
    try:
        log_file = setup_text_logger(cfg, prefix="z_ib_system_log")
        assert log_file.name.startswith("z_ib_system_log_")
        assert log_file.parent == cfg.parent  # fallback to config dir when LOG_BASE unset
    finally:
        logging.getLogger().handlers.clear()
