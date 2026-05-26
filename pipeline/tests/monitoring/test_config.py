"""Tests for MonitorConfig JSON loader + validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from okmich_quant_pipeline.monitoring import MonitorConfig


def _write_config(path: Path, **overrides) -> Path:
    defaults: dict = {
        "symbols": ["GBPUSD.r", "EURUSD.r"],
        "artifact_base_dir": "./artefacts",
        "variant_with_lag": "hmm_lambda_L3",
        "inference_log_base_dir": "./logs",
        "strategy_name_template": "{symbol}_fl3_hmm",
        "raw_data_dir": "./raw",
        "output_dir": "./monitor_out",
        "feature_engineering_callable": "my_pkg.features:compute",
        "tail_n": 500,
        "violation_counter_threshold": 3,
    }
    defaults.update(overrides)
    path.write_text(json.dumps(defaults), encoding="utf-8")
    return path


def test_config_from_json_happy_path(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json")

    cfg = MonitorConfig.from_json(cfg_path)

    assert cfg.symbols == ("GBPUSD.r", "EURUSD.r")
    assert cfg.variant_with_lag == "hmm_lambda_L3"
    assert cfg.tail_n == 500
    assert cfg.violation_counter_threshold == 3
    # Defaults filled in for unspecified gate thresholds
    assert cfg.max_entropy_abs_z == 3.0
    assert cfg.max_ks_statistic == 0.1


def test_config_resolves_paths_relative_to_config_file(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json")
    cfg = MonitorConfig.from_json(cfg_path)

    assert cfg.artifact_base_dir == (tmp_path / "artefacts").resolve()
    assert cfg.raw_data_dir == (tmp_path / "raw").resolve()
    assert cfg.output_dir == (tmp_path / "monitor_out").resolve()


def test_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        MonitorConfig.from_json(tmp_path / "missing.json")


def test_config_rejects_missing_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"symbols": ["A"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        MonitorConfig.from_json(path)


def test_config_rejects_empty_symbols(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json", symbols=[])
    with pytest.raises(ValueError, match="symbols.*non-empty"):
        MonitorConfig.from_json(cfg_path)


def test_config_rejects_template_without_symbol_placeholder(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json", strategy_name_template="fixed_name")

    with pytest.raises(ValueError, match=r"must contain"):
        MonitorConfig.from_json(cfg_path)


def test_config_rejects_invalid_tail_n(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json", tail_n=0)
    with pytest.raises(ValueError, match="tail_n.*>= 1"):
        MonitorConfig.from_json(cfg_path)


def test_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json", violation_counter_threshold=0)
    with pytest.raises(ValueError, match="violation_counter_threshold.*>= 1"):
        MonitorConfig.from_json(cfg_path)


def test_config_rejects_malformed_feature_callable(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "monitor.json", feature_engineering_callable="missing_colon_separator")
    with pytest.raises(ValueError, match="'pkg.module:func_name'"):
        MonitorConfig.from_json(cfg_path)


def test_config_optional_notifier_defaults_to_none(tmp_path: Path) -> None:
    cfg = MonitorConfig.from_json(_write_config(tmp_path / "monitor.json"))
    assert cfg.notifier is None


def test_config_accepts_notifier_block(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "monitor.json",
        notifier={"type": "telegram", "bot_token": "tok", "chat_id": "chat"},
    )
    cfg = MonitorConfig.from_json(cfg_path)
    assert cfg.notifier == {"type": "telegram", "bot_token": "tok", "chat_id": "chat"}
