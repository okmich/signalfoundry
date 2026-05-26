"""Tests for the core inference-logging primitives.

Covers:
- InferenceLogRecord: to_dict / from_dict round-trip, ISO timestamp handling,
  None timestamps, frozen dataclass, missing-key validation.
- JsonlInferenceLogger: write+read-back, daily UTC rotation (mock datetime to
  cross date boundary), filename convention, append semantics across multiple
  writes within a day, close() releases the handle.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from okmich_quant_core import BaseInferenceLogger, InferenceLogRecord, JsonlInferenceLogger
from okmich_quant_core.logging import jsonl as jsonl_module


# ---- InferenceLogRecord ------------------------------------------------------

def _example_record(**overrides) -> InferenceLogRecord:
    defaults = dict(
        wall_clock_utc=pd.Timestamp("2026-05-25T14:30:00.123456+00:00"),
        asof_bar_ts=pd.Timestamp("2026-05-25T14:30:00+00:00"),
        label_bar_ts=pd.Timestamp("2026-05-25T14:15:00+00:00"),
        bar_close=81234.5,
        features={"tsi": 0.12, "dbl_smoothed_log_rets": -0.0001, "smoothed_atr": 57.3},
        direction=1,
        confidence=0.974,
        extras={"probs": [0.013, 0.013, 0.974], "loglik": -2.34, "lag": 3},
    )
    defaults.update(overrides)
    return InferenceLogRecord(**defaults)


def test_inference_log_record_round_trip() -> None:
    original = _example_record()

    payload = original.to_dict()
    restored = InferenceLogRecord.from_dict(payload)

    assert restored.wall_clock_utc == original.wall_clock_utc
    assert restored.asof_bar_ts == original.asof_bar_ts
    assert restored.label_bar_ts == original.label_bar_ts
    assert restored.bar_close == original.bar_close
    assert restored.features == original.features
    assert restored.direction == original.direction
    assert restored.confidence == original.confidence
    assert restored.extras == original.extras


def test_inference_log_record_to_dict_emits_iso_strings_for_timestamps() -> None:
    payload = _example_record().to_dict()

    assert isinstance(payload["wall_clock_utc"], str)
    assert isinstance(payload["asof_bar_ts"], str)
    assert isinstance(payload["label_bar_ts"], str)
    assert "T" in payload["wall_clock_utc"]


def test_inference_log_record_handles_none_label_bar_ts() -> None:
    record = _example_record(label_bar_ts=None, bar_close=None, direction=None, confidence=None)

    payload = record.to_dict()

    assert payload["label_bar_ts"] is None
    assert payload["bar_close"] is None
    assert payload["direction"] is None
    assert payload["confidence"] is None


def test_inference_log_record_round_trips_with_none_fields() -> None:
    original = _example_record(label_bar_ts=None, bar_close=None, direction=None,
                                confidence=None, extras={})

    restored = InferenceLogRecord.from_dict(original.to_dict())

    assert restored.label_bar_ts is None
    assert restored.bar_close is None
    assert restored.direction is None
    assert restored.confidence is None
    assert restored.extras == {}


def test_inference_log_record_is_frozen() -> None:
    record = _example_record()

    with pytest.raises(Exception):
        record.direction = -1  # type: ignore[misc]


def test_inference_log_record_from_dict_rejects_missing_keys() -> None:
    incomplete = {"wall_clock_utc": "2026-05-25T14:30:00+00:00", "features": {}}

    with pytest.raises(ValueError, match=r"missing required keys"):
        InferenceLogRecord.from_dict(incomplete)


def test_inference_log_record_default_extras_is_empty_dict() -> None:
    record = InferenceLogRecord(
        wall_clock_utc=pd.Timestamp("2026-05-25T14:30:00+00:00"),
        asof_bar_ts=pd.Timestamp("2026-05-25T14:30:00+00:00"),
        label_bar_ts=None,
        bar_close=None,
        features={"x": 1.0},
        direction=0,
        confidence=None,
    )

    assert record.extras == {}


# ---- JsonlInferenceLogger ----------------------------------------------------

def test_jsonl_logger_implements_base_interface(tmp_path: Path) -> None:
    logger = JsonlInferenceLogger(tmp_path, strategy_name="TEST")

    assert isinstance(logger, BaseInferenceLogger)
    logger.close()


def test_jsonl_logger_rejects_empty_strategy_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strategy_name must be non-empty"):
        JsonlInferenceLogger(tmp_path, strategy_name="")


def test_jsonl_logger_writes_record_to_dated_file(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 5, 25, 14, 30, 0, tzinfo=timezone.utc)
    with patch.object(jsonl_module, "datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        logger = JsonlInferenceLogger(tmp_path, strategy_name="TEST", fsync=False)
        logger.write(_example_record())
        logger.close()

    expected_path = tmp_path / "inference_TEST_20260525.jsonl"
    assert expected_path.is_file()
    contents = expected_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    record_dict = json.loads(contents[0])
    assert record_dict["extras"]["loglik"] == -2.34
    assert record_dict["direction"] == 1
    assert "strategy_name" not in record_dict  # strategy identity is on the logger, not the record


def test_jsonl_logger_appends_within_same_day(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 5, 25, 14, 30, 0, tzinfo=timezone.utc)
    with patch.object(jsonl_module, "datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        logger = JsonlInferenceLogger(tmp_path, strategy_name="APPEND", fsync=False)
        logger.write(_example_record(confidence=0.5))
        logger.write(_example_record(confidence=0.6))
        logger.write(_example_record(confidence=0.7))
        logger.close()

    expected_path = tmp_path / "inference_APPEND_20260525.jsonl"
    lines = expected_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    confidences = [json.loads(line)["confidence"] for line in lines]
    assert confidences == [0.5, 0.6, 0.7]


def test_jsonl_logger_rotates_on_utc_date_flip(tmp_path: Path) -> None:
    day1 = datetime(2026, 5, 25, 23, 59, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 5, 26, 0, 1, 0, tzinfo=timezone.utc)
    call_count = {"n": 0}

    def fake_now(tz=None):
        call_count["n"] += 1
        # Robust to extra internal datetime.now() calls (rotation logic could grow more):
        # the first call to _ensure_handle returns day1, all subsequent return day2.
        return day1 if call_count["n"] == 1 else day2

    with patch.object(jsonl_module, "datetime") as mock_dt:
        mock_dt.now.side_effect = fake_now
        logger = JsonlInferenceLogger(tmp_path, strategy_name="ROTATE", fsync=False)
        logger.write(_example_record(confidence=0.1))
        logger.write(_example_record(confidence=0.2))
        logger.close()

    file_day1 = tmp_path / "inference_ROTATE_20260525.jsonl"
    file_day2 = tmp_path / "inference_ROTATE_20260526.jsonl"
    assert file_day1.is_file()
    assert file_day2.is_file()
    assert len(file_day1.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert len(file_day2.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_jsonl_logger_records_round_trip_via_from_dict(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 5, 25, 14, 30, 0, tzinfo=timezone.utc)
    with patch.object(jsonl_module, "datetime") as mock_dt:
        mock_dt.now.return_value = fixed_now
        logger = JsonlInferenceLogger(tmp_path, strategy_name="RT", fsync=False)
        original = _example_record()
        logger.write(original)
        logger.close()

    path = tmp_path / "inference_RT_20260525.jsonl"
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    restored = InferenceLogRecord.from_dict(payload)

    assert restored.direction == original.direction
    assert restored.extras["probs"] == original.extras["probs"]
    assert restored.wall_clock_utc == original.wall_clock_utc


def test_jsonl_logger_close_is_idempotent(tmp_path: Path) -> None:
    logger = JsonlInferenceLogger(tmp_path, strategy_name="CLOSE", fsync=False)
    logger.write(_example_record())
    logger.close()
    logger.close()  # second call must not raise


def test_jsonl_logger_creates_log_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir"

    logger = JsonlInferenceLogger(nested, strategy_name="MKDIR", fsync=False)
    logger.write(_example_record())
    logger.close()

    assert nested.is_dir()
    files = list(nested.glob("inference_MKDIR_*.jsonl"))
    assert len(files) == 1


def test_jsonl_logger_exposes_log_dir_and_strategy_name(tmp_path: Path) -> None:
    logger = JsonlInferenceLogger(tmp_path, strategy_name="PROPS", fsync=False)

    assert logger.log_dir == tmp_path
    assert logger.strategy_name == "PROPS"
    logger.close()
