"""Tests for the log-based monitoring I/O layer.

Covers:
- InferenceLog construction, immutability, tail()
- read_inference_log: round-trip with known JSONL, warmup-row drop, multi-file
  concatenation, schema validation (missing keys, mismatched feature names,
  mismatched n_states), invalid JSON, file-not-found
- MonitoringCycleReport AND aggregation
- run_streaming_gates wiring + window-metadata capture
- load_posterior_and_loglik_baselines_from_metadata round-trip + missing-block errors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.posterior_inference import (
    FeatureHealthBaselines,
    InferenceLogFrame,
    LoglikDriftBaselines,
    MonitoringCycleReport,
    PosteriorHealthBaselines,
    fit_feature_health_baselines,
    fit_loglik_drift_baselines,
    fit_posterior_health_baselines,
    load_posterior_and_loglik_baselines_from_metadata,
    read_inference_log,
    run_streaming_gates,
)


# ---- fixtures ----------------------------------------------------------------

def _make_record(ts: str, features: dict[str, float], probs: list[float] | None, loglik: float | None,
                 bar_close: float = 100.0, label_ts: str | None = None) -> dict:
    """One inference-log record matching the core InferenceLogRecord schema.

    HMM-specific fields (probs, loglik, state_id, etc.) live under ``extras``.
    """
    if label_ts is None:
        label_ts = ts
    return {
        "wall_clock_utc": ts,
        "asof_bar_ts": ts,
        "label_bar_ts": label_ts,
        "bar_close": bar_close,
        "features": features,
        "direction": 0,
        "confidence": None if probs is None else float(max(probs)),
        "extras": {
            "engine": "fl3",
            "lag": 3,
            "progressive_skip": 1,
            "state_id": 0 if probs is None else int(np.argmax(probs)),
            "probs": probs,
            "loglik": loglik,
            "confirm_chain": [],
            "smoothed_dir_last10": [0] * 10,
            "signal": {"entries_long": 0, "entries_short": 0, "exits_long": 0, "exits_short": 0},
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---- InferenceLog ------------------------------------------------------------

def test_inference_log_arrays_are_read_only() -> None:
    log = InferenceLogFrame(
        label_timestamps=pd.DatetimeIndex(["2026-05-01", "2026-05-02"]),
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        feature_names=("a", "b"),
        posteriors=np.array([[0.5, 0.5], [0.7, 0.3]], dtype=float),
        logliks=np.array([-1.0, -1.2], dtype=float),
    )

    with pytest.raises(ValueError, match="read-only|assignment destination"):
        log.features[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        log.posteriors[0, 0] = 99.0
    with pytest.raises(ValueError, match="read-only|assignment destination"):
        log.logliks[0] = 99.0


def test_inference_log_shape_properties() -> None:
    log = InferenceLogFrame(
        label_timestamps=pd.DatetimeIndex(["2026-05-01", "2026-05-02", "2026-05-03"]),
        features=np.zeros((3, 4), dtype=float),
        feature_names=("a", "b", "c", "d"),
        posteriors=np.zeros((3, 2), dtype=float),
        logliks=np.zeros(3, dtype=float),
    )

    assert log.n_bars == 3
    assert log.n_states == 2
    assert log.n_features == 4


def test_inference_log_tail_returns_last_n_rows() -> None:
    log = InferenceLogFrame(
        label_timestamps=pd.DatetimeIndex(["2026-05-01", "2026-05-02", "2026-05-03"]),
        features=np.array([[1.0], [2.0], [3.0]], dtype=float),
        feature_names=("a",),
        posteriors=np.array([[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], dtype=float),
        logliks=np.array([-1.0, -1.1, -1.2], dtype=float),
    )

    tail = log.tail(2)

    assert tail.n_bars == 2
    np.testing.assert_array_equal(tail.features, [[2.0], [3.0]])
    np.testing.assert_array_equal(tail.logliks, [-1.1, -1.2])
    assert tail.label_timestamps[0] == pd.Timestamp("2026-05-02")


def test_inference_log_tail_n_greater_than_length_returns_self() -> None:
    log = InferenceLogFrame(
        label_timestamps=pd.DatetimeIndex(["2026-05-01"]),
        features=np.zeros((1, 1), dtype=float),
        feature_names=("a",),
        posteriors=np.zeros((1, 2), dtype=float),
        logliks=np.zeros(1, dtype=float),
    )

    tail = log.tail(10)

    assert tail is log


def test_inference_log_tail_rejects_negative_n() -> None:
    log = InferenceLogFrame(
        label_timestamps=pd.DatetimeIndex(["2026-05-01"]),
        features=np.zeros((1, 1), dtype=float),
        feature_names=("a",),
        posteriors=np.zeros((1, 2), dtype=float),
        logliks=np.zeros(1, dtype=float),
    )

    with pytest.raises(ValueError, match="n must be >= 0"):
        log.tail(-1)


# ---- read_inference_log ------------------------------------------------------

def test_read_inference_log_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "inference_fl3_test.jsonl"
    records = [
        _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": -0.001}, [0.7, 0.2, 0.1], -2.34),
        _make_record("2026-05-01T00:05:00", {"tsi": 0.2, "dbl_rets": 0.002}, [0.1, 0.8, 0.1], -2.10),
    ]
    _write_jsonl(path, records)

    log = read_inference_log(path)

    assert log.n_bars == 2
    assert log.feature_names == ("tsi", "dbl_rets")
    assert log.n_states == 3
    np.testing.assert_allclose(log.features, [[0.1, -0.001], [0.2, 0.002]])
    np.testing.assert_allclose(log.posteriors, [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    np.testing.assert_allclose(log.logliks, [-2.34, -2.10])


def test_read_inference_log_drops_warmup_rows(tmp_path: Path) -> None:
    path = tmp_path / "inference_fl3.jsonl"
    records = [
        _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": -0.001}, None, None),
        _make_record("2026-05-01T00:05:00", {"tsi": 0.2, "dbl_rets": 0.002}, [0.1, 0.8, 0.1], -2.10),
        _make_record("2026-05-01T00:10:00", {"tsi": 0.3, "dbl_rets": 0.003}, None, -3.0),
        _make_record("2026-05-01T00:15:00", {"tsi": 0.4, "dbl_rets": 0.004}, [0.5, 0.3, 0.2], None),
        _make_record("2026-05-01T00:20:00", {"tsi": 0.5, "dbl_rets": 0.005}, [0.3, 0.4, 0.3], -2.5),
    ]
    _write_jsonl(path, records)

    log = read_inference_log(path)

    assert log.n_bars == 2
    np.testing.assert_allclose(log.logliks, [-2.10, -2.5])


def test_read_inference_log_drops_null_label_bar_ts_rows(tmp_path: Path) -> None:
    path = tmp_path / "inference_fl3.jsonl"
    valid = _make_record("2026-05-01T00:05:00", {"tsi": 0.2, "dbl_rets": 0.002}, [0.1, 0.8, 0.1], -2.10)
    null_label = _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": -0.001}, [0.7, 0.2, 0.1], -2.0)
    null_label["label_bar_ts"] = None
    _write_jsonl(path, [null_label, valid])

    log = read_inference_log(path)

    assert log.n_bars == 1
    np.testing.assert_allclose(log.logliks, [-2.10])


def test_read_inference_log_concatenates_multiple_files(tmp_path: Path) -> None:
    p1 = tmp_path / "inference_a.jsonl"
    p2 = tmp_path / "inference_b.jsonl"
    _write_jsonl(p1, [
        _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": -0.001}, [0.7, 0.3], -1.0),
    ])
    _write_jsonl(p2, [
        _make_record("2026-05-01T00:05:00", {"tsi": 0.2, "dbl_rets": 0.002}, [0.4, 0.6], -1.5),
        _make_record("2026-05-01T00:10:00", {"tsi": 0.3, "dbl_rets": 0.003}, [0.5, 0.5], -1.2),
    ])

    log = read_inference_log([p1, p2])

    assert log.n_bars == 3
    np.testing.assert_allclose(log.logliks, [-1.0, -1.5, -1.2])


def test_read_inference_log_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "inference.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(json.dumps(_make_record("2026-05-01T00:00:00", {"tsi": 0.1}, [0.7, 0.3], -1.0)) + "\n")
        fh.write("   \n")

    log = read_inference_log(path)

    assert log.n_bars == 1


def test_read_inference_log_raises_on_invalid_json_in_middle(tmp_path: Path) -> None:
    """Invalid JSON in the middle of the file is corruption, not torn-write — raise."""
    path = tmp_path / "inference.jsonl"
    good = _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": -0.001}, [0.7, 0.2, 0.1], -2.0)
    good_after = _make_record("2026-05-01T00:10:00", {"tsi": 0.2, "dbl_rets": 0.002}, [0.4, 0.3, 0.3], -2.0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write("{not valid json}\n")
        fh.write(json.dumps(good_after) + "\n")

    with pytest.raises(ValueError, match="invalid JSON"):
        read_inference_log(path)


def test_read_inference_log_tolerates_torn_last_line(tmp_path: Path) -> None:
    """A torn final line (trader killed mid-write) is skipped with a warning, not raised."""
    path = tmp_path / "inference.jsonl"
    good = _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": -0.001}, [0.7, 0.2, 0.1], -2.0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(good) + "\n")
        fh.write('{"label_bar_ts": "2026-05-01T00:05:00", "features": {"tsi": 0.2, "db')  # truncated

    with pytest.warns(RuntimeWarning, match="truncated/invalid last line"):
        log = read_inference_log(path)

    assert log.n_bars == 1


def test_read_inference_log_rejects_missing_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "inference.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"label_bar_ts": "2026-05-01T00:00:00", "features": {}}) + "\n")

    with pytest.raises(ValueError, match=r"missing required keys \['extras'\]"):
        read_inference_log(path)


def test_read_inference_log_rejects_non_mapping_extras(tmp_path: Path) -> None:
    path = tmp_path / "inference.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "label_bar_ts": "2026-05-01T00:00:00", "features": {"tsi": 0.1}, "extras": ["wrong", "type"],
        }) + "\n")

    with pytest.raises(ValueError, match="'extras' must be a mapping"):
        read_inference_log(path)


def test_read_inference_log_rejects_feature_name_change(tmp_path: Path) -> None:
    path = tmp_path / "inference.jsonl"
    _write_jsonl(path, [
        _make_record("2026-05-01T00:00:00", {"tsi": 0.1, "dbl_rets": 0.0}, [0.7, 0.3], -1.0),
        _make_record("2026-05-01T00:05:00", {"tsi": 0.2, "different_name": 0.0}, [0.4, 0.6], -1.5),
    ])

    with pytest.raises(ValueError, match="feature-name set changed"):
        read_inference_log(path)


def test_read_inference_log_rejects_n_states_change(tmp_path: Path) -> None:
    path = tmp_path / "inference.jsonl"
    _write_jsonl(path, [
        _make_record("2026-05-01T00:00:00", {"tsi": 0.1}, [0.7, 0.3], -1.0),
        _make_record("2026-05-01T00:05:00", {"tsi": 0.2}, [0.4, 0.3, 0.3], -1.5),
    ])

    with pytest.raises(ValueError, match="posterior length changed"):
        read_inference_log(path)


def test_read_inference_log_rejects_all_warmup(tmp_path: Path) -> None:
    path = tmp_path / "inference.jsonl"
    _write_jsonl(path, [
        _make_record("2026-05-01T00:00:00", {"tsi": 0.1}, None, None),
        _make_record("2026-05-01T00:05:00", {"tsi": 0.2}, None, None),
    ])

    with pytest.raises(ValueError, match="no fully-populated rows found"):
        read_inference_log(path)


def test_read_inference_log_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="log file not found"):
        read_inference_log(tmp_path / "does_not_exist.jsonl")


def test_read_inference_log_rejects_empty_paths() -> None:
    with pytest.raises(ValueError, match="paths is empty"):
        read_inference_log([])


# ---- run_streaming_gates -----------------------------------------------------

def _build_realistic_log(n: int = 500, n_features: int = 3, n_states: int = 3, seed: int = 0) -> InferenceLogFrame:
    """In-distribution log of n bars built deterministically for gate testing."""
    rng = np.random.default_rng(seed)
    return InferenceLogFrame(
        label_timestamps=pd.date_range("2026-05-01", periods=n, freq="5min"),
        features=rng.normal(size=(n, n_features)),
        feature_names=tuple(f"f{i}" for i in range(n_features)),
        posteriors=rng.dirichlet(np.ones(n_states), size=n),
        logliks=rng.normal(loc=-2.0, scale=0.3, size=n),
    )


def test_run_streaming_gates_returns_report_with_window_metadata() -> None:
    log = _build_realistic_log(n=500, seed=1)
    posterior_baselines = fit_posterior_health_baselines(log.posteriors, window=20)
    feature_baselines = fit_feature_health_baselines(log.features, feature_names=list(log.feature_names))
    loglik_baselines = fit_loglik_drift_baselines(log.logliks, window=20)

    report = run_streaming_gates(log, posterior_baselines, feature_baselines, loglik_baselines)

    assert isinstance(report, MonitoringCycleReport)
    assert report.log_window_n == 500
    assert report.log_window_start_ts == pd.Timestamp("2026-05-01 00:00:00")
    assert report.log_window_end_ts == log.label_timestamps[-1]


def test_run_streaming_gates_overall_ok_is_and_of_components() -> None:
    log = _build_realistic_log(n=500, seed=2)
    posterior_baselines = fit_posterior_health_baselines(log.posteriors, window=20)
    feature_baselines = fit_feature_health_baselines(log.features, feature_names=list(log.feature_names))
    loglik_baselines = fit_loglik_drift_baselines(log.logliks, window=20)

    report = run_streaming_gates(log, posterior_baselines, feature_baselines, loglik_baselines)

    expected = report.posterior.overall_ok and report.feature.overall_ok and report.loglik.overall_ok
    assert report.overall_ok == expected


def test_run_streaming_gates_passes_when_log_drawn_from_baseline_distribution() -> None:
    base = _build_realistic_log(n=3000, seed=10)
    posterior_baselines = fit_posterior_health_baselines(base.posteriors, window=30)
    feature_baselines = fit_feature_health_baselines(base.features, feature_names=list(base.feature_names))
    loglik_baselines = fit_loglik_drift_baselines(base.logliks, window=30)

    # Live log: same generation process, different seed but same distribution.
    live = _build_realistic_log(n=500, seed=11)

    report = run_streaming_gates(live, posterior_baselines, feature_baselines, loglik_baselines)

    assert report.posterior.overall_ok is True
    assert report.feature.overall_ok is True
    assert report.loglik.overall_ok is True
    assert report.overall_ok is True


def test_run_streaming_gates_rejects_empty_log() -> None:
    empty_log = InferenceLogFrame(
        label_timestamps=pd.DatetimeIndex([], dtype="datetime64[ns]"),
        features=np.zeros((0, 2), dtype=float),
        feature_names=("a", "b"),
        posteriors=np.zeros((0, 3), dtype=float),
        logliks=np.zeros(0, dtype=float),
    )
    posterior_baselines = fit_posterior_health_baselines(np.random.default_rng(0).dirichlet([1, 1, 1], size=200),
                                                         window=10)
    feature_baselines = fit_feature_health_baselines(np.random.default_rng(0).normal(size=(200, 2)),
                                                     feature_names=["a", "b"])
    loglik_baselines = fit_loglik_drift_baselines(np.random.default_rng(0).normal(size=200), window=10)

    with pytest.raises(ValueError, match="log is empty"):
        run_streaming_gates(empty_log, posterior_baselines, feature_baselines, loglik_baselines)


# ---- load_posterior_and_loglik_baselines_from_metadata -----------------------

def test_load_baselines_from_metadata_round_trips() -> None:
    rng = np.random.default_rng(20)
    posterior_baselines = fit_posterior_health_baselines(rng.dirichlet([1, 1, 1], size=500), window=20)
    loglik_baselines = fit_loglik_drift_baselines(rng.normal(size=500), window=20)
    metadata = {
        "monitoring_baselines": {
            "window": 20,
            "posterior": posterior_baselines.to_dict(),
            "loglik": loglik_baselines.to_dict(),
        }
    }

    restored_posterior, restored_loglik = load_posterior_and_loglik_baselines_from_metadata(metadata)

    assert restored_posterior.window == posterior_baselines.window
    assert restored_posterior.entropy_mean == pytest.approx(posterior_baselines.entropy_mean, abs=1e-15)
    np.testing.assert_allclose(restored_posterior.occupancy, posterior_baselines.occupancy)
    assert restored_loglik.window == loglik_baselines.window
    assert restored_loglik.loglik_mean == pytest.approx(loglik_baselines.loglik_mean, abs=1e-15)


def test_load_baselines_rejects_missing_monitoring_baselines_block() -> None:
    with pytest.raises(ValueError, match="missing 'monitoring_baselines' block"):
        load_posterior_and_loglik_baselines_from_metadata({"unrelated": "field"})


def test_load_baselines_rejects_missing_posterior_sub_block() -> None:
    metadata = {"monitoring_baselines": {"window": 20, "loglik": {}}}

    with pytest.raises(ValueError, match="missing 'posterior' sub-block"):
        load_posterior_and_loglik_baselines_from_metadata(metadata)


def test_load_baselines_rejects_missing_loglik_sub_block() -> None:
    metadata = {"monitoring_baselines": {"window": 20, "posterior": {}}}

    with pytest.raises(ValueError, match="missing 'loglik' sub-block"):
        load_posterior_and_loglik_baselines_from_metadata(metadata)


def test_load_baselines_rejects_non_mapping_block() -> None:
    metadata = {"monitoring_baselines": ["wrong", "type"]}

    with pytest.raises(ValueError, match="must be a mapping"):
        load_posterior_and_loglik_baselines_from_metadata(metadata)


# ---- end-to-end writer→reader contract --------------------------------------

def test_jsonl_writer_round_trips_through_reader(tmp_path: Path) -> None:
    """End-to-end: write records via the core JsonlInferenceLogger, read them back via
    read_inference_log, verify the contract is round-trippable.

    Catches schema drift between writer and reader that the hand-rolled fixture cannot.
    """
    from okmich_quant_core import InferenceLogRecord, JsonlInferenceLogger
    import pandas as pd

    logger = JsonlInferenceLogger(tmp_path, strategy_name="E2E", fsync=False)
    base_ts = pd.Timestamp("2026-05-01T00:00:00+00:00")
    records = []
    for i in range(3):
        ts = base_ts + pd.Timedelta(minutes=5 * i)
        records.append(InferenceLogRecord(
            wall_clock_utc=ts, asof_bar_ts=ts, label_bar_ts=ts,
            bar_close=100.0 + i, features={"tsi": 0.1 * i, "dbl_rets": 0.001 * i},
            direction=1, confidence=0.9,
            extras={
                "probs": [0.7 - 0.1 * i, 0.2, 0.1 + 0.1 * i],
                "loglik": -2.0 - 0.1 * i,
            },
        ))
    for r in records:
        logger.write(r)
    logger.close()

    log_files = list(tmp_path.glob("inference_E2E_*.jsonl"))
    assert len(log_files) == 1

    frame = read_inference_log(log_files[0])

    assert frame.n_bars == 3
    assert frame.feature_names == ("tsi", "dbl_rets")
    assert frame.n_states == 3
    np.testing.assert_allclose(frame.features[:, 0], [0.0, 0.1, 0.2])
    np.testing.assert_allclose(frame.logliks, [-2.0, -2.1, -2.2])
    np.testing.assert_allclose(frame.posteriors[0], [0.7, 0.2, 0.1])
