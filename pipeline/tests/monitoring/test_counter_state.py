"""Tests for ViolationCounterState — counter advancement, threshold trip, arm/re-arm, persistence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.posterior_inference import (
    FeatureHealthReport,
    LoglikDriftReport,
    MonitoringCycleReport,
    PosteriorHealthReport,
)
from okmich_quant_pipeline.monitoring import ViolationCounterState


def _make_report(posterior_ok: bool, feature_ok: bool, loglik_ok: bool) -> MonitoringCycleReport:
    posterior = PosteriorHealthReport(
        overall_ok=posterior_ok, entropy_staleness_z=0.5, entropy_staleness_ok=posterior_ok,
        occupancy_drift_l1=0.05, occupancy_drift_ok=posterior_ok,
        flip_rate_drift_signed=0.01, flip_rate_drift_ok=posterior_ok,
    )
    feature = FeatureHealthReport(
        overall_ok=feature_ok, ks_statistics=np.array([0.05, 0.06]),
        p_values=np.array([0.5, 0.4]), per_feature_ok=np.array([feature_ok, feature_ok]),
        feature_names=("f0", "f1"),
    )
    loglik = LoglikDriftReport(overall_ok=loglik_ok, loglik_drift_z=0.5, loglik_drift_ok=loglik_ok)
    return MonitoringCycleReport(
        overall_ok=posterior_ok and feature_ok and loglik_ok,
        posterior=posterior, feature=feature, loglik=loglik,
        log_window_n=500, log_window_start_ts=pd.Timestamp("2026-05-01T00:00:00Z"),
        log_window_end_ts=pd.Timestamp("2026-05-01T23:55:00Z"),
    )


def test_counter_state_default_init() -> None:
    state = ViolationCounterState()
    assert state.posterior_consecutive_failures == 0
    assert state.posterior_alert_armed is True
    assert state.last_cycle_utc is None


def test_counter_state_passing_cycle_keeps_counters_zero() -> None:
    state = ViolationCounterState()
    tripped = state.advance(_make_report(True, True, True), threshold=3)

    assert tripped == []
    assert state.posterior_consecutive_failures == 0
    assert state.feature_consecutive_failures == 0
    assert state.loglik_consecutive_failures == 0


def test_counter_state_increments_on_failure() -> None:
    state = ViolationCounterState()

    tripped = state.advance(_make_report(False, True, True), threshold=3)
    assert tripped == []
    assert state.posterior_consecutive_failures == 1


def test_counter_state_trips_at_threshold() -> None:
    state = ViolationCounterState()
    for _ in range(2):
        assert state.advance(_make_report(False, True, True), threshold=3) == []
    tripped = state.advance(_make_report(False, True, True), threshold=3)
    assert tripped == ["posterior"]
    assert state.posterior_consecutive_failures == 3
    assert state.posterior_alert_armed is False


def test_counter_state_does_not_double_trip_when_already_disarmed() -> None:
    state = ViolationCounterState()
    for _ in range(3):
        state.advance(_make_report(False, True, True), threshold=3)
    # 4th and 5th continued-failing cycles should not re-trip
    tripped_4 = state.advance(_make_report(False, True, True), threshold=3)
    tripped_5 = state.advance(_make_report(False, True, True), threshold=3)
    assert tripped_4 == []
    assert tripped_5 == []
    assert state.posterior_consecutive_failures == 5


def test_counter_state_rearms_after_passing_cycle() -> None:
    state = ViolationCounterState()
    for _ in range(3):
        state.advance(_make_report(False, True, True), threshold=3)
    # Pass: counter resets, alert re-arms
    state.advance(_make_report(True, True, True), threshold=3)
    assert state.posterior_consecutive_failures == 0
    assert state.posterior_alert_armed is True
    # Fail 3 more times → trips again
    for _ in range(2):
        state.advance(_make_report(False, True, True), threshold=3)
    tripped = state.advance(_make_report(False, True, True), threshold=3)
    assert tripped == ["posterior"]


def test_counter_state_independent_per_gate() -> None:
    state = ViolationCounterState()
    # Cycle 1: posterior fails, others ok      → p=1, f=0, l=0
    # Cycle 2: posterior+feature fail           → p=2, f=1, l=0
    # Cycle 3: posterior recovers, feature fails → p=0, f=2, l=0
    # Cycle 4: feature fails again              → p=0, f=3 → trips
    state.advance(_make_report(False, True, True), threshold=3)
    state.advance(_make_report(False, False, True), threshold=3)
    state.advance(_make_report(True, False, True), threshold=3)
    tripped = state.advance(_make_report(True, False, True), threshold=3)
    assert tripped == ["feature"]
    assert state.posterior_consecutive_failures == 0
    assert state.feature_consecutive_failures == 3
    assert state.loglik_consecutive_failures == 0


def test_counter_state_multiple_gates_trip_in_same_cycle() -> None:
    state = ViolationCounterState()
    for _ in range(2):
        state.advance(_make_report(False, False, False), threshold=3)
    tripped = state.advance(_make_report(False, False, False), threshold=3)
    assert set(tripped) == {"posterior", "feature", "loglik"}


def test_counter_state_threshold_one_trips_immediately() -> None:
    state = ViolationCounterState()
    tripped = state.advance(_make_report(False, True, True), threshold=1)
    assert tripped == ["posterior"]


def test_counter_state_advance_rejects_invalid_threshold() -> None:
    state = ViolationCounterState()
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        state.advance(_make_report(True, True, True), threshold=0)


def test_counter_state_save_load_round_trip(tmp_path: Path) -> None:
    state = ViolationCounterState()
    state.advance(_make_report(False, True, True), threshold=3)
    state.advance(_make_report(False, True, True), threshold=3)
    path = tmp_path / "counter_state.json"
    state.save(path)

    restored = ViolationCounterState.load_or_init(path)
    assert restored.posterior_consecutive_failures == 2
    assert restored.posterior_alert_armed is True
    assert restored.last_overall_ok is False


def test_counter_state_load_or_init_missing_file_returns_default(tmp_path: Path) -> None:
    state = ViolationCounterState.load_or_init(tmp_path / "does_not_exist.json")
    assert state.posterior_consecutive_failures == 0
    assert state.posterior_alert_armed is True
