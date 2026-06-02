"""LOGGING_CONTRACT v1.0.0 foundation tests: identity, records, factory, schema, JsonlEventLogger."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone

import jsonschema
import pytest

from okmich_quant_core.logging.base import (
    LOG_SCHEMA_VERSION,
    BarOutcome,
    LogBinding,
    LogEventType,
    SystemRecordFactory,
    UnknownSchemaMajorError,
    _iso_utc,
    load_schema,
    record_from_dict,
)
from okmich_quant_core.logging.identity import LogRootConfigError, LogicalSystemIdentity, RunnerIdentity
from okmich_quant_core.logging.jsonl import JsonlEventLogger
from okmich_quant_core.logging.runner_status import RunnerStatus


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture
def runner() -> RunnerIdentity:
    return RunnerIdentity(runner_id="deriv_hmm-4321", runner_start_token="tok-abc",
                          broker="Deriv-Server-01", account_id="123456", broker_session_id="terminal-xyz")


@pytest.fixture
def logical() -> LogicalSystemIdentity:
    return LogicalSystemIdentity(strategy="deriv_hmm_posterior", symbol="EURUSD", timeframe_minutes=5)


@pytest.fixture
def sys_factory(runner, logical) -> SystemRecordFactory:
    # fixed clock so wall_clock_utc is deterministic
    return SystemRecordFactory(runner, logical, order_tag=777,
                               clock=lambda: datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))


def _validate(record) -> dict:
    payload = record.to_dict()
    jsonschema.validate(payload, load_schema(record.envelope.event))
    return payload


# --------------------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------------------

def test_logical_system_id_and_path_parts(logical):
    assert logical.logical_system_id == "deriv_hmm_posterior/EURUSD/5"
    assert logical.path_parts() == ("deriv_hmm_posterior", "EURUSD", "5")


def test_runner_identity_generate_id_and_unique_token():
    a = RunnerIdentity.generate(name="sys", broker="B", account_id="1", pid=999)
    b = RunnerIdentity.generate(name="sys", broker="B", account_id="1", pid=999)
    assert a.runner_id == "sys-999"
    assert a.runner_start_token != b.runner_start_token  # fresh token per process incarnation


def test_path_parts_sanitises_separators():
    ident = LogicalSystemIdentity(strategy="a/b", symbol="X\\Y", timeframe_minutes=15)
    assert ident.path_parts() == ("a_b", "X_Y", "15")


# --------------------------------------------------------------------------------------
# _iso_utc
# --------------------------------------------------------------------------------------

def test_iso_utc_assumes_utc_for_naive():
    assert _iso_utc(datetime(2026, 1, 1, 12, 0)) == "2026-01-01T12:00:00+00:00"


def test_iso_utc_converts_aware_to_utc():
    from datetime import timedelta
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    assert _iso_utc(aware) == "2026-01-01T10:00:00+00:00"


def test_iso_utc_none_passthrough():
    assert _iso_utc(None) is None


# --------------------------------------------------------------------------------------
# records + factory + schema validation + round-trip
# --------------------------------------------------------------------------------------

def test_bar_record_has_full_envelope_and_validates(sys_factory):
    rec = sys_factory.bar(asof_bar_ts=datetime(2026, 6, 1, 11, 55, tzinfo=timezone.utc), outcome=BarOutcome.OK,
                          bar_close=1.2345, direction=1, confidence=0.8,
                          features={"f1": 0.5}, extras={"probs": [0.1, 0.9], "loglik": -2.3})
    payload = _validate(rec)
    assert payload["event"] == "bar"
    assert payload["log_schema_version"] == LOG_SCHEMA_VERSION
    assert payload["runner_id"] == "deriv_hmm-4321"
    assert payload["logical_system_id"] == "deriv_hmm_posterior/EURUSD/5"
    assert payload["symbol"] == "EURUSD" and payload["timeframe"] == 5
    assert payload["order_tag"] == 777
    assert payload["asof_bar_ts"] == "2026-06-01T11:55:00+00:00"
    assert payload["outcome"] == "ok"
    assert payload["extras"] == {"probs": [0.1, 0.9], "loglik": -2.3}


def test_bar_record_emits_explicit_nulls_for_absent_tier1(sys_factory):
    rec = sys_factory.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.ERROR)
    payload = _validate(rec)
    # null, not omitted — consumers distinguish "no signal" from "field missing"
    for k in ("label_bar_ts", "direction", "confidence", "bar_close"):
        assert k in payload and payload[k] is None
    assert payload["features"] == {} and payload["extras"] == {}
    assert "tier1_error" not in payload  # only present when Tier-1 actually failed


def test_bar_record_tier1_error_annotation_present_only_when_set(sys_factory):
    rec = sys_factory.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK,
                          tier1_error="non-serialisable extras")
    payload = _validate(rec)
    assert payload["tier1_error"] == "non-serialisable extras"


def test_circuit_breaker_record_validates(sys_factory):
    rec = sys_factory.circuit_breaker_tripped(consecutive_errors=5, last_error="boom")
    payload = _validate(rec)
    assert payload["event"] == "circuit_breaker_tripped"
    assert payload["consecutive_errors"] == 5 and payload["last_error"] == "boom"
    assert payload["symbol"] == "EURUSD"  # per-strategy scope keeps full envelope


def test_strategy_reenabled_record_validates(sys_factory):
    payload = _validate(sys_factory.strategy_reenabled(reason="operator"))
    assert payload["event"] == "strategy_reenabled" and payload["reason"] == "operator"


def test_record_round_trip_via_record_from_dict(sys_factory):
    rec = sys_factory.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK, direction=-1)
    back = record_from_dict(rec.to_dict())
    assert back.to_dict() == rec.to_dict()
    assert back.outcome is BarOutcome.OK and isinstance(back.outcome, BarOutcome)
    assert back.envelope.event is LogEventType.BAR


def test_record_from_dict_rejects_unknown_major(sys_factory):
    payload = sys_factory.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK).to_dict()
    payload["log_schema_version"] = "2.0.0"
    with pytest.raises(UnknownSchemaMajorError):
        record_from_dict(payload)


def test_record_from_dict_accepts_same_major_higher_minor(sys_factory):
    payload = sys_factory.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK).to_dict()
    payload["log_schema_version"] = "1.7.0"  # additive minor — consumers accept
    rec = record_from_dict(payload)
    assert rec.envelope.event is LogEventType.BAR


def test_record_from_dict_rejects_missing_event():
    with pytest.raises(ValueError):
        record_from_dict({"log_schema_version": "1.0.0"})


def test_retired_logging_names_raise_migration_error():
    import okmich_quant_core
    import okmich_quant_core.logging as core_logging
    for name in ("InferenceLogRecord", "BaseInferenceLogger", "JsonlInferenceLogger"):
        with pytest.raises(ImportError, match="removed in 0.7.0"):
            getattr(okmich_quant_core, name)
    # The runner-JSONL lifecycle names retired in this round also raise a migration hint.
    for name in ("RunnerRecordFactory", "build_runner_logger", "StartupRecord", "ShutdownRecord",
                 "JsonlInferenceLogger"):
        with pytest.raises(ImportError, match="removed in 0.7.0"):
            getattr(core_logging, name)


# --------------------------------------------------------------------------------------
# RunnerStatus — per-system status file (runner lifecycle)
# --------------------------------------------------------------------------------------

def test_runner_status_writes_running_then_stopped(runner, logical, tmp_path):
    rs = RunnerStatus(runner, [logical], log_base=tmp_path, pid=4321, library_versions={"okmich-quant-core": "0.7.0"})
    status_path = tmp_path / "deriv_hmm_posterior" / "EURUSD" / "5" / "status.json"

    rs.mark_started()
    running = json.loads(status_path.read_text(encoding="utf-8"))
    jsonschema.validate(running, load_schema("runner_status"))
    assert running["state"] == "running"
    assert running["runner_id"] == runner.runner_id and running["pid"] == 4321
    assert running["broker_disconnected"] is None and running["started_at"]
    assert running["logical_systems"][0]["symbol"] == "EURUSD"

    rs.mark_stopped(broker_disconnected=True, clean=True, reason="operator_stop")
    stopped = json.loads(status_path.read_text(encoding="utf-8"))
    jsonschema.validate(stopped, load_schema("runner_status"))
    assert stopped["state"] == "stopped"
    assert stopped["broker_disconnected"] is True and stopped["clean"] is True
    assert stopped["reason"] == "operator_stop" and stopped["stopped_at"]
    # atomic write leaves no temp file behind
    assert not list(status_path.parent.glob("*.tmp"))


def test_runner_status_multi_system_writes_each_path(runner, tmp_path):
    a = LogicalSystemIdentity(strategy="hmm", symbol="EURUSD", timeframe_minutes=5)
    b = LogicalSystemIdentity(strategy="hmm", symbol="GBPUSD", timeframe_minutes=5)
    rs = RunnerStatus(runner, [a, b], log_base=tmp_path)
    rs.mark_started()
    for sym in ("EURUSD", "GBPUSD"):
        p = tmp_path / "hmm" / sym / "5" / "status.json"
        assert p.is_file() and json.loads(p.read_text(encoding="utf-8"))["state"] == "running"


# --------------------------------------------------------------------------------------
# LogBinding two-phase lifecycle
# --------------------------------------------------------------------------------------

def test_log_binding_refuses_to_emit_before_bind(logical, runner, tmp_path):
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    try:
        binding = LogBinding(logical, logger, order_tag=42)
        assert binding.is_bound is False
        with pytest.raises(RuntimeError):
            binding.system_factory()
        binding.bind(runner)
        assert binding.is_bound is True
        fac = binding.system_factory()
        rec = fac.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK)
        assert rec.envelope.runner_id == runner.runner_id
        assert rec.envelope.order_tag == 42
    finally:
        logger.close()


# --------------------------------------------------------------------------------------
# JsonlEventLogger — path, durability split, back-pressure, drain, rotation
# --------------------------------------------------------------------------------------

def test_logger_uses_okmich_quant_log_base_env(logical, tmp_path, monkeypatch):
    env_root = tmp_path / "env_logs"
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(env_root))
    logger = JsonlEventLogger(logical)
    try:
        assert logger.directory == env_root / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference"
    finally:
        logger.close()


def test_logger_explicit_log_base_overrides_env(logical, tmp_path, monkeypatch):
    env_root = tmp_path / "env_logs"
    explicit_root = tmp_path / "explicit_logs"
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(env_root))
    logger = JsonlEventLogger(logical, log_base=explicit_root)
    try:
        assert logger.directory == explicit_root / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference"
    finally:
        logger.close()


def test_logger_fails_fast_without_log_base_or_env(logical, monkeypatch):
    monkeypatch.delenv("OKMICH_QUANT_LOG_BASE", raising=False)
    with pytest.raises(LogRootConfigError, match="OKMICH_QUANT_LOG_BASE"):
        JsonlEventLogger(logical)


def test_logger_fails_fast_on_blank_env(logical, monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", "   ")
    with pytest.raises(LogRootConfigError, match="OKMICH_QUANT_LOG_BASE"):
        JsonlEventLogger(logical)


def test_logger_path_layout(logical, tmp_path, monkeypatch):
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime",
                        _FrozenDatetime(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)))
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    try:
        f = SystemRecordFactory(_runner(), logical)
        logger.write(f.circuit_breaker_tripped(consecutive_errors=1))  # lifecycle = sync
        expected = tmp_path / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference" / "inference_20260601.jsonl"
        assert expected.is_file()
        line = json.loads(expected.read_text(encoding="utf-8").strip())
        assert line["event"] == "circuit_breaker_tripped"
    finally:
        logger.close()


def test_lifecycle_is_synchronous_bar_is_queued(logical, tmp_path, monkeypatch):
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime",
                        _FrozenDatetime(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)))
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    path = tmp_path / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference" / "inference_20260601.jsonl"
    try:
        f = SystemRecordFactory(_runner(), logical)
        # block the writer so the bar cannot be flushed yet
        entered, release = threading.Event(), threading.Event()
        orig = logger._write_sync

        def blocking(rec):
            if rec.envelope.event is LogEventType.BAR:
                entered.set()
                release.wait(5)
            orig(rec)

        logger._write_sync = blocking
        logger.write(f.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK))  # queued
        assert entered.wait(2)
        # lifecycle write must NOT wait on the (blocked) bar writer — it is synchronous on this thread
        logger.write(f.circuit_breaker_tripped(consecutive_errors=2))
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert [x["event"] for x in lines] == ["circuit_breaker_tripped"]  # bar not yet on disk
        release.set()
        logger._write_sync = orig
        logger.drain()
        events = sorted(json.loads(x)["event"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
        assert events == ["bar", "circuit_breaker_tripped"]
    finally:
        release.set()
        logger.close()


def test_backpressure_surfaces_and_never_blocks(logical, tmp_path):
    logger = JsonlEventLogger(logical, log_base=tmp_path, bar_queue_maxsize=1)
    try:
        f = SystemRecordFactory(_runner(), logical)
        entered, release = threading.Event(), threading.Event()

        def blocking(rec):
            entered.set()
            release.wait(5)

        logger._write_sync = blocking
        ts = "2026-06-01T11:55:00+00:00"
        logger.write(f.bar(asof_bar_ts=ts, outcome=BarOutcome.OK))  # pulled by writer -> blocks
        assert entered.wait(2)
        logger.write(f.bar(asof_bar_ts=ts, outcome=BarOutcome.OK))  # fills queue (maxsize=1)
        logger.write(f.bar(asof_bar_ts=ts, outcome=BarOutcome.OK))  # Full -> surfaced drop
        logger.write(f.bar(asof_bar_ts=ts, outcome=BarOutcome.OK))  # Full -> surfaced drop
        assert logger.backpressure_dropped == 2  # surfaced, not silent
        release.set()
    finally:
        release.set()
        logger.close()


def test_drain_flushes_queued_bars(logical, tmp_path, monkeypatch):
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime",
                        _FrozenDatetime(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)))
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    path = tmp_path / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference" / "inference_20260601.jsonl"
    try:
        f = SystemRecordFactory(_runner(), logical)
        for _ in range(20):
            logger.write(f.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK))
        logger.drain()
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(lines) == 20
    finally:
        logger.close()


def test_logger_rotates_on_utc_date_flip(logical, tmp_path, monkeypatch):
    clock = _FrozenDatetime(datetime(2026, 6, 1, 23, 59, tzinfo=timezone.utc))
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime", clock)
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    base = tmp_path / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference"
    try:
        f = SystemRecordFactory(_runner(), logical)
        logger.write(f.circuit_breaker_tripped(consecutive_errors=1))  # day 01
        clock.value = datetime(2026, 6, 2, 0, 1, tzinfo=timezone.utc)   # flip
        logger.write(f.circuit_breaker_tripped(consecutive_errors=2))  # day 02
        assert (base / "inference_20260601.jsonl").is_file()
        assert (base / "inference_20260602.jsonl").is_file()
    finally:
        logger.close()


def test_logger_close_is_idempotent(logical, tmp_path):
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    logger.close()
    logger.close()  # must not raise


def test_lifecycle_write_is_fsync_durable(logical, tmp_path, monkeypatch):
    """The durability-split claim: a lifecycle record fsyncs before write() returns (§10)."""
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime",
                        _FrozenDatetime(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)))
    calls: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.os.fsync",
                        lambda fd: (calls.append(fd), real_fsync(fd))[1])
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    try:
        f = SystemRecordFactory(_runner(), logical)
        logger.write(f.circuit_breaker_tripped(consecutive_errors=1))  # lifecycle = sync path
        assert calls, "lifecycle write must fsync before returning (durability contract §10)"
    finally:
        logger.close()


def test_fsync_disabled_skips_fsync(logical, tmp_path, monkeypatch):
    """fsync=False (test/throughput knob) must not call os.fsync — proves the flag is wired."""
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime",
                        _FrozenDatetime(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)))
    calls: list[int] = []
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.os.fsync", lambda fd: calls.append(fd))
    logger = JsonlEventLogger(logical, log_base=tmp_path, fsync=False)
    try:
        f = SystemRecordFactory(_runner(), logical)
        logger.write(f.circuit_breaker_tripped(consecutive_errors=1))
        assert calls == []
    finally:
        logger.close()


def test_drain_timeout_gives_up_without_hanging(logical, tmp_path):
    """The ops-critical non-blocking-shutdown guarantee: a wedged writer can't hang drain forever."""
    logger = JsonlEventLogger(logical, log_base=tmp_path, bar_queue_maxsize=10)
    release = threading.Event()
    try:
        f = SystemRecordFactory(_runner(), logical)
        logger._write_sync = lambda rec: release.wait(10)  # wedge the writer
        logger.write(f.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK))
        start = time.monotonic()
        logger.drain(timeout=0.2)  # must return ~promptly, not block on the wedged writer
        assert time.monotonic() - start < 5.0
        assert logger._queue.unfinished_tasks > 0  # the wedged record is still pending (not lost)
    finally:
        release.set()
        logger.close()


def test_write_after_close_is_surfaced_not_silent(logical, tmp_path):
    """A late write (e.g. a stray IB callback after shutdown) is rejected + counted, not silently queued."""
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    f = SystemRecordFactory(_runner(), logical)
    logger.close()
    logger.write(f.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK))
    logger.write(f.circuit_breaker_tripped(consecutive_errors=1))
    assert logger.post_close_dropped == 2  # surfaced via the counter, not a silent drop


def test_writer_thread_survives_a_bad_bar_write(logical, tmp_path, monkeypatch):
    """A bad write in the writer thread bumps write_errors and the thread keeps draining (jsonl.py)."""
    monkeypatch.setattr("okmich_quant_core.logging.jsonl.datetime",
                        _FrozenDatetime(datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)))
    logger = JsonlEventLogger(logical, log_base=tmp_path)
    path = tmp_path / "deriv_hmm_posterior" / "EURUSD" / "5" / "inference" / "inference_20260601.jsonl"
    try:
        f = SystemRecordFactory(_runner(), logical)
        orig = logger._write_sync
        state = {"first": True}

        def flaky(rec):
            if state["first"]:
                state["first"] = False
                raise OSError("disk hiccup")
            orig(rec)

        logger._write_sync = flaky
        logger.write(f.bar(asof_bar_ts="2026-06-01T11:55:00+00:00", outcome=BarOutcome.OK))  # fails
        logger.write(f.bar(asof_bar_ts="2026-06-01T11:56:00+00:00", outcome=BarOutcome.OK))  # succeeds
        logger.drain()
        assert logger.write_errors == 1  # surfaced
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(lines) == 1  # writer survived the first failure and wrote the second
    finally:
        logger.close()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _runner() -> RunnerIdentity:
    return RunnerIdentity(runner_id="r-1", runner_start_token="t", broker="B", account_id="1",
                          broker_session_id=None)


class _FrozenDatetime:
    """Minimal stand-in for the ``datetime`` module symbol used in jsonl.py (``datetime.now(tz)``)."""

    def __init__(self, value: datetime):
        self.value = value

    def now(self, tz=None):
        return self.value if tz is None else self.value.astimezone(tz)
