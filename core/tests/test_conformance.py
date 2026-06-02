"""LOGGING_CONTRACT v1.0.0 conformance harness (§12/§13).

The gate a developer's PR must pass: drive a real strategy through the framework — a successful bar,
an error bar, a breaker trip + re-enable (dispatch), and a runner startup + shutdown — then validate
EVERY emitted inference-log record against the shipped JSON Schema for its event, AND the runner
status file (running + stopped) against the status schema. Also assert the sealing guard rejects a
run()/_emit override and that a Tier-1 failure still yields a valid Tier-0 record. Covers all three
inference-log LogEventTypes + the runner-lifecycle status file end-to-end.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import jsonschema
import pytest

from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.config import RunLoopConfig, StrategyConfig
from okmich_quant_core.logging import BarOutcome, BaseEventLogger, LogEventType, RunnerIdentity, load_schema
from okmich_quant_core.multi_trader import MultiTrader
from okmich_quant_core.run_loop import RunLoop
from okmich_quant_core.signal import BaseSignal


class _RecordingLogger(BaseEventLogger):
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class _CtxSignal(BaseSignal):
    def __init__(self, ctx=None, raise_exc=False):
        super().__init__()
        self._ctx = ctx
        self._raise = raise_exc

    def get_signal_context(self):
        if self._raise:
            raise RuntimeError("ctx boom")
        return self._ctx


class _Strat(BaseStrategy):
    def __init__(self, name="conf", signal=None):
        self.rec = _RecordingLogger()
        super().__init__(StrategyConfig(name=name, symbol="EURUSD", timeframe=5, magic=99),
                         signal if signal is not None else BaseSignal(), inference_logger=self.rec)
        self.should_fail = False

    def is_new_bar(self, run_dt):
        return True

    def on_new_bar(self):
        if self.should_fail:
            raise ValueError("conformance boom")


class _FakeSession:
    broker = "Deriv-Server-01"
    account_id = "987654"
    broker_session_id = "terminal-7"

    def disconnect(self) -> bool:
        return True


_seq = [0]


def _dt():
    _seq[0] += 1
    return datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(minutes=_seq[0])


def _assert_valid(record):
    """Every emitted record MUST validate against the shipped JSON Schema for its event (§13)."""
    jsonschema.validate(record.to_dict(), load_schema(record.envelope.event))


def test_all_emitted_records_validate_against_schema(tmp_path):
    s = _Strat(signal=_CtxSignal({"direction": 1, "confidence": 0.8, "features": {"f": 0.1},
                                   "extras": {"probs": [0.2, 0.8], "loglik": -2.0}}))
    rl = RunLoop(RunLoopConfig(), MultiTrader([s], max_consecutive_errors=2),
                 broker_session=_FakeSession(), runner_name="conf_sys", log_base=tmp_path)
    status_path = tmp_path / "conf" / "EURUSD" / "5" / "status.json"

    rl._startup()                 # writes running status
    jsonschema.validate(json.loads(status_path.read_text(encoding="utf-8")), load_schema("runner_status"))
    rl.trader.run(_dt())          # bar ok
    s.should_fail = True
    rl.trader.run(_dt())          # bar error (consecutive 1)
    rl.trader.run(_dt())          # bar error (consecutive 2) -> circuit_breaker_tripped
    rl.trader.run(_dt())          # skipped_disabled bar
    rl.trader.enable_all()        # strategy_reenabled
    rl._shutdown("operator")      # writes stopped status

    seen_events = set()
    bar_outcomes = set()
    for record in s.rec.records:
        _assert_valid(record)
        seen_events.add(record.envelope.event)
        if record.envelope.event is LogEventType.BAR:
            bar_outcomes.add(record.outcome)

    # All three inference-log event types exercised + schema-validated end-to-end.
    assert seen_events == set(LogEventType)
    # ...and every BarOutcome was actually emitted (the harness drives ok, error, AND the dispatch
    # layer's skipped_disabled bar — the schema's `event: bar` const doesn't prove the outcome enum
    # was covered, so assert it explicitly here).
    assert bar_outcomes == set(BarOutcome)
    # The runner-lifecycle status file (stopped, proven disconnect) is schema-valid too.
    stopped = json.loads(status_path.read_text(encoding="utf-8"))
    jsonschema.validate(stopped, load_schema("runner_status"))
    assert stopped["state"] == "stopped" and stopped["broker_disconnected"] is True


def test_conformance_sealing_rejects_run_override():
    with pytest.raises(TypeError):
        class Bad(BaseStrategy):
            def run(self, run_dt):
                pass

            def on_new_bar(self):
                pass

            def is_new_bar(self, run_dt):
                return True


def test_conformance_tier1_failure_yields_valid_tier0():
    s = _Strat(signal=_CtxSignal(raise_exc=True))
    s.bind_runner_identity(RunnerIdentity(runner_id="r", runner_start_token="t",
                                          broker="b", account_id="a", broker_session_id=None))
    s.run(_dt())
    rec = s.rec.records[0]
    assert rec.envelope.event is LogEventType.BAR
    assert rec.tier1_error is not None       # Tier 1 failed...
    _assert_valid(rec)                       # ...but the Tier 0 record is still schema-valid
