"""Phase 2 tests — the sealed BaseStrategy emission seam (LOGGING_CONTRACT §5/§5.1/§8/§5.3)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.config import StrategyConfig
from okmich_quant_core.logging import BaseEventLogger, BarOutcome, JsonlEventLogger, LogEventType, RunnerIdentity
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
    def __init__(self, *, signal=None, fail=False, new=True, logger=None):
        super().__init__(StrategyConfig(name="s", symbol="X", timeframe=5, magic=7),
                         signal if signal is not None else BaseSignal(),
                         inference_logger=logger if logger is not None else _RecordingLogger())
        self.bind_runner_identity(RunnerIdentity(runner_id="r-1", runner_start_token="tok",
                                                 broker="b", account_id="a", broker_session_id="sess"))
        self._fail = fail
        self._new = new
        self.calls = 0

    def is_new_bar(self, run_dt):
        return self._new

    def on_new_bar(self):
        self.calls += 1
        if self._fail:
            raise ValueError("boom")


def _records(strat):
    return strat.log_binding.logger.records


# --------------------------------------------------------------------------------------
# §5.1 sealing
# --------------------------------------------------------------------------------------

def test_sealing_rejects_run_override():
    with pytest.raises(TypeError, match="sealed BaseStrategy.run"):
        class Bad(BaseStrategy):
            def run(self, run_dt):
                pass

            def on_new_bar(self):
                pass

            def is_new_bar(self, run_dt):
                return True


def test_sealing_rejects_emit_and_bind_overrides():
    with pytest.raises(TypeError, match="_emit_bar_record"):
        class BadEmit(BaseStrategy):
            def _emit_bar_record(self, *, asof_bar_ts, outcome):
                pass

            def on_new_bar(self):
                pass

            def is_new_bar(self, run_dt):
                return True

    with pytest.raises(TypeError, match="bind_runner_identity"):
        class BadBind(BaseStrategy):
            def bind_runner_identity(self, runner):
                pass

            def on_new_bar(self):
                pass

            def is_new_bar(self, run_dt):
                return True


def test_sealing_allows_hook_overrides():
    # Overriding the developer hooks is allowed (this is the sandbox, §5.2).
    class Good(BaseStrategy):
        def on_new_bar(self):
            pass

        def is_new_bar(self, run_dt):
            return False

        def manage_positions(self, run_dt, flag=False):
            return 0

    assert issubclass(Good, BaseStrategy)


# --------------------------------------------------------------------------------------
# §5 fail-closed default logger
# --------------------------------------------------------------------------------------

def test_default_logger_constructed_when_none_injected(tmp_path):
    class S(BaseStrategy):
        def on_new_bar(self):
            pass

        def is_new_bar(self, run_dt):
            return True

    s = S(StrategyConfig(name="dh", symbol="EURUSD", timeframe=5, magic=1), BaseSignal(), log_base=tmp_path)
    try:
        assert isinstance(s.log_binding.logger, JsonlEventLogger)
        assert s.log_binding.logger.directory == tmp_path / "dh" / "EURUSD" / "5" / "inference"
        assert s.log_binding.logical.logical_system_id == "dh/EURUSD/5"
    finally:
        s.cleanup()


# --------------------------------------------------------------------------------------
# §7.2 heartbeat: one bar per new bar, ok / error
# --------------------------------------------------------------------------------------

def test_bar_emitted_on_successful_new_bar():
    s = _Strat()
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    recs = _records(s)
    assert len(recs) == 1
    assert recs[0].envelope.event is LogEventType.BAR
    assert recs[0].outcome is BarOutcome.OK
    assert s.calls == 1


def test_bar_emitted_on_error_and_exception_reraised():
    s = _Strat(fail=True)
    with pytest.raises(ValueError, match="boom"):
        s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    recs = _records(s)
    assert len(recs) == 1
    assert recs[0].outcome is BarOutcome.ERROR  # heartbeat captured the failed cycle...
    assert s.calls == 1                          # ...and the exception still propagated (breaker bookkeeping)


def test_manage_positions_failure_on_new_bar_emits_error_and_reraises():
    # A position-manager failure on a NEW bar must still produce the Tier 0 outcome=error heartbeat
    # (the per-new-bar floor) and then re-raise for StrategyHealth — not be lost before emission.
    class _PMFail(BaseStrategy):
        def __init__(self):
            super().__init__(StrategyConfig(name="pm", symbol="X", timeframe=5, magic=1),
                             BaseSignal(), inference_logger=_RecordingLogger())
            self.bind_runner_identity(RunnerIdentity(runner_id="r", runner_start_token="t",
                                                     broker="b", account_id="a", broker_session_id=None))
            self.on_new_bar_called = False

        def is_new_bar(self, run_dt):
            return True

        def manage_positions(self, run_dt, flag=False):
            raise RuntimeError("pm boom")

        def on_new_bar(self):
            self.on_new_bar_called = True

    s = _PMFail()
    with pytest.raises(RuntimeError, match="pm boom"):
        s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    recs = _records(s)
    assert len(recs) == 1 and recs[0].outcome is BarOutcome.ERROR  # heartbeat captured the failed cycle
    assert s.on_new_bar_called is False                            # manage_positions raised first


def test_no_bar_when_not_a_new_bar():
    s = _Strat(new=False)
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    assert _records(s) == []
    assert s.calls == 0


def test_dup_guard_coalesces_subsecond_runs():
    s = _Strat()
    t = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    s.run(t)
    s.run(t.replace(microsecond=500_000))  # < 1s later → coalesced
    assert len(_records(s)) == 1
    assert s.calls == 1


def test_asof_bar_ts_is_utc_and_floored_to_last_complete_bar():
    s = _Strat()
    s.run(datetime(2026, 6, 1, 12, 3, tzinfo=timezone.utc))  # inside the 12:00 bar
    assert _records(s)[0].asof_bar_ts == "2026-06-01T11:55:00+00:00"  # last COMPLETE 5m bar


# --------------------------------------------------------------------------------------
# Tier 1 from get_signal_context + §8 isolation
# --------------------------------------------------------------------------------------

def test_tier1_populated_from_get_signal_context():
    ctx = {"direction": 1, "confidence": 0.7, "bar_close": 1.23,
           "label_bar_ts": "2026-06-01T11:50:00+00:00",
           "features": {"f1": 0.5}, "extras": {"probs": [0.2, 0.8], "loglik": -2.1}}
    s = _Strat(signal=_CtxSignal(ctx))
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    rec = _records(s)[0]
    assert rec.direction == 1 and rec.confidence == 0.7 and rec.bar_close == 1.23
    assert rec.label_bar_ts == "2026-06-01T11:50:00+00:00"
    assert rec.features == {"f1": 0.5}
    assert rec.extras == {"probs": [0.2, 0.8], "loglik": -2.1}
    assert rec.tier1_error is None


def test_tier1_serialisation_failure_preserves_tier0():
    s = _Strat(signal=_CtxSignal({"extras": {"x": object()}}))  # non-JSON value
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    rec = _records(s)[0]
    assert rec.outcome is BarOutcome.OK            # Tier 0 floor survives
    assert rec.asof_bar_ts is not None
    assert rec.extras == {}                        # toxic Tier 1 dropped
    assert rec.tier1_error is not None             # ...but annotated, not silent


def test_tier1_non_finite_float_preserves_tier0():
    # NaN/Inf (common from indicators on warm-up bars) must NOT be written as bare NaN/Infinity
    # tokens (invalid JSON for strict readers) — the guard degrades to Tier 0 + annotation.
    s = _Strat(signal=_CtxSignal({"features": {"f": float("nan")}}))
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    rec = _records(s)[0]
    assert rec.outcome is BarOutcome.OK
    assert rec.features == {} and rec.tier1_error is not None


def test_cleanup_isolates_notifier_failure_from_logger_close():
    # The inference logger (which drains queued bars) must be closed even if the notifier raises.
    class _ClosableLogger(_RecordingLogger):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    class _BoomNotifier:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            raise RuntimeError("notifier boom")

    log = _ClosableLogger()
    s = _Strat(logger=log)
    s.notifier = _BoomNotifier()
    s.cleanup()  # must not raise despite the notifier blowing up
    assert log.closed is True       # logger drained/closed regardless
    assert s.notifier.closed is True


def test_tier1_get_signal_context_exception_preserves_tier0():
    s = _Strat(signal=_CtxSignal(raise_exc=True))
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    rec = _records(s)[0]
    assert rec.outcome is BarOutcome.OK
    assert rec.tier1_error is not None and "ctx boom" in rec.tier1_error


def test_error_bar_has_asof_ts_and_no_tier1():
    # §5.3: the error heartbeat is fully populated (asof_bar_ts derived before any fetch). The cycle
    # RAISED, so get_signal_context() may hold the previous bar's values — Tier 1 must NOT be read
    # for an error outcome (it would be a stale-data lie), even though the signal has a context.
    ctx = {"direction": 1, "confidence": 0.9, "features": {"f1": 0.5}}
    s = _Strat(signal=_CtxSignal(ctx), fail=True)
    with pytest.raises(ValueError, match="boom"):
        s.run(datetime(2026, 6, 1, 12, 3, tzinfo=timezone.utc))
    rec = _records(s)[0]
    assert rec.outcome is BarOutcome.ERROR
    assert rec.asof_bar_ts == "2026-06-01T11:55:00+00:00"   # present despite the failure
    assert rec.direction is None and rec.confidence is None  # no stale Tier 1 on an error bar
    assert rec.features == {} and rec.extras == {}


def test_default_logger_emits_to_disk_end_to_end(tmp_path):
    # Closes the invariant-2 gap: a strategy with NO injected logger must not only construct a
    # default JsonlEventLogger but actually write a bar through it to the on-disk JSONL.
    import json

    class S(BaseStrategy):
        def on_new_bar(self):
            pass

        def is_new_bar(self, run_dt):
            return True

    s = S(StrategyConfig(name="dh", symbol="EURUSD", timeframe=5, magic=1), BaseSignal(), log_base=tmp_path)
    s.bind_runner_identity(RunnerIdentity(runner_id="r-1", runner_start_token="tok",
                                          broker="b", account_id="a", broker_session_id=None))
    s.run(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
    s.cleanup()  # drains the bounded bar queue + closes the handle

    # exactly one inference file under the OPS §7 path, with one bar record
    files = list((tmp_path / "dh" / "EURUSD" / "5" / "inference").glob("inference_*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(x) for x in files[0].read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1 and lines[0]["event"] == "bar" and lines[0]["outcome"] == "ok"
    assert lines[0]["logical_system_id"] == "dh/EURUSD/5"
