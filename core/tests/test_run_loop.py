"""RunLoop runner-lifecycle tests — the status file (LOGGING_CONTRACT runner-lifecycle channel).

Exercises the extracted ``_startup`` / ``_shutdown`` (not the infinite ``run()`` loop): startup binds
every strategy and writes a ``running`` status; shutdown closes the trader (bars drained), disconnects,
then marks the status ``stopped`` with the PROVEN disconnect — read directly from status.json.
"""
from __future__ import annotations

import json

from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.config import RunLoopConfig, StrategyConfig
from okmich_quant_core.logging import BaseEventLogger
from okmich_quant_core.run_loop import RunLoop
from okmich_quant_core.signal import BaseSignal


class _NoopLogger(BaseEventLogger):
    """No-op inference logger — these tests exercise runner lifecycle (status file), not bars."""

    def write(self, record):
        pass

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class _Strat(BaseStrategy):
    def __init__(self, name="s", symbol="EURUSD"):
        super().__init__(StrategyConfig(name=name, symbol=symbol, timeframe=5, magic=1),
                         BaseSignal(), inference_logger=_NoopLogger())

    def is_new_bar(self, run_dt):
        return False

    def on_new_bar(self):
        pass


class _FakeTrader:
    def __init__(self, strategies):
        self.strategies = strategies
        self.closed = False

    def run(self, run_dt):
        pass

    def check_positions(self, run_dt):
        pass

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, proven=True):
        self.broker = "Deriv-Server"
        self.account_id = "123456"
        self.broker_session_id = "term-1"
        self._proven = proven
        self.disconnect_calls = 0

    def disconnect(self) -> bool:
        self.disconnect_calls += 1
        return self._proven


def _status(tmp_path, strategy="s") -> dict:
    # Runner-scoped status: ONE file at the runner root <log_base>/<strategy>/status.json (a single
    # Trader → no -multi suffix; LOGGING_CONTRACT §7.1).
    return json.loads((tmp_path / strategy / "status.json").read_text(encoding="utf-8"))


def test_startup_binds_strategies_and_writes_running_status(tmp_path):
    s = _Strat(symbol="EURUSD")
    rl = RunLoop(RunLoopConfig(), _FakeTrader([s]), broker_session=_FakeSession(),
                 runner_name="sys", log_base=tmp_path)
    rl._startup()
    assert s.log_binding.is_bound
    st = _status(tmp_path)
    assert st["state"] == "running"
    assert st["broker"] == "Deriv-Server" and st["account_id"] == "123456"
    assert st["logical_systems"][0]["symbol"] == "EURUSD" and st["logical_systems"][0]["timeframe"] == 5
    assert st["broker_disconnected"] is None
    rl._startup()  # idempotent — still running
    assert _status(tmp_path)["state"] == "running"


def test_shutdown_closes_trader_disconnects_and_marks_stopped(tmp_path):
    trader = _FakeTrader([_Strat()])
    session = _FakeSession(proven=True)
    rl = RunLoop(RunLoopConfig(), trader, broker_session=session, log_base=tmp_path)
    rl._startup()
    rl._shutdown("operator")
    assert trader.closed is True             # bars drained via trader.close() before disconnect
    assert session.disconnect_calls == 1
    st = _status(tmp_path)
    assert st["state"] == "stopped" and st["broker_disconnected"] is True
    assert st["clean"] is True and st["reason"] == "operator"
    rl._shutdown("again")  # idempotent — exactly one runner shutdown
    assert session.disconnect_calls == 1
    assert _status(tmp_path)["reason"] == "operator"


def test_shutdown_broker_disconnected_false_when_not_proven(tmp_path):
    rl = RunLoop(RunLoopConfig(), _FakeTrader([_Strat()]), broker_session=_FakeSession(proven=False),
                 log_base=tmp_path)
    rl._startup()
    rl._shutdown("x")
    assert _status(tmp_path)["broker_disconnected"] is False


def test_request_stop_sets_flag(tmp_path):
    rl = RunLoop(RunLoopConfig(), _FakeTrader([_Strat()]), log_base=tmp_path)
    assert rl._stop_requested is False
    rl._request_stop()
    assert rl._stop_requested is True


def test_run_shuts_down_on_stop_request(monkeypatch, tmp_path):
    """run() must mark the status stopped on the stop path (not only KeyboardInterrupt)."""
    import pytest
    trader = _FakeTrader([_Strat()])
    rl = RunLoop(RunLoopConfig(), trader, broker_session=_FakeSession(), log_base=tmp_path)
    monkeypatch.setattr(rl, "_install_signal_handlers", lambda: None)  # don't touch the test process's SIGTERM
    rl._stop_requested = True  # loop body never runs; exercises startup + finally-shutdown + exit
    with pytest.raises(SystemExit):
        rl.run()
    assert trader.closed is True
    st = _status(tmp_path)
    assert st["state"] == "stopped" and st["broker_disconnected"] is True


def test_startup_failure_still_runs_shutdown(monkeypatch, tmp_path):
    """A failure during _startup must still disconnect the broker + close the trader (finally path)."""
    import pytest
    trader = _FakeTrader([_Strat()])
    session = _FakeSession()
    rl = RunLoop(RunLoopConfig(), trader, broker_session=session, log_base=tmp_path)
    monkeypatch.setattr(rl, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(rl, "_startup", lambda: (_ for _ in ()).throw(RuntimeError("startup boom")))
    with pytest.raises(RuntimeError, match="startup boom"):
        rl.run()
    assert trader.closed is True            # finally ran _shutdown despite the startup failure
    assert session.disconnect_calls == 1


def test_no_broker_session_degrades_but_still_writes_status(tmp_path):
    rl = RunLoop(RunLoopConfig(), _FakeTrader([_Strat()]), log_base=tmp_path)  # no session injected
    rl._startup()
    rl._shutdown("x")
    st = _status(tmp_path)
    assert st["state"] == "stopped" and st["broker_disconnected"] is False


def test_runloop_config_rejects_bad_intervals():
    """RunLoopConfig must reject intervals that break the `second % interval` scheduler (review hardening)."""
    import pytest
    from pydantic import ValidationError
    for bad in (0, -5, 30.5):
        with pytest.raises(ValidationError):
            RunLoopConfig(chk_position_interval=bad)
    with pytest.raises(ValidationError):
        RunLoopConfig(sleep_interval=0)
    assert RunLoopConfig(chk_position_interval=30.0).chk_position_interval == 30.0
