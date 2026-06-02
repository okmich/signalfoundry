"""Phase 4 tests — dispatch-layer inference records (LOGGING_CONTRACT §7.3).

Trader / MultiTrader hold each strategy's logger and emit circuit_breaker_tripped on trip,
strategy_reenabled on re-enable, and a skipped_disabled bar for a disabled strategy they skip.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.config import StrategyConfig
from okmich_quant_core.logging import BaseEventLogger, BarOutcome, LogEventType, RunnerIdentity
from okmich_quant_core.multi_trader import MultiTrader
from okmich_quant_core.signal import BaseSignal
from okmich_quant_core.trader import Trader


class _RecordingLogger(BaseEventLogger):
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class _Strat(BaseStrategy):
    def __init__(self, name="s", symbol="X", magic=1):
        self.rec = _RecordingLogger()
        super().__init__(StrategyConfig(name=name, symbol=symbol, timeframe=1, magic=magic),
                         BaseSignal(), inference_logger=self.rec)
        self.bind_runner_identity(RunnerIdentity(runner_id="r", runner_start_token="t",
                                                 broker="b", account_id="a", broker_session_id=None))
        self.should_fail = False
        self.calls = 0

    def is_new_bar(self, run_dt):
        return True

    def on_new_bar(self):
        self.calls += 1
        if self.should_fail:
            raise ValueError("boom")


_seq = [0]


def _dt() -> datetime:
    _seq[0] += 1
    return datetime(2026, 6, 1) + timedelta(minutes=_seq[0])


def _events(rec, ev):
    return [r for r in rec.records if r.envelope.event is ev]


# ---- Trader ----

def test_trader_breaker_emits_circuit_breaker_tripped():
    s = _Strat()
    s.should_fail = True
    t = Trader(s, max_consecutive_errors=2)
    t.run(_dt())
    t.run(_dt())
    cbt = _events(s.rec, LogEventType.CIRCUIT_BREAKER_TRIPPED)
    assert len(cbt) == 1
    assert cbt[0].consecutive_errors == 2 and "boom" in cbt[0].last_error


def test_trader_skips_disabled_with_skipped_disabled_bar():
    s = _Strat()
    s.should_fail = True
    t = Trader(s, max_consecutive_errors=1)
    t.run(_dt())  # trips
    runs_before = s.calls
    t.run(_dt())  # disabled → skipped
    assert s.calls == runs_before  # on_new_bar not executed
    assert any(b.outcome is BarOutcome.SKIPPED_DISABLED for b in _events(s.rec, LogEventType.BAR))


def test_trader_reenable_emits_strategy_reenabled():
    s = _Strat()
    s.should_fail = True
    t = Trader(s, max_consecutive_errors=1)
    t.run(_dt())  # trips
    t.enable()
    assert len(_events(s.rec, LogEventType.STRATEGY_REENABLED)) == 1


# ---- MultiTrader ----

def test_multitrader_breaker_skip_and_reenable():
    s = _Strat(name="m")
    s.should_fail = True
    mt = MultiTrader([s], max_consecutive_errors=1)
    mt.run(_dt())  # trips → circuit_breaker_tripped
    mt.run(_dt())  # disabled → skipped_disabled
    assert len(_events(s.rec, LogEventType.CIRCUIT_BREAKER_TRIPPED)) == 1
    assert any(b.outcome is BarOutcome.SKIPPED_DISABLED for b in _events(s.rec, LogEventType.BAR))
    mt.enable_strategy("m")
    assert len(_events(s.rec, LogEventType.STRATEGY_REENABLED)) == 1


def test_multitrader_error_isolation_emits_per_strategy():
    bad = _Strat(name="bad")
    good = _Strat(name="good")
    bad.should_fail = True
    mt = MultiTrader([bad, good], max_consecutive_errors=5)
    mt.run(_dt())
    # bad: one error bar; good: one ok bar — each on its own logger
    assert _events(bad.rec, LogEventType.BAR)[-1].outcome is BarOutcome.ERROR
    assert _events(good.rec, LogEventType.BAR)[-1].outcome is BarOutcome.OK
