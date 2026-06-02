"""Phase 3 tests — IB near-par inference logging (LOGGING_CONTRACT §5/§7.2/§7.3/§7.4).

Drives the sealed async ``_on_bar_close`` seam directly with a recording logger, so no live IB
connection is needed. Covers: per-bar heartbeat (ok/error), partial-skip, the IB circuit breaker
(trip → circuit_breaker_tripped → skipped_disabled → reenable), sealing, and the proven disconnect.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from okmich_quant_core import BarOutcome, LogEventType, RunnerIdentity
from okmich_quant_core.logging import BaseEventLogger
from okmich_quant_core.signal import BaseSignal
from okmich_quant_ib.broker_session import IBBrokerSession
from okmich_quant_ib.contract import IBContractConfig, SecType
from okmich_quant_ib.strategy import BaseIBStrategy


class _RecordingLogger(BaseEventLogger):
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


def _config(max_pos=1):
    from okmich_quant_core import PositionSizingConfig, PositionSizingType
    return SimpleNamespace(
        name="ib_sys", symbol="AAPL", timeframe="5 mins", magic=42, signal_params={},
        position_sizing=PositionSizingConfig(type=PositionSizingType.FIXED, units=1.0),
        max_number_of_open_positions=max_pos, bars_to_copy=10, position_manager=None, filters=[],
    )


def _make_strategy(max_consecutive_errors=5):
    rec = _RecordingLogger()

    class _Concrete(BaseIBStrategy):
        def __init__(self):
            super().__init__(_config(), BaseSignal(),
                             IBContractConfig(sec_type=SecType.STK, exchange="SMART", currency="USD"),
                             max_consecutive_errors=max_consecutive_errors, inference_logger=rec)
            self.fail = False
            self.calls = 0

        def _append_bar(self, completed_bar):  # no PriceBuffer interaction in unit tests
            pass

        async def on_new_bar(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError("ib boom")

    s = _Concrete()
    s.bind_runner_identity(RunnerIdentity(runner_id="ib-1", runner_start_token="tok",
                                          broker="IB", account_id="DU1", broker_session_id="c1"))
    return s, rec


def _bar(minute=0, partial=False):
    return {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            "time": datetime(2026, 6, 1, 12, minute, tzinfo=timezone.utc),
            "sample_count": 60, "partial": partial}


# --------------------------------------------------------------------------------------
# heartbeat
# --------------------------------------------------------------------------------------

async def test_successful_bar_emits_ok_heartbeat():
    s, rec = _make_strategy()
    await s._on_bar_close(_bar(0))
    assert len(rec.records) == 1
    assert rec.records[0].envelope.event is LogEventType.BAR
    assert rec.records[0].outcome is BarOutcome.OK
    assert rec.records[0].asof_bar_ts == "2026-06-01T12:00:00+00:00"  # framework bar boundary
    assert s.calls == 1


async def test_failed_bar_emits_error_heartbeat_without_reraise():
    s, rec = _make_strategy()
    s.fail = True
    await s._on_bar_close(_bar(5))  # must NOT raise (no MultiTrader above the IB seam)
    assert rec.records[0].outcome is BarOutcome.ERROR
    assert s.health.consecutive_errors == 1


async def test_partial_bar_emits_nothing():
    s, rec = _make_strategy()
    await s._on_bar_close(_bar(0, partial=True))
    assert rec.records == []
    assert s.calls == 0


# --------------------------------------------------------------------------------------
# IB circuit breaker → near-par with MT5 (§7.3)
# --------------------------------------------------------------------------------------

async def test_breaker_trips_and_emits_circuit_breaker_then_skipped():
    s, rec = _make_strategy(max_consecutive_errors=2)
    s.fail = True
    await s._on_bar_close(_bar(0))   # error 1
    await s._on_bar_close(_bar(5))   # error 2 -> trips
    assert s.health.is_enabled is False
    events = [r.envelope.event for r in rec.records]
    assert events == [LogEventType.BAR, LogEventType.BAR, LogEventType.CIRCUIT_BREAKER_TRIPPED]
    cbt = rec.records[-1]
    assert cbt.consecutive_errors == 2 and "ib boom" in cbt.last_error

    # subsequent bar while disabled -> skipped_disabled, on_new_bar NOT run
    calls_before = s.calls
    await s._on_bar_close(_bar(10))
    assert rec.records[-1].outcome is BarOutcome.SKIPPED_DISABLED
    assert s.calls == calls_before


async def test_reenable_emits_strategy_reenabled():
    s, rec = _make_strategy(max_consecutive_errors=1)
    s.fail = True
    await s._on_bar_close(_bar(0))  # trips immediately
    assert s.health.is_enabled is False
    s.reenable()
    assert s.health.is_enabled is True
    assert rec.records[-1].envelope.event is LogEventType.STRATEGY_REENABLED


# --------------------------------------------------------------------------------------
# §5.1 sealing of the IB seam
# --------------------------------------------------------------------------------------

def test_sealing_rejects_on_bar_close_override():
    with pytest.raises(TypeError, match="_on_bar_close"):
        class Bad(BaseIBStrategy):
            async def _on_bar_close(self, completed_bar):
                pass

            async def on_new_bar(self):
                pass


def test_sealing_allows_on_new_bar_override():
    class Good(BaseIBStrategy):
        async def on_new_bar(self):
            pass

    assert issubclass(Good, BaseIBStrategy)


# --------------------------------------------------------------------------------------
# §7.4 proven, idempotent disconnect
# --------------------------------------------------------------------------------------

class _FakeIB:
    def __init__(self):
        self._connected = True
        self.disconnect_calls = 0

    def managedAccounts(self):
        return ["DU123"]

    def isConnected(self):
        return self._connected

    def disconnect(self):
        self.disconnect_calls += 1
        self._connected = False


def test_ib_broker_session_identity_and_proven_idempotent_disconnect():
    ib = _FakeIB()
    sess = IBBrokerSession(ib, "127.0.0.1", 4002, 7)
    assert sess.broker == "IB"
    assert sess.account_id == "DU123"
    assert sess.broker_session_id == "clientId=7@127.0.0.1:4002"

    assert sess.disconnect() is True          # proven down via isConnected()
    assert ib.disconnect_calls == 1
    assert sess.disconnect() is True          # idempotent: cached, no second disconnect
    assert ib.disconnect_calls == 1
