"""Inference-logging parity with ``ib/tests/test_inference_logging.py`` (LOGGING_CONTRACT §5/§7.2/§7.3/§7.4).

Drives the sealed async ``_on_bar_close`` seam directly with a recording logger: per-bar heartbeat (ok / error),
stale bars (crypto's analogue of IB's partial bars), the circuit breaker, sealing, and the proven disconnect.
"""
from datetime import datetime, timezone

import pytest

from okmich_quant_core import BarOutcome, LogEventType
from okmich_quant_crypto import BaseCryptoStrategy, CryptoBrokerSession, VenueEnvironment
from okmich_quant_crypto.models import ClosedBar

from .conftest import make_strategy

BAR_TS = int(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)


def _bar(minute=0):
    return ClosedBar(open_ms=BAR_TS + minute * 60_000, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)


async def test_successful_bar_emits_ok_heartbeat():
    s, rec = make_strategy()
    await s._on_bar_close(_bar(0), True)
    assert len(rec.records) == 1
    assert rec.records[0].envelope.event is LogEventType.BAR
    assert rec.records[0].outcome is BarOutcome.OK
    assert rec.records[0].asof_bar_ts == "2026-06-01T12:00:00+00:00"  # framework bar label = candle open
    assert len(s.cycles) == 1


async def test_failed_bar_emits_error_heartbeat_without_reraise():
    s, rec = make_strategy()
    s.fail = True
    await s._on_bar_close(_bar(5), True)  # must NOT raise (no MultiTrader above the seam)
    assert rec.records[0].outcome is BarOutcome.ERROR
    assert s.health.consecutive_errors == 1


async def test_stale_bar_is_buffered_but_emits_nothing():
    s, rec = make_strategy()
    await s._on_bar_close(_bar(0), False)
    assert rec.records == [] and s.cycles == []
    assert len(s.fetch_price_bars()) == 1


async def test_breaker_trips_and_emits_circuit_breaker_then_skipped():
    s, rec = make_strategy(max_consecutive_errors=2)
    s.fail = True
    await s._on_bar_close(_bar(0), True)
    await s._on_bar_close(_bar(5), True)
    assert s.health.is_enabled is False
    events = [r.envelope.event for r in rec.records]
    assert events == [LogEventType.BAR, LogEventType.BAR, LogEventType.CIRCUIT_BREAKER_TRIPPED]
    assert rec.records[-1].consecutive_errors == 2 and "crypto boom" in rec.records[-1].last_error
    cycles = len(s.cycles)
    await s._on_bar_close(_bar(10), True)
    assert rec.records[-1].outcome is BarOutcome.SKIPPED_DISABLED
    assert len(s.cycles) == cycles


async def test_reenable_emits_strategy_reenabled():
    s, rec = make_strategy(max_consecutive_errors=1)
    s.fail = True
    await s._on_bar_close(_bar(0), True)
    assert s.health.is_enabled is False
    s.reenable()
    assert s.health.is_enabled is True
    assert rec.records[-1].envelope.event is LogEventType.STRATEGY_REENABLED


def test_sealing_rejects_on_bar_close_override():
    with pytest.raises(TypeError, match="_on_bar_close"):
        class Bad(BaseCryptoStrategy):
            async def _on_bar_close(self, bar, live):
                pass

            async def on_new_bar(self):
                pass


def test_sealing_rejects_core_seam_overrides():
    with pytest.raises(TypeError, match="sealed BaseStrategy.run"):
        class Bad(BaseCryptoStrategy):
            def run(self, run_dt):
                pass

            async def on_new_bar(self):
                pass


def test_sealing_allows_on_new_bar_override():
    class Good(BaseCryptoStrategy):
        async def on_new_bar(self):
            pass

    assert issubclass(Good, BaseCryptoStrategy)


def test_requires_crypto_config():
    from okmich_quant_core import BaseSignal, StrategyConfig

    class Good(BaseCryptoStrategy):
        async def on_new_bar(self):
            pass

    with pytest.raises(TypeError, match="CryptoStrategyConfig"):
        Good(StrategyConfig(name="x", symbol="X", timeframe="5m", magic=1), BaseSignal())


# ---------------------------------------------------------------------- §7.4 proven, idempotent disconnect

class _FakeCCXT:
    def __init__(self, leaks=False):
        self.session = object()
        self.clients = {"ws": object()}
        self.tcp_connector = None
        self.close_calls = 0
        self.leaks = leaks

    async def close(self):
        self.close_calls += 1
        self.session = None
        if not self.leaks:
            self.clients = {}


def test_session_identity():
    sess = CryptoBrokerSession(_FakeCCXT(), "bybit", VenueEnvironment.DEMO, "key-abc", session_id="s-1")
    assert sess.broker == "bybit:demo"
    assert sess.account_id == "key-abc"
    assert sess.broker_session_id == "s-1"
    assert CryptoBrokerSession(_FakeCCXT(), "bybit", VenueEnvironment.LIVE, "1").broker_session_id.startswith("ccxt-")


async def test_aclose_is_proven_and_idempotent():
    ex = _FakeCCXT()
    sess = CryptoBrokerSession(ex, "bybit", VenueEnvironment.DEMO, "1")
    assert await sess.aclose() is True
    assert await sess.aclose() is True
    assert sess.disconnect() is True          # cached proof, no second close
    assert ex.close_calls == 1


async def test_unproven_close_reports_false():
    sess = CryptoBrokerSession(_FakeCCXT(leaks=True), "bybit", VenueEnvironment.DEMO, "1")
    assert await sess.aclose() is False


async def test_disconnect_inside_a_running_loop_before_aclose_is_not_proven():
    ex = _FakeCCXT()
    sess = CryptoBrokerSession(ex, "bybit", VenueEnvironment.DEMO, "1")
    assert sess.disconnect() is False
    assert ex.close_calls == 0


def test_disconnect_without_a_loop_runs_the_close():
    ex = _FakeCCXT()
    sess = CryptoBrokerSession(ex, "bybit", VenueEnvironment.DEMO, "1")
    assert sess.disconnect() is True and ex.close_calls == 1
    assert sess.disconnect() is True and ex.close_calls == 1
