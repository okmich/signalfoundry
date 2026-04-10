"""
Tests for the notification subpackage: _AsyncDispatcher, BaseNotifier, TelegramNotifier.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from okmich_quant_core.notification.base import BaseNotifier, _AsyncDispatcher
from okmich_quant_core.notification.telegram import TelegramNotifier


# ---------------------------------------------------------------------------
# _AsyncDispatcher
# ---------------------------------------------------------------------------


class TestAsyncDispatcher:
    def test_messages_are_delivered(self):
        delivered = []
        dispatcher = _AsyncDispatcher(lambda msg: delivered.append(msg))
        dispatcher.dispatch("hello")
        dispatcher.dispatch("world")
        dispatcher.flush()
        assert delivered == ["hello", "world"]

    def test_dispatch_is_non_blocking(self):
        """dispatch() must return before the slow send_fn completes."""
        started = []

        def slow_send(msg):
            time.sleep(0.1)
            started.append(msg)

        dispatcher = _AsyncDispatcher(slow_send)
        t0 = time.monotonic()
        dispatcher.dispatch("msg")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, "dispatch() blocked the caller"
        dispatcher.flush()
        assert started == ["msg"]

    def test_send_fn_exception_does_not_crash_worker(self):
        calls = []

        def failing_then_ok(msg):
            if msg == "bad":
                raise RuntimeError("oops")
            calls.append(msg)

        dispatcher = _AsyncDispatcher(failing_then_ok)
        dispatcher.dispatch("bad")
        dispatcher.dispatch("good")
        dispatcher.flush()
        assert calls == ["good"]

    def test_close_stops_worker(self):
        dispatcher = _AsyncDispatcher(lambda msg: None)
        dispatcher.close()
        assert not dispatcher._thread.is_alive()

    def test_close_flushes_before_stopping(self):
        delivered = []
        dispatcher = _AsyncDispatcher(lambda msg: delivered.append(msg))
        dispatcher.dispatch("a")
        dispatcher.dispatch("b")
        dispatcher.close()
        assert delivered == ["a", "b"]


# ---------------------------------------------------------------------------
# TelegramNotifier (send_fn mocked — no real HTTP calls)
# ---------------------------------------------------------------------------


def _make_notifier(strategy_name="TestStrategy"):
    """Return a TelegramNotifier with the HTTP client patched out."""
    with patch("okmich_quant_core.notification.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        notifier = TelegramNotifier(
            bot_token="fake_token",
            chat_id="12345",
            strategy_name=strategy_name,
        )
        # Replace the dispatcher's send_fn with a spy after construction
        sent = []
        notifier._dispatcher._send_fn = lambda msg: sent.append(msg)
        return notifier, sent


class TestTelegramNotifier:
    def test_on_trade_opened(self):
        notifier, sent = _make_notifier()
        notifier.on_trade_opened("EURUSD", "BUY", 0.1, 1.1234, 1.1200, 1.1300, 1001, 555)
        notifier._dispatcher.flush()
        assert len(sent) == 1
        assert "OPENED" in sent[0]
        assert "EURUSD" in sent[0]
        assert "BUY" in sent[0]

    def test_on_trade_closed(self):
        notifier, sent = _make_notifier()
        notifier.on_trade_closed("GBPUSD", 42, 12.50)
        notifier._dispatcher.flush()
        assert "CLOSED" in sent[0]
        assert "42" in sent[0]

    def test_on_trade_modified(self):
        notifier, sent = _make_notifier()
        notifier.on_trade_modified("USDJPY", 99, 140.0, 145.0)
        notifier._dispatcher.flush()
        assert "MODIFIED" in sent[0]
        assert "99" in sent[0]

    def test_on_error(self):
        notifier, sent = _make_notifier()
        notifier.on_error("MyStrat", "something broke")
        notifier._dispatcher.flush()
        assert "ERROR" in sent[0]
        assert "MyStrat" in sent[0]
        assert "something broke" in sent[0]

    def test_on_circuit_breaker_tripped(self):
        notifier, sent = _make_notifier()
        notifier.on_circuit_breaker_tripped("MyStrat", 5)
        notifier._dispatcher.flush()
        assert "CIRCUIT BREAKER" in sent[0]
        assert "5" in sent[0]

    def test_on_connection_lost(self):
        notifier, sent = _make_notifier()
        notifier.on_connection_lost("MyStrat")
        notifier._dispatcher.flush()
        assert "CONNECTION LOST" in sent[0]

    def test_on_connection_restored(self):
        notifier, sent = _make_notifier()
        notifier.on_connection_restored("MyStrat")
        notifier._dispatcher.flush()
        assert "CONNECTION RESTORED" in sent[0]

    def test_close_stops_worker(self):
        notifier, _ = _make_notifier()
        notifier.close()
        assert not notifier._dispatcher._thread.is_alive()

    def test_multiple_messages_ordered(self):
        notifier, sent = _make_notifier()
        notifier.on_error("S", "err1")
        notifier.on_error("S", "err2")
        notifier.on_error("S", "err3")
        notifier._dispatcher.flush()
        assert [m for m in sent if "err1" in m or "err2" in m or "err3" in m] == sent
        assert "err1" in sent[0]
        assert "err2" in sent[1]
        assert "err3" in sent[2]


# ---------------------------------------------------------------------------
# BaseStrategy integration — notifier wired in
# ---------------------------------------------------------------------------


class TestBaseStrategyNotifierIntegration:
    def test_cleanup_closes_notifier(self):
        from datetime import datetime
        from unittest.mock import Mock

        from okmich_quant_core.base_strategy import BaseStrategy

        class ConcreteStrategy(BaseStrategy):
            def is_new_bar(self, run_dt):
                return True

            def on_new_bar(self):
                pass

        mock_config = Mock()
        mock_config.name = "Test"
        mock_signal = Mock()
        mock_notifier = Mock(spec=BaseNotifier)

        strategy = ConcreteStrategy(mock_config, mock_signal, notifier=mock_notifier)
        strategy.cleanup()

        mock_notifier.close.assert_called_once()

    def test_no_notifier_cleanup_is_safe(self):
        from unittest.mock import Mock

        from okmich_quant_core.base_strategy import BaseStrategy

        class ConcreteStrategy(BaseStrategy):
            def is_new_bar(self, run_dt):
                return True

            def on_new_bar(self):
                pass

        mock_config = Mock()
        mock_config.name = "Test"
        strategy = ConcreteStrategy(mock_config, Mock())
        strategy.cleanup()  # must not raise


# ---------------------------------------------------------------------------
# MultiTrader notifier integration
# ---------------------------------------------------------------------------


class TestMultiTraderNotifierIntegration:
    def _make_strategy(self, name, should_fail=False, consecutive_errors=1):
        from unittest.mock import Mock

        from okmich_quant_core.base_strategy import BaseStrategy

        class S(BaseStrategy):
            def is_new_bar(self, run_dt):
                return True

            def on_new_bar(self):
                if should_fail:
                    raise RuntimeError("boom")

        cfg = Mock()
        cfg.name = name
        notifier = Mock(spec=BaseNotifier)
        s = S(cfg, Mock(), notifier=notifier)
        s.should_fail = should_fail
        return s

    def test_on_error_called_when_strategy_raises(self):
        from datetime import datetime

        from okmich_quant_core.multi_trader import MultiTrader

        strat = self._make_strategy("S1", should_fail=True)
        mt = MultiTrader([strat], max_consecutive_errors=5)
        mt.run(datetime.now())

        strat.notifier.on_error.assert_called_once()
        args = strat.notifier.on_error.call_args[0]
        assert args[0] == "S1"
        assert "boom" in args[1]

    def test_on_circuit_breaker_called_when_disabled(self):
        from datetime import datetime

        from okmich_quant_core.multi_trader import MultiTrader

        strat = self._make_strategy("S1", should_fail=True)
        mt = MultiTrader([strat], max_consecutive_errors=1)
        mt.run(datetime.now())

        strat.notifier.on_circuit_breaker_tripped.assert_called_once()
        args = strat.notifier.on_circuit_breaker_tripped.call_args[0]
        assert args[0] == "S1"

    def test_notifier_not_called_on_success(self):
        from datetime import datetime

        from okmich_quant_core.multi_trader import MultiTrader

        strat = self._make_strategy("S1", should_fail=False)
        mt = MultiTrader([strat])
        mt.run(datetime.now())

        strat.notifier.on_error.assert_not_called()
        strat.notifier.on_circuit_breaker_tripped.assert_not_called()