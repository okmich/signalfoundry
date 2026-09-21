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


def _make_notifier(strategy_name="TestStrategy", broker=""):
    """Return a TelegramNotifier with the HTTP client patched out."""
    with patch("okmich_quant_core.notification.telegram.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200, raise_for_status=lambda: None)
        notifier = TelegramNotifier(
            bot_token="fake_token",
            chat_id="12345",
            strategy_name=strategy_name,
            broker=broker,
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

    def test_on_trade_failed(self):
        notifier, sent = _make_notifier()
        notifier.on_trade_failed(
            symbol="BTCUSD", direction="buy", reason="AutoTrading disabled by client",
            retcode=10089, context={"strategy_name": "MyStrat"},
        )
        notifier._dispatcher.flush()
        assert len(sent) == 1
        assert "TRADE FAILED" in sent[0]
        assert "BTCUSD" in sent[0]
        assert "buy" in sent[0]
        assert "MyStrat" in sent[0]
        assert "AutoTrading disabled by client" in sent[0]
        assert "10089" in sent[0]

    def test_on_trade_failed_no_retcode(self):
        notifier, sent = _make_notifier()
        notifier.on_trade_failed(
            symbol="EURUSD", direction="CLOSE", reason="ticket=42 not found",
            context={"strategy_name": "MyStrat"},
        )
        notifier._dispatcher.flush()
        assert "TRADE FAILED" in sent[0]
        assert "CLOSE" in sent[0]
        assert "ticket=42 not found" in sent[0]
        assert "retcode" not in sent[0]

    def test_on_trade_failed_html_escaping(self):
        """HTML-special chars in reason must be escaped or Telegram drops the message."""
        notifier, sent = _make_notifier()
        notifier.on_trade_failed(
            symbol="BTCUSD", direction="buy",
            reason="'<NoneType>' object has no attribute 'retcode' & state=<bad>",
            context={"strategy_name": "MyStrat"},
        )
        notifier._dispatcher.flush()
        # Raw HTML metacharacters from the reason must not appear unescaped
        assert "<NoneType>" not in sent[0]
        assert "<bad>" not in sent[0]
        assert "&lt;NoneType&gt;" in sent[0]
        assert "&amp;" in sent[0]


# Every TelegramNotifier message type, as (expected bold label, call).
_ALL_MESSAGES = [
    ("🔵 OPENED", lambda n: n.on_trade_opened("EURUSD", "buy", 0.1, 1.1, 1.0, 1.2, 1001, 555)),
    ("🟢 CLOSED", lambda n: n.on_trade_closed("EURUSD", 555, 12.5)),
    ("🔴 CLOSED", lambda n: n.on_trade_closed("EURUSD", 555, -12.5)),
    ("✏️ MODIFIED", lambda n: n.on_trade_modified("EURUSD", 555, 1.0, 1.2)),
    ("🛑 TRADE FAILED", lambda n: n.on_trade_failed("EURUSD", "buy", "rejected")),
    ("⚠️ ERROR", lambda n: n.on_error("", "boom")),
    ("🚫 CIRCUIT BREAKER", lambda n: n.on_circuit_breaker_tripped("", 5)),
    ("📡 CONNECTION LOST", lambda n: n.on_connection_lost("")),
    ("✅ CONNECTION RESTORED", lambda n: n.on_connection_restored("")),
]


class TestTelegramNotifierSystemTag:
    """The notifier's strategy_name (the sending system, e.g. SystemConfig.name) heads every message."""

    @pytest.mark.parametrize("label, send", _ALL_MESSAGES, ids=[label for label, _ in _ALL_MESSAGES])
    def test_system_tag_follows_label_on_every_message(self, label, send):
        notifier, sent = _make_notifier(strategy_name="trend-hmm-live")
        send(notifier)
        notifier._dispatcher.flush()
        assert sent[0].startswith(f"<b>{label}</b> [trend-hmm-live]")

    @pytest.mark.parametrize("label, send", _ALL_MESSAGES, ids=[label for label, _ in _ALL_MESSAGES])
    def test_no_system_name_means_no_empty_tag(self, label, send):
        notifier, sent = _make_notifier(strategy_name="")
        send(notifier)
        notifier._dispatcher.flush()
        assert "[]" not in sent[0]
        assert not sent[0].startswith(f"<b>{label}</b> [")

    def test_trade_message_tags_system_then_broker(self):
        notifier, sent = _make_notifier(strategy_name="trend-hmm-live", broker="MT5")
        notifier.on_trade_opened("EURUSD", "buy", 0.1, 1.0842, 1.08, 1.09, 1001, 123)
        notifier._dispatcher.flush()
        assert sent[0].startswith("<b>🔵 OPENED</b> [trend-hmm-live] [MT5] EURUSD buy 0.1L @ 1.0842\n")

    def test_trade_failed_tags_system_then_strategy_then_broker(self):
        notifier, sent = _make_notifier(strategy_name="trend-hmm-live", broker="MT5")
        notifier.on_trade_failed("BTCUSD", "buy", "rejected", context={"strategy_name": "MyStrat"})
        notifier._dispatcher.flush()
        assert sent[0].startswith("<b>🛑 TRADE FAILED</b> [trend-hmm-live] [MyStrat] [MT5] BTCUSD buy\n")

    def test_per_call_name_follows_system_name(self):
        notifier, sent = _make_notifier(strategy_name="trend-hmm-live")
        notifier.on_error("S1", "boom")
        notifier._dispatcher.flush()
        assert sent[0] == "<b>⚠️ ERROR</b> [trend-hmm-live] [S1]\nboom"

    def test_per_call_name_equal_to_system_name_shown_once(self):
        notifier, sent = _make_notifier(strategy_name="trend-hmm-live")
        notifier.on_circuit_breaker_tripped("trend-hmm-live", 5)
        notifier._dispatcher.flush()
        assert sent[0] == "<b>🚫 CIRCUIT BREAKER</b> [trend-hmm-live] tripped after 5 errors"

    def test_per_call_name_kept_when_no_system_name(self):
        notifier, sent = _make_notifier(strategy_name="")
        notifier.on_connection_lost("S1")
        notifier._dispatcher.flush()
        assert sent[0] == "<b>📡 CONNECTION LOST</b> [S1]"

    def test_system_name_is_html_escaped(self):
        notifier, sent = _make_notifier(strategy_name="S&P <live>")
        notifier.on_trade_closed("US500", 7, 1.0)
        notifier._dispatcher.flush()
        assert "[S&amp;P &lt;live&gt;]" in sent[0]
        assert "<live>" not in sent[0]


class TestTelegramNotifierErrorEscaping:
    def test_on_error_escapes_exception_text(self):
        """str(exception) routinely contains '<'; unescaped, Telegram rejects the message and the alert is lost."""
        notifier, sent = _make_notifier()
        notifier.on_error("S1", "'<' not supported between instances of 'NoneType' and 'float' & <more>")
        notifier._dispatcher.flush()
        body = sent[0].split("\n", 1)[1]
        assert "<" not in body and ">" not in body
        assert "&lt;&#x27; not supported" in body
        assert "&amp; &lt;more&gt;" in body

    def test_on_error_escapes_strategy_name(self):
        notifier, sent = _make_notifier(strategy_name="")
        notifier.on_error("<S1>", "boom")
        notifier._dispatcher.flush()
        assert sent[0] == "<b>⚠️ ERROR</b> [&lt;S1&gt;]\nboom"


class TestBaseNotifierDefaults:
    """on_trade_failed has a default impl that delegates to on_error so
    third-party notifiers that don't override it still produce some signal."""

    def test_default_on_trade_failed_calls_on_error(self):
        calls = []

        class CustomNotifier(BaseNotifier):
            def on_trade_opened(self, *a, **k): pass
            def on_trade_closed(self, *a, **k): pass
            def on_trade_modified(self, *a, **k): pass
            def on_error(self, strategy_name, error_message, context=None):
                calls.append((strategy_name, error_message, context))
            def on_circuit_breaker_tripped(self, *a, **k): pass
            def on_connection_lost(self, *a, **k): pass
            def on_connection_restored(self, *a, **k): pass
            def close(self): pass

        n = CustomNotifier()
        n.on_trade_failed(
            symbol="BTCUSD", direction="buy", reason="Trade disabled",
            retcode=10017, context={"strategy_name": "MyStrat"},
        )
        assert len(calls) == 1
        strategy_name, error_message, context = calls[0]
        assert strategy_name == "MyStrat"
        assert "BTCUSD" in error_message
        assert "buy" in error_message
        assert "Trade disabled" in error_message
        assert "10017" in error_message

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
        mock_config.symbol = "EURUSD"
        mock_config.timeframe = 5
        mock_config.magic = 1
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
        mock_config.symbol = "EURUSD"
        mock_config.timeframe = 5
        mock_config.magic = 1
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
        cfg.symbol = "EURUSD"
        cfg.timeframe = 5
        cfg.magic = 1
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
