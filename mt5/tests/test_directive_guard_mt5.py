"""The account-directive guard on MT5: the entry choke point in BaseMt5Strategy's order methods and the forced sweep's
MT5 hooks (ACCOUNT_ADMIN_SPEC §10.1, §14.4)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from okmich_quant_core import StrategyConfig
from okmich_quant_core.account_admin import (AccountDirective, DirectiveAccount, DirectiveFile, directive_path,
                                             write_directive)
from okmich_quant_core.directive_guard import PROCESS_READER
from okmich_quant_core.logging import GuardedOp, LogEventType, RunnerIdentity

LOGIN, SERVER = 51234567, "ICMarketsSC-Demo"


@pytest.fixture
def mt5_env(tmp_path, monkeypatch):
    live = tmp_path / "live"
    monkeypatch.setenv("OKMICH_QUANT_LIVE_BASE", str(live))
    with patch("okmich_quant_mt5.strategy.fetch_symbol_info") as info, \
         patch("okmich_quant_mt5.strategy.PriceBuffer"), \
         patch("okmich_quant_mt5.strategy.get_position_manager"), \
         patch("okmich_quant_mt5.strategy.get_positions") as get_pos, \
         patch("okmich_quant_mt5.strategy.open_position") as open_pos, \
         patch("okmich_quant_mt5.strategy.place_pending_order") as place_pending, \
         patch("okmich_quant_mt5.strategy.close_position") as close_pos, \
         patch("okmich_quant_mt5.strategy.cancel_pending_order") as cancel, \
         patch("okmich_quant_mt5.strategy.mt5") as mt5:
        info.return_value = {"symbol": "EURUSD", "point": 0.00001, "filling_mode": 1}
        get_pos.return_value = []
        mt5.account_info.return_value = SimpleNamespace(login=LOGIN, server=SERVER)
        mt5.orders_get.return_value = ()
        yield SimpleNamespace(live=live, get_positions=get_pos, open_position=open_pos, place_pending=place_pending,
                              close_position=close_pos, cancel=cancel, mt5=mt5)


def publish(live, directive):
    now = datetime.now(timezone.utc)
    write_directive(directive_path(live, "icmarkets.demo"), DirectiveFile(
        DirectiveAccount("icmarkets", SERVER, LOGIN, "USD"), directive, "test", ("test",), now, now, 240, 1, 1))
    PROCESS_READER.invalidate()


def strategy(signal=None):
    from okmich_quant_mt5.strategy import GenericBasicStrategy
    s = GenericBasicStrategy(StrategyConfig(name="gb", symbol="EURUSD", timeframe=5, magic=12345), signal or Mock())
    s.bind_runner_identity(RunnerIdentity.generate(name="r", broker=SERVER, account_id=str(LOGIN)))
    s.records = []
    s._log_binding.logger.write = s.records.append
    return s


def ops(s):
    return [r for r in s.records if r.envelope.event is LogEventType.ACCOUNT_DIRECTIVE_OP]


def test_a_buy_signal_never_reaches_the_broker_under_no_entry_ops(mt5_env):
    publish(mt5_env.live, AccountDirective.NO_ENTRY_OPS)
    signal = Mock()
    signal.generate.return_value = (np.array([0, 1]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0]))
    s = strategy(signal)
    s.fetch_ohlcv = Mock(return_value=Mock())
    s.fetch_latest_tick_info = Mock(return_value={"ask": 1.1000, "bid": 1.0999})
    s.notifier = Mock()
    s.latest_run_dt = datetime(2026, 9, 24, 13, 0)
    s.on_new_bar()
    mt5_env.open_position.assert_not_called()
    s.notifier.on_trade_opened.assert_not_called()
    assert [o.op for o in ops(s)] == [GuardedOp.OPEN_LONG] and ops(s)[0].signal_bar_utc is not None


@pytest.mark.parametrize("order_type,op", [("buy", GuardedOp.OPEN_LONG), ("sell", GuardedOp.OPEN_SHORT),
                                           ("buy_limit", GuardedOp.PLACE_PENDING), ("sell_stop", GuardedOp.PLACE_PENDING)])
def test_place_order_is_the_choke_point(mt5_env, order_type, op):
    publish(mt5_env.live, AccountDirective.NO_ENTRY_OPS)
    s = strategy()
    assert s.place_order(order_type, price=1.1) is False
    mt5_env.open_position.assert_not_called()
    mt5_env.place_pending.assert_not_called()
    assert [o.op for o in ops(s)] == [op]


def test_all_ops_sends_as_before(mt5_env):
    publish(mt5_env.live, AccountDirective.ALL_OPS)
    s = strategy()
    assert s.place_order("buy_limit", price=1.09) is True and s.open_position("buy", 1.1) is True
    mt5_env.place_pending.assert_called_once()
    mt5_env.open_position.assert_called_once()


def test_own_exit_closes_under_no_ops(mt5_env):
    publish(mt5_env.live, AccountDirective.NO_OPS)
    s = strategy()
    assert s.close_position(555, reason="exit_signal") is True
    mt5_env.close_position.assert_called_once()


def test_sweep_cancels_own_pending_and_closes_own_positions_under_no_ops(mt5_env):
    publish(mt5_env.live, AccountDirective.NO_OPS)
    mt5_env.mt5.orders_get.return_value = (SimpleNamespace(ticket=11, magic=12345, type=2, volume_current=0.1, price_open=1.09),
                                           SimpleNamespace(ticket=12, magic=999, type=2, volume_current=0.1, price_open=1.09))
    mt5_env.get_positions.return_value = [{"ticket": 21, "type": 1, "volume": 0.1, "price_open": 1.1, "profit": -2.5}]
    s = strategy()
    s.enforce_account_directive(datetime(2026, 9, 24, 13, 0))
    mt5_env.cancel.assert_called_once_with(11)                         # another system's order (magic 999) is left alone
    mt5_env.close_position.assert_called_once()
    assert mt5_env.close_position.call_args[0][0] == 21
    assert [o.op for o in ops(s)] == [GuardedOp.CANCEL_PENDING, GuardedOp.CLOSE_SHORT]
    assert s._open_trades["21"]["close_intent"] == "account_directive:NO_OPS"   # the reconciler attributes it right


def test_a_failed_forced_close_does_not_raise_or_spam_trade_failed(mt5_env):
    publish(mt5_env.live, AccountDirective.NO_OPS)
    mt5_env.get_positions.return_value = [{"ticket": 21, "type": 0, "volume": 0.1, "price_open": 1.1, "profit": 0.0}]
    mt5_env.close_position.side_effect = RuntimeError("market closed")
    s = strategy()
    s.notifier = Mock()
    for _ in range(3):
        s.enforce_account_directive(datetime(2026, 9, 24, 13, 0))
    s.notifier.on_trade_failed.assert_not_called()
    assert s.notifier.on_account_event.call_count == 1                  # the failure, alerted once this episode
