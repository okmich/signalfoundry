"""MT5 half of closed-position reconciliation: deal-history resolution and the sweep debounce.

The debounce test exists because the original ``<=`` made a runner polling at exactly the debounce interval
silently do NO position management at all — and the symptom is indistinguishable from a quiet market.
"""
from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from okmich_quant_core.closed_trade import CloseReason
from okmich_quant_mt5.functions.mt5 import DataFetchError, fetch_closed_deals, select_history_window

_Deal = namedtuple("_Deal", "ticket position entry reason volume price profit commission swap time time_msc")


def _deal(entry=1, reason=5, volume=0.1, price=1.2, profit=10.0, commission=-0.5, swap=-0.1, t=1_700_000_000,
          ticket=1, position=99):
    return _Deal(ticket, position, entry, reason, volume, price, profit, commission, swap, t, t * 1000)


class _FakeMt5:
    """Only the surface fetch_closed_deals touches."""

    DEAL_ENTRY_OUT = 1

    def __init__(self, deals):
        self._deals = deals
        self.selected = None

    def history_deals_get(self, position=None):
        return self._deals

    def history_select(self, start, end):
        self.selected = (start, end)
        return True

    @staticmethod
    def last_error():
        return (1, "boom")


# --------------------------------------------------------------------------------------
# fetch_closed_deals
# --------------------------------------------------------------------------------------

def test_query_failure_raises_rather_than_reporting_no_deals():
    """None means 'could not ask'. Returning [] here would report a phantom close with zero P/L."""
    fake = _FakeMt5(None)
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        with pytest.raises(DataFetchError):
            fetch_closed_deals(99)


def test_genuinely_no_closing_deal_yet_returns_empty():
    fake = _FakeMt5(())
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        assert fetch_closed_deals(99) == []


def test_entry_deal_is_excluded_so_volume_is_not_double_counted():
    fake = _FakeMt5((_deal(entry=0, ticket=1), _deal(entry=1, ticket=2)))
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        deals = fetch_closed_deals(99)
    assert [d["ticket"] for d in deals] == [2]


def test_deal_reason_is_named():
    fake = _FakeMt5((_deal(reason=4), ))
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        assert fetch_closed_deals(99)[0]["reason_name"] == "stop_loss"


def test_unknown_reason_code_degrades_to_unknown():
    fake = _FakeMt5((_deal(reason=99), ))
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        assert fetch_closed_deals(99)[0]["reason_name"] == "unknown"


def test_deals_are_ordered_oldest_first():
    fake = _FakeMt5((_deal(t=200, ticket=2), _deal(t=100, ticket=1)))
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        assert [d["ticket"] for d in fetch_closed_deals(99)] == [1, 2]


# --------------------------------------------------------------------------------------
# select_history_window — a convenience, never a precondition
# --------------------------------------------------------------------------------------

def test_history_preload_is_skipped_when_the_build_lacks_it():
    """An older or stubbed MetaTrader5 must not stop a strategy from constructing."""
    with patch("okmich_quant_mt5.functions.mt5.mt5", SimpleNamespace()):
        assert select_history_window() is False


def test_history_preload_swallows_a_raise():
    class _Boom(SimpleNamespace):
        @staticmethod
        def history_select(*_a):
            raise RuntimeError("terminal busy")

    with patch("okmich_quant_mt5.functions.mt5.mt5", _Boom()):
        assert select_history_window() is False


def test_history_preload_covers_the_requested_window():
    fake = _FakeMt5(())
    with patch("okmich_quant_mt5.functions.mt5.mt5", fake):
        assert select_history_window(days=3) is True
    start, end = fake.selected
    assert timedelta(days=2, hours=23) < (datetime.now(timezone.utc) - start) < timedelta(days=3, hours=1)
    assert end > datetime.now(timezone.utc)


# --------------------------------------------------------------------------------------
# resolve_closed_trade — realised, not last-seen
# --------------------------------------------------------------------------------------

class _Resolver:
    """BaseMt5Strategy.resolve_closed_trade lifted off the class, so the test needs no live terminal."""

    def __init__(self, symbol="EURUSD", magic=7):
        self.strategy_config = SimpleNamespace(symbol=symbol, magic=magic)

    resolve_closed_trade = None      # bound below


def _make_resolver():
    from okmich_quant_mt5.strategy import BaseMt5Strategy
    r = _Resolver()
    r.resolve_closed_trade = BaseMt5Strategy.resolve_closed_trade.__get__(r, _Resolver)
    return r


def test_resolution_reports_realised_profit_and_broker_reason():
    r = _make_resolver()
    with patch("okmich_quant_mt5.strategy.fetch_closed_deals",
               return_value=[dict(_deal(reason=5, price=1.25, profit=12.0, commission=-0.4, swap=-0.2)._asdict(),
                                  reason_name="take_profit")]):
        trade = r.resolve_closed_trade("99", {"price_open": 1.2})
    assert trade.reason == CloseReason.TAKE_PROFIT
    assert trade.exit_price == 1.25
    assert trade.entry_price == 1.2
    assert trade.profit == 12.0
    assert trade.net_profit == pytest.approx(11.4)
    assert trade.resolved is True


def test_partial_closes_are_summed_and_volume_weighted():
    """Scaling out must report one honest average, not whichever leg happened to be last."""
    r = _make_resolver()
    legs = [dict(_deal(volume=0.1, price=1.20, profit=5.0, t=100)._asdict(), reason_name="take_profit"),
            dict(_deal(volume=0.3, price=1.30, profit=15.0, t=200)._asdict(), reason_name="take_profit")]
    with patch("okmich_quant_mt5.strategy.fetch_closed_deals", return_value=legs):
        trade = r.resolve_closed_trade("99", {"price_open": 1.1})
    assert trade.volume == pytest.approx(0.4)
    assert trade.exit_price == pytest.approx(1.275)          # (0.1*1.20 + 0.3*1.30) / 0.4
    assert trade.profit == pytest.approx(20.0)


def test_no_closing_deal_yet_is_unresolved_not_a_fabricated_zero():
    r = _make_resolver()
    with patch("okmich_quant_mt5.strategy.fetch_closed_deals", return_value=[]):
        assert r.resolve_closed_trade("99", {"price_open": 1.2}) is None


def test_stop_out_maps_to_its_own_reason():
    r = _make_resolver()
    with patch("okmich_quant_mt5.strategy.fetch_closed_deals",
               return_value=[dict(_deal(reason=6)._asdict(), reason_name="stop_out")]):
        assert r.resolve_closed_trade("99", {}).reason == CloseReason.STOP_OUT


# --------------------------------------------------------------------------------------
# the debounce
# --------------------------------------------------------------------------------------

def test_debounce_is_strict_so_polling_at_the_interval_is_not_swallowed():
    """chk_position_interval == _MIN_POSITION_CHK_SECONDS must still sweep."""
    from okmich_quant_mt5.strategy import BaseMt5Strategy

    stub = SimpleNamespace(prev_position_chk_dt=None, position_manager=None,
                           strategy_config=SimpleNamespace(symbol="EURUSD", magic=7),
                           _MIN_POSITION_CHK_SECONDS=BaseMt5Strategy._MIN_POSITION_CHK_SECONDS)
    manage = BaseMt5Strategy.manage_positions.__get__(stub, type(stub))
    t0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)

    with patch("okmich_quant_mt5.strategy.get_positions", return_value=[{"ticket": 1}]):
        assert manage(t0) == [{"ticket": 1}]
        assert manage(t0 + timedelta(seconds=5)) == [{"ticket": 1}]      # exactly at the interval: allowed
        assert manage(t0 + timedelta(seconds=7)) is None                 # inside it: unobserved, NOT []


def test_debounced_sweep_returns_none_never_empty():
    """The whole point: an unobserved book must be distinguishable from a flat one."""
    from okmich_quant_mt5.strategy import BaseMt5Strategy

    t0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
    stub = SimpleNamespace(prev_position_chk_dt=t0, position_manager=None,
                           strategy_config=SimpleNamespace(symbol="EURUSD", magic=7),
                           _MIN_POSITION_CHK_SECONDS=BaseMt5Strategy._MIN_POSITION_CHK_SECONDS)
    manage = BaseMt5Strategy.manage_positions.__get__(stub, type(stub))
    with patch("okmich_quant_mt5.strategy.get_positions", return_value=[]) as q:
        assert manage(t0 + timedelta(seconds=1)) is None
        q.assert_not_called()


def test_query_failure_propagates_rather_than_reporting_flat():
    from okmich_quant_mt5.strategy import BaseMt5Strategy

    stub = SimpleNamespace(prev_position_chk_dt=None, position_manager=None,
                           strategy_config=SimpleNamespace(symbol="EURUSD", magic=7),
                           _MIN_POSITION_CHK_SECONDS=BaseMt5Strategy._MIN_POSITION_CHK_SECONDS)
    manage = BaseMt5Strategy.manage_positions.__get__(stub, type(stub))
    with patch("okmich_quant_mt5.strategy.get_positions", side_effect=DataFetchError("terminal down")):
        with pytest.raises(DataFetchError):
            manage(datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc))
