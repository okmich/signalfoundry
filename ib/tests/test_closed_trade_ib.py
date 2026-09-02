"""IB half of closed-position reconciliation.

Written after review found two defects here that no test covered: ``Fill`` has no ``.order`` (so the close-reason
heuristic silently never fired), and ``CommissionReport.realizedPNL`` defaults to ``0.0`` and is set to
``UNSET_DOUBLE`` when unreported (so an unguarded read reports a profit of 1.8e308). Both are pure-function
concerns, so they are tested off the class without a live IB connection.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from okmich_quant_core.closed_trade import CloseReason
from okmich_quant_ib.strategy import BaseIBStrategy

UNSET_DOUBLE = 1.7976931348623157e+308


def _trade(order_type):
    return SimpleNamespace(order=SimpleNamespace(orderType=order_type))


# --------------------------------------------------------------------------------------
# close-reason inference reads the ORDER, which lives on the trade and not the fill
# --------------------------------------------------------------------------------------

def test_stop_order_is_a_stop_loss():
    assert BaseIBStrategy._infer_close_reason(_trade("STP")) == CloseReason.STOP_LOSS
    assert BaseIBStrategy._infer_close_reason(_trade("STP LMT")) == CloseReason.STOP_LOSS


def test_limit_order_is_a_take_profit():
    assert BaseIBStrategy._infer_close_reason(_trade("LMT")) == CloseReason.TAKE_PROFIT


def test_market_order_is_unknown_and_left_for_intent_to_refine():
    assert BaseIBStrategy._infer_close_reason(_trade("MKT")) == CloseReason.UNKNOWN


def test_a_fill_is_not_a_trade():
    """Regression: ib_async's Fill is (contract, execution, commissionReport, time) — it has NO .order.

    Passing a Fill here made the heuristic return UNKNOWN for every exit, silently, forever.
    """
    fill = SimpleNamespace(contract=object(), execution=object(), commissionReport=object(), time=None)
    assert not hasattr(fill, "order")
    assert BaseIBStrategy._infer_close_reason(fill) == CloseReason.UNKNOWN


def test_missing_order_does_not_raise():
    assert BaseIBStrategy._infer_close_reason(None) == CloseReason.UNKNOWN
    assert BaseIBStrategy._infer_close_reason(SimpleNamespace(order=None)) == CloseReason.UNKNOWN


# --------------------------------------------------------------------------------------
# realised P/L: unset must be unresolved, never a number
# --------------------------------------------------------------------------------------

def test_unset_pnl_sentinel_is_not_reported_as_profit():
    """Regression: UNSET_DOUBLE arrives as a real float and would poison every downstream sum."""
    assert BaseIBStrategy._realised_pnl(SimpleNamespace(realizedPNL=UNSET_DOUBLE)) is None
    assert BaseIBStrategy._realised_pnl(SimpleNamespace(realizedPNL=-UNSET_DOUBLE)) is None


def test_absent_commission_report_is_unresolved():
    assert BaseIBStrategy._realised_pnl(None) is None
    assert BaseIBStrategy._realised_pnl(SimpleNamespace()) is None


def test_a_genuine_zero_pnl_is_a_value_not_a_missing_one():
    assert BaseIBStrategy._realised_pnl(SimpleNamespace(realizedPNL=0.0)) == 0.0


def test_real_pnl_passes_through():
    assert BaseIBStrategy._realised_pnl(SimpleNamespace(realizedPNL=-12.5)) == -12.5


def test_finite_rejects_junk_without_raising():
    assert BaseIBStrategy._finite("not a number") is None
    assert BaseIBStrategy._finite(None) is None
    assert BaseIBStrategy._finite(3) == 3.0


# --------------------------------------------------------------------------------------
# resolution end to end
# --------------------------------------------------------------------------------------

class _Resolver:
    def __init__(self):
        self.strategy_config = SimpleNamespace(symbol="AAPL", magic=7)
        self._last_fills = {}

    # staticmethod/classmethod must be re-wrapped: reading them off the class yields plain functions, and
    # assigning those here would silently turn them into instance methods (self would land in the first arg).
    resolve_closed_trade = BaseIBStrategy.resolve_closed_trade
    _realised_pnl = classmethod(BaseIBStrategy._realised_pnl.__func__)
    _finite = classmethod(BaseIBStrategy._finite.__func__)
    _infer_close_reason = staticmethod(BaseIBStrategy._infer_close_reason)
    _UNSET_DOUBLE = BaseIBStrategy._UNSET_DOUBLE


def _fill(price=101.0, shares=10.0, pnl=25.0, commission=-1.0):
    return SimpleNamespace(
        contract=SimpleNamespace(conId=42),
        execution=SimpleNamespace(price=price, shares=shares, orderRef="ref"),
        commissionReport=SimpleNamespace(realizedPNL=pnl, commission=commission),
        time=None)


def test_resolution_uses_the_trades_order_type_and_the_fills_pnl():
    r = _Resolver()
    r._last_fills["42"] = (_trade("LMT"), _fill())
    trade = r.resolve_closed_trade("42", {"avg_cost": 100.0})
    assert trade.reason == CloseReason.TAKE_PROFIT
    assert trade.entry_price == 100.0
    assert trade.exit_price == 101.0
    assert trade.volume == 10.0
    assert trade.profit == 25.0
    assert trade.net_profit == pytest.approx(24.0)
    assert trade.resolved is True


def test_resolution_without_a_reported_pnl_is_flagged_unresolved():
    r = _Resolver()
    r._last_fills["42"] = (_trade("STP"), _fill(pnl=UNSET_DOUBLE))
    trade = r.resolve_closed_trade("42", {"avg_cost": 100.0})
    assert trade.resolved is False
    assert trade.profit == 0.0                      # not 1.8e308
    assert trade.reason == CloseReason.STOP_LOSS    # still describable
    assert "UNRESOLVED" in trade.describe()


def test_short_position_reports_positive_volume():
    r = _Resolver()
    r._last_fills["42"] = (_trade("LMT"), _fill(shares=-5.0))
    assert r.resolve_closed_trade("42", {}).volume == 5.0


def test_a_key_with_no_stored_fill_is_unresolvable():
    r = _Resolver()
    assert r.resolve_closed_trade("999", {}) is None


def test_the_stored_fill_is_consumed_so_it_cannot_be_reused():
    r = _Resolver()
    r._last_fills["42"] = (_trade("LMT"), _fill())
    assert r.resolve_closed_trade("42", {}) is not None
    assert r.resolve_closed_trade("42", {}) is None
    assert r._last_fills == {}


# --------------------------------------------------------------------------------------
# a close that did not happen leaves no claim behind
# --------------------------------------------------------------------------------------

def _close_stub(calls):
    """BaseIBStrategy.close_position lifted onto a stub, with intent recording spied on."""
    stub = SimpleNamespace(
        ib=object(), strategy_config=SimpleNamespace(symbol="EURUSD", magic=7),
        _position_key=lambda position: str(position["conId"]),
        note_close_intent=lambda k, r: calls.append(("note", k, r)),
        clear_close_intent=lambda k: calls.append(("clear", k)),
        _notify_trade_failed=lambda *a, **kw: None,
    )
    return BaseIBStrategy.close_position.__get__(stub, type(stub))


def test_a_failed_ib_close_withdraws_its_intent():
    """Intent is recorded before the request. Left behind after a failure it outlives the call, and a human
    closing that position later has their MANUAL close reported as ours."""
    import asyncio
    from okmich_quant_ib.resilience import IBPermanentError

    calls = []
    close = _close_stub(calls)
    with patch("okmich_quant_ib.strategy.ib_close_position",
               side_effect=IBPermanentError("rejected", code=201)):
        assert asyncio.run(close({"conId": 42}, reason="exit_signal")) is False
    assert calls == [("note", "42", "exit_signal"), ("clear", "42")]


def test_a_successful_ib_close_keeps_its_intent():
    import asyncio

    async def _ok(*a, **kw):
        return True

    calls = []
    close = _close_stub(calls)
    with patch("okmich_quant_ib.strategy.ib_close_position", side_effect=_ok):
        assert asyncio.run(close({"conId": 42}, reason="exit_signal")) is True
    assert calls == [("note", "42", "exit_signal")]


# --------------------------------------------------------------------------------------
# reconnect must not leave handlers registered twice
# --------------------------------------------------------------------------------------

class _Event:
    """ib_async's Event surface, as much of it as subscribe/unsubscribe touch."""

    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)          # raises when absent, as ib_async's does
        return self


class _FakeIB:
    def __init__(self):
        self.fillEvent = _Event()
        self.positionEvent = _Event()
        self.execDetailsEvent = _Event()
        self.errorEvent = _Event()

    def reqRealTimeBars(self, *a, **kw):
        return SimpleNamespace(updateEvent=_Event())

    def reqMktData(self, *a, **kw):
        return SimpleNamespace()

    def cancelRealTimeBars(self, *a, **kw):
        pass

    def cancelMktData(self, *a, **kw):
        pass


def test_reconnect_does_not_register_handlers_twice():
    """ib_async keeps handlers on the IB object across a reconnect, so re-subscribing without detaching
    first leaves every handler registered twice — and each reconnect adds another copy, so one fill is
    applied to the position cache N times. Silent, and it compounds."""
    import asyncio
    from okmich_quant_ib.contract import SecType

    ib = _FakeIB()

    async def _resync(_ib):
        return None

    stub = SimpleNamespace(
        ib=ib, contract=object(), contract_cfg=SimpleNamespace(sec_type=SecType.CASH),
        _rt_bars=None, _ticker=None,
        _bar_aggregator=SimpleNamespace(on_realtime_bar=lambda *a: None, _reset=lambda: None),
        _position_cache=SimpleNamespace(on_fill=lambda *a: None, on_position=lambda *a: None, resync=_resync),
        _on_position_fill=lambda *a: None, _on_fill=lambda *a: None, _on_error=lambda *a: None,
    )
    cls = type(stub)
    stub._subscribe = BaseIBStrategy._subscribe.__get__(stub, cls)
    stub._unsubscribe = BaseIBStrategy._unsubscribe.__get__(stub, cls)
    resubscribe = BaseIBStrategy._resubscribe.__get__(stub, cls)

    asyncio.run(stub._subscribe(ib))
    events = ("fillEvent", "positionEvent", "execDetailsEvent", "errorEvent")
    baseline = {name: len(getattr(ib, name).handlers) for name in events}
    assert baseline["fillEvent"] == 2                       # the cache, then close detection

    for _ in range(3):                                      # three reconnects
        asyncio.run(resubscribe(ib))

    for name, count in baseline.items():
        assert len(getattr(ib, name).handlers) == count, f"{name} accumulated duplicate handlers"

    fills = ib.fillEvent.handlers
    assert fills.index(stub._position_cache.on_fill) < fills.index(stub._on_position_fill), \
        "close detection must stay registered AFTER the cache, so the fill is applied before the diff"
