"""IB half of closed-position reconciliation.

Written after review found two defects here that no test covered: ``Fill`` has no ``.order`` (so the close-reason
heuristic silently never fired), and ``CommissionReport.realizedPNL`` defaults to ``0.0`` and is set to
``UNSET_DOUBLE`` when unreported (so an unguarded read reports a profit of 1.8e308). Both are pure-function
concerns, so they are tested off the class without a live IB connection.
"""
from __future__ import annotations

from types import SimpleNamespace

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
