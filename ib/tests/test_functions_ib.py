"""Unit tests for low-level functions/ib.py."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from okmich_quant_ib.functions.ib import get_pending_orders, get_positions


def _trade(symbol, conId, action, orderRef, status="Submitted"):
    order = SimpleNamespace(
        orderId=1, action=action, orderType="MKT", totalQuantity=10, lmtPrice=0.0,
        auxPrice=0.0, orderRef=orderRef,
    )
    return SimpleNamespace(
        order=order,
        contract=SimpleNamespace(symbol=symbol, conId=conId),
        orderStatus=SimpleNamespace(status=status),
    )


def _fill_record(symbol, conId, side, shares, price, orderRef):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol, conId=conId),
        execution=SimpleNamespace(side=side, shares=shares, price=price, orderRef=orderRef),
    )


class TestGetPendingOrders:
    def test_filter_by_orderref(self):
        ib = MagicMock()
        ib.openTrades.return_value = [
            _trade("AAPL", 12345, "BUY", "sf_42"),
            _trade("AAPL", 12345, "BUY", "sf_99"),
        ]
        result = get_pending_orders(ib, "AAPL", magic=42)
        assert len(result) == 1

    def test_filter_by_conid_preferred(self):
        ib = MagicMock()
        ib.openTrades.return_value = [
            _trade("ES", 11111, "BUY", "sf_42"),
            _trade("ES", 22222, "BUY", "sf_42"),
        ]
        result = get_pending_orders(ib, "ES", magic=42, con_id=22222)
        assert len(result) == 1
        assert result[0]["trade"].contract.conId == 22222

    def test_action_filter_excludes_protective_stops(self):
        ib = MagicMock()
        ib.openTrades.return_value = [
            _trade("AAPL", 12345, "BUY", "sf_42"),
            _trade("AAPL", 12345, "SELL", "sf_42"),  # protective stop
        ]
        result = get_pending_orders(ib, "AAPL", magic=42, action_filter="BUY")
        assert len(result) == 1
        assert result[0]["action"] == "BUY"


class TestGetPositions:
    def test_aggregates_by_orderref(self):
        ib = MagicMock()
        ib.fills.return_value = [
            _fill_record("AAPL", 12345, "BOT", 50, 100.0, "sf_42"),
            _fill_record("AAPL", 12345, "BOT", 50, 101.0, "sf_42"),
        ]
        result = get_positions(ib, "AAPL", magic=42)
        assert len(result) == 1
        assert result[0]["position"] == 100

    def test_conid_filter_excludes_other_expiries(self):
        ib = MagicMock()
        ib.fills.return_value = [
            _fill_record("ES", 11111, "BOT", 5, 4500, "sf_42"),
            _fill_record("ES", 22222, "BOT", 5, 4501, "sf_42"),
        ]
        result = get_positions(ib, "ES", magic=42, con_id=11111)
        assert len(result) == 1
        assert result[0]["contract"].conId == 11111

    def test_wrong_orderref_excluded(self):
        ib = MagicMock()
        ib.fills.return_value = [
            _fill_record("AAPL", 12345, "BOT", 50, 100.0, "sf_99"),
        ]
        assert get_positions(ib, "AAPL", magic=42) == []
