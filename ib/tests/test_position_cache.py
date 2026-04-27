"""Unit tests for IBPositionCache."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from okmich_quant_ib.position_cache import IBPositionCache


def _trade():
    return SimpleNamespace(orderStatus=SimpleNamespace(status="Filled"))


def _fill(side="BOT", shares=10, price=100.0, orderRef="sf_42", conId=12345):
    return SimpleNamespace(
        contract=SimpleNamespace(conId=conId, symbol="AAPL"),
        execution=SimpleNamespace(side=side, shares=shares, price=price, orderRef=orderRef),
    )


def _position(conId=12345, position=10, avgCost=100.0):
    return SimpleNamespace(
        contract=SimpleNamespace(conId=conId, symbol="AAPL"),
        position=position, avgCost=avgCost,
    )


class TestIBPositionCacheFills:
    def test_single_full_fill(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=100, price=50.0))
        positions = cache.get_open()
        assert len(positions) == 1
        assert positions[0]["position"] == 100
        assert positions[0]["avg_cost"] == 50.0

    def test_partial_fills_aggregate(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=40, price=50.0))
        cache.on_fill(_trade(), _fill(side="BOT", shares=60, price=51.0))
        positions = cache.get_open()
        assert positions[0]["position"] == 100
        # weighted avg: (40*50 + 60*51) / 100 = 50.6
        assert positions[0]["avg_cost"] == pytest.approx(50.6, rel=1e-9)

    def test_partial_exit_preserves_avg_cost(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=100, price=50.0))
        cache.on_fill(_trade(), _fill(side="SLD", shares=40, price=55.0))
        positions = cache.get_open()
        assert positions[0]["position"] == 60
        # avg_cost preserved on same-side reduction
        assert positions[0]["avg_cost"] == 50.0

    def test_reversal_resets_avg_cost(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=100, price=10.0))
        cache.on_fill(_trade(), _fill(side="SLD", shares=150, price=12.0))
        positions = cache.get_open()
        assert positions[0]["position"] == -50
        # On reversal, avg_cost is set to the price of the reversal fill
        assert positions[0]["avg_cost"] == 12.0

    def test_full_exit_removes_position(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=100, price=50.0))
        cache.on_fill(_trade(), _fill(side="SLD", shares=100, price=51.0))
        assert cache.get_open() == []

    def test_wrong_orderref_ignored(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(orderRef="sf_99", shares=100))
        assert cache.get_open() == []

    def test_wrong_conid_ignored(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(conId=99999, shares=100))
        assert cache.get_open() == []


class TestIBPositionCachePositions:
    def test_position_event_zero_removes(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=100, price=50.0))
        cache.on_position(_position(position=0))
        assert cache.get_open() == []

    def test_position_event_drift_trusts_broker(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_fill(_trade(), _fill(side="BOT", shares=100, price=50.0))
        cache.on_position(_position(position=120, avgCost=51.0))
        positions = cache.get_open()
        assert positions[0]["position"] == 120
        assert positions[0]["avg_cost"] == 51.0

    def test_position_event_unknown_adopted(self):
        cache = IBPositionCache("AAPL", 12345, 42)
        cache.on_position(_position(position=80, avgCost=50.0))
        positions = cache.get_open()
        assert len(positions) == 1
        assert positions[0]["position"] == 80


@pytest.mark.asyncio
async def test_resync_rebuilds_from_broker():
    ib = MagicMock()
    ib.reqPositionsAsync = AsyncMock(return_value=[
        _position(conId=12345, position=42, avgCost=99.0),
        _position(conId=99999, position=10, avgCost=50.0),  # different conId, ignored
    ])
    cache = IBPositionCache("AAPL", 12345, 42)
    await cache.resync(ib)
    positions = cache.get_open()
    assert len(positions) == 1
    assert positions[0]["position"] == 42
