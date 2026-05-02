"""Smoke tests for IB strategy quantity sizing and bracket-rejection bookkeeping.

Full end-to-end strategy tests would require mocking ``IB``, ``Contract``,
and the BarAggregator pipeline; those scenarios live in ``integration/`` for
paper-account verification. This module exercises the surface-level guards.
"""
from types import SimpleNamespace

import pytest

from okmich_quant_core import PositionSizingConfig, PositionSizingType
from okmich_quant_ib.contract import IBContractConfig, SecType
from okmich_quant_ib.strategy import BaseIBStrategy


class _DummySignal:
    def generate(self, df):
        return [], [], [], []


class _DummyConfig(SimpleNamespace):
    pass


def _make_strategy(*, position_sizing=None, position_manager=None, max_pos=1) -> BaseIBStrategy:
    if position_sizing is None:
        position_sizing = PositionSizingConfig(type=PositionSizingType.FIXED, units=1.0)
    cfg = _DummyConfig(
        name="t", symbol="AAPL", timeframe="5 mins", magic=42,
        signal_params={}, position_sizing=position_sizing,
        max_number_of_open_positions=max_pos, bars_to_copy=10,
        position_manager=position_manager, filters=[],
    )

    class _Concrete(BaseIBStrategy):
        async def on_new_bar(self):
            return None

    contract_cfg = IBContractConfig(
        sec_type=SecType.STK, exchange="SMART", currency="USD",
    )
    return _Concrete(cfg, _DummySignal(), contract_cfg)


def test_calculate_quantity_uses_fixed_lot_and_rounds_to_increment():
    s = _make_strategy(position_sizing=PositionSizingConfig(type=PositionSizingType.FIXED, units=12.7))
    s.contract_info = {"size_increment": 1.0, "min_size": 1.0}
    assert s.calculate_quantity() == 13.0


def test_calculate_quantity_respects_min_size():
    s = _make_strategy(position_sizing=PositionSizingConfig(type=PositionSizingType.FIXED, units=0.0001))
    s.contract_info = {"size_increment": 1.0, "min_size": 1.0}
    assert s.calculate_quantity() == 1.0


def test_calculate_quantity_risk_pct_raises():
    s = _make_strategy(position_sizing=PositionSizingConfig(
        type=PositionSizingType.RISK_PCT_OF_EQUITY, risk_pct=0.01,
    ))
    s.contract_info = {}
    with pytest.raises(NotImplementedError, match="risk_pct_of_equity"):
        s.calculate_quantity()


def test_register_bracket_keys_by_every_orderid():
    s = _make_strategy()
    trades = [
        SimpleNamespace(order=SimpleNamespace(orderId=10), orderStatus=SimpleNamespace(status="Submitted")),
        SimpleNamespace(order=SimpleNamespace(orderId=11), orderStatus=SimpleNamespace(status="Submitted")),
        SimpleNamespace(order=SimpleNamespace(orderId=12), orderStatus=SimpleNamespace(status="Submitted")),
    ]
    s._register_bracket(trades)
    assert s._bracket_trades[10] is trades
    assert s._bracket_trades[11] is trades
    assert s._bracket_trades[12] is trades


@pytest.mark.asyncio
async def test_place_order_with_sl_or_tp_raises():
    s = _make_strategy()
    s.contract_info = {"size_increment": 1.0, "min_size": 1.0}
    s.ib = SimpleNamespace()
    s.contract = SimpleNamespace()
    # Catching NotImplementedError specifically — it is re-raised through the
    # except clause path because place_order wraps only IBxxxError types.
    with pytest.raises(NotImplementedError, match="atomic"):
        await s.place_order("buy", price=100.0, sl=99.0)
