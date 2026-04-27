"""Paper-account order roundtrip. Runs only when IB_INTEGRATION=1.

Places a far-from-market limit order, verifies it is visible in
``get_pending_orders``, then cancels it and verifies cancellation.
"""
import pytest

from okmich_quant_ib.contract import IBContractConfig, SecType, resolve_contract
from okmich_quant_ib.functions.ib import (
    cancel_order, get_pending_orders, place_limit_order,
)


@pytest.mark.asyncio
async def test_limit_order_lifecycle(ib_paper):
    cfg = IBContractConfig(sec_type=SecType.CASH, exchange="IDEALPRO", currency="USD")
    contract = await resolve_contract(ib_paper, "EUR.USD", cfg)
    # Far below market — will sit as a resting limit.
    trade = await place_limit_order(
        ib_paper, contract, action="BUY", quantity=20000,
        limit_price=0.01, magic=88888,
    )
    pending = get_pending_orders(ib_paper, "EUR.USD", magic=88888,
                                 con_id=contract.conId)
    assert any(p["order_id"] == trade.order.orderId for p in pending)

    cancelled = await cancel_order(ib_paper, trade)
    assert cancelled
