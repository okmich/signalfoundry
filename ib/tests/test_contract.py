"""Unit tests for contract.py — focus on make_order_ref and resolve_contract logic."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from okmich_quant_ib.contract import (
    IBContractConfig, SecType, make_order_ref, resolve_contract,
)


def test_make_order_ref():
    assert make_order_ref(42) == "sf_42"
    assert make_order_ref(0) == "sf_0"


@pytest.mark.asyncio
async def test_resolve_contract_no_match_raises():
    ib = SimpleNamespace(reqContractDetailsAsync=AsyncMock(return_value=[]))
    cfg = IBContractConfig(sec_type=SecType.STK, exchange="SMART", currency="USD")
    with pytest.raises(ValueError, match="No contract found"):
        await resolve_contract(ib, "NOPE", cfg)


@pytest.mark.asyncio
async def test_resolve_contract_ambiguous_raises():
    contract_a = SimpleNamespace()
    contract_b = SimpleNamespace()
    ib = SimpleNamespace(reqContractDetailsAsync=AsyncMock(return_value=[
        SimpleNamespace(contract=contract_a),
        SimpleNamespace(contract=contract_b),
    ]))
    cfg = IBContractConfig(sec_type=SecType.STK, exchange="SMART", currency="USD")
    with pytest.raises(ValueError, match="Ambiguous"):
        await resolve_contract(ib, "AAPL", cfg)


@pytest.mark.asyncio
async def test_resolve_contract_single_match_returns():
    contract = SimpleNamespace(symbol="AAPL")
    ib = SimpleNamespace(reqContractDetailsAsync=AsyncMock(return_value=[
        SimpleNamespace(contract=contract),
    ]))
    cfg = IBContractConfig(
        sec_type=SecType.STK, exchange="SMART", currency="USD",
        primary_exchange="NASDAQ",
    )
    result = await resolve_contract(ib, "AAPL", cfg)
    assert result is contract


@pytest.mark.asyncio
async def test_resolve_contract_normalises_forex_symbol():
    contract = SimpleNamespace()
    ib = SimpleNamespace(reqContractDetailsAsync=AsyncMock(return_value=[
        SimpleNamespace(contract=contract),
    ]))
    cfg = IBContractConfig(sec_type=SecType.CASH, exchange="IDEALPRO", currency="USD")
    # Should not raise — the dotted form is normalised before being passed to Forex().
    await resolve_contract(ib, "EUR.USD", cfg)
    ib.reqContractDetailsAsync.assert_called_once()
