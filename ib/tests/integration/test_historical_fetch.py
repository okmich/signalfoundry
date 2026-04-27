"""Paper-account historical fetch smoke test. Runs only when IB_INTEGRATION=1."""
import pytest

from okmich_quant_ib.contract import IBContractConfig, SecType, resolve_contract
from okmich_quant_ib.functions.ib import fetch_historical_bars


@pytest.mark.asyncio
async def test_fetch_5mins_two_days(ib_paper):
    cfg = IBContractConfig(sec_type=SecType.CASH, exchange="IDEALPRO", currency="USD")
    contract = await resolve_contract(ib_paper, "EUR.USD", cfg)
    df = await fetch_historical_bars(
        ib_paper, contract, bar_size="5 mins", bars_to_copy=200,
        what_to_show="MIDPOINT", use_rth=False,
    )
    assert len(df) >= 200
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
