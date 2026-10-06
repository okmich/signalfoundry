"""CryptoEventLoop: startup against a fake exchange, account-stream dispatch, and the shutdown contract."""
import json

import pytest

from okmich_quant_crypto import CryptoEventLoop, OrderSide
from okmich_quant_crypto.models import Credentials

from .conftest import RecordingStrategy, make_cfg, seed_candles
from .fakes import MIN, perp_market

SYM = "BTC/USDT:USDT"


def _loop(exchange, venue, clock, *strategies, credentials=Credentials(api_key="k", secret="s"), **kw):
    async def factory(profile, venue_cfg, creds):
        return exchange

    loop = CryptoEventLoop(venue, credentials=credentials, exchange_factory=factory, clock=clock, sleep=clock.sleep,
                           **kw)
    for s in strategies:
        loop.add_strategy(s)
    return loop


async def test_startup_binds_bootstraps_and_shutdown_keeps_stops(exchange, venue, clock, tmp_path):
    seed_candles(exchange, SYM, 5 * MIN, (clock() // (5 * MIN)) * 5 * MIN - 5 * MIN, 10)
    s = RecordingStrategy(make_cfg(position_sizing={"type": "fixed", "units": 0.1}))
    loop = _loop(exchange, venue, clock, s)
    await loop._startup()
    assert s.log_binding.is_bound and s.spec.symbol == SYM
    assert loop._broker_session.broker == "fakex:demo"
    assert loop._broker_session.account_id == Credentials(api_key="k").fingerprint()

    # A filled position with standalone stops, plus a resting limit entry that never filled.
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry_id = [o for o in exchange.orders.values() if o["type"] == "market"][0]["id"]
    clock.advance(1_000)
    await loop._dispatch("fills", exchange.fill(entry_id, 100.0))
    await loop._dispatch("positions", exchange.positions[SYM])
    s.stops.set_pending(None, None)
    assert await s.open_position(OrderSide.BUY, limit_price=90.0)
    limit = [o for o in exchange.orders.values() if o["type"] == "limit"][0]

    await loop.close()
    assert exchange.orders[limit["id"]]["status"] == "canceled"                 # working entry cancelled
    stops = [o for o in exchange.orders.values() if o["trigger"]]
    assert len(stops) == 2 and all(o["status"] == "open" for o in stops)         # protection kept
    assert exchange.closed
    status_files = list(_status_files())
    assert len(status_files) == 1
    status = json.loads(status_files[0].read_text(encoding="utf-8"))
    assert status["state"] == "stopped" and status["broker"] == "fakex:demo"
    assert status["broker_disconnected"] is True and status["clean"] is True
    assert status["logical_systems"][0]["symbol"] == "BTC/USDT-USDT"


def _status_files():
    import os
    from pathlib import Path
    return Path(os.environ["OKMICH_QUANT_LOG_BASE"]).glob("**/status.json")


async def test_multi_trader_sleeves_share_the_runner_root(exchange, venue, clock):
    """As core's RunLoop: the sleeves' shared strategy name + "-multi" holds every inference path and the ONE
    status.json - the folder the Fleet Supervisor tails."""
    eth = "ETH/USDT:USDT"
    exchange.markets[eth] = perp_market(eth)
    last_closed = (clock() // (5 * MIN)) * 5 * MIN - 5 * MIN
    for symbol in (SYM, eth):
        seed_candles(exchange, symbol, 5 * MIN, last_closed, 10)
    btc = RecordingStrategy(make_cfg(name="rsi", magic=1))
    eth_sleeve = RecordingStrategy(make_cfg(name="rsi", magic=2, market_symbol=eth))
    loop = _loop(exchange, venue, clock, btc, eth_sleeve)
    await loop._startup()
    assert [s.log_binding.logical.strategy for s in (btc, eth_sleeve)] == ["rsi-multi", "rsi-multi"]
    await loop.close()
    [status] = list(_status_files())
    assert status.parent.name == "rsi-multi"
    assert {d["symbol"] for d in json.loads(status.read_text(encoding="utf-8"))["logical_systems"]} == {
        "BTC/USDT-USDT", "ETH/USDT-USDT"}


@pytest.mark.parametrize("multi,expected", [(None, "s1"), (False, "s1"), (True, "s1-multi")])
async def test_single_strategy_runner_root_follows_multi(exchange, venue, clock, multi, expected):
    seed_candles(exchange, SYM, 5 * MIN, (clock() // (5 * MIN)) * 5 * MIN - 5 * MIN, 10)
    s = RecordingStrategy(make_cfg())
    loop = _loop(exchange, venue, clock, s, multi=multi)
    await loop._startup()
    assert s.log_binding.logical.strategy == expected
    await loop.close()


async def test_dispatch_routes_by_symbol_only_to_streaming_strategies(exchange, venue, clock):
    seed_candles(exchange, SYM, 5 * MIN, (clock() // (5 * MIN)) * 5 * MIN - 5 * MIN, 10)
    s = RecordingStrategy(make_cfg())
    loop = _loop(exchange, venue, clock, s)
    await loop._startup()
    seen = []

    async def spy(item):
        seen.append(item["symbol"])

    s.on_venue_position = spy
    await loop._dispatch("positions", {"symbol": "ETH/USDT:USDT", "contracts": 1})
    await loop._dispatch("positions", {"symbol": SYM, "contracts": 0})
    assert seen == [SYM]
    await loop.close()


async def test_missing_credentials_fail_before_connecting(exchange, venue, clock):
    s = RecordingStrategy(make_cfg())
    loop = _loop(exchange, venue, clock, s, credentials=Credentials())
    with pytest.raises(ValueError, match="no API credentials"):
        await loop._startup()
    assert loop.exchange is None


async def test_venue_removed_from_the_list_is_refused_at_startup(exchange, venue, clock, monkeypatch):
    from okmich_quant_crypto.venue import registry
    loop = _loop(exchange, venue, clock, RecordingStrategy(make_cfg()))
    monkeypatch.delitem(registry._SUPPORTED, "fakex")
    with pytest.raises(Exception, match="not a supported exchange"):
        await loop._startup()
    assert loop.exchange is None


async def test_no_strategies_is_an_error(exchange, venue, clock):
    with pytest.raises(ValueError, match="no strategies"):
        await _loop(exchange, venue, clock)._startup()
