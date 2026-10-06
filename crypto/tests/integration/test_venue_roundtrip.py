"""Real-venue (testnet / demo) checks. Skipped unless the CRYPTO_IT_* variables are set (see conftest)."""
import asyncio

import pytest

from okmich_quant_crypto import MarketType, StopTrigger
from okmich_quant_crypto.client_order_id import make_client_order_id
from okmich_quant_crypto.functions.crypto import cancel_order_safe, get_open_orders, place_order_idempotent
from okmich_quant_crypto.markets import MarketSpec
from okmich_quant_crypto.data.history import fetch_candles_range

PERP, SPOT = "BTC/USDT:USDT", "BTC/USDT"
MAGIC = 990001


async def test_connect_and_markets(it_exchange):
    exchange, profile, _ = it_exchange
    assert PERP in exchange.markets and SPOT in exchange.markets
    assert exchange.has["fetchOHLCV"] and exchange.has["createOrder"]


async def test_ohlcv_pagination(it_exchange):
    exchange, profile, _ = it_exchange
    now = exchange.milliseconds()
    rows = await fetch_candles_range(exchange, "fetch_ohlcv", PERP, "1m", now - 2500 * 60_000, now,
                                     profile.ohlcv_limit(exchange))
    ts = [r[0] for r in rows]
    assert len(rows) >= 2000 and ts == sorted(set(ts))
    assert ts[-1] + 60_000 <= now  # no forming bar


@pytest.mark.parametrize("symbol,market_type", [(SPOT, MarketType.SPOT), (PERP, MarketType.LINEAR_PERP)])
async def test_far_limit_order_place_find_cancel(it_exchange, symbol, market_type):
    exchange, profile, _ = it_exchange
    spec = MarketSpec.from_market(exchange, exchange.market(symbol), market_type)
    price = float((await exchange.fetch_ticker(symbol))["last"]) * 0.5
    amount = spec.checked_amount(max(spec.amount_to_base(spec.min_amount or 0), 10.0 / price) * 1.2, price)
    cid = make_client_order_id(MAGIC, profile.client_id_rule)
    order = await place_order_idempotent(exchange, profile, spec, "limit", "buy", amount, spec.round_price(price),
                                         profile.order_params(market_type), cid)
    try:
        found = await profile.find_order_by_client_id(exchange, spec, cid)
        assert found is not None and str(found["id"]) == str(order["id"])
        assert any(str(o["id"]) == str(order["id"]) for o in await get_open_orders(exchange, spec, MAGIC, profile))
    finally:
        assert await cancel_order_safe(exchange, profile, spec, str(order["id"]))


async def test_min_size_perp_round_trip_with_native_stops(it_exchange):
    exchange, profile, _ = it_exchange
    spec = MarketSpec.from_market(exchange, exchange.market(PERP), MarketType.LINEAR_PERP)
    await profile.ensure_one_way(exchange, spec)
    last = float((await exchange.fetch_ticker(PERP))["last"])
    amount = spec.checked_amount(max(spec.amount_to_base(spec.min_amount or 0), 6.0 / last), last)
    params = profile.order_params(MarketType.LINEAR_PERP)
    params.update(profile.attached_stop_params(MarketType.LINEAR_PERP, spec.round_price(last * 0.9),
                                               spec.round_price(last * 1.1), StopTrigger.MARK))
    cid = make_client_order_id(MAGIC, profile.client_id_rule)
    await place_order_idempotent(exchange, profile, spec, "market", "buy", amount, None, params, cid)
    try:
        position = None
        for _ in range(20):
            positions = [p for p in await exchange.fetch_positions([PERP]) if float(p.get("contracts") or 0)]
            if positions:
                position = positions[0]
                break
            await asyncio.sleep(0.5)
        assert position is not None
        sl, tp = profile.position_stop_levels(position)
        assert sl and tp
        caps = profile.stop_capabilities(exchange, MarketType.LINEAR_PERP)
        if caps.position_level:
            await profile.set_position_stops(exchange, spec, spec.round_price(last * 0.85), None, StopTrigger.LAST)
    finally:
        close = profile.order_params(MarketType.LINEAR_PERP, reduce_only=True)
        await place_order_idempotent(exchange, profile, spec, "market", "sell", amount, None, close,
                                     make_client_order_id(MAGIC, profile.client_id_rule))
