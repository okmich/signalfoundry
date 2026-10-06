"""Real-venue (testnet / demo) checks. Skipped unless the CRYPTO_IT_* variables are set (see conftest).

Written against the venue's declared capabilities, so the same tests exercise Bybit (stops attached on entry, then
moved at position level) and Binance (separate reduce-only conditional "algo" orders).
"""
import asyncio

import pytest

from okmich_quant_crypto import MarketType, StopTrigger
from okmich_quant_crypto.client_order_id import make_client_order_id
from okmich_quant_crypto.data.history import fetch_candles_range
from okmich_quant_crypto.functions.crypto import cancel_order_safe, get_open_orders, place_order_idempotent
from okmich_quant_crypto.markets import MarketSpec

PERP, SPOT = "BTC/USDT:USDT", "BTC/USDT"
MAGIC = 990001
#: Resting bid this far below the last price: far enough not to fill during a test, inside the tightest band seen
#: (Binance USDⓈ-M rejects limit prices more than 5 % from the mark price).
PASSIVE_BID = 0.97


def _small_amount(spec: MarketSpec, price: float) -> float:
    """The smallest order the venue accepts at ``price`` (20 % headroom over its minimum amount / notional)."""
    notional = max(spec.min_cost or 0.0, 10.0) * 1.2
    base = max(spec.amount_to_base(spec.min_amount or 0.0) * 1.2, notional / price)
    return spec.checked_amount(base, price)


async def _open_position(exchange) -> dict | None:
    for _ in range(20):
        positions = [p for p in await exchange.fetch_positions([PERP]) if float(p.get("contracts") or 0)]
        if positions:
            return positions[0]
        await asyncio.sleep(0.5)
    return None


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
    price = spec.round_price(float((await exchange.fetch_ticker(symbol))["last"]) * PASSIVE_BID)
    amount = _small_amount(spec, price)
    cid = make_client_order_id(MAGIC, profile.client_id_rule)
    order = await place_order_idempotent(exchange, profile, spec, "limit", "buy", amount, price,
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
    caps = profile.stop_capabilities(exchange, MarketType.LINEAR_PERP)
    assert caps.any_native, f"{profile.exchange_id} declares no native perp stops"
    await profile.ensure_one_way(exchange, spec)
    last = float((await exchange.fetch_ticker(PERP))["last"])
    amount = _small_amount(spec, last)
    sl, tp = spec.round_price(last * 0.9), spec.round_price(last * 1.1)
    params = profile.order_params(MarketType.LINEAR_PERP)
    if caps.attached_on_market_entry:
        params.update(profile.attached_stop_params(MarketType.LINEAR_PERP, sl, tp, StopTrigger.MARK))
    await place_order_idempotent(exchange, profile, spec, "market", "buy", amount, None, params,
                                 make_client_order_id(MAGIC, profile.client_id_rule))
    stop_params = profile.stop_order_params(MarketType.LINEAR_PERP)
    stop_ids: list[str] = []
    try:
        position = await _open_position(exchange)
        assert position is not None
        if caps.attached_on_market_entry:
            assert all(profile.position_stop_levels(position))
        if caps.position_level:
            await profile.set_position_stops(exchange, spec, spec.round_price(last * 0.85), None, StopTrigger.LAST)
        elif caps.standalone_conditional:
            for level, is_stop_loss in ((sl, True), (tp, False)):
                cid = make_client_order_id(MAGIC, profile.client_id_rule)
                stop = profile.standalone_stop_params(MarketType.LINEAR_PERP, level, is_stop_loss, StopTrigger.MARK)
                order = await place_order_idempotent(exchange, profile, spec, "market", "sell", amount, None, stop,
                                                     cid)
                stop_ids.append(str(order["id"]))
                found = await profile.find_order_by_client_id(exchange, spec, cid)
                assert found is not None and str(found["id"]) == str(order["id"])
            listed = await get_open_orders(exchange, spec, MAGIC, profile, stop_params)
            assert set(stop_ids) <= {str(o["id"]) for o in listed}
    finally:
        try:
            close = profile.order_params(MarketType.LINEAR_PERP, reduce_only=True)
            await place_order_idempotent(exchange, profile, spec, "market", "sell", amount, None, close,
                                         make_client_order_id(MAGIC, profile.client_id_rule))
        finally:
            # Reduce-only stops cannot open a position, but a leftover could close the NEXT one: always cancel.
            cancelled = [await cancel_order_safe(exchange, profile, spec, oid, stop_params) for oid in stop_ids]
            assert all(cancelled)
