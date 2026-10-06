"""End-to-end strategy lifecycle against the fake exchange: entry, stops, modification, close, closed-trade P&L."""
import asyncio
from typing import Optional

import pytest

from okmich_quant_core import CloseReason
from okmich_quant_crypto import MarketType, OrderRole, OrderSide, StopMode, StopTrigger
from okmich_quant_crypto.markets import MarketSpec
from okmich_quant_crypto.models import FundingPayment
from okmich_quant_crypto.stops import ManagedStopController, NativeStopController
from okmich_quant_crypto.venue.base import StopCapabilities, VenueProfile

from .conftest import RecordingNotifier, bootstrap, make_cfg, make_strategy, seed_candles
from .fakes import MIN

SYM = "BTC/USDT:USDT"
TF = 5 * MIN


class PositionLevelProfile(VenueProfile):
    """A venue that keeps SL/TP on the position (like Bybit perps) and reports funding."""

    def __init__(self, exchange_id="okx", funding=None):
        super().__init__(exchange_id)
        self.stop_calls = []
        self.funding = funding

    def stop_capabilities(self, exchange, market_type):
        return StopCapabilities(attached_on_market_entry=True, attached_on_limit_entry=True,
                                standalone_conditional=True, position_level=True)

    async def set_position_stops(self, exchange, spec, stop_loss, take_profit, trigger):
        self.stop_calls.append((stop_loss, take_profit, trigger))
        pos = exchange.positions.get(spec.symbol)
        if pos is not None:
            pos["stopLossPrice"], pos["takeProfitPrice"] = stop_loss, take_profit

    async def fetch_funding(self, exchange, spec, since_ms, until_ms):
        return self.funding


async def _ready(exchange, venue, clock, cfg=None, profile=None, notifier=None):
    cfg = cfg or make_cfg(position_sizing={"type": "fixed", "units": 0.1})
    seed_candles(exchange, cfg.market_symbol, TF, (clock() // TF) * TF - TF, 10)
    s, rec = make_strategy(cfg, notifier=notifier)
    await bootstrap(s, exchange, venue, clock, profile)
    return s


def _creates(exchange):
    return [c for c in exchange.calls if c[0] == "create_order"]


async def _settle(strategy):
    for _ in range(5):
        tasks = [t for t in strategy._tasks if not t.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks)


async def _open_and_fill(s, exchange, clock, price=100.0, sl: Optional[float] = 95.0, tp: Optional[float] = 110.0):
    assert await s.open_position(OrderSide.BUY, stop_loss=sl, take_profit=tp)
    entry = _creates(exchange)[-1]
    order_id = [o for o in exchange.orders.values() if o["clientOrderId"] == entry[6]["clientOrderId"]][0]["id"]
    clock.advance(1_000)
    trade = exchange.fill(order_id, price, fee={"cost": 0.006, "currency": "USDT"})
    await s.on_venue_fill(trade)
    await s.on_venue_position(exchange.positions[SYM])
    return entry


async def _close_and_fill(s, exchange, clock, price):
    pos = s.get_open_positions()[0]
    assert await s.close_position(pos, "signal_exit")
    close = _creates(exchange)[-1]
    order_id = [o for o in exchange.orders.values() if o["clientOrderId"] == close[6]["clientOrderId"]][0]["id"]
    clock.advance(60_000)
    trade = exchange.fill(order_id, price, fee={"cost": 0.0063, "currency": "USDT"})
    await s.on_venue_fill(trade)
    await s.on_venue_position({"symbol": SYM, "contracts": 0, "side": None})
    await _settle(s)
    return close


# ---------------------------------------------------------------------- generic venue: standalone stop orders

async def test_generic_native_stops_are_standalone_orders_never_attached(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    assert s.caps.stop_mode is StopMode.NATIVE and isinstance(s.stops, NativeStopController)
    entry = await _open_and_fill(s, exchange, clock)
    assert "stopLoss" not in entry[6] and "takeProfit" not in entry[6]  # no venue-created stops we cannot move
    stops = [c for c in _creates(exchange)[1:]]
    assert {next(iter(k for k in c[6] if k.endswith("Price"))) for c in stops} == {"stopLossPrice", "takeProfitPrice"}
    assert all(c[6]["reduceOnly"] is True and c[3] == "sell" for c in stops)
    pos = s.get_open_positions()[0]
    assert pos["stop_loss"] == 95.0 and pos["take_profit"] == 110.0
    roles = {s.orders.role_of(client_order_id=c[6]["clientOrderId"]) for c in stops}
    assert roles == {OrderRole.STOP_LOSS, OrderRole.TAKE_PROFIT}


async def test_standalone_stop_is_replaced_new_before_old(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    await _open_and_fill(s, exchange, clock)
    old_sl = [o for o in exchange.orders.values() if o["params"].get("stopLossPrice") == 95.0][0]
    exchange.calls.clear()
    assert await s._apply_levels(s.get_open_positions()[0], 97.0, 110.0)
    kinds = [c[0] for c in exchange.calls]
    assert kinds == ["create_order", "cancel_order"]                     # placed first, then the old one cancelled
    assert exchange.calls[0][6]["stopLossPrice"] == 97.0 and exchange.calls[1][1] == old_sl["id"]
    assert s.get_open_positions()[0]["stop_loss"] == 97.0


async def test_close_resolves_pnl_fees_and_reason(exchange, venue, clock):
    notifier = RecordingNotifier()
    s = await _ready(exchange, venue, clock, notifier=notifier)
    await _open_and_fill(s, exchange, clock)
    pos_id = s.get_open_positions()[0]["position_id"]
    await _close_and_fill(s, exchange, clock, 105.0)
    closed = notifier.of("closed")
    assert len(closed) == 1
    _, symbol, ticket, profit, price, reason = closed[0]
    assert symbol == "BTC/USDT-USDT" and ticket == pos_id
    assert profit == pytest.approx(0.1 * 5 - 0.006 - 0.0063)              # gross - fees (generic: no funding)
    assert price == pytest.approx(105.0) and reason == CloseReason.STRATEGY.value
    # The leftover standalone stops were cancelled with the position.
    assert not [o for o in exchange.orders.values() if o["trigger"] and o["status"] == "open"]


async def test_stop_fill_on_a_certified_style_venue_is_a_stop_loss(exchange, venue, clock):
    notifier = RecordingNotifier()
    s = await _ready(exchange, venue, clock, notifier=notifier)
    await _open_and_fill(s, exchange, clock)
    sl_order = [o for o in exchange.orders.values() if o["params"].get("stopLossPrice") == 95.0][0]
    clock.advance(60_000)
    trade = exchange.fill(sl_order["id"], 95.0, fee={"cost": 0.0057, "currency": "USDT"})
    await s.on_venue_fill(trade)
    await s.on_venue_position({"symbol": SYM, "contracts": 0, "side": None})
    await _settle(s)
    assert notifier.of("closed")[0][5] == CloseReason.STOP_LOSS.value


# ---------------------------------------------------------------------- position-level venue (Bybit-like)

async def test_position_level_stops_attach_on_entry_and_replace_in_place(exchange, venue, clock):
    funding = [FundingPayment(timestamp_ms=0, amount=-0.02), FundingPayment(timestamp_ms=0, amount=0.005)]
    profile = PositionLevelProfile(funding=funding)
    notifier = RecordingNotifier()
    s = await _ready(exchange, venue, clock, profile=profile, notifier=notifier)
    entry = await _open_and_fill(s, exchange, clock)
    assert entry[6]["stopLoss"] == {"triggerPrice": 95.0} and entry[6]["takeProfit"] == {"triggerPrice": 110.0}
    assert len(_creates(exchange)) == 1                                  # no standalone stop orders
    assert await s._apply_levels(s.get_open_positions()[0], 99.0, 110.0)
    assert profile.stop_calls[-1] == (99.0, 110.0, StopTrigger.LAST)
    await _close_and_fill(s, exchange, clock, 105.0)
    profit = notifier.of("closed")[0][3]
    assert profit == pytest.approx(0.5 - 0.006 - 0.0063 - 0.015)        # funding lands in swap


# ---------------------------------------------------------------------- managed stops

async def test_managed_stop_fires_once_and_is_attributed(exchange, venue, clock):
    exchange.features["swap"]["linear"]["createOrder"] = {}
    notifier = RecordingNotifier()
    s = await _ready(exchange, venue, clock, cfg=make_cfg(position_sizing={"type": "fixed", "units": 0.1}),
                     notifier=notifier)
    assert isinstance(s.stops, ManagedStopController)
    await _open_and_fill(s, exchange, clock)
    assert len(_creates(exchange)) == 1                                  # nothing on the venue
    await s.on_venue_ticker({"last": 96.0, "bid": 95.9, "ask": 96.1})
    assert len(_creates(exchange)) == 1
    await s.on_venue_ticker({"last": 94.9})
    await s.on_venue_ticker({"last": 94.0})                               # debounced: one close only
    closes = _creates(exchange)[1:]
    assert len(closes) == 1 and closes[0][3] == "sell" and closes[0][6]["reduceOnly"] is True
    assert s.orders.role_of(client_order_id=closes[0][6]["clientOrderId"]) is OrderRole.STOP_LOSS
    order_id = [o for o in exchange.orders.values() if o["clientOrderId"] == closes[0][6]["clientOrderId"]][0]["id"]
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(order_id, 94.9))
    await s.on_venue_position({"symbol": SYM, "contracts": 0, "side": None})
    await _settle(s)
    assert notifier.of("closed")[0][5] == CloseReason.STOP_LOSS.value


async def test_managed_levels_survive_restart_and_offline_crossing_closes(exchange, venue, clock):
    exchange.features["swap"]["linear"]["createOrder"] = {}
    s = await _ready(exchange, venue, clock)
    await _open_and_fill(s, exchange, clock)
    await s.on_venue_ticker({"last": 100.0})
    s.stops.persist_heartbeat()
    went_dark = clock()
    # Process "dies". While down, a 1m candle traded through the stop.
    exchange.add_candles(SYM, "1m", [[(went_dark // MIN + 2) * MIN, 99.0, 99.5, 94.0, 96.0, 5.0]])
    clock.advance(10 * MIN)
    s2, _ = make_strategy(make_cfg(position_sizing={"type": "fixed", "units": 0.1}))
    await bootstrap(s2, exchange, venue, clock)
    assert s2.get_open_positions()[0]["stop_loss"] == 95.0               # restored from the state file
    closes = [c for c in _creates(exchange) if c[6].get("reduceOnly") and c[3] == "sell"]
    assert len(closes) == 1 and s2.orders.role_of(client_order_id=closes[0][6]["clientOrderId"]) is OrderRole.STOP_LOSS


# ---------------------------------------------------------------------- sizing + rejections

async def test_risk_sizing_uses_stop_distance_and_equity(exchange, venue, clock):
    cfg = make_cfg(position_sizing={"type": "risk_pct_of_equity", "risk_pct": 0.01})
    s = await _ready(exchange, venue, clock, cfg=cfg)
    assert await s.calculate_quantity(100.0, 95.0) == pytest.approx(10_000 * 0.01 / 5)
    with pytest.raises(Exception, match="needs a stop-loss distance"):
        await s.calculate_quantity(100.0, None)


async def test_order_below_venue_minimum_is_a_logged_non_crashing_failure(exchange, venue, clock):
    notifier = RecordingNotifier()
    cfg = make_cfg(position_sizing={"type": "fixed", "units": 0.0001})
    s = await _ready(exchange, venue, clock, cfg=cfg, notifier=notifier)
    assert await s.open_position(OrderSide.BUY) is False
    assert _creates(exchange) == [] and notifier.of("failed")


async def test_spot_strategies_cannot_short(exchange, venue, clock):
    cfg = make_cfg(name="spot", market_symbol="BTC/USDT", market_type="spot", magic=7,
                   position_sizing={"type": "fixed", "units": 0.01})
    s = await _ready(exchange, venue, clock, cfg=cfg)
    with pytest.raises(ValueError, match="cannot open short"):
        await s.open_position(OrderSide.SELL)


async def test_bootstrap_validates_venue_and_market(exchange, venue, clock):
    s, _ = make_strategy(make_cfg(timeframe="3m"))
    with pytest.raises(ValueError, match="not offered by this venue"):
        await bootstrap(s, exchange, venue, clock)
    s, _ = make_strategy(make_cfg(market_symbol="ETH/USDT:USDT"))
    with pytest.raises(Exception, match="ETH/USDT:USDT"):
        await bootstrap(s, exchange, venue, clock)


async def test_bootstrap_sets_one_way_and_leverage(exchange, venue, clock):
    await _ready(exchange, venue, clock, cfg=make_cfg(leverage=3, position_sizing={"type": "fixed", "units": 0.1}))
    assert ("set_position_mode", False, SYM) in exchange.calls
    assert ("set_leverage", 3.0, SYM) in exchange.calls


def test_spec_helper_shape(exchange):
    spec = MarketSpec.from_market(exchange, exchange.market(SYM), MarketType.LINEAR_PERP)
    assert spec.settle == "USDT" and spec.market_id == "BTCUSDT"
