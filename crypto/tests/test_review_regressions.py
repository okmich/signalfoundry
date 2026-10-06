"""Regression tests for the issues found in the independent review of the crypto package."""
import asyncio

import pytest
from ccxt.base import errors as e

from okmich_quant_crypto import OrderRole, OrderSide
from okmich_quant_crypto.enums import StopTrigger
from okmich_quant_crypto.models import Fill
from okmich_quant_crypto.pnl import last_round_trip
from okmich_quant_crypto.resilience import ErrorClass, classify_ccxt_error
from okmich_quant_crypto.state_store import StateStore
from okmich_quant_crypto.stops import NativeStopController
from okmich_quant_crypto.venue.bybit import BybitProfile

from .conftest import bootstrap, make_cfg, make_strategy, seed_candles
from .fakes import MIN
from .test_strategy_lifecycle import PositionLevelProfile, _creates

SYM, SPOT, TF = "BTC/USDT:USDT", "BTC/USDT", 5 * MIN


async def _ready(exchange, venue, clock, cfg=None, profile=None):
    cfg = cfg or make_cfg(position_sizing={"type": "fixed", "units": 0.1})
    seed_candles(exchange, cfg.market_symbol, TF, (clock() // TF) * TF - TF, 10)
    s, _ = make_strategy(cfg)
    await bootstrap(s, exchange, venue, clock, profile)
    return s


def _order_for(exchange, call):
    return [o for o in exchange.orders.values() if o["clientOrderId"] == call[6]["clientOrderId"]][0]


def _open_triggers(exchange):
    return [o for o in exchange.orders.values() if o["trigger"] and o["status"] == "open"]


async def _no_sleep(*_a, **_k):
    return None


# ---------------------------------------------------------------------- 1. lifecycle ids never repeat

async def test_lifecycle_ids_do_not_come_from_the_venue_timestamp(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    cache = s.position_cache
    exchange.set_position(SYM, "long", 0.1, 100.0, timestamp=12345)          # Bybit createdTime never changes
    opened, _ = cache.apply_position(exchange.positions[SYM])
    clock.advance(1_000)
    _, ended = cache.apply_position(None)
    clock.advance(1_000)
    exchange.set_position(SYM, "long", 0.1, 100.0, timestamp=12345)
    reopened, _ = cache.apply_position(exchange.positions[SYM])
    assert ended.position_id == opened["position_id"] != reopened["position_id"]
    assert "12345" not in opened["position_id"]


async def test_adopted_position_uses_the_venue_time_only_to_widen_the_window(exchange, venue, clock):
    exchange.set_position(SYM, "long", 0.1, 100.0, timestamp=clock() - 3 * 3600_000)
    s = await _ready(exchange, venue, clock)
    pos = s.position_cache.get_open()[0]
    assert pos["opened_ms"] == clock() - 3 * 3600_000
    assert pos["position_id"] == f"{SYM}@{clock()}"


async def test_flip_ends_one_lifecycle_and_starts_another(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    exchange.set_position(SYM, "long", 0.1)
    opened, _ = s.position_cache.apply_position(exchange.positions[SYM])
    clock.advance(1_000)
    exchange.set_position(SYM, "short", 0.1)
    reopened, ended = s.position_cache.apply_position(exchange.positions[SYM])
    assert ended.position_id == opened["position_id"] and reopened["type"] == 1
    assert reopened["position_id"] != opened["position_id"]


# ---------------------------------------------------------------------- 2. managed stops: no spurious stop-out

async def _managed_long(exchange, venue, clock):
    exchange.features["swap"]["linear"]["createOrder"] = {}
    s = await _ready(exchange, venue, clock)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    order = _order_for(exchange, _creates(exchange)[-1])
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(order["id"], 100.0))
    await s.on_venue_position(exchange.positions[SYM])
    return s


async def test_dip_before_the_position_opened_is_not_a_crossing(exchange, venue, clock):
    outage_start = clock() - 120 * MIN
    exchange.add_candles(SYM, "1m", [[(clock() // MIN - 60) * MIN, 99, 99, 94, 96, 1]])  # dip an hour before entry
    s = await _managed_long(exchange, venue, clock)
    await s.on_reconnected(outage_start, price_outage=True)
    assert len(_creates(exchange)) == 1                                         # entry only, no stop-out


async def test_account_stream_outage_does_not_trigger_the_offline_scan(exchange, venue, clock):
    s = await _managed_long(exchange, venue, clock)
    clock.advance(5 * MIN)
    exchange.add_candles(SYM, "1m", [[(clock() // MIN - 2) * MIN, 99, 99, 94, 96, 1]])
    await s.on_reconnected(clock() - 10 * MIN)                                  # account stream, price feed alive
    assert len(_creates(exchange)) == 1
    await s.on_reconnected(clock() - 10 * MIN, price_outage=True)               # prices really were unwatched
    assert len(_creates(exchange)) == 2


# ---------------------------------------------------------------------- 3. failed spot close re-arms stops

async def test_failed_spot_close_puts_the_stops_back(exchange, venue, clock):
    cfg = make_cfg(name="spot", market_symbol=SPOT, market_type="spot", magic=7,
                   position_sizing={"type": "fixed", "units": 0.1})
    s = await _ready(exchange, venue, clock, cfg=cfg)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = _order_for(exchange, _creates(exchange)[-1])
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    assert len(_open_triggers(exchange)) == 2
    exchange.create_script = [e.InsufficientFunds("rejected")]
    assert await s.close_position(s.get_open_positions()[0]) is False
    triggers = _open_triggers(exchange)
    assert len(triggers) == 2                                                   # re-armed after the failed sell
    assert {t["params"].get("stopLossPrice") or t["params"].get("takeProfitPrice") for t in triggers} == {95.0, 110.0}


async def test_managed_stop_rearms_when_its_close_raises(exchange, venue, clock):
    s = await _managed_long(exchange, venue, clock)

    async def boom(*_a, **_k):
        raise RuntimeError("network died mid-close")

    s.stops.close = boom
    with pytest.raises(RuntimeError):
        await s.stops.on_price(s.get_open_positions()[0], 94.0)
    assert s.stops._fired_for is None


# ---------------------------------------------------------------------- 4. ambiguous entry keeps its stops

async def test_unknown_entry_keeps_pending_levels(exchange, venue, clock, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    s = await _ready(exchange, venue, clock)
    exchange.create_script = [e.RequestTimeout("t")] * 3
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0) is False
    assert s.stops._state["pending"] == {"stop_loss": 95.0, "take_profit": 110.0}


def test_bybit_spot_duplicate_and_finished_codes():
    prof = BybitProfile()
    assert classify_ccxt_error(e.InvalidOrder('bybit {"retCode":170141}'), prof) is ErrorClass.DUPLICATE
    assert classify_ccxt_error(e.InvalidOrder('bybit {"retCode":170139}'), prof) is ErrorClass.ALREADY_DONE
    assert classify_ccxt_error(e.InvalidOrder('bybit {"retCode":170146}'), prof) is ErrorClass.UNKNOWN_STATE


# ---------------------------------------------------------------------- 5/6. stops follow size; no duplicates

async def test_stops_resize_when_partial_fills_grow_the_position(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = _order_for(exchange, _creates(exchange)[-1])
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0, qty=0.05))
    await s.on_venue_position(exchange.positions[SYM])
    assert {t["amount"] for t in _open_triggers(exchange)} == {0.05}
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0, qty=0.05))
    await s.on_venue_position(exchange.positions[SYM])
    triggers = _open_triggers(exchange)
    assert len(triggers) == 2 and all(t["amount"] == pytest.approx(0.1) for t in triggers)


async def test_concurrent_reconciles_do_not_duplicate_stop_orders(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = _order_for(exchange, _creates(exchange)[-1])
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    await s.on_venue_position(exchange.positions[SYM])
    exchange.set_position(SYM, "long", 0.2, 100.0)                              # grew while streams were down
    await asyncio.gather(*(s.on_reconnected(None) for _ in range(3)))
    triggers = _open_triggers(exchange)
    assert len(triggers) == 2 and all(t["amount"] == pytest.approx(0.2) for t in triggers)


async def test_untracked_leftover_stop_orders_are_cancelled(exchange, venue, clock):
    s = await _ready(exchange, venue, clock)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = _order_for(exchange, _creates(exchange)[-1])
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    await s.on_venue_position(exchange.positions[SYM])
    stray = exchange._new_order(SYM, "market", "sell", 0.1, None, {"clientOrderId": "sf42xstray", "stopLossPrice": 90,
                                                                    "reduceOnly": True})
    await s.on_reconnected(None)
    assert exchange.orders[stray["id"]]["status"] == "canceled"
    assert len(_open_triggers(exchange)) == 2


async def test_orphaned_cancel_is_retried(exchange, tmp_path):
    from okmich_quant_crypto.markets import MarketSpec
    from okmich_quant_crypto import MarketType
    from okmich_quant_crypto.venue.base import VenueProfile
    results = [False, True]
    cancelled = []

    async def cancel(order_id, params=None):
        cancelled.append(order_id)
        return results.pop(0)

    async def no_ids():
        return set()

    spec = MarketSpec.from_market(exchange, exchange.market(SYM), MarketType.LINEAR_PERP)
    ctl = NativeStopController("t", spec, StateStore.open(tmp_path, "s"), StopTrigger.LAST, exchange=exchange,
                               profile=VenueProfile("okx"), position_level=False, submit=None, cancel=cancel,
                               fetch_open_stop_ids=no_ids)
    ctl._bind("p1")
    ctl._orders()[OrderRole.STOP_LOSS.value] = {"order_id": "9", "level": 95.0, "qty": 0.1}
    await ctl.on_position_closed("p1")      # first cancel unconfirmed -> orphan, retried at once -> gone
    assert cancelled == ["9", "9"] and ctl._orphans() == []


# ---------------------------------------------------------------------- 7. position-level venue stop cleared

async def test_cleared_venue_stop_is_reapplied_after_reconnect(exchange, venue, clock):
    profile = PositionLevelProfile()
    s = await _ready(exchange, venue, clock, profile=profile)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = _order_for(exchange, _creates(exchange)[-1])
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    await s.on_venue_position(exchange.positions[SYM])
    calls = len(profile.stop_calls)
    exchange.positions[SYM]["stopLossPrice"] = None                             # someone removed it on the venue
    exchange.positions[SYM]["takeProfitPrice"] = None
    await s.on_reconnected(None)
    assert len(profile.stop_calls) == calls + 1 and profile.stop_calls[-1][:2] == (95.0, 110.0)


# ---------------------------------------------------------------------- 8. back-to-back trades resolve

def _fill(side, qty, price, ts):
    return Fill(trade_id=f"{side}{ts}", order_id=None, client_order_id=None, timestamp_ms=ts, side=OrderSide(side),
                price=price, base_qty=qty)


def test_window_starting_inside_the_previous_trip_still_resolves():
    # The previous long was opened BEFORE the window; only its closing sell is visible.
    fills = [_fill("sell", 1, 100, 1), _fill("buy", 2, 90, 2), _fill("sell", 2, 95, 3)]
    trip, complete = last_round_trip(fills)
    assert complete and [f.timestamp_ms for f in trip] == [2, 3]


# ---------------------------------------------------------------------- 11. late, older spot fill is caught up

async def test_spot_catch_up_overlaps_the_last_fill(exchange, venue, clock):
    cfg = make_cfg(name="spot", market_symbol=SPOT, market_type="spot", magic=7,
                   position_sizing={"type": "fixed", "units": 0.1})
    s = await _ready(exchange, venue, clock, cfg=cfg)
    newer = exchange.make_trade(SPOT, "buy", 0.2, 100.0, client_order_id="sf7xb", ts=clock() - MIN)
    await s.on_venue_fill(newer)                                                # the stream delivered the NEWER fill
    exchange.make_trade(SPOT, "buy", 0.1, 100.0, client_order_id="sf7xa", ts=clock() - 5 * MIN)  # older, missed
    await s.on_reconnected(None)
    assert s.get_open_positions()[0]["position"] == pytest.approx(0.3)
