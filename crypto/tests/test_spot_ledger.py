"""Spot inventory ledger: ownership by client-order-id, partial fills, base-coin fees, persistence, sell sizing."""
import pytest

from okmich_quant_crypto import MarketType, OrderSide
from okmich_quant_crypto.markets import MarketSpec
from okmich_quant_crypto.models import Fill
from okmich_quant_crypto.orders import OrderRegistry
from okmich_quant_crypto.spot_ledger import SpotInventoryLedger
from okmich_quant_crypto.state_store import StateStore
from okmich_quant_crypto.venue.base import VenueProfile

from .conftest import bootstrap, make_cfg, make_strategy, seed_candles
from .fakes import MIN

SPOT = "BTC/USDT"


def _ledger(exchange, tmp_path, magic=42):
    store = StateStore.open(tmp_path, "ledger")
    spec = MarketSpec.from_market(exchange, exchange.market(SPOT), MarketType.SPOT)
    return SpotInventoryLedger(spec, SPOT, magic, OrderRegistry(store), store), store


def _fill(tid, side, qty, price, ts, cid="sf42xa", fee_base=0.0, fee_quote=0.0):
    return Fill(trade_id=tid, order_id=None, client_order_id=cid, timestamp_ms=ts, side=OrderSide(side), price=price,
                base_qty=qty, fee_quote=fee_quote, fee_base=fee_base)


def test_only_our_fills_are_owned(exchange, tmp_path):
    ledger, _ = _ledger(exchange, tmp_path)
    assert ledger.owns(_fill("1", "buy", 1, 100, 1))
    assert not ledger.owns(_fill("2", "buy", 1, 100, 1, cid="sf421xa"))
    assert not ledger.owns(_fill("3", "buy", 1, 100, 1, cid=None))


def test_partial_fills_base_fees_and_lifecycle(exchange, tmp_path):
    ledger, _ = _ledger(exchange, tmp_path)
    opened, ended = ledger.apply_fill(_fill("1", "buy", 0.4, 100, 1, fee_base=0.0004, fee_quote=0.04))
    assert opened is not None and ended is None
    position_id = opened["position_id"]
    opened, ended = ledger.apply_fill(_fill("2", "buy", 0.6, 102, 2, fee_base=0.0006, fee_quote=0.0612))
    assert opened is None and ended is None
    assert ledger.qty == pytest.approx(0.999)                     # what was actually received
    assert ledger.get_open()[0]["position_id"] == position_id
    _, ended = ledger.apply_fill(_fill("3", "sell", 0.999, 110, 3, fee_quote=0.11))
    assert ended is not None and ended.position_id == position_id and len(ended.fills) == 3
    assert ledger.get_open() == [] and ledger.qty == 0


def test_duplicate_fills_are_applied_once(exchange, tmp_path):
    ledger, _ = _ledger(exchange, tmp_path)
    ledger.apply_fill(_fill("1", "buy", 0.5, 100, 1))
    ledger.apply_fill(_fill("1", "buy", 0.5, 100, 1))
    assert ledger.qty == pytest.approx(0.5)


def test_ledger_persists_across_restarts(exchange, tmp_path):
    ledger, _ = _ledger(exchange, tmp_path)
    ledger.apply_fill(_fill("1", "buy", 0.5, 100, 1))
    again, _ = _ledger(exchange, tmp_path)
    assert again.qty == pytest.approx(0.5) and again.get_open()[0]["avg_cost"] == pytest.approx(100)
    again.apply_fill(_fill("1", "buy", 0.5, 100, 1))              # seen ids persisted too
    assert again.qty == pytest.approx(0.5)


def test_sell_never_exceeds_free_balance(exchange, tmp_path):
    ledger, _ = _ledger(exchange, tmp_path)
    ledger.apply_fill(_fill("1", "buy", 1.0, 100, 1))
    assert ledger.sellable_qty(0.7) == pytest.approx(0.7)          # someone moved coins: sell only what is free
    assert ledger.sellable_qty(5.0) == pytest.approx(1.0)          # never balance we did not buy


def test_untracked_oversell_clamps_to_flat(exchange, tmp_path):
    ledger, _ = _ledger(exchange, tmp_path)
    ledger.apply_fill(_fill("1", "buy", 1.0, 100, 1))
    _, ended = ledger.apply_fill(_fill("2", "sell", 1.5, 100, 2))
    assert ended is not None and ledger.qty == 0


async def test_rebuild_adopts_only_tagged_history(exchange, venue, clock):
    ts = clock() - 60 * MIN
    exchange.make_trade(SPOT, "buy", 0.3, 100.0, client_order_id="sf7xold1", ts=ts)
    exchange.make_trade(SPOT, "buy", 2.0, 100.0, client_order_id=None, ts=ts + 1)          # manual buy: not ours
    exchange.make_trade(SPOT, "buy", 0.2, 101.0, client_order_id="sf8xother", ts=ts + 2)   # another strategy
    exchange.make_trade(SPOT, "buy", 0.1, 102.0, client_order_id="sf7xold2", ts=ts + 3,
                        fee={"cost": 0.0001, "currency": "BTC"})
    seed_candles(exchange, SPOT, 5 * MIN, (clock() // (5 * MIN)) * 5 * MIN - 5 * MIN, 10)
    s, _ = make_strategy(make_cfg(name="spot", market_symbol=SPOT, market_type="spot", magic=7,
                                  position_sizing={"type": "fixed", "units": 0.1}))
    await bootstrap(s, exchange, venue, clock, VenueProfile("okx"))
    pos = s.get_open_positions()
    assert len(pos) == 1 and pos[0]["position"] == pytest.approx(0.3 + 0.1 - 0.0001)
    assert pos[0]["type"] == 0


async def test_spot_close_sells_min_of_ledger_and_free(exchange, venue, clock):
    seed_candles(exchange, SPOT, 5 * MIN, (clock() // (5 * MIN)) * 5 * MIN - 5 * MIN, 10)
    exchange.balance["free"]["BTC"] = 0.25
    exchange.make_trade(SPOT, "buy", 0.3, 100.0, client_order_id="sf7xa", ts=clock() - MIN)
    exchange.balance["free"]["BTC"] = 0.25                      # 0.05 locked elsewhere
    s, _ = make_strategy(make_cfg(name="spot", market_symbol=SPOT, market_type="spot", magic=7,
                                  position_sizing={"type": "fixed", "units": 0.1}))
    await bootstrap(s, exchange, venue, clock, VenueProfile("okx"))
    assert await s.close_position(s.get_open_positions()[0])
    sell = [c for c in exchange.calls if c[0] == "create_order" and c[3] == "sell"][-1]
    assert sell[4] == pytest.approx(0.25) and "reduceOnly" not in sell[6]
