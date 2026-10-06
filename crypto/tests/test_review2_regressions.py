"""Regression tests for the issues found in the second independent review of the crypto package."""
import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest
from ccxt.base import errors as e

from okmich_quant_crypto import OrderSide
from okmich_quant_crypto.data.history import HistoryNotServedError
from okmich_quant_crypto.data.recorder import MarketRecorder, dedupe_rows
from okmich_quant_crypto.enums import Dataset, RecordStream
from okmich_quant_crypto.models import Fill
from okmich_quant_crypto.pnl import last_round_trip, summarize
from okmich_quant_crypto.utils import crypto_data_fetcher as f

from .conftest import bootstrap, make_cfg, make_strategy, seed_candles
from .fakes import MIN, FakeClock
from .test_data_history import FakeDataExchange, _trade
from .test_market_recorder import FakeStreamExchange, _cfg

SYM, SPOT, TF = "BTC/USDT:USDT", "BTC/USDT", 5 * MIN
H = 3_600_000
D0 = int(datetime(2026, 6, 5, tzinfo=timezone.utc).timestamp() * 1000)


def _dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _open_triggers(exchange):
    return [o for o in exchange.orders.values() if o["trigger"] and o["status"] == "open"]


async def _ready(exchange, venue, clock, cfg):
    seed_candles(exchange, cfg.market_symbol, TF, (clock() // TF) * TF - TF, 10)
    s, _ = make_strategy(cfg)
    await bootstrap(s, exchange, venue, clock)
    return s


# ---------------------------------------------------------------------- 1. half-lifted spot stops are restored

async def test_spot_close_that_cannot_lift_all_stops_restores_them(exchange, venue, clock):
    cfg = make_cfg(name="spot", market_symbol=SPOT, market_type="spot", magic=7,
                   position_sizing={"type": "fixed", "units": 0.1})
    s = await _ready(exchange, venue, clock, cfg)
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = [o for o in exchange.orders.values() if o["type"] == "market"][0]
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    tp = [o for o in _open_triggers(exchange) if o["params"].get("takeProfitPrice")][0]
    real_cancel = exchange.cancel_order

    async def cancel(id, symbol=None, params=None):
        if id == tp["id"]:
            raise e.ExchangeError("cancel rejected")      # the TP cannot be lifted; the SL already was
        return await real_cancel(id, symbol, params)

    exchange.cancel_order = cancel
    assert await s.close_position(s.get_open_positions()[0]) is False
    live_sl = [o for o in _open_triggers(exchange) if o["params"].get("stopLossPrice") == 95.0]
    assert len(live_sl) == 1                                                     # SL re-armed


# ---------------------------------------------------------------------- 2. trades: same latest page for two starts

async def test_recent_start_on_a_venue_that_ignores_since_is_refused(monkeypatch, tmp_path):
    clock = FakeClock(D0 + 72 * H)
    ex = FakeDataExchange(clock, page=1000)
    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    now = clock()
    ex.public_trades = [_trade(i, now - 17 * MIN + i * 1000) for i in range(1000)]   # the latest page only
    ex.trades_ignore_since = True
    with pytest.raises(HistoryNotServedError, match="same latest page"):
        await f.fetch_trades(SYM, "x", _dt(now - 30 * MIN), _dt(now), str(tmp_path / "t"))
    assert not list((tmp_path / "t").glob("*.parquet")) if (tmp_path / "t").exists() else True


# ---------------------------------------------------------------------- 3. distinct rows survive compaction

def _frame(rows):
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True), name="date")
    return df


def test_compaction_keeps_distinct_rows_that_share_a_millisecond():
    liqs = _frame([{"ts": 1, "side": "sell", "price": 100.0 + i, "contracts": 1.0, "recv_ms": 5} for i in range(5)])
    assert len(dedupe_rows(pd.concat([liqs, liqs]), "liquidations")) == 5
    gaps = _frame([{"ts": 1, "stream": s, "end_ms": 9, "error": "x"} for s in ("trades", "ticker", "order_book")])
    assert len(dedupe_rows(gaps, "gaps")) == 3
    no_id = _frame([{"ts": 1, "trade_id": None, "price": 100.0 + i, "recv_ms": 5} for i in range(4)])
    assert len(dedupe_rows(no_id, "trades")) == 4
    assert len(dedupe_rows(pd.concat([no_id, no_id]), "trades")) == 4               # exact copies do collapse


def test_downloaded_trades_without_ids_are_kept():
    from okmich_quant_crypto.data.storage import merge_on_key
    rows = _frame([{"ts": 1, "trade_id": None, "price": float(i)} for i in range(3)])
    assert len(merge_on_key(None, rows, "trade_id")) == 3


# ---------------------------------------------------------------------- 4. outage marker survives failed reconcile

async def test_failed_reconcile_reports_failure(exchange, venue, clock):
    from okmich_quant_crypto import CryptoEventLoop
    from okmich_quant_crypto.models import Credentials
    from .conftest import RecordingStrategy

    async def factory(*_a):
        return exchange

    seed_candles(exchange, SYM, TF, (clock() // TF) * TF - TF, 10)
    s = RecordingStrategy(make_cfg())
    loop = CryptoEventLoop(venue, credentials=Credentials(api_key="k", secret="s"), exchange_factory=factory,
                           clock=clock, sleep=clock.sleep)
    loop.add_strategy(s)
    await loop._startup()

    async def boom(*_a, **_k):
        raise RuntimeError("REST down")

    s.on_reconnected = boom
    assert await loop._reconcile_all(clock() - 60_000) is False
    await loop.close()


# ---------------------------------------------------------------------- 5/6. trades streamed per day; empty windows

async def test_trades_are_written_day_by_day(monkeypatch, tmp_path):
    clock = FakeClock(D0 + 72 * H)
    ex = FakeDataExchange(clock, page=2)
    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    ex.public_trades = [_trade(i, D0 + i * 12 * H) for i in range(6)]          # three UTC days
    written = []
    real = f.write_daily_partitions

    def spy(frame, output, key):
        written.append(sorted(set(frame.index.strftime("%Y-%m-%d"))))
        return real(frame, output, key)

    monkeypatch.setattr(f, "write_daily_partitions", spy)
    assert await f.fetch_trades(SYM, "x", _dt(D0), _dt(D0 + 70 * H), str(tmp_path / "t")) == 6
    assert len(written) >= 3 and written[0] == ["2026-06-05"]                   # day 1 written before day 3 fetched


async def test_empty_windows_are_stepped_over_not_treated_as_the_end(monkeypatch, tmp_path):
    clock = FakeClock(D0 + 30 * 24 * H)
    ex = FakeDataExchange(clock, page=3)
    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    listing = D0 + 10 * 24 * H
    ex.funding = [{"timestamp": listing + i * 8 * H, "fundingRate": 0.0001} for i in range(6)]

    async def windowed(symbol=None, since=None, limit=None, params=None):
        # Like CCXT's Bybit: the venue is asked for [since, since + 50h] only.
        rows = [r for r in ex.funding if since <= r["timestamp"] <= since + 50 * H]
        return [dict(r) for r in rows[:3]]

    ex.fetch_funding_rate_history = windowed
    df = await f.fetch_dataset(Dataset.FUNDING_RATE, SYM, "x", _dt(D0), _dt(D0 + 20 * 24 * H), str(tmp_path / "f"))
    assert len(df) == 6


# ---------------------------------------------------------------------- 7. a stream that never connects: gap row

async def test_stream_down_at_shutdown_records_its_gap(tmp_path):
    clock = FakeClock(D0)
    ex = FakeStreamExchange(clock)
    ex.scripts["trades"] = [e.NetworkError("down")] * 50
    rec = MarketRecorder(_cfg(tmp_path), exchange=ex, clock=clock, sleep=asyncio.sleep)
    task = asyncio.create_task(rec._watch_loop(SYM, RecordStream.TRADES, ex.watch_trades, rec.on_trades))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gaps = rec.writer._buffers.get((SYM, "gaps"), [])
    assert len(gaps) == 1 and gaps[0]["stream"] == "trades"


# ---------------------------------------------------------------------- 8/10. stop sizing and failed moves

async def test_stop_change_uses_the_live_position_size(exchange, venue, clock):
    s = await _ready(exchange, venue, clock, make_cfg(position_sizing={"type": "fixed", "units": 0.1}))
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = [o for o in exchange.orders.values() if o["type"] == "market"][0]
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    await s.on_venue_position(exchange.positions[SYM])
    stale = s.get_open_positions()[0]                                            # snapshot at 0.1
    exchange.set_position(SYM, "long", 0.3, 100.0)
    s.position_cache.apply_position(exchange.positions[SYM])                     # the venue moved on
    assert await s._apply_levels(stale, 97.0, 110.0)
    assert all(o["amount"] == pytest.approx(0.3) for o in _open_triggers(exchange))


async def test_failed_stop_move_keeps_the_old_level_on_record(exchange, venue, clock):
    s = await _ready(exchange, venue, clock, make_cfg(position_sizing={"type": "fixed", "units": 0.1}))
    assert await s.open_position(OrderSide.BUY, stop_loss=95.0, take_profit=110.0)
    entry = [o for o in exchange.orders.values() if o["type"] == "market"][0]
    clock.advance(1_000)
    await s.on_venue_fill(exchange.fill(entry["id"], 100.0))
    await s.on_venue_position(exchange.positions[SYM])
    exchange.create_script = [e.InvalidOrder("rejected")]
    await s._apply_levels(s.get_open_positions()[0], 97.0, 110.0)
    assert s.get_open_positions()[0]["stop_loss"] == 95.0                        # what is really on the venue


# ---------------------------------------------------------------------- 9. a flip resolves into two trades

def _fill(side, qty, price, ts):
    return Fill(trade_id=f"{side}{ts}", order_id=None, client_order_id=None, timestamp_ms=ts, side=OrderSide(side),
                price=price, base_qty=qty)


def test_flip_fill_is_split_between_the_two_round_trips():
    fills = [_fill("buy", 1, 100, 1), _fill("sell", 2, 115, 2)]                 # long 1, then sell 2 -> short 1
    trip, ok = last_round_trip(fills, final_qty=-1.0)
    assert ok and summarize(trip).realized == pytest.approx(15.0)
    assert trip[-1].base_qty == pytest.approx(1.0)
    later = fills + [_fill("buy", 1, 110, 3)]                                   # the short closes at 110
    trip2, ok2 = last_round_trip(later)
    assert ok2 and trip2[0].side is OrderSide.SELL and trip2[0].base_qty == pytest.approx(1.0)
    assert summarize(trip2).realized == pytest.approx(5.0)


async def test_cache_flip_carries_the_residual(exchange, venue, clock):
    s = await _ready(exchange, venue, clock, make_cfg())
    exchange.set_position(SYM, "long", 0.1)
    s.position_cache.apply_position(exchange.positions[SYM])
    exchange.set_position(SYM, "short", 0.2)
    _, ended = s.position_cache.apply_position(exchange.positions[SYM])
    assert ended.residual_qty == pytest.approx(-0.2)


# ---------------------------------------------------------------------- 11/13. downloader edges

async def test_full_refetch_keeps_history_outside_the_range(monkeypatch, tmp_path):
    clock = FakeClock(D0 + 72 * H)
    ex = FakeDataExchange(clock)
    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    ex.funding = [{"timestamp": D0 + i * 8 * H, "fundingRate": 0.0001} for i in range(9)]
    out = str(tmp_path / "f.parquet")
    await f.fetch_dataset(Dataset.FUNDING_RATE, SYM, "x", _dt(D0), _dt(D0 + 70 * H), out)
    df = await f.fetch_dataset(Dataset.FUNDING_RATE, SYM, "x", _dt(D0 + 40 * H), _dt(D0 + 70 * H), out, resume=False)
    assert len(df) == 9


async def test_naive_datetimes_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        await f.fetch_dataset(Dataset.FUNDING_RATE, SYM, "x", datetime(2026, 1, 1), datetime(2026, 1, 2),
                              str(tmp_path / "x.pq"))


# ---------------------------------------------------------------------- 14. spot lifecycle ids never repeat

def test_spot_lifecycle_id_not_reused(exchange, tmp_path):
    from okmich_quant_crypto import MarketType
    from okmich_quant_crypto.markets import MarketSpec
    from okmich_quant_crypto.orders import OrderRegistry
    from okmich_quant_crypto.spot_ledger import SpotInventoryLedger
    from okmich_quant_crypto.state_store import StateStore
    store = StateStore.open(tmp_path, "l")
    spec = MarketSpec.from_market(exchange, exchange.market(SPOT), MarketType.SPOT)
    ledger = SpotInventoryLedger(spec, SPOT, 42, OrderRegistry(store), store)
    f1 = Fill("1", None, "sf42xa", 1000, OrderSide.BUY, 100.0, 1.0)
    opened, _ = ledger.apply_fill(f1)
    ledger.apply_fill(Fill("2", None, "sf42xa", 1000, OrderSide.SELL, 101.0, 1.0))
    reopened, _ = ledger.apply_fill(Fill("3", None, "sf42xa", 1000, OrderSide.BUY, 100.0, 1.0))
    assert reopened["position_id"] != opened["position_id"]


# ---------------------------------------------------------------------- 15. sampler hardening

async def test_ticker_not_sampled_while_its_stream_is_down(tmp_path):
    clock = FakeClock(D0)
    rec = MarketRecorder(_cfg(tmp_path), exchange=FakeStreamExchange(clock), clock=clock)
    rec.on_ticker(SYM, {"bid": 1.0, "ask": 2.0, "timestamp": D0})
    assert rec.sample_ticker(SYM, D0 + 1000) is not None
    rec.state[SYM].ticker_live = False
    assert rec.sample_ticker(SYM, D0 + 2000) is None


async def test_sampler_never_samples_a_grid_point_twice_and_survives_errors(tmp_path):
    clock = FakeClock(D0)
    rec = MarketRecorder(_cfg(tmp_path), exchange=FakeStreamExchange(clock), clock=clock)
    seen = []

    def sample(tick):
        seen.append(tick)
        if len(seen) == 2:
            raise RuntimeError("one bad sample")
        if len(seen) >= 4:
            raise asyncio.CancelledError

    async def early_sleep(_seconds):          # a timer that wakes early: the clock does not move
        await asyncio.sleep(0)

    rec.sleep = early_sleep
    with pytest.raises(asyncio.CancelledError):
        await rec._sampler(1.0, sample)
    assert seen == sorted(set(seen)) and len(seen) == 4


def test_minimum_sampling_interval(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="0.1 seconds"):
        _cfg(tmp_path, book_interval_seconds=0.0001)
