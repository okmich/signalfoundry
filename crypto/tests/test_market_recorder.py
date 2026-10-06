"""Live market recorder: configuration, book sampling, depth fallback, trades, gaps, hourly parquet layout."""
import asyncio

import pandas as pd
import pytest
from ccxt.base.errors import BadRequest, NetworkError
from pydantic import ValidationError

from okmich_quant_crypto.data.recorder import MarketRecorder, PartitionWriter, RecorderConfig, book_columns
from okmich_quant_crypto.enums import RecordStream
from okmich_quant_crypto.resilience import VenueUnsupportedError

from .fakes import FakeClock, FakeExchange

PERP, SPOT = "BTC/USDT:USDT", "BTC/USDT"
HOUR = 3_600_000
T = 1_780_002_000_000 - (1_780_002_000_000 % HOUR) + 10 * 60_000     # 10 minutes into an hour


class FakeStreamExchange(FakeExchange):
    """WebSocket side: each watch_* call pops the next scripted item (an exception is raised); empty = wait forever."""

    def __init__(self, clock, allowed_depths=(50, 200), **kw):
        super().__init__(clock, **kw)
        self.has.update({"watchOrderBook": True, "watchTrades": True, "watchTicker": True, "watchLiquidations": True})
        self.allowed_depths = allowed_depths
        self.scripts = {"book": [], "trades": [], "ticker": [], "liq": []}
        self.book_limits = []

    async def _next(self, kind):
        script = self.scripts[kind]
        if not script:
            await asyncio.sleep(3600)
        item = script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def watch_order_book(self, symbol, limit=None, params=None):
        self.book_limits.append(limit)
        if limit is not None and limit not in self.allowed_depths:
            raise BadRequest(f"fakex watchOrderBook(): limit can be one of {list(self.allowed_depths)}")
        return await self._next("book")

    async def watch_trades(self, symbol, since=None, limit=None, params=None):
        return await self._next("trades")

    async def watch_ticker(self, symbol, params=None):
        return await self._next("ticker")

    async def watch_liquidations(self, symbol, since=None, limit=None, params=None):
        return await self._next("liq")


def _cfg(tmp_path, **kw):
    data = {"exchange_id": "fakex", "symbols": [PERP], "output_dir": str(tmp_path / "rec"), "book_levels": 3,
            "book_interval_seconds": 1.0}
    data.update(kw)
    return RecorderConfig(**data)


def _book(n_bids=5, n_asks=5, ts=T):
    return {"bids": [[100.0 - i, 1.0 + i] for i in range(n_bids)],
            "asks": [[100.5 + i, 2.0 + i] for i in range(n_asks)], "timestamp": ts}


# ---------------------------------------------------------------------- config

def test_config_defaults_and_names(tmp_path):
    cfg = RecorderConfig(exchange_id="bybit", symbols=[PERP], output_dir=str(tmp_path))
    assert cfg.book_levels == 50 and cfg.book_interval_seconds == 1.0 and cfg.streams == list(RecordStream)
    assert cfg.book_stream_name == "order_book_l50_1s" and cfg.ticker_stream_name == "ticker_1s"
    cfg = _cfg(tmp_path, book_levels=30, book_interval_seconds=3, ticker_interval_seconds=0.5)
    assert cfg.book_stream_name == "order_book_l30_3s" and cfg.ticker_stream_name == "ticker_0p5s"


@pytest.mark.parametrize("kw,msg", [({"book_levels": 0}, "book_levels"), ({"book_interval_seconds": 0}, "intervals"),
                                    ({"symbols": []}, "at least one symbol"), ({"symbols": [PERP, PERP]}, "duplicate"),
                                    ({"book_subscribe_depth": 2}, "book_subscribe_depth"), ({"streams": []}, "stream"),
                                    ({"bogus": 1}, "Extra inputs")])
def test_config_validation(tmp_path, kw, msg):
    with pytest.raises(ValidationError, match=msg):
        _cfg(tmp_path, **kw)


# ---------------------------------------------------------------------- sampling

def test_book_snapshot_columns_padding_and_staleness(tmp_path):
    clock = FakeClock(T)
    rec = MarketRecorder(_cfg(tmp_path, book_levels=3), exchange=FakeStreamExchange(clock), clock=clock)
    assert rec.sample_book(PERP, T) is None                                   # no book yet
    rec.on_book(PERP, _book(n_bids=2))                                         # thinner than 3 levels on the bid side
    row = rec.sample_book(PERP, T + 1000)
    assert [k for k in row if k != "ts"] == book_columns(3)
    assert row["bid_px_1"] == 100.0 and row["bid_sz_2"] == 2.0 and row["bid_px_3"] is None
    assert row["ask_px_3"] == 102.5 and row["book_age_ms"] == 1000
    assert rec.sample_book(PERP, T + 31_000) is None                           # 31 s without an update: stale


def test_book_copy_is_taken_at_sample_time(tmp_path):
    clock = FakeClock(T)
    rec = MarketRecorder(_cfg(tmp_path), exchange=FakeStreamExchange(clock), clock=clock)
    live = _book()
    rec.on_book(PERP, live)
    first = rec.sample_book(PERP, T + 1000)
    live["bids"][0] = [99.9, 7.0]                                              # CCXT mutates its book in place
    second = rec.sample_book(PERP, T + 2000)
    assert first["bid_px_1"] == 100.0 and second["bid_px_1"] == 99.9


async def test_unsupported_depth_falls_back_to_venue_default(tmp_path):
    clock = FakeClock(T)
    ex = FakeStreamExchange(clock, allowed_depths=(50, 200))
    ex.scripts["book"] = [_book()]
    rec = MarketRecorder(_cfg(tmp_path, book_levels=30), exchange=ex, clock=clock)
    book = await rec.subscribe_book(PERP)
    assert book["timestamp"] == T and ex.book_limits == [30, None]
    assert rec.state[PERP].subscribe_depth is None


async def test_explicit_subscribe_depth_is_used(tmp_path):
    clock = FakeClock(T)
    ex = FakeStreamExchange(clock)
    ex.scripts["book"] = [_book()]
    rec = MarketRecorder(_cfg(tmp_path, book_levels=30, book_subscribe_depth=200), exchange=ex, clock=clock)
    await rec.subscribe_book(PERP)
    assert ex.book_limits == [200]


def test_trades_are_deduplicated_and_tickers_sampled(tmp_path):
    clock = FakeClock(T)
    rec = MarketRecorder(_cfg(tmp_path), exchange=FakeStreamExchange(clock), clock=clock)
    trades = [{"id": "a", "timestamp": T, "side": "Buy", "price": 100.0, "amount": 0.1, "cost": 10.0},
              {"id": "b", "timestamp": T + 1, "side": "sell", "price": 100.1, "amount": 0.2, "cost": 20.02}]
    rec.on_trades(PERP, trades)
    rec.on_trades(PERP, trades[1:])                                           # redelivered
    rows = rec.writer._buffers[(PERP, "trades")]
    assert [r["trade_id"] for r in rows] == ["a", "b"] and rows[0]["side"] == "buy"
    rec.on_ticker(PERP, {"bid": 99.9, "ask": 100.1, "last": 100.0, "markPrice": 100.02, "timestamp": T})
    row = rec.sample_ticker(PERP, T + 1000)
    assert row["mark"] == 100.02 and row["ticker_age_ms"] == 1000


# ---------------------------------------------------------------------- gaps + streams

async def test_stream_outage_is_recorded_as_a_gap(tmp_path):
    clock = FakeClock(T)
    ex = FakeStreamExchange(clock)
    ex.scripts["trades"] = [NetworkError("reset"), NetworkError("reset"),
                            [{"id": "x", "timestamp": T + 5000, "side": "buy", "price": 1, "amount": 1, "cost": 1}]]
    rec = MarketRecorder(_cfg(tmp_path), exchange=ex, clock=clock, sleep=clock.sleep)
    task = asyncio.create_task(rec._watch_loop(PERP, RecordStream.TRADES, ex.watch_trades, rec.on_trades))
    for _ in range(20):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gaps = rec.writer._buffers[(PERP, "gaps")]
    assert len(gaps) == 1 and gaps[0]["stream"] == "trades" and gaps[0]["end_ms"] > gaps[0]["ts"]
    assert "NetworkError" in gaps[0]["error"]
    assert [r["trade_id"] for r in rec.writer._buffers[(PERP, "trades")]] == ["x"]


async def test_start_checks_capabilities_and_skips_liquidations_for_spot(tmp_path):
    clock = FakeClock(T)
    ex = FakeStreamExchange(clock)
    rec = MarketRecorder(_cfg(tmp_path, symbols=[PERP, SPOT]), exchange=ex, clock=clock, sleep=clock.sleep)
    await rec.start()
    assert RecordStream.LIQUIDATIONS in rec.streams[PERP] and RecordStream.LIQUIDATIONS not in rec.streams[SPOT]
    await rec.stop()
    assert ex.closed
    ex2 = FakeStreamExchange(clock)
    ex2.has["watchOrderBook"] = False
    with pytest.raises(VenueUnsupportedError, match="watchOrderBook"):
        await MarketRecorder(_cfg(tmp_path), exchange=ex2, clock=clock).start()


# ---------------------------------------------------------------------- storage

def _rows(n, start=T, step=1000):
    return [{"ts": start + i * step, "v": float(i)} for i in range(n)]


def test_hourly_parts_then_compaction(tmp_path):
    clock = FakeClock(T)
    w = PartitionWriter(tmp_path, "fakex", clock)
    for r in _rows(3):
        w.add(PERP, "order_book_l3_1s", r)
    w.flush()
    day = pd.Timestamp(T, unit="ms").strftime("%Y-%m-%d")
    day_dir = tmp_path / "fakex" / "BTC_USDT_USDT" / "order_book_l3_1s" / day
    hour = pd.Timestamp(T, unit="ms").strftime("%H")
    assert len(list(day_dir.glob(f"{hour}.part-*.parquet"))) == 1 and not (day_dir / f"{hour}.parquet").exists()
    for r in _rows(2, start=T + 10_000):
        w.add(PERP, "order_book_l3_1s", r)
    clock.advance(HOUR)                                                        # the hour has closed
    w.flush()
    compacted = pd.read_parquet(day_dir / f"{hour}.parquet")
    assert len(compacted) == 5 and compacted.index.is_monotonic_increasing and compacted.index.name == "date"
    assert not list(day_dir.glob(f"{hour}.part-*.parquet"))


def test_trades_dedupe_on_compaction_and_close_compacts_open_hour(tmp_path):
    clock = FakeClock(T)
    w = PartitionWriter(tmp_path, "fakex", clock, symbols=(PERP,))
    same_ms = [{"ts": T, "trade_id": "a", "price": 1.0}, {"ts": T, "trade_id": "b", "price": 2.0}]
    for r in same_ms:
        w.add(PERP, "trades", r)
    w.flush()
    w.add(PERP, "trades", {"ts": T, "trade_id": "b", "price": 2.0})              # duplicate from a reconnect
    w.close()
    day_dir = tmp_path / "fakex" / "BTC_USDT_USDT" / "trades" / pd.Timestamp(T, unit="ms").strftime("%Y-%m-%d")
    files = list(day_dir.glob("*.parquet"))
    assert len(files) == 1 and ".part-" not in files[0].name
    df = pd.read_parquet(files[0])
    assert sorted(df["trade_id"]) == ["a", "b"]                                  # same-ms trades kept, dup dropped


def test_leftover_parts_from_a_crash_are_compacted_on_start(tmp_path):
    clock = FakeClock(T)
    w = PartitionWriter(tmp_path, "fakex", clock)
    for r in _rows(2):
        w.add(PERP, "ticker_1s", r)
    w.flush()                                                                  # "crash": parts left behind
    clock.advance(2 * HOUR)
    PartitionWriter(tmp_path, "fakex", clock, symbols=(PERP,)).compact()
    assert not list(tmp_path.rglob("*.part-*.parquet")) and len(list(tmp_path.rglob("*.parquet"))) == 1


class _StubRecorder:
    """Stands in for MarketRecorder.run(): waits for request_stop like the real one."""

    def __init__(self):
        self._stop_event = None
        self.stopped = False

    async def run(self):
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()
        self.stopped = True

    def request_stop(self):
        if self._stop_event is not None:
            self._stop_event.set()


async def test_duration_stops_the_recorder_gracefully():
    from okmich_quant_crypto.utils.crypto_market_recorder import record
    stub = _StubRecorder()
    await asyncio.wait_for(record(stub, duration_hours=0.05 / 3600), timeout=5)   # 50 ms
    assert stub.stopped


async def test_duration_must_be_positive():
    from okmich_quant_crypto.utils.crypto_market_recorder import record
    with pytest.raises(ValueError, match="duration_hours"):
        await record(_StubRecorder(), duration_hours=0)
