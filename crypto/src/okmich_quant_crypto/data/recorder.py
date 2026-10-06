"""Live market recorder through CCXT's unified WebSocket methods - venue-agnostic.

Order-book history is not served by any unified REST method, so the only generic way to have it is to record it.
The recorder captures, per symbol:

* ``order_book`` - a snapshot of the top ``book_levels`` levels every ``book_interval_seconds`` (on a UTC grid: a 1 s
  interval samples at :00.000, :01.000, ...). CCXT maintains the book from the venue's updates (sequence checks and
  resyncs are its job); the recorder samples it. This is a SAMPLED book, not an event-by-event replay - right for
  depth / imbalance / spread / resilience features at minute horizons, not for tick-level queue modelling;
* ``trades`` - every public trade (aggressor side, price, amount), de-duplicated by trade id;
* ``ticker`` - bid / ask / last / mark / index sampled every ``ticker_interval_seconds``;
* ``liquidations`` - where the venue streams them;
* ``gaps`` - one row per outage of any stream (start, end, error), so research can mask the holes.

Depth: venues only accept specific subscription depths (Bybit: 1 / 50 / 200 / 1000), so the recorder subscribes at
``book_subscribe_depth`` if set, else tries ``book_levels``, and falls back to the venue's default depth when the venue
rejects that number - recording the top ``book_levels`` of whatever it receives (and warning once if the book is
thinner than asked).

Layout (index ``date`` = UTC sample / event time)::

    <output_dir>/<exchange>/<SYMBOL>/<stream>/<YYYY-MM-DD>/<HH>.parquet

Rows are flushed every ``flush_seconds`` as small ``<HH>.part-*.parquet`` files (a crash loses at most one flush), and
an hour's parts are compacted into ``<HH>.parquet`` once the hour has passed. The book stream's directory name carries
its depth and interval (``order_book_l50_1s``) so files with different column sets never mix.
"""
import asyncio
import logging
import signal
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from ccxt.base.errors import ArgumentsRequired, BadRequest, NotSupported
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..enums import RecordStream, VenueEnvironment
from ..timeframe_utils import ms_to_utc, utc_now_ms
from .exchange import make_public_exchange, require_capability
from .storage import load_existing, safe_symbol, save_atomically

logger = logging.getLogger(__name__)

_CAPABILITY = {RecordStream.ORDER_BOOK: "watchOrderBook", RecordStream.TRADES: "watchTrades",
               RecordStream.TICKER: "watchTicker", RecordStream.LIQUIDATIONS: "watchLiquidations"}
#: A stream that has been failing this long is recorded as a gap even if it never recovers before shutdown.
_GAP_STREAM = "gaps"
_SEEN_TRADE_IDS = 20_000


class RecorderConfig(BaseModel):
    """What to record, and how finely. JSON-loadable (``load_from_file``)."""
    model_config = ConfigDict(extra="forbid")

    exchange_id: str
    environment: VenueEnvironment = VenueEnvironment.LIVE
    symbols: list[str]
    output_dir: str
    streams: list[RecordStream] = Field(default_factory=lambda: list(RecordStream))
    #: Levels per side kept in each book snapshot.
    book_levels: int = 50
    #: Seconds between book snapshots.
    book_interval_seconds: float = 1.0
    #: Seconds between ticker samples (defaults to the book interval).
    ticker_interval_seconds: Optional[float] = None
    #: Depth to SUBSCRIBE at, when the venue only accepts specific values (e.g. 200 on Bybit for 100 levels).
    book_subscribe_depth: Optional[int] = None
    #: A book (or ticker) not updated for this long is not sampled (the snapshot would be stale).
    stale_book_seconds: float = 30.0
    flush_seconds: float = 60.0
    ccxt_options: dict = Field(default_factory=dict)

    @field_validator("symbols")
    def _symbols(cls, v: list[str]) -> list[str]:
        v = [s.strip() for s in v if s and s.strip()]
        if not v:
            raise ValueError("at least one symbol is required")
        if len(set(v)) != len(v):
            raise ValueError("duplicate symbols")
        return v

    @field_validator("book_levels")
    def _levels(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("book_levels must be > 0")
        return v

    @field_validator("book_interval_seconds", "ticker_interval_seconds")
    def _sample_interval(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0.1:
            raise ValueError("intervals must be at least 0.1 seconds")
        return v

    @field_validator("stale_book_seconds", "flush_seconds")
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("intervals must be > 0 seconds")
        return v

    @model_validator(mode="after")
    def _streams(self) -> "RecorderConfig":
        if not self.streams:
            raise ValueError("at least one stream is required")
        if self.book_subscribe_depth is not None and self.book_subscribe_depth < self.book_levels:
            raise ValueError("book_subscribe_depth must be >= book_levels")
        return self

    @property
    def ticker_interval(self) -> float:
        return self.ticker_interval_seconds or self.book_interval_seconds

    @property
    def book_stream_name(self) -> str:
        return f"order_book_l{self.book_levels}_{_fmt_seconds(self.book_interval_seconds)}s"

    @property
    def ticker_stream_name(self) -> str:
        return f"ticker_{_fmt_seconds(self.ticker_interval)}s"

    @classmethod
    def load_from_file(cls, path) -> "RecorderConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.model_validate_json(fh.read())


def book_columns(levels: int) -> list[str]:
    """Book snapshot columns: per side, level by level (``bid_px_1, bid_sz_1, bid_px_2, ...``)."""
    cols = ["exchange_ts_ms", "book_age_ms"]
    for side in ("bid", "ask"):
        for i in range(1, levels + 1):
            cols += [f"{side}_px_{i}", f"{side}_sz_{i}"]
    return cols


TICKER_COLUMNS = ["bid", "ask", "bid_size", "ask_size", "last", "mark", "index", "base_volume", "quote_volume",
                  "exchange_ts_ms", "ticker_age_ms"]
TRADE_COLUMNS = ["trade_id", "side", "price", "amount", "cost", "recv_ms"]
LIQUIDATION_COLUMNS = ["side", "price", "contracts", "contract_size", "base_value", "quote_value", "recv_ms"]
GAP_COLUMNS = ["stream", "end_ms", "error"]


# ---------------------------------------------------------------------- storage

def dedupe_rows(df: pd.DataFrame, stream: str) -> pd.DataFrame:
    """Drop rows stored twice (a crash between writing an hour file and deleting its parts re-merges them).

    Distinct events must survive: trades are keyed by id (rows WITHOUT an id are only dropped as exact copies);
    liquidations and gaps by the whole row (several can share a millisecond); book / ticker samples by their grid time.
    """
    if df.empty:
        return df
    keyed = df.reset_index()
    if stream == RecordStream.TRADES.value and "trade_id" in keyed.columns:
        has_id = keyed["trade_id"].notna()
        dup = (has_id & keyed.duplicated(subset=["trade_id"], keep="last")) | \
            (~has_id & keyed.duplicated(keep="last"))
    elif stream in (RecordStream.LIQUIDATIONS.value, _GAP_STREAM):
        dup = keyed.duplicated(keep="last")
    else:
        dup = keyed.duplicated(subset=[keyed.columns[0]], keep="last")
    return df[~dup.to_numpy()]


class PartitionWriter:
    """Buffers rows per (symbol, stream); writes hourly part files and compacts closed hours.

    ``take`` runs on the event loop (it swaps the buffers); ``write`` and ``compact`` do file I/O and may run in a
    worker thread. Rows whose part file could not be written are handed back by ``write`` for ``requeue`` - a failed
    write never drops data. Compaction only touches this writer's own symbols: day directories it wrote parts to,
    plus its symbols' leftovers from a previous crash (scanned once).
    """

    def __init__(self, root: str | Path, exchange_id: str, clock: Callable[[], int] = utc_now_ms, symbols: tuple = ()):
        self.root = Path(root)
        self.exchange_id = exchange_id
        self.clock = clock
        self.symbols = tuple(symbols)
        self._buffers: dict[tuple[str, str], list[dict]] = {}
        self._dirty: set[Path] = set()
        self._scanned = False
        self._seq = 0
        self.rows_written: dict[str, int] = {}

    def add(self, symbol: str, stream: str, row: dict) -> None:
        self._buffers.setdefault((symbol, stream), []).append(row)

    def stream_dir(self, symbol: str, stream: str) -> Path:
        return self.root / self.exchange_id / safe_symbol(symbol) / stream

    def take(self) -> dict[tuple[str, str], list[dict]]:
        batches, self._buffers = self._buffers, {}
        return batches

    def requeue(self, failed: dict[tuple[str, str], list[dict]]) -> None:
        for key, rows in failed.items():
            self._buffers[key] = rows + self._buffers.get(key, [])

    def write(self, batches: dict[tuple[str, str], list[dict]]) -> dict[tuple[str, str], list[dict]]:
        """Write batches as part files. Returns the batches that could NOT be written (to requeue)."""
        failed: dict[tuple[str, str], list[dict]] = {}
        for (symbol, stream), rows in batches.items():
            if not rows:
                continue
            try:
                df = pd.DataFrame(rows)
                df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True), name="date")
                for hour, part in df.groupby(df.index.floor("h")):
                    directory = self.stream_dir(symbol, stream) / hour.strftime("%Y-%m-%d")
                    self._seq += 1
                    save_atomically(part.sort_index(kind="stable"),
                                    directory / f"{hour.strftime('%H')}.part-{self.clock()}-{self._seq:06d}.parquet")
                    self._dirty.add(directory)
                self.rows_written[stream] = self.rows_written.get(stream, 0) + len(df)
            except Exception:
                logger.exception("%s %s: writing %d rows failed; kept for the next flush", symbol, stream, len(rows))
                failed[(symbol, stream)] = rows
        return failed

    def flush(self) -> int:
        """Synchronous take + write + requeue + compact (tests, shutdown). Returns rows written."""
        batches = self.take()
        failed = self.write(batches)
        self.requeue(failed)
        self.compact()
        return sum(len(r) for k, r in batches.items() if k not in failed)

    def _scan_leftovers(self) -> None:
        for symbol in self.symbols:
            base = self.root / self.exchange_id / safe_symbol(symbol)
            if base.exists():
                self._dirty.update(p.parent for p in base.rglob("*.part-*.parquet"))
        self._scanned = True

    def compact(self, include_open_hour: bool = False) -> int:
        """Merge closed hours' part files into ``HH.parquet``. ``include_open_hour`` also merges the current hour
        (used at shutdown; a later run's parts for that hour are merged into it again)."""
        if not self._scanned:
            self._scan_leftovers()
        open_hour_start = (self.clock() // 3_600_000) * 3_600_000
        compacted = 0
        for directory in list(self._dirty):
            groups: dict[Path, list[Path]] = {}
            for part in directory.glob("*.part-*.parquet"):
                groups.setdefault(directory / f"{part.name.split('.part-')[0]}.parquet", []).append(part)
            pending = False
            for target, parts in groups.items():
                hour_start = pd.Timestamp(f"{directory.name} {target.stem}:00:00", tz="UTC")
                if not include_open_hour and hour_start.value // 1_000_000 >= open_hour_start:
                    pending = True
                    continue
                frames = [f for f in [load_existing(target)] + [pd.read_parquet(p) for p in sorted(parts)]
                          if f is not None]
                merged = dedupe_rows(pd.concat(frames).sort_index(kind="stable"), directory.parent.name)
                save_atomically(merged, target)
                for p in parts:
                    p.unlink(missing_ok=True)
                compacted += 1
            if not pending:
                self._dirty.discard(directory)
        return compacted

    def close(self) -> None:
        self.flush()
        self.compact(include_open_hour=True)


# ---------------------------------------------------------------------- recorder

@dataclass
class _SymbolState:
    book: Any = None
    book_recv_ms: Optional[int] = None
    book_live: bool = False
    warned_thin: bool = False
    stale_logged: bool = False
    ticker: Optional[dict] = None
    ticker_recv_ms: Optional[int] = None
    ticker_live: bool = False
    seen_trades: dict = field(default_factory=dict)
    subscribe_depth: Optional[int] = None
    depth_resolved: bool = False


class MarketRecorder:
    """Records the configured streams for every symbol until :meth:`stop` (or Ctrl+C under :meth:`run`)."""

    def __init__(self, cfg: RecorderConfig, *, exchange=None, clock: Callable[[], int] = utc_now_ms,
                 sleep: Callable = asyncio.sleep):
        self.cfg = cfg
        self.exchange = exchange
        self.clock = clock
        self.sleep = sleep
        self.writer = PartitionWriter(cfg.output_dir, cfg.exchange_id, clock, symbols=tuple(cfg.symbols))
        self.state = {s: _SymbolState(subscribe_depth=cfg.book_subscribe_depth or cfg.book_levels) for s in cfg.symbols}
        self.streams: dict[str, list[RecordStream]] = {}
        self._tasks: list[asyncio.Task] = []
        self._stop_event: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self.exchange is None:
            self.exchange = make_public_exchange(self.cfg.exchange_id, self.cfg.environment, streaming=True,
                                                 ccxt_options=self.cfg.ccxt_options)
        await self.exchange.load_markets()
        for symbol in self.cfg.symbols:
            self.exchange.market(symbol)  # BadSymbol early
            self.streams[symbol] = self._streams_for(symbol)
        self.writer.compact()  # parts left by a previous crash
        for symbol, streams in self.streams.items():
            if RecordStream.ORDER_BOOK in streams:
                self._spawn(self._book_loop(symbol), f"book:{symbol}")
                self._spawn(self._sampler(self.cfg.book_interval_seconds, partial(self.sample_book, symbol)),
                            f"book-sampler:{symbol}")
            if RecordStream.TRADES in streams:
                self._spawn(self._watch_loop(symbol, RecordStream.TRADES, self.exchange.watch_trades, self.on_trades),
                            f"trades:{symbol}")
            if RecordStream.TICKER in streams:
                self._spawn(self._watch_loop(symbol, RecordStream.TICKER, self.exchange.watch_ticker, self.on_ticker),
                            f"ticker:{symbol}")
                self._spawn(self._sampler(self.cfg.ticker_interval, partial(self.sample_ticker, symbol)),
                            f"ticker-sampler:{symbol}")
            if RecordStream.LIQUIDATIONS in streams:
                self._spawn(self._watch_loop(symbol, RecordStream.LIQUIDATIONS, self.exchange.watch_liquidations,
                                             self.on_liquidations), f"liquidations:{symbol}")
        self._spawn(self._flush_loop(), "flush")
        logger.info("recording %s on %s: %s; book %d levels every %ss, ticker every %ss -> %s",
                    self.cfg.symbols, self.cfg.exchange_id, [s.value for s in self.cfg.streams], self.cfg.book_levels,
                    self.cfg.book_interval_seconds, self.cfg.ticker_interval, self.cfg.output_dir)

    def _streams_for(self, symbol: str) -> list[RecordStream]:
        streams = []
        is_contract = bool(self.exchange.market(symbol).get("contract"))
        for stream in self.cfg.streams:
            if stream is RecordStream.LIQUIDATIONS:
                if not is_contract or not (self.exchange.has or {}).get(_CAPABILITY[stream]):
                    logger.warning("%s: %s does not stream liquidations for this market through CCXT; skipped",
                                   symbol, self.cfg.exchange_id)
                    continue
            else:
                require_capability(self.exchange, _CAPABILITY[stream], f"a {stream.value} stream")
            streams.append(stream)
        return streams

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        try:
            self.writer.close()
        finally:
            if self.exchange is not None:
                await self.exchange.close()
        logger.info("recorder stopped; rows written: %s", self.writer.rows_written)

    async def run(self) -> None:
        """Record until SIGINT / SIGTERM, then flush everything and close."""
        self._stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._stop_event.set)
        except NotImplementedError:
            def _request_stop(*_args):
                loop.call_soon_threadsafe(self._stop_event.set)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, _request_stop)
        try:
            await self.start()
            await self._stop_event.wait()
        finally:
            await self.stop()

    def request_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def _spawn(self, coro, name: str) -> None:
        self._tasks.append(asyncio.create_task(coro, name=name))

    # ------------------------------------------------------------------ loops
    async def subscribe_book(self, symbol: str):
        """One ``watch_order_book`` call at the resolved depth; falls back to the venue default if it rejects it."""
        st = self.state[symbol]
        try:
            if st.subscribe_depth is None:
                return await self.exchange.watch_order_book(symbol)
            return await self.exchange.watch_order_book(symbol, st.subscribe_depth)
        except (BadRequest, ArgumentsRequired, NotSupported) as exc:
            if st.subscribe_depth is None or st.depth_resolved:
                raise
            logger.warning("%s: %s does not accept a %d-level book subscription (%s); using its default depth and "
                           "keeping the top %d levels", symbol, self.cfg.exchange_id, st.subscribe_depth, exc,
                           self.cfg.book_levels)
            st.subscribe_depth = None
            return await self.exchange.watch_order_book(symbol)

    async def _book_loop(self, symbol: str) -> None:
        async def watch():
            return await self.subscribe_book(symbol)
        await self._watch_loop(symbol, RecordStream.ORDER_BOOK, None, self.on_book, watch=watch)

    async def _watch_loop(self, symbol: str, stream: RecordStream, method, handler, watch=None) -> None:
        backoff = 1.0
        down_since: Optional[int] = None
        last_error = ""
        while True:
            try:
                data = await (watch() if watch is not None else method(symbol))
                if down_since is not None:
                    self.record_gap(symbol, stream, down_since, self.clock(), last_error)
                    down_since = None
                if stream is RecordStream.ORDER_BOOK:
                    self.state[symbol].depth_resolved = True
                handler(symbol, data)
                backoff = 1.0
            except asyncio.CancelledError:
                if down_since is not None:
                    self.record_gap(symbol, stream, down_since, self.clock(), last_error)
                    down_since = None
                raise
            except Exception as exc:
                if down_since is None:
                    down_since = self.clock()
                    logger.warning("%s %s stream lost: %s: %s", symbol, stream.value, type(exc).__name__, exc)
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                if stream is RecordStream.ORDER_BOOK:
                    self.state[symbol].book_live = False
                elif stream is RecordStream.TICKER:
                    self.state[symbol].ticker_live = False
                try:
                    await self.sleep(backoff)
                except asyncio.CancelledError:
                    # Shutdown usually lands here, in the backoff: the outage must still be recorded.
                    self.record_gap(symbol, stream, down_since, self.clock(), last_error)
                    raise
                backoff = min(backoff * 2, 60.0)

    async def _sampler(self, interval_seconds: float, sample: Callable[[int], None]) -> None:
        interval_ms = int(interval_seconds * 1000)
        last_tick = 0
        while True:
            now = self.clock()
            next_tick = max((now // interval_ms + 1) * interval_ms, last_tick + interval_ms)
            await self.sleep(max(0.0, (next_tick - now) / 1000.0))
            last_tick = next_tick  # a timer that wakes early must not sample the same grid point twice
            try:
                sample(next_tick)
            except Exception:
                logger.exception("sampling at %d failed; the sampler keeps running", next_tick)

    async def _flush_loop(self) -> None:
        while True:
            await self.sleep(self.cfg.flush_seconds)
            batches = self.writer.take()
            try:
                failed = await asyncio.to_thread(self.writer.write, batches)
            except Exception:
                logger.exception("recorder flush failed; rows kept for the next flush")
                failed = batches
            self.writer.requeue(failed)
            try:
                await asyncio.to_thread(self.writer.compact)
            except Exception:
                logger.exception("recorder compaction failed; retried on the next flush")

    # ------------------------------------------------------------------ handlers (public so tests drive them)
    def on_book(self, symbol: str, book: Any) -> None:
        st = self.state[symbol]
        st.book, st.book_recv_ms, st.book_live = book, self.clock(), True

    def sample_book(self, symbol: str, sample_ms: int) -> Optional[dict]:
        st = self.state[symbol]
        if st.book is None or not st.book_live:
            return None
        age = sample_ms - (st.book_recv_ms or sample_ms)
        if age > self.cfg.stale_book_seconds * 1000:
            if not st.stale_logged:
                logger.warning("%s: book not updated for %.0fs - not sampling a stale book", symbol, age / 1000)
                st.stale_logged = True
            return None
        st.stale_logged = False
        n = self.cfg.book_levels
        bids, asks = list(st.book.get("bids") or [])[:n], list(st.book.get("asks") or [])[:n]
        if (len(bids) < n or len(asks) < n) and not st.warned_thin:
            logger.warning("%s: book has %d bids / %d asks, fewer than the %d levels recorded (missing levels are "
                           "NaN)", symbol, len(bids), len(asks), n)
            st.warned_thin = True
        row = {"ts": sample_ms, "exchange_ts_ms": st.book.get("timestamp"), "book_age_ms": age}
        for side, levels in (("bid", bids), ("ask", asks)):
            for i in range(n):
                level = levels[i] if i < len(levels) else None
                row[f"{side}_px_{i + 1}"] = float(level[0]) if level is not None else None
                row[f"{side}_sz_{i + 1}"] = float(level[1]) if level is not None else None
        self.writer.add(symbol, self.cfg.book_stream_name, row)
        return row

    def on_trades(self, symbol: str, trades: list) -> None:
        st = self.state[symbol]
        now = self.clock()
        for t in trades or []:
            trade_id = str(t.get("id")) if t.get("id") is not None else None
            if trade_id is not None:
                if trade_id in st.seen_trades:
                    continue
                st.seen_trades[trade_id] = None
                if len(st.seen_trades) > _SEEN_TRADE_IDS:
                    st.seen_trades.pop(next(iter(st.seen_trades)))
            if t.get("timestamp") is None:
                continue
            side = t.get("side")
            self.writer.add(symbol, RecordStream.TRADES.value, {
                "ts": int(t["timestamp"]), "trade_id": trade_id, "side": str(side).lower() if side else None,
                "price": _num(t.get("price")), "amount": _num(t.get("amount")), "cost": _num(t.get("cost")),
                "recv_ms": now})

    def on_ticker(self, symbol: str, ticker: dict) -> None:
        st = self.state[symbol]
        st.ticker, st.ticker_recv_ms, st.ticker_live = ticker, self.clock(), True

    def sample_ticker(self, symbol: str, sample_ms: int) -> Optional[dict]:
        st = self.state[symbol]
        if st.ticker is None or not st.ticker_live:
            return None
        if sample_ms - (st.ticker_recv_ms or sample_ms) > self.cfg.stale_book_seconds * 1000:
            return None  # the last ticker is too old to stand for this moment
        t = st.ticker
        row = {"ts": sample_ms, "bid": _num(t.get("bid")), "ask": _num(t.get("ask")),
               "bid_size": _num(t.get("bidVolume")), "ask_size": _num(t.get("askVolume")), "last": _num(t.get("last")),
               "mark": _num(t.get("markPrice")), "index": _num(t.get("indexPrice")),
               "base_volume": _num(t.get("baseVolume")), "quote_volume": _num(t.get("quoteVolume")),
               "exchange_ts_ms": t.get("timestamp"), "ticker_age_ms": sample_ms - (st.ticker_recv_ms or sample_ms)}
        self.writer.add(symbol, self.cfg.ticker_stream_name, row)
        return row

    def on_liquidations(self, symbol: str, liquidations: list) -> None:
        now = self.clock()
        for liq in liquidations or []:
            if liq.get("timestamp") is None:
                continue
            side = (liq.get("info") or {}).get("side") if liq.get("side") is None else liq.get("side")
            self.writer.add(symbol, RecordStream.LIQUIDATIONS.value, {
                "ts": int(liq["timestamp"]), "side": str(side).lower() if side else None,
                "price": _num(liq.get("price")), "contracts": _num(liq.get("contracts")),
                "contract_size": _num(liq.get("contractSize")), "base_value": _num(liq.get("baseValue")),
                "quote_value": _num(liq.get("quoteValue")), "recv_ms": now})

    def record_gap(self, symbol: str, stream: RecordStream, start_ms: int, end_ms: int, error: str) -> None:
        logger.warning("%s %s stream gap %s .. %s (%s)", symbol, stream.value, ms_to_utc(start_ms).isoformat(),
                       ms_to_utc(end_ms).isoformat(), error)
        self.writer.add(symbol, _GAP_STREAM, {"ts": start_ms, "stream": stream.value, "end_ms": end_ms, "error": error})


def _num(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:g}".replace(".", "p")
