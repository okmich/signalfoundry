"""The closed-bar contract shared by the STREAM and POLL feeds.

Both feeds end in the same two components, which is what makes their output identical:

* :class:`BarReconciler` - the REST read of a candle. A bar is accepted only when the venue returns the candle with
  the expected open timestamp AND the wall clock is past its close. The last row CCXT returns is the forming candle;
  it is never taken as closed.
* :class:`BarSequencer` - exactly one emission per timeframe boundary, in order. A gap (reconnect, missed poll) is
  backfilled from REST first. Bars are delivered with a ``live`` flag: a bar that closed more than
  ``close_max_wait_seconds`` ago is STALE - it updates the price buffer but must not run the strategy, because acting
  on a signal from an old bar is worse than skipping it.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

from ..models import ClosedBar
from ..timeframe_utils import last_closed_bar_open_ms, timeframe_to_ms, utc_now_ms

logger = logging.getLogger(__name__)

OnBar = Callable[[ClosedBar, bool], Awaitable[None]]
OnMissed = Callable[[int], None]
Clock = Callable[[], int]


class BarReconciler:
    """REST reads of closed candles for one (symbol, timeframe)."""

    def __init__(self, exchange, symbol: str, timeframe: str, ohlcv_limit: int, clock: Clock = utc_now_ms):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.tf_ms = timeframe_to_ms(timeframe)
        self.ohlcv_limit = max(2, int(ohlcv_limit))
        self.clock = clock

    def _is_closed(self, open_ms: int) -> bool:
        return self.clock() >= open_ms + self.tf_ms

    async def fetch_closed(self, open_ms: int) -> Optional[ClosedBar]:
        """The closed candle opening at ``open_ms``, or ``None`` if it is not (yet) available."""
        if not self._is_closed(open_ms):
            return None
        rows = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, open_ms, 2)
        for row in rows or []:
            if int(row[0]) == open_ms:
                return ClosedBar.from_row(row)
        return None

    async def fetch_range(self, first_open_ms: int, last_open_ms: int) -> list[ClosedBar]:
        """Closed candles with open time in ``[first_open_ms, last_open_ms]``, paged; venue gaps are simply absent."""
        bars: dict[int, ClosedBar] = {}
        cursor = first_open_ms
        while cursor <= last_open_ms:
            rows = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, cursor, self.ohlcv_limit)
            if not rows:
                break
            for row in rows:
                ts = int(row[0])
                if first_open_ms <= ts <= last_open_ms and self._is_closed(ts):
                    bars[ts] = ClosedBar.from_row(row)
            newest = max(int(r[0]) for r in rows)
            if newest < cursor:
                break
            cursor = newest + self.tf_ms
        return [bars[ts] for ts in sorted(bars)]

    async def fetch_recent(self, count: int) -> list[ClosedBar]:
        """The ``count`` most recent closed candles (fewer if the venue has less history)."""
        last = last_closed_bar_open_ms(self.clock(), self.tf_ms)
        first = last - (count - 1) * self.tf_ms
        return await self.fetch_range(first, last)


class BarSequencer:
    """Exactly-once, in-order delivery of reconciled bars, with gap backfill and the staleness rule."""

    def __init__(self, reconciler: BarReconciler, on_bar: OnBar, max_wait_ms: int, clock: Clock = utc_now_ms):
        self.reconciler = reconciler
        self.on_bar = on_bar
        self.tf_ms = reconciler.tf_ms
        self.max_wait_ms = max_wait_ms
        self.clock = clock
        self.last_open_ms: Optional[int] = None
        self._lock = asyncio.Lock()

    def is_live(self, open_ms: int) -> bool:
        return self.clock() - (open_ms + self.tf_ms) <= self.max_wait_ms

    async def offer(self, open_ms: int) -> bool:
        """Deliver the bar opening at ``open_ms`` (and any gap before it). True once it is delivered or already was."""
        async with self._lock:
            if self.last_open_ms is not None and open_ms <= self.last_open_ms:
                return True
            if self.last_open_ms is not None and open_ms > self.last_open_ms + self.tf_ms:
                await self._backfill(self.last_open_ms + self.tf_ms, open_ms - self.tf_ms)
            bar = await self.reconciler.fetch_closed(open_ms)
            if bar is None:
                return False
            await self._emit(bar, self.is_live(open_ms))
            return True

    async def _backfill(self, first_open_ms: int, last_open_ms: int) -> None:
        expected = (last_open_ms - first_open_ms) // self.tf_ms + 1
        bars = await self.reconciler.fetch_range(first_open_ms, last_open_ms)
        if len(bars) < expected:
            logger.warning("%s %s: backfill %d..%d returned %d of %d bars (venue gap)", self.reconciler.symbol,
                           self.reconciler.timeframe, first_open_ms, last_open_ms, len(bars), expected)
        for bar in bars:
            if self.last_open_ms is None or bar.open_ms > self.last_open_ms:
                # Gap bars are by definition late: buffer-only, never a strategy cycle.
                await self._emit(bar, False)

    async def _emit(self, bar: ClosedBar, live: bool) -> None:
        self.last_open_ms = bar.open_ms
        await self.on_bar(bar, live)


class ClosedBarSource(ABC):
    """Emits exactly one reconciled closed bar per timeframe boundary to ``on_bar(bar, live)``."""

    def __init__(self, exchange, symbol: str, timeframe: str, on_bar: OnBar, *, ohlcv_limit: int,
                 close_grace_seconds: float, close_max_wait_seconds: float, clock: Clock = utc_now_ms,
                 sleep: Callable[[float], Awaitable] = asyncio.sleep, on_missed: Optional[OnMissed] = None):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.clock = clock
        self.sleep = sleep
        self.close_grace_ms = int(close_grace_seconds * 1000)
        self.close_max_wait_ms = int(close_max_wait_seconds * 1000)
        self.reconciler = BarReconciler(exchange, symbol, timeframe, ohlcv_limit, clock)
        self.sequencer = BarSequencer(self.reconciler, on_bar, self.close_max_wait_ms, clock)
        self.tf_ms = self.reconciler.tf_ms
        self.on_missed = on_missed
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    @property
    def last_open_ms(self) -> Optional[int]:
        return self.sequencer.last_open_ms

    async def seed(self, count: int) -> list[ClosedBar]:
        """Warm-up history; also sets the sequencer's position so the first live bar is the next boundary."""
        bars = await self.reconciler.fetch_recent(count)
        if bars:
            self.sequencer.last_open_ms = bars[-1].open_ms
        return bars

    async def close_boundary(self, open_ms: int) -> bool:
        """Deliver the bar opening at ``open_ms``, retrying REST until ``close_max_wait`` after its close."""
        deadline = open_ms + self.tf_ms + self.close_max_wait_ms
        delay = 0.5
        while True:
            try:
                if await self.sequencer.offer(open_ms):
                    return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("%s %s: closing bar %d failed: %s", self.symbol, self.timeframe, open_ms, exc)
            if self._stopping:
                return False
            if self.clock() >= deadline:
                break
            await self.sleep(delay)
            delay = min(delay * 2, 5.0)
        logger.error("%s %s: bar opening %d not available within %.0fs of its close - MISSED (it will be backfilled "
                     "as stale on the next boundary)", self.symbol, self.timeframe, open_ms,
                     self.close_max_wait_ms / 1000)
        if self.on_missed is not None:
            try:
                self.on_missed(open_ms)
            except Exception:
                logger.exception("on_missed callback failed")
        return False

    async def resync(self) -> None:
        """Catch up to the latest closed bar (after a reconnect or a stall)."""
        await self.close_boundary(last_closed_bar_open_ms(self.clock(), self.tf_ms))

    def start(self) -> None:
        self._stopping = False
        self._tasks = [asyncio.create_task(coro, name=f"{type(self).__name__}:{self.symbol}") for coro in self._loops()]

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []

    @abstractmethod
    def _loops(self) -> list:
        """The coroutines this source runs while started."""

    async def _sleep_until(self, target_ms: int) -> None:
        delay = (target_ms - self.clock()) / 1000.0
        if delay > 0:
            await self.sleep(delay)
