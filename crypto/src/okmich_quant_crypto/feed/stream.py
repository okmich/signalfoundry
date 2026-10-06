"""STREAM feed: ``watch_ohlcv`` triggers, REST confirms.

Two loops run while started:

* the WebSocket loop feeds candles to :class:`CandleCloseDetector`; every close it detects is delivered through the
  REST-reconciling sequencer. A WebSocket error backs off, resets the detector and resyncs from REST.
* a boundary watchdog wakes ``close_max_wait / 2`` after each boundary and closes the bar from REST if the socket has
  not. A quiet market, a stalled socket, or a lost final update must never cost a bar.
"""
import asyncio
import logging

from ..bar_aggregator import CandleCloseDetector
from ..timeframe_utils import bar_open_ms, last_closed_bar_open_ms
from .base import ClosedBarSource

logger = logging.getLogger(__name__)


class StreamBarSource(ClosedBarSource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = CandleCloseDetector(self.tf_ms)

    def _loops(self) -> list:
        return [self._ws_loop(), self._watchdog_loop()]

    async def handle_rows(self, rows: list) -> None:
        """Process one ``watch_ohlcv`` delivery. Public so tests can script the stream."""
        for open_ms in self.detector.feed(rows):
            await self.close_boundary(open_ms)

    async def _ws_loop(self) -> None:
        backoff = 1.0
        while not self._stopping:
            try:
                rows = await self.exchange.watch_ohlcv(self.symbol, self.timeframe)
                await self.handle_rows(rows)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("%s %s: watch_ohlcv failed (%s: %s); reconnecting in %.0fs", self.symbol, self.timeframe,
                               type(exc).__name__, exc, backoff)
                self.detector.reset()
                await self.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                try:
                    await self.resync()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("%s %s: REST resync after stream error failed", self.symbol, self.timeframe)

    async def watchdog_tick(self) -> None:
        """Close the most recent boundary from REST if the socket has not. Public so tests can drive it."""
        expected = last_closed_bar_open_ms(self.clock(), self.tf_ms)
        if self.last_open_ms is None or self.last_open_ms < expected:
            logger.info("%s %s: stream has not closed bar %d yet; closing it from REST", self.symbol, self.timeframe,
                        expected)
            await self.close_boundary(expected)

    async def _watchdog_loop(self) -> None:
        while not self._stopping:
            next_boundary = bar_open_ms(self.clock(), self.tf_ms) + self.tf_ms
            await self._sleep_until(next_boundary + self.close_max_wait_ms // 2)
            await self.watchdog_tick()
