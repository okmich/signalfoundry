"""POLL feed: wake at each bar boundary plus ``close_grace_seconds``, read the closed candle over REST.

A bar is accepted only when the candle with the expected open timestamp is returned AND the wall clock is past its
close; otherwise it is retried with short backoff up to ``close_max_wait_seconds`` and then reported missed (it is
backfilled as stale on the next boundary). All requests go through CCXT's rate limiter.
"""
import logging

from ..timeframe_utils import bar_open_ms
from .base import ClosedBarSource

logger = logging.getLogger(__name__)


class PollBarSource(ClosedBarSource):

    def _loops(self) -> list:
        return [self._poll_loop()]

    async def poll_once(self) -> bool:
        """Close the most recent boundary. Public so tests can drive the poller step by step."""
        expected = bar_open_ms(self.clock(), self.tf_ms) - self.tf_ms
        return await self.close_boundary(expected)

    async def _poll_loop(self) -> None:
        while not self._stopping:
            next_boundary = bar_open_ms(self.clock(), self.tf_ms) + self.tf_ms
            await self._sleep_until(next_boundary + self.close_grace_ms)
            await self.poll_once()
