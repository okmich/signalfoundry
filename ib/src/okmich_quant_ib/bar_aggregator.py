"""Aggregate IB 5-second real-time bars into a target timeframe."""
import logging
import math
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class BarAggregator:
    """Accumulate IB 5-second real-time bars into a target timeframe.

    Fires on_bar_close(completed_bar) when the target period boundary is crossed.
    Boundary detection uses UTC-epoch alignment:
        period = epoch_seconds // target_seconds

    The completed bar dict has keys: open, high, low, close, volume, time,
    sample_count, partial. Skip partial bars for execution decisions.
    """

    def __init__(self, target_minutes: int,
                 on_bar_close: Callable[[dict], Coroutine[Any, Any, None]],
                 gap_reset_seconds: int = 60, track_volume: bool = True):
        if target_minutes < 1:
            raise ValueError(f"target_minutes must be >= 1, got {target_minutes}")
        if target_minutes >= 1440:
            raise ValueError(
                f"BarAggregator does not support daily or longer bars (target_minutes={target_minutes}). "
                "IB daily bars follow exchange session definitions, not UTC midnight. "
                "Use reqHistoricalDataAsync with barSizeSetting='1 day' directly."
            )
        self.target_seconds = target_minutes * 60
        self._expected_samples = self.target_seconds // 5
        self.on_bar_close = on_bar_close
        self._gap_reset_seconds = max(gap_reset_seconds, 10)
        self._track_volume = track_volume
        self._reset()
        self._last_bar_epoch: Optional[int] = None

    async def on_realtime_bar(self, bars, has_new_bar: bool) -> None:
        """Subscribe this coroutine to bars.updateEvent.

        ib_async detects coroutine handlers and schedules them on the running
        asyncio loop automatically.
        """
        if not has_new_bar or len(bars) == 0:
            return
        bar = bars[-1]
        bar_epoch = self._extract_epoch(bar.time)

        if self._last_bar_epoch is not None:
            gap = bar_epoch - self._last_bar_epoch
            if gap == 0:
                return
            if gap < 0:
                logger.warning(f"Out-of-order bar (gap={gap}s) — resetting aggregator state")
                self._reset()
                self._last_bar_epoch = None
            elif gap > self._gap_reset_seconds:
                logger.warning(f"Bar gap of {gap}s detected — resetting aggregator state")
                self._reset()
        self._last_bar_epoch = bar_epoch

        if self._bar_start_epoch is not None:
            current_period = bar_epoch // self.target_seconds
            open_period = self._bar_start_epoch // self.target_seconds
            if current_period != open_period:
                await self._emit()

        if self._open is None:
            self._open = float(bar.open_)
            self._bar_start_epoch = (bar_epoch // self.target_seconds) * self.target_seconds
            self._high = float(bar.high)
            self._low = float(bar.low)
        else:
            self._high = max(self._high, float(bar.high))
            self._low = min(self._low, float(bar.low))
        self._close = float(bar.close)
        self._volume += float(bar.volume) if self._track_volume else 0.0
        self._sample_count += 1

    async def _emit(self):
        await self.on_bar_close({
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._volume,
            "time": datetime.fromtimestamp(self._bar_start_epoch, tz=timezone.utc),
            "sample_count": self._sample_count,
            "partial": self._sample_count < self._expected_samples,
        })
        self._reset()

    def _reset(self):
        self._open: Optional[float] = None
        self._high: float = -math.inf
        self._low: float = math.inf
        self._close: float = 0.0
        self._volume: float = 0.0
        self._bar_start_epoch: Optional[int] = None
        self._sample_count: int = 0

    @staticmethod
    def _extract_epoch(t) -> int:
        if isinstance(t, datetime):
            return int(t.timestamp())
        return int(t)
