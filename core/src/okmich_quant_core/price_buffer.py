import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict

import pandas as pd


logger = logging.getLogger(__name__)


class PriceBuffer:
    """
    Efficiently buffers OHLCV price data by maintaining a rolling window and fetching only new bars.

    The buffer maintains exactly buffer_size + 1 bars and automatically determines whether to:
    - Do a full reload (when empty, stale, or validation fails)
    - Do an incremental fetch (only fetch new bars since last update)

    Timezone handling:
        The buffer infers its timezone from the first data it receives and uses that timezone
        consistently for all subsequent operations. No pre-configured or system timezone is assumed.
        Callers must pass timezone-aware datetimes to get_fetch_params() and update().

    Deduplication:
        When new data overlaps with buffered data, the most recently supplied data always wins.
        Deduplication is applied immediately after merging, before any trimming or validation.

    Bar boundary:
        At any wall-clock time, the buffer anchors to the last COMPLETE bar. For a 5-minute
        timeframe at 12:00:00, the forming bar (12:00) is ignored and the boundary is set to
        11:55:00. The 12:00 bar may not yet be available or stable from the broker.
    """

    def __init__(self, symbol: str, timeframe: int, buffer_size: int, exclude_columns: list = None,
                 timeframe_minutes: int = None):
        """
        Initialize price buffer for a specific symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_M5)
            buffer_size: Number of historical bars to maintain (buffer will hold buffer_size + 1 bars)
            exclude_columns: Column names to drop on arrival (e.g., ['open', 'spread', 'real_volume']).
                             Defaults to ["real_volume"].
            timeframe_minutes: Bar duration in minutes. Required.
        """
        if timeframe_minutes is None:
            raise ValueError(f"timeframe_minutes is required for symbol {symbol}")

        self.symbol = symbol
        self.timeframe = timeframe
        self.buffer_size = buffer_size
        self.exclude_columns = exclude_columns if exclude_columns is not None else ["real_volume"]
        self.timeframe_minutes = timeframe_minutes

        self.data: pd.DataFrame = pd.DataFrame()
        self.last_update_dt: Optional[datetime] = None
        self._data_tz = None           # inferred from first data received; never pre-set
        self._stable_boundary: Optional[datetime] = None  # last complete bar boundary as of last update

        excluded_info = (
            f", excluding columns: {self.exclude_columns}" if self.exclude_columns else ""
        )
        logger.info(
            f"PriceBuffer initialized: {symbol} @ {self.timeframe_minutes}min, "
            f"buffer_size={buffer_size} (will maintain {buffer_size + 1} bars){excluded_info}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_fetch_params(self, current_dt: datetime) -> Dict[str, object]:
        """
        Determine what data needs to be fetched from the broker.

        Args:
            current_dt: Current wall-clock time. Must be timezone-aware.

        Returns a dict the caller uses to request data:
            - 'start_dt': datetime to fetch from (None = position-based full reload)
            - 'count':    number of bars to fetch (0 = buffer already current, no fetch needed)
            - 'rounded_current_dt': last complete bar boundary used as the fetch ceiling
        """
        current_dt = self._require_tz_aware(current_dt, context="get_fetch_params")
        current_dt = self._coerce_tz(current_dt)
        rounded_current_dt = self._last_complete_bar_boundary(current_dt)

        if self._needs_full_reload(rounded_current_dt):
            logger.info(f"[{self.symbol}] Full reload required")
            return {
                "start_dt": None,
                "count": self.buffer_size + 1,
                "rounded_current_dt": rounded_current_dt,
            }

        # Start from the bar immediately after what is already stored.
        # e.g. last_bar=11:50, boundary=11:55 → fetch from 11:55, count=1
        next_bar_start = self.last_bar_time + timedelta(minutes=self.timeframe_minutes)

        if next_bar_start > rounded_current_dt:
            # Buffer is already at or ahead of the stable boundary — nothing to fetch.
            logger.debug(f"[{self.symbol}] Buffer already current (last_bar={self.last_bar_time})")
            return {
                "start_dt": next_bar_start,
                "count": 0,
                "rounded_current_dt": rounded_current_dt,
            }

        gap_minutes = (rounded_current_dt - next_bar_start).total_seconds() / 60
        count = int(gap_minutes / self.timeframe_minutes) + 1
        logger.debug(
            f"[{self.symbol}] Incremental fetch: last_bar={self.last_bar_time}, "
            f"start={next_bar_start}, boundary={rounded_current_dt}, "
            f"gap={gap_minutes:.1f}min, count={count}"
        )
        return {
            "start_dt": next_bar_start,
            "count": count,
            "rounded_current_dt": rounded_current_dt,
        }

    def update(self, new_data: pd.DataFrame, current_dt: datetime) -> bool:
        """
        Merge new_data into the buffer.

        Args:
            new_data: Incoming OHLCV bars.
            current_dt: Wall-clock time of this update. Must be timezone-aware.

        New data always wins on duplicate timestamps. Returns True on success.
        """
        if new_data is None or len(new_data) == 0:
            logger.warning(f"[{self.symbol}] Received empty data, skipping update")
            return False

        current_dt = self._require_tz_aware(current_dt, context="update")
        new_data = new_data.copy()

        # ── Timezone: require tz-aware index, infer buffer tz on first load ──
        if not hasattr(new_data.index, "tz") or new_data.index.tz is None:
            raise ValueError(
                f"[{self.symbol}] update: received a timezone-naive data index. "
                f"All data passed to PriceBuffer must have a timezone-aware DatetimeIndex."
            )
        if self._data_tz is None:
            self._data_tz = new_data.index.tz
            logger.info(f"[{self.symbol}] Timezone inferred from data: {self._data_tz}")
        elif str(new_data.index.tz) != str(self._data_tz):
            logger.warning(
                f"[{self.symbol}] Incoming tz ({new_data.index.tz}) differs from "
                f"buffer tz ({self._data_tz}); converting"
            )
            new_data.index = new_data.index.tz_convert(self._data_tz)

        # ── Drop excluded columns ──────────────────────────────────────
        if self.exclude_columns:
            cols_to_drop = [c for c in self.exclude_columns if c in new_data.columns]
            if cols_to_drop:
                new_data = new_data.drop(columns=cols_to_drop)
                logger.debug(f"[{self.symbol}] Dropped columns: {cols_to_drop}")

        # ── Stable boundary: only complete bars enter the buffer ──────
        # current_dt tells us where the clock is. Any bar at or beyond the
        # forming bar's open time is incomplete and must be excluded.
        stable_boundary = self._last_complete_bar_boundary(self._coerce_tz(current_dt))
        new_data = new_data[new_data.index <= stable_boundary]
        if new_data.empty:
            # All incoming bars were beyond the stable boundary (broker sent the forming bar
            # early, or we were called redundantly). Buffer is already current — not an error.
            logger.debug(
                f"[{self.symbol}] All incoming bars are beyond stable boundary "
                f"({stable_boundary}); buffer already current"
            )
            self._stable_boundary = stable_boundary
            self.last_update_dt = current_dt
            return True

        # ── Merge: new_data appended last so keep='last' prefers it ───
        # Dedup is always applied — covers both initial fill (incoming data may
        # itself contain duplicate timestamps) and incremental updates.
        combined = new_data if self.is_empty() else pd.concat([self.data, new_data])
        self.data = (
            combined[~combined.index.duplicated(keep="last")]
            .sort_index()
        )

        # ── Trim to buffer_size + 1 ────────────────────────────────────
        if len(self.data) > self.buffer_size + 1:
            self.data = self.data.iloc[-(self.buffer_size + 1):]

        if not self._validate_data():
            logger.error(f"[{self.symbol}] Validation failed after update, resetting buffer")
            self.reset()
            return False

        self._stable_boundary = stable_boundary
        self.last_update_dt = current_dt
        logger.debug(
            f"[{self.symbol}] Buffer updated: {len(self.data)} bars "
            f"(stable up to {stable_boundary})"
        )
        return True

    def get_data(self) -> pd.DataFrame:
        """
        Return a copy of the buffered OHLCV data.

        All bars returned are guaranteed to be complete and stable as of the
        current_dt supplied to the last update() call. The forming bar is
        never included, regardless of when this method is called.
        """
        return self.data.copy()

    def reset(self) -> None:
        """Clear all buffer state. Next get_fetch_params call will trigger a full reload."""
        logger.info(f"[{self.symbol}] Buffer reset")
        self.data = pd.DataFrame()
        self.last_update_dt = None
        self._stable_boundary = None
        # Intentionally preserve _data_tz across resets — the timezone does not change
        # when we reconnect to the same broker feed.

    def is_empty(self) -> bool:
        return len(self.data) == 0

    @property
    def last_bar_time(self) -> Optional[datetime]:
        """Timestamp of the most recent bar held in the buffer."""
        if self.is_empty():
            return None
        return self.data.index[-1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_tz_aware(self, dt: datetime, context: str = "") -> datetime:
        """Raise ValueError if dt is timezone-naive. Libraries must not guess timezone."""
        if dt.tzinfo is None:
            raise ValueError(
                f"[{self.symbol}] {context}: received a timezone-naive datetime. "
                f"All datetimes passed to PriceBuffer must be timezone-aware."
            )
        return dt

    def _coerce_tz(self, dt: datetime) -> datetime:
        """Convert dt to the buffer's established timezone, if one is known."""
        if self._data_tz is None:
            return dt
        return dt.astimezone(self._data_tz)

    def _last_complete_bar_boundary(self, dt: datetime) -> datetime:
        """
        Return the open timestamp of the last COMPLETE bar before dt.

        The bar currently open (forming) is excluded — it may be incomplete or
        not yet available from the broker.

        Examples (5-minute bars):
            12:00:00  →  11:55:00  (12:00 bar just opened; 11:55 bar is last complete)
            12:03:47  →  11:55:00  (still inside the 12:00 bar)
            11:59:59  →  11:50:00  (still inside the 11:55 bar)
        """
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)

        # Integer arithmetic avoids floating-point drift at exact bar boundaries
        total_seconds = int((dt_utc - epoch).total_seconds())
        total_minutes = total_seconds // 60

        # Index of the bar currently open (may be incomplete)
        current_bar_idx = total_minutes // self.timeframe_minutes
        # Step back one to get the last bar that has fully closed
        last_complete_idx = current_bar_idx - 1

        boundary_utc = epoch + timedelta(minutes=last_complete_idx * self.timeframe_minutes)

        # Return in the buffer's established timezone, or the original tz if not yet known
        target_tz = self._data_tz or (dt.tzinfo if dt.tzinfo else timezone.utc)
        return boundary_utc.astimezone(target_tz)

    def _needs_full_reload(self, current_dt: datetime) -> bool:
        """
        Return True if the buffer must be fully reloaded rather than incrementally updated.

        Triggers when:
        - Buffer is empty
        - Gap between last bar and current boundary exceeds 2 timeframe periods
          (indicates missed bars, reconnect after outage, etc.)
        """
        if self.is_empty():
            return True

        actual_gap_minutes = (current_dt - self.last_bar_time).total_seconds() / 60
        max_gap_minutes = self.timeframe_minutes * 2

        if actual_gap_minutes > max_gap_minutes:
            logger.warning(
                f"[{self.symbol}] Stale buffer: {actual_gap_minutes:.1f}min gap "
                f"(threshold {max_gap_minutes}min) — full reload required"
            )
            return True

        return False

    def _validate_data(self) -> bool:
        """
        Validate buffer integrity.

        Checks:
        1. No NaN values
        2. No duplicate timestamps (dedup should have cleared these)
        3. No sub-timeframe gaps (indicates out-of-order or clock-skew data)

        Large gaps (weekends, holidays) are logged at DEBUG and accepted — enforcing
        a hard upper delta cap causes false failures for instruments with session breaks.
        """
        if self.is_empty():
            return True

        if self.data.isnull().any().any():
            logger.error(f"[{self.symbol}] Validation failed: NaN values present")
            return False

        if self.data.index.duplicated().any():
            logger.error(f"[{self.symbol}] Validation failed: duplicate timestamps present")
            return False

        if len(self.data) >= 2:
            expected_delta = timedelta(minutes=self.timeframe_minutes)
            deltas = self.data.index.to_series().diff().dropna()
            too_small = deltas[deltas < expected_delta]
            if not too_small.empty:
                logger.error(
                    f"[{self.symbol}] Validation failed: sub-timeframe gaps detected "
                    f"at {too_small.index.tolist()}"
                )
                return False

            large_gaps = deltas[deltas > expected_delta]
            for ts, delta in large_gaps.items():
                logger.debug(
                    f"[{self.symbol}] Session gap at {ts}: {delta} "
                    f"(expected {expected_delta}) — treated as normal market closure"
                )

        return True