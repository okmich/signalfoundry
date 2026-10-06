"""Candle-close detection for streamed OHLCV.

Unlike IB (5-second bars aggregated up), crypto venues stream the target timeframe directly, so nothing is
aggregated here: this module only decides WHEN a candle has closed. Two signals, whichever comes first:

* the venue's own "final" flag, where the profile's exchange class carries it as a 7th field (Bybit ``confirm``);
* rollover - a candle with a newer open timestamp appears, so every older candle is closed.

The last WebSocket update of a candle can be missed, so a detected close is only a TRIGGER: the bar the strategy sees
is always re-read over REST (``feed.base.BarReconciler``). That is also what makes STREAM and POLL produce identical
bars.
"""
from typing import Optional


class CandleCloseDetector:
    """Feed it OHLCV rows as they stream in; it returns the open timestamps of candles that have just closed."""

    def __init__(self, tf_ms: int):
        if tf_ms <= 0:
            raise ValueError(f"tf_ms must be positive (got {tf_ms})")
        self.tf_ms = tf_ms
        self._current_open: Optional[int] = None
        self._reported: Optional[int] = None

    def reset(self) -> None:
        """Forget the forming candle (after a reconnect, the stream restarts from whatever is current)."""
        self._current_open = None

    def feed(self, rows: list) -> list[int]:
        closed: set[int] = set()
        for row in sorted(rows or [], key=lambda r: r[0]):
            open_ms = int(row[0])
            if open_ms % self.tf_ms != 0:
                # A candle off the UTC grid means the venue labels this timeframe differently from core's bar grid.
                raise ValueError(f"candle open {open_ms} is not aligned to the {self.tf_ms} ms UTC grid")
            confirmed = len(row) > 6 and row[6] is True
            if confirmed:
                closed.add(open_ms)
            if self._current_open is None:
                self._current_open = open_ms
            elif open_ms > self._current_open:
                closed.add(self._current_open)
                self._current_open = open_ms
        fresh = sorted(ts for ts in closed if self._reported is None or ts > self._reported)
        if fresh:
            self._reported = fresh[-1]
        return fresh
