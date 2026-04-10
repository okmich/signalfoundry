import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable

logger = logging.getLogger(__name__)


class _AsyncDispatcher:
    """
    Internal background-thread dispatcher. Puts messages on a bounded queue and delivers them via *send_fn* from a
    single daemon thread so the caller never blocks.

    Queue backpressure: if the queue is full (delivery is stalled, e.g. network
    outage), the *oldest* pending message is dropped and a warning is logged to
    prevent unbounded memory growth.  The default capacity (``maxsize``) is 200
    messages, enough for several minutes of high-frequency alerts.
    """

    DEFAULT_MAXSIZE = 200

    def __init__(self, send_fn: Callable[[str], bool], maxsize: int = DEFAULT_MAXSIZE):
        self._send_fn = send_fn
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            message = self._queue.get()
            if message is None:  # sentinel — stop worker
                self._queue.task_done()
                break
            try:
                self._send_fn(message)
            except Exception as e:
                logger.warning(f"Notification delivery failed: {e}")
            finally:
                self._queue.task_done()

    def dispatch(self, message: str):
        """Non-blocking: enqueue *message* and return immediately.

        If the queue is full, the oldest message is dropped and a warning is
        logged.  This prevents unbounded memory growth during network outages.
        """
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.warning(
                    f"Notification queue full (maxsize={self._queue.maxsize}): "
                    f"dropped oldest message to make room. "
                    f"Dropped: {str(dropped)[:80]!r}"
                )
            except queue.Empty:
                pass
            # Retry once after making room
            try:
                self._queue.put_nowait(message)
            except queue.Full:
                logger.warning(
                    f"Notification queue still full after drop — message discarded: "
                    f"{str(message)[:80]!r}"
                )

    def flush(self, timeout: float = 5.0):
        """Block until every queued message has been delivered."""
        self._queue.join()

    def close(self):
        """Drain the queue then stop the worker thread."""
        self._queue.put(None)  # sentinel
        self._thread.join()


class BaseNotifier(ABC):
    """
    Abstract notification interface. Concrete implementations own an _AsyncDispatcher so every on_xxx() call is
    non-blocking relative to the trading loop.

    Lifecycle:
        - Instantiate once per strategy.
        - Call close() (or rely on BaseStrategy.cleanup()) on shutdown to flush
          any queued messages before the process exits.
    """

    @abstractmethod
    def on_trade_opened(self, symbol: str, direction: str, volume: float, price: float,
                        sl: float, tp: float, magic: int, ticket: int):
        ...

    @abstractmethod
    def on_trade_closed(self, symbol: str, ticket: int, profit: float):
        ...

    @abstractmethod
    def on_trade_modified(self, symbol: str, ticket: int, sl: float, tp: float):
        ...

    @abstractmethod
    def on_error(self, strategy_name: str, error_message: str, context: dict = None):
        ...

    @abstractmethod
    def on_circuit_breaker_tripped(self, strategy_name: str, consecutive_errors: int):
        ...

    @abstractmethod
    def on_connection_lost(self, strategy_name: str):
        ...

    @abstractmethod
    def on_connection_restored(self, strategy_name: str):
        ...

    @abstractmethod
    def close(self):
        """Flush pending messages and release resources."""
        ...