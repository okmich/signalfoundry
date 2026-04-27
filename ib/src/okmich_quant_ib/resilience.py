"""IB error hierarchy, classification, retry decorator, and connection monitor."""
import asyncio
import logging
import time
from enum import StrEnum
from functools import wraps
from typing import Any, Callable, Coroutine, Optional

from ib_async import IB

logger = logging.getLogger(__name__)


class IBConnectionError(Exception):
    """Connection to Gateway lost or never established."""


class IBTransientError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


class IBPermanentError(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


TRANSIENT_CODES: set[int] = {
    1100, 1101, 1102, 2110, 10225, 504, 162,
}

PERMANENT_CODES: set[int] = {
    200, 201, 202, 203, 321, 10147, 10148,
}


class ErrorClass(StrEnum):
    WARNING = "warning"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


def classify_ib_error(code: int) -> ErrorClass:
    if code in TRANSIENT_CODES:
        return ErrorClass.TRANSIENT
    if code in PERMANENT_CODES:
        return ErrorClass.PERMANENT
    if 2000 <= code <= 2999:
        return ErrorClass.WARNING
    return ErrorClass.UNKNOWN


def is_ib_connected(ib: IB) -> bool:
    return ib.isConnected()


def with_retry(max_retries: int = 3, initial_delay: float = 1.0,
               backoff_factor: float = 2.0, max_delay: float = 30.0) -> Callable:
    """Async retry decorator. Retries on IBTransientError / IBConnectionError; re-raises IBPermanentError immediately."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = min(initial_delay, max_delay)
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except IBTransientError as e:
                    if attempt >= max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    logger.warning(f"{func.__name__} transient error {e.code}: retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except IBConnectionError as e:
                    if attempt >= max_retries:
                        raise
                    logger.warning(f"{func.__name__} connection lost: retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except IBPermanentError:
                    raise
        return wrapper
    return decorator


class ConnectionMonitor:
    """Reactive connection health tracker."""

    def __init__(self, ib: IB, check_interval: float = 60.0,
                 reconnect_callback: Optional[Callable[[], Coroutine[Any, Any, None]]] = None):
        self.ib = ib
        self.check_interval = check_interval
        self.reconnect_callback = reconnect_callback
        self.last_check_time = 0.0
        self.consecutive_failures = 0
        self.is_healthy = True
        ib.disconnectedEvent += self._on_disconnect

    async def _on_disconnect(self):
        self.is_healthy = False
        logger.warning("IB disconnected (event)")
        if self.reconnect_callback:
            await self.reconnect_callback()

    def check_connection(self, force: bool = False) -> bool:
        current = time.time()
        if not force and (current - self.last_check_time) < self.check_interval:
            return self.is_healthy
        self.last_check_time = current
        if is_ib_connected(self.ib):
            if not self.is_healthy:
                logger.info("IB connection restored")
            self.is_healthy = True
            self.consecutive_failures = 0
            return True
        self.consecutive_failures += 1
        self.is_healthy = False
        return False

    def get_health_status(self) -> dict:
        return {
            "is_healthy": self.is_healthy,
            "consecutive_failures": self.consecutive_failures,
            "last_check_time": self.last_check_time,
            "is_connected": is_ib_connected(self.ib),
        }
