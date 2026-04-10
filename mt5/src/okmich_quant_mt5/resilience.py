"""
Error handling and resilience utilities for MT5 operations.

Provides retry logic, exponential backoff, connection monitoring, and error classification.
"""
import logging
import time
from functools import wraps
from typing import Callable, Any, Optional, Set
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


# MT5 return codes that indicate transient errors (safe to retry)
TRANSIENT_ERROR_CODES: Set[int] = {
    mt5.TRADE_RETCODE_REQUOTE,          # Requote - price changed
    mt5.TRADE_RETCODE_CONNECTION,       # Connection lost
    mt5.TRADE_RETCODE_PRICE_CHANGED,    # Price changed during execution
    mt5.TRADE_RETCODE_TIMEOUT,          # Request timeout
    mt5.TRADE_RETCODE_PRICE_OFF,        # Invalid price (may be temporary)
    mt5.TRADE_RETCODE_REJECT,           # Request rejected (broker busy)
    mt5.TRADE_RETCODE_ERROR,            # Common error (may be transient)
}

# MT5 return codes that indicate permanent errors (do not retry)
PERMANENT_ERROR_CODES: Set[int] = {
    mt5.TRADE_RETCODE_INVALID,          # Invalid request
    mt5.TRADE_RETCODE_INVALID_VOLUME,   # Invalid volume
    mt5.TRADE_RETCODE_INVALID_PRICE,    # Invalid price
    mt5.TRADE_RETCODE_INVALID_STOPS,    # Invalid stops
    mt5.TRADE_RETCODE_TRADE_DISABLED,   # Trade is disabled
    mt5.TRADE_RETCODE_MARKET_CLOSED,    # Market is closed
    mt5.TRADE_RETCODE_NO_MONEY,         # Not enough money
    mt5.TRADE_RETCODE_FROZEN,           # Frozen (trade operations disabled)
    mt5.TRADE_RETCODE_INVALID_FILL,     # Invalid fill mode
    mt5.TRADE_RETCODE_INVALID_ORDER,    # Invalid order
    mt5.TRADE_RETCODE_INVALID_EXPIRATION, # Invalid expiration
    mt5.TRADE_RETCODE_LONG_ONLY,        # Only long positions allowed
    mt5.TRADE_RETCODE_SHORT_ONLY,       # Only short positions allowed
    mt5.TRADE_RETCODE_CLOSE_ONLY,       # Only close positions allowed
    mt5.TRADE_RETCODE_FIFO_CLOSE,       # FIFO rule violation
}


class MT5ConnectionError(Exception):
    """Raised when MT5 connection is lost."""
    pass


class MT5TransientError(Exception):
    """Raised for transient MT5 errors that can be retried."""
    def __init__(self, message: str, retcode: int):
        super().__init__(message)
        self.retcode = retcode


class MT5PermanentError(Exception):
    """Raised for permanent MT5 errors that should not be retried."""
    def __init__(self, message: str, retcode: int):
        super().__init__(message)
        self.retcode = retcode


def is_mt5_connected() -> bool:
    """
    Check if MT5 terminal is connected and responsive.

    Returns:
        True if connected, False otherwise
    """
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        return False

    # Check if terminal is connected to the trade server
    return terminal_info.connected


def classify_mt5_error(retcode: int) -> str:
    """
    Classify MT5 error as transient or permanent.

    Args:
        retcode: MT5 return code

    Returns:
        'transient', 'permanent', or 'unknown'
    """
    if retcode in TRANSIENT_ERROR_CODES:
        return 'transient'
    elif retcode in PERMANENT_ERROR_CODES:
        return 'permanent'
    else:
        return 'unknown'


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    retry_on_connection_loss: bool = True,
):
    """
    Decorator that adds retry logic with exponential backoff to MT5 operations.

    Automatically retries on:
    - Transient MT5 errors (requote, connection, timeout, etc.)
    - Connection loss (if retry_on_connection_loss=True)

    Does NOT retry on:
    - Permanent errors (invalid parameters, insufficient funds, etc.)
    - Successful operations

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        backoff_factor: Multiplier for delay after each retry (default: 2.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
        retry_on_connection_loss: Whether to retry on connection loss (default: True)

    Example:
        @with_retry(max_retries=3, initial_delay=1.0)
        def open_position(...):
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise MT5TransientError("Order failed", result.retcode)
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = min(initial_delay, max_delay)
            last_exception = None

            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Check connection before each attempt
                    if retry_on_connection_loss and not is_mt5_connected():
                        raise MT5ConnectionError("MT5 connection lost")

                    # Execute function
                    return func(*args, **kwargs)

                except MT5TransientError as e:
                    last_exception = e
                    if attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries. "
                            f"Last error: {e} (retcode: {e.retcode})"
                        )
                        raise

                    error_type = classify_mt5_error(e.retcode)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} failed "
                        f"with {error_type} error (retcode: {e.retcode}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

                except MT5ConnectionError as e:
                    last_exception = e
                    if not retry_on_connection_loss or attempt >= max_retries:
                        logger.error(
                            f"{func.__name__} failed due to connection loss after "
                            f"{attempt} attempts"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} connection lost on attempt {attempt + 1}/"
                        f"{max_retries + 1}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

                except MT5PermanentError as e:
                    # Don't retry permanent errors
                    logger.error(
                        f"{func.__name__} failed with permanent error (retcode: {e.retcode}): {e}"
                    )
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


class ConnectionMonitor:
    """
    Monitor MT5 connection health and trigger reconnection when needed.

    Usage:
        monitor = ConnectionMonitor(check_interval=60.0)

        # In trading loop
        if not monitor.check_connection():
            logger.error("MT5 connection unhealthy")
    """

    def __init__(
        self,
        check_interval: float = 60.0,
        reconnect_callback: Optional[Callable[[], bool]] = None
    ):
        """
        Initialize connection monitor.

        Args:
            check_interval: Minimum seconds between connection checks
            reconnect_callback: Optional function to call when reconnection is needed.
                               Should return True if reconnection successful.
        """
        self.check_interval = check_interval
        self.reconnect_callback = reconnect_callback
        self.last_check_time = 0.0
        self.consecutive_failures = 0
        self.is_healthy = True

    def check_connection(self, force: bool = False) -> bool:
        """
        Check if MT5 connection is healthy.

        Args:
            force: Force check even if check_interval hasn't elapsed

        Returns:
            True if connection is healthy, False otherwise
        """
        current_time = time.time()

        # Rate limit checks unless forced
        if not force and (current_time - self.last_check_time) < self.check_interval:
            return self.is_healthy

        self.last_check_time = current_time

        # Check connection
        if is_mt5_connected():
            if not self.is_healthy:
                logger.info("MT5 connection restored")
            self.is_healthy = True
            self.consecutive_failures = 0
            return True
        else:
            self.consecutive_failures += 1
            logger.warning(
                f"MT5 connection unhealthy (consecutive failures: {self.consecutive_failures})"
            )

            # Attempt reconnection if callback provided
            if self.reconnect_callback:
                try:
                    logger.info("Attempting to reconnect MT5...")
                    if self.reconnect_callback():
                        logger.info("MT5 reconnection successful")
                        self.is_healthy = True
                        self.consecutive_failures = 0
                        return True
                    else:
                        logger.error("MT5 reconnection failed")
                except Exception as e:
                    logger.error(f"MT5 reconnection error: {e}")

            self.is_healthy = False
            return False

    def get_health_status(self) -> dict:
        """
        Get detailed connection health status.

        Returns:
            Dict with health metrics
        """
        return {
            'is_healthy': self.is_healthy,
            'consecutive_failures': self.consecutive_failures,
            'last_check_time': self.last_check_time,
            'is_connected': is_mt5_connected(),
        }