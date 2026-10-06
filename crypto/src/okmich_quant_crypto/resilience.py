"""CCXT exception classification, retry, and the package's own error types.

CCXT's exception tree is close to what a trading loop needs but not exact, and the gaps are where real money leaks:

* ``ExchangeNotAvailable`` is a ``NetworkError`` - yet a non-JSON HTTP 403/451 (IP ban, geo-block) lands there too.
  A naive "retry every NetworkError" loop hammers a ban. Those are classified ``BANNED``.
* ``RequestTimeout`` on an order placement does not mean the order failed. It means its state is UNKNOWN and must be
  looked up by client order id before anything is resent.
* ``RateLimitExceeded`` is a sibling of ``DDoSProtection``, not a subclass; both are retried with backoff.
* Venues report "already done" / "not modified" / "duplicate id" as generic ``InvalidOrder`` or ``BadRequest``. The
  venue profile maps those codes first (``VenueProfile.classify_venue_error``), before the generic tree below.

Classification is most-specific-first: the isinstance chain below is ordered so a subclass is always tested before
its parent.
"""
import asyncio
import logging
import re
from enum import StrEnum
from functools import wraps
from typing import Any, Callable, Optional

from ccxt.base import errors as ccxt_errors

logger = logging.getLogger(__name__)


class ErrorClass(StrEnum):
    TRANSIENT = "transient"            # retry with backoff; the request had no effect
    PERMANENT = "permanent"            # do not retry; the request is wrong or not allowed
    UNKNOWN_STATE = "unknown_state"    # an order placement may or may not have happened; look it up
    BANNED = "banned"                  # IP ban / geo-block / auth revoked mid-run; stop and alert, never hammer
    NO_CHANGE = "no_change"            # the requested state is already in place; treat as success
    NOT_FOUND = "not_found"            # the order is not there (cancel: already gone)
    ALREADY_DONE = "already_done"      # the order is already filled or cancelled
    DUPLICATE = "duplicate"            # the client order id already exists; look it up instead of resending
    CLOCK_SKEW = "clock_skew"          # timestamp outside the venue's receive window; resync the clock, then retry


class CryptoError(Exception):
    """Base of this package's errors. ``error_class`` says how the caller should react."""

    def __init__(self, message: str, error_class: ErrorClass = ErrorClass.PERMANENT,
                 cause: BaseException | None = None):
        super().__init__(message)
        self.error_class = error_class
        self.cause = cause


class CryptoTransientError(CryptoError):
    def __init__(self, message: str, cause: BaseException | None = None):
        super().__init__(message, ErrorClass.TRANSIENT, cause)


class CryptoPermanentError(CryptoError):
    def __init__(self, message: str, cause: BaseException | None = None):
        super().__init__(message, ErrorClass.PERMANENT, cause)


class CryptoBannedError(CryptoError):
    def __init__(self, message: str, cause: BaseException | None = None):
        super().__init__(message, ErrorClass.BANNED, cause)


class OrderStateUnknownError(CryptoError):
    """A placement could not be confirmed either way after every lookup. The order MAY be live on the venue."""

    def __init__(self, message: str, client_order_id: str, cause: BaseException | None = None):
        super().__init__(message, ErrorClass.UNKNOWN_STATE, cause)
        self.client_order_id = client_order_id


class VenueUnsupportedError(CryptoPermanentError):
    """The venue (or its CCXT implementation) lacks a capability this configuration needs."""


#: HTTP statuses that mean "this IP / region is refused", carried in CCXT's message as `` <code> <reason> ``.
_BAN_STATUS_RE = re.compile(r"\s(403|418|451)\s")


def classify_ccxt_error(exc: BaseException, profile: Any = None, *, placing_order: bool = False) -> ErrorClass:
    """Map an exception to the reaction the caller should take.

    ``profile`` (a ``VenueProfile``) gets the first word for venue-specific codes. ``placing_order`` turns ambiguous
    transport failures into ``UNKNOWN_STATE``: a request that died in flight may already have created the order.
    """
    if isinstance(exc, CryptoError):
        return exc.error_class
    if profile is not None:
        venue_class = profile.classify_venue_error(exc)
        if venue_class is not None:
            return venue_class
    e = ccxt_errors
    if isinstance(exc, (e.NoChange, e.MarginModeAlreadySet)):
        return ErrorClass.NO_CHANGE
    if isinstance(exc, e.OrderNotFound):
        return ErrorClass.NOT_FOUND
    if isinstance(exc, e.DuplicateOrderId):
        return ErrorClass.DUPLICATE
    if isinstance(exc, e.InvalidNonce):
        return ErrorClass.CLOCK_SKEW
    if isinstance(exc, (e.RateLimitExceeded, e.DDoSProtection)):
        if _BAN_STATUS_RE.search(str(exc)):
            return ErrorClass.BANNED
        return ErrorClass.TRANSIENT
    if isinstance(exc, e.RequestTimeout):
        return ErrorClass.UNKNOWN_STATE if placing_order else ErrorClass.TRANSIENT
    if isinstance(exc, e.ExchangeNotAvailable):
        if _BAN_STATUS_RE.search(str(exc)):
            return ErrorClass.BANNED
        return ErrorClass.UNKNOWN_STATE if placing_order else ErrorClass.TRANSIENT
    if isinstance(exc, e.NetworkError):
        return ErrorClass.UNKNOWN_STATE if placing_order else ErrorClass.TRANSIENT
    if isinstance(exc, (e.AccountSuspended, e.PermissionDenied, e.AuthenticationError)):
        return ErrorClass.BANNED if isinstance(exc, e.AccountSuspended) else ErrorClass.PERMANENT
    if isinstance(exc, (e.InsufficientFunds, e.BadSymbol, e.BadRequest, e.ArgumentsRequired, e.InvalidOrder,
                        e.NotSupported, e.OperationRejected, e.ExchangeError)):
        return ErrorClass.PERMANENT
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError)):
        return ErrorClass.UNKNOWN_STATE if placing_order else ErrorClass.TRANSIENT
    return ErrorClass.PERMANENT


def wrap_ccxt_error(exc: BaseException, context: str, profile: Any = None) -> CryptoError:
    """Convert a CCXT exception into this package's error type, keeping the original as ``cause``."""
    if isinstance(exc, CryptoError):
        return exc
    error_class = classify_ccxt_error(exc, profile)
    message = f"{context}: {type(exc).__name__}: {exc}"
    if error_class is ErrorClass.BANNED:
        return CryptoBannedError(message, exc)
    if error_class in (ErrorClass.TRANSIENT, ErrorClass.CLOCK_SKEW):
        return CryptoTransientError(message, exc)
    return CryptoError(message, error_class, exc)


def with_retry(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0, max_delay: float = 30.0,
               profile_attr: Optional[str] = None) -> Callable:
    """Async retry decorator for IDEMPOTENT reads only (never for order placement - see ``functions.crypto``).

    Retries ``TRANSIENT`` and ``CLOCK_SKEW`` failures with exponential backoff; everything else is re-raised at once.
    ``profile_attr`` names an attribute of the first positional argument that holds the venue profile, if any.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = min(initial_delay, max_delay)
            profile = getattr(args[0], profile_attr, None) if (profile_attr and args) else None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    error_class = classify_ccxt_error(exc, profile)
                    if error_class not in (ErrorClass.TRANSIENT, ErrorClass.CLOCK_SKEW) or attempt >= max_retries:
                        raise
                    logger.warning("%s: %s (%s), retrying in %.1fs", func.__name__, type(exc).__name__,
                                   error_class.value, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
        return wrapper
    return decorator


async def call_with_retry(coro_factory: Callable, *, what: str, profile: Any = None, max_retries: int = 3,
                          initial_delay: float = 1.0, max_delay: float = 30.0, on_clock_skew: Callable | None = None):
    """Run an idempotent call built by ``coro_factory`` with retry on transient failures. Returns its result."""
    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            error_class = classify_ccxt_error(exc, profile)
            if error_class is ErrorClass.CLOCK_SKEW and on_clock_skew is not None:
                try:
                    await on_clock_skew()
                except Exception:
                    logger.exception("%s: clock resync failed", what)
            if error_class not in (ErrorClass.TRANSIENT, ErrorClass.CLOCK_SKEW) or attempt >= max_retries:
                raise
            logger.warning("%s failed (%s: %s), retry %d/%d in %.1fs", what, type(exc).__name__, error_class.value,
                           attempt + 1, max_retries, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, max_delay)
