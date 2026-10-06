"""Thin async helpers over a CCXT exchange - the crypto counterpart of ``functions/ib.py``.

Order placement is the one place retries are dangerous: a request that timed out may already have created the order.
:func:`place_order_idempotent` therefore never resends blindly. Every attempt reuses the SAME client order id; after
an ambiguous failure the venue is asked for that id first, and a "duplicate id" rejection is treated as "it exists,
go and find it".
"""
import asyncio
import logging
import os
from typing import Any, Optional

from ..client_order_id import is_ours
from ..config import CryptoVenueConfig
from ..markets import MarketSpec
from ..models import Credentials
from ..resilience import (
    CryptoBannedError, CryptoPermanentError, CryptoTransientError, ErrorClass, OrderStateUnknownError,
    call_with_retry, classify_ccxt_error,
)

logger = logging.getLogger(__name__)


# ===== Connection =====

def load_credentials(venue: CryptoVenueConfig, environ: Optional[dict] = None) -> Credentials:
    """Read credentials from the environment variables NAMED in the venue config (see ``core.env_loader``)."""
    env = environ if environ is not None else os.environ
    return Credentials(api_key=env.get(venue.api_key_env) or None, secret=env.get(venue.secret_env) or None,
                       password=(env.get(venue.password_env) or None) if venue.password_env else None)


async def connect_exchange(profile, venue: CryptoVenueConfig, credentials: Credentials):
    """Build the exchange for ``venue`` via its profile, load markets and sync the clock. Closes it on failure."""
    exchange = profile.build_exchange(venue.environment, credentials, rate_limit_ms=venue.rate_limit_ms,
                                      ccxt_options=venue.ccxt_options)
    try:
        await call_with_retry(exchange.load_markets, what=f"{venue.exchange_id} load_markets", profile=profile)
        if exchange.options.get("adjustForTimeDifference") and hasattr(exchange, "load_time_difference"):
            await call_with_retry(exchange.load_time_difference, what=f"{venue.exchange_id} time sync", profile=profile)
        return exchange
    except Exception:
        try:
            await exchange.close()
        except Exception:
            pass
        raise


async def resync_clock(exchange) -> None:
    if hasattr(exchange, "load_time_difference"):
        await exchange.load_time_difference()


# ===== Market data =====

async def fetch_tick_info(exchange, symbol: str) -> dict[str, float]:
    """Bid/ask/last snapshot in the shape the filters expect (parity with IB's ``_current_tick_info``)."""
    ticker = await exchange.fetch_ticker(symbol)
    return ticker_to_tick_info(ticker)


def ticker_to_tick_info(ticker: dict) -> dict[str, float]:
    return {"bid": float(ticker.get("bid") or 0.0), "ask": float(ticker.get("ask") or 0.0),
            "last": float(ticker.get("last") or 0.0), "bid_size": float(ticker.get("bidVolume") or 0.0),
            "ask_size": float(ticker.get("askVolume") or 0.0),
            "mark": float(ticker.get("markPrice") or 0.0), "index": float(ticker.get("indexPrice") or 0.0)}


async def fetch_quote_equity(exchange, currency: str) -> Optional[float]:
    """Account equity in ``currency`` (total balance incl. unrealised P&L where the venue reports it)."""
    balance = await exchange.fetch_balance()
    total = (balance.get("total") or {}).get(currency)
    return float(total) if total is not None else None


async def fetch_free_balance(exchange, currency: str) -> float:
    balance = await exchange.fetch_balance()
    return float((balance.get("free") or {}).get(currency) or 0.0)


# ===== Orders =====

async def place_order_idempotent(exchange, profile, spec: MarketSpec, order_type: str, side: str, amount: float,
                                 price: Optional[float], params: dict, client_order_id: str, *,
                                 max_attempts: int = 3, lookup_attempts: int = 3) -> dict:
    """Create an order exactly once, or raise.

    Raises ``CryptoPermanentError`` (rejected), ``CryptoBannedError`` (stop trading), or ``OrderStateUnknownError``
    (could not confirm either way - the order MAY be live; the caller must reconcile, never just resend).
    """
    profile.client_id_rule.check(client_order_id)
    params = {**params, "clientOrderId": client_order_id}
    delay = 0.5
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await exchange.create_order(spec.symbol, order_type, side, amount, price, params)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_exc = exc
            error_class = classify_ccxt_error(exc, profile, placing_order=True)
            if error_class is ErrorClass.BANNED:
                raise CryptoBannedError(f"{spec.symbol} order {client_order_id}: {exc}", exc) from exc
            if error_class in (ErrorClass.UNKNOWN_STATE, ErrorClass.DUPLICATE):
                existing = await _lookup(exchange, profile, spec, client_order_id, lookup_attempts)
                if existing is not None:
                    logger.warning("%s: order %s was placed despite %s; using the venue's record", spec.symbol,
                                   client_order_id, type(exc).__name__)
                    return existing
                if error_class is ErrorClass.DUPLICATE:
                    raise OrderStateUnknownError(f"{spec.symbol}: venue reports client id {client_order_id} as a "
                                                 f"duplicate but no such order can be found", client_order_id,
                                                 exc) from exc
            elif error_class is ErrorClass.CLOCK_SKEW:
                try:
                    await resync_clock(exchange)
                except Exception:
                    logger.exception("clock resync failed")
            elif error_class is not ErrorClass.TRANSIENT:
                raise CryptoPermanentError(f"{spec.symbol} {side} {order_type} {amount}: {type(exc).__name__}: {exc}",
                                           exc) from exc
            if attempt < max_attempts:
                logger.warning("%s: placement of %s failed (%s, %s); retrying with the same client id", spec.symbol,
                               client_order_id, type(exc).__name__, error_class.value)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
    if last_exc is not None and classify_ccxt_error(last_exc, profile, placing_order=True) is ErrorClass.TRANSIENT:
        raise CryptoTransientError(f"{spec.symbol}: order {client_order_id} rejected after {max_attempts} attempts: "
                                   f"{last_exc}", last_exc)
    raise OrderStateUnknownError(f"{spec.symbol}: order {client_order_id} could not be confirmed after "
                                 f"{max_attempts} attempts", client_order_id, last_exc)


async def _lookup(exchange, profile, spec: MarketSpec, client_order_id: str, attempts: int) -> Optional[dict]:
    delay = 0.5
    for attempt in range(attempts):
        try:
            return await profile.find_order_by_client_id(exchange, spec, client_order_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("%s: lookup of order %s failed (%s)", spec.symbol, client_order_id, exc)
            await asyncio.sleep(delay)
            delay *= 2
    return None


async def cancel_order_safe(exchange, profile, spec: MarketSpec, order_id: str, params: Optional[dict] = None) -> bool:
    """Cancel; True when the order is gone afterwards (cancelled now, or already filled / cancelled / unknown)."""
    try:
        await call_with_retry(lambda: exchange.cancel_order(order_id, spec.symbol, params or {}),
                              what=f"cancel {order_id}", profile=profile)
        return True
    except Exception as exc:
        error_class = classify_ccxt_error(exc, profile)
        if error_class in (ErrorClass.NOT_FOUND, ErrorClass.ALREADY_DONE):
            return True
        logger.error("%s: cancel of %s failed: %s", spec.symbol, order_id, exc)
        return False


async def get_open_orders(exchange, spec: MarketSpec, magic: int, profile, params: Optional[dict] = None) -> list[dict]:
    """This strategy's open orders on ``spec`` (by client-order-id prefix)."""
    orders = await call_with_retry(lambda: exchange.fetch_open_orders(spec.symbol, None, None, params or {}),
                                   what=f"{spec.symbol} open orders", profile=profile)
    return [o for o in orders or [] if is_ours(profile.client_order_id_of(o), magic)]


def order_summary(order: dict[str, Any]) -> str:
    return (f"{order.get('side')} {order.get('type')} {order.get('amount')} @ {order.get('price')} "
            f"[{order.get('status')}] id={order.get('id')}")
