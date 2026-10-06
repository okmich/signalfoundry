"""Crypto concrete of the broker-neutral :class:`okmich_quant_core.BrokerSession` (LOGGING_CONTRACT §7.4).

* ``broker`` = ``"{exchange_id}:{environment}"``;
* ``account_id`` = the venue's account UID where the profile can read it, otherwise a non-reversible fingerprint of
  the API key (never the key itself);
* ``broker_session_id`` identifies this connection (CCXT version + a random token per session).

CCXT's ``close()`` is async while the Protocol's ``disconnect()`` is sync. The event loop therefore awaits
:meth:`CryptoBrokerSession.aclose` during shutdown; it closes the exchange, PROVES the release (no HTTP session, no
open WebSocket client, no connector) and caches the result. ``disconnect()`` returns that cached proof and is
idempotent. Called before ``aclose`` with no event loop running, it runs ``aclose`` itself; called from inside a
running loop it cannot block on the close, so it reports "not proven" rather than lie.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import ccxt

from .enums import VenueEnvironment

logger = logging.getLogger(__name__)


def is_exchange_closed(exchange) -> bool:
    """True when every network resource CCXT holds for ``exchange`` has been released."""
    return (getattr(exchange, "session", None) is None and not getattr(exchange, "clients", None)
            and getattr(exchange, "tcp_connector", None) is None)


class CryptoBrokerSession:
    """:class:`okmich_quant_core.BrokerSession` adapter wrapping a CCXT async exchange."""

    def __init__(self, exchange, exchange_id: str, environment: VenueEnvironment, account_id: str,
                 session_id: Optional[str] = None):
        self._exchange = exchange
        self._broker = f"{exchange_id}:{environment.value}"
        self._account_id = account_id
        self._session_id = session_id or f"ccxt-{ccxt.__version__}:{uuid.uuid4().hex[:12]}"
        self._disconnected: bool | None = None  # cached proven result (idempotent)

    @property
    def broker(self) -> str:
        return self._broker

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def broker_session_id(self) -> str | None:
        return self._session_id

    async def aclose(self) -> bool:
        """Close the exchange and return whether the release is PROVEN. Idempotent."""
        if self._disconnected is not None:
            return self._disconnected
        try:
            await self._exchange.close()
        except Exception:
            logger.exception("CryptoBrokerSession: error during exchange.close()")
        self._disconnected = is_exchange_closed(self._exchange)
        if not self._disconnected:
            logger.error("CryptoBrokerSession: close could not be proven - CCXT still holds a session or socket")
        return self._disconnected

    def disconnect(self) -> bool:
        """Release the session and return whether it is PROVEN disconnected. Idempotent (see module docstring)."""
        if self._disconnected is not None:
            return self._disconnected
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aclose())
        logger.error("CryptoBrokerSession.disconnect() called inside a running event loop before aclose(); "
                     "await aclose() instead - reporting the disconnect as NOT proven")
        return False
