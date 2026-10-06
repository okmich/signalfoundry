"""Public-data exchange instances built from CCXT alone (no venue profile, no credentials).

The data tools are deliberately venue-agnostic: they use only CCXT's unified methods and work with any exchange CCXT
supports, whether or not the package trades on it.
"""
from typing import Optional

import ccxt.async_support as ccxt_async
import ccxt.pro as ccxtpro

from ..enums import VenueEnvironment
from ..resilience import VenueUnsupportedError


def make_public_exchange(exchange_id: str, environment: VenueEnvironment = VenueEnvironment.LIVE, *,
                         streaming: bool = False, ccxt_options: Optional[dict] = None):
    """A rate-limited CCXT exchange for public data. ``streaming`` needs CCXT's WebSocket (ccxt.pro) class."""
    exchange_id = exchange_id.strip().lower()
    cls = getattr(ccxtpro, exchange_id, None)
    if cls is None:
        if streaming:
            raise VenueUnsupportedError(f"{exchange_id!r} has no WebSocket support in CCXT")
        cls = getattr(ccxt_async, exchange_id, None)
    if cls is None:
        raise VenueUnsupportedError(f"unknown CCXT exchange id {exchange_id!r}")
    exchange = cls({"enableRateLimit": True, "options": dict(ccxt_options or {})})
    if environment is VenueEnvironment.TESTNET:
        if not (exchange.urls or {}).get("test"):
            raise VenueUnsupportedError(f"{exchange_id} has no testnet in CCXT")
        exchange.set_sandbox_mode(True)
    elif environment is VenueEnvironment.DEMO:
        # Demo copies serve LIVE market data; only the account side differs. Use the demo hosts where CCXT has them.
        if hasattr(exchange, "enable_demo_trading"):
            exchange.enable_demo_trading(True)
    return exchange


def require_capability(exchange, capability: str, what: str) -> None:
    if not (exchange.has or {}).get(capability):
        raise VenueUnsupportedError(f"{exchange.id} does not offer {what} through CCXT ({capability} is not supported)")
