"""The supported-exchange allowlist.

Trading runs ONLY on the exchanges listed in ``_SUPPORTED``. Each entry is a hand-written venue profile that was
added on its own feature branch together with integration tests run on the venue's demo / testnet (see "Adding an
exchange" in ``crypto/README.md``). There is deliberately no "any CCXT exchange" mode: CCXT's metadata for a venue is a
starting point for writing its profile, not something to trade on.

Read-only tools (``fetch-crypto-data``) may use :func:`get_profile` for any CCXT exchange: it falls back to the base
``VenueProfile`` for public data. Everything that trades goes through :func:`resolve_profile`, which enforces the list.
"""
from ..resilience import VenueUnsupportedError
from .base import VenueProfile
from .bybit import BybitProfile

_SUPPORTED: dict[str, type[VenueProfile]] = {
    "bybit": BybitProfile,
}


def supported_venues() -> list[str]:
    return sorted(_SUPPORTED)


def is_supported(exchange_id: str) -> bool:
    return exchange_id.strip().lower() in _SUPPORTED


def resolve_profile(exchange_id: str) -> VenueProfile:
    """The profile of a SUPPORTED exchange. Raises :class:`VenueUnsupportedError` for anything else."""
    exchange_id = exchange_id.strip().lower()
    cls = _SUPPORTED.get(exchange_id)
    if cls is None:
        raise VenueUnsupportedError(f"{exchange_id!r} is not a supported exchange (supported: {supported_venues()}). "
                                    f"A new exchange is added on its own feature branch: venue profile, registry "
                                    f"entry, integration tests.")
    return cls(exchange_id)


def get_profile(exchange_id: str) -> VenueProfile:
    """For READ-ONLY use (public market data): the supported profile, or the base profile for any CCXT exchange."""
    exchange_id = exchange_id.strip().lower()
    cls = _SUPPORTED.get(exchange_id)
    return cls(exchange_id) if cls is not None else VenueProfile(exchange_id)


def register_profile(exchange_id: str, profile_cls: type[VenueProfile]) -> None:
    """Add a venue to the supported list (used by a new venue's module, and by test doubles)."""
    _SUPPORTED[exchange_id.strip().lower()] = profile_cls
