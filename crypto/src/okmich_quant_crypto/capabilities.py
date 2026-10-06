"""The compatibility gate: what a strategy needs from a venue, checked before anything is subscribed or ordered.

"Compatible CCXT exchange" means: passes :func:`resolve_capabilities` for the strategy's configuration. Missing a
REQUIRED capability fails fast, listing everything missing at once. Missing an OPTIONAL one degrades explicitly and
says so in the log: no WebSocket bars -> REST polling for bars; no WebSocket account streams -> REST polling for
orders / fills / positions; no native stop that can protect an existing position -> managed stops (AUTO only).
"""
import logging
from dataclasses import dataclass

from .config import CryptoStrategyConfig, check_managed_stop_latency
from .enums import FeedMode, MarketType, StopMode
from .resilience import VenueUnsupportedError
from .venue.base import StopCapabilities, VenueProfile

logger = logging.getLogger(__name__)

REQUIRED_ALWAYS = ("fetchOHLCV", "createOrder", "cancelOrder", "fetchOpenOrders", "fetchBalance", "fetchTicker",
                   "fetchMyTrades")
REQUIRED_PERP = ("fetchPositions",)
STREAM_ACCOUNT = ("watchOrders", "watchMyTrades")
STREAM_ACCOUNT_PERP = ("watchPositions",)


@dataclass(frozen=True)
class ResolvedCapabilities:
    """The concrete plan for one strategy on one venue."""
    stop_mode: StopMode
    stop_caps: StopCapabilities
    stream_bars: bool
    stream_ticker: bool
    stream_account: bool
    ohlcv_limit: int

    def describe(self) -> str:
        return (f"stop_mode={self.stop_mode.value} bars={'stream' if self.stream_bars else 'poll'} "
                f"ticker={'stream' if self.stream_ticker else 'poll'} "
                f"account={'stream' if self.stream_account else 'poll'} ohlcv_limit={self.ohlcv_limit}")


def resolve_capabilities(exchange, profile: VenueProfile, cfg: CryptoStrategyConfig) -> ResolvedCapabilities:
    has = exchange.has or {}
    required = list(REQUIRED_ALWAYS)
    if cfg.market_type is MarketType.LINEAR_PERP:
        required += list(REQUIRED_PERP)
        if cfg.leverage is not None:
            required.append("setLeverage")
    missing = [cap for cap in required if not has.get(cap)]
    if missing:
        raise VenueUnsupportedError(f"{profile.exchange_id} cannot run {cfg.name!r} ({cfg.market_type.value}): CCXT "
                                    f"reports these required capabilities missing: {missing}")

    stop_caps = profile.stop_capabilities(exchange, cfg.market_type)
    stop_mode = _resolve_stop_mode(cfg, stop_caps, profile)

    stream = cfg.feed_mode is FeedMode.STREAM
    stream_bars = stream and bool(has.get("watchOHLCV"))
    account_caps = STREAM_ACCOUNT + (STREAM_ACCOUNT_PERP if cfg.market_type is MarketType.LINEAR_PERP else ())
    stream_account = stream and all(has.get(cap) for cap in account_caps)
    stream_ticker = stream and bool(has.get("watchTicker"))
    if stream and not stream_bars:
        logger.warning("%s: %s has no watchOHLCV; bars fall back to REST polling", cfg.name, profile.exchange_id)
    if stream and not stream_account:
        logger.warning("%s: %s lacks one of %s; orders/fills/positions fall back to REST polling", cfg.name,
                       profile.exchange_id, list(account_caps))

    resolved = ResolvedCapabilities(stop_mode=stop_mode, stop_caps=stop_caps, stream_bars=stream_bars,
                                    stream_ticker=stream_ticker, stream_account=stream_account,
                                    ohlcv_limit=profile.ohlcv_limit(exchange))
    if stop_mode is StopMode.MANAGED:
        check_managed_stop_latency(cfg)
        if not stream_ticker and cfg.feed_mode is FeedMode.STREAM:
            logger.warning("%s: managed stops will poll the ticker every %.1fs (no watchTicker)", cfg.name,
                           cfg.managed_stop_poll_seconds)
    return resolved


def _resolve_stop_mode(cfg: CryptoStrategyConfig, caps: StopCapabilities, profile: VenueProfile) -> StopMode:
    # Native means "survives this process being down". That needs a way to protect a position that ALREADY exists
    # (after a restart, an adoption, or a market entry the venue cannot attach to); attach-on-entry alone is not
    # enough.
    native_ok = caps.can_protect_after_fill
    if cfg.stop_mode is StopMode.NATIVE:
        if not native_ok:
            raise VenueUnsupportedError(f"{cfg.name}: NATIVE stops requested but {profile.exchange_id} "
                                        f"({cfg.market_type.value}) cannot place a stop on an existing position")
        return StopMode.NATIVE
    if cfg.stop_mode is StopMode.MANAGED:
        return StopMode.MANAGED
    return StopMode.NATIVE if native_ok else StopMode.MANAGED
