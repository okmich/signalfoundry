"""Crypto strategy lifecycle and the ``GenericBasicCryptoStrategy`` execution loop fired on closed bars.

How core's contracts are met without changing core (see ``README.md`` for the full mapping):

* the per-bar heartbeat lives behind the sealed async seam :meth:`BaseCryptoStrategy._on_bar_close`, exactly as IB's
  ``_on_bar_close`` does: Tier 0 ``bar`` record on every LIVE bar, per-strategy circuit breaker, no re-raise;
* a position is keyed by its lifecycle id (``position_id``), so core's close reconciliation can hold an unresolved
  close for ``_CLOSE_RESOLUTION_GRACE_SECONDS`` without the key ever being reused;
* closes are resolved broker-side (fills + funding fetched asynchronously) and handed to core as a finished
  ``ClosedTrade``; funding goes into ``swap`` and fees into ``commission`` (negative), so ``net_profit`` is honest.
"""
import asyncio
import collections
import logging
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional, Union

import pandas as pd

from okmich_quant_core import (
    BarOutcome, BaseSignal, BaseStrategy, ClosedTrade, CloseReason, OrderType, PositionSizingType, StrategyHealth,
)
from okmich_quant_core.notification.base import BaseNotifier
from okmich_quant_core.price_buffer import PriceBuffer

from .capabilities import ResolvedCapabilities, resolve_capabilities
from .client_order_id import make_client_order_id
from .config import CryptoStrategyConfig, CryptoVenueConfig
from .enums import FillKind, MarginModeScope, MarketType, OrderRole, OrderSide, StopMode
from .feed import ClosedBarSource, PollBarSource, StreamBarSource
from .filters import create_filter
from .functions.crypto import (
    cancel_order_safe, fetch_free_balance, fetch_quote_equity, fetch_tick_info, get_open_orders,
    place_order_idempotent, ticker_to_tick_info,
)
from .markets import MarketSpec, OrderSizeError
from .models import ClosedBar, Fill
from .orders import OrderRegistry
from .pnl import last_round_trip, summarize
from .position_cache import CryptoPositionCache, EndedLifecycle
from .position_manager import get_position_manager
from .resilience import CryptoBannedError, CryptoError, OrderStateUnknownError
from .spot_ledger import SpotInventoryLedger
from .state_store import StateStore
from .stops import ManagedStopController, NativeStopController, StopController, close_side
from .timeframe_utils import ms_to_utc, timeframe_to_minutes, utc_now_ms, validate_venue_timeframe
from .venue.base import VenueProfile

logger = logging.getLogger(__name__)

_FILL_REASONS = {FillKind.STOP_LOSS: CloseReason.STOP_LOSS, FillKind.TAKE_PROFIT: CloseReason.TAKE_PROFIT,
                 FillKind.LIQUIDATION: CloseReason.STOP_OUT, FillKind.ADL: CloseReason.STOP_OUT}
_ROLE_REASONS = {OrderRole.STOP_LOSS: CloseReason.STOP_LOSS, OrderRole.TAKE_PROFIT: CloseReason.TAKE_PROFIT,
                 OrderRole.EXIT: CloseReason.STRATEGY, OrderRole.ENTRY: CloseReason.STRATEGY}
_TERMINAL_ORDER_STATUSES = {"canceled", "cancelled", "rejected", "expired"}


@dataclass
class VenueContext:
    """What the event loop hands every strategy at bootstrap."""
    exchange: Any
    profile: VenueProfile
    venue: CryptoVenueConfig
    clock: Callable[[], int] = utc_now_ms
    sleep: Callable = asyncio.sleep


def bars_frame(bars: list[ClosedBar]) -> pd.DataFrame:
    return pd.DataFrame({"open": [b.open for b in bars], "high": [b.high for b in bars], "low": [b.low for b in bars],
                         "close": [b.close for b in bars], "volume": [b.volume for b in bars]},
                        index=pd.DatetimeIndex([b.time for b in bars], name="date"))


def _last(values) -> bool:
    """Last element of a signal series (numpy array, pandas Series or list) as a plain bool."""
    if hasattr(values, "iloc"):
        return bool(values.iloc[-1])
    return bool(values[-1])


class BaseCryptoStrategy(BaseStrategy):
    #: The async per-bar seam is sealed: it owns the inference-log heartbeat (§5.1).
    _CRYPTO_SEALED = frozenset({"_on_bar_close"})
    #: Lifecycle ids are never reused, so an unresolved close may be held while fills / funding settle (see core).
    _CLOSE_RESOLUTION_GRACE_SECONDS: float = 180.0
    #: Fill history is read from this long before a lifecycle's open (see ``pnl.last_round_trip``).
    RESOLUTION_LOOKBACK_MS = 3600 * 1000
    MAX_RESOLUTION_WINDOW_MS = 30 * 24 * 3600 * 1000
    #: A fresh spot ledger adopts our tagged fills from this far back.
    SPOT_LEDGER_LOOKBACK_MS = 7 * 24 * 3600 * 1000
    RESOLUTION_RETRY_SECONDS = 30.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # also runs BaseStrategy's seal (run / _emit_bar_record / bind)
        for name in BaseCryptoStrategy._CRYPTO_SEALED:
            if name in cls.__dict__:
                raise TypeError(f"{cls.__name__} may not override sealed BaseCryptoStrategy.{name}() - the "
                                f"inference-log heartbeat lives there (LOGGING_CONTRACT §5.1). "
                                f"Implement on_new_bar().")

    def __init__(self, config: CryptoStrategyConfig, signal: BaseSignal, notifier: Optional[BaseNotifier] = None, *,
                 max_consecutive_errors: int = 5, **kwargs):
        if not isinstance(config, CryptoStrategyConfig):
            raise TypeError(f"{type(self).__name__} needs a CryptoStrategyConfig (got {type(config).__name__})")
        tf_min = timeframe_to_minutes(config.timeframe)
        super().__init__(config, signal, notifier, timeframe_minutes=tf_min, **kwargs)
        self.strategy_config: CryptoStrategyConfig = config
        self._crypto_health = StrategyHealth(config.name, max_consecutive_errors)
        self.max_number_of_open_positions = config.max_number_of_open_positions
        self.price_buffer = PriceBuffer(symbol=config.symbol, timeframe=config.timeframe,
                                        buffer_size=config.bars_to_copy, timeframe_minutes=tf_min)
        self.filter_chain = create_filter(config)

        self.exchange = None
        self.profile: Optional[VenueProfile] = None
        self.venue: Optional[CryptoVenueConfig] = None
        self.spec: Optional[MarketSpec] = None
        self.caps: Optional[ResolvedCapabilities] = None
        self.store: Optional[StateStore] = None
        self.orders: Optional[OrderRegistry] = None
        self.position_cache: Optional[CryptoPositionCache] = None
        self.spot_ledger: Optional[SpotInventoryLedger] = None
        self.stops: Optional[StopController] = None
        self.bar_source: Optional[ClosedBarSource] = None
        self.position_manager = None
        self._clock: Callable[[], int] = utc_now_ms
        self._sleep: Callable = asyncio.sleep
        self._tasks: list[asyncio.Task] = []
        self._bar_lock = asyncio.Lock()
        self._book_lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._known_sizes: dict[str, float] = {}
        self._resolved_closes: dict[str, ClosedTrade] = {}
        self._seen_fills: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._ticker: Optional[dict] = None
        self._offline_since_ms: Optional[int] = None
        self._warned_no_funding = False

    # ------------------------------------------------------------------ properties
    @property
    def health(self) -> StrategyHealth:
        return self._crypto_health

    @property
    def is_perp(self) -> bool:
        return self.strategy_config.market_type is MarketType.LINEAR_PERP

    def _now_dt(self) -> datetime:
        return ms_to_utc(self._clock())

    # ------------------------------------------------------------------ lifecycle
    async def _bootstrap(self, ctx: VenueContext) -> None:
        cfg = self.strategy_config
        self.exchange, self.profile, self.venue = ctx.exchange, ctx.profile, ctx.venue
        self._clock, self._sleep = ctx.clock, ctx.sleep
        self.spec = MarketSpec.from_market(self.exchange, self.exchange.market(cfg.market_symbol), cfg.market_type)
        validate_venue_timeframe(cfg.timeframe, getattr(self.exchange, "timeframes", None))
        self.caps = resolve_capabilities(self.exchange, self.profile, cfg)
        logger.info("%s: %s on %s/%s - %s", cfg.name, cfg.market_symbol, self.profile.exchange_id,
                    self.venue.environment.value, self.caps.describe())

        self.store = StateStore.open(self.venue.state_dir, self.venue.exchange_id, self.venue.environment.value,
                                     self.venue.sub_account, cfg.name, cfg.market_symbol)
        self.orders = OrderRegistry(self.store)
        if self.is_perp:
            await self.profile.ensure_one_way(self.exchange, self.spec)
            if cfg.leverage is not None:
                await self.profile.apply_leverage(self.exchange, cfg.leverage, self.spec)
            if self.venue.margin_mode is not None and self.profile.margin_mode_scope is MarginModeScope.SYMBOL:
                await self.profile.apply_margin_mode(self.exchange, self.venue.margin_mode, self.spec)
            self.position_cache = CryptoPositionCache(self.spec, cfg.symbol, self.profile, self.store, self._clock)
        else:
            self.spot_ledger = SpotInventoryLedger(self.spec, cfg.symbol, cfg.magic, self.orders, self.store,
                                                   self._clock)

        self.position_manager = get_position_manager(cfg, price_buffer=self.price_buffer, spec=self.spec)
        self.stops = self._make_stop_controller()
        logger.info("%s: %s", cfg.name, self.stops.describe())

        source_cls = StreamBarSource if self.caps.stream_bars else PollBarSource
        self.bar_source = source_cls(self.exchange, cfg.market_symbol, cfg.timeframe, self._on_bar_close,
                                     ohlcv_limit=self.caps.ohlcv_limit, close_grace_seconds=cfg.close_grace_seconds,
                                     close_max_wait_seconds=cfg.close_max_wait_seconds, clock=self._clock,
                                     sleep=self._sleep, on_missed=self._on_missed_bar)
        seed = await self.bar_source.seed(cfg.bars_to_copy + 1)
        if seed:
            self.price_buffer.update(bars_frame(seed), self._now_dt())
        if len(seed) < cfg.bars_to_copy:
            logger.warning("%s: only %d of %d warm-up bars available", cfg.name, len(seed), cfg.bars_to_copy)

        # Startup is an outage of everything: adopt what the venue holds, re-check protection, and (managed stops)
        # scan for crossings since prices were last watched.
        unwatched_since = self.stops.last_price_ms if isinstance(self.stops, ManagedStopController) else None
        await self._sync_book(after_outage=True, price_outage_since=unwatched_since, adopting=True)

    def _make_stop_controller(self) -> StopController:
        cfg = self.strategy_config
        if self.caps.stop_mode is StopMode.NATIVE:
            return NativeStopController(cfg.name, self.spec, self.store, cfg.stop_trigger, exchange=self.exchange,
                                        profile=self.profile, position_level=self.caps.stop_caps.position_level,
                                        submit=self._submit_order, cancel=self._cancel_order,
                                        fetch_open_stop_ids=self._open_stop_order_ids, clock=self._clock,
                                        live_positions=self._venue_positions)
        return ManagedStopController(cfg.name, self.spec, self.store, cfg.stop_trigger, close=self._close_for_stop,
                                     clock=self._clock)

    def start(self, stream_account: bool) -> None:
        """Start the bar feed and this strategy's background loops (called by the event loop after bootstrap)."""
        cfg = self.strategy_config
        self.bar_source.start()
        if self.caps.stop_mode is StopMode.MANAGED:
            self._spawn(self._price_loop(), "price")
            latency = "WebSocket ticker" if self.caps.stream_ticker else f"{cfg.managed_stop_poll_seconds:.1f}s polling"
            logger.warning("%s: managed stop latency = %s (+ order round trip)", cfg.name, latency)
        if stream_account and self.caps.stream_account:
            self._spawn(self._periodic(self._reconcile, cfg.reconcile_seconds), "reconcile")
        else:
            self._spawn(self._periodic(self._reconcile, cfg.position_poll_seconds), "account-poll")

    async def stop(self) -> None:
        if self.bar_source is not None:
            await self.bar_source.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        if isinstance(self.stops, ManagedStopController):
            self.stops.persist_heartbeat()

    def _spawn(self, coro, label: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=f"{self.strategy_config.name}:{label}")
        self._tasks = [t for t in self._tasks if not t.done()]
        self._tasks.append(task)
        return task

    async def _periodic(self, func, interval: float) -> None:
        interval = max(float(interval), 1.0)
        while True:
            await self._sleep(interval)
            try:
                await func()
            except asyncio.CancelledError:
                raise
            except CryptoBannedError as exc:
                self._on_banned(exc)
            except Exception:
                logger.exception("%s: periodic %s failed", self.strategy_config.name, getattr(func, "__name__", func))

    async def on_reconnected(self, offline_since_ms: Optional[int], *, price_outage: bool = False) -> None:
        """After an outage: REST-reconcile the book and re-check protection (and, for a PRICE outage, managed-stop
        crossings while prices went unwatched)."""
        await self._sync_book(after_outage=True, price_outage_since=offline_since_ms if price_outage else None)

    # ------------------------------------------------------------------ the sealed per-bar seam
    async def _on_bar_close(self, bar: ClosedBar, live: bool) -> None:
        """SEALED - the un-bypassable per-bar heartbeat (§5.1), the crypto twin of IB's ``_on_bar_close``.

        Every bar updates the price buffer. Only a LIVE bar runs a cycle and writes a ``bar`` record: a stale or
        backfilled bar (it closed more than ``close_max_wait_seconds`` ago) is buffer-only, so the heartbeat shows the
        outage as a gap instead of pretending the strategy acted on time. No re-raise: the breaker is the bookkeeping
        target, as on IB.
        """
        self._append_bar(bar)
        if not live:
            logger.info("%s: bar %s is stale - buffered, no cycle", self.strategy_config.name, bar.time.isoformat())
            return
        asof_bar_ts = bar.time
        async with self._bar_lock:
            if not self._crypto_health.is_enabled:
                self._emit_bar_record(asof_bar_ts=asof_bar_ts, outcome=BarOutcome.SKIPPED_DISABLED)
                return
            error: Optional[Exception] = None
            try:
                self.latest_run_dt = bar.time
                await self.on_new_bar()
            except Exception as e:
                error = e
                logger.exception("Error in on_new_bar for %s: %s", self.strategy_config.symbol, e)
                if self.notifier:
                    self.notifier.on_error(self.strategy_config.name, str(e))
                if isinstance(e, CryptoBannedError):
                    self._on_banned(e)
            self._emit_bar_record(asof_bar_ts=asof_bar_ts, outcome=BarOutcome.ERROR if error else BarOutcome.OK)
            self._record_bar_health(error)

    def _record_bar_health(self, error: Optional[Exception]) -> None:
        was_enabled = self._crypto_health.is_enabled
        if error is None:
            self._crypto_health.record_success(0.0)
            return
        self._crypto_health.record_error(0.0, last_error=str(error))
        if was_enabled and not self._crypto_health.is_enabled:
            self.emit_circuit_breaker_tripped(consecutive_errors=self._crypto_health.consecutive_errors,
                                              last_error=str(error))
            if self.notifier:
                self.notifier.on_circuit_breaker_tripped(self.strategy_config.name,
                                                         self._crypto_health.consecutive_errors)

    def reenable(self) -> None:
        if not self._crypto_health.is_enabled:
            self._crypto_health.enable()
            self.emit_strategy_reenabled(reason="manual")

    def _on_banned(self, exc: BaseException) -> None:
        """A ban / revoked key: stop trading at once and alert. Native stops stay on the venue."""
        logger.critical("%s: venue refused us (%s) - strategy DISABLED", self.strategy_config.name, exc)
        if self._crypto_health.is_enabled:
            self._crypto_health.disable()
            self.emit_circuit_breaker_tripped(consecutive_errors=self._crypto_health.consecutive_errors,
                                              last_error=f"banned: {exc}")
            if self.notifier:
                self.notifier.on_error(self.strategy_config.name, f"venue refused the account: {exc}")

    def _on_missed_bar(self, open_ms: int) -> None:
        if self.notifier:
            self.notifier.on_error(self.strategy_config.name, f"bar {ms_to_utc(open_ms).isoformat()} not available "
                                                              f"within {self.strategy_config.close_max_wait_seconds}s")

    def _append_bar(self, bar: ClosedBar) -> None:
        self.price_buffer.update(bars_frame([bar]), self._now_dt())

    # ------------------------------------------------------------------ BaseStrategy ABC
    def is_new_bar(self, run_dt) -> bool:
        """Event-driven: bar boundaries come from the feed, never from a polled clock."""
        return False

    @abstractmethod
    async def on_new_bar(self):
        """Read the price buffer, generate signals, manage positions, place orders."""

    # ------------------------------------------------------------------ venue events (from the event loop)
    async def on_venue_order(self, order: dict) -> None:
        cid = self.profile.client_order_id_of(order)
        if not cid or self.orders.role_of(client_order_id=cid) is None:
            return
        self.orders.attach_order_id(cid, order.get("id"))
        status = str(order.get("status") or "").lower()
        if status in _TERMINAL_ORDER_STATUSES and self.orders.role_of(client_order_id=cid) is OrderRole.ENTRY \
                and not float(order.get("filled") or 0):
            logger.warning("%s: entry order %s ended %s unfilled", self.strategy_config.name, cid, status)
            self.stops.clear_pending()

    async def on_venue_fill(self, trade: dict) -> None:
        fill = self.profile.normalize_fill(trade, self.spec)
        if fill.kind in (FillKind.FUNDING, FillKind.SETTLEMENT) or not self._first_sight(fill.trade_id):
            return
        async with self._book_lock:
            if self.is_perp:
                self.position_cache.note_fill_time(fill.timestamp_ms)
            elif self.spot_ledger.owns(fill):
                await self._handle_book_events(*self.spot_ledger.apply_fill(fill))
                await self._resize_stops_if_needed()
            else:
                return
        logger.info("%s: fill %s %.10g @ %s (%s)", self.strategy_config.name, fill.side.value, fill.base_qty,
                    fill.price, fill.kind.value)
        if self.notifier:
            self.notifier.on_trade_filled(symbol=self.strategy_config.symbol, order_id=fill.order_id,
                                          qty=fill.base_qty, avg_price=fill.price)

    async def on_venue_position(self, position: dict) -> None:
        if not self.is_perp:
            return
        async with self._book_lock:
            await self._handle_book_events(*self.position_cache.apply_position(position))
            await self._resize_stops_if_needed()

    async def on_venue_ticker(self, ticker: dict) -> None:
        self._ticker = ticker
        price = self.profile.trigger_price(ticker, self.strategy_config.stop_trigger)
        if self.spot_ledger is not None and ticker.get("last"):
            self.spot_ledger.update_price(float(ticker["last"]))
        if price is None or not isinstance(self.stops, ManagedStopController):
            return
        for position in self.get_open_positions():
            await self.stops.on_price(position, price)

    def _first_sight(self, trade_id: str) -> bool:
        if trade_id in self._seen_fills:
            return False
        self._seen_fills[trade_id] = None
        while len(self._seen_fills) > 5000:
            self._seen_fills.popitem(last=False)
        return True

    # ------------------------------------------------------------------ book reconciliation
    async def _reconcile(self) -> None:
        await self._sync_book(after_outage=False)

    async def _sync_book(self, after_outage: bool, price_outage_since: Optional[int] = None,
                         adopting: bool = False) -> None:
        """REST truth for the book - positions (perp) or our fills (spot) - then protection checks.

        Serialised: three account streams recovering at once, the price loop and the periodic reconcile can all ask
        for a sync, and two concurrent protection passes would place duplicate stop orders. ``price_outage_since`` is
        set only when PRICES went unwatched (process down, price feed lost): only then can a managed stop have been
        crossed unseen. An account-stream outage alone does not blind managed stops.
        """
        async with self._sync_lock:
            async with self._book_lock:
                if self.is_perp:
                    await self._handle_book_events(*await self.position_cache.resync(self.exchange, adopting=adopting))
                else:
                    for opened, ended in await self.spot_ledger.catch_up(self.exchange, self.profile,
                                                                         self.SPOT_LEDGER_LOOKBACK_MS):
                        await self._handle_book_events(opened, ended)
                await self._resize_stops_if_needed()
            for position in self.get_open_positions():
                self.register_open_position(position)
            if after_outage or isinstance(self.stops, NativeStopController):
                # Native stops are verified on every reconcile: a stop lost to anything (a failed close, a manual
                # cancel, a venue-side expiry) is put back within one reconcile interval, not at the next outage.
                for raw in self._venue_positions():
                    await self.stops.ensure_protection(raw)
            if price_outage_since and isinstance(self.stops, ManagedStopController):
                for position in self.get_open_positions():
                    await self.stops.check_offline_window(self.exchange, position, price_outage_since)
            # Re-read: positions may have closed during the awaits above, and observing a stale book would re-track
            # a key whose close was already announced.
            self.observe_open_positions(self.get_open_positions())

    def _venue_positions(self) -> list[dict]:
        """Open positions as the VENUE reports them (stop fields not merged with the controller's levels)."""
        if self.position_cache is not None:
            return self.position_cache.get_open()
        if self.spot_ledger is not None:
            return self.spot_ledger.get_open()
        return []

    async def _resize_stops_if_needed(self) -> None:
        """Standalone stop orders carry a quantity; when the position grows or shrinks they must follow it."""
        if not isinstance(self.stops, NativeStopController) or self.stops.position_level:
            return
        for raw in self._venue_positions():
            size = abs(raw["position"])
            known = self._known_sizes.get(raw["position_id"])
            if known is None:
                self._known_sizes = {raw["position_id"]: size}
            elif abs(known - size) > self._qty_eps():
                logger.info("%s: position %s size %.10g -> %.10g; resizing stop orders", self.strategy_config.name,
                            raw["position_id"], known, size)
                try:
                    await self.stops.ensure_protection(raw)
                    self._known_sizes = {raw["position_id"]: size}  # only once the orders follow it
                except Exception:
                    logger.exception("%s: resizing stops failed; retried on the next update",
                                     self.strategy_config.name)

    async def _handle_book_events(self, opened: Optional[dict], ended: Optional[EndedLifecycle]) -> None:
        if ended is not None:
            await self._on_lifecycle_ended(ended)
        if opened is not None:
            await self._on_position_opened(opened)

    async def _on_position_opened(self, position: dict) -> None:
        cfg = self.strategy_config
        self.register_open_position(position)
        logger.info("%s: position %s opened: %.10g @ %s", cfg.name, position["position_id"], position["position"],
                    position["price_open"])
        try:
            await self.stops.on_position_opened(self._with_stop_levels(position))
            if self.position_manager is not None:
                # Initial levels now, not at the next bar close: a fresh position must not wait a bar for its stop.
                current = [self._with_stop_levels(p) for p in self.get_open_positions()]
                await self.position_manager.manage_positions(current, self._apply_levels, self.close_position)
        except CryptoBannedError as exc:
            self._on_banned(exc)
        except Exception:
            logger.exception("%s: protecting new position %s failed - it may be UNPROTECTED", cfg.name,
                             position["position_id"])
            if self.notifier:
                self.notifier.on_error(cfg.name, f"position {position['position_id']} may be unprotected")

    async def _on_lifecycle_ended(self, ended: EndedLifecycle) -> None:
        # Register first: a position that closed while this process was down was never registered in THIS process,
        # and core only announces closes of positions it is tracking.
        self.register_open_position({**ended.last_seen, "position_id": ended.position_id})
        try:
            await self.stops.on_position_closed(ended.position_id)
        except Exception:
            logger.exception("%s: stop cleanup for %s failed", self.strategy_config.name, ended.position_id)
        self._spawn(self._resolve_and_announce(ended), f"resolve:{ended.position_id}")

    async def _resolve_and_announce(self, ended: EndedLifecycle) -> None:
        attempts = max(1, int(self._CLOSE_RESOLUTION_GRACE_SECONDS // self.RESOLUTION_RETRY_SECONDS))
        for attempt in range(attempts):
            trade = None
            try:
                trade = await self._resolve_lifecycle(ended)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s: resolving %s failed", self.strategy_config.name, ended.position_id)
            if trade is None:
                trade = self._unresolved_trade(ended)
            self._resolved_closes[ended.position_id] = trade
            self.observe_open_positions(self.get_open_positions())
            if trade.resolved or ended.position_id not in self._pending_closes:
                return
            await self._sleep(self.RESOLUTION_RETRY_SECONDS)
        self._resolved_closes.pop(ended.position_id, None)

    async def _resolve_lifecycle(self, ended: EndedLifecycle) -> Optional[ClosedTrade]:
        spot = not self.is_perp
        if spot:
            fills, complete = list(ended.fills), bool(ended.fills)
        else:
            since = max(ended.opened_ms - self.RESOLUTION_LOOKBACK_MS, ended.closed_ms - self.MAX_RESOLUTION_WINDOW_MS)
            history = await self.profile.fetch_fills(self.exchange, self.spec, since, ended.closed_ms)
            trades = [f for f in history if f.kind not in (FillKind.FUNDING, FillKind.SETTLEMENT)]
            fills, complete = last_round_trip(trades, spot=False, eps=self._qty_eps(), final_qty=ended.residual_qty)
        if not complete:
            logger.warning("%s: fills for %s do not form a complete round trip yet", self.strategy_config.name,
                           ended.position_id)
            return None
        expected_long = ended.last_seen.get("type", 0) == 0 if ended.last_seen else None
        if expected_long is not None and (fills[0].side is OrderSide.BUY) != expected_long:
            logger.warning("%s: round trip for %s has the wrong direction", self.strategy_config.name,
                           ended.position_id)
            return None
        book = summarize(fills, spot=spot, eps=self._qty_eps())
        funding = 0.0
        if not spot:
            payments = await self.profile.fetch_funding(self.exchange, self.spec, fills[0].timestamp_ms,
                                                        fills[-1].timestamp_ms)
            if payments is None:
                if not self._warned_no_funding:
                    logger.warning("%s: %s cannot report funding; closed-trade P&L EXCLUDES funding",
                                   self.strategy_config.name, self.profile.exchange_id)
                    self._warned_no_funding = True
            else:
                funding = sum(p.amount for p in payments)
        return ClosedTrade(key=ended.position_id, symbol=self.strategy_config.symbol, magic=self.strategy_config.magic,
                           reason=self._close_reason(fills[-1]), volume=book.closed_qty, entry_price=book.avg_entry,
                           exit_price=book.avg_exit, profit=book.realized, commission=-book.fees_quote, swap=funding,
                           opened_at=ms_to_utc(fills[0].timestamp_ms), closed_at=ms_to_utc(fills[-1].timestamp_ms),
                           last_seen=dict(ended.last_seen), resolved=not book.fee_unresolved)

    def _unresolved_trade(self, ended: EndedLifecycle) -> ClosedTrade:
        last = ended.last_seen or {}
        return ClosedTrade(key=ended.position_id, symbol=self.strategy_config.symbol, magic=self.strategy_config.magic,
                           reason=CloseReason.UNKNOWN, volume=abs(float(last.get("volume") or 0.0)),
                           entry_price=float(last.get("price_open") or 0.0), opened_at=ms_to_utc(ended.opened_ms),
                           closed_at=ms_to_utc(ended.closed_ms), last_seen=dict(last), resolved=False)

    def _close_reason(self, last_fill: Fill) -> CloseReason:
        if last_fill.kind in _FILL_REASONS:
            return _FILL_REASONS[last_fill.kind]
        role = self.orders.role_of(order_id=last_fill.order_id, client_order_id=last_fill.client_order_id)
        if role is not None:
            return _ROLE_REASONS[role]
        # Not our order and not a venue stop we can recognise. Where the profile labels venue stop fills that is a
        # human; otherwise the venue's own stop may simply be unlabelled, so it is honestly unknown.
        return CloseReason.MANUAL if self.profile.labels_stop_fills else CloseReason.UNKNOWN

    def resolve_closed_trade(self, key: str, last_seen: dict) -> Optional[ClosedTrade]:
        """Core hook: the payload resolved asynchronously by :meth:`_resolve_and_announce` (IB pattern)."""
        trade = self._resolved_closes.get(key)
        if trade is not None and trade.resolved:
            self._resolved_closes.pop(key, None)
        return trade

    def _qty_eps(self) -> float:
        if self.spec is not None and self.spec.min_amount:
            return self.spec.amount_to_base(self.spec.min_amount) * 1e-3
        return 1e-12

    # ------------------------------------------------------------------ convenience API
    def fetch_price_bars(self) -> pd.DataFrame:
        return self.price_buffer.get_data()

    def get_open_positions(self) -> list[dict]:
        if self.position_cache is not None:
            return [self._with_stop_levels(p) for p in self.position_cache.get_open()]
        if self.spot_ledger is not None:
            return [self._with_stop_levels(p) for p in self.spot_ledger.get_open()]
        return []

    def _with_stop_levels(self, position: dict) -> dict:
        """Merge the stop controller's levels into a position (the venue's own levels are kept when it has none)."""
        if self.stops is None:
            return position
        sl, tp = self.stops.levels(position["position_id"])
        if sl is None and tp is None:
            return position
        return {**position, "stop_loss": sl, "take_profit": tp}

    async def current_tick_info(self) -> Optional[dict]:
        if self._ticker is not None:
            return ticker_to_tick_info(self._ticker)
        try:
            return await fetch_tick_info(self.exchange, self.spec.symbol)
        except Exception as exc:
            logger.warning("%s: ticker unavailable (%s)", self.strategy_config.name, exc)
            return None

    def _notify_trade_failed(self, direction: str, reason: str, retcode: Any = None) -> None:
        if self.notifier:
            self.notifier.on_trade_failed(symbol=self.strategy_config.symbol, direction=direction, reason=reason,
                                          retcode=retcode, context={"strategy_name": self.strategy_config.name})

    async def pending_entry_orders(self) -> list[dict]:
        orders = await get_open_orders(self.exchange, self.spec, self.strategy_config.magic, self.profile)
        return [o for o in orders if self.orders.role_of(client_order_id=self.profile.client_order_id_of(o),
                                                         order_id=o.get("id")) is OrderRole.ENTRY]

    # ------------------------------------------------------------------ sizing
    async def calculate_quantity(self, price: float, stop_loss: Optional[float] = None) -> float:
        """Base quantity for an entry at ``price``.

        FIXED: ``units`` in the configured ``sizing_unit``. RISK_PCT_OF_EQUITY and KELLY_CRITERION: size so that a
        loss at ``stop_loss`` equals ``fraction * equity`` (``risk_pct`` / ``kelly_fraction``; the package does not
        estimate Kelly - the configured fraction IS the bet). Both need a stop distance; without one the order is
        rejected. Raises :class:`OrderSizeError`.
        """
        cfg = self.strategy_config
        sizing = cfg.position_sizing
        if sizing.type is PositionSizingType.FIXED:
            return self.spec.units_to_base(float(sizing.units), cfg.sizing_unit, price)
        fraction = sizing.risk_pct if sizing.type is PositionSizingType.RISK_PCT_OF_EQUITY else sizing.kelly_fraction
        if stop_loss is None or stop_loss <= 0 or stop_loss == price:
            raise OrderSizeError(f"{cfg.name}: {sizing.type.value} sizing needs a stop-loss distance and none is set")
        currency = self.spec.settle or self.spec.quote
        equity = await fetch_quote_equity(self.exchange, currency)
        if not equity or equity <= 0:
            raise OrderSizeError(f"{cfg.name}: no positive {currency} equity to size against")
        return equity * float(fraction) / abs(price - stop_loss)

    def _entry_levels(self, price: float, is_long: bool, stop_loss: Optional[float], take_profit: Optional[float]):
        """Explicit levels win; otherwise the position manager's initial levels (needed for risk-based sizing)."""
        if (stop_loss is None or take_profit is None) and self.position_manager is not None:
            m_sl, m_tp = self.position_manager.initial_levels(price, is_long)
            stop_loss = stop_loss if stop_loss is not None else m_sl
            take_profit = take_profit if take_profit is not None else m_tp
        return stop_loss, take_profit

    # ------------------------------------------------------------------ orders
    async def _submit_order(self, order_type: str, side: OrderSide, base_qty: float, price: Optional[float],
                            params: dict, role: OrderRole) -> dict:
        """Every order goes through here: precision, venue minimums, client id, registry, idempotent placement."""
        reference = price or float((await self.current_tick_info() or {}).get("last") or 0.0)
        amount = self.spec.checked_amount(base_qty, reference)
        client_id = make_client_order_id(self.strategy_config.magic, self.profile.client_id_rule)
        self.orders.record(client_id, role)
        order_price = self.spec.round_price(price) if price else None
        order = await place_order_idempotent(self.exchange, self.profile, self.spec, order_type, side.value, amount,
                                             order_price, params, client_id)
        self.orders.attach_order_id(client_id, order.get("id"))
        logger.info("%s: %s order %s %s %s x %s @ %s -> id=%s", self.strategy_config.name, role.value, client_id,
                    side.value, order_type, amount, order_price, order.get("id"))
        return order

    async def _cancel_order(self, order_id: str, params: Optional[dict] = None) -> bool:
        return await cancel_order_safe(self.exchange, self.profile, self.spec, order_id, params)

    async def _open_stop_order_ids(self) -> set[str]:
        """Ids of OUR open conditional orders, recognised by client-order-id prefix (not by the registry, whose
        entries age out) so that untracked leftovers are found too."""
        orders = await get_open_orders(self.exchange, self.spec, self.strategy_config.magic, self.profile,
                                       self.profile.stop_order_params(self.spec.market_type))
        return {str(o.get("id")) for o in orders}

    async def open_position(self, side: Union[OrderSide, str], quantity: Optional[float] = None, *,
                            stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                            limit_price: Optional[float] = None) -> bool:
        """Enter with a market (or limit) order. Stops: attached on entry where the venue can, otherwise applied
        as soon as the position appears. Returns False (logged + notified) on any rejection."""
        side = OrderSide(str(side).lower())
        if not self.is_perp and side is OrderSide.SELL:
            raise ValueError("spot strategies cannot open short positions")
        try:
            tick = await self.current_tick_info() or {}
            price = limit_price or float(tick.get("ask" if side is OrderSide.BUY else "bid") or tick.get("last") or 0)
            if price <= 0:
                raise OrderSizeError(f"{self.strategy_config.name}: no price to size the entry against")
            is_long = side is OrderSide.BUY
            stop_loss, take_profit = self._entry_levels(price, is_long, stop_loss, take_profit)
            qty = quantity if quantity is not None else await self.calculate_quantity(price, stop_loss)
            params = self.profile.order_params(self.spec.market_type)
            attach = self._can_attach(limit_price is not None) and (stop_loss is not None or take_profit is not None)
            if attach:
                params.update(self.profile.attached_stop_params(self.spec.market_type, stop_loss, take_profit,
                                                                self.strategy_config.stop_trigger))
            self.stops.set_pending(stop_loss, take_profit)
            order_type = "limit" if limit_price is not None else "market"
            await self._submit_order(order_type, side, qty, limit_price, params, OrderRole.ENTRY)
            return True
        except CryptoBannedError as exc:
            self._on_banned(exc)
            self._notify_trade_failed(side.value, str(exc))
            return False
        except OrderStateUnknownError as exc:
            # The entry MAY be live: keep the pending levels so a position that does appear is protected at once.
            logger.error("%s: entry %s state unknown (%s); keeping its stop levels pending", self.strategy_config.name,
                         side.value, exc)
            self._notify_trade_failed(side.value, str(exc), exc.error_class)
            return False
        except (CryptoError, OrderSizeError, ValueError) as exc:
            logger.error("%s: entry %s failed: %s", self.strategy_config.name, side.value, exc)
            self.stops.clear_pending()
            self._notify_trade_failed(side.value, str(exc), getattr(exc, "error_class", None))
            return False

    def _can_attach(self, is_limit: bool) -> bool:
        """Attach SL/TP to the entry only where the venue keeps them as replaceable POSITION-level stops.

        Elsewhere an attached stop becomes a venue-created order this package holds no id for: it could not be moved
        later, and the stop controller's own standalone orders would stack a second stop on top of it. There the
        stops are placed as standalone orders as soon as the position appears.
        """
        caps = self.caps.stop_caps
        if self.caps.stop_mode is not StopMode.NATIVE or not caps.position_level:
            return False
        return caps.attached_on_limit_entry if is_limit else caps.attached_on_market_entry

    async def place_order(self, order_type: Union[str, OrderType], price: float = 0.0, sl: float = 0.0,
                          tp: float = 0.0, quantity: Optional[float] = None) -> bool:
        """IB/MT5-parity entry point. Market and limit entries; SL/TP are applied by the stop controller."""
        ot = (order_type.value if isinstance(order_type, OrderType) else str(order_type)).lower()
        sl_level = sl or None
        tp_level = tp or None
        if ot in ("buy", "sell"):
            return await self.open_position(ot, quantity, stop_loss=sl_level, take_profit=tp_level)
        if ot in ("buy_limit", "sell_limit"):
            return await self.open_position(ot.split("_")[0], quantity, stop_loss=sl_level, take_profit=tp_level,
                                            limit_price=price)
        if ot in ("buy_stop", "sell_stop"):
            raise NotImplementedError("stop-entry orders are not supported in v1; use a market entry on the signal bar")
        raise ValueError(f"Invalid order_type: {order_type}")

    async def close_position(self, position: dict, reason: str = "strategy_close") -> bool:
        """Close with a reduce-only (perp) market order. ``reason`` is recorded as intent; the fill announces it."""
        return await self._close(position, reason, OrderRole.EXIT)

    async def _close_for_stop(self, position: dict, reason: str, role: OrderRole) -> bool:
        return await self._close(position, reason, role)

    async def _close(self, position: dict, reason: str, role: OrderRole) -> bool:
        """Close the position. Any failure after ``before_close`` (which may have lifted spot stops) puts the
        protection back; a close whose outcome is UNKNOWN keeps its intent and is re-checked shortly instead."""
        key = self._position_key(position)
        lifted = True  # before_close may lift protection even when it then fails part-way
        try:
            if not await self.stops.before_close(position):
                self._notify_trade_failed("CLOSE", "protective orders could not be lifted for the close")
                await self._restore_protection()
                return False
            if key is not None:
                self.note_close_intent(key, reason)
            qty = abs(position["position"])
            if not self.is_perp:
                qty = await self._spot_sellable_qty()
            params = self.profile.order_params(self.spec.market_type, reduce_only=True)
            await self._submit_order("market", close_side(position), qty, None, params, role)
            return True
        except OrderStateUnknownError as exc:
            logger.error("%s: close of %s state unknown (%s); re-checking the book shortly", self.strategy_config.name,
                         key, exc)
            self._notify_trade_failed("CLOSE", str(exc), exc.error_class)
            self._spawn(self._recheck_protection(), f"recheck:{key}")
            return False
        except CryptoBannedError as exc:
            self._on_banned(exc)
        except Exception as exc:
            logger.error("%s: close of %s failed: %s", self.strategy_config.name, key, exc)
            self._notify_trade_failed("CLOSE", str(exc), getattr(exc, "error_class", None))
        if key is not None:
            self.clear_close_intent(key)
        if lifted:
            await self._restore_protection()
        return False

    async def _spot_sellable_qty(self) -> float:
        """Sellable spot quantity. Cancelled stop orders can take a moment to release their locked balance."""
        free = await fetch_free_balance(self.exchange, self.spec.base)
        for _ in range(3):
            if free + self._qty_eps() >= self.spot_ledger.qty:
                break
            await self._sleep(0.5)
            free = await fetch_free_balance(self.exchange, self.spec.base)
        return self.spot_ledger.sellable_qty(free)

    async def _restore_protection(self) -> None:
        if not isinstance(self.stops, NativeStopController):
            return
        for raw in self._venue_positions():
            try:
                await self.stops.ensure_protection(raw)
            except Exception:
                logger.critical("%s: close failed AND stops could not be re-armed - %s may be UNPROTECTED",
                                self.strategy_config.name, raw.get("position_id"), exc_info=True)

    async def _recheck_protection(self, delay: float = 5.0) -> None:
        await self._sleep(delay)
        try:
            await self._sync_book(after_outage=False)
            await self._restore_protection()
        except Exception:
            logger.exception("%s: re-check after an unknown close failed", self.strategy_config.name)

    async def _apply_levels(self, position: dict, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
        try:
            return await self.stops.set_levels(position, stop_loss, take_profit)
        except CryptoBannedError as exc:
            self._on_banned(exc)
        except Exception as exc:
            logger.error("%s: setting SL=%s TP=%s on %s failed: %s", self.strategy_config.name, stop_loss, take_profit,
                         position.get("position_id"), exc)
            self._notify_trade_failed("STOPS", str(exc))
        return False

    async def manage_open_positions(self) -> set[str]:
        """Run the configured position manager over the open book. Returns ids a close was submitted for."""
        if self.position_manager is None:
            return set()
        positions = self.get_open_positions()
        if not positions:
            return set()
        return await self.position_manager.manage_positions(positions, self._apply_levels, self.close_position)

    async def cancel_working_orders(self, roles: tuple[OrderRole, ...] = (OrderRole.ENTRY, OrderRole.EXIT)) -> int:
        """Cancel this strategy's open orders with the given roles (shutdown keeps STOP_LOSS / TAKE_PROFIT)."""
        cancelled = 0
        for order in await get_open_orders(self.exchange, self.spec, self.strategy_config.magic, self.profile):
            role = self.orders.role_of(client_order_id=self.profile.client_order_id_of(order), order_id=order.get("id"))
            if role in roles and await self._cancel_order(str(order.get("id"))):
                cancelled += 1
        return cancelled

    # ------------------------------------------------------------------ prices for managed stops
    async def _price_loop(self) -> None:
        backoff = 1.0
        while True:
            try:
                if self.caps.stream_ticker:
                    ticker = await self.exchange.watch_ticker(self.spec.symbol)
                else:
                    ticker = await self.exchange.fetch_ticker(self.spec.symbol)
                if self._offline_since_ms is not None:
                    # Cleared only after the reconciliation succeeds: a failed one must not lose when prices
                    # stopped being watched, or the crossing check for that outage would start too late.
                    await self.on_reconnected(self._offline_since_ms, price_outage=True)
                    self._offline_since_ms = None
                await self.on_venue_ticker(ticker)
                backoff = 1.0
                if not self.caps.stream_ticker:
                    await self._sleep(self.strategy_config.managed_stop_poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._offline_since_ms is None:
                    self._offline_since_ms = self._clock()
                    logger.critical("%s: price feed lost (%s) - MANAGED STOPS ARE NOT LIVE until it recovers",
                                    self.strategy_config.name, exc)
                await self._sleep(backoff)
                backoff = min(backoff * 2, 30.0)


class GenericBasicCryptoStrategy(BaseCryptoStrategy):
    """Event-driven counterpart of MT5 ``GenericBasicStrategy`` / ``GenericBasicIBStrategy``."""

    async def on_new_bar(self):
        cfg = self.strategy_config
        logger.info("GenericBasicCryptoStrategy (%s) on_new_bar @ %s", cfg, self.latest_run_dt)
        # Exceptions propagate to the sealed _on_bar_close seam (heartbeat + breaker) - do not swallow them here.
        price_bars = self.fetch_price_bars()
        if price_bars is None or len(price_bars) < cfg.bars_to_copy:
            logger.warning("Insufficient price data for %s", cfg.symbol)
            return

        entries_long, exits_long, entries_short, exits_short = self.signal_generator.generate(price_bars)
        entries_long, exits_long = _last(entries_long), _last(exits_long)
        entries_short, exits_short = _last(entries_short), _last(exits_short)
        logger.info("Signal %s(%s): L=%s,%s S=%s,%s", cfg.symbol, cfg.magic, entries_long, exits_long, entries_short,
                    exits_short)

        closing = await self.manage_open_positions()
        positions = self.get_open_positions()
        for pos in positions:
            if pos["position_id"] in closing:
                continue
            is_long = pos["position"] > 0
            if (exits_long and is_long) or (exits_short and not is_long):
                if await self.close_position(pos, "signal_exit"):
                    closing.add(pos["position_id"])
        if closing:
            logger.info("Skipping entry for %s: close submitted this bar", cfg.symbol)
            return

        if entries_long and entries_short:
            logger.warning("Ambiguous signal for %s: both long and short entry on the same bar - skipping", cfg.symbol)
            return
        if not (entries_long or entries_short):
            return
        if entries_short and not self.is_perp:
            logger.info("Short entry ignored for spot %s", cfg.symbol)
            return
        pending = await self.pending_entry_orders()
        if len(positions) + len(pending) >= self.max_number_of_open_positions:
            logger.info("Entry ignored - %d open + %d pending for %s", len(positions), len(pending), cfg.symbol)
            return
        tick_info = await self.current_tick_info()
        filter_context = {"datetime": self.latest_run_dt, "symbol_info": {"market": self.spec.symbol},
                          "open_positions": len(positions), "signal_type": "long" if entries_long else "short",
                          "tick_info": tick_info}
        if not self.filter_chain(filter_context):
            logger.info("Filter chain blocked entry for %s", cfg.symbol)
            return
        side = OrderSide.BUY if entries_long else OrderSide.SELL
        if await self.open_position(side):
            logger.info("Entry order submitted for %s side=%s", cfg.symbol, side.value)
        else:
            logger.error("Entry order failed for %s side=%s", cfg.symbol, side.value)
