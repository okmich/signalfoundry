"""Stop-loss / take-profit ownership for the strategy's single position.

Two implementations behind one interface, chosen once at bootstrap from the resolved ``StopMode``:

* :class:`NativeStopController` - levels live on the venue and keep working while this process is down. Where the
  profile supports position-level stops (Bybit perps) a change REPLACES both levels in one call, so there is never a
  window without protection. Otherwise each level is a separate conditional order, changed new-before-old: the
  replacement is placed BEFORE the old order is cancelled. Every operation runs under one lock (reconnect handlers,
  the bar cycle and position events can all reach it at once), and an order whose cancel could not be confirmed is
  kept as an orphan and retried - never forgotten while it may still be live.
* :class:`ManagedStopController` - levels live here (persisted), prices are watched, and a crossing sends a
  reduce-only market close. It is NOT live while the process is down; after a PRICE-feed or process outage it checks
  the candles of the outage - clamped to when the position opened and its levels were set - for a crossing.

Desired levels are persisted per lifecycle id, so a restart re-applies (native) or keeps watching (managed) the same
levels.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

from ccxt.base.errors import InsufficientFunds

from .enums import MarketType, OrderRole, OrderSide, StopMode, StopTrigger
from .markets import MarketSpec
from .resilience import CryptoPermanentError
from .state_store import StateStore
from .timeframe_utils import utc_now_ms

logger = logging.getLogger(__name__)

#: submit(order_type, side, base_qty, price, params, role) -> order dict
Submit = Callable[..., Awaitable[dict]]
#: cancel(order_id, params) -> bool (True when the order is gone)
Cancel = Callable[..., Awaitable[bool]]
#: close(position, reason, role) -> bool
Close = Callable[..., Awaitable[bool]]

_KINDS = ((OrderRole.STOP_LOSS, "stop_loss"), (OrderRole.TAKE_PROFIT, "take_profit"))
#: Longest outage scanned for managed-stop crossings (1m candles, paged).
_MAX_OFFLINE_SCAN_MS = 7 * 24 * 3600 * 1000


def close_side(position: dict) -> OrderSide:
    return OrderSide.SELL if position["position"] > 0 else OrderSide.BUY


def validate_levels(position: dict, stop_loss: Optional[float], take_profit: Optional[float]) -> None:
    """Reject levels on the wrong side of the entry (a long's take-profit below its entry would close it at once)."""
    entry = float(position.get("price_open") or 0.0)
    if entry <= 0:
        return
    is_long = position["position"] > 0
    if take_profit is not None and ((is_long and take_profit <= entry) or (not is_long and take_profit >= entry)):
        raise ValueError(f"take-profit {take_profit} is on the wrong side of entry {entry} for a "
                         f"{'long' if is_long else 'short'}")
    if stop_loss is not None and stop_loss <= 0:
        raise ValueError(f"stop-loss must be positive (got {stop_loss})")


class StopController(ABC):
    mode: StopMode

    def __init__(self, name: str, spec: MarketSpec, store: StateStore, trigger: StopTrigger, clock=utc_now_ms):
        self.name = name
        self.spec = spec
        self.store = store
        self.trigger = trigger
        self.clock = clock
        self._state = store.section("stops")
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ levels
    def levels(self, position_id: Optional[str]) -> tuple[Optional[float], Optional[float]]:
        if position_id is None or self._state.get("position_id") != position_id:
            return None, None
        return self._state.get("stop_loss"), self._state.get("take_profit")

    def set_pending(self, stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        """Levels requested with an entry order, before the position (and its id) exists."""
        self._state["pending"] = {"stop_loss": stop_loss, "take_profit": take_profit}
        self.store.save()

    def clear_pending(self) -> None:
        if self._state.pop("pending", None) is not None:
            self.store.save()

    def _bind(self, position_id: str) -> None:
        """Make ``position_id`` the position this state describes, dropping a previous position's leftovers.

        Must run BEFORE any order for the position is recorded: binding afterwards would wipe the order ids just
        placed for it. Pending entry levels and orphaned orders survive the rebind.
        """
        if self._state.get("position_id") != position_id:
            keep = {k: self._state[k] for k in ("pending", "orphans") if k in self._state}
            self._state.clear()
            self._state.update(keep)
            self._state["position_id"] = position_id

    def _remember(self, position_id: str, stop_loss: Optional[float], take_profit: Optional[float]) -> None:
        self._bind(position_id)
        if (self._state.get("stop_loss"), self._state.get("take_profit")) != (stop_loss, take_profit):
            self._state["levels_set_ms"] = self.clock()
        self._state["stop_loss"] = stop_loss
        self._state["take_profit"] = take_profit
        self.store.save()

    # ------------------------------------------------------------------ lifecycle (all under the lock)
    async def on_position_opened(self, position: dict) -> None:
        """Bind pending entry levels to the new position and make sure they are in force."""
        async with self._lock:
            pending = self._state.pop("pending", None)
            if pending and (pending.get("stop_loss") is not None or pending.get("take_profit") is not None):
                await self._set_levels(position, pending.get("stop_loss"), pending.get("take_profit"))
            else:
                self._bind(position["position_id"])
                self.store.save()

    async def on_position_closed(self, position_id: str) -> None:
        async with self._lock:
            await self._on_closed(position_id)

    async def _on_closed(self, position_id: str) -> None:
        if self._state.get("position_id") == position_id:
            keep = {k: self._state[k] for k in ("pending", "orphans") if k in self._state}
            self._state.clear()
            self._state.update(keep)
            self.store.save()

    async def set_levels(self, position: dict, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
        """Put ``(stop_loss, take_profit)`` in force for ``position``. ``None`` removes a level."""
        async with self._lock:
            return await self._set_levels(position, stop_loss, take_profit)

    async def ensure_protection(self, position: dict) -> None:
        """After a restart, reconnect or size change: make sure the desired levels are actually in force.

        ``position`` must be the VENUE's view (its own stop fields), not one already merged with this controller's
        levels - otherwise a missing venue stop is invisible.
        """
        async with self._lock:
            await self._ensure_protection(position)

    async def before_close(self, position: dict) -> bool:
        """Called before a strategy close. False aborts the close (e.g. a stop could not be confirmed cancelled)."""
        return True

    @abstractmethod
    async def _set_levels(self, position: dict, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
        ...

    @abstractmethod
    async def _ensure_protection(self, position: dict) -> None:
        ...

    def describe(self) -> str:
        return f"{self.mode.value} stops ({self.trigger.value} trigger)"


class NativeStopController(StopController):
    mode = StopMode.NATIVE

    def __init__(self, name: str, spec: MarketSpec, store: StateStore, trigger: StopTrigger, *, exchange, profile,
                 position_level: bool, submit: Submit, cancel: Cancel, fetch_open_stop_ids, clock=utc_now_ms,
                 live_positions: Optional[Callable[[], list]] = None):
        super().__init__(name, spec, store, trigger, clock)
        #: () -> the venue's CURRENT positions; standalone stop quantities are taken from here, never from a
        #: caller's snapshot that a concurrent fill may already have outdated.
        self.live_positions = live_positions
        self.exchange = exchange
        self.profile = profile
        self.position_level = position_level
        self.submit = submit
        self.cancel = cancel
        #: () -> set of ids of OUR open stop orders on the venue (by client-order-id prefix)
        self.fetch_open_stop_ids = fetch_open_stop_ids

    async def _set_levels(self, position: dict, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
        validate_levels(position, stop_loss, take_profit)
        if self.position_level:
            await self.profile.set_position_stops(self.exchange, self.spec, stop_loss, take_profit, self.trigger)
            self._remember(position["position_id"], stop_loss, take_profit)
            logger.info("%s: position stops set SL=%s TP=%s (%s)", self.name, stop_loss, take_profit,
                        self.trigger.value)
            return True
        position = self._live(position)
        if position is None:
            return False
        self._bind(position["position_id"])
        ok = True
        for role, key in _KINDS:
            level = stop_loss if role is OrderRole.STOP_LOSS else take_profit
            ok = await self._replace_order(position, role, key, level) and ok
        # Remember what is ACTUALLY on the venue: a failed replacement leaves the old order (and level) in force.
        sl_order, tp_order = self._order(OrderRole.STOP_LOSS), self._order(OrderRole.TAKE_PROFIT)
        self._remember(position["position_id"], sl_order["level"] if sl_order else None,
                       tp_order["level"] if tp_order else None)
        return ok

    def _live(self, position: dict) -> Optional[dict]:
        """The venue's current view of ``position`` (None if it is no longer open)."""
        if self.live_positions is None:
            return position
        for live in self.live_positions():
            if live.get("position_id") == position["position_id"]:
                return live
        logger.info("%s: position %s is no longer open; stop change skipped", self.name, position["position_id"])
        return None

    def _stop_params(self) -> dict:
        return self.profile.stop_order_params(self.spec.market_type)

    def _orders(self) -> dict:
        return self._state.setdefault("orders", {})

    def _order(self, role: OrderRole) -> Optional[dict]:
        return self._orders().get(role.value)

    def _orphans(self) -> list:
        return self._state.setdefault("orphans", [])

    async def _cancel_or_orphan(self, order_id: str, what: str) -> bool:
        """Cancel; if that cannot be confirmed, keep the id as an orphan to retry (it may still be live)."""
        if await self.cancel(order_id, self._stop_params()):
            return True
        logger.error("%s: could not confirm cancel of %s %s - kept for retry", self.name, what, order_id)
        if order_id not in self._orphans():
            self._orphans().append(order_id)
        self.store.save()
        return False

    async def _retry_orphans(self) -> None:
        for order_id in list(self._orphans()):
            if await self.cancel(order_id, self._stop_params()):
                self._orphans().remove(order_id)
        self.store.save()

    async def _replace_order(self, position: dict, role: OrderRole, key: str, level: Optional[float]) -> bool:
        old = self._order(role)
        qty = abs(position["position"])
        if old is not None and level is not None and old.get("level") == level \
                and abs(float(old.get("qty", 0)) - qty) <= 1e-12:
            return True
        if level is None:
            if old is not None:
                self._orders().pop(role.value, None)
                await self._cancel_or_orphan(old["order_id"], key)
            return True
        params = self.profile.standalone_stop_params(self.spec.market_type, level, role is OrderRole.STOP_LOSS,
                                                     self.trigger)
        try:
            new = await self.submit("market", close_side(position), qty, None, params, role)
        except CryptoPermanentError as exc:
            locked = isinstance(exc.cause, InsufficientFunds) and self.spec.market_type is MarketType.SPOT
            if old is None or not locked:
                logger.error("%s: could not place %s at %s: %s", self.name, key, level, exc)
                return False
            # Spot venues may lock the balance for the existing stop, so new-before-old fails for lack of funds.
            # Fall back to old-before-new and say so: for a moment the position is unprotected.
            logger.critical("%s: replacing %s old-before-new (venue locks balance); position briefly unprotected",
                            self.name, key)
            if not await self.cancel(old["order_id"], self._stop_params()):
                return False
            self._orders().pop(role.value, None)
            new = await self.submit("market", close_side(position), qty, None, params, role)
            old = None
        self._orders()[role.value] = {"order_id": str(new.get("id")), "level": level, "qty": qty}
        self.store.save()
        if old is not None and str(old.get("order_id")) != str(new.get("id")):
            await self._cancel_or_orphan(old["order_id"], key)
        return True

    async def before_close(self, position: dict) -> bool:
        if self.position_level or self.spec.market_type is not MarketType.SPOT:
            # Position-level stops die with the position; perp conditional orders are reduce-only and cannot open
            # a new position. Spot stops are plain sells: they must go BEFORE the close or they sell later.
            return True
        async with self._lock:
            for role, key in _KINDS:
                order = self._order(role)
                if order is None:
                    continue
                if not await self.cancel(order["order_id"], self._stop_params()):
                    logger.critical("%s: aborting close - spot %s %s could not be confirmed cancelled", self.name,
                                    key, order["order_id"])
                    return False
                self._orders().pop(role.value, None)
            self.store.save()
            return True

    async def _on_closed(self, position_id: str) -> None:
        if not self.position_level and self._state.get("position_id") == position_id:
            for role, key in _KINDS:
                order = self._order(role)
                if order is not None:
                    await self._cancel_or_orphan(order["order_id"], key)
        await super()._on_closed(position_id)
        await self._retry_orphans()

    async def _ensure_protection(self, position: dict) -> None:
        sl, tp = self.levels(position["position_id"])
        if self.position_level:
            venue_sl, venue_tp = position.get("stop_loss"), position.get("take_profit")
            if sl is None and tp is None:
                if venue_sl is not None or venue_tp is not None:
                    # Adopt what the venue has (attached on entry, or set before a restart that lost local state).
                    self._remember(position["position_id"], venue_sl, venue_tp)
                return
            if not _same(sl, venue_sl) or not _same(tp, venue_tp):
                logger.warning("%s: venue stops SL=%s TP=%s differ from desired SL=%s TP=%s; re-applying", self.name,
                               venue_sl, venue_tp, sl, tp)
                await self._set_levels(position, sl, tp)
            return
        await self._retry_orphans()
        open_ids = await self.fetch_open_stop_ids()
        tracked = {o["order_id"] for o in self._orders().values()}
        for order_id in open_ids - tracked:
            # Ours (client-id prefix) but not tracked: left behind by a crash between placement and bookkeeping.
            logger.warning("%s: cancelling untracked stop order %s", self.name, order_id)
            await self._cancel_or_orphan(order_id, "untracked stop")
        qty = abs(position["position"])
        stale = False
        for role, _ in _KINDS:
            order = self._order(role)
            if order is None:
                continue
            if order["order_id"] not in open_ids:
                self._orders().pop(role.value, None)   # filled, cancelled outside, or expired: re-place below
                stale = True
            elif abs(float(order.get("qty", 0)) - qty) > 1e-12:
                stale = True
        wanted_missing = (sl is not None and self._order(OrderRole.STOP_LOSS) is None) or \
                         (tp is not None and self._order(OrderRole.TAKE_PROFIT) is None)
        if stale or wanted_missing:
            logger.warning("%s: stop orders missing or sized for another quantity; re-placing SL=%s TP=%s", self.name,
                           sl, tp)
            await self._set_levels(position, sl, tp)


class ManagedStopController(StopController):
    mode = StopMode.MANAGED

    def __init__(self, name: str, spec: MarketSpec, store: StateStore, trigger: StopTrigger, *, close: Close,
                 clock=utc_now_ms):
        super().__init__(name, spec, store, trigger, clock)
        self.close = close
        self._fired_for: Optional[str] = None
        logger.warning("%s: MANAGED stops - levels are enforced by this process only and are NOT live while it is "
                       "disconnected or stopped", name)

    async def _set_levels(self, position: dict, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
        validate_levels(position, stop_loss, take_profit)
        self._remember(position["position_id"], stop_loss, take_profit)
        self._fired_for = None
        logger.info("%s: managed stops SL=%s TP=%s", self.name, stop_loss, take_profit)
        return True

    def crossed(self, position: dict, low: float, high: float) -> Optional[OrderRole]:
        """Which level (if any) a price range [low, high] crossed. Stop-loss wins when a range spans both."""
        sl, tp = self.levels(position["position_id"])
        is_long = position["position"] > 0
        if sl is not None and ((is_long and low <= sl) or (not is_long and high >= sl)):
            return OrderRole.STOP_LOSS
        if tp is not None and ((is_long and high >= tp) or (not is_long and low <= tp)):
            return OrderRole.TAKE_PROFIT
        return None

    async def on_price(self, position: dict, price: float) -> Optional[OrderRole]:
        self._state["last_price_ms"] = self.clock()
        role = self.crossed(position, price, price)
        if role is not None:
            await self._fire(position, role, f"price {price}")
        return role

    async def check_offline_window(self, exchange, position: dict, since_ms: int,
                                   timeframe: str = "1m") -> Optional[OrderRole]:
        """After a price-feed / process outage: did price cross a level while nobody watched?

        The window is clamped to start no earlier than the position's open and the moment its current levels were set
        - a dip BEFORE the position existed, or before a level was moved, is not a crossing of this stop.
        """
        sl, tp = self.levels(position["position_id"])
        if sl is None and tp is None:
            return None
        since = max(int(since_ms), int(position.get("opened_ms") or 0), int(self._state.get("levels_set_ms") or 0))
        now = self.clock()
        since = max(since, now - _MAX_OFFLINE_SCAN_MS)
        low, high = None, None
        cursor = since - since % 60_000
        while cursor < now:
            rows = await exchange.fetch_ohlcv(self.spec.symbol, timeframe, cursor, 1000)
            rows = [r for r in rows or [] if int(r[0]) + 60_000 > since]
            if not rows:
                break
            low = min([float(r[3]) for r in rows] + ([low] if low is not None else []))
            high = max([float(r[2]) for r in rows] + ([high] if high is not None else []))
            newest = max(int(r[0]) for r in rows)
            if newest < cursor:
                break
            cursor = newest + 60_000
        if low is None:
            return None
        role = self.crossed(position, low, high)
        if role is not None:
            logger.critical("%s: managed %s level was crossed while unwatched (range %s..%s since %d); closing now",
                            self.name, role.value, low, high, since)
            await self._fire(position, role, "crossed while unwatched")
        return role

    async def _fire(self, position: dict, role: OrderRole, why: str) -> None:
        if self._fired_for == position["position_id"]:
            return  # a close is already on its way for this position
        self._fired_for = position["position_id"]
        logger.warning("%s: managed %s triggered (%s)", self.name, role.value, why)
        ok = False
        try:
            ok = await self.close(position, f"managed_{role.value}", role)
        finally:
            if not ok:
                self._fired_for = None  # re-arm: the next price crossing tries again

    async def _ensure_protection(self, position: dict) -> None:
        sl, tp = self.levels(position["position_id"])
        logger.warning("%s: managed stops for %s SL=%s TP=%s are live again", self.name, position["position_id"],
                       sl, tp)

    @property
    def last_price_ms(self) -> Optional[int]:
        value = self._state.get("last_price_ms")
        return int(value) if value else None

    def persist_heartbeat(self) -> None:
        self.store.save()


def _same(a: Optional[float], b: Optional[float], rel: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)
