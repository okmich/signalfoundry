"""Strategy-owned spot inventory.

A spot "position" is not something the venue keeps: a BTC balance is a balance, whoever bought it. So the strategy
keeps its own ledger, built ONLY from fills of orders it placed (client-order-id prefix or a registered order id),
and never touches balance it did not buy. Sell size is ``min(ledger qty, free base balance)``.

Fees charged in the base coin on a buy are not received, so the ledger holds the received quantity (see ``pnl``).
The ledger is persisted after every fill, and on startup it catches up from the venue's fill history since the last
fill it saw - so a restart costs one paged history read, not a full rebuild.
"""
import logging
from typing import Optional

from .client_order_id import is_ours
from .enums import FillKind
from .markets import MarketSpec
from .models import Fill
from .orders import OrderRegistry
from .pnl import Book
from .position_cache import EndedLifecycle, make_position_id, position_dict
from .state_store import StateStore
from .timeframe_utils import utc_now_ms

logger = logging.getLogger(__name__)

BookEvents = tuple[Optional[dict], Optional[EndedLifecycle]]

#: How far before the newest applied fill a catch-up re-reads history.
_CATCH_UP_OVERLAP_MS = 30 * 60 * 1000
#: Seen trade ids kept for de-duplication (stream and REST reconciliation deliver the same fills).
_SEEN_LIMIT = 2000


class SpotInventoryLedger:

    def __init__(self, spec: MarketSpec, log_symbol: str, magic: int, orders: OrderRegistry, store: StateStore,
                 clock=utc_now_ms):
        self.spec = spec
        self.log_symbol = log_symbol
        self.magic = magic
        self.orders = orders
        self.store = store
        self.clock = clock
        self._state = store.section("spot_ledger")
        self._state.setdefault("seen", [])
        self._state.setdefault("round_trip", [])
        self._book = Book(qty=float(self._state.get("qty", 0.0)), avg_price=float(self._state.get("avg_price", 0.0)))
        self._last_price: Optional[float] = None

    # ------------------------------------------------------------------ attribution
    def owns(self, fill: Fill) -> bool:
        return is_ours(fill.client_order_id, self.magic) or (fill.order_id is not None
                                                             and fill.order_id in self.orders.order_ids())

    # ------------------------------------------------------------------ updates
    def apply_fill(self, fill: Fill) -> tuple[Optional[dict], Optional[EndedLifecycle]]:
        """Apply one of OUR fills. Returns ``(opened, ended)`` like ``CryptoPositionCache.apply_position``."""
        if fill.kind in (FillKind.FUNDING, FillKind.SETTLEMENT) or fill.trade_id in self._state["seen"]:
            return None, None
        self._remember(fill.trade_id)
        self._last_price = fill.price
        was_open = bool(self._state.get("position_id"))
        self._book.apply(fill, spot=True)
        self._state["last_fill_ms"] = max(int(self._state.get("last_fill_ms", 0)), fill.timestamp_ms)
        opened = ended = None
        if self._book.qty < 0:
            # The strategy sold more than its ledger held - balance it did not buy. Clamp: never go short on spot.
            logger.error("%s: spot ledger went negative (%.10f) - an untracked sell; clamping to 0", self.spec.symbol,
                         self._book.qty)
            self._book.qty, self._book.avg_price = 0.0, 0.0
        if not was_open and not self.spec.is_dust(self._book.qty):
            opened_ms = fill.timestamp_ms
            while make_position_id(self.spec.symbol, opened_ms) == self._state.get("last_position_id"):
                opened_ms += 1  # never reuse the id of the lifecycle that just closed (core keys closes on it)
            self._state["position_id"] = make_position_id(self.spec.symbol, opened_ms)
            self._state["opened_ms"] = fill.timestamp_ms
            self._state["round_trip"] = []
            opened = True
        if self._state.get("position_id"):
            self._state["round_trip"].append(fill.to_state())
        if was_open and self.spec.is_dust(self._book.qty):
            if self._book.qty > 0:
                logger.info("%s: writing off dust %.10f left after the close", self.spec.symbol, self._book.qty)
            ended = EndedLifecycle(position_id=self._state["position_id"], opened_ms=int(self._state["opened_ms"]),
                                   closed_ms=fill.timestamp_ms, last_seen=dict(self._state.get("last_seen") or {}),
                                   fills=[Fill.from_state(f) for f in self._state["round_trip"]])
            self._book.qty, self._book.avg_price = 0.0, 0.0
            self._state["last_position_id"] = self._state["position_id"]
            for key in ("position_id", "opened_ms", "last_seen"):
                self._state.pop(key, None)
            self._state["round_trip"] = []
        self._persist()
        if opened:
            return self.get_open()[0], ended
        return None, ended

    def _remember(self, trade_id: str) -> None:
        seen = self._state["seen"]
        seen.append(trade_id)
        if len(seen) > _SEEN_LIMIT:
            del seen[: len(seen) - _SEEN_LIMIT]

    def _persist(self) -> None:
        self._state["qty"] = self._book.qty
        self._state["avg_price"] = self._book.avg_price
        if self._state.get("position_id"):
            pos = self.get_open()
            if pos:
                self._state["last_seen"] = {k: pos[0].get(k) for k in ("position_id", "symbol", "market_symbol", "type",
                                                                       "position", "volume", "avg_cost", "price_open",
                                                                       "opened_ms")}
        self.store.save()

    async def catch_up(self, exchange, profile, lookback_ms: int) -> list[BookEvents]:
        """Apply our fills since the last one seen (or ``lookback_ms`` back on a fresh ledger)."""
        # Overlap the last fill seen: a fill that arrived late on the stream can be OLDER than the newest one
        # applied. Already-applied fills are skipped by the persisted seen-ids.
        last = int(self._state.get("last_fill_ms") or 0)
        since = (last - _CATCH_UP_OVERLAP_MS) if last else (self.clock() - lookback_ms)
        fills = await profile.fetch_fills(exchange, self.spec, since)
        return [self.apply_fill(f) for f in fills if self.owns(f)]

    # ------------------------------------------------------------------ reads
    @property
    def qty(self) -> float:
        return self._book.qty

    def sellable_qty(self, free_base: float) -> float:
        if free_base + 1e-12 < self._book.qty:
            logger.warning("%s: free %s balance %.10f is below the ledger's %.10f (fees, a manual sale, or a lock); "
                           "selling only what is free", self.spec.symbol, self.spec.base, free_base, self._book.qty)
        return max(0.0, min(self._book.qty, free_base))

    def update_price(self, price: float) -> None:
        self._last_price = price

    def get_open(self) -> list[dict]:
        if not self._state.get("position_id") or self.spec.is_dust(self._book.qty):
            return []
        return [position_dict(position_id=self._state["position_id"], symbol=self.log_symbol,
                              market_symbol=self.spec.symbol, qty=self._book.qty, avg_price=self._book.avg_price,
                              price_current=self._last_price, stop_loss=None, take_profit=None,
                              opened_ms=int(self._state["opened_ms"]))]
