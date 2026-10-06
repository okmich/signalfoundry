"""The strategy's single perpetual position, with a lifecycle id.

A venue nets every fill on a symbol into one position, and it has no id: the symbol is reused on every re-entry.
Core's closed-trade reconciliation needs a key that is NOT reused while a close is pending (``BaseStrategy``:
"a position key must not be REUSED while a close on that key is still pending"). So each flat-to-flat lifecycle gets
its own id, ``{market_symbol}@{detected_ms}`` from the LOCAL clock, persisted so a restart mid-position keeps the same
id. Never the venue's position timestamp: Bybit's ``createdTime`` is the first time a position was EVER created on the
symbol, so it repeats across round trips. A flip (long <-> short in one update) ends one lifecycle and starts another.

The position itself always comes from the venue (``fetch_positions`` / ``watch_positions``): strategy isolation
guarantees nothing else trades this (account, symbol), so the venue position IS the strategy's position.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from .markets import MarketSpec
from .state_store import StateStore
from .timeframe_utils import utc_now_ms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndedLifecycle:
    """A position that went flat: what is needed to resolve it into a ``ClosedTrade``."""
    position_id: str
    opened_ms: int
    closed_ms: int
    last_seen: dict
    fills: list = field(default_factory=list)
    #: Signed size of the position that REPLACED this one in the same update (a flip); 0 for a plain close.
    residual_qty: float = 0.0


def make_position_id(market_symbol: str, opened_ms: int) -> str:
    return f"{market_symbol}@{opened_ms}"


def position_dict(*, position_id: str, symbol: str, market_symbol: str, qty: float, avg_price: float,
                  price_current: Optional[float], stop_loss: Optional[float], take_profit: Optional[float],
                  opened_ms: int, unrealized_pnl: Optional[float] = None) -> dict:
    """The position shape handed to core, filters and position managers.

    ``position_id`` is what core's ``_position_key`` picks up; there is deliberately no ``ticket`` / ``conId`` key.
    Unset stop levels are ``None`` (never 0.0).
    """
    return {"position_id": position_id, "symbol": symbol, "market_symbol": market_symbol,
            "type": 0 if qty > 0 else 1, "position": qty, "volume": abs(qty), "avg_cost": avg_price,
            "price_open": avg_price, "price_current": price_current if price_current else avg_price,
            "stop_loss": stop_loss, "take_profit": take_profit, "opened_ms": opened_ms,
            "unrealized_pnl": unrealized_pnl}


class CryptoPositionCache:
    """Mirror of one perp position for one strategy."""

    def __init__(self, spec: MarketSpec, log_symbol: str, profile, store: StateStore, clock=utc_now_ms):
        self.spec = spec
        self.log_symbol = log_symbol
        self.profile = profile
        self.store = store
        self.clock = clock
        self._lifecycle = store.section("lifecycle")
        self._snapshot: Optional[dict] = None

    # ------------------------------------------------------------------ updates
    def apply_position(self, position: Optional[dict], *,
                       adopting: bool = False) -> tuple[Optional[dict], Optional[EndedLifecycle]]:
        """Apply a venue position (``None`` / zero size = flat). Returns ``(opened, ended)``: the position dict if a
        lifecycle just started, and the lifecycle that just ended, if any (both on a flip long <-> short).

        The lifecycle id comes from the LOCAL clock at detection, never from the venue: Bybit's position
        ``createdTime`` is the first time a position was ever created on the symbol, so it would repeat across round
        trips. ``adopting`` (a position found at startup that this process never saw open) lets the venue timestamp
        widen the fill-history window used to resolve it - and nothing else.
        """
        qty = self._signed_qty(position)
        was_open = bool(self._lifecycle.get("position_id"))
        ended = None
        if was_open and (qty == 0.0 or (qty > 0) != (self._lifecycle.get("side") == "long")):
            ended = self._end(residual_qty=qty)
            was_open = False
        if qty == 0.0:
            self._snapshot = None
            return None, ended
        opened = None
        if not was_open:
            detected = self.clock()
            window_start = detected
            venue_ts = int(position.get("timestamp") or 0)
            if adopting and 0 < venue_ts < detected:
                window_start = venue_ts
            self._start(detected, window_start, "long" if qty > 0 else "short")
            opened = True
        self._snapshot = position
        return (self.get_open()[0] if opened else None), ended

    def _end(self, residual_qty: float = 0.0) -> EndedLifecycle:
        ended = EndedLifecycle(position_id=self._lifecycle["position_id"], opened_ms=int(self._lifecycle["opened_ms"]),
                               closed_ms=self.clock(), last_seen=dict(self._lifecycle.get("last_seen") or {}),
                               residual_qty=residual_qty)
        last_id = self._lifecycle["position_id"]
        self._lifecycle.clear()
        self._lifecycle["last_position_id"] = last_id
        self.store.save()
        return ended

    def note_fill_time(self, timestamp_ms: int) -> None:
        """A fill is the most precise open time there is; pull ``opened_ms`` back to the first fill seen."""
        if self._lifecycle.get("position_id") and 0 < timestamp_ms < int(self._lifecycle["opened_ms"]):
            # The id was already handed to core; only the window start used for resolution moves.
            self._lifecycle["opened_ms"] = timestamp_ms
            self.store.save()

    def _start(self, detected_ms: int, window_start_ms: int, side: str) -> None:
        position_id = make_position_id(self.spec.symbol, detected_ms)
        last_id = self._lifecycle.get("last_position_id")
        while position_id == last_id:  # two lifecycles detected in the same millisecond
            detected_ms += 1
            position_id = make_position_id(self.spec.symbol, detected_ms)
        self._lifecycle.clear()
        self._lifecycle.update({"position_id": position_id, "opened_ms": window_start_ms, "side": side,
                                "last_position_id": last_id})
        self.store.save()

    def _signed_qty(self, position: Optional[dict]) -> float:
        if not position:
            return 0.0
        contracts = float(position.get("contracts") or 0.0)
        if contracts == 0.0:
            return 0.0
        base = self.spec.amount_to_base(abs(contracts))
        if self.spec.is_dust(base):
            return 0.0
        side = str(position.get("side") or "").lower()
        if side not in ("long", "short"):
            logger.warning("%s: position without a side, cannot track it: %r", self.spec.symbol, position)
            return 0.0
        return base if side == "long" else -base

    # ------------------------------------------------------------------ reads
    @property
    def position_id(self) -> Optional[str]:
        return self._lifecycle.get("position_id")

    @property
    def opened_ms(self) -> Optional[int]:
        value = self._lifecycle.get("opened_ms")
        return int(value) if value is not None else None

    def get_open(self) -> list[dict]:
        if self._snapshot is None or not self.position_id:
            return []
        qty = self._signed_qty(self._snapshot)
        sl, tp = self.profile.position_stop_levels(self._snapshot)
        price = self._snapshot.get("markPrice") or self._snapshot.get("lastPrice")
        pos = position_dict(position_id=self.position_id, symbol=self.log_symbol, market_symbol=self.spec.symbol,
                            qty=qty, avg_price=float(self._snapshot.get("entryPrice") or 0.0),
                            price_current=float(price) if price else None, stop_loss=sl, take_profit=tp,
                            opened_ms=int(self._lifecycle["opened_ms"]),
                            unrealized_pnl=self._snapshot.get("unrealizedPnl"))
        if self._lifecycle.get("last_seen") != _persistable(pos):
            self._lifecycle["last_seen"] = _persistable(pos)
            self.store.save()
        return [pos]

    # ------------------------------------------------------------------ venue sync
    async def resync(self, exchange, *, adopting: bool = False) -> tuple[Optional[dict], Optional[EndedLifecycle]]:
        """REST rebuild: the venue position replaces whatever the cache believed."""
        positions = await exchange.fetch_positions([self.spec.symbol])
        mine = [p for p in positions or [] if p.get("symbol") == self.spec.symbol and float(p.get("contracts") or 0)]
        if len(mine) > 1:
            raise RuntimeError(f"{self.spec.symbol}: {len(mine)} open positions - the account is in hedge mode")
        return self.apply_position(mine[0] if mine else None, adopting=adopting)


#: The stable part of a position dict, persisted as the closed-trade fallback description. Prices that move every
#: update are left out so the state file is not rewritten on every tick.
_LAST_SEEN_KEYS = ("position_id", "symbol", "market_symbol", "type", "position", "volume", "avg_cost", "price_open",
                   "stop_loss", "take_profit", "opened_ms")


def _persistable(pos: dict) -> dict:
    return {k: pos.get(k) for k in _LAST_SEEN_KEYS}
