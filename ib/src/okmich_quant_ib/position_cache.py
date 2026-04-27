"""Fill-driven in-memory position mirror for one strategy."""
import logging
from typing import TYPE_CHECKING

from ib_async import IB

from .contract import make_order_ref

if TYPE_CHECKING:
    from ib_async import Fill, Position, Trade

logger = logging.getLogger(__name__)


class IBPositionCache:
    """In-memory mirror of open positions for one strategy.

    Maintained by:
      fillEvent     — aggregates partial fills into running positions
      positionEvent — broker-authoritative drift correction (conId-level)

    Each fill is filtered by orderRef AND conId so only fills for this
    strategy's specific contract affect the cache.

    Same-symbol limitation: IB position snapshots carry no orderRef, so
    running two strategies on the same symbol on the same account is
    unsupported. Use sub-accounts or disjoint symbol sets.
    """

    def __init__(self, symbol: str, con_id: int, magic: int):
        self.symbol = symbol
        self.con_id = con_id
        self.order_ref = make_order_ref(magic)
        self._positions: dict[int, dict] = {}

    def on_fill(self, trade: "Trade", fill: "Fill") -> None:
        if fill.execution.orderRef != self.order_ref:
            return
        if fill.contract.conId != self.con_id:
            return

        conId = fill.contract.conId
        shares = fill.execution.shares
        signed_qty = shares if fill.execution.side == "BOT" else -shares

        pos = self._positions.setdefault(conId, {
            "contract": fill.contract,
            "position": 0.0,
            "cost_basis": 0.0,
            "avg_cost": 0.0,
        })
        old_position = pos["position"]
        new_position = old_position + signed_qty

        # Reversal: net qty crosses zero AND stays non-zero (e.g. long 100, sell 150 → short 50).
        # Partial / full exit (same-side reduction): avg_cost preserved, cost_basis scales.
        # New entry / add: avg_cost becomes weighted average.
        if old_position != 0.0 and new_position != 0.0 and (old_position > 0) != (new_position > 0):
            pos["avg_cost"] = fill.execution.price
            pos["cost_basis"] = new_position * fill.execution.price
        elif old_position != 0.0 and (
            (old_position > 0 and signed_qty < 0) or (old_position < 0 and signed_qty > 0)
        ):
            pos["cost_basis"] = new_position * pos["avg_cost"]
        else:
            pos["cost_basis"] += signed_qty * fill.execution.price
            pos["avg_cost"] = (pos["cost_basis"] / new_position) if new_position else 0.0
        pos["position"] = new_position

        if abs(pos["position"]) < 1e-9:
            del self._positions[conId]

    def on_position(self, pos: "Position") -> None:
        """Broker-authoritative snapshot — used to detect drift from cache."""
        if pos.contract.conId != self.con_id:
            return
        conId = pos.contract.conId

        if pos.position == 0.0:
            self._positions.pop(conId, None)
            return

        cached = self._positions.get(conId)
        if cached is None:
            logger.warning(
                f"Adopting unexpected broker position for {self.symbol} "
                f"(conId={conId}, qty={pos.position}). Ensure all managed instruments are flat at startup."
            )
            self._positions[conId] = {
                "contract": pos.contract,
                "position": pos.position,
                "cost_basis": pos.position * pos.avgCost,
                "avg_cost": pos.avgCost,
            }
            return

        if abs(cached["position"] - pos.position) > 1e-9:
            logger.warning(
                f"Position drift for {self.symbol}: cache={cached['position']}, "
                f"broker={pos.position}. Trusting broker."
            )
            cached["position"] = pos.position
            cached["avg_cost"] = pos.avgCost
            cached["cost_basis"] = pos.position * pos.avgCost

    def get_open(self) -> list[dict]:
        return [{
            "symbol": p["contract"].symbol,
            "contract": p["contract"],
            "position": p["position"],
            "avg_cost": p["avg_cost"],
        } for p in self._positions.values()]

    async def resync(self, ib: IB) -> None:
        """Post-reconnect rebuild from broker state. Clears and refills the cache."""
        self._positions.clear()
        for pos in await ib.reqPositionsAsync():
            if pos.contract.conId == self.con_id and pos.position != 0.0:
                self._positions[pos.contract.conId] = {
                    "contract": pos.contract,
                    "position": pos.position,
                    "cost_basis": pos.position * pos.avgCost,
                    "avg_cost": pos.avgCost,
                }
