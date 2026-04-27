"""Base IB position manager — async cancel-then-close lifecycle, in-place stop modification."""
import logging
from abc import abstractmethod
from typing import Any, List, Dict, Optional

from ib_async import IB, Contract

from okmich_quant_core import BasePositionManager, StrategyConfig

from ..functions.ib import (
    cancel_order, close_position as ib_close, modify_order, place_stop_order,
)

logger = logging.getLogger(__name__)


class BaseIBPositionManager(BasePositionManager):
    """Base IB position manager.

    SL is implemented as a post-entry protective stop order submitted after the
    position exists — NOT an atomic bracket child. Strategies needing atomic
    entry+stop should use ``place_bracket_order`` directly.

    Stop lifecycle: ``cancel_protective_stop`` MUST be called before every close
    (manager-driven or signal-driven) to prevent an orphan stop from firing on a
    flat position.
    """

    def __init__(self, ib: IB, contract: Contract, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config)
        self.ib = ib
        self.contract = contract
        self.strategy_config = strategy_config
        self.kwargs = kwargs
        self._sl_trades: dict[int, Any] = {}

    @abstractmethod
    def _evaluate_position(
        self, position: dict
    ) -> tuple[bool, Optional[float], Optional[float]]:
        """Return (should_close, new_sl, new_tp).

        ``should_close`` — True to close immediately with a market order.
        ``new_sl`` — updated stop-loss price, or None to leave unchanged.
        ``new_tp`` — updated take-profit price, or None to leave unchanged.
        """

    async def manage_positions(self, positions: list[dict]) -> set[int]:
        """Async management dispatch.

        Calls ``_evaluate_position`` for each position, then awaits close /
        modify as directed. Returns the set of conIds for positions where a
        close order was submitted, so the strategy can avoid double-closing on
        the same bar before async fill events update the cache.
        """
        closing_con_ids: set[int] = set()
        for position in positions:
            should_close, sl, tp = self._evaluate_position(position)
            if should_close:
                if await self.close_position(position):
                    closing_con_ids.add(position["contract"].conId)
            elif sl is not None or tp is not None:
                await self.modify_position(position, sl=sl, tp=tp)
        return closing_con_ids

    async def cancel_protective_stop(self, position: dict) -> bool:
        """Cancel the protective stop and return True ONLY when confirmed.

        The handle is retained on failure so the caller can retry; the caller
        MUST NOT proceed with closing the position until this returns True.
        """
        conId = position["contract"].conId
        existing = self._sl_trades.get(conId)
        if existing is None:
            return True
        cancelled = await cancel_order(self.ib, existing)
        if cancelled:
            self._sl_trades.pop(conId, None)
            return True
        logger.warning(
            f"Protective stop for {position['contract'].symbol} (orderId={existing.order.orderId}) "
            "could not be confirmed cancelled — keeping handle, aborting close."
        )
        return False

    async def close_position(self, position: dict) -> bool:
        """Cancel protective stop (confirmed) then close with a market order."""
        if not await self.cancel_protective_stop(position):
            return False
        try:
            await ib_close(self.ib, position, self.strategy_config.magic)
            return True
        except Exception:
            return False

    async def modify_position(self, position: dict, sl: Optional[float] = None,
                              tp: Optional[float] = None) -> bool:
        """Update the protective stop. In-place modification preserves protection.

        TP is not implemented in the base class — subclasses needing a TP limit
        order must override this method.
        """
        if tp is not None:
            raise NotImplementedError(
                "TP modification is not implemented in BaseIBPositionManager. "
                "Override modify_position in a concrete subclass to add take-profit logic."
            )
        if sl is None:
            return True
        conId = position["contract"].conId
        size = position["position"]
        action = "SELL" if size > 0 else "BUY"
        existing = self._sl_trades.get(conId)

        if existing:
            await modify_order(self.ib, existing, stop_price=sl, quantity=abs(size))
        else:
            trade = await place_stop_order(
                self.ib, position["contract"], action, abs(size), sl,
                self.strategy_config.magic,
            )
            self._sl_trades[conId] = trade
        return True

    # ---- BasePositionManager abstract stubs ----
    # The IB workflow drives management through manage_positions(positions)
    # and _evaluate_position. The polled MT5-style abstracts below are not used
    # but must exist for ABC instantiation.

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Not used in IB. The strategy sources positions from IBPositionCache."""
        return []

    def manage_long_position(self, position: Dict[str, Any], flag: bool):
        """Not used in IB; manage_positions(positions) is the entry point."""
        raise NotImplementedError(
            "IB position managers dispatch via manage_positions(positions); "
            "manage_long_position is not used."
        )

    def manage_short_position(self, position: Dict[str, Any], flag: bool):
        raise NotImplementedError(
            "IB position managers dispatch via manage_positions(positions); "
            "manage_short_position is not used."
        )
