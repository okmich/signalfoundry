"""The two book actions MT5 offers the Account Admin (ACCOUNT_ADMIN_SPEC §7.5, §12.1): cancel a pending order and close
a position in full. Nothing else is here: no open, no add, no place, no modify (invariant 3)."""

from __future__ import annotations

import logging
from typing import Any

from okmich_quant_core.account_admin import BookActionOutcome, BookActionResult, PendingOrder, Position, PositionSide

from ..functions.mt5 import resolve_filling_mode

logger = logging.getLogger(__name__)

ADMIN_COMMENT = "account_admin"


class Mt5BrokerActions:
    """:class:`okmich_quant_core.account_admin.BrokerActions` over the MT5 terminal. ``mt5`` is injectable for tests."""

    def __init__(self, mt5: Any):
        self.mt5 = mt5
        self._done = {int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)), int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008))}

    def _send(self, request: dict) -> tuple[bool, str | None, int | None]:
        result = self.mt5.order_send(request)
        if result is None:
            return False, f"order_send returned None: {self.mt5.last_error()}", None
        retcode = int(getattr(result, "retcode", -1))
        return retcode in self._done, f"{getattr(result, 'comment', '')} (retcode {retcode})", retcode

    def _gone(self, rows) -> bool | None:
        """An answered empty query is 'gone'; None is 'the terminal did not answer' (never read as gone)."""
        if rows is None:
            return None
        return len(rows) == 0

    def cancel_pending(self, order: PendingOrder) -> BookActionResult:
        gone = self._gone(self.mt5.orders_get(ticket=order.ticket))
        if gone is None:
            return BookActionResult(BookActionOutcome.FAILED, f"orders_get failed: {self.mt5.last_error()}")
        if gone:
            return BookActionResult(BookActionOutcome.SKIPPED_FILLED, "no longer on the book")
        ok, detail, retcode = self._send({"action": int(getattr(self.mt5, "TRADE_ACTION_REMOVE", 8)),
                                          "order": int(order.ticket), "comment": ADMIN_COMMENT})
        if ok:
            return BookActionResult(BookActionOutcome.DONE, retcode=retcode)
        if self._gone(self.mt5.orders_get(ticket=order.ticket)):   # it filled or was removed while we asked
            return BookActionResult(BookActionOutcome.SKIPPED_FILLED, detail, retcode)
        return BookActionResult(BookActionOutcome.FAILED, detail, retcode)

    def close_position(self, position: Position) -> BookActionResult:
        live = self.mt5.positions_get(ticket=position.ticket)
        gone = self._gone(live)
        if gone is None:
            return BookActionResult(BookActionOutcome.FAILED, f"positions_get failed: {self.mt5.last_error()}")
        if gone:
            return BookActionResult(BookActionOutcome.SKIPPED_CLOSED, "no longer on the book")
        live = live[0]
        tick = self.mt5.symbol_info_tick(position.symbol)
        if tick is None:
            return BookActionResult(BookActionOutcome.FAILED, f"no quote for {position.symbol}")
        closing_sell = position.side is PositionSide.LONG
        request = {"action": int(getattr(self.mt5, "TRADE_ACTION_DEAL", 1)), "position": int(position.ticket),
                   "symbol": position.symbol, "volume": float(live.volume),
                   "type": int(getattr(self.mt5, "ORDER_TYPE_SELL" if closing_sell else "ORDER_TYPE_BUY", 1 if closing_sell else 0)),
                   "price": float(tick.bid if closing_sell else tick.ask), "deviation": 20,
                   "magic": int(position.magic), "comment": ADMIN_COMMENT}
        filling = resolve_filling_mode(position.symbol)
        if filling is not None:
            request["type_filling"] = filling
        ok, detail, retcode = self._send(request)
        if ok:
            return BookActionResult(BookActionOutcome.DONE, retcode=retcode)
        if self._gone(self.mt5.positions_get(ticket=position.ticket)):
            return BookActionResult(BookActionOutcome.SKIPPED_CLOSED, detail, retcode)
        return BookActionResult(BookActionOutcome.FAILED, detail, retcode)
