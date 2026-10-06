"""The governed account read from the MT5 terminal, as the Admin's broker-neutral values (ACCOUNT_ADMIN_SPEC §7.1).

Read-only: nothing here sends an order. Every server-clock time is converted to UTC with the stated
:class:`ServerClock`. A failed read raises (never returns an empty book): the host turns it into a degraded cycle, since
an unobserved book must never read as a flat one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from okmich_quant_core.account_admin import (AccountInfo, Deal, DealEntry, DealKind, PendingOrder, PendingOrderType,
                                             Position, PositionSide)

from .server_clock import ServerClock

logger = logging.getLogger(__name__)


class Mt5ReadError(RuntimeError):
    """The terminal did not answer a read."""


def _const(mt5: Any, name: str, default: int) -> int:
    return int(getattr(mt5, name, default))


class Mt5AccountSource:
    """:class:`okmich_quant_core.account_admin.AccountSource` over the process-global MT5 terminal.

    ``mt5`` is the ``MetaTrader5`` module (injectable for tests).
    """

    #: History queries are widened by this margin and filtered after conversion, so a query's own clock edge can
    #: never drop a deal at the window's boundary.
    QUERY_MARGIN = timedelta(days=1)

    def __init__(self, mt5: Any, clock: ServerClock):
        self.mt5 = mt5
        self.clock = clock
        self._order_types = {
            _const(mt5, "ORDER_TYPE_BUY_LIMIT", 2): PendingOrderType.BUY_LIMIT,
            _const(mt5, "ORDER_TYPE_SELL_LIMIT", 3): PendingOrderType.SELL_LIMIT,
            _const(mt5, "ORDER_TYPE_BUY_STOP", 4): PendingOrderType.BUY_STOP,
            _const(mt5, "ORDER_TYPE_SELL_STOP", 5): PendingOrderType.SELL_STOP,
            _const(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6): PendingOrderType.BUY_STOP_LIMIT,
            _const(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7): PendingOrderType.SELL_STOP_LIMIT,
        }
        self._deal_kinds = {
            _const(mt5, "DEAL_TYPE_BUY", 0): DealKind.BUY, _const(mt5, "DEAL_TYPE_SELL", 1): DealKind.SELL,
            _const(mt5, "DEAL_TYPE_BALANCE", 2): DealKind.BALANCE, _const(mt5, "DEAL_TYPE_CREDIT", 3): DealKind.CREDIT,
            _const(mt5, "DEAL_TYPE_BONUS", 6): DealKind.BONUS,
        }
        self._deal_entries = {
            _const(mt5, "DEAL_ENTRY_IN", 0): DealEntry.IN, _const(mt5, "DEAL_ENTRY_OUT", 1): DealEntry.OUT,
            _const(mt5, "DEAL_ENTRY_INOUT", 2): DealEntry.INOUT, _const(mt5, "DEAL_ENTRY_OUT_BY", 3): DealEntry.OUT_BY,
        }
        self._position_buy = _const(mt5, "POSITION_TYPE_BUY", 0)

    def account_info(self) -> AccountInfo | None:
        info = self.mt5.account_info()
        if info is None:
            return None
        return AccountInfo(login=int(info.login), server=str(info.server), currency=str(info.currency),
                           balance=float(info.balance), equity=float(info.equity))

    def positions(self) -> list[Position]:
        rows = self.mt5.positions_get()
        if rows is None:
            raise Mt5ReadError(f"positions_get failed: {self.mt5.last_error()}")
        return [Position(ticket=int(p.ticket), symbol=str(p.symbol), magic=int(p.magic),
                         side=PositionSide.LONG if int(p.type) == self._position_buy else PositionSide.SHORT,
                         volume=float(p.volume), price_open=float(p.price_open),
                         opened_utc=self.clock.to_utc(p.time) if getattr(p, "time", 0) else None,
                         profit=float(p.profit)) for p in rows]

    def pending_orders(self) -> list[PendingOrder]:
        rows = self.mt5.orders_get()
        if rows is None:
            raise Mt5ReadError(f"orders_get failed: {self.mt5.last_error()}")
        out: list[PendingOrder] = []
        for o in rows:
            kind = self._order_types.get(int(o.type))
            if kind is None:
                continue   # a market order in flight is not a resting pending order
            setup = getattr(o, "time_setup", 0)
            out.append(PendingOrder(ticket=int(o.ticket), symbol=str(o.symbol), magic=int(o.magic), order_type=kind,
                                    volume=float(getattr(o, "volume_current", 0.0) or getattr(o, "volume_initial", 0.0)),
                                    price=float(o.price_open), setup_utc=self.clock.to_utc(setup) if setup else None))
        return out

    def deals(self, from_utc: datetime, to_utc: datetime) -> list[Deal]:
        start = self.clock.to_server_epoch(from_utc - self.QUERY_MARGIN)
        end = self.clock.to_server_epoch(to_utc + self.QUERY_MARGIN)
        rows = self.mt5.history_deals_get(start, end)
        if rows is None:
            raise Mt5ReadError(f"history_deals_get failed: {self.mt5.last_error()}")
        out: list[Deal] = []
        for d in rows:
            when = self.clock.to_utc(d.time)
            if not from_utc <= when <= to_utc:
                continue
            out.append(Deal(ticket=int(d.ticket), order=int(d.order), position_id=int(d.position_id), symbol=str(d.symbol),
                            magic=int(d.magic), kind=self._deal_kinds.get(int(d.type), DealKind.OTHER),
                            entry=self._deal_entries.get(int(d.entry), DealEntry.NONE), volume=float(d.volume),
                            price=float(d.price), profit=float(d.profit), commission=float(getattr(d, "commission", 0.0)),
                            swap=float(getattr(d, "swap", 0.0)), fee=float(getattr(d, "fee", 0.0)), time_utc=when))
        return sorted(out, key=lambda d: (d.time_utc, d.ticket))

    def freshest_tick_problem(self, symbols: list[str], now: datetime) -> str | None:
        """Cross-check the stated server clock against live ticks; the first contradiction found, or ``None``."""
        for symbol in symbols:
            try:
                tick = self.mt5.symbol_info_tick(symbol)
            except Exception:
                continue
            if tick is not None and getattr(tick, "time", 0):
                problem = self.clock.tick_contradicts(tick.time, now)
                if problem:
                    return f"{symbol}: {problem}"
        return None
