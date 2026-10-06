"""Fakes for Account Admin tests: an in-memory broker account and a recording notifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from okmich_quant_core.account_admin import (AccountInfo, BookActionOutcome, BookActionResult, Deal, DealEntry, DealKind,
                                             PendingOrder, PendingOrderType, Position, PositionSide)
from okmich_quant_core.account_admin.owners import MagicOwner
from okmich_quant_core.account_admin.tasks.base import TaskContext

UTC = timezone.utc
LOGIN = 12345678
SERVER = "ICMarketsSC-Demo"


def t(s: str) -> datetime:
    """``t("2026-09-24 13:00")`` -> aware UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


@dataclass
class FakeAccount:
    """A broker account the tests move by hand."""

    balance: float = 100_000.0
    equity: float = 100_000.0
    login: int = LOGIN
    server: str = SERVER
    currency: str = "USD"
    open_positions: list[Position] = field(default_factory=list)
    orders: list[PendingOrder] = field(default_factory=list)
    deal_list: list[Deal] = field(default_factory=list)
    fail_reads: bool = False
    info_none: bool = False
    cancel_outcomes: dict[int, BookActionOutcome] = field(default_factory=dict)
    cancelled: list[int] = field(default_factory=list)
    closed: list[int] = field(default_factory=list)
    touched: int = 0

    # AccountSource
    def account_info(self):
        self.touched += 1
        if self.fail_reads:
            raise ConnectionError("terminal IPC timeout")
        if self.info_none:
            return None
        return AccountInfo(self.login, self.server, self.currency, self.balance, self.equity)

    def positions(self):
        return list(self.open_positions)

    def pending_orders(self):
        return list(self.orders)

    def deals(self, from_utc: datetime, to_utc: datetime):
        return sorted((d for d in self.deal_list if from_utc <= d.time_utc <= to_utc), key=lambda d: d.time_utc)

    # BrokerActions
    def cancel_pending(self, order: PendingOrder) -> BookActionResult:
        outcome = self.cancel_outcomes.get(order.ticket, BookActionOutcome.DONE)
        if outcome is BookActionOutcome.DONE:
            self.cancelled.append(order.ticket)
            self.orders = [o for o in self.orders if o.ticket != order.ticket]
            return BookActionResult(outcome)
        if outcome is BookActionOutcome.SKIPPED_FILLED:
            self.orders = [o for o in self.orders if o.ticket != order.ticket]
            return BookActionResult(outcome, "order not found")
        return BookActionResult(outcome, "trade context busy", 10018)

    def close_position(self, position: Position) -> BookActionResult:
        self.closed.append(position.ticket)
        return BookActionResult(BookActionOutcome.DONE)

    # helpers
    def add_deal(self, ticket: int, when: datetime, net: float = 0.0, *, kind: DealKind = DealKind.SELL,
                 entry: DealEntry = DealEntry.OUT, magic: int = 1, symbol: str = "EURUSD", volume: float = 0.1) -> Deal:
        deal = Deal(ticket=ticket, order=ticket, position_id=ticket, symbol=symbol, magic=magic, kind=kind, entry=entry,
                    volume=volume, price=1.1, profit=net, commission=0.0, swap=0.0, fee=0.0, time_utc=when)
        self.deal_list.append(deal)
        return deal


def order(ticket: int, setup: datetime | None, *, magic: int = 7, kind: PendingOrderType = PendingOrderType.BUY_LIMIT,
          symbol: str = "EURUSD") -> PendingOrder:
    return PendingOrder(ticket=ticket, symbol=symbol, magic=magic, order_type=kind, volume=0.1, price=1.05,
                        setup_utc=setup)


def position(ticket: int, *, magic: int = 7, symbol: str = "EURUSD") -> Position:
    return Position(ticket=ticket, symbol=symbol, magic=magic, side=PositionSide.LONG, volume=0.1, price_open=1.1,
                    opened_utc=None, profit=0.0)


class RecordingNotifier:
    def __init__(self):
        self.alerts: list[tuple[str, str, str]] = []

    def send_alert(self, level, title, body):
        self.alerts.append((str(level), title, body))

    def titles(self) -> list[str]:
        return [a[1].split(" [")[0] for a in self.alerts]


class RecordingPort:
    def __init__(self, account: FakeAccount, degraded: bool = False):
        self.account, self.degraded, self.calls = account, degraded, []

    def cancel_pending(self, order, *, reason):
        self.calls.append(("cancel", order.ticket, reason))
        if self.degraded:
            return BookActionResult(BookActionOutcome.REFUSED, "degraded")
        return self.account.cancel_pending(order)

    def close_position(self, position, *, reason):
        self.calls.append(("close", position.ticket, reason))
        return self.account.close_position(position)


def make_ctx(account: FakeAccount, now: datetime, slot: dict | None = None, *, degraded: bool = False,
             requests: list | None = None, owners: dict[int, MagicOwner] | None = None, first_cycle: bool = False,
             state_load: str = "ok", previous_output: dict | None = None, valid_for_s: int = 240,
             port: RecordingPort | None = None) -> TaskContext:
    from okmich_quant_core.account_admin import AccountSnapshot
    snapshot = None
    if not degraded:
        info = account.account_info()
        snapshot = AccountSnapshot(taken_utc=now, info=info, positions=tuple(account.open_positions),
                                   orders=tuple(account.orders))
    return TaskContext(now=now, account="icmarkets.demo", degraded=degraded, degraded_reason="test degraded" if degraded else None,
                       snapshot=snapshot, slot=slot or {}, requests=requests or [],
                       actions=port or RecordingPort(account, degraded), deals=account.deals,
                       owners=lambda: dict(owners or {}), valid_for_s=valid_for_s, broker_label="icmarkets",
                       expected_login=LOGIN, expected_server=SERVER, first_cycle=first_cycle, state_load=state_load,
                       previous_output=previous_output)


def prop_guard_entry(**policy_overrides: Any) -> dict:
    policy = {
        "initial_capital": 100_000.0, "account_start_utc": "2026-09-01T00:00:00Z", "day_tz": "America/New_York",
        "day_start_hour": 17, "daily_base": "day_start_balance", "daily_limit_reference": "initial_capital",
        "daily_loss_pct": 5.0, "daily_warn_fraction": 0.6, "max_loss_pct": 10.0, "max_loss_mode": "static",
        "max_loss_trail_locks_at_initial": False, "max_warn_fraction": 0.8, "profit_target_pct": 10.0,
        "target_measure": "balance",
        "conditions": {"daily_loss": {"directive": "NO_OPS", "latch": "trading_day"},
                       "daily_warn": {"directive": "NO_ENTRY_OPS", "latch": "none"},
                       "max_loss": {"directive": "NO_OPS", "latch": "manual"},
                       "max_warn": {"directive": "NO_ENTRY_OPS", "latch": "none"},
                       "target": {"directive": "NO_OPS", "latch": "manual"}},
        "calendar": [],
    }
    policy.update(policy_overrides)
    return {"kind": "prop_guard", "checks": {"close_grace_s": 120, "obey_grace_s": 30}, "max_override_s": 86_400,
            "policy": policy}


def admin_config(tasks: list[dict] | None = None, **clock_overrides: Any) -> dict:
    clock = {"cycle_s": 20, "jitter_s": 5, "valid_for_s": 240,
             "blackout": [{"every_s": 60, "offset_s": 0, "length_s": 3}, {"every_s": 300, "offset_s": 0, "length_s": 8},
                          {"every_s": 30, "offset_s": 0, "length_s": 2}]}
    clock.update(clock_overrides)
    return {"kind": "account_admin", "runner": "_account_admin", "clock": clock, "requests": {"request_ttl_s": 900},
            "tasks": tasks if tasks is not None else [prop_guard_entry()]}
