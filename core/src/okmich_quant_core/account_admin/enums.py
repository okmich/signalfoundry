"""Closed vocabularies of the Account Admin (ACCOUNT_ADMIN_SPEC). Every fixed set of values is an enum here, so the
writer (the Admin) and every reader (the trading-system guard, ops tooling) share one definition."""

from __future__ import annotations

import enum
from typing import Iterable


class AccountDirective(enum.StrEnum):
    """What every system on the account may do right now (spec §5). Ordered by restrictiveness."""

    ALL_OPS = "ALL_OPS"             # every operation allowed
    NO_ENTRY_OPS = "NO_ENTRY_OPS"   # nothing that opens or adds; everything else as normal
    NO_OPS = "NO_OPS"               # hold nothing: close what is held, open nothing

    @property
    def rank(self) -> int:
        return _DIRECTIVE_RANK[self]


_DIRECTIVE_RANK = {AccountDirective.ALL_OPS: 0, AccountDirective.NO_ENTRY_OPS: 1, AccountDirective.NO_OPS: 2}

#: The directives a condition, calendar window or override may impose: they can only tighten (spec §6.4-§6.6, §8.2).
RESTRICTIVE_DIRECTIVES = frozenset({AccountDirective.NO_ENTRY_OPS, AccountDirective.NO_OPS})


def most_restrictive(directives: Iterable[AccountDirective]) -> AccountDirective:
    """The most restrictive of ``directives``; ``ALL_OPS`` when there are none (spec §6.6)."""
    return max(directives, key=lambda d: d.rank, default=AccountDirective.ALL_OPS)


class DirectiveSource(enum.StrEnum):
    """Why a reader applies the directive it applies (spec §10.2)."""

    ABSENT = "absent"
    FILE = "file"
    STALE = "stale"
    INVALID = "invalid"
    WRONG_ACCOUNT = "wrong_account"


class Condition(enum.StrEnum):
    """One reason the directive may be restrictive (spec §6.4)."""

    DAILY_LOSS = "daily_loss"
    DAILY_WARN = "daily_warn"
    MAX_LOSS = "max_loss"
    MAX_WARN = "max_warn"
    TARGET = "target"
    CALENDAR = "calendar"
    OVERRIDE = "override"
    BALANCE_OPERATION = "balance_operation"
    STATE_LOST = "state_lost"
    ADMIN_DEGRADED = "admin_degraded"


#: The loss/target conditions whose directive and latch the policy configures (spec §6.1 `conditions`).
POLICY_CONDITIONS = (Condition.DAILY_LOSS, Condition.DAILY_WARN, Condition.MAX_LOSS, Condition.MAX_WARN,
                     Condition.TARGET)


class LatchScope(enum.StrEnum):
    NONE = "none"                   # active only while the condition holds
    TRADING_DAY = "trading_day"     # active until the next day boundary
    MANUAL = "manual"               # active until an operator latch reset


class DailyBase(enum.StrEnum):
    DAY_START_BALANCE = "day_start_balance"
    DAY_START_MAX_BALANCE_EQUITY = "day_start_max_balance_equity"


class LimitReference(enum.StrEnum):
    INITIAL_CAPITAL = "initial_capital"
    DAILY_BASE = "daily_base"


class MaxLossMode(enum.StrEnum):
    STATIC = "static"
    TRAILING_BALANCE = "trailing_balance"
    TRAILING_EQUITY = "trailing_equity"


class TargetMeasure(enum.StrEnum):
    BALANCE = "balance"
    EQUITY = "equity"


class CalendarKind(enum.StrEnum):
    WEEKLY = "weekly"
    DATED = "dated"


class RequestKind(enum.StrEnum):
    """``prop_guard``'s operator requests (spec §8.2)."""

    RESET_LATCH = "reset_latch"
    OVERRIDE = "override"
    CLEAR_OVERRIDE = "clear_override"


class RequestOutcome(enum.StrEnum):
    APPLIED = "applied"
    REJECTED = "rejected"


class AdminEvent(enum.StrEnum):
    """Audit-log event types (spec §7.7)."""

    ADMIN_STARTED = "admin_started"
    ADMIN_STOPPED = "admin_stopped"
    CYCLE = "cycle"
    DIRECTIVE_CHANGED = "directive_changed"
    LATCH_TRIPPED = "latch_tripped"
    LATCH_CLEARED = "latch_cleared"
    DAY_ROLLED = "day_rolled"
    REQUEST_APPLIED = "request_applied"
    REQUEST_REJECTED = "request_rejected"
    OBEDIENCE_BREACH = "obedience_breach"
    ORPHAN = "orphan"
    WRITE_FAILED = "write_failed"
    ADMIN_DEGRADED = "admin_degraded"
    TASK_FAILED = "task_failed"
    BOOK_ACTION = "book_action"
    OVERRIDE_EXPIRED = "override_expired"


class AdminTaskKind(enum.StrEnum):
    """The registered admin tasks (spec §12). A new kind is specified in the spec before it is added here."""

    PROP_GUARD = "prop_guard"
    PENDING_ORDER_CLEANUP = "pending_order_cleanup"


class CleanupScope(enum.StrEnum):
    ALL = "all"
    ORPHANED_ONLY = "orphaned_only"
    MAGICS = "magics"


class BookActionKind(enum.StrEnum):
    """The only two things an admin task can do to the book (spec §12.1, invariant 3)."""

    CANCEL_PENDING = "cancel_pending"
    CLOSE_POSITION = "close_position"


class BookActionOutcome(enum.StrEnum):
    DONE = "done"
    FAILED = "failed"
    SKIPPED_FILLED = "skipped_filled"   # the pending order was no longer there (filled or removed) when cancelled
    SKIPPED_CLOSED = "skipped_closed"   # the position was no longer there when closed
    REFUSED = "refused"                 # the host refused it (a degraded cycle)


class PendingOrderType(enum.StrEnum):
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"
    BUY_STOP_LIMIT = "buy_stop_limit"
    SELL_STOP_LIMIT = "sell_stop_limit"


class PositionSide(enum.StrEnum):
    LONG = "long"
    SHORT = "short"


class DealEntry(enum.StrEnum):
    IN = "in"           # opens or adds
    OUT = "out"         # closes or reduces
    INOUT = "inout"     # reversal: closes and opens
    OUT_BY = "out_by"   # closed by an opposite position
    NONE = "none"       # not a trade (balance operations, fees)


class DealKind(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"
    BALANCE = "balance"     # deposit or withdrawal
    CREDIT = "credit"
    BONUS = "bonus"
    OTHER = "other"         # fees, commissions, corrections booked as their own deals

    @property
    def is_balance_operation(self) -> bool:
        """A movement of money into or out of the account, not a trading result (spec §6.4 `balance_operation`)."""
        return self in (DealKind.BALANCE, DealKind.CREDIT, DealKind.BONUS)


class AlertLevel(enum.StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
