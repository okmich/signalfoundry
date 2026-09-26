"""The account as the Admin sees it, broker-neutral, and the two ports a broker library implements for it.

The Admin's tasks decide from these value objects only, so every decision is testable without a terminal and the same
tasks serve any broker. A broker library (MT5 today) supplies:

* an :class:`AccountSource`, read once per cycle for the snapshot plus deal history on demand; and
* a :class:`BrokerActions`, the only two things any task can do to the book (spec §12.1, invariant 3).

Every time here is an aware UTC instant: the broker library converts its server clock before it builds these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from .enums import BookActionOutcome, DealEntry, DealKind, PendingOrderType, PositionSide


@dataclass(frozen=True)
class AccountInfo:
    login: int
    server: str
    currency: str
    balance: float
    equity: float


@dataclass(frozen=True)
class Position:
    ticket: int
    symbol: str
    magic: int
    side: PositionSide
    volume: float
    price_open: float
    opened_utc: datetime | None
    profit: float


@dataclass(frozen=True)
class PendingOrder:
    ticket: int
    symbol: str
    magic: int
    order_type: PendingOrderType
    volume: float
    price: float
    setup_utc: datetime | None   # None when the broker's clock could not be converted: age unknown, never acted on


@dataclass(frozen=True)
class Deal:
    ticket: int
    order: int
    position_id: int
    symbol: str
    magic: int
    kind: DealKind
    entry: DealEntry
    volume: float
    price: float
    profit: float
    commission: float
    swap: float
    fee: float
    time_utc: datetime

    @property
    def net(self) -> float:
        """What this deal did to the balance: profit (or the amount of a balance operation) plus costs."""
        return self.profit + self.commission + self.swap + self.fee


@dataclass(frozen=True)
class AccountSnapshot:
    """One cycle's read of the account, shared by every task (spec §7.1 step 2)."""

    taken_utc: datetime
    info: AccountInfo
    positions: tuple[Position, ...] = field(default_factory=tuple)
    orders: tuple[PendingOrder, ...] = field(default_factory=tuple)


@runtime_checkable
class AccountSource(Protocol):
    """Read-only access to the governed account. Methods raise on a failed read; the host turns that into a degraded
    cycle. Nothing here may send an order."""

    def account_info(self) -> AccountInfo | None:
        """The live terminal's account, or ``None`` when the terminal cannot report it."""
        ...

    def positions(self) -> list[Position]:
        ...

    def pending_orders(self) -> list[PendingOrder]:
        ...

    def deals(self, from_utc: datetime, to_utc: datetime) -> list[Deal]:
        """Every deal with ``from_utc <= time_utc <= to_utc``, oldest first."""
        ...


@dataclass(frozen=True)
class BookActionResult:
    outcome: BookActionOutcome
    error: str | None = None
    retcode: int | None = None


@runtime_checkable
class BrokerActions(Protocol):
    """The two book actions a broker library provides to the Admin, and nothing else. Both reduce risk."""

    def cancel_pending(self, order: PendingOrder) -> BookActionResult:
        """Remove a resting pending order. ``SKIPPED_FILLED`` when it is no longer there."""
        ...

    def close_position(self, position: Position) -> BookActionResult:
        """Close a position in full by ticket. ``SKIPPED_CLOSED`` when it is no longer there."""
        ...
