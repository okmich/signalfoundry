"""Broker-neutral description of a position that has left the book.

The detection of a close is broker-specific and cannot be shared: MT5 is polled and only ever learns a position
ended by noticing it is no longer in ``get_positions``, while IB is event-driven and is told by ``fillEvent``. What
IS shared is everything after detection — deciding it really closed, attributing WHY, announcing it exactly once.
:class:`ClosedTrade` is the handoff between the two halves: each broker builds one from whatever it has (an MT5
history deal, an IB fill) and passes it to ``BaseStrategy._on_position_closed``, which owns the shared half.

Carrying the payload rather than a callback is deliberate. ``BaseIBStrategy`` is async top to bottom; if the shared
handler had to ask the broker to resolve the close, that call would be awaitable on IB and plain on MT5, and the
core would have to be async-aware to host it. Resolution therefore happens broker-side, where the right calling
convention is already in scope, and the core stays synchronous.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


class CloseReason(enum.StrEnum):
    """Why a position left the book, normalised across brokers.

    ``STRATEGY`` means this system asked for the close; the specific rule is carried separately in
    :attr:`ClosedTrade.strategy_reason` because only the strategy knows it (``"ctl_flip"``, ``"session_flatten"``).
    ``MANUAL`` means a human closed it in the terminal — operationally very different from a strategy close and
    worth alerting on differently, so it is not folded into ``STRATEGY``.
    """
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    STOP_OUT = "stop_out"                # margin liquidation
    STRATEGY = "strategy"                # this system requested it
    MANUAL = "manual"                    # closed by a human, outside the system
    EXPIRED = "expired"
    UNKNOWN = "unknown"                  # gone from the book, cause unresolvable


@dataclass(frozen=True)
class ClosedTrade:
    """One position, after the fact. ``key`` is the broker-neutral identity from ``BaseStrategy._position_key``.

    ``profit`` is the broker's REALISED figure, not a snapshot of unrealised P/L taken before the close request —
    that distinction is the whole point of resolving from broker history rather than reporting what the position
    dict said on the way out.
    """
    key: str
    symbol: str
    magic: Optional[int] = None
    reason: CloseReason = CloseReason.UNKNOWN
    #: The strategy's own label for a close it requested (free-form, strategy-defined). Only meaningful when
    #: ``reason`` is STRATEGY, and only present when the close was recorded via ``_note_close_intent``.
    strategy_reason: Optional[str] = None
    volume: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    #: Last position snapshot seen while it was still open. The fallback description when resolution fails: a
    #: position that has already vanished cannot be queried, so without this a failed lookup leaves only an id.
    last_seen: dict[str, Any] = field(default_factory=dict)
    #: False when the broker could not tell us how it ended and the fields above are the last-seen values rather
    #: than resolved ones. Consumers that aggregate P/L MUST check this before summing ``profit``.
    resolved: bool = True

    @property
    def net_profit(self) -> float:
        """Realised P/L including the costs the broker books separately (MT5 reports swap and commission apart)."""
        return self.profit + self.commission + self.swap

    def describe(self) -> str:
        """One-line human summary for the text log and notifications."""
        why = self.reason.value
        if self.reason == CloseReason.STRATEGY and self.strategy_reason:
            why = f"{why}:{self.strategy_reason}"
        if not self.resolved:
            why = f"{why} (UNRESOLVED)"
        return (f"{self.symbol} {self.key} closed [{why}] vol={self.volume} entry={self.entry_price} "
                f"exit={self.exit_price} profit={self.net_profit:+.2f}")
