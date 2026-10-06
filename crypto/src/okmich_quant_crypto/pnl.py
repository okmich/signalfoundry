"""Average-cost bookkeeping shared by the perp position resolution and the spot inventory ledger.

Conventions (they match ``core.ClosedTrade``, where ``net_profit = profit + commission + swap``):

* ``realized`` - gross P&L in quote currency, before fees and funding;
* fees are accumulated as a POSITIVE cost in ``fees_quote`` and reported to core as a NEGATIVE ``commission``;
* a spot BUY whose fee is charged in the base coin delivers ``base_qty - fee_base``: the ledger holds what was
  received, and the base fee is valued at the fill price inside ``fees_quote`` - so gross + fees still equals the
  actual quote cash in and out.
"""
from dataclasses import dataclass, replace

from .enums import OrderSide
from .models import Fill


@dataclass
class Book:
    """A one-way average-cost book for one symbol. ``qty`` is signed base quantity."""
    qty: float = 0.0
    avg_price: float = 0.0
    realized: float = 0.0
    fees_quote: float = 0.0
    fee_unresolved: bool = False
    entry_qty: float = 0.0
    entry_notional: float = 0.0
    closed_qty: float = 0.0
    exit_notional: float = 0.0
    eps: float = 1e-12

    def qty_delta(self, fill: Fill, spot: bool) -> float:
        if not spot or fill.fee_base == 0.0:
            return fill.signed_qty
        if fill.side is OrderSide.BUY:
            return fill.base_qty - fill.fee_base
        return -(fill.base_qty + fill.fee_base)

    def apply(self, fill: Fill, spot: bool = False) -> None:
        delta = self.qty_delta(fill, spot)
        price = fill.price
        self.fees_quote += fill.fee_quote
        self.fee_unresolved = self.fee_unresolved or fill.fee_unresolved
        if delta == 0.0:
            return
        if self.is_flat or (self.qty > 0) == (delta > 0):
            new_qty = self.qty + delta
            self.avg_price = (self.avg_price * abs(self.qty) + price * abs(delta)) / abs(new_qty)
            self.qty = new_qty
            self.entry_qty += abs(delta)
            self.entry_notional += price * abs(delta)
            return
        closing = min(abs(delta), abs(self.qty))
        direction = 1.0 if self.qty > 0 else -1.0
        self.realized += (price - self.avg_price) * closing * direction
        self.closed_qty += closing
        self.exit_notional += price * closing
        remainder = self.qty + delta
        if abs(remainder) <= self.eps:
            self.qty, self.avg_price = 0.0, 0.0
        elif (remainder > 0) != (self.qty > 0):
            # Reversal: the excess opens a new position at this fill's price.
            self.qty, self.avg_price = remainder, price
            self.entry_qty += abs(remainder)
            self.entry_notional += price * abs(remainder)
        else:
            self.qty = remainder

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) <= self.eps

    @property
    def avg_entry(self) -> float:
        return self.entry_notional / self.entry_qty if self.entry_qty else 0.0

    @property
    def avg_exit(self) -> float:
        return self.exit_notional / self.closed_qty if self.closed_qty else 0.0


def scale_fill(fill: Fill, ratio: float) -> Fill:
    """The part ``ratio`` of a fill (quantity and fees pro rata) - used to split a fill that flips a position."""
    return replace(fill, base_qty=fill.base_qty * ratio, fee_quote=fill.fee_quote * ratio,
                   fee_base=fill.fee_base * ratio)


def last_round_trip(fills: list[Fill], spot: bool = False, eps: float = 1e-12,
                    final_qty: float = 0.0) -> tuple[list[Fill], bool]:
    """The fills of the LAST round trip in ``fills`` (ordered by time), and whether it is complete.

    ``fills`` may start anywhere before the round trip (the caller fetches with a generous lookback) but must END at
    the close being resolved. ``final_qty`` is the signed position AFTER the last fill: 0 for a plain close, the new
    position's size when the close was a flip (long -> short in one fill). A fill that crosses zero belongs to two
    round trips and is split pro rata: its closing part ends the earlier trip, its opening part starts the later one.

    ``complete`` is False when no start point (flat, or a crossing) exists inside the window - it does not reach
    back to the round trip's first fill. The caller must then report the close as UNRESOLVED rather than describe it
    wrongly.
    """
    # Walk BACKWARD from the close: the position before fill i is the position after it minus its delta. Walking
    # forward from the window start would instead assume the book was flat THERE - wrong whenever the window opens
    # inside an earlier round trip (back-to-back trades within the lookback).
    if not fills:
        return [], False
    fills = list(fills)
    deltas = [Book().qty_delta(f, spot) for f in fills]
    tail: list[Fill] = []
    position_after = 0.0
    if abs(final_qty) > eps:
        # Strip the NEW position's fills (from its opening back to the fill that started it) off the end.
        after = final_qty
        for j in range(len(fills) - 1, -1, -1):
            before = after - deltas[j]
            if abs(before) <= eps:              # the new position opened from flat: the old trip closed earlier
                fills, deltas = fills[:j], deltas[:j]
                break
            if (before > 0) != (after > 0):     # fill j flipped: its closing part ends the old trip
                tail = [scale_fill(fills[j], abs(before) / abs(deltas[j]))]
                fills, deltas, position_after = fills[:j], deltas[:j], before
                break
            after = before
        else:
            return [], False
    for i in range(len(fills) - 1, -1, -1):
        position_before = position_after - deltas[i]
        if abs(position_before) <= eps:
            return _checked(fills[i:] + tail, spot, eps)
        if abs(position_after) > eps and (position_before > 0) != (position_after > 0):
            # Fill i flipped the position: only its opening part (|position_after|) belongs to this trip.
            opening = scale_fill(fills[i], abs(position_after) / abs(deltas[i]))
            return _checked([opening] + fills[i + 1:] + tail, spot, eps)
        position_after = position_before
    return [], False


def _checked(trip: list[Fill], spot: bool, eps: float) -> tuple[list[Fill], bool]:
    book = summarize(trip, spot, eps)
    # Sanity: the trip must open from flat, never pass through flat before its end, and end flat.
    if book.is_flat and book.entry_qty > eps:
        return trip, True
    return [], False


def summarize(fills: list[Fill], spot: bool = False, eps: float = 1e-12) -> Book:
    book = Book(eps=eps)
    for fill in fills:
        book.apply(fill, spot)
    return book
