"""Plain value objects passed between the venue profiles and the rest of the package."""
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .enums import FillKind, OrderSide
from .timeframe_utils import ms_to_utc


@dataclass(frozen=True)
class Credentials:
    """API credentials. Never logged, never serialised: ``__repr__`` is redacted."""
    api_key: Optional[str] = None
    secret: Optional[str] = None
    password: Optional[str] = None

    def __repr__(self) -> str:
        return f"Credentials(api_key={'***' if self.api_key else None})"

    @property
    def present(self) -> bool:
        return bool(self.api_key and self.secret)

    def fingerprint(self) -> Optional[str]:
        """A non-reversible, stable label for the key (used as account id where the venue cannot report one)."""
        if not self.api_key:
            return None
        return "key-" + hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class ClosedBar:
    """One closed, REST-reconciled candle. ``open_ms`` is the bar's open (label) time, as exchanges label candles."""
    open_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def time(self) -> datetime:
        return ms_to_utc(self.open_ms)

    @classmethod
    def from_row(cls, row: list) -> "ClosedBar":
        return cls(int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                   float(row[5]) if row[5] is not None else 0.0)


@dataclass(frozen=True)
class Fill:
    """One execution, normalised. Quantities are BASE units; fees are split by the currency they were charged in.

    ``fee_quote`` is the total fee valued in the quote currency (fees charged in base are converted at the fill
    price). ``fee_base`` is the part charged in base units, which a spot buyer never receives. ``fee_unresolved`` is
    set when a fee was charged in a third currency (e.g. BNB) this package cannot value without a rate.
    """
    trade_id: str
    order_id: Optional[str]
    client_order_id: Optional[str]
    timestamp_ms: int
    side: OrderSide
    price: float
    base_qty: float
    fee_quote: float = 0.0
    fee_base: float = 0.0
    fee_unresolved: bool = False
    kind: FillKind = FillKind.TRADE
    info: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def signed_qty(self) -> float:
        return self.base_qty if self.side is OrderSide.BUY else -self.base_qty

    def to_state(self) -> dict:
        return {"trade_id": self.trade_id, "order_id": self.order_id, "client_order_id": self.client_order_id,
                "timestamp_ms": self.timestamp_ms, "side": self.side.value, "price": self.price,
                "base_qty": self.base_qty, "fee_quote": self.fee_quote, "fee_base": self.fee_base,
                "fee_unresolved": self.fee_unresolved, "kind": self.kind.value}

    @classmethod
    def from_state(cls, d: dict) -> "Fill":
        return cls(trade_id=str(d["trade_id"]), order_id=d.get("order_id"), client_order_id=d.get("client_order_id"),
                   timestamp_ms=int(d["timestamp_ms"]), side=OrderSide(d["side"]), price=float(d["price"]),
                   base_qty=float(d["base_qty"]), fee_quote=float(d.get("fee_quote", 0.0)),
                   fee_base=float(d.get("fee_base", 0.0)), fee_unresolved=bool(d.get("fee_unresolved", False)),
                   kind=FillKind(d.get("kind", FillKind.TRADE.value)))


@dataclass(frozen=True)
class FundingPayment:
    """A funding settlement on a perpetual. ``amount`` is in the settle currency, POSITIVE = received."""
    timestamp_ms: int
    amount: float
    currency: Optional[str] = None
