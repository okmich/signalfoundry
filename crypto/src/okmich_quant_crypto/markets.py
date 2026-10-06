"""Market metadata, precision and size conversion.

Every quantity and price that leaves this package goes through a :class:`MarketSpec`. CCXT's own
``amount_to_precision`` / ``price_to_precision`` do the rounding (they know each venue's precision mode); this module
adds the checks CCXT leaves to the caller - minimum amount, minimum notional - and the conversion between the three
sizing units, because a "contract" is 1 BTC on Bybit linear but 0.01 BTC on OKX's BTC-USDT-SWAP.

Order amounts in CCXT are in CONTRACTS for derivatives and in BASE units for spot. Everything the strategy and the
position dicts see is in BASE units; conversion happens only here.
"""
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from ccxt.base.errors import InvalidOrder

from .enums import MarketType, SizingUnit

logger = logging.getLogger(__name__)


class OrderSizeError(ValueError):
    """An order is below the venue's minimum amount or notional after rounding."""


@dataclass(frozen=True)
class MarketSpec:
    """The slice of a CCXT market dict that sizing and rounding need, plus the exchange for its rounding helpers."""
    symbol: str
    market_id: str
    market_type: MarketType
    base: str
    quote: str
    settle: Optional[str]
    contract_size: float
    min_amount: Optional[float]
    min_cost: Optional[float]
    tick_size: Optional[float]
    exchange: Any = None

    @classmethod
    def from_market(cls, exchange: Any, market: dict, market_type: MarketType) -> "MarketSpec":
        """Build from ``exchange.market(symbol)``, validating the market really is of ``market_type``."""
        validate_market_type(market, market_type)
        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        cost_limits = limits.get("cost") or {}
        precision = market.get("precision") or {}
        contract_size = market.get("contractSize") if market_type is MarketType.LINEAR_PERP else 1.0
        return cls(symbol=market["symbol"], market_id=str(market.get("id")), market_type=market_type,
                   base=market.get("base"), quote=market.get("quote"), settle=market.get("settle"),
                   contract_size=float(contract_size or 1.0), min_amount=_positive(amount_limits.get("min")),
                   min_cost=_positive(cost_limits.get("min")), tick_size=_positive(precision.get("price")),
                   exchange=exchange)

    # ------------------------------------------------------------------ conversion
    def base_to_amount(self, base_qty: float) -> float:
        """Base units -> the unit CCXT order amounts use (contracts for perps, base for spot)."""
        return base_qty / self.contract_size if self.market_type is MarketType.LINEAR_PERP else base_qty

    def amount_to_base(self, amount: float) -> float:
        return amount * self.contract_size if self.market_type is MarketType.LINEAR_PERP else amount

    def units_to_base(self, units: float, unit: SizingUnit, price: float) -> float:
        """A FIXED size in ``unit`` -> base quantity at ``price``."""
        if unit is SizingUnit.BASE_QTY:
            return units
        if unit is SizingUnit.QUOTE_NOTIONAL:
            if price <= 0:
                raise OrderSizeError(f"{self.symbol}: cannot size {units} {self.quote} notional without a "
                                     f"positive price")
            return units / price
        if unit is SizingUnit.CONTRACTS:
            if self.market_type is not MarketType.LINEAR_PERP:
                raise OrderSizeError("CONTRACTS sizing applies to perpetuals only")
            return units * self.contract_size
        raise ValueError(f"unknown sizing unit {unit!r}")

    # ------------------------------------------------------------------ rounding
    def round_amount(self, base_qty: float) -> float:
        """Base quantity -> venue order amount, truncated (never rounded UP: rounding up can exceed balance/risk)."""
        amount = self.base_to_amount(abs(base_qty))
        if self.exchange is None:
            return amount
        try:
            # CCXT truncates (TRUNCATE) and raises InvalidOrder when the result is zero.
            return float(self.exchange.amount_to_precision(self.symbol, amount))
        except InvalidOrder:
            return 0.0

    def round_price(self, price: float) -> float:
        if self.exchange is None:
            return price
        return float(self.exchange.price_to_precision(self.symbol, price))

    def checked_amount(self, base_qty: float, price: float) -> float:
        """Round ``base_qty`` to an order amount and enforce the venue minimums. Raises :class:`OrderSizeError`."""
        if not math.isfinite(base_qty) or base_qty <= 0:
            raise OrderSizeError(f"{self.symbol}: order quantity must be positive and finite (got {base_qty})")
        amount = self.round_amount(base_qty)
        if amount <= 0:
            raise OrderSizeError(f"{self.symbol}: quantity {base_qty} rounds to zero at the venue's amount precision")
        if self.min_amount is not None and amount < self.min_amount:
            raise OrderSizeError(f"{self.symbol}: amount {amount} is below the venue minimum {self.min_amount}")
        notional = self.amount_to_base(amount) * price if price and price > 0 else None
        if self.min_cost is not None and notional is not None and notional < self.min_cost:
            raise OrderSizeError(f"{self.symbol}: notional {notional:.8f} {self.quote} is below the venue minimum "
                                 f"{self.min_cost}")
        return amount

    def is_dust(self, base_qty: float) -> bool:
        """True when a residual quantity is too small to trade - treated as flat."""
        if base_qty <= 0:
            return True
        amount = self.base_to_amount(base_qty)
        if self.min_amount is not None:
            return amount < self.min_amount
        return amount <= 0


def validate_market_type(market: dict, market_type: MarketType) -> None:
    """Fail fast when a configured symbol is not the market type the strategy declares."""
    symbol = market.get("symbol")
    if market_type is MarketType.SPOT:
        if not market.get("spot"):
            raise ValueError(f"{symbol} is not a spot market (type={market.get('type')!r})")
        return
    if market_type is MarketType.LINEAR_PERP:
        if not (market.get("swap") and market.get("linear")):
            raise ValueError(f"{symbol} is not a linear perpetual swap (type={market.get('type')!r}, "
                             f"linear={market.get('linear')!r})")
        if not market.get("settle") or market.get("settle") != market.get("quote"):
            raise ValueError(f"{symbol}: linear perp must settle in its quote currency "
                             f"(settle={market.get('settle')!r}, quote={market.get('quote')!r})")
        return
    raise ValueError(f"unknown market type {market_type!r}")


def _positive(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None
