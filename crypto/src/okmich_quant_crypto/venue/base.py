"""Venue profiles: every per-exchange difference lives behind this class.

Generic code in this package never branches on an exchange id. It asks the profile - for capabilities, for the
parameters an order needs, for how to look an order up by client id, for what a venue error code means. The base
class derives what it can from CCXT's ``has`` / ``features`` metadata and makes conservative choices elsewhere; it is
the STARTING POINT for a venue's profile. A supported venue subclasses it and overrides what CCXT gets wrong or
leaves out, and only venues in ``venue.registry`` can trade (the base class alone serves read-only public data).
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import ccxt.pro as ccxtpro

from ..client_order_id import ClientIdRule
from ..enums import (
    FillKind, MarginMode, MarginModeScope, MarketType, OrderSide, StopTrigger, VenueEnvironment,
)
from ..markets import MarketSpec
from ..models import Credentials, Fill, FundingPayment
from ..resilience import ErrorClass, VenueUnsupportedError, classify_ccxt_error

logger = logging.getLogger(__name__)

#: Raw-``info`` keys venues use for the client order id when CCXT does not lift it into the unified structure.
_CLIENT_ID_INFO_KEYS = ("clientOrderId", "orderLinkId", "clOrdId", "newClientOrderId", "client_oid", "clientOid",
                        "text")


@dataclass(frozen=True)
class StopCapabilities:
    """What native stop placement a venue supports for one market type."""
    #: SL/TP attached to a MARKET entry order (exchange creates them on fill).
    attached_on_market_entry: bool = False
    #: SL/TP attached to a LIMIT entry order.
    attached_on_limit_entry: bool = False
    #: Separate reduce-only (or spot TP/SL) conditional orders.
    standalone_conditional: bool = False
    #: Position-level stops that are REPLACED in place (no window where the position is unprotected).
    position_level: bool = False

    @property
    def any_native(self) -> bool:
        return self.attached_on_market_entry or self.attached_on_limit_entry or self.standalone_conditional \
            or self.position_level

    @property
    def can_protect_after_fill(self) -> bool:
        """Can a stop be placed on a position that already exists (e.g. after a restart, or after a market entry)."""
        return self.standalone_conditional or self.position_level


class VenueProfile:
    """Base profile derived from CCXT metadata. Subclass per supported venue; register it in ``venue.registry``."""

    #: Whether this profile recognises venue-triggered stop / take-profit fills (``fill_kind``). When it does, a close
    #: by an order that is neither ours nor a recognised venue stop is a human's (MANUAL); otherwise it is UNKNOWN.
    labels_stop_fills: bool = False
    #: Letters and digits, 32 chars: the strictest rule among the major venues (OKX).
    client_id_rule: ClientIdRule = ClientIdRule(max_length=32)
    margin_mode_scope: MarginModeScope = MarginModeScope.SYMBOL
    #: Whether the profile's exchange class carries a "candle is final" flag as a 7th OHLCV field.
    supports_bar_confirm: bool = False
    #: Fallback OHLCV page size when CCXT's features do not state one. Conservative on purpose.
    default_ohlcv_limit: int = 100

    def __init__(self, exchange_id: str):
        self.exchange_id = exchange_id

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.exchange_id!r})"

    # ------------------------------------------------------------------ construction
    def exchange_class(self) -> type:
        cls = getattr(ccxtpro, self.exchange_id, None)
        if cls is None:
            raise VenueUnsupportedError(f"ccxt.pro has no exchange class {self.exchange_id!r} (no WebSocket support)")
        return cls

    def exchange_options(self, ccxt_options: dict | None = None) -> dict:
        """CCXT ``options`` for a new exchange instance. The venue's own options are merged last."""
        options = {"adjustForTimeDifference": True}
        options.update(ccxt_options or {})
        return options

    def build_exchange(self, environment: VenueEnvironment, credentials: Credentials, *,
                       rate_limit_ms: int | None = None, ccxt_options: dict | None = None):
        config: dict[str, Any] = {"enableRateLimit": True, "options": self.exchange_options(ccxt_options)}
        if credentials.api_key:
            config["apiKey"] = credentials.api_key
        if credentials.secret:
            config["secret"] = credentials.secret
        if credentials.password:
            config["password"] = credentials.password
        if rate_limit_ms is not None:
            config["rateLimit"] = rate_limit_ms
        exchange = self.exchange_class()(config)
        self.apply_environment(exchange, environment)
        return exchange

    def apply_environment(self, exchange, environment: VenueEnvironment) -> None:
        """Point the instance at LIVE / TESTNET / DEMO. Exactly one switch, applied once, at construction."""
        if environment is VenueEnvironment.LIVE:
            return
        if environment is VenueEnvironment.TESTNET:
            if not (exchange.urls or {}).get("test"):
                raise VenueUnsupportedError(f"{self.exchange_id} has no testnet in CCXT")
            exchange.set_sandbox_mode(True)
            return
        raise VenueUnsupportedError(f"{self.exchange_id}: DEMO trading is not supported by the generic profile")

    # ------------------------------------------------------------------ capabilities
    def features_for(self, exchange, market_type: MarketType) -> dict:
        """CCXT's resolved ``features`` block for the market type (``{}`` when the venue publishes none)."""
        features = getattr(exchange, "features", None) or {}
        if market_type is MarketType.SPOT:
            block = features.get("spot")
        else:
            block = (features.get("swap") or {}).get("linear")
        return block or {}

    def stop_capabilities(self, exchange, market_type: MarketType) -> StopCapabilities:
        create = self.features_for(exchange, market_type).get("createOrder") or {}
        attached = bool(create.get("attachedStopLossTakeProfit"))
        standalone = bool(create.get("stopLossPrice")) and bool(create.get("takeProfitPrice"))
        return StopCapabilities(attached_on_market_entry=attached, attached_on_limit_entry=attached,
                                standalone_conditional=standalone, position_level=False)

    def ohlcv_limit(self, exchange) -> int:
        """Largest OHLCV page the venue returns. Always passed explicitly: CCXT defaults are often much smaller."""
        for market_type in (MarketType.LINEAR_PERP, MarketType.SPOT):
            limit = (self.features_for(exchange, market_type).get("fetchOHLCV") or {}).get("limit")
            if limit:
                return int(limit)
        return self.default_ohlcv_limit

    def check_strategy(self, cfg) -> None:
        """Reject a strategy configuration this venue cannot honour (called before anything is subscribed)."""
        return None

    def balance_params(self, market_type: MarketType) -> dict:
        """Params that make ``fetch_balance`` read the account that funds ``market_type``."""
        return {}

    def account_stream_calls(self, exchange, *, spot: bool, perp_symbols: list[str]) -> list[tuple[str, Callable]]:
        """The account-wide WebSocket watchers to run: ``(kind, zero-arg coroutine factory)`` with kind in
        ``orders`` / ``fills`` / ``positions``. Venues whose spot and futures accounts stream separately return one
        watcher per account."""
        calls: list[tuple[str, Callable]] = [("orders", exchange.watch_orders), ("fills", exchange.watch_my_trades)]
        if perp_symbols:
            calls.append(("positions", exchange.watch_positions))
        return calls

    # ------------------------------------------------------------------ order parameters
    def order_params(self, market_type: MarketType, *, reduce_only: bool = False) -> dict:
        """Base params for a plain order (the client order id is added at placement)."""
        params: dict[str, Any] = {}
        if reduce_only and market_type is MarketType.LINEAR_PERP:
            # Spot has no reduce-only concept; venues reject or ignore the flag there.
            params["reduceOnly"] = True
        return params

    def stop_order_params(self, market_type: MarketType) -> dict:
        """Params that address conditional (stop) orders when listing or cancelling them."""
        return {"trigger": True}

    def attached_stop_params(self, market_type: MarketType, stop_loss: Optional[float], take_profit: Optional[float],
                             trigger: StopTrigger) -> dict:
        """CCXT's unified attached SL/TP parameters for an entry order."""
        params: dict[str, Any] = {}
        if stop_loss is not None:
            params["stopLoss"] = {"triggerPrice": stop_loss}
        if take_profit is not None:
            params["takeProfit"] = {"triggerPrice": take_profit}
        if trigger is not StopTrigger.LAST and params:
            logger.warning("%s: generic profile cannot select a %s stop trigger; venue default applies",
                           self.exchange_id, trigger.value)
        return params

    def standalone_stop_params(self, market_type: MarketType, level: float, is_stop_loss: bool,
                               trigger: StopTrigger) -> dict:
        """Parameters for a separate conditional order that closes the position at ``level``."""
        params: dict[str, Any] = {"stopLossPrice" if is_stop_loss else "takeProfitPrice": level}
        if market_type is MarketType.LINEAR_PERP:
            params["reduceOnly"] = True
        return params

    async def set_position_stops(self, exchange, spec: MarketSpec, stop_loss: Optional[float],
                                 take_profit: Optional[float], trigger: StopTrigger) -> None:
        """Replace the position-level SL/TP in place. Only for profiles whose capabilities say ``position_level``."""
        raise VenueUnsupportedError(f"{self.exchange_id}: position-level stops are not supported by this profile")

    # ------------------------------------------------------------------ lookups
    async def find_order_by_client_id(self, exchange, spec: MarketSpec, client_order_id: str) -> Optional[dict]:
        """The order with ``client_order_id``, open or recently closed, or ``None`` if the venue has no such order."""
        fetchers = [exchange.fetch_open_orders]
        if exchange.has.get("fetchClosedOrders"):
            fetchers.append(exchange.fetch_closed_orders)
        if exchange.has.get("fetchCanceledOrders"):
            fetchers.append(exchange.fetch_canceled_orders)
        for fetch in fetchers:
            for order in await fetch(spec.symbol) or []:
                if self.client_order_id_of(order) == client_order_id:
                    return order
        return None

    async def fetch_fills(self, exchange, spec: MarketSpec, since_ms: int, until_ms: int | None = None) -> list[Fill]:
        """Our account's fills on ``spec`` since ``since_ms`` (all of them; attribution is the caller's job)."""
        fills: list[Fill] = []
        cursor = since_ms
        seen: set[str] = set()
        for _ in range(50):
            trades = await exchange.fetch_my_trades(spec.symbol, cursor, 100)
            new = [t for t in trades or [] if str(t.get("id")) not in seen]
            if not new:
                break
            for t in new:
                seen.add(str(t.get("id")))
                fill = self.normalize_fill(t, spec)
                if until_ms is None or fill.timestamp_ms <= until_ms:
                    fills.append(fill)
            last_ts = max(int(t.get("timestamp") or 0) for t in new)
            if last_ts <= cursor or len(trades) < 100 or (until_ms is not None and last_ts > until_ms):
                break
            cursor = last_ts
        return sorted(fills, key=lambda f: (f.timestamp_ms, f.trade_id))

    async def fetch_funding(self, exchange, spec: MarketSpec, since_ms: int,
                            until_ms: int) -> Optional[list[FundingPayment]]:
        """Funding payments on ``spec`` in ``[since_ms, until_ms]``, sign normalised (positive = received).

        ``None`` means "this profile cannot tell": the base profile does not trust CCXT's funding sign convention,
        which differs per venue, so a venue profile must implement this (else funding is left out of P&L, warned).
        """
        return None

    async def account_uid(self, exchange) -> Optional[str]:
        """The venue's account id for the API key, where an endpoint exists for it."""
        return None

    # ------------------------------------------------------------------ account setup
    async def ensure_one_way(self, exchange, spec: MarketSpec) -> None:
        """Make sure ``spec``'s positions are in one-way mode, or fail fast."""
        if exchange.has.get("setPositionMode"):
            try:
                await exchange.set_position_mode(False, spec.symbol)
                return
            except Exception as exc:
                if classify_ccxt_error(exc, self) is ErrorClass.NO_CHANGE:
                    return
                raise VenueUnsupportedError(f"{spec.symbol}: could not ensure one-way position mode ({exc}). Hedge "
                                            f"mode is unsupported; switch the account to one-way with no open "
                                            f"positions.") from exc
        positions = await exchange.fetch_positions([spec.symbol]) if exchange.has.get("fetchPositions") else []
        if any(p.get("hedged") for p in positions or []):
            raise VenueUnsupportedError(f"{spec.symbol}: account is in hedge mode; switch it to one-way")

    async def apply_margin_mode(self, exchange, mode: MarginMode, spec: Optional[MarketSpec]) -> None:
        if not exchange.has.get("setMarginMode"):
            raise VenueUnsupportedError(f"{self.exchange_id} cannot set the margin mode through CCXT")
        try:
            await exchange.set_margin_mode(mode.value, spec.symbol if spec else None)
        except Exception as exc:
            if classify_ccxt_error(exc, self) is not ErrorClass.NO_CHANGE:
                raise

    async def apply_leverage(self, exchange, leverage: float, spec: MarketSpec) -> None:
        if not exchange.has.get("setLeverage"):
            raise VenueUnsupportedError(f"{self.exchange_id} cannot set leverage through CCXT")
        try:
            await exchange.set_leverage(leverage, spec.symbol)
        except Exception as exc:
            if classify_ccxt_error(exc, self) is not ErrorClass.NO_CHANGE:
                raise

    # ------------------------------------------------------------------ interpretation
    def classify_venue_error(self, exc: BaseException) -> Optional[ErrorClass]:
        """Venue-specific error codes, checked before CCXT's generic exception tree. ``None`` = no opinion."""
        return None

    def client_order_id_of(self, record: dict) -> Optional[str]:
        """Client order id of a unified order or trade, falling back to the raw ``info`` keys venues use."""
        value = record.get("clientOrderId")
        if value:
            return str(value)
        info = record.get("info") or {}
        for key in _CLIENT_ID_INFO_KEYS:
            if info.get(key):
                return str(info[key])
        return None

    def fill_kind(self, trade: dict) -> FillKind:
        return FillKind.TRADE

    def normalize_fill(self, trade: dict, spec: MarketSpec) -> Fill:
        """Unified CCXT trade -> :class:`Fill` in base units with fees split by currency."""
        price = float(trade.get("price") or 0.0)
        amount = float(trade.get("amount") or 0.0)
        base_qty = spec.amount_to_base(amount)
        fee_quote, fee_base, unresolved = 0.0, 0.0, False
        fees = trade.get("fees") or ([trade["fee"]] if trade.get("fee") else [])
        for fee in fees:
            cost = fee.get("cost")
            if cost is None:
                continue
            cost = float(cost)
            currency = self._fee_currency(trade, fee)
            if currency == spec.quote or (currency is not None and currency == spec.settle):
                fee_quote += cost
            elif currency == spec.base:
                fee_base += cost
                fee_quote += cost * price
            elif cost != 0.0:
                unresolved = True
        side = OrderSide(str(trade.get("side")).lower())
        return Fill(trade_id=str(trade.get("id")), order_id=_str_or_none(trade.get("order")),
                    client_order_id=self.client_order_id_of(trade), timestamp_ms=int(trade.get("timestamp") or 0),
                    side=side, price=price, base_qty=base_qty, fee_quote=fee_quote, fee_base=fee_base,
                    fee_unresolved=unresolved, kind=self.fill_kind(trade), info=trade.get("info") or {})

    def _fee_currency(self, trade: dict, fee: dict) -> Optional[str]:
        return fee.get("currency")

    def position_stop_levels(self, position: dict) -> tuple[Optional[float], Optional[float]]:
        """(stop_loss, take_profit) the venue reports on a unified position; unset levels (0 / empty) -> ``None``."""
        return _level(position.get("stopLossPrice")), _level(position.get("takeProfitPrice"))

    def bar_confirmed(self, row: list) -> Optional[bool]:
        """The "candle is final" flag when the profile's exchange class carries one (7th field), else ``None``."""
        return bool(row[6]) if len(row) > 6 and row[6] is not None else None

    @staticmethod
    def trigger_price(ticker: dict, trigger: StopTrigger) -> Optional[float]:
        """The price a managed stop compares against, from a unified ticker."""
        key = {StopTrigger.LAST: "last", StopTrigger.MARK: "markPrice", StopTrigger.INDEX: "indexPrice"}[trigger]
        value = ticker.get(key)
        if value is None and trigger is not StopTrigger.LAST:
            value = ticker.get("last")
        return float(value) if value is not None else None


def _level(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _str_or_none(value) -> Optional[str]:
    return str(value) if value is not None else None
