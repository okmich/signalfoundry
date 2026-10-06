"""Bybit (v5, unified trading account) - the first supported venue.

Every override here exists because CCXT 4.5.85's generic path is wrong or incomplete for Bybit (verified against the
source and live public calls):

* Demo trading must be switched on with ``enable_demo_trading(True)``; the constructor option does nothing, and
  enabling sandbox after demo silently lands on testnet. Demo has no WS order entry and no UID endpoint.
* ``watch_ohlcv`` drops Bybit's kline ``confirm`` flag. The exchange subclass below keeps it as a 7th field.
* ``watch_positions`` takes a REST snapshot on first use that can hang if it fails; it is switched off here and the
  package takes its own snapshot.
* The unified ``triggerPriceType`` is ignored by ``create_order``; Bybit's own ``slTriggerBy`` / ``tpTriggerBy`` are
  sent instead.
* Position stops are replaced through ``/v5/position/trading-stop`` in Full mode with ``positionIdx=0`` (CCXT's
  route omits ``positionIdx`` and, in Partial mode, ADDS a stop on every call instead of replacing it).
* ``fetch_funding_history`` sends the wrong page-size parameter and hard-codes USDT; funding is read from the
  execution list directly, paged in the 7-day windows Bybit allows. Bybit's sign is positive = PAID.
* ``fetch_my_trades`` drops liquidation (BustTrade) and ADL fills; fills are read from the execution list unfiltered.
* "Not modified" / "already filled" / "duplicate orderLinkId" arrive as generic InvalidOrder / BadRequest; the codes
  are mapped in :meth:`BybitProfile.classify_venue_error`.
* ``set_margin_mode`` on a unified account changes the WHOLE account and ignores the symbol.
"""
import logging
import re
from typing import Any, Optional

import ccxt.pro as ccxtpro

from ..client_order_id import ClientIdRule
from ..enums import FillKind, MarginModeScope, MarketType, StopTrigger, VenueEnvironment
from ..markets import MarketSpec
from ..models import Fill, FundingPayment
from ..resilience import ErrorClass, VenueUnsupportedError
from .base import StopCapabilities, VenueProfile

logger = logging.getLogger(__name__)

_RET_CODE_RE = re.compile(r'"ret_?[cC]ode"\s*:\s*"?(\d+)')
#: Bybit allows at most 7 days between startTime and endTime on the execution list.
_EXECUTION_WINDOW_MS = 7 * 24 * 3600 * 1000 - 1
_EXECUTION_PAGE_LIMIT = 100

_TRIGGER_BY = {StopTrigger.LAST: "LastPrice", StopTrigger.MARK: "MarkPrice", StopTrigger.INDEX: "IndexPrice"}

#: Bybit retCode -> reaction. Only codes whose meaning is documented and stable are listed.
_VENUE_CODES: dict[str, ErrorClass] = {
    "10002": ErrorClass.CLOCK_SKEW,      # request timestamp outside recv_window
    "10006": ErrorClass.TRANSIENT,       # too many visits (rate limit)
    "10018": ErrorClass.TRANSIENT,       # IP rate limit
    "10016": ErrorClass.TRANSIENT,       # internal system error
    "10003": ErrorClass.PERMANENT,       # invalid API key
    "10004": ErrorClass.PERMANENT,       # signature error
    "10005": ErrorClass.PERMANENT,       # permission denied for the key
    "10010": ErrorClass.BANNED,          # unmatched IP (key bound to other IPs)
    "33004": ErrorClass.BANNED,          # API key expired
    "110001": ErrorClass.NOT_FOUND,      # order does not exist
    "110008": ErrorClass.ALREADY_DONE,   # order already filled or cancelled
    "110010": ErrorClass.ALREADY_DONE,   # order already cancelled
    "110025": ErrorClass.NO_CHANGE,      # position mode not modified
    "110026": ErrorClass.NO_CHANGE,      # margin mode not modified
    "110043": ErrorClass.NO_CHANGE,      # leverage not modified
    "34040": ErrorClass.NO_CHANGE,       # trading-stop: not modified
    "110072": ErrorClass.DUPLICATE,      # orderLinkId is duplicate
    # spot
    "170139": ErrorClass.ALREADY_DONE,   # order has been filled
    "170141": ErrorClass.DUPLICATE,      # duplicate clientOrderId
    "170142": ErrorClass.ALREADY_DONE,   # order has been cancelled
    "170143": ErrorClass.NOT_FOUND,      # cannot be found on the order book
    "170146": ErrorClass.UNKNOWN_STATE,  # order creation timeout: it may exist
    "170147": ErrorClass.TRANSIENT,      # order cancellation timeout
}

_STOP_LOSS_TYPES = {"StopLoss", "PartialStopLoss", "TrailingStop"}
_TAKE_PROFIT_TYPES = {"TakeProfit", "PartialTakeProfit"}
_EXEC_KINDS = {"BustTrade": FillKind.LIQUIDATION, "AdlTrade": FillKind.ADL, "Funding": FillKind.FUNDING,
               "Settle": FillKind.SETTLEMENT}


class BybitPro(ccxtpro.bybit):
    """CCXT's Bybit with the kline ``confirm`` flag kept as a 7th OHLCV field (it survives CCXT's OHLCV cache)."""

    def parse_ws_ohlcv(self, ohlcv, market=None) -> list:
        row = super().parse_ws_ohlcv(ohlcv, market)
        row.append(bool(ohlcv.get("confirm")) if isinstance(ohlcv, dict) else None)
        return row


class BybitProfile(VenueProfile):
    labels_stop_fills = True
    client_id_rule = ClientIdRule(max_length=36, pattern=r"^[A-Za-z0-9_-]+$")
    margin_mode_scope = MarginModeScope.ACCOUNT
    supports_bar_confirm = True
    default_ohlcv_limit = 1000

    def __init__(self, exchange_id: str = "bybit"):
        super().__init__(exchange_id)

    # ------------------------------------------------------------------ construction
    def exchange_class(self) -> type:
        return BybitPro

    def exchange_options(self, ccxt_options: dict | None = None) -> dict:
        options = super().exchange_options()
        options["watchPositions"] = {"fetchPositionsSnapshot": False, "awaitPositionsSnapshot": False}
        options.update(ccxt_options or {})
        return options

    def apply_environment(self, exchange, environment: VenueEnvironment) -> None:
        if environment is VenueEnvironment.DEMO:
            # Never combine with sandbox: sandbox-then-demo raises, demo-then-sandbox silently lands on testnet.
            exchange.enable_demo_trading(True)
            return
        super().apply_environment(exchange, environment)

    # ------------------------------------------------------------------ capabilities
    def stop_capabilities(self, exchange, market_type: MarketType) -> StopCapabilities:
        if market_type is MarketType.SPOT:
            # Bybit attaches TP/SL to spot LIMIT orders only; CCXT raises InvalidOrder for spot market + TP/SL.
            return StopCapabilities(attached_on_market_entry=False, attached_on_limit_entry=True,
                                    standalone_conditional=True, position_level=False)
        return StopCapabilities(attached_on_market_entry=True, attached_on_limit_entry=True,
                                standalone_conditional=True, position_level=True)

    # ------------------------------------------------------------------ order parameters
    def attached_stop_params(self, market_type: MarketType, stop_loss: Optional[float], take_profit: Optional[float],
                             trigger: StopTrigger) -> dict:
        params: dict[str, Any] = {}
        # Scalars, not dicts: a scalar maps to Bybit's position-level (Full) TP/SL; a dict with a limit price would
        # switch the order into Partial mode.
        if stop_loss is not None:
            params["stopLoss"] = stop_loss
            if market_type is MarketType.LINEAR_PERP:
                params["slTriggerBy"] = _TRIGGER_BY[trigger]
        if take_profit is not None:
            params["takeProfit"] = take_profit
            if market_type is MarketType.LINEAR_PERP:
                params["tpTriggerBy"] = _TRIGGER_BY[trigger]
        if market_type is MarketType.SPOT and trigger is not StopTrigger.LAST and params:
            logger.warning("bybit spot TP/SL trigger on last price only; %s ignored", trigger.value)
        return params

    def stop_order_params(self, market_type: MarketType) -> dict:
        # Spot TP/SL orders are their own order class on Bybit (orderFilter=tpslOrder), not "StopOrder".
        return {"orderFilter": "tpslOrder"} if market_type is MarketType.SPOT else {"trigger": True}

    def standalone_stop_params(self, market_type: MarketType, level: float, is_stop_loss: bool,
                               trigger: StopTrigger) -> dict:
        params = super().standalone_stop_params(market_type, level, is_stop_loss, trigger)
        if market_type is MarketType.LINEAR_PERP:
            params["triggerBy"] = _TRIGGER_BY[trigger]
        return params

    async def set_position_stops(self, exchange, spec: MarketSpec, stop_loss: Optional[float],
                                 take_profit: Optional[float], trigger: StopTrigger) -> None:
        """Replace BOTH position levels atomically (Full mode). ``None`` cancels a level ("0" to Bybit)."""
        if spec.market_type is not MarketType.LINEAR_PERP:
            raise VenueUnsupportedError("bybit position-level stops exist for derivatives only")
        request = {
            "category": "linear", "symbol": spec.market_id, "tpslMode": "Full", "positionIdx": 0,
            "stopLoss": str(spec.round_price(stop_loss)) if stop_loss else "0",
            "takeProfit": str(spec.round_price(take_profit)) if take_profit else "0",
            "slTriggerBy": _TRIGGER_BY[trigger], "tpTriggerBy": _TRIGGER_BY[trigger],
        }
        try:
            await exchange.private_post_v5_position_trading_stop(request)
        except Exception as exc:
            if self.classify_venue_error(exc) is ErrorClass.NO_CHANGE:
                return
            raise

    # ------------------------------------------------------------------ lookups
    async def find_order_by_client_id(self, exchange, spec: MarketSpec, client_order_id: str) -> Optional[dict]:
        category = self._category(spec)
        market = exchange.market(spec.symbol)
        for endpoint in (exchange.private_get_v5_order_realtime, exchange.private_get_v5_order_history):
            response = await endpoint({"category": category, "symbol": spec.market_id, "orderLinkId": client_order_id})
            rows = ((response or {}).get("result") or {}).get("list") or []
            for row in rows:
                if row.get("orderLinkId") == client_order_id:
                    return exchange.parse_order(row, market)
        return None

    async def fetch_fills(self, exchange, spec: MarketSpec, since_ms: int, until_ms: int | None = None) -> list[Fill]:
        rows = await self._execution_rows(exchange, spec, since_ms, until_ms, exec_type=None)
        market = exchange.market(spec.symbol)
        fills = []
        for row in rows:
            if row.get("execType") == "Funding":
                continue
            fills.append(self.normalize_fill(exchange.parse_trade(row, market), spec))
        return sorted(fills, key=lambda f: (f.timestamp_ms, f.trade_id))

    async def fetch_funding(self, exchange, spec: MarketSpec, since_ms: int,
                            until_ms: int) -> Optional[list[FundingPayment]]:
        if spec.market_type is not MarketType.LINEAR_PERP:
            return []
        rows = await self._execution_rows(exchange, spec, since_ms, until_ms, exec_type="Funding")
        payments = []
        for row in rows:
            fee = row.get("execFee")
            if fee in (None, ""):
                continue
            # Bybit: execFee > 0 means the position PAID funding. Normalise to positive = received.
            payments.append(FundingPayment(timestamp_ms=int(row.get("execTime") or 0), amount=-float(fee),
                                           currency=row.get("feeCurrency") or spec.settle))
        return payments

    async def _execution_rows(self, exchange, spec: MarketSpec, since_ms: int, until_ms: int | None,
                              exec_type: Optional[str]) -> list[dict]:
        until = until_ms if until_ms is not None else exchange.milliseconds()
        rows: list[dict] = []
        start = since_ms
        while start <= until:
            end = min(start + _EXECUTION_WINDOW_MS, until)
            cursor = None
            for _ in range(100):
                request = {"category": self._category(spec), "symbol": spec.market_id, "startTime": start,
                           "endTime": end, "limit": _EXECUTION_PAGE_LIMIT}
                if exec_type:
                    request["execType"] = exec_type
                if cursor:
                    request["cursor"] = cursor
                response = await exchange.private_get_v5_execution_list(request)
                result = (response or {}).get("result") or {}
                rows.extend(result.get("list") or [])
                cursor = result.get("nextPageCursor")
                if not cursor:
                    break
            start = end + 1
        return rows

    async def account_uid(self, exchange) -> Optional[str]:
        if exchange.options.get("enableDemoTrading"):
            return None  # /v5/user/query-api is not available on demo trading
        try:
            response = await exchange.private_get_v5_user_query_api()
        except Exception as exc:
            logger.warning("bybit: could not read the account UID (%s); falling back to a key fingerprint", exc)
            return None
        uid = ((response or {}).get("result") or {}).get("userID")
        return str(uid) if uid else None

    # ------------------------------------------------------------------ interpretation
    def classify_venue_error(self, exc: BaseException) -> Optional[ErrorClass]:
        match = _RET_CODE_RE.search(str(exc))
        return _VENUE_CODES.get(match.group(1)) if match else None

    def client_order_id_of(self, record: dict) -> Optional[str]:
        info = record.get("info") or {}
        return info.get("orderLinkId") or super().client_order_id_of(record)

    def fill_kind(self, trade: dict) -> FillKind:
        info = trade.get("info") or {}
        exec_type = info.get("execType")
        if exec_type in _EXEC_KINDS:
            return _EXEC_KINDS[exec_type]
        stop_type = info.get("stopOrderType") or ""
        if stop_type in _STOP_LOSS_TYPES:
            return FillKind.STOP_LOSS
        if stop_type in _TAKE_PROFIT_TYPES:
            return FillKind.TAKE_PROFIT
        return FillKind.TRADE

    def _fee_currency(self, trade: dict, fee: dict) -> Optional[str]:
        # CCXT infers the fee currency from side and sign; Bybit states it (spot) in feeCurrency. Prefer the venue.
        info = trade.get("info") or {}
        return info.get("feeCurrency") or fee.get("currency")

    @staticmethod
    def _category(spec: MarketSpec) -> str:
        return "spot" if spec.market_type is MarketType.SPOT else "linear"
