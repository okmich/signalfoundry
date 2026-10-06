"""Binance (spot + USDⓈ-M linear perpetuals) - supported venue.

Everything here was checked against CCXT 4.5.85's source (``binance.py`` / ``pro/binance.py``) and Binance's API docs;
the overrides exist because CCXT's generic path is wrong or incomplete for Binance:

* One ``ccxt.pro.binance`` instance routes by the symbol's market (``/api/v3`` for spot, ``/fapi`` for USDⓈ-M).
  ``defaultType`` is set to ``swap`` because CCXT renews the futures user-stream listenKey only when the DEFAULT type
  is futures (otherwise the key expires after 60 min and the futures stream goes silent). Spot user data uses the
  WS-API subscription, which needs no listenKey. Calls without a symbol therefore default to futures, which is why
  balances are read with explicit ``type`` params (:meth:`BinanceProfile.balance_params`).
* Spot and futures user data are SEPARATE streams: :meth:`account_stream_calls` watches each account that has
  strategies. ``watch_positions`` is given the symbols (CCXT sends ``subType=None`` otherwise) and its REST snapshot is
  disabled (it can hang forever if that snapshot fails).
* Paper trading is Binance DEMO trading (``enable_demo_trading``: spot + USDⓈ-M, live-like prices). CCXT no longer
  supports Binance's futures testnet, so ``testnet`` is refused. USDⓈ-M demo prices, funding and lot steps are NOT
  production's - do not measure fills or funding there.
* USDⓈ-M stop orders are "algo" orders since Binance's 2025-12 migration: placed through ``/fapi/v1/algoOrder`` with
  ``clientAlgoId`` (CCXT routes ``stopLossPrice`` / ``takeProfitPrice`` there) and listed / cancelled / looked up
  ONLY with ``{'trigger': True}``. The trigger price type is Binance's ``workingType`` (CONTRACT_PRICE = last, or
  MARK_PRICE); there is no index option. There are no attached or position-level stops.
* Spot stops are not used natively: Binance does not state that a spot stop leaves the base balance free, so an SL
  and a TP on the same quantity may not coexist. AUTO resolves spot stops to MANAGED.
* ``fetch_my_trades`` windows are bounded (7 days USDⓈ-M, 24 h spot) and its trades carry no client order id;
  fills are paged window by window. Liquidation / ADL fills are not identifiable there, so venue-side closes that are
  not our orders are reported UNKNOWN (``labels_stop_fills`` is off).
* Error codes CCXT maps too coarsely: ``-4059`` (position side unchanged) is ``OperationRejected``, a duplicate
  client id is a plain ``InvalidOrder`` (``-4116`` futures, ``-2010 "Duplicate order sent."`` spot), and ``-2015``
  after a successful call becomes ``DDoSProtection``.
"""
import logging
import re
from typing import Callable, Optional

from ccxt.base.errors import DDoSProtection

from ..client_order_id import ClientIdRule
from ..enums import MarketType, StopTrigger, VenueEnvironment
from ..markets import MarketSpec
from ..models import Fill, FundingPayment
from ..resilience import ErrorClass, VenueUnsupportedError, classify_ccxt_error
from .base import StopCapabilities, VenueProfile

logger = logging.getLogger(__name__)

_CODE_RE = re.compile(r'"code"\s*:\s*"?(-?\d+)')
_WORKING_TYPE = {StopTrigger.LAST: "CONTRACT_PRICE", StopTrigger.MARK: "MARK_PRICE"}
#: Longest windows Binance accepts between startTime and endTime on the trade-history endpoints.
_FUTURES_TRADES_WINDOW_MS = 7 * 24 * 3600 * 1000 - 1
_SPOT_TRADES_WINDOW_MS = 24 * 3600 * 1000 - 1
_PAGE = 1000

_VENUE_CODES: dict[str, ErrorClass] = {
    "-1021": ErrorClass.CLOCK_SKEW,      # timestamp outside recvWindow
    "-1015": ErrorClass.TRANSIENT,       # too many new orders
    "-2011": ErrorClass.NOT_FOUND,       # unknown order sent (cancel)
    "-2013": ErrorClass.NOT_FOUND,       # order does not exist
    "-2015": ErrorClass.BANNED,          # invalid API key, IP, or permissions: stop and alert
    "-4046": ErrorClass.NO_CHANGE,       # no need to change margin type
    "-4059": ErrorClass.NO_CHANGE,       # no need to change position side
    "-4116": ErrorClass.DUPLICATE,       # futures: clientOrderId is duplicated
}


class BinanceProfile(VenueProfile):
    client_id_rule = ClientIdRule(max_length=36, pattern=r"^[A-Za-z0-9_-]+$")
    labels_stop_fills = False
    default_ohlcv_limit = 1000

    def __init__(self, exchange_id: str = "binance"):
        super().__init__(exchange_id)

    # ------------------------------------------------------------------ construction
    def exchange_options(self, ccxt_options: dict | None = None) -> dict:
        options = super().exchange_options()
        options["defaultType"] = "swap"
        options["watchPositions"] = {"fetchPositionsSnapshot": False, "awaitPositionsSnapshot": False}
        options.update(ccxt_options or {})
        return options

    def apply_environment(self, exchange, environment: VenueEnvironment) -> None:
        if environment is VenueEnvironment.DEMO:
            exchange.enable_demo_trading(True)
            return
        if environment is VenueEnvironment.TESTNET:
            raise VenueUnsupportedError("binance: use environment 'demo' (Binance demo trading covers spot and USDⓈ-M; "
                                        "CCXT no longer supports Binance's futures testnet)")
        super().apply_environment(exchange, environment)

    def ohlcv_limit(self, exchange) -> int:
        # CCXT clamps both spot and USDⓈ-M to 1000; its USDⓈ-M features entry (500) understates it.
        return self.default_ohlcv_limit

    # ------------------------------------------------------------------ capabilities
    def stop_capabilities(self, exchange, market_type: MarketType) -> StopCapabilities:
        if market_type is MarketType.SPOT:
            return StopCapabilities()
        return StopCapabilities(standalone_conditional=True)

    def check_strategy(self, cfg) -> None:
        if cfg.market_type is MarketType.LINEAR_PERP and cfg.stop_trigger is StopTrigger.INDEX:
            raise VenueUnsupportedError(f"{cfg.name}: Binance USDⓈ-M stops trigger on the last (contract) or the mark "
                                        f"price only; stop_trigger 'index' is not available")

    def balance_params(self, market_type: MarketType) -> dict:
        return {"type": "spot"} if market_type is MarketType.SPOT else {"type": "swap"}

    def account_stream_calls(self, exchange, *, spot: bool, perp_symbols: list[str]) -> list[tuple[str, Callable]]:
        calls: list[tuple[str, Callable]] = []
        if perp_symbols:
            futures = {"type": "swap"}
            symbols = list(perp_symbols)
            calls += [("orders", lambda: exchange.watch_orders(None, None, None, dict(futures))),
                      ("fills", lambda: exchange.watch_my_trades(None, None, None, dict(futures))),
                      ("positions", lambda: exchange.watch_positions(symbols))]
        if spot:
            calls += [("orders", lambda: exchange.watch_orders(None, None, None, {"type": "spot"})),
                      ("fills", lambda: exchange.watch_my_trades(None, None, None, {"type": "spot"}))]
        return calls

    # ------------------------------------------------------------------ order parameters
    def standalone_stop_params(self, market_type: MarketType, level: float, is_stop_loss: bool,
                               trigger: StopTrigger) -> dict:
        params = super().standalone_stop_params(market_type, level, is_stop_loss, trigger)
        if market_type is MarketType.LINEAR_PERP:
            params["workingType"] = _WORKING_TYPE.get(trigger, "CONTRACT_PRICE")
        return params

    # ------------------------------------------------------------------ lookups
    async def find_order_by_client_id(self, exchange, spec: MarketSpec, client_order_id: str) -> Optional[dict]:
        attempts = [{"clientOrderId": client_order_id}]
        if spec.market_type is MarketType.LINEAR_PERP:
            attempts.append({"clientOrderId": client_order_id, "trigger": True})  # algo (stop) orders
        for params in attempts:
            try:
                return await exchange.fetch_order(None, spec.symbol, params)
            except Exception as exc:
                if classify_ccxt_error(exc, self) is not ErrorClass.NOT_FOUND:
                    raise
        return None

    async def fetch_fills(self, exchange, spec: MarketSpec, since_ms: int, until_ms: int | None = None) -> list[Fill]:
        until = until_ms if until_ms is not None else exchange.milliseconds()
        window = _SPOT_TRADES_WINDOW_MS if spec.market_type is MarketType.SPOT else _FUTURES_TRADES_WINDOW_MS
        trades: dict[str, dict] = {}
        start = since_ms
        while start <= until:
            end = min(start + window, until)
            cursor = start
            for _ in range(1000):
                page = await exchange.fetch_my_trades(spec.symbol, cursor, _PAGE, {"until": end}) or []
                for t in page:
                    trades[str(t.get("id"))] = t
                if len(page) < _PAGE:
                    break
                newest = max(int(t.get("timestamp") or 0) for t in page)
                if newest <= cursor:
                    break
                cursor = newest  # same-millisecond trades are re-read and de-duplicated by id
            start = end + 1
        fills = [self.normalize_fill(t, spec) for t in trades.values()]
        return sorted(fills, key=lambda f: (f.timestamp_ms, f.trade_id))

    async def fetch_funding(self, exchange, spec: MarketSpec, since_ms: int,
                            until_ms: int) -> Optional[list[FundingPayment]]:
        """USDⓈ-M funding from the income history (``incomeType=FUNDING_FEE``). Binance reports ``income`` from the
        account's side - positive is credited - which is already this package's convention (positive = received)."""
        if spec.market_type is not MarketType.LINEAR_PERP:
            return []
        payments: dict[str, FundingPayment] = {}
        cursor = since_ms
        for _ in range(1000):
            page = await exchange.fetch_funding_history(spec.symbol, cursor, _PAGE, {"until": until_ms}) or []
            for r in page:
                if r.get("amount") is None or r.get("timestamp") is None:
                    continue
                key = str(r.get("id") or (r["timestamp"], r["amount"]))
                payments[key] = FundingPayment(timestamp_ms=int(r["timestamp"]), amount=float(r["amount"]),
                                               currency=r.get("code") or spec.settle)
            if len(page) < _PAGE:
                break
            newest = max(int(r["timestamp"]) for r in page if r.get("timestamp") is not None)
            if newest < cursor:
                break
            cursor = newest + 1
        return sorted(payments.values(), key=lambda p: p.timestamp_ms)

    async def account_uid(self, exchange) -> Optional[str]:
        try:
            account = await exchange.private_get_account()
        except Exception as exc:
            logger.warning("binance: could not read the account UID (%s); falling back to a key fingerprint", exc)
            return None
        uid = (account or {}).get("uid")
        return str(uid) if uid else None

    # ------------------------------------------------------------------ interpretation
    def classify_venue_error(self, exc: BaseException) -> Optional[ErrorClass]:
        message = str(exc)
        if isinstance(exc, DDoSProtection) and ("418" in message or "banned" in message.lower()):
            return ErrorClass.BANNED                      # IP ban: never hammer it
        if "Duplicate order sent" in message:
            return ErrorClass.DUPLICATE                   # spot duplicate client order id
        match = _CODE_RE.search(message)
        if not match:
            return None
        code = match.group(1)
        if code == "-1003":
            return ErrorClass.BANNED if "banned" in message.lower() else ErrorClass.TRANSIENT
        return _VENUE_CODES.get(code)

    def client_order_id_of(self, record: dict) -> Optional[str]:
        value = record.get("clientOrderId")
        if value:
            return str(value)
        info = record.get("info") or {}
        # WebSocket user-data events carry the client order id as 'c' (both spot and USDⓈ-M); algo orders as
        # 'clientAlgoId' / 'caid'.
        for key in ("c", "clientAlgoId", "caid"):
            if info.get(key):
                return str(info[key])
        return super().client_order_id_of(record)
