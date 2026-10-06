"""Venue profiles: generic capability derivation, stop-mode resolution, the certification gate, and Bybit specifics."""
import pytest

from okmich_quant_crypto import MarketType, StopMode, StopTrigger, VenueEnvironment
from okmich_quant_crypto.capabilities import resolve_capabilities
from okmich_quant_crypto.enums import FillKind
from okmich_quant_crypto.markets import MarketSpec
from okmich_quant_crypto.resilience import VenueUnsupportedError
from okmich_quant_crypto.venue import (
    BybitPro, BybitProfile, VenueProfile, get_profile, is_supported, resolve_profile, supported_venues,
)

from .conftest import make_cfg
from .fakes import FakeExchange, perp_market, spot_market


# ---------------------------------------------------------------------- registry + gate

def test_supported_list():
    assert [v for v in supported_venues() if v != "fakex"] == ["binance", "bybit"]   # fakex: test-only venue
    assert is_supported(" ByBit ") and is_supported("binance") and not is_supported("okx")
    assert isinstance(resolve_profile("bybit"), BybitProfile)


def test_unsupported_exchanges_cannot_trade():
    for exchange_id in ("okx", "kraken", "gate"):
        with pytest.raises(VenueUnsupportedError, match="not a supported exchange"):
            resolve_profile(exchange_id)


def test_read_only_profile_is_available_for_any_ccxt_exchange():
    assert isinstance(get_profile("bybit"), BybitProfile)
    generic = get_profile("okx")
    assert type(generic) is VenueProfile and generic.exchange_id == "okx"


# ---------------------------------------------------------------------- capability gate + stop mode

def test_missing_required_capabilities_fail_fast_listing_all(clock):
    ex = FakeExchange(clock)
    ex.has["fetchPositions"] = False
    ex.has["fetchMyTrades"] = False
    with pytest.raises(VenueUnsupportedError, match=r"\['fetchMyTrades', 'fetchPositions'\]"):
        resolve_capabilities(ex, VenueProfile("okx"), make_cfg())


def test_auto_resolves_native_when_a_live_position_can_be_protected(exchange):
    caps = resolve_capabilities(exchange, VenueProfile("okx"), make_cfg())
    assert caps.stop_mode is StopMode.NATIVE and caps.stream_bars and caps.stream_account


def test_auto_falls_back_to_managed(exchange):
    exchange.features["swap"]["linear"]["createOrder"] = {"attachedStopLossTakeProfit": {"price": True}}
    caps = resolve_capabilities(exchange, VenueProfile("okx"), make_cfg())
    # Attach-on-entry alone cannot protect a position after a restart, so AUTO does not call it native.
    assert caps.stop_mode is StopMode.MANAGED


def test_explicit_native_on_a_venue_without_it_fails(exchange):
    exchange.features["swap"]["linear"]["createOrder"] = {}
    with pytest.raises(VenueUnsupportedError, match="NATIVE stops requested"):
        resolve_capabilities(exchange, VenueProfile("okx"), make_cfg(stop_mode="native"))


def test_managed_with_slow_poll_rejected_after_auto_resolution(exchange):
    exchange.features["swap"]["linear"]["createOrder"] = {}
    cfg = make_cfg(feed_mode="poll", managed_stop_poll_seconds=9)
    with pytest.raises(ValueError, match="managed_stop_poll_seconds"):
        resolve_capabilities(exchange, VenueProfile("okx"), cfg)


def test_missing_streams_degrade_to_polling(exchange):
    exchange.has["watchPositions"] = False
    exchange.has["watchOHLCV"] = False
    caps = resolve_capabilities(exchange, VenueProfile("okx"), make_cfg())
    assert not caps.stream_bars and not caps.stream_account
    assert resolve_capabilities(exchange, VenueProfile("okx"), make_cfg(feed_mode="poll")).stream_ticker is False


def test_ohlcv_limit_comes_from_features_or_a_conservative_default(exchange):
    assert VenueProfile("okx").ohlcv_limit(exchange) == 3
    exchange.features = {}
    assert VenueProfile("okx").ohlcv_limit(exchange) == 100


# ---------------------------------------------------------------------- generic fill normalisation

def test_fee_split_by_currency(exchange):
    spot = MarketSpec.from_market(exchange, exchange.market("BTC/USDT"), MarketType.SPOT)
    prof = VenueProfile("okx")
    base_fee = prof.normalize_fill({"id": "1", "side": "buy", "price": 100.0, "amount": 1.0, "timestamp": 5,
                                    "fee": {"cost": 0.001, "currency": "BTC"}}, spot)
    assert base_fee.fee_base == 0.001 and base_fee.fee_quote == pytest.approx(0.1) and not base_fee.fee_unresolved
    third = prof.normalize_fill({"id": "2", "side": "sell", "price": 100.0, "amount": 1.0, "timestamp": 5,
                                 "fee": {"cost": 0.01, "currency": "BNB"}}, spot)
    assert third.fee_unresolved
    perp = MarketSpec.from_market(exchange, exchange.market("BTC/USDT:USDT"), MarketType.LINEAR_PERP)
    f = prof.normalize_fill({"id": "3", "side": "sell", "price": 100.0, "amount": 2.0, "timestamp": 5,
                             "fees": [{"cost": 0.05, "currency": "USDT"}], "info": {"clOrdId": "sf1xab"}}, perp)
    assert f.fee_quote == 0.05 and f.client_order_id == "sf1xab" and f.base_qty == 2.0


# ---------------------------------------------------------------------- Bybit

class _BybitStub:
    """Just the implicit-API surface BybitProfile calls."""

    def __init__(self, clock):
        self.clock = clock
        self.options = {}
        self.requests = []
        self.execution_pages = {}
        self.markets = {m["symbol"]: m for m in (perp_market(), spot_market())}

    def market(self, symbol):
        return self.markets[symbol]

    def milliseconds(self):
        return self.clock()

    def price_to_precision(self, symbol, price):
        return repr(round(price, 1))

    async def private_post_v5_position_trading_stop(self, request):
        self.requests.append(("trading-stop", request))
        return {"retCode": 0, "result": {}}

    async def private_get_v5_execution_list(self, request):
        self.requests.append(("execution", dict(request)))
        key = (request.get("execType"), request["startTime"], request.get("cursor"))
        return {"result": self.execution_pages.get(key, {"list": []})}

    async def private_get_v5_order_realtime(self, request):
        self.requests.append(("realtime", request))
        return {"result": {"list": []}}

    async def private_get_v5_order_history(self, request):
        self.requests.append(("history", request))
        return {"result": {"list": [{"orderLinkId": request["orderLinkId"], "orderId": "9"}]}}

    def parse_order(self, row, market):
        return {"id": row["orderId"], "clientOrderId": row["orderLinkId"], "info": row}

    def parse_trade(self, row, market):
        return {"id": row["execId"], "order": row.get("orderId"), "side": row["side"].lower(),
                "price": float(row["execPrice"]), "amount": float(row["execQty"]), "timestamp": int(row["execTime"]),
                "fee": {"cost": float(row["execFee"]), "currency": "USDT"}, "info": row}


def _bspec(stub, symbol="BTC/USDT:USDT", mt=MarketType.LINEAR_PERP):
    return MarketSpec.from_market(stub, stub.market(symbol), mt)


def test_bybit_stop_capabilities():
    caps = BybitProfile().stop_capabilities(None, MarketType.LINEAR_PERP)
    assert caps.position_level and caps.attached_on_market_entry
    spot = BybitProfile().stop_capabilities(None, MarketType.SPOT)
    assert not spot.attached_on_market_entry and spot.attached_on_limit_entry and spot.standalone_conditional
    assert not spot.position_level


def test_bybit_attached_params_are_scalars_with_native_trigger():
    p = BybitProfile().attached_stop_params(MarketType.LINEAR_PERP, 95.0, 110.0, StopTrigger.MARK)
    assert p == {"stopLoss": 95.0, "slTriggerBy": "MarkPrice", "takeProfit": 110.0, "tpTriggerBy": "MarkPrice"}
    spot = BybitProfile().attached_stop_params(MarketType.SPOT, 95.0, None, StopTrigger.LAST)
    assert spot == {"stopLoss": 95.0}
    assert BybitProfile().stop_order_params(MarketType.SPOT) == {"orderFilter": "tpslOrder"}


async def test_bybit_trading_stop_replaces_both_levels_in_full_mode(clock):
    stub = _BybitStub(clock)
    await BybitProfile().set_position_stops(stub, _bspec(stub), 95.04, None, StopTrigger.LAST)
    kind, req = stub.requests[-1]
    assert kind == "trading-stop"
    assert req["tpslMode"] == "Full" and req["positionIdx"] == 0 and req["category"] == "linear"
    assert req["stopLoss"] == "95.0" and req["takeProfit"] == "0"   # None cancels the level
    assert req["symbol"] == "BTCUSDT"


async def test_bybit_funding_is_paged_in_7_day_windows_and_sign_normalised(clock):
    stub = _BybitStub(clock)
    day = 24 * 3600 * 1000
    since = 0
    stub.execution_pages[("Funding", 0, None)] = {"list": [{"execFee": "0.5", "execTime": "100"}],
                                                  "nextPageCursor": "c2"}
    stub.execution_pages[("Funding", 0, "c2")] = {"list": [{"execFee": "-0.2", "execTime": "200"}]}
    stub.execution_pages[("Funding", 7 * day, None)] = {"list": [{"execFee": "0.1", "execTime": str(8 * day)}]}
    payments = await BybitProfile().fetch_funding(stub, _bspec(stub), since, 10 * day)
    assert [p.amount for p in payments] == [-0.5, 0.2, -0.1]          # positive execFee = PAID
    windows = [r for k, r in stub.requests if k == "execution"]
    assert all(r["endTime"] - r["startTime"] < 7 * day for r in windows)
    assert all(r["limit"] == 100 for r in windows)


async def test_bybit_fills_include_liquidations_but_not_funding(clock):
    stub = _BybitStub(clock)
    rows = [{"execId": "a", "side": "Buy", "execPrice": "100", "execQty": "1", "execTime": "1", "execFee": "0.1",
             "execType": "Trade"},
            {"execId": "b", "side": "Sell", "execPrice": "80", "execQty": "1", "execTime": "2", "execFee": "0.1",
             "execType": "BustTrade"},
            {"execId": "c", "side": "Sell", "execPrice": "0", "execQty": "1", "execTime": "3", "execFee": "0.3",
             "execType": "Funding"}]
    stub.execution_pages[(None, 0, None)] = {"list": rows}
    fills = await BybitProfile().fetch_fills(stub, _bspec(stub), 0, 10)
    assert [(f.trade_id, f.kind) for f in fills] == [("a", FillKind.TRADE), ("b", FillKind.LIQUIDATION)]


def test_bybit_fill_kinds():
    prof = BybitProfile()
    assert prof.fill_kind({"info": {"execType": "Trade", "stopOrderType": "StopLoss"}}) is FillKind.STOP_LOSS
    assert prof.fill_kind({"info": {"execType": "Trade", "stopOrderType": "TakeProfit"}}) is FillKind.TAKE_PROFIT
    assert prof.fill_kind({"info": {"execType": "AdlTrade"}}) is FillKind.ADL
    assert prof.fill_kind({"info": {"execType": "Trade", "stopOrderType": ""}}) is FillKind.TRADE
    assert prof.client_order_id_of({"clientOrderId": None, "info": {"orderLinkId": "sf1xa"}}) == "sf1xa"


async def test_bybit_finds_orders_by_link_id_in_realtime_then_history(clock):
    stub = _BybitStub(clock)
    order = await BybitProfile().find_order_by_client_id(stub, _bspec(stub), "sf42xabc")
    assert order["id"] == "9"
    assert [k for k, _ in stub.requests] == ["realtime", "history"]


async def test_bybit_demo_has_no_uid_endpoint(clock):
    stub = _BybitStub(clock)
    stub.options["enableDemoTrading"] = True
    assert await BybitProfile().account_uid(stub) is None


def test_bybit_demo_uses_demo_switch_not_sandbox():
    ex = BybitPro()
    BybitProfile().apply_environment(ex, VenueEnvironment.DEMO)
    assert ex.options.get("enableDemoTrading") is True
    assert "demo" in str(ex.urls["api"]) and not ex.isSandboxModeEnabled
    assert ex.options["watchPositions"]["fetchPositionsSnapshot"] is True  # default instance; profile sets False


def test_bybit_profile_builds_its_exchange_with_safe_options():
    from okmich_quant_crypto.models import Credentials
    ex = BybitProfile().build_exchange(VenueEnvironment.DEMO, Credentials(api_key="k", secret="s"))
    assert isinstance(ex, BybitPro)
    assert ex.options["watchPositions"] == {"fetchPositionsSnapshot": False, "awaitPositionsSnapshot": False}
    assert ex.options["enableDemoTrading"] is True and ex.options["adjustForTimeDifference"] is True


def test_bybit_ws_ohlcv_keeps_the_confirm_flag():
    row = BybitPro().parse_ws_ohlcv({"start": 60000, "open": "1", "high": "2", "low": "0.5", "close": "1.5",
                                     "volume": "10", "confirm": True}, {"inverse": False})
    assert row == [60000, 1.0, 2.0, 0.5, 1.5, 10.0, True]
    assert BybitProfile().bar_confirmed(row) is True
