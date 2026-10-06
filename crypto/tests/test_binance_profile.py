"""Binance venue profile: environments, stops (algo orders), streams, lookups, windowed history, error codes."""
import pytest
from ccxt.base import errors as e

from okmich_quant_crypto import BinanceProfile, MarketType, StopMode, StopTrigger, VenueEnvironment
from okmich_quant_crypto.capabilities import resolve_capabilities
from okmich_quant_crypto.markets import MarketSpec
from okmich_quant_crypto.models import Credentials
from okmich_quant_crypto.resilience import ErrorClass, VenueUnsupportedError, classify_ccxt_error
from okmich_quant_crypto.venue import resolve_profile, supported_venues

from .conftest import make_cfg
from .fakes import FakeExchange, perp_market, spot_market

DAY = 24 * 3600 * 1000
PERP, SPOT = "BTC/USDT:USDT", "BTC/USDT"


def test_binance_is_supported():
    assert "binance" in supported_venues() and isinstance(resolve_profile("binance"), BinanceProfile)


def test_demo_hosts_and_safe_options():
    ex = BinanceProfile().build_exchange(VenueEnvironment.DEMO, Credentials(api_key="k", secret="s"))
    assert ex.options["enableDemoTrading"] is True and ex.options["defaultType"] == "swap"
    assert "demo-fapi.binance.com" in ex.urls["api"]["fapiPrivate"]
    assert "demo-api.binance.com" in ex.urls["api"]["private"]
    assert ex.options["watchPositions"] == {"fetchPositionsSnapshot": False, "awaitPositionsSnapshot": False}


def test_testnet_is_refused_in_favour_of_demo():
    with pytest.raises(VenueUnsupportedError, match="use environment 'demo'"):
        BinanceProfile().build_exchange(VenueEnvironment.TESTNET, Credentials())


def test_stop_capabilities_and_auto_resolution(exchange):
    prof = BinanceProfile()
    perp = prof.stop_capabilities(None, MarketType.LINEAR_PERP)
    assert perp.standalone_conditional and not perp.position_level and not perp.attached_on_market_entry
    assert not prof.stop_capabilities(None, MarketType.SPOT).any_native
    assert resolve_capabilities(exchange, prof, make_cfg()).stop_mode is StopMode.NATIVE
    spot_cfg = make_cfg(market_symbol=SPOT, market_type="spot")
    assert resolve_capabilities(exchange, prof, spot_cfg).stop_mode is StopMode.MANAGED
    with pytest.raises(VenueUnsupportedError, match="NATIVE stops requested"):
        resolve_capabilities(exchange, prof, make_cfg(market_symbol=SPOT, market_type="spot", stop_mode="native"))


def test_index_trigger_is_rejected_for_perps(exchange):
    with pytest.raises(VenueUnsupportedError, match="index"):
        resolve_capabilities(exchange, BinanceProfile(), make_cfg(stop_trigger="index"))


def test_algo_stop_params_carry_working_type():
    prof = BinanceProfile()
    p = prof.standalone_stop_params(MarketType.LINEAR_PERP, 95.0, True, StopTrigger.MARK)
    assert p == {"stopLossPrice": 95.0, "reduceOnly": True, "workingType": "MARK_PRICE"}
    p = prof.standalone_stop_params(MarketType.LINEAR_PERP, 110.0, False, StopTrigger.LAST)
    assert p["takeProfitPrice"] == 110.0 and p["workingType"] == "CONTRACT_PRICE"
    assert prof.stop_order_params(MarketType.LINEAR_PERP) == {"trigger": True}   # algo orders need it everywhere
    assert prof.attached_stop_params is not None


def test_balance_params_select_the_account():
    prof = BinanceProfile()
    assert prof.balance_params(MarketType.SPOT) == {"type": "spot"}
    assert prof.balance_params(MarketType.LINEAR_PERP) == {"type": "swap"}


class _Stub:
    def __init__(self, clock=None):
        self.calls = []
        self.trades = []
        self.funding = []
        self.orders = {}
        self.now = 100 * DAY
        self.markets = {m["symbol"]: m for m in (perp_market(), spot_market())}

    def market(self, s):
        return self.markets[s]

    def milliseconds(self):
        return self.now

    async def watch_orders(self, symbol=None, since=None, limit=None, params=None):
        self.calls.append(("watch_orders", params))
        return []

    async def watch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        self.calls.append(("watch_my_trades", params))
        return []

    async def watch_positions(self, symbols=None, since=None, limit=None, params=None):
        self.calls.append(("watch_positions", symbols))
        return []

    async def fetch_order(self, id, symbol=None, params=None):
        self.calls.append(("fetch_order", dict(params or {})))
        key = (params.get("clientOrderId"), bool(params.get("trigger")))
        if key not in self.orders:
            raise e.OrderNotFound('binance {"code":-2013,"msg":"Order does not exist."}')
        return self.orders[key]

    async def fetch_my_trades(self, symbol=None, since=None, limit=None, params=None):
        until = params["until"]
        self.calls.append(("fetch_my_trades", since, until))
        assert until - since < 7 * DAY if self.markets[symbol]["swap"] else until - since < DAY
        rows = [t for t in self.trades if since <= t["timestamp"] <= until]
        return rows[:limit]

    async def fetch_funding_history(self, symbol=None, since=None, limit=None, params=None):
        rows = [r for r in self.funding if since <= r["timestamp"] <= params["until"]]
        return rows[:limit]

    async def private_get_account(self):
        return {"uid": 123456}


async def test_account_streams_one_per_account():
    stub = _Stub()
    calls = BinanceProfile().account_stream_calls(stub, spot=True, perp_symbols=[PERP])
    assert [k for k, _ in calls] == ["orders", "fills", "positions", "orders", "fills"]
    for _, factory in calls:
        await factory()
    assert ("watch_orders", {"type": "swap"}) in stub.calls and ("watch_orders", {"type": "spot"}) in stub.calls
    assert ("watch_positions", [PERP]) in stub.calls                              # symbols always passed
    only_perp = BinanceProfile().account_stream_calls(stub, spot=False, perp_symbols=[PERP])
    assert len(only_perp) == 3


async def test_lookup_tries_regular_then_algo_orders():
    stub = _Stub()
    spec = MarketSpec.from_market(stub, stub.market(PERP), MarketType.LINEAR_PERP)
    stub.orders[("sf1xstop", True)] = {"id": "algo-9", "clientOrderId": "sf1xstop"}
    found = await BinanceProfile().find_order_by_client_id(stub, spec, "sf1xstop")
    assert found["id"] == "algo-9"
    assert [c[1] for c in stub.calls] == [{"clientOrderId": "sf1xstop"}, {"clientOrderId": "sf1xstop", "trigger": True}]
    assert await BinanceProfile().find_order_by_client_id(stub, spec, "sf1xnone") is None
    spot_spec = MarketSpec.from_market(stub, stub.market(SPOT), MarketType.SPOT)
    stub.calls.clear()
    assert await BinanceProfile().find_order_by_client_id(stub, spot_spec, "sf1xnone") is None
    assert len(stub.calls) == 1                                                  # spot has no algo orders


def _trade(i, ts):
    return {"id": str(i), "order": "o1", "timestamp": ts, "side": "buy", "price": 100.0, "amount": 0.01,
            "fee": {"cost": 0.001, "currency": "USDT"}, "info": {}}


async def test_fills_are_read_in_venue_windows():
    stub = _Stub()
    stub.trades = [_trade(i, i * 4 * DAY) for i in range(5)]                      # sparse: gaps longer than a window
    spec = MarketSpec.from_market(stub, stub.market(PERP), MarketType.LINEAR_PERP)
    fills = await BinanceProfile().fetch_fills(stub, spec, 0, 17 * DAY)
    assert [f.trade_id for f in fills] == ["0", "1", "2", "3", "4"]
    spot_spec = MarketSpec.from_market(stub, stub.market(SPOT), MarketType.SPOT)
    stub.calls.clear()
    await BinanceProfile().fetch_fills(stub, spot_spec, 0, 3 * DAY - 1)              # until is inclusive
    assert len([c for c in stub.calls if c[0] == "fetch_my_trades"]) == 3         # 24 h windows


async def test_funding_keeps_binance_income_sign():
    stub = _Stub()
    stub.funding = [{"id": "1", "timestamp": 10, "amount": -0.375, "code": "USDT"},
                    {"id": "2", "timestamp": 20, "amount": 0.12, "code": "USDT"}]
    spec = MarketSpec.from_market(stub, stub.market(PERP), MarketType.LINEAR_PERP)
    payments = await BinanceProfile().fetch_funding(stub, spec, 0, 100)
    assert [p.amount for p in payments] == [-0.375, 0.12]                          # positive = received
    spot_spec = MarketSpec.from_market(stub, stub.market(SPOT), MarketType.SPOT)
    assert await BinanceProfile().fetch_funding(stub, spot_spec, 0, 100) == []


async def test_account_uid():
    assert await BinanceProfile().account_uid(_Stub()) == "123456"

    class _NoUid(_Stub):
        async def private_get_account(self):
            raise e.NotSupported("demo")

    assert await BinanceProfile().account_uid(_NoUid()) is None


@pytest.mark.parametrize("exc,expected", [
    (e.OperationRejected('binance {"code":-4059,"msg":"No need to change position side."}'), ErrorClass.NO_CHANGE),
    (e.InvalidOrder('binance {"code":-4116,"msg":"ClientOrderId is duplicated."}'), ErrorClass.DUPLICATE),
    (e.InvalidOrder('binance {"code":-2010,"msg":"Duplicate order sent."}'), ErrorClass.DUPLICATE),
    (e.DDoSProtection('binance {"code":-2015,"msg":"Invalid API-key, IP, or permissions for action."}'),
     ErrorClass.BANNED),
    (e.DDoSProtection('binance 418 {"code":-1003,"msg":"Way too many requests; IP banned until 1700000000000."}'),
     ErrorClass.BANNED),
    (e.RateLimitExceeded('binance {"code":-1003,"msg":"Too many requests."}'), ErrorClass.TRANSIENT),
    (e.InvalidNonce('binance {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}'),
     ErrorClass.CLOCK_SKEW),
    (e.OrderNotFound('binance {"code":-2011,"msg":"Unknown order sent."}'), ErrorClass.NOT_FOUND),
])
def test_binance_error_codes(exc, expected):
    assert classify_ccxt_error(exc, BinanceProfile(), placing_order=True) is expected


def test_client_id_from_websocket_and_algo_payloads():
    prof = BinanceProfile()
    assert prof.client_order_id_of({"clientOrderId": None, "info": {"c": "sf1xa"}}) == "sf1xa"
    assert prof.client_order_id_of({"clientOrderId": None, "info": {"clientAlgoId": "sf1xb"}}) == "sf1xb"
    assert prof.client_order_id_of({"clientOrderId": "sf1xc", "info": {}}) == "sf1xc"
    prof.client_id_rule.check("sf5001xabcdefghijklmnop")


async def test_strategy_reads_the_right_balance(exchange, venue, clock):
    from .conftest import bootstrap, make_strategy, seed_candles
    seen = []
    real = exchange.fetch_balance

    async def spy(params=None):
        seen.append(params)
        return await real(params)

    exchange.fetch_balance = spy
    seed_candles(exchange, PERP, 5 * 60_000, (clock() // 300_000) * 300_000 - 300_000, 10)
    s, _ = make_strategy(make_cfg(position_sizing={"type": "risk_pct_of_equity", "risk_pct": 0.01}))
    await bootstrap(s, exchange, venue, clock, BinanceProfile("fakex"))
    await s.calculate_quantity(100.0, 95.0)
    assert seen[-1] == {"type": "swap"}


def test_fake_exchange_still_constructs(clock):
    assert FakeExchange(clock).id == "fakex"
