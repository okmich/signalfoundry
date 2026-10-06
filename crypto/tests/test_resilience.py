"""CCXT exception classification, venue code overrides, and idempotent order placement."""
import pytest
from ccxt.base import errors as e

from okmich_quant_crypto import MarketType
from okmich_quant_crypto.functions.crypto import cancel_order_safe, place_order_idempotent
from okmich_quant_crypto.markets import MarketSpec
from okmich_quant_crypto.resilience import (
    CryptoBannedError, CryptoPermanentError, ErrorClass, OrderStateUnknownError, call_with_retry,
    classify_ccxt_error,
)
from okmich_quant_crypto.venue.base import VenueProfile
from okmich_quant_crypto.venue.bybit import BybitProfile


@pytest.mark.parametrize("exc,placing,expected", [
    (e.RequestTimeout("bybit POST url"), False, ErrorClass.TRANSIENT),
    (e.RequestTimeout("bybit POST url"), True, ErrorClass.UNKNOWN_STATE),
    (e.NetworkError("reset"), True, ErrorClass.UNKNOWN_STATE),
    (e.RateLimitExceeded("too many"), True, ErrorClass.TRANSIENT),
    (e.DDoSProtection("slow down"), False, ErrorClass.TRANSIENT),
    (e.DDoSProtection("binance GET https://x 418 I'm a teapot ip banned"), False, ErrorClass.BANNED),
    (e.ExchangeNotAvailable("bybit GET https://x 403 Forbidden <html>"), False, ErrorClass.BANNED),
    (e.ExchangeNotAvailable("bybit GET https://x 451 Unavailable For Legal Reasons"), False, ErrorClass.BANNED),
    (e.ExchangeNotAvailable("bybit GET https://x 503 Service Unavailable"), False, ErrorClass.TRANSIENT),
    (e.OnMaintenance("maintenance"), False, ErrorClass.TRANSIENT),
    (e.InvalidNonce("recv window"), False, ErrorClass.CLOCK_SKEW),
    (e.NoChange("not modified"), False, ErrorClass.NO_CHANGE),
    (e.MarginModeAlreadySet("already"), False, ErrorClass.NO_CHANGE),
    (e.OrderNotFound("gone"), False, ErrorClass.NOT_FOUND),
    (e.DuplicateOrderId("dup"), True, ErrorClass.DUPLICATE),
    (e.InsufficientFunds("no money"), True, ErrorClass.PERMANENT),
    (e.InvalidOrder("bad qty"), True, ErrorClass.PERMANENT),
    (e.BadSymbol("nope"), False, ErrorClass.PERMANENT),
    (e.AuthenticationError("bad key"), False, ErrorClass.PERMANENT),
    (e.PermissionDenied("no perm"), False, ErrorClass.PERMANENT),
    (e.AccountSuspended("suspended"), False, ErrorClass.BANNED),
    (e.ExchangeError("weird"), False, ErrorClass.PERMANENT),
])
def test_generic_classification(exc, placing, expected):
    assert classify_ccxt_error(exc, VenueProfile("okx"), placing_order=placing) is expected


@pytest.mark.parametrize("code,expected", [
    ("110072", ErrorClass.DUPLICATE), ("110043", ErrorClass.NO_CHANGE), ("110025", ErrorClass.NO_CHANGE),
    ("110008", ErrorClass.ALREADY_DONE), ("110010", ErrorClass.ALREADY_DONE), ("110001", ErrorClass.NOT_FOUND),
    ("10002", ErrorClass.CLOCK_SKEW), ("10006", ErrorClass.TRANSIENT), ("10016", ErrorClass.TRANSIENT),
    ("10010", ErrorClass.BANNED), ("34040", ErrorClass.NO_CHANGE),
])
def test_bybit_codes_override_the_generic_tree(code, expected):
    # CCXT raises these as InvalidOrder / BadRequest / ExchangeError with the JSON body in the message.
    exc = e.InvalidOrder(f'bybit {{"retCode":{code},"retMsg":"x","result":{{}}}}')
    assert classify_ccxt_error(exc, BybitProfile()) is expected


def test_unknown_bybit_code_falls_through_to_generic():
    exc = e.InsufficientFunds('bybit {"retCode":110007,"retMsg":"ab not enough"}')
    assert classify_ccxt_error(exc, BybitProfile()) is ErrorClass.PERMANENT


async def test_call_with_retry_retries_transient_only():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise e.RequestTimeout("t")
        return "ok"

    assert await call_with_retry(flaky, what="t", initial_delay=0) == "ok"
    assert calls["n"] == 3

    async def broken():
        raise e.InvalidOrder("bad")

    with pytest.raises(e.InvalidOrder):
        await call_with_retry(broken, what="b", initial_delay=0)


# ---------------------------------------------------------------------- idempotent placement

def _spec(exchange):
    return MarketSpec.from_market(exchange, exchange.market("BTC/USDT:USDT"), MarketType.LINEAR_PERP)


async def _place(exchange, cid="sf42xabc", **kw):
    return await place_order_idempotent(exchange, VenueProfile("okx"), _spec(exchange), "market", "buy", 0.01, None,
                                        {}, cid, **kw)


def _creates(exchange):
    return [c for c in exchange.calls if c[0] == "create_order"]


async def test_timeout_after_accept_finds_the_order_and_does_not_resend(exchange, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    exchange.create_script = [("accept_then", e.RequestTimeout("timed out"))]
    order = await _place(exchange)
    assert order["clientOrderId"] == "sf42xabc"
    assert len(_creates(exchange)) == 1
    assert len([o for o in exchange.orders.values() if o["clientOrderId"] == "sf42xabc"]) == 1


async def test_timeout_before_accept_resends_with_the_same_client_id(exchange, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    exchange.create_script = [e.RequestTimeout("timed out")]
    order = await _place(exchange)
    creates = _creates(exchange)
    assert len(creates) == 2
    assert {c[6]["clientOrderId"] for c in creates} == {"sf42xabc"}
    assert order["clientOrderId"] == "sf42xabc"


async def test_duplicate_id_is_resolved_by_lookup(exchange, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    exchange._new_order("BTC/USDT:USDT", "market", "buy", 0.01, None, {"clientOrderId": "sf42xabc"})
    exchange.create_script = [e.DuplicateOrderId("dup")]
    order = await _place(exchange)
    assert order["clientOrderId"] == "sf42xabc"
    assert len(_creates(exchange)) == 1


async def test_duplicate_that_cannot_be_found_is_unknown_state(exchange, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    exchange.create_script = [e.DuplicateOrderId("dup")]
    with pytest.raises(OrderStateUnknownError):
        await _place(exchange)


async def test_permanent_rejection_is_not_retried(exchange):
    exchange.create_script = [e.InsufficientFunds("no money")]
    with pytest.raises(CryptoPermanentError):
        await _place(exchange)
    assert len(_creates(exchange)) == 1


async def test_ban_raises_banned(exchange):
    exchange.create_script = [e.ExchangeNotAvailable("okx POST https://x 403 Forbidden")]
    with pytest.raises(CryptoBannedError):
        await _place(exchange)


async def test_persistent_timeouts_end_in_unknown_state(exchange, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    exchange.create_script = [e.RequestTimeout("t")] * 3
    with pytest.raises(OrderStateUnknownError) as info:
        await _place(exchange)
    assert info.value.client_order_id == "sf42xabc"


async def test_invalid_client_id_never_leaves_the_process(exchange):
    with pytest.raises(ValueError):
        await _place(exchange, cid="has spaces and is way too long for any venue rule")
    assert _creates(exchange) == []


async def test_cancel_of_a_gone_order_counts_as_cancelled(exchange):
    assert await cancel_order_safe(exchange, VenueProfile("okx"), _spec(exchange), "999") is True


async def _no_sleep(*_args, **_kwargs):
    return None
