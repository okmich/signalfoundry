"""Precision / minimums / contract size, candle-close detection, and average-cost P&L."""
import pytest

from okmich_quant_crypto import MarketType, OrderSide, SizingUnit
from okmich_quant_crypto.bar_aggregator import CandleCloseDetector
from okmich_quant_crypto.markets import MarketSpec, OrderSizeError, validate_market_type
from okmich_quant_crypto.models import Fill
from okmich_quant_crypto.pnl import Book, last_round_trip, summarize

from .fakes import FakeExchange, perp_market, spot_market


def _spec(exchange, symbol="BTC/USDT:USDT", mt=MarketType.LINEAR_PERP):
    return MarketSpec.from_market(exchange, exchange.market(symbol), mt)


def test_amount_is_truncated_never_rounded_up(exchange):
    spec = _spec(exchange)
    assert spec.checked_amount(0.0129, 1000.0) == pytest.approx(0.012)


def test_below_minimum_amount_and_notional_are_rejected(exchange):
    spec = _spec(exchange)
    with pytest.raises(OrderSizeError, match="rounds to zero"):
        spec.checked_amount(0.0004, 100.0)
    with pytest.raises(OrderSizeError, match="notional"):
        spec.checked_amount(0.002, 100.0)  # 0.2 USDT < 5 USDT min cost
    with pytest.raises(OrderSizeError, match="positive"):
        spec.checked_amount(-1.0, 100.0)


def test_contract_size_conversion(clock):
    okx_like = FakeExchange(clock, markets=[perp_market(contract_size=0.01, amount_step=1, min_amount=1)])
    spec = _spec(okx_like)
    assert spec.base_to_amount(0.05) == pytest.approx(5)          # 0.05 BTC = 5 contracts
    assert spec.amount_to_base(5) == pytest.approx(0.05)
    assert spec.checked_amount(0.0549, 1000.0) == 5               # truncated to whole contracts
    assert spec.units_to_base(3, SizingUnit.CONTRACTS, 100.0) == pytest.approx(0.03)


def test_sizing_units(exchange):
    spec = _spec(exchange)
    assert spec.units_to_base(0.5, SizingUnit.BASE_QTY, 100.0) == 0.5
    assert spec.units_to_base(1000, SizingUnit.QUOTE_NOTIONAL, 100.0) == 10
    with pytest.raises(OrderSizeError):
        spec.units_to_base(1000, SizingUnit.QUOTE_NOTIONAL, 0.0)
    spot = _spec(exchange, "BTC/USDT", MarketType.SPOT)
    with pytest.raises(OrderSizeError):
        spot.units_to_base(1, SizingUnit.CONTRACTS, 100.0)


def test_market_type_is_validated():
    with pytest.raises(ValueError, match="not a spot market"):
        validate_market_type(perp_market(), MarketType.SPOT)
    with pytest.raises(ValueError, match="not a linear perpetual"):
        validate_market_type(spot_market(), MarketType.LINEAR_PERP)
    inverse = perp_market() | {"linear": False, "inverse": True}
    with pytest.raises(ValueError, match="not a linear perpetual"):
        validate_market_type(inverse, MarketType.LINEAR_PERP)


def test_dust(exchange):
    spec = _spec(exchange)
    assert spec.is_dust(0.0005) and not spec.is_dust(0.001) and spec.is_dust(0.0)


# ---------------------------------------------------------------------- candle-close detection

TF = 300_000


def test_rollover_closes_the_previous_candle():
    d = CandleCloseDetector(TF)
    assert d.feed([[0, 1, 1, 1, 1, 1]]) == []
    assert d.feed([[0, 1, 2, 1, 2, 2]]) == []          # same candle, updated
    assert d.feed([[TF, 2, 2, 2, 2, 1]]) == [0]        # newer open -> 0 closed
    assert d.feed([[TF, 2, 3, 2, 3, 2]]) == []


def test_confirm_flag_closes_without_waiting_for_rollover():
    d = CandleCloseDetector(TF)
    assert d.feed([[0, 1, 1, 1, 1, 1, False]]) == []
    assert d.feed([[0, 1, 1, 1, 1, 1, True]]) == [0]
    assert d.feed([[TF, 1, 1, 1, 1, 1, False]]) == []  # rollover does not report 0 twice


def test_missed_final_update_and_skipped_candles():
    d = CandleCloseDetector(TF)
    d.feed([[0, 1, 1, 1, 1, 1]])
    assert d.feed([[3 * TF, 1, 1, 1, 1, 1]]) == [0]   # only the candle it saw; the gap is the sequencer's job


def test_misaligned_candle_raises():
    with pytest.raises(ValueError, match="UTC grid"):
        CandleCloseDetector(TF).feed([[123, 1, 1, 1, 1, 1]])


# ---------------------------------------------------------------------- P&L

def _fill(side, qty, price, ts, fee_quote=0.0, fee_base=0.0, tid=None):
    return Fill(trade_id=tid or f"{side}{ts}", order_id=None, client_order_id=None, timestamp_ms=ts,
                side=OrderSide(side), price=price, base_qty=qty, fee_quote=fee_quote, fee_base=fee_base)


def test_long_round_trip_average_cost():
    book = summarize([_fill("buy", 1, 100, 1, 0.1), _fill("buy", 1, 110, 2, 0.1), _fill("sell", 2, 120, 3, 0.2)])
    assert book.is_flat
    assert book.realized == pytest.approx((120 - 105) * 2)
    assert book.fees_quote == pytest.approx(0.4)
    assert book.avg_entry == pytest.approx(105) and book.avg_exit == pytest.approx(120)
    assert book.closed_qty == pytest.approx(2)


def test_short_round_trip_with_partial_exits():
    book = summarize([_fill("sell", 2, 100, 1), _fill("buy", 1, 90, 2), _fill("buy", 1, 95, 3)])
    assert book.is_flat and book.realized == pytest.approx(10 + 5)


def test_spot_base_fee_is_not_received_but_is_a_cost():
    # Buy 1 BTC @100 paying 0.001 BTC fee -> hold 0.999; sell 0.999 @110 paying 0.1 USDT.
    book = summarize([_fill("buy", 1, 100, 1, fee_quote=0.1, fee_base=0.001), _fill("sell", 0.999, 110, 2, 0.1)],
                     spot=True)
    assert book.is_flat
    net = book.realized - book.fees_quote
    cash = -100 + 0.999 * 110 - 0.1
    assert net == pytest.approx(cash)


def test_last_round_trip_discards_earlier_trips_in_the_lookback():
    fills = [_fill("buy", 1, 50, 1), _fill("sell", 1, 60, 2),        # an older trip
             _fill("sell", 1, 100, 3), _fill("buy", 1, 90, 4)]       # the one being resolved
    trip, complete = last_round_trip(fills)
    assert complete and [f.timestamp_ms for f in trip] == [3, 4]


def test_round_trip_is_incomplete_when_the_window_started_mid_position():
    # A long opened before the window: only its closing sell is visible.
    trip, complete = last_round_trip([_fill("sell", 1, 100, 5)])
    assert not complete and trip == []


def test_reversal_opens_the_remainder_at_the_fill_price():
    b = Book()
    b.apply(_fill("buy", 1, 100, 1))
    b.apply(_fill("sell", 3, 110, 2))
    assert b.qty == pytest.approx(-2) and b.avg_price == 110 and b.realized == pytest.approx(10)
