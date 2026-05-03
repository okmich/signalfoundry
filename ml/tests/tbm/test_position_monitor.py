"""Tests for ml.tbm.position_monitor — long, short, side semantics, OHLC ticks."""

import math
from datetime import datetime, timezone

import pytest

from okmich_quant_ml.tbm.position_monitor import (
    BarrierHit,
    PositionBook,
    PositionMonitor,
    PositionResult,
)


def _make_monitor(pid="p1", entry_price=100.0, vol=0.02, pt=1.0, sl=1.0,
                  expiry=datetime(2026, 1, 1, 1, 0), side=1, sign_vertical=False):
    return PositionMonitor(
        position_id=pid, entry_price=entry_price, entry_time=datetime(2026, 1, 1),
        volatility=vol, pt_multiplier=pt, sl_multiplier=sl, expiry_time=expiry,
        side=side, sign_vertical=sign_vertical,
    )


class TestLongPositionMonitor:
    def test_open_when_no_barrier_hit(self):
        m = _make_monitor()
        # entry 100 * (1 + 0.02) = 102 upper; (1 - 0.02) = 98 lower
        assert m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.5) is None

    def test_upper_hit_via_close(self):
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=102.0)
        assert result.barrier_hit == BarrierHit.UPPER
        assert result.label == 1
        assert result.side == 1
        assert result.ret > 0

    def test_lower_hit_via_close(self):
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=98.0)
        assert result.barrier_hit == BarrierHit.LOWER
        assert result.label == -1
        assert result.ret < 0

    def test_upper_hit_via_intrabar_high(self):
        m = _make_monitor()
        # Close inside barriers; high reaches upper
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=101.0, high=102.5, low=100.5)
        assert result.barrier_hit == BarrierHit.UPPER
        assert result.label == 1

    def test_horizontal_priority_over_vertical(self):
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 1, 0), price=102.0)
        assert result.barrier_hit == BarrierHit.UPPER

    def test_vertical_unsigned(self):
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 1, 0), price=100.5)
        assert result.barrier_hit == BarrierHit.VERTICAL
        assert result.label == 0

    def test_vertical_signed_long(self):
        m = _make_monitor(sign_vertical=True)
        assert m.on_bar(datetime(2026, 1, 1, 1, 0), price=100.5).label == 1
        m2 = _make_monitor(sign_vertical=True)
        assert m2.on_bar(datetime(2026, 1, 1, 1, 0), price=99.5).label == -1


class TestShortPositionMonitor:
    def test_short_lower_hit_is_win(self):
        # Short at 100; pt=1.0 -> profit target at 100*(1-0.02)=98 (the LOWER barrier)
        m = _make_monitor(side=-1)
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=98.0)
        assert result.barrier_hit == BarrierHit.LOWER
        assert result.label == 1  # bet won
        assert result.ret > 0  # short gained from price drop

    def test_short_upper_hit_is_loss(self):
        # Short stop at 100*(1+0.02)=102 (the UPPER barrier)
        m = _make_monitor(side=-1)
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=102.0)
        assert result.barrier_hit == BarrierHit.UPPER
        assert result.label == -1
        assert result.ret < 0

    def test_short_asymmetric_pt_sl(self):
        # pt=2 (target far below), sl=1 (stop nearer above)
        m = _make_monitor(side=-1, pt=2.0, sl=1.0)
        # Profit target = 100 * (1 - 2*0.02) = 96; stop = 100 * (1 + 1*0.02) = 102
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=95.0)
        assert result.barrier_hit == BarrierHit.LOWER
        assert result.label == 1

    def test_short_vertical_signed(self):
        m = _make_monitor(side=-1, sign_vertical=True)
        # Price below entry at expiry -> short won
        result = m.on_bar(datetime(2026, 1, 1, 1, 0), price=99.5)
        assert result.label == 1

    def test_invalid_side(self):
        with pytest.raises(ValueError):
            _make_monitor(side=0)
        with pytest.raises(ValueError):
            _make_monitor(side=2)


class TestPriceValidation:
    def test_nan_price_rejected(self):
        m = _make_monitor()
        with pytest.raises(ValueError):
            m.on_bar(datetime(2026, 1, 1, 0, 30), price=math.nan)

    def test_zero_price_rejected(self):
        m = _make_monitor()
        with pytest.raises(ValueError):
            m.on_bar(datetime(2026, 1, 1, 0, 30), price=0.0)

    def test_nan_high_low_rejected(self):
        m = _make_monitor()
        with pytest.raises(ValueError):
            m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.0, high=math.nan, low=99.0)

    def test_low_above_price_rejected(self):
        m = _make_monitor()
        with pytest.raises(ValueError, match="OHLC inconsistent"):
            m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.0, high=101.0, low=100.5)

    def test_high_below_price_rejected(self):
        m = _make_monitor()
        with pytest.raises(ValueError, match="OHLC inconsistent"):
            m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.0, high=99.5, low=99.0)

    def test_inverted_high_low_rejected(self):
        m = _make_monitor()
        # high < low (vendor swap on weekend / corrupt feed). price must satisfy
        # low <= price <= high, which already rejects this. Use price equal to
        # high to bypass that check and verify the high<low check catches it.
        with pytest.raises(ValueError, match="OHLC inconsistent"):
            m.on_bar(datetime(2026, 1, 1, 0, 30), price=99.0, high=99.0, low=100.0)


class TestTimezoneConsistency:
    def test_tz_mismatch_at_construction(self):
        with pytest.raises(ValueError, match="tz-awareness"):
            PositionMonitor(
                position_id="p1", entry_price=100.0, entry_time=datetime(2026, 1, 1),
                volatility=0.02, pt_multiplier=1.0, sl_multiplier=1.0,
                expiry_time=datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
            )

    def test_tz_mismatch_in_on_bar(self):
        m = _make_monitor()
        with pytest.raises(ValueError, match="tz-awareness"):
            m.on_bar(datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc), price=100.5)


class TestPositionResult:
    def test_immutable(self):
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=102.0)
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            result.label = 99


class TestPositionBookErrorIsolation:
    def test_one_bad_position_does_not_kill_book(self, caplog):
        book = PositionBook()
        book.open_position(_make_monitor(pid="p1"))
        book.open_position(_make_monitor(pid="p2"))
        # p1 gets a bad bar; p2 a clean one (price hits upper).
        prices = {"p1": math.nan, "p2": 102.0}
        with caplog.at_level("WARNING"):
            exits = book.on_bar(datetime(2026, 1, 1, 0, 30), prices)
        assert len(exits) == 1
        assert exits[0].position_id == "p2"
        # p1 still in book (not closed by the bad bar)
        assert "p1" in book.open_positions
        assert any("on_bar error" in m for m in caplog.messages)


class TestEWMAVolatilityEstimatorValidation:
    def test_negative_warm_up_rejected(self):
        from okmich_quant_ml.tbm.vol_estimator import EWMAVolatilityEstimator
        with pytest.raises(ValueError, match="warm_up_bars"):
            EWMAVolatilityEstimator(span=10, warm_up_bars=-1)


class TestExitAtBarrierPrice:
    def test_horizontal_hit_exit_at_barrier_not_close(self):
        # entry=100, vol=0.02, pt=1 -> upper=102. Wick high reaches 103 with close at 101.
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=101.0, high=103.0, low=100.5)
        assert result.barrier_hit == BarrierHit.UPPER
        # Exit price is the barrier (102), NOT the close (101)
        assert result.exit_price == pytest.approx(102.0)
        # ret derived from barrier
        assert result.ret == pytest.approx(math.log(102.0 / 100.0))

    def test_lower_hit_exit_at_barrier_not_close(self):
        m = _make_monitor()  # lower = 98
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=99.0, high=99.5, low=97.0)
        assert result.barrier_hit == BarrierHit.LOWER
        assert result.exit_price == pytest.approx(98.0)
        assert result.ret == pytest.approx(math.log(98.0 / 100.0))

    def test_vertical_exit_at_close(self):
        m = _make_monitor()
        # No barrier touch; expiry hits with close=100.5
        result = m.on_bar(datetime(2026, 1, 1, 1, 0), price=100.5)
        assert result.barrier_hit == BarrierHit.VERTICAL
        assert result.exit_price == pytest.approx(100.5)


class TestSameBarPolicy:
    def test_default_worst_case_lower_wins(self):
        # Bar where high >= upper AND low <= lower
        m = _make_monitor()
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.0, high=103.0, low=97.0)
        assert result.barrier_hit == BarrierHit.LOWER
        assert result.label == -1

    def test_upper_first_policy(self):
        from okmich_quant_ml.tbm.position_monitor import BarrierTiePolicy
        m = PositionMonitor(
            position_id="p", entry_price=100.0, entry_time=datetime(2026, 1, 1),
            volatility=0.02, pt_multiplier=1.0, sl_multiplier=1.0,
            expiry_time=datetime(2026, 1, 1, 1, 0), side=1,
            same_bar_policy=BarrierTiePolicy.UPPER_FIRST,
        )
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.0, high=103.0, low=97.0)
        assert result.barrier_hit == BarrierHit.UPPER
        assert result.label == 1

    def test_short_worst_case_means_short_loses(self):
        # For a SHORT, lower=profit and upper=stop. Same-bar WORST_CASE picks LOWER barrier
        # (path-direction) but for a short the BET label is side*path = -1*-1 = +1 (won).
        # Wait — that's the OPPOSITE of "worst case for the bet". The convention is
        # path-directional worst-case (downside dominant in PRICE terms), not bet-worst-case.
        # Document & assert this behavior so future readers don't get confused.
        from okmich_quant_ml.tbm.position_monitor import BarrierTiePolicy
        m = PositionMonitor(
            position_id="s", entry_price=100.0, entry_time=datetime(2026, 1, 1),
            volatility=0.02, pt_multiplier=1.0, sl_multiplier=1.0,
            expiry_time=datetime(2026, 1, 1, 1, 0), side=-1,
            same_bar_policy=BarrierTiePolicy.WORST_CASE,
        )
        result = m.on_bar(datetime(2026, 1, 1, 0, 30), price=100.0, high=103.0, low=97.0)
        # WORST_CASE picks LOWER barrier; for short, lower=profit -> bet won
        assert result.barrier_hit == BarrierHit.LOWER
        assert result.label == 1  # short won (lower hit means price dropped)


class TestPositionBook:
    def test_simultaneous_exits(self):
        book = PositionBook()
        for i in range(5):
            book.open_position(_make_monitor(pid=f"p{i}"))
        prices = {f"p{i}": 102.0 for i in range(5)}
        exits = book.on_bar(datetime(2026, 1, 1, 0, 30), prices)
        assert len(exits) == 5
        assert book.n_open == 0

    def test_partial_exits(self):
        book = PositionBook()
        for i in range(5):
            book.open_position(_make_monitor(pid=f"p{i}"))
        prices = {f"p{i}": (102.0 if i < 2 else 100.5) for i in range(5)}
        exits = book.on_bar(datetime(2026, 1, 1, 0, 30), prices)
        assert len(exits) == 2
        assert book.n_open == 3

    def test_missing_price_skips_with_warning(self, caplog):
        book = PositionBook()
        book.open_position(_make_monitor(pid="p1"))
        book.open_position(_make_monitor(pid="p2"))
        with caplog.at_level("WARNING"):
            exits = book.on_bar(datetime(2026, 1, 1, 0, 30), {"p1": 102.0})
        assert len(exits) == 1
        assert exits[0].position_id == "p1"
        assert any("missing price" in m for m in caplog.messages)

    def test_duplicate_open_raises(self):
        book = PositionBook()
        book.open_position(_make_monitor(pid="p1"))
        with pytest.raises(ValueError):
            book.open_position(_make_monitor(pid="p1"))

    def test_book_with_intrabar_high_low(self):
        book = PositionBook()
        book.open_position(_make_monitor(pid="p1"))
        # Close inside, intrabar high reaches barrier
        prices = {"p1": 101.0}
        highs = {"p1": 102.5}
        lows = {"p1": 100.0}
        exits = book.on_bar(datetime(2026, 1, 1, 0, 30), prices, highs=highs, lows=lows)
        assert len(exits) == 1
        assert exits[0].barrier_hit == BarrierHit.UPPER
