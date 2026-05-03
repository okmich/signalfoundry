"""Tests for ml.tbm.barriers — pure geometry."""

import math

import pytest

from okmich_quant_ml.tbm.barriers import check_barrier_touch, compute_barrier_levels


class TestComputeBarrierLevels:
    def test_two_sided_unitless_vol_contract(self):
        # entry=100, vol=0.01 (1% return-vol), pt=1.5 -> upper = 100*(1+0.015) = 101.5
        levels = compute_barrier_levels(entry_price=100.0, volatility=0.01,
                                        pt_multiplier=1.5, sl_multiplier=1.0)
        assert levels["upper"] == pytest.approx(101.5)
        assert levels["lower"] == pytest.approx(99.0)

    def test_one_sided_no_pt(self):
        levels = compute_barrier_levels(100.0, 0.02, pt_multiplier=0.0, sl_multiplier=1.0)
        assert levels["upper"] is None
        assert levels["lower"] == pytest.approx(98.0)

    def test_one_sided_no_sl(self):
        levels = compute_barrier_levels(100.0, 0.02, pt_multiplier=1.5, sl_multiplier=0.0)
        assert levels["upper"] == pytest.approx(103.0)
        assert levels["lower"] is None

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf, -math.inf])
    def test_invalid_entry_price(self, bad_value):
        with pytest.raises(ValueError):
            compute_barrier_levels(bad_value, 0.01, 1.0, 1.0)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, math.nan, math.inf])
    def test_invalid_volatility(self, bad_value):
        with pytest.raises(ValueError):
            compute_barrier_levels(100.0, bad_value, 1.0, 1.0)

    @pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
    def test_invalid_pt_multiplier(self, bad_value):
        with pytest.raises(ValueError):
            compute_barrier_levels(100.0, 0.01, bad_value, 1.0)

    @pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
    def test_invalid_sl_multiplier(self, bad_value):
        with pytest.raises(ValueError):
            compute_barrier_levels(100.0, 0.01, 1.0, bad_value)

    def test_rejects_non_positive_lower_barrier(self):
        # sl * vol >= 1 -> lower = entry * (1 - 1) <= 0
        with pytest.raises(ValueError, match="non-positive lower barrier"):
            compute_barrier_levels(100.0, 0.5, pt_multiplier=1.0, sl_multiplier=2.0)
        with pytest.raises(ValueError, match="non-positive lower barrier"):
            compute_barrier_levels(100.0, 1.0, pt_multiplier=1.0, sl_multiplier=1.0)


class TestCheckBarrierTouch:
    def test_neither(self):
        assert check_barrier_touch(price=100.0, upper=105.0, lower=95.0) == 0

    def test_upper_hit(self):
        assert check_barrier_touch(105.0, 105.0, 95.0) == 1
        assert check_barrier_touch(106.0, 105.0, 95.0) == 1

    def test_lower_hit(self):
        assert check_barrier_touch(95.0, 105.0, 95.0) == -1
        assert check_barrier_touch(94.0, 105.0, 95.0) == -1

    def test_gap_through_both_upper_wins(self):
        assert check_barrier_touch(price=100.0, upper=99.0, lower=101.0) == 1

    def test_upper_only_barrier(self):
        assert check_barrier_touch(105.0, 105.0, None) == 1
        assert check_barrier_touch(50.0, 105.0, None) == 0

    def test_lower_only_barrier(self):
        assert check_barrier_touch(95.0, None, 95.0) == -1
        assert check_barrier_touch(150.0, None, 95.0) == 0

    def test_both_none(self):
        assert check_barrier_touch(100.0, None, None) == 0
