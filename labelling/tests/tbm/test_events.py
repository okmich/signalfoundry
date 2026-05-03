"""Tests for labelling.tbm.events — CUSUM and vertical barrier."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.tbm.events import cusum_filter, get_vertical_barrier


class TestCusumFilter:
    def test_hand_computed_reference(self):
        # Threshold = 1.0, raw values: 0.5, 0.3, 0.4, -0.6, -0.8
        # i=0: s_pos=0.5, s_neg=0;     no trigger
        # i=1: s_pos=0.8, s_neg=0;     no trigger
        # i=2: s_pos=1.2 -> trigger; reset both
        # i=3: s_pos=0,   s_neg=-0.6;  no trigger
        # i=4: s_pos=0,   s_neg=-1.4 -> trigger
        idx = pd.date_range("2026-01-01", periods=5, freq="1min")
        series = pd.Series([0.5, 0.3, 0.4, -0.6, -0.8], index=idx)
        events = cusum_filter(series, threshold=1.0)
        assert list(events) == [idx[2], idx[4]]

    def test_raw_values_not_diff(self):
        # If we had used .diff() this would emit different events.
        # Constant returns of 0.6 each step -> sum > 1 fast -> trigger every 2 bars.
        idx = pd.date_range("2026-01-01", periods=10, freq="1min")
        series = pd.Series([0.6] * 10, index=idx)
        events = cusum_filter(series, threshold=1.0)
        assert len(events) > 0
        # First trigger when cumulative sum >= 1.0: i=1 since 0.6+0.6=1.2
        assert events[0] == idx[1]

    def test_both_arms_reset_on_trigger(self):
        # Designed to distinguish proper both-arm reset from a bug that only resets s_pos.
        # Sequence: -0.4, -0.4, +1.5 (triggers s_pos at i=2), -0.5.
        # With proper reset: at i=2 BOTH arms zero out, so at i=3 s_neg = -0.5, no trigger.
        # With s_pos-only reset: s_neg carries -0.8 across the i=2 trigger, then at i=3
        # s_neg = -1.3 <= -1.0 -> spurious second trigger at i=3.
        idx = pd.date_range("2026-01-01", periods=4, freq="1min")
        series = pd.Series([-0.4, -0.4, 1.5, -0.5], index=idx)
        events = cusum_filter(series, threshold=1.0)
        assert list(events) == [idx[2]]

    def test_empty_input(self):
        s = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        events = cusum_filter(s, threshold=1.0)
        assert len(events) == 0

    def test_invalid_threshold(self, gbm_close):
        with pytest.raises(ValueError):
            cusum_filter(np.log(gbm_close / gbm_close.shift(1)).dropna(), threshold=0.0)


class TestGetVerticalBarrier:
    def test_num_bars_iloc_positional(self):
        # Build sparse index — Mondays in January (5 bars), then another month later
        idx = pd.DatetimeIndex(["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26",
                                "2026-02-02", "2026-02-09", "2026-02-16"])
        close = pd.Series(range(len(idx)), index=idx)
        events = pd.DatetimeIndex([idx[0], idx[2]])
        t1 = get_vertical_barrier(events, close, num_bars=2)
        # iloc-positional: 2 bars after 0 = idx[2]; 2 bars after 2 = idx[4]
        assert t1.iloc[0] == idx[2]
        assert t1.iloc[1] == idx[4]

    def test_num_bars_clipped_to_last(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="1h")
        close = pd.Series(range(10), index=idx)
        events = pd.DatetimeIndex([idx[8]])
        t1 = get_vertical_barrier(events, close, num_bars=5)
        assert t1.iloc[0] == idx[-1]

    def test_num_days_calendar_arithmetic(self):
        idx = pd.date_range("2026-01-01", periods=20, freq="1D")
        close = pd.Series(range(20), index=idx)
        events = pd.DatetimeIndex([idx[0]])
        t1 = get_vertical_barrier(events, close, num_days=5)
        assert t1.iloc[0] == pd.Timestamp("2026-01-06")

    def test_num_days_with_gaps(self):
        # 1-day gap; ask for 3 days from t0 — should land on first index >= t0 + 3D
        idx = pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-08"])
        close = pd.Series(range(len(idx)), index=idx)
        events = pd.DatetimeIndex([idx[0]])
        t1 = get_vertical_barrier(events, close, num_days=3)
        assert t1.iloc[0] == pd.Timestamp("2026-01-05")

    def test_neither_or_both_raises(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        close = pd.Series(range(5), index=idx)
        events = pd.DatetimeIndex([idx[0]])
        with pytest.raises(ValueError):
            get_vertical_barrier(events, close)
        with pytest.raises(ValueError):
            get_vertical_barrier(events, close, num_bars=1, num_days=1)

    def test_empty_events(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        close = pd.Series(range(5), index=idx)
        t1 = get_vertical_barrier(pd.DatetimeIndex([]), close, num_bars=1)
        assert len(t1) == 0

    def test_num_days_membership_check(self):
        # num_days mode now also requires events ⊆ close.index (matches num_bars).
        idx = pd.date_range("2026-01-01", periods=5, freq="1D")
        close = pd.Series(range(5), index=idx)
        events = pd.DatetimeIndex(["2099-01-01"])
        with pytest.raises(KeyError, match="not present in close.index"):
            get_vertical_barrier(events, close, num_days=1)

    def test_tz_mismatch_rejected(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        close = pd.Series(range(5), index=idx)
        tz_events = pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")])
        with pytest.raises(ValueError, match="tz="):
            get_vertical_barrier(tz_events, close, num_bars=1)


class TestCusumFilterInputValidation:
    def test_non_datetime_index_rejected(self):
        s = pd.Series([0.1, -0.2, 0.3], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            cusum_filter(s, threshold=0.5)

    def test_unsorted_index_rejected(self):
        idx = pd.DatetimeIndex(["2026-01-02", "2026-01-01", "2026-01-03"])
        s = pd.Series([0.1, -0.2, 0.3], index=idx)
        with pytest.raises(ValueError, match="monotonic"):
            cusum_filter(s, threshold=0.5)

    def test_duplicate_index_rejected(self):
        idx = pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-02"])
        s = pd.Series([0.1, 0.2, 0.3], index=idx)
        with pytest.raises(ValueError, match="unique"):
            cusum_filter(s, threshold=0.5)
