"""Tests for labelling.tbm.labeling — get_labels and apply_min_return_filter."""

from math import log

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.tbm.labeling import (
    BarrierTiePolicy,
    apply_min_return_filter,
    get_labels,
)


def _make_prices(close_arr, index=None, hi_buf=0.0, lo_buf=0.0):
    """Build OHLC frame from a close array. high = close + hi_buf, low = close - lo_buf."""
    if index is None:
        index = pd.date_range("2026-01-01", periods=len(close_arr), freq="1h")
    close = pd.Series(close_arr, index=index)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + hi_buf,
        "low": close - lo_buf,
        "close": close,
    }, index=index)


class TestGetLabels:
    def test_upper_hit_via_close(self):
        # vol=0.01, pt=1.0 -> upper at entry*(1+0.01) = 101 (entry=100)
        prices = _make_prices(np.linspace(100.0, 102.0, 20))
        events = pd.Series([prices.index[15]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)
        row = result.iloc[0]
        assert row["label"] == 1
        assert row["barrier"] == "upper"

    def test_lower_hit(self):
        prices = _make_prices(np.linspace(100.0, 98.0, 20))
        events = pd.Series([prices.index[15]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.iloc[0]["label"] == -1
        assert result.iloc[0]["barrier"] == "lower"

    def test_intrabar_high_triggers_upper(self):
        # Close stays well inside barriers; only an intrabar spike on high punches through.
        # entry = close[0] = 100.0, vol = 0.01, pt = 1.0 -> upper = 100*(1.01) = 101.0
        n = 20
        close = np.full(n, 100.0)
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"open": close, "high": close.copy(), "low": close.copy(), "close": close}, index=idx)
        # Spike high on bar 5 to 101.5 (close-only walk would NOT trigger)
        df.loc[df.index[5], "high"] = 101.5
        events = pd.Series([df.index[15]], index=[df.index[0]])
        vol = pd.Series(0.01, index=df.index)
        result = get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.iloc[0]["barrier"] == "upper"
        assert result.iloc[0]["label"] == 1

    def test_intrabar_low_triggers_lower(self):
        n = 20
        close = np.full(n, 100.0)
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"open": close, "high": close.copy(), "low": close.copy(), "close": close}, index=idx)
        # Lower = 100*(0.99) = 99.0; spike low to 98.5
        df.loc[df.index[5], "low"] = 98.5
        events = pd.Series([df.index[15]], index=[df.index[0]])
        vol = pd.Series(0.01, index=df.index)
        result = get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.iloc[0]["barrier"] == "lower"
        assert result.iloc[0]["label"] == -1

    def test_vertical_expiry(self):
        prices = _make_prices(np.full(20, 100.0))
        events = pd.Series([prices.index[10]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.iloc[0]["label"] == 0
        assert result.iloc[0]["barrier"] == "vertical"

    def test_log_returns(self):
        close = np.linspace(100.0, 110.0, 10)
        prices = _make_prices(close)
        events = pd.Series([prices.index[9]], index=[prices.index[0]])
        # Wide-but-valid barriers (sl*vol < 1) -> no horizontal hit, vertical exit at last close
        vol = pd.Series(0.5, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)
        # entry=100, upper=150, lower=50; path 100->110 never hits -> vertical
        assert result.iloc[0]["barrier"] == "vertical"
        expected_ret = log(close[-1] / close[0])
        assert result.iloc[0]["ret"] == pytest.approx(expected_ret)

    def test_min_ret_filter_zeros_label_keeps_barrier(self):
        prices = _make_prices(np.linspace(100.0, 100.5, 20))
        events = pd.Series([prices.index[15]], index=[prices.index[0]])
        # Tiny vol -> barriers nearly at entry; small return triggers a hit
        vol = pd.Series(0.0001, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol, min_ret=0.05)
        assert result.iloc[0]["label"] == 0  # zeroed by min_ret
        assert result.iloc[0]["barrier"] in ("upper", "vertical", "lower")  # untouched

    def test_one_sided_pt_only(self):
        prices = _make_prices(np.linspace(100.0, 95.0, 20))
        events = pd.Series([prices.index[15]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 0.0], volatility=vol)
        # No lower; upper not hit (price falls); vertical exit
        assert result.iloc[0]["barrier"] == "vertical"
        assert result.iloc[0]["label"] == 0

    def test_invalid_pt_sl(self):
        prices = _make_prices(np.full(5, 100.0))
        events = pd.Series([prices.index[3]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        with pytest.raises(ValueError):
            get_labels(events, prices, pt_sl=[0.0, 0.0], volatility=vol)
        with pytest.raises(ValueError):
            get_labels(events, prices, pt_sl=[1.0], volatility=vol)
        with pytest.raises(ValueError):
            get_labels(events, prices, pt_sl=[-1.0, 1.0], volatility=vol)

    def test_missing_columns(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        bad = pd.DataFrame({"close": [100]*5}, index=idx)
        events = pd.Series([idx[3]], index=[idx[0]])
        vol = pd.Series(0.01, index=idx)
        with pytest.raises(ValueError, match="missing columns"):
            get_labels(events, bad, pt_sl=[1.0, 1.0], volatility=vol)

    def test_unsorted_index_rejected(self):
        idx = pd.DatetimeIndex(["2026-01-02", "2026-01-01", "2026-01-03"])
        df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                           "close": [1, 2, 3]}, index=idx)
        events = pd.Series([idx[0]], index=[idx[1]])
        vol = pd.Series(0.01, index=idx)
        with pytest.raises(ValueError, match="monotonic"):
            get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol)

    def test_non_unique_events_index_rejected(self):
        prices = _make_prices(np.linspace(100, 105, 20))
        # Two events at the same t0
        events = pd.Series([prices.index[10], prices.index[12]],
                           index=[prices.index[0], prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        with pytest.raises(ValueError, match="events.index must be unique"):
            get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)

    def test_non_numeric_close_rejected(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        df = pd.DataFrame({
            "open": [1.0, 2, 3, 4, 5], "high": [1, 2, 3, 4, 5], "low": [1, 2, 3, 4, 5],
            "close": ["a", "b", "c", "d", "e"],
        }, index=idx)
        events = pd.Series([idx[3]], index=[idx[0]])
        vol = pd.Series(0.01, index=idx)
        with pytest.raises(ValueError, match="numeric dtype"):
            get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol)

    def test_vol_no_overlap_rejected(self):
        prices = _make_prices(np.linspace(100, 105, 20))
        events = pd.Series([prices.index[10]], index=[prices.index[0]])
        # Vol on a totally different timeframe
        vol_idx = pd.date_range("2099-01-01", periods=5, freq="1h")
        vol = pd.Series([0.01] * 5, index=vol_idx)
        with pytest.raises(ValueError, match="overlap"):
            get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)

    def test_tz_mismatch_rejected(self):
        prices = _make_prices(np.linspace(100, 105, 20))
        events = pd.Series([prices.index[10]],
                           index=pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")]))
        vol = pd.Series(0.01, index=prices.index)
        with pytest.raises(ValueError, match="tz"):
            get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)

    def test_horizontal_exit_at_barrier_not_close(self):
        # Wick spike to upper but close stays BELOW entry. Old code (close-as-exit):
        # ret = log(close[5]/entry) < 0 with label=+1 -> sign mismatch. New code:
        # exit price = barrier price -> ret = log(upper/entry) > 0, sign matches.
        n = 20
        # Entry bar close = 100. Other bars hover at 99.5 (above lower=99, below upper=101).
        # Bar 5 has a wick to high=101.5 (touches upper) but close stays at 99.5.
        close = np.array([100.0] + [99.5] * (n - 1))
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"open": close, "high": close.copy(), "low": close.copy(), "close": close}, index=idx)
        df.loc[df.index[5], "high"] = 101.5
        events = pd.Series([df.index[15]], index=[df.index[0]])
        vol = pd.Series(0.01, index=df.index)
        result = get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.iloc[0]["barrier"] == "upper"
        assert result.iloc[0]["label"] == 1
        # Exit price = barrier (101), not bar close (99.5)
        assert result.iloc[0]["ret"] == pytest.approx(log(101.0 / 100.0), rel=1e-6)
        # Sanity: close on hit bar is BELOW entry; old behavior would have flipped sign.
        assert df["close"].iloc[5] < 100.0

    def test_same_bar_default_worst_case(self):
        # Bar 5 has high>=upper AND low<=lower. WORST_CASE default => label = -1.
        n = 20
        close = np.full(n, 100.0)
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"open": close, "high": close.copy(), "low": close.copy(), "close": close}, index=idx)
        df.loc[df.index[5], "high"] = 102.0  # touches upper=101 (vol=0.01, pt=1)
        df.loc[df.index[5], "low"] = 98.5    # touches lower=99
        events = pd.Series([df.index[15]], index=[df.index[0]])
        vol = pd.Series(0.01, index=df.index)
        result = get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.iloc[0]["barrier"] == "lower"  # worst-case -> lower wins
        assert result.iloc[0]["label"] == -1

    def test_same_bar_upper_first_policy(self):
        n = 20
        close = np.full(n, 100.0)
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"open": close, "high": close.copy(), "low": close.copy(), "close": close}, index=idx)
        df.loc[df.index[5], "high"] = 102.0
        df.loc[df.index[5], "low"] = 98.5
        events = pd.Series([df.index[15]], index=[df.index[0]])
        vol = pd.Series(0.01, index=df.index)
        result = get_labels(events, df, pt_sl=[1.0, 1.0], volatility=vol,
                            same_bar_policy=BarrierTiePolicy.UPPER_FIRST)
        assert result.iloc[0]["barrier"] == "upper"
        assert result.iloc[0]["label"] == 1

    def test_rejects_price_unit_volatility_via_attrs(self):
        prices = _make_prices(np.linspace(100, 105, 20))
        events = pd.Series([prices.index[10]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        vol.attrs["vol_kind"] = "price"
        with pytest.raises(ValueError, match="vol_kind="):
            get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)

    def test_rejects_annualized_volatility_via_attrs(self):
        prices = _make_prices(np.linspace(100, 105, 20))
        events = pd.Series([prices.index[10]], index=[prices.index[0]])
        vol = pd.Series(0.01, index=prices.index)
        vol.attrs["vol_kind"] = "annualized"
        with pytest.raises(ValueError, match="vol_kind="):
            get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)

    def test_rejects_sl_vol_geq_one(self):
        prices = _make_prices(np.linspace(100, 105, 20))
        events = pd.Series([prices.index[10]], index=[prices.index[0]])
        vol = pd.Series(0.5, index=prices.index)  # sl*vol = 1.0
        # Per-event check inside _process_block warns + skips; output is empty
        result = get_labels(events, prices, pt_sl=[1.0, 2.0], volatility=vol)
        # sl=2, vol=0.5 -> sl*vol=1.0 -> non-positive lower; event skipped
        assert len(result) == 0

    def test_object_dtype_events_handled(self):
        # Series with python datetime values (object dtype) — must still work via cast.
        from datetime import datetime as _dt
        prices = _make_prices(np.linspace(100, 105, 20))
        t0 = prices.index[0].to_pydatetime()
        t1 = prices.index[10].to_pydatetime()
        events = pd.Series([t1], index=[t0])
        events = events.astype(object)
        vol = pd.Series(0.01, index=prices.index)
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)
        assert len(result) == 1

    def test_empty_events(self):
        prices = _make_prices(np.full(5, 100.0))
        vol = pd.Series(0.01, index=prices.index)
        empty = pd.Series([], dtype="datetime64[ns]")
        result = get_labels(empty, prices, pt_sl=[1.0, 1.0], volatility=vol)
        assert result.empty
        assert list(result.columns) == ["t1", "ret", "label", "barrier"]

    def test_vol_alignment_ffill(self):
        idx = pd.date_range("2026-01-01", periods=24, freq="1h")
        prices = _make_prices(np.linspace(100, 105, 24), index=idx)
        # vol indexed at coarser frequency
        vol_idx = idx[::6]
        vol = pd.Series([0.01] * len(vol_idx), index=vol_idx)
        events = pd.Series([idx[20]], index=[idx[3]])
        result = get_labels(events, prices, pt_sl=[1.0, 1.0], volatility=vol)
        assert len(result) == 1


class TestApplyMinReturnFilter:
    def test_zeros_below_threshold(self):
        df = pd.DataFrame({
            "t1": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "ret": [0.001, 0.05, -0.1],
            "label": [1, 1, -1],
            "barrier": ["upper", "upper", "lower"],
        })
        out = apply_min_return_filter(df, min_ret=0.01)
        assert list(out["label"]) == [0, 1, -1]
        assert list(out["barrier"]) == ["upper", "upper", "lower"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({
            "t1": pd.to_datetime(["2026-01-01"]),
            "ret": [0.001],
            "label": [1],
            "barrier": ["upper"],
        })
        _ = apply_min_return_filter(df, min_ret=0.01)
        assert df["label"].iloc[0] == 1
