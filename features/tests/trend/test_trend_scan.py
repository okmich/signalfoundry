import datetime

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import trend_scan_features
from okmich_quant_features.trend.trend_scan import _hysteresis_kernel

SCAN_COLUMNS = ["ts_direction", "ts_strength", "ts_r2", "ts_window", "ts_t_value", "ts_slope", "ts_agreement",
                "ts_line_gap"]


def _random_walk(n: int = 300, seed: int = 7, freq: str = "5min") -> pd.Series:
    rng = np.random.default_rng(seed)
    close = 1.30 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    return pd.Series(close, index=pd.date_range("2026-01-05", periods=n, freq=freq), name="close")


def _reference_row(y: np.ndarray, t: int, min_window: int, max_window: int, first: int = 0) -> dict:
    """Independent OLS (lstsq + covariance matrix) over every trailing window ending at t."""
    best, signs = None, []
    for length in range(min_window, max_window + 1):
        lo = t - length + 1
        if lo < first:
            break
        w = y[lo:t + 1]
        design = np.column_stack([np.ones(length), np.arange(length, dtype=float)])
        coef, *_ = np.linalg.lstsq(design, w, rcond=None)
        resid = w - design @ coef
        s2 = resid @ resid / (length - 2)
        t_value = coef[1] / np.sqrt(s2 * np.linalg.inv(design.T @ design)[1, 1])
        r2 = 1 - resid @ resid / np.sum((w - w.mean()) ** 2)
        signs.append(np.sign(coef[1]))
        if best is None or abs(t_value) > abs(best["ts_t_value"]):
            best = {"ts_t_value": t_value, "ts_window": length, "ts_r2": r2, "ts_slope": coef[1],
                    "ts_line_gap": resid[-1] / np.sqrt(s2)}
    best["ts_direction"] = np.sign(best["ts_slope"])
    best["ts_strength"] = best["ts_direction"] * best["ts_r2"]
    best["ts_agreement"] = np.mean(np.array(signs) == best["ts_direction"])
    return best


class TestAgainstReference:
    @pytest.mark.parametrize("t", [0, 4, 5, 9, 40, 71, 150, 299])
    def test_every_column_matches_independent_ols(self, t):
        prices = _random_walk()
        result = trend_scan_features(prices, min_window=6, max_window=48)
        row = result.iloc[t]
        if t < 5:                                                      # warmup: fewer than min_window bars
            assert row.isna().all()
            return
        expected = _reference_row(np.log(prices.to_numpy()), t, 6, 48)
        for column, value in expected.items():
            assert row[column] == pytest.approx(value, rel=1e-7, abs=1e-10), column

    def test_raw_prices_when_log_prices_is_false(self):
        prices = _random_walk()
        result = trend_scan_features(prices, min_window=6, max_window=36, log_prices=False)
        expected = _reference_row(prices.to_numpy(), 200, 6, 36)
        assert result["ts_t_value"].iloc[200] == pytest.approx(expected["ts_t_value"], rel=1e-7)
        assert result["ts_window"].iloc[200] == expected["ts_window"]


class TestCausality:
    @pytest.mark.parametrize("cut", [7, 50, 137, 250])
    def test_appending_future_bars_never_changes_a_past_row(self, cut):
        prices = _random_walk()
        full = trend_scan_features(prices, max_window=48, state_enter=0.7, state_exit=0.35)
        prefix = trend_scan_features(prices.iloc[:cut], max_window=48, state_enter=0.7, state_exit=0.35)
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut])

    def test_live_value_equals_last_row_of_trailing_slice(self):
        prices = _random_walk()
        full = trend_scan_features(prices, max_window=48)
        tail = trend_scan_features(prices.iloc[-48:], max_window=48)
        pd.testing.assert_series_equal(tail.iloc[-1], full.iloc[-1])


class TestBreaks:
    def test_no_window_spans_a_break(self):
        prices = _random_walk()
        index = prices.index.to_numpy().copy()
        index[100:] = index[100:] + np.timedelta64(2, "D")          # a weekend between bars 99 and 100
        prices.index = pd.DatetimeIndex(index)
        result = trend_scan_features(prices, min_window=6, max_window=48, break_gap=pd.Timedelta(minutes=30))
        assert result.iloc[100:105].isna().all().all()                  # warmup restarts after the break
        bars_since_break = np.arange(len(prices)) - 100 + 1
        after = slice(105, 160)
        assert (result["ts_window"].iloc[after].to_numpy() <= bars_since_break[after]).all()
        expected = _reference_row(np.log(prices.to_numpy()), 120, 6, 48, first=100)
        assert result["ts_t_value"].iloc[120] == pytest.approx(expected["ts_t_value"], rel=1e-7)

    def test_without_break_gap_windows_cross_the_gap(self):
        prices = _random_walk()
        prices.index = prices.index.where(np.arange(len(prices)) < 100, prices.index + pd.Timedelta(days=2))
        result = trend_scan_features(prices, min_window=6, max_window=48)
        assert result.iloc[100:105].notna().all().all()

    @pytest.mark.parametrize("gap", [pd.Timedelta(minutes=30), "30min", datetime.timedelta(minutes=30),
                                     np.timedelta64(30, "m")])
    def test_break_gap_accepts_any_duration_type(self, gap):
        prices = _random_walk()
        prices.index = prices.index.where(np.arange(len(prices)) < 100, prices.index + pd.Timedelta(days=2))
        result = trend_scan_features(prices, max_window=48, break_gap=gap)
        assert result.iloc[100:105].isna().all().all() and result.iloc[105:].notna().all().all()

    @pytest.mark.parametrize("gap", [30, 1800, 30.0, "30", " 30 ", True, np.int64(30)])
    def test_a_break_gap_without_a_unit_is_refused(self, gap):
        with pytest.raises(ValueError, match="unit"):
            trend_scan_features(_random_walk(), break_gap=gap)  # pd.Timedelta(30) would mean 30 nanoseconds

    def test_a_break_gap_shorter_than_the_bar_spacing_is_refused(self):
        with pytest.raises(ValueError, match="spacing"):
            trend_scan_features(_random_walk(), break_gap="1min")  # 5-minute bars: every bar would be a break

    def test_break_gap_needs_a_time_sorted_index(self):
        with pytest.raises(ValueError, match="sorted"):
            trend_scan_features(_random_walk().iloc[::-1], break_gap="30min")

    def test_segment_labels_split_windows_like_a_break(self):
        prices = _random_walk()
        gapped = prices.copy()
        gapped.index = prices.index.where(np.arange(len(prices)) < 100, prices.index + pd.Timedelta(days=2))
        labels = np.where(np.arange(len(prices)) < 100, 0, 1)  # e.g. a rollover that leaves no time gap
        by_label = trend_scan_features(prices.to_numpy(), max_window=48, segment=labels)
        by_gap = trend_scan_features(gapped, max_window=48, break_gap="30min")
        np.testing.assert_array_equal(by_label.to_numpy(), by_gap.to_numpy())
        assert by_label.iloc[100:105].isna().all().all()

    def test_segment_and_break_gap_combine(self):
        prices = _random_walk()
        prices.index = prices.index.where(np.arange(len(prices)) < 100, prices.index + pd.Timedelta(days=2))
        labels = np.where(np.arange(len(prices)) < 200, "a", "b")
        result = trend_scan_features(prices, max_window=48, break_gap="30min", segment=labels)
        assert result.iloc[100:105].isna().all().all() and result.iloc[200:205].isna().all().all()
        assert result.iloc[105:200].notna().all().all()

    @pytest.mark.parametrize("labels", [np.zeros(5), np.r_[np.zeros(299), np.nan], np.zeros((300, 2))])
    def test_bad_segment_labels_raise(self, labels):
        with pytest.raises(ValueError, match="segment"):
            trend_scan_features(_random_walk(), segment=labels)

    def test_break_gap_works_on_microsecond_index(self):
        prices = _random_walk()
        prices.index = prices.index.as_unit("us")
        result = trend_scan_features(prices, max_window=24, break_gap=pd.Timedelta(minutes=30))
        assert result.iloc[5:].notna().all().all()                      # regular 5-min bars: no false breaks


class TestBehaviour:
    def test_clean_uptrend_and_downtrend(self):
        n = 200
        noise = np.random.default_rng(3).normal(0, 1e-4, n)
        up = pd.Series(np.exp(np.linspace(0, 0.05, n) + noise))
        down = pd.Series(np.exp(np.linspace(0, -0.05, n) + noise))
        r_up = trend_scan_features(up, max_window=48).iloc[60:]
        r_down = trend_scan_features(down, max_window=48).iloc[60:]
        assert (r_up["ts_direction"] == 1).all() and (r_down["ts_direction"] == -1).all()
        assert (r_up["ts_r2"] > 0.9).all() and (r_up["ts_agreement"] == 1.0).all()
        assert (r_up["ts_slope"] > 0).all() and (r_down["ts_strength"] < -0.9).all()

    def test_line_gap_sign_follows_the_last_bar(self):
        base = np.exp(np.linspace(0, 0.02, 60) + np.random.default_rng(1).normal(0, 2e-4, 60))
        spike_up, spike_down = base.copy(), base.copy()
        spike_up[-1] *= 1.004
        spike_down[-1] *= 0.996
        assert trend_scan_features(spike_up, max_window=24)["ts_line_gap"].iloc[-1] > 0
        assert trend_scan_features(spike_down, max_window=24)["ts_line_gap"].iloc[-1] < 0

    def test_flat_prices_are_neutral_not_nan(self):
        result = trend_scan_features(np.full(50, 1.25), max_window=24).iloc[5:]
        assert (result["ts_direction"] == 0).all() and (result["ts_r2"] == 0).all()
        assert (result["ts_t_value"] == 0).all() and (result["ts_line_gap"] == 0).all()
        assert (result["ts_agreement"] == 1.0).all()

    def test_a_perfectly_straight_window_has_an_undefined_t(self):
        result = trend_scan_features(np.arange(38_000, 38_300, 5.0), log_prices=False, max_window=24).iloc[5:]
        assert (result["ts_direction"] == 1).all() and (result["ts_r2"] == 1).all()
        assert result["ts_t_value"].isna().all()  # an exact equal-step ladder: t is undefined, never +-inf
        assert not np.isinf(result.to_numpy(dtype=float)).any()

    def test_near_straight_windows_keep_a_finite_t(self):
        result = trend_scan_features(np.exp(np.linspace(0, 0.1, 40)), max_window=24).iloc[5:]
        assert (result["ts_direction"] == 1).all()
        t_value = result["ts_t_value"]
        assert (t_value.isna() | (t_value.abs() > 1e6)).all() and not np.isinf(t_value).any()

    def test_ndarray_and_series_give_the_same_values(self):
        prices = _random_walk()
        from_series = trend_scan_features(prices, max_window=36)
        from_array = trend_scan_features(prices.to_numpy(), max_window=36)
        assert isinstance(from_array.index, pd.RangeIndex)
        np.testing.assert_array_equal(from_array.to_numpy(), from_series.to_numpy())

    def test_strength_never_leaves_unit_interval_on_exact_lines(self):
        for prices in (np.exp(np.linspace(0, 1, 120)), np.exp(np.linspace(0, -1, 120)), np.linspace(1, 2, 120)):
            result = trend_scan_features(prices, max_window=72).iloc[5:]
            assert result["ts_r2"].between(0, 1).all() and result["ts_strength"].abs().le(1).all()

    def test_empty_input(self):
        result = trend_scan_features(pd.Series([], dtype=float), state_enter=0.5)
        assert result.empty and list(result.columns) == SCAN_COLUMNS + ["ts_state"]

    def test_random_walk_integrity(self):
        result = trend_scan_features(_random_walk(n=2000, seed=11), max_window=72)
        assert list(result.columns) == SCAN_COLUMNS
        assert result.iloc[:5].isna().all().all() and result.iloc[5:].notna().all().all()
        assert not np.isinf(result.to_numpy(dtype=float)[5:]).any()
        assert result["ts_r2"].iloc[5:].between(0, 1).all()
        assert result["ts_agreement"].iloc[5:].between(0, 1).all()
        assert result["ts_window"].iloc[5:].between(6, 72).all()


class TestState:
    def test_state_column_only_when_enabled(self):
        assert "ts_state" not in trend_scan_features(_random_walk(), max_window=24).columns

    def test_state_follows_a_trend_reversal(self):
        noise = np.random.default_rng(5).normal(0, 1e-4, 300)
        path = np.r_[np.linspace(0, 0.03, 150), np.linspace(0.03, 0.0, 150)] + noise
        result = trend_scan_features(np.exp(path), max_window=36, state_enter=0.8, state_exit=0.4)
        state = result["ts_state"]
        assert state.iloc[:5].isna().all()
        assert set(state.dropna().unique()) <= {-1.0, 0.0, 1.0}
        assert (state.iloc[60:150] == 1).all()
        assert (state.iloc[210:] == -1).all()

    def test_hysteresis_holds_between_exit_and_enter(self):
        result = trend_scan_features(_random_walk(n=3000, seed=2), max_window=48, state_enter=0.8, state_exit=0.4)
        strength, state = result["ts_strength"].to_numpy(), result["ts_state"].to_numpy()
        held_up = (state[:-1] == 1) & (strength[1:] >= 0.4) & (strength[1:] > -0.8)
        assert (state[1:][held_up] == 1).all()
        entered_up = (state[1:] == 1) & (state[:-1] != 1)
        assert (strength[1:][entered_up] >= 0.8).all()

    def test_hysteresis_down_side_mirrors_the_up_side(self):
        state = _hysteresis_kernel(np.array([-0.9, -0.5, -0.31, -0.2, -0.85, 0.85]), 0.8, 0.3)
        np.testing.assert_array_equal(state, [-1, -1, -1, 0, -1, 1])  # holds to -exit, leaves above it, flips direct

    def test_state_restarts_from_neutral_after_a_gap(self):
        # UP before the gap; afterwards 0.5 sits between exit (0.3) and enter (0.8): a carried-over UP would stay UP
        state = _hysteresis_kernel(np.array([0.9, 0.5, np.nan, 0.5, 0.85]), 0.8, 0.3)
        np.testing.assert_array_equal(state, [1, 1, np.nan, 0, 1])

    def test_state_resets_after_a_break(self):
        prices = _random_walk(n=400)
        prices.index = prices.index.where(np.arange(len(prices)) < 200, prices.index + pd.Timedelta(days=2))
        result = trend_scan_features(prices, max_window=36, break_gap=pd.Timedelta(minutes=30), state_enter=0.6)
        assert result["ts_state"].iloc[200:205].isna().all()


class TestValidation:
    @pytest.mark.parametrize("kwargs", [{"min_window": 2}, {"min_window": 10, "max_window": 9},
                                        {"state_enter": 0.0}, {"state_enter": 1.2},
                                        {"state_enter": 0.5, "state_exit": 0.6},
                                        {"state_enter": 0.5, "state_exit": -0.1}, {"break_gap": pd.Timedelta(0)},
                                        {"break_gap": "soon"}, {"break_gap": pd.NaT}, {"min_window": 6.5},
                                        {"max_window": 48.5}, {"min_window": True}, {"min_window": 4},
                                        {"min_window": "6"}, {"min_window": None}, {"max_window": 1e20},
                                        {"max_window": float("nan")}, {"state_exit": 0.4},
                                        {"state_enter": "0.5"}, {"state_enter": 0.5, "state_exit": None}])
    def test_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            trend_scan_features(_random_walk(), **kwargs)

    def test_non_finite_prices(self):
        prices = _random_walk()
        prices.iloc[10] = np.nan
        with pytest.raises(ValueError):
            trend_scan_features(prices)

    def test_non_positive_prices_need_log_prices_false(self):
        values = np.linspace(-1.0, 1.0, 50)
        with pytest.raises(ValueError):
            trend_scan_features(values)
        assert trend_scan_features(values, log_prices=False, max_window=24)["ts_direction"].iloc[-1] == 1

    def test_two_dimensional_input_is_rejected(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            trend_scan_features(_random_walk().to_frame())

    def test_whole_number_floats_are_accepted(self):
        prices = _random_walk()
        pd.testing.assert_frame_equal(trend_scan_features(prices, min_window=6.0, max_window=48.0),
                                      trend_scan_features(prices, min_window=6, max_window=48))

    def test_break_gap_needs_a_datetime_index(self):
        with pytest.raises(ValueError):
            trend_scan_features(_random_walk().to_numpy(), break_gap=pd.Timedelta(minutes=30))
