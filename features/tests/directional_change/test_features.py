"""
Tests for log_r, dc_live_features, and normalise_minmax.

Hand-calculated live features for [100, 112, 100, 90, 100, 115, 103], theta=0.1, alpha=1.0:
  i=0: NaN (no DC confirmed yet)
  i=1: tmv=1.2,           t=1, dir=+1,  upward_dcc=F, downward_dcc=F  (uptrend from 100@0; init→mode1, not a confirmed DCC)
  i=2: tmv=12/11.2≈1.071, t=1, dir=-1,  upward_dcc=F, downward_dcc=T  (downward DC confirmed; mode1→-1)
  i=3: tmv=22/11.2≈1.964, t=2, dir=-1,  upward_dcc=F, downward_dcc=F
  i=4: tmv=10/9≈1.111,    t=1, dir=+1,  upward_dcc=T, downward_dcc=F  (upward DC confirmed; mode-1→1)
  i=5: tmv=25/9≈2.778,    t=2, dir=+1,  upward_dcc=F, downward_dcc=F
  i=6: tmv=12/11.5≈1.043, t=1, dir=-1,  upward_dcc=F, downward_dcc=T  (downward DC confirmed; mode1→-1)
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import (
    dc_live_features,
    log_r,
    normalise_minmax,
    parse_dc_events,
)

THETA = 0.1
REF_PRICES = [100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0]


@pytest.fixture
def ref_series():
    return pd.Series(REF_PRICES)


@pytest.fixture
def ref_trends(ref_series):
    return parse_dc_events(ref_series, THETA)


@pytest.fixture
def ref_live(ref_series):
    return dc_live_features(ref_series, THETA)


class TestLogR:
    def test_log_r_matches_log_of_r(self, ref_trends):
        expected = np.log(ref_trends["r"])
        result = log_r(ref_trends)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_log_r_is_finite_for_positive_r(self, ref_trends):
        result = log_r(ref_trends)
        assert result.notna().all()
        assert np.isfinite(result).all()

    def test_log_r_length_matches_trends(self, ref_trends):
        assert len(log_r(ref_trends)) == len(ref_trends)

    def test_log_r_returns_nan_for_zero_r(self):
        trends = pd.DataFrame({"r": [0.1, 0.0, 0.05]})
        result = log_r(trends)
        assert np.isnan(result.iloc[1])

    def test_log_r_returns_nan_for_negative_r(self):
        trends = pd.DataFrame({"r": [0.1, -0.01, 0.05]})
        result = log_r(trends)
        assert np.isnan(result.iloc[1])

    def test_log_r_does_not_distort_non_positive(self):
        # Ensure the old clip behaviour is gone — non-positive must become NaN, not ~-27
        trends = pd.DataFrame({"r": [0.0]})
        result = log_r(trends)
        assert np.isnan(result.iloc[0])

    def test_log_r_preserves_positive_values(self):
        trends = pd.DataFrame({"r": [0.1, np.nan, 0.05]})
        result = log_r(trends)
        assert result.iloc[0] == pytest.approx(np.log(0.1))
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(np.log(0.05))


class TestDcLiveFeatures:
    def test_output_index_matches_input(self, ref_series, ref_live):
        assert ref_live.index.equals(ref_series.index)

    def test_columns_present(self, ref_live):
        assert set(ref_live.columns) == {"tmv_current", "t_current", "direction", "upward_dcc", "downward_dcc", "rdc_current"}

    def test_nan_before_first_dc(self, ref_live):
        # Bar 0: no DC confirmed yet → all NaN
        assert np.isnan(ref_live["tmv_current"].iloc[0])
        assert np.isnan(ref_live["t_current"].iloc[0])
        assert np.isnan(ref_live["direction"].iloc[0])

    def test_tmv_current_values(self, ref_live):
        expected = [np.nan, 1.2, 12 / 11.2, 22 / 11.2, 10 / 9, 25 / 9, 12 / 11.5]
        for i, exp in enumerate(expected):
            val = ref_live["tmv_current"].iloc[i]
            if np.isnan(exp):
                assert np.isnan(val), f"Bar {i}: expected NaN, got {val}"
            else:
                assert val == pytest.approx(exp, rel=1e-6), f"Bar {i}: expected {exp}, got {val}"

    def test_t_current_values(self, ref_live):
        expected = [np.nan, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0]
        for i, exp in enumerate(expected):
            val = ref_live["t_current"].iloc[i]
            if np.isnan(exp):
                assert np.isnan(val)
            else:
                assert val == pytest.approx(exp)

    def test_direction_values(self, ref_live):
        expected = [np.nan, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        for i, exp in enumerate(expected):
            val = ref_live["direction"].iloc[i]
            if np.isnan(exp):
                assert np.isnan(val)
            else:
                assert val == pytest.approx(exp)

    def test_direction_only_plus_minus_one(self, ref_live):
        valid = ref_live["direction"].dropna()
        assert set(valid.unique()).issubset({1.0, -1.0})

    def test_tmv_current_always_positive(self, ref_live):
        assert (ref_live["tmv_current"].dropna() > 0).all()

    def test_t_current_always_positive(self, ref_live):
        assert (ref_live["t_current"].dropna() > 0).all()

    def test_consistency_with_parser_at_ext_end(self, ref_series, ref_trends):
        """
        At the bar just before a DC confirmation (the last bar of a completed trend's
        OS phase), live TMV should equal the parser's TMV for that trend.
        """
        live = dc_live_features(ref_series, THETA)
        # Trend 0 (up, ext_end@1): last bar in OS is bar 1 (before DC at bar 2)
        assert live["tmv_current"].iloc[1] == pytest.approx(ref_trends["tmv"].iloc[0])
        # Trend 1 (down, ext_end@3): last bar in OS is bar 3 (before DC at bar 4)
        assert live["tmv_current"].iloc[3] == pytest.approx(ref_trends["tmv"].iloc[1])
        # Trend 2 (up, ext_end@5): last bar in OS is bar 5 (before DC at bar 6)
        assert live["tmv_current"].iloc[5] == pytest.approx(ref_trends["tmv"].iloc[2])

    def test_upward_dcc_values(self, ref_live):
        # upward_dcc fires at i=4 only (mode -1→1 transition from an established downtrend)
        expected = [False, False, False, False, True, False, False]
        for i, exp in enumerate(expected):
            assert bool(ref_live["upward_dcc"].iloc[i]) == exp, f"Bar {i}: expected upward_dcc={exp}"

    def test_downward_dcc_values(self, ref_live):
        # downward_dcc fires at i=2 and i=6 (mode 1→-1 transitions from established uptrends)
        expected = [False, False, True, False, False, False, True]
        for i, exp in enumerate(expected):
            assert bool(ref_live["downward_dcc"].iloc[i]) == exp, f"Bar {i}: expected downward_dcc={exp}"

    def test_upward_dcc_not_fired_on_init_to_mode1(self, ref_live):
        # i=1: mode transitions from init(0) to 1 — should NOT fire upward_dcc
        assert not ref_live["upward_dcc"].iloc[1]

    def test_dcc_flags_are_mutually_exclusive_per_bar(self, ref_live):
        # No bar should have both upward_dcc and downward_dcc True simultaneously
        both = ref_live["upward_dcc"] & ref_live["downward_dcc"]
        assert not both.any()

    def test_alpha_asymmetric_fires_downward_dcc_sooner(self):
        # With alpha=0.5, downward DCC fires when price falls 5% from high.
        # [100, 115, 108]: 108 <= 115*0.95=109.25 → downward_dcc fires at i=2 with alpha=0.5
        # With alpha=1.0: 108 <= 115*0.9=103.5 → does NOT fire.
        prices = pd.Series([100.0, 115.0, 108.0])
        live_sym = dc_live_features(prices, theta=0.1, alpha=1.0)
        live_asym = dc_live_features(prices, theta=0.1, alpha=0.5)
        assert not live_sym["downward_dcc"].any()
        assert live_asym["downward_dcc"].iloc[2]

    def test_alpha_does_not_affect_upward_dcc(self):
        # [100, 90, 100]: upward DC fires at i=2 for all alpha values
        prices = pd.Series([100.0, 90.0, 100.0])
        live_sym = dc_live_features(prices, theta=0.1, alpha=1.0)
        live_asym = dc_live_features(prices, theta=0.1, alpha=0.5)
        assert live_sym["upward_dcc"].iloc[2]
        assert live_asym["upward_dcc"].iloc[2]

    def test_empty_series_returns_empty_frame(self):
        live = dc_live_features(pd.Series([], dtype=float), THETA)
        assert live.empty

    def test_single_bar_returns_nan_frame(self):
        live = dc_live_features(pd.Series([100.0]), THETA)
        assert len(live) == 1
        assert np.isnan(live["tmv_current"].iloc[0])
        assert np.isnan(live["direction"].iloc[0])
        assert not live["upward_dcc"].iloc[0]

    def test_invalid_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            dc_live_features(pd.Series(REF_PRICES), theta=0.0)
        with pytest.raises(ValueError, match="theta"):
            dc_live_features(pd.Series(REF_PRICES), theta=-0.01)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            dc_live_features(pd.Series(REF_PRICES), theta=THETA, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            dc_live_features(pd.Series(REF_PRICES), theta=THETA, alpha=1.1)

    def test_nan_price_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            dc_live_features(pd.Series([100.0, np.nan, 112.0]), THETA)

    def test_inf_price_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            dc_live_features(pd.Series([100.0, np.inf, 112.0]), THETA)

    def test_flat_series_all_nan(self):
        series = pd.Series([100.0] * 20)
        live = dc_live_features(series, THETA)
        assert live["tmv_current"].isna().all()

    def test_datetime_index_preserved(self):
        dates = pd.date_range("2024-01-01", periods=len(REF_PRICES), freq="h")
        series = pd.Series(REF_PRICES, index=dates)
        live = dc_live_features(series, THETA)
        assert live.index.equals(dates)


class TestRdcCurrent:
    """
    rdc_current = tmv_current * theta / t_current = |p - last_ext| / last_ext / t

    Hand-calculated for [100, 112, 100, 90, 100, 115, 103], theta=0.1:
      i=0: NaN  (init phase)
      i=1: 1.2   * 0.1 / 1  = 0.12
      i=2: (12/11.2) * 0.1 / 1  = 12/112  ≈ 0.10714
      i=3: (22/11.2) * 0.1 / 2  = 22/224  ≈ 0.09821
      i=4: (10/9)   * 0.1 / 1  = 10/90   ≈ 0.11111
      i=5: (25/9)   * 0.1 / 2  = 25/180  ≈ 0.13889
      i=6: (12/11.5)* 0.1 / 1  = 12/115  ≈ 0.10435
    """

    def test_nan_during_init(self, ref_live):
        assert np.isnan(ref_live["rdc_current"].iloc[0])

    def test_rdc_current_values(self, ref_live):
        expected = [np.nan, 0.12, 12 / 112, 22 / 224, 10 / 90, 25 / 180, 12 / 115]
        for i, exp in enumerate(expected):
            val = ref_live["rdc_current"].iloc[i]
            if np.isnan(exp):
                assert np.isnan(val), f"Bar {i}: expected NaN, got {val}"
            else:
                assert val == pytest.approx(exp, rel=1e-6), f"Bar {i}: expected {exp}, got {val}"

    def test_rdc_current_equals_tmv_times_theta_over_t(self, ref_live):
        live = ref_live
        mask = live["t_current"].notna() & (live["t_current"] > 0)
        expected = live.loc[mask, "tmv_current"] * THETA / live.loc[mask, "t_current"]
        pd.testing.assert_series_equal(live.loc[mask, "rdc_current"], expected, check_names=False)

    def test_rdc_current_always_positive_when_not_nan(self, ref_live):
        valid = ref_live["rdc_current"].dropna()
        assert (valid > 0).all()

    def test_rdc_current_nan_during_init_phase(self):
        # Series too short to produce any DC → rdc_current all NaN
        live = dc_live_features(pd.Series([100.0, 105.0]), THETA)
        assert live["rdc_current"].isna().all()

    def test_single_bar_rdc_is_nan(self):
        live = dc_live_features(pd.Series([100.0]), THETA)
        assert np.isnan(live["rdc_current"].iloc[0])

    def test_rdc_current_with_alpha(self):
        # Asymmetric alpha does not change the formula — only when DC fires
        prices = pd.Series(REF_PRICES)
        live_sym = dc_live_features(prices, THETA, alpha=1.0)
        live_asym = dc_live_features(prices, THETA, alpha=0.5)
        # Both should satisfy rdc = tmv * theta / t wherever t > 0
        for live in (live_sym, live_asym):
            mask = live["t_current"].notna() & (live["t_current"] > 0)
            expected = live.loc[mask, "tmv_current"] * THETA / live.loc[mask, "t_current"]
            pd.testing.assert_series_equal(live.loc[mask, "rdc_current"], expected, check_names=False)


class TestNormaliseMinmax:
    def test_nan_in_series_raises(self):
        s = pd.Series([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            normalise_minmax(s)

    def test_inf_in_series_raises(self):
        s = pd.Series([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            normalise_minmax(s)

    def test_neg_inf_in_series_raises(self):
        s = pd.Series([1.0, -np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            normalise_minmax(s)

    def test_training_range_is_zero_to_one(self):
        s = pd.Series([1.2, 2.5, 1.8, 3.1, 1.5])
        normed, mn, mx = normalise_minmax(s)
        assert normed.min() == pytest.approx(0.0)
        assert normed.max() == pytest.approx(1.0)

    def test_returns_correct_min_max(self):
        s = pd.Series([2.0, 4.0, 6.0])
        _, mn, mx = normalise_minmax(s)
        assert mn == pytest.approx(2.0)
        assert mx == pytest.approx(6.0)

    def test_reuse_training_stats_on_test_data(self):
        train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        _, mn, mx = normalise_minmax(train)

        # Test value within training range
        test_in = pd.Series([3.0])
        normed, _, _ = normalise_minmax(test_in, min_val=mn, max_val=mx)
        assert normed.iloc[0] == pytest.approx(0.5)

        # Test value outside training range → extrapolates beyond [0, 1]
        test_out = pd.Series([6.0])
        normed_out, _, _ = normalise_minmax(test_out, min_val=mn, max_val=mx)
        assert normed_out.iloc[0] == pytest.approx(1.25)

    def test_constant_series_returns_zeros(self):
        s = pd.Series([5.0, 5.0, 5.0])
        normed, mn, mx = normalise_minmax(s)
        assert (normed == 0.0).all()
        assert mn == mx

    def test_formula_correctness(self):
        s = pd.Series([2.0, 4.0, 6.0])
        normed, mn, mx = normalise_minmax(s)
        expected = (s - mn) / (mx - mn)
        pd.testing.assert_series_equal(normed, expected)
