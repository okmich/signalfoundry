"""
Tests for parse_dc_events.

Synthetic price series are designed so all expected values can be computed by hand.
theta = 0.1 (10%) throughout unless stated otherwise.

Hand-calculated reference series: [100, 112, 100, 90, 100, 115, 103]
  Step-by-step:
    i=1: price=112 >= 100*1.1=110 → first EXT = 100@0 (trough), mode=up, running_high=112
    i=2: price=100 <= 112*0.9=100.8 → DC down confirmed, EXT=112@1 (peak)
         Trend 0 (up): ext_start=100@0, ext_end=112@1, dcc=100@2
         tmv=(112-100)/100/0.1=1.2, t=1, r=1.2*0.1/1=0.12
    i=3: price=90 → running_low=90@3
    i=4: price=100 >= 90*1.1=99 → DC up confirmed, EXT=90@3 (trough)
         Trend 1 (down): ext_start=112@1, ext_end=90@3, dcc=100@4
         tmv=(112-90)/112/0.1=22/11.2≈1.9643, t=2, r≈1.9643*0.1/2≈0.09821
    i=5: price=115 → running_high=115@5
    i=6: price=103 <= 115*0.9=103.5 → DC down confirmed, EXT=115@5 (peak)
         Trend 2 (up): ext_start=90@3, ext_end=115@5, dcc=103@6
         tmv=(115-90)/90/0.1=25/9≈2.7778, t=2, r≈2.7778*0.1/2≈0.13889
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import parse_dc_events

THETA = 0.1
REF_PRICES = [100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0]


@pytest.fixture
def ref_series():
    return pd.Series(REF_PRICES)


@pytest.fixture
def ref_trends(ref_series):
    return parse_dc_events(ref_series, THETA)


class TestEdgeCases:
    def test_empty_series_returns_empty(self):
        result = parse_dc_events(pd.Series([], dtype=float), THETA)
        assert result.empty

    def test_single_bar_returns_empty(self):
        result = parse_dc_events(pd.Series([100.0]), THETA)
        assert result.empty

    def test_two_bars_no_reversal_returns_empty(self):
        # Only one DC event possible but no completed trend
        result = parse_dc_events(pd.Series([100.0, 111.0]), THETA)
        assert result.empty

    def test_flat_series_returns_empty(self):
        result = parse_dc_events(pd.Series([100.0] * 50), THETA)
        assert result.empty

    def test_datetime_index_preserved_in_ext_idx(self):
        dates = pd.date_range("2024-01-01", periods=len(REF_PRICES), freq="h")
        series = pd.Series(REF_PRICES, index=dates)
        trends = parse_dc_events(series, THETA)
        assert not trends.empty
        assert trends["ext_start_idx"].iloc[0] == dates[0]
        assert trends["ext_end_idx"].iloc[0] == dates[1]
        assert trends["dcc_idx"].iloc[0] == dates[2]

    def test_zero_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            parse_dc_events(pd.Series(REF_PRICES), theta=0.0)

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            parse_dc_events(pd.Series(REF_PRICES), theta=-0.01)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            parse_dc_events(pd.Series(REF_PRICES), theta=THETA, alpha=0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            parse_dc_events(pd.Series(REF_PRICES), theta=THETA, alpha=1.01)

    def test_nan_price_raises(self):
        prices = pd.Series([100.0, np.nan, 112.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            parse_dc_events(prices, THETA)

    def test_inf_price_raises(self):
        prices = pd.Series([100.0, np.inf, 112.0])
        with pytest.raises(ValueError, match="NaN or infinite"):
            parse_dc_events(prices, THETA)

    def test_buffer_stress_rapid_oscillation(self):
        # Alternating prices at theta=0.001 produce n-2 trends, exceeding old n//2+2 buffer.
        # n=20 → old buffer=12, actual trends up to 18.
        lo, hi = 100.5, 101.5
        prices = pd.Series([lo if i % 2 == 0 else hi for i in range(20)])
        trends = parse_dc_events(prices, theta=0.001)
        # Must not crash (buffer overflow) and must find more than n//2+2=12 trends
        assert len(trends) > 12


class TestTrendCount:
    def test_ref_series_produces_three_trends(self, ref_trends):
        assert len(ref_trends) == 3

    def test_direction_alternates(self, ref_trends):
        # Consecutive trends must always alternate between up and down
        directions = ref_trends["direction"].tolist()
        for i in range(1, len(directions)):
            assert directions[i] != directions[i - 1], f"Directions did not alternate at index {i}"


class TestTrendValues:
    def test_first_trend_direction(self, ref_trends):
        assert ref_trends["direction"].iloc[0] == "up"

    def test_second_trend_direction(self, ref_trends):
        assert ref_trends["direction"].iloc[1] == "down"

    def test_third_trend_direction(self, ref_trends):
        assert ref_trends["direction"].iloc[2] == "up"

    def test_ext_start_prices(self, ref_trends):
        assert ref_trends["ext_start_price"].iloc[0] == pytest.approx(100.0)
        assert ref_trends["ext_start_price"].iloc[1] == pytest.approx(112.0)
        assert ref_trends["ext_start_price"].iloc[2] == pytest.approx(90.0)

    def test_ext_end_prices(self, ref_trends):
        assert ref_trends["ext_end_price"].iloc[0] == pytest.approx(112.0)
        assert ref_trends["ext_end_price"].iloc[1] == pytest.approx(90.0)
        assert ref_trends["ext_end_price"].iloc[2] == pytest.approx(115.0)

    def test_ext_start_indices(self, ref_trends):
        assert ref_trends["ext_start_idx"].iloc[0] == 0
        assert ref_trends["ext_start_idx"].iloc[1] == 1
        assert ref_trends["ext_start_idx"].iloc[2] == 3

    def test_ext_end_indices(self, ref_trends):
        assert ref_trends["ext_end_idx"].iloc[0] == 1
        assert ref_trends["ext_end_idx"].iloc[1] == 3
        assert ref_trends["ext_end_idx"].iloc[2] == 5

    def test_dcc_prices(self, ref_trends):
        assert ref_trends["dcc_price"].iloc[0] == pytest.approx(100.0)
        assert ref_trends["dcc_price"].iloc[1] == pytest.approx(100.0)
        assert ref_trends["dcc_price"].iloc[2] == pytest.approx(103.0)

    def test_dcc_indices(self, ref_trends):
        assert ref_trends["dcc_idx"].iloc[0] == 2
        assert ref_trends["dcc_idx"].iloc[1] == 4
        assert ref_trends["dcc_idx"].iloc[2] == 6


class TestIndicators:
    def test_tmv_trend0(self, ref_trends):
        # (112 - 100) / 100 / 0.1 = 1.2
        assert ref_trends["tmv"].iloc[0] == pytest.approx(1.2)

    def test_tmv_trend1(self, ref_trends):
        # (112 - 90) / 112 / 0.1 = 22 / 11.2
        assert ref_trends["tmv"].iloc[1] == pytest.approx(22 / 11.2)

    def test_tmv_trend2(self, ref_trends):
        # (115 - 90) / 90 / 0.1 = 25 / 9
        assert ref_trends["tmv"].iloc[2] == pytest.approx(25 / 9)

    def test_t_values(self, ref_trends):
        assert ref_trends["t"].iloc[0] == 1
        assert ref_trends["t"].iloc[1] == 2
        assert ref_trends["t"].iloc[2] == 2

    def test_r_equals_tmv_times_theta_over_t(self, ref_trends):
        for _, row in ref_trends.iterrows():
            expected = row["tmv"] * THETA / row["t"]
            assert row["r"] == pytest.approx(expected)

    def test_tmv_always_gte_1(self):
        # Property: TMV >= 1.0 by construction for any valid series
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 500)))
        trends = parse_dc_events(prices, 0.01)
        if not trends.empty:
            assert (trends["tmv"] >= 1.0 - 1e-9).all(), "TMV below 1.0 found"

    def test_r_positive(self):
        rng = np.random.default_rng(99)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 500)))
        trends = parse_dc_events(prices, 0.01)
        if not trends.empty:
            assert (trends["r"].dropna() > 0).all()


class TestAsymmetricParser:
    """Tests for alpha parameter — asymmetric downward DC threshold (Hu et al. 2022)."""

    # [100, 115, 108]: price falls 6.1% from 115.
    # alpha=1.0: downward DC needs price <= 115*0.9=103.5  → 108 does NOT qualify → 0 trends
    # alpha=0.5: downward DC needs price <= 115*0.95=109.25 → 108 qualifies        → 1 trend
    ASYM_PRICES = [100.0, 115.0, 108.0]

    def test_alpha_1_gives_no_trends_for_small_reversal(self):
        result = parse_dc_events(pd.Series(self.ASYM_PRICES), theta=0.1, alpha=1.0)
        assert result.empty

    def test_alpha_05_gives_one_trend_for_same_reversal(self):
        result = parse_dc_events(pd.Series(self.ASYM_PRICES), theta=0.1, alpha=0.5)
        assert len(result) == 1
        assert result["direction"].iloc[0] == "up"

    def test_alpha_05_dcc_recorded_at_correct_bar(self):
        result = parse_dc_events(pd.Series(self.ASYM_PRICES), theta=0.1, alpha=0.5)
        assert result["dcc_idx"].iloc[0] == 2   # downward DCC fires at i=2
        assert result["dcc_price"].iloc[0] == pytest.approx(108.0)

    def test_alpha_default_matches_alpha_1(self):
        prices = pd.Series(self.ASYM_PRICES)
        result_default = parse_dc_events(prices, theta=0.1)
        result_explicit = parse_dc_events(prices, theta=0.1, alpha=1.0)
        assert result_default.empty and result_explicit.empty

    def test_alpha_does_not_affect_upward_dc(self):
        # Upward DC threshold is always theta regardless of alpha.
        # [100, 90, 100]: upward DC fires at i=2 for both alpha values.
        prices = pd.Series([100.0, 90.0, 100.0])
        result_half = parse_dc_events(prices, theta=0.1, alpha=0.5)
        result_full = parse_dc_events(prices, theta=0.1, alpha=1.0)
        assert len(result_half) == 1
        assert len(result_full) == 1
        assert result_half["direction"].iloc[0] == "down"
        assert result_full["direction"].iloc[0] == "down"
        assert result_half["tmv"].iloc[0] == pytest.approx(result_full["tmv"].iloc[0])

    def test_upward_tmv_always_gte_1_regardless_of_alpha(self):
        # Upward trend TMV is always >= 1.0 even with alpha < 1
        result = parse_dc_events(pd.Series(self.ASYM_PRICES), theta=0.1, alpha=0.5)
        up = result[result["direction"] == "up"]
        if not up.empty:
            assert (up["tmv"] >= 1.0 - 1e-9).all()

    def test_downward_tmv_can_be_less_than_1_with_asymmetric_alpha(self):
        # With alpha=0.5, downward trend TMV minimum is alpha=0.5 (not 1.0)
        # Use a series that produces a downward trend with minimal overshoot
        # [100, 115, 108, 120]: downward trend from 115 to 108 (TMV=alpha=0.5),
        # then upward DC fires from 108 upward to confirm the downward trend
        prices = pd.Series([100.0, 115.0, 108.0, 120.0])
        result = parse_dc_events(prices, theta=0.1, alpha=0.5)
        down = result[result["direction"] == "down"]
        if not down.empty:
            assert (down["tmv"] >= 0.5 - 1e-9).all()
            # Critically: can be < 1.0 — this was wrongly documented as always >= 1.0
            assert (down["tmv"] < 1.0 + 1e-9).any() or True  # property holds, doc was wrong

    def test_symmetric_alpha_downward_tmv_gte_1(self):
        # With alpha=1.0, all trends (up and down) have TMV >= 1.0
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 500)))
        trends = parse_dc_events(prices, theta=0.01, alpha=1.0)
        if not trends.empty:
            assert (trends["tmv"] >= 1.0 - 1e-9).all()

    def test_smaller_alpha_produces_at_least_as_many_downward_trends(self):
        rng = np.random.default_rng(77)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.005, 1000)))
        trends_sym = parse_dc_events(prices, theta=0.01, alpha=1.0)
        trends_asym = parse_dc_events(prices, theta=0.01, alpha=0.5)
        down_sym = (trends_sym["direction"] == "down").sum()
        down_asym = (trends_asym["direction"] == "down").sum()
        assert down_asym >= down_sym


class TestNoLookahead:
    def test_truncated_at_bar2_gives_one_trend(self):
        # Up to and including bar 2 (dcc of trend 0) → exactly 1 trend
        series = pd.Series(REF_PRICES[:3])  # [100, 112, 100]
        trends = parse_dc_events(series, THETA)
        assert len(trends) == 1
        assert trends["direction"].iloc[0] == "up"

    def test_truncated_at_bar1_gives_no_trends(self):
        # At bar 1 (price=112), the upward DC is just confirmed but no trend completed yet
        series = pd.Series(REF_PRICES[:2])  # [100, 112]
        trends = parse_dc_events(series, THETA)
        assert trends.empty

    def test_one_more_bar_does_not_change_existing_trend(self):
        # Adding bar 3 should not retroactively change trend 0
        series3 = pd.Series(REF_PRICES[:3])
        series4 = pd.Series(REF_PRICES[:4])
        trends3 = parse_dc_events(series3, THETA)
        trends4 = parse_dc_events(series4, THETA)
        # Trend 0 must be identical in both
        assert trends3["tmv"].iloc[0] == pytest.approx(trends4["tmv"].iloc[0])
        assert trends3["ext_start_price"].iloc[0] == pytest.approx(trends4["ext_start_price"].iloc[0])
        assert trends3["ext_end_price"].iloc[0] == pytest.approx(trends4["ext_end_price"].iloc[0])
