"""
Tests for idc_parse.

Hand-calculated reference series: [100, 112, 100, 90, 100, 115, 103]
theta=0.1, alpha=1.0

Bar-by-bar trace:
  i=0: init, ph=100, pl=100, t_dc0=0
  i=1: p=112 >= 100*1.1=110 -> first upward DC from init
       upturn_dc=T, mode=1, ph=112, pl=100(init_low), t_dc0=1, rdc=NaN (init DC)
  i=2: p=100 <= 112*0.9=100.8 -> downward DC confirmed
       downturn_dc=T, mode=-1, ph=112, pl=100, t_dc0=2
       rdc = |100-112|/112/(2-1) = 12/112 ~ 0.10714
  i=3: p=90 < pl=100 -> new low
       new_low=T, mode=-1, ph=112, pl=90, t_dc0=3
  i=4: p=100 >= 90*1.1=99 -> upward DC confirmed
       upturn_dc=T, mode=1, ph=100, pl=90, t_dc0=4
       rdc = |100-90|/90/(4-3) = 10/90 ~ 0.11111
  i=5: p=115 > ph=100 -> new high
       new_high=T, mode=1, ph=115, pl=90, t_dc0=5
  i=6: p=103 <= 115*0.9=103.5 -> downward DC confirmed
       downturn_dc=T, mode=-1, ph=115, pl=103, t_dc0=6
       rdc = |103-115|/115/(6-5) = 12/115 ~ 0.10435
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import idc_parse, parse_dc_events

THETA = 0.1
REF_PRICES = [100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0]


@pytest.fixture
def ref_series():
    return pd.Series(REF_PRICES)


@pytest.fixture
def ref_idc(ref_series):
    return idc_parse(ref_series, THETA)


class TestEdgeCases:
    def test_empty_series_returns_empty(self):
        result = idc_parse(pd.Series([], dtype=float), THETA)
        assert result.empty

    def test_single_bar_returns_single_nan_row(self):
        result = idc_parse(pd.Series([100.0]), THETA)
        assert len(result) == 1
        assert np.isnan(result["direction"].iloc[0])
        assert np.isnan(result["rdc"].iloc[0])

    def test_flat_series_direction_all_nan(self):
        result = idc_parse(pd.Series([100.0] * 20), THETA)
        assert result["direction"].isna().all()

    def test_flat_series_no_dc_events(self):
        result = idc_parse(pd.Series([100.0] * 20), THETA)
        assert not result["upturn_dc"].any()
        assert not result["downturn_dc"].any()

    def test_two_bars_no_reversal_direction_all_nan(self):
        result = idc_parse(pd.Series([100.0, 101.0]), THETA)
        assert result["direction"].isna().all()

    def test_output_index_matches_input(self, ref_series, ref_idc):
        assert ref_idc.index.equals(ref_series.index)

    def test_output_length_matches_input(self, ref_series, ref_idc):
        assert len(ref_idc) == len(ref_series)

    def test_columns_present(self, ref_idc):
        expected = {"direction", "ph", "pl", "t_dc0", "upturn_dc", "downturn_dc", "new_high", "new_low", "rdc"}
        assert set(ref_idc.columns) == expected

    def test_datetime_index_preserved(self):
        dates = pd.date_range("2024-01-01", periods=len(REF_PRICES), freq="h")
        series = pd.Series(REF_PRICES, index=dates)
        result = idc_parse(series, THETA)
        assert result.index.equals(dates)

    def test_zero_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            idc_parse(pd.Series(REF_PRICES), theta=0.0)

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            idc_parse(pd.Series(REF_PRICES), theta=-0.01)

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            idc_parse(pd.Series(REF_PRICES), theta=THETA, alpha=0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            idc_parse(pd.Series(REF_PRICES), theta=THETA, alpha=1.01)

    def test_nan_price_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            idc_parse(pd.Series([100.0, np.nan, 112.0]), THETA)

    def test_inf_price_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            idc_parse(pd.Series([100.0, np.inf, 112.0]), THETA)


class TestDirectionValues:
    def test_bar0_direction_nan(self, ref_idc):
        assert np.isnan(ref_idc["direction"].iloc[0])

    def test_direction_after_first_dc(self, ref_idc):
        # Bar 1: first upward DC -> upturn (1.0)
        assert ref_idc["direction"].iloc[1] == pytest.approx(1.0)

    def test_direction_sequence(self, ref_idc):
        expected = [np.nan, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        for i, exp in enumerate(expected):
            val = ref_idc["direction"].iloc[i]
            if np.isnan(exp):
                assert np.isnan(val), f"Bar {i}: expected NaN, got {val}"
            else:
                assert val == pytest.approx(exp), f"Bar {i}: expected {exp}, got {val}"

    def test_direction_only_valid_values(self, ref_idc):
        valid = ref_idc["direction"].dropna()
        assert set(valid.unique()).issubset({1.0, -1.0})


class TestPhPl:
    def test_ph_bar0(self, ref_idc):
        assert ref_idc["ph"].iloc[0] == pytest.approx(100.0)

    def test_pl_bar0(self, ref_idc):
        assert ref_idc["pl"].iloc[0] == pytest.approx(100.0)

    def test_ph_sequence(self, ref_idc):
        # ph: 100, 112, 112, 112, 100, 115, 115
        expected_ph = [100.0, 112.0, 112.0, 112.0, 100.0, 115.0, 115.0]
        for i, exp in enumerate(expected_ph):
            assert ref_idc["ph"].iloc[i] == pytest.approx(exp), f"Bar {i}: ph expected {exp}"

    def test_pl_sequence(self, ref_idc):
        # pl: 100, 100, 100, 90, 90, 90, 103
        expected_pl = [100.0, 100.0, 100.0, 90.0, 90.0, 90.0, 103.0]
        for i, exp in enumerate(expected_pl):
            assert ref_idc["pl"].iloc[i] == pytest.approx(exp), f"Bar {i}: pl expected {exp}"

    def test_ph_nondecreasing_during_upturn(self, ref_idc):
        for i in range(1, len(ref_idc)):
            if ref_idc["direction"].iloc[i] == 1.0 and ref_idc["direction"].iloc[i - 1] == 1.0:
                assert ref_idc["ph"].iloc[i] >= ref_idc["ph"].iloc[i - 1]

    def test_pl_nonincreasing_during_downturn(self, ref_idc):
        for i in range(1, len(ref_idc)):
            if ref_idc["direction"].iloc[i] == -1.0 and ref_idc["direction"].iloc[i - 1] == -1.0:
                assert ref_idc["pl"].iloc[i] <= ref_idc["pl"].iloc[i - 1]

    def test_ph_always_positive(self, ref_idc):
        assert (ref_idc["ph"] > 0).all()

    def test_pl_always_positive(self, ref_idc):
        assert (ref_idc["pl"] > 0).all()


class TestDCSignals:
    def test_upturn_dc_bars(self, ref_idc):
        expected = [False, True, False, False, True, False, False]
        for i, exp in enumerate(expected):
            assert bool(ref_idc["upturn_dc"].iloc[i]) == exp, f"Bar {i}: upturn_dc expected {exp}"

    def test_downturn_dc_bars(self, ref_idc):
        expected = [False, False, True, False, False, False, True]
        for i, exp in enumerate(expected):
            assert bool(ref_idc["downturn_dc"].iloc[i]) == exp, f"Bar {i}: downturn_dc expected {exp}"

    def test_new_high_bars(self, ref_idc):
        expected = [False, False, False, False, False, True, False]
        for i, exp in enumerate(expected):
            assert bool(ref_idc["new_high"].iloc[i]) == exp, f"Bar {i}: new_high expected {exp}"

    def test_new_low_bars(self, ref_idc):
        expected = [False, False, False, True, False, False, False]
        for i, exp in enumerate(expected):
            assert bool(ref_idc["new_low"].iloc[i]) == exp, f"Bar {i}: new_low expected {exp}"

    def test_upturn_dc_and_new_high_mutually_exclusive(self, ref_idc):
        assert not (ref_idc["upturn_dc"] & ref_idc["new_high"]).any()

    def test_downturn_dc_and_new_low_mutually_exclusive(self, ref_idc):
        assert not (ref_idc["downturn_dc"] & ref_idc["new_low"]).any()

    def test_upturn_dc_and_downturn_dc_mutually_exclusive(self, ref_idc):
        assert not (ref_idc["upturn_dc"] & ref_idc["downturn_dc"]).any()

    def test_init_upturn_dc_is_flagged(self, ref_idc):
        # Bar 1: first DC from init is still flagged as upturn_dc
        assert ref_idc["upturn_dc"].iloc[1]

    def test_dc_fires_only_after_init(self, ref_idc):
        # No DC event on bar 0 (init bar)
        assert not ref_idc["upturn_dc"].iloc[0]
        assert not ref_idc["downturn_dc"].iloc[0]


class TestRDC:
    def test_rdc_nan_at_bar0(self, ref_idc):
        assert np.isnan(ref_idc["rdc"].iloc[0])

    def test_rdc_nan_at_init_dc(self, ref_idc):
        # Bar 1: first DC from init -> rdc is NaN (no completed prior trend)
        assert np.isnan(ref_idc["rdc"].iloc[1])

    def test_rdc_at_downturn_dc_bar2(self, ref_idc):
        # |100 - 112| / 112 / (2 - 1) = 12/112
        assert ref_idc["rdc"].iloc[2] == pytest.approx(12.0 / 112.0)

    def test_rdc_at_upturn_dc_bar4(self, ref_idc):
        # |100 - 90| / 90 / (4 - 3) = 10/90
        assert ref_idc["rdc"].iloc[4] == pytest.approx(10.0 / 90.0)

    def test_rdc_at_downturn_dc_bar6(self, ref_idc):
        # |103 - 115| / 115 / (6 - 5) = 12/115
        assert ref_idc["rdc"].iloc[6] == pytest.approx(12.0 / 115.0)

    def test_rdc_nan_on_non_dc_bars(self, ref_idc):
        non_dc = ~(ref_idc["upturn_dc"] | ref_idc["downturn_dc"])
        assert ref_idc["rdc"][non_dc].isna().all()

    def test_rdc_positive_where_not_nan(self, ref_idc):
        rdc_vals = ref_idc["rdc"].dropna()
        assert (rdc_vals > 0).all()


class TestTDC0:
    def test_t_dc0_sequence(self, ref_idc):
        # t_dc0: 0, 1, 2, 3, 4, 5, 6
        expected = [0, 1, 2, 3, 4, 5, 6]
        for i, exp in enumerate(expected):
            assert ref_idc["t_dc0"].iloc[i] == exp, f"Bar {i}: t_dc0 expected {exp}"

    def test_t_dc0_nonnegative(self, ref_idc):
        assert (ref_idc["t_dc0"] >= 0).all()

    def test_t_dc0_leq_bar_index(self, ref_idc):
        for i in range(len(ref_idc)):
            assert ref_idc["t_dc0"].iloc[i] <= i


class TestConsistencyWithParser:
    def test_downturn_dc_aligns_with_parser_dcc(self):
        # In parse_dc_events, DCC bars of 'up' trends correspond to bars where
        # a downward DC fires. The non-init downturn_dc bars in idc_parse must
        # match the dcc_idx values of 'up' trends in parse_dc_events.
        series = pd.Series(REF_PRICES)
        idc = idc_parse(series, THETA)
        trends = parse_dc_events(series, THETA)

        up_dcc_bars = set(trends[trends["direction"] == "up"]["dcc_idx"].tolist())
        # Bar 1 is the init upturn DC — has no corresponding trend in parse_dc_events
        # Non-init downturn_dc bars: bars 2 and 6
        downturn_dc_bars = set(idc.index[idc["downturn_dc"] & ~idc["upturn_dc"]].tolist()) - {
            idc.index[idc["upturn_dc"] & idc.index.isin([1])].tolist()[0]
            if (idc["upturn_dc"] & (idc.index == 1)).any()
            else -1
        }
        non_init_downturn_dc = set(idc.index[idc["downturn_dc"]].tolist())
        assert non_init_downturn_dc == up_dcc_bars

    def test_upturn_dc_aligns_with_parser_dcc(self):
        # In parse_dc_events, DCC bars of 'down' trends correspond to bars where
        # an upward DC fires (excluding the init DC which has no prior trend).
        series = pd.Series(REF_PRICES)
        idc = idc_parse(series, THETA)
        trends = parse_dc_events(series, THETA)

        down_dcc_bars = set(trends[trends["direction"] == "down"]["dcc_idx"].tolist())
        # Non-init upturn_dc bars: bar 4
        init_dc_bar = idc.index[idc["upturn_dc"]].tolist()[0]  # first upturn_dc = init
        non_init_upturn_dc = set(idc.index[idc["upturn_dc"]].tolist()) - {init_dc_bar}
        assert non_init_upturn_dc == down_dcc_bars

    def test_total_dc_events_consistent(self):
        # Non-init DC events in idc_parse should equal number of completed trends
        # in parse_dc_events. Each completed trend ends with a DC event (one that
        # is not the init DC).
        series = pd.Series(REF_PRICES)
        idc = idc_parse(series, THETA)
        trends = parse_dc_events(series, THETA)

        all_dc_bars = idc.index[idc["upturn_dc"] | idc["downturn_dc"]].tolist()
        non_init_dc_count = len(all_dc_bars) - 1  # subtract the init DC
        assert non_init_dc_count == len(trends)


class TestAsymmetricAlpha:
    # [100, 115, 108]: price falls 6.1% from 115.
    # alpha=1.0: downward DC needs 115*0.9=103.5  -> 108 does NOT qualify
    # alpha=0.5: downward DC needs 115*0.95=109.25 -> 108 qualifies
    ASYM_PRICES = [100.0, 115.0, 108.0]

    def test_alpha_1_no_downturn_dc(self):
        result = idc_parse(pd.Series(self.ASYM_PRICES), theta=0.1, alpha=1.0)
        assert not result["downturn_dc"].any()

    def test_alpha_05_fires_downturn_dc_at_bar2(self):
        result = idc_parse(pd.Series(self.ASYM_PRICES), theta=0.1, alpha=0.5)
        assert result["downturn_dc"].iloc[2]

    def test_alpha_does_not_affect_upward_dc(self):
        # [100, 90, 100]: upward DC fires at bar 2 for all alpha values
        prices = pd.Series([100.0, 90.0, 100.0])
        result_half = idc_parse(prices, theta=0.1, alpha=0.5)
        result_full = idc_parse(prices, theta=0.1, alpha=1.0)
        assert result_half["upturn_dc"].iloc[2]
        assert result_full["upturn_dc"].iloc[2]

    def test_ph_held_fixed_during_downturn(self):
        # During downturn phase, ph must not change
        prices = pd.Series([100.0, 115.0, 108.0, 107.0, 105.0])
        result = idc_parse(prices, theta=0.1, alpha=0.5)
        # After downturn DC at bar 2, ph should remain 115 throughout downturn
        for i in range(2, len(result)):
            if result["direction"].iloc[i] == -1.0:
                assert result["ph"].iloc[i] == pytest.approx(115.0), f"Bar {i}: ph changed during downturn"

    def test_pl_held_fixed_during_upturn(self):
        # During upturn phase, pl must not change
        prices = pd.Series([100.0, 90.0, 99.0, 102.0, 105.0])
        result = idc_parse(prices, theta=0.1, alpha=1.0)
        # After upturn DC at bar 2, pl should remain 90 throughout upturn
        for i in range(2, len(result)):
            if result["direction"].iloc[i] == 1.0:
                assert result["pl"].iloc[i] == pytest.approx(90.0), f"Bar {i}: pl changed during upturn"
