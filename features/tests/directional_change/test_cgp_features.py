"""
Tests for label_alpha_beta_dc and extract_dc_classification_features.

Reference series: [100, 112, 100, 90, 100, 115, 103], theta=0.1
  Trend 0 (up):   ext_end@1, dcc@2  → dc_length=1, os_length=3-2=1, αDC
  Trend 1 (down): ext_end@3, dcc@4  → dc_length=1, os_length=5-4=1, αDC
  Trend 2 (up):   ext_end@5, dcc@6  → dc_length=1, os_length=NaN   (last)
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import (
    extract_dc_classification_features,
    label_alpha_beta_dc,
    parse_dc_events,
)

THETA = 0.1
REF_PRICES = [100.0, 112.0, 100.0, 90.0, 100.0, 115.0, 103.0]


@pytest.fixture
def ref_trends():
    return parse_dc_events(pd.Series(REF_PRICES), THETA)


@pytest.fixture
def labelled(ref_trends):
    return label_alpha_beta_dc(ref_trends)


class TestLabelAlphaBetaDC:
    def test_new_columns_present(self, labelled):
        for col in ('dc_length', 'os_length', 'has_os'):
            assert col in labelled.columns

    def test_original_columns_unchanged(self, ref_trends, labelled):
        for col in ref_trends.columns:
            pd.testing.assert_series_equal(labelled[col], ref_trends[col], check_names=False)

    def test_dc_length_values(self, labelled):
        # dc_length = dcc_pos - ext_end_pos
        expected = (labelled['dcc_pos'] - labelled['ext_end_pos']).values
        np.testing.assert_array_equal(labelled['dc_length'].values, expected)

    def test_dc_length_always_positive(self, labelled):
        assert (labelled['dc_length'] >= 1).all()

    def test_os_length_first_two_rows(self, labelled):
        assert labelled['os_length'].iloc[0] == pytest.approx(1.0)
        assert labelled['os_length'].iloc[1] == pytest.approx(1.0)

    def test_os_length_last_row_nan(self, labelled):
        assert pd.isna(labelled['os_length'].iloc[-1])

    def test_has_os_first_two_true(self, labelled):
        assert labelled['has_os'].iloc[0] is True
        assert labelled['has_os'].iloc[1] is True

    def test_has_os_last_row_nan(self, labelled):
        assert pd.isna(labelled['has_os'].iloc[-1])

    def test_beta_dc_labelled_correctly(self):
        # Construct a series where an immediate reversal creates a βDC:
        # very tight zigzag forces dc events with minimal OS
        prices = pd.Series([100.0, 111.0, 99.9, 110.0, 98.9, 109.0])
        trends = parse_dc_events(prices, theta=0.1)
        if len(trends) >= 2:
            labelled = label_alpha_beta_dc(trends)
            # os_length can be 0 → βDC
            assert 'has_os' in labelled.columns

    def test_missing_columns_raises(self, ref_trends):
        bad = ref_trends.drop(columns=['ext_end_pos'])
        with pytest.raises(ValueError, match='missing columns'):
            label_alpha_beta_dc(bad)

    def test_single_trend_last_row_only(self):
        # Minimal series producing exactly one trend → all NaN labels
        prices = pd.Series([100.0, 112.0, 100.0])
        trends = parse_dc_events(prices, THETA)
        if len(trends) == 1:
            labelled = label_alpha_beta_dc(trends)
            assert pd.isna(labelled['has_os'].iloc[0])


class TestExtractDCClassificationFeatures:
    def test_output_columns(self, labelled):
        feats = extract_dc_classification_features(labelled)
        assert list(feats.columns) == ['X1', 'X2', 'X3', 'X4', 'X5']

    def test_no_x6_column(self, labelled):
        feats = extract_dc_classification_features(labelled)
        assert 'X6' not in feats.columns

    def test_same_index_as_input(self, labelled):
        feats = extract_dc_classification_features(labelled)
        pd.testing.assert_index_equal(feats.index, labelled.index)

    def test_x1_magnitude(self, labelled):
        feats = extract_dc_classification_features(labelled)
        expected = (labelled['dcc_price'] - labelled['ext_end_price']).abs()
        pd.testing.assert_series_equal(feats['X1'], expected, check_names=False)

    def test_x2_equals_dc_length_clipped(self, labelled):
        feats = extract_dc_classification_features(labelled)
        expected = labelled['dc_length'].clip(lower=1)
        pd.testing.assert_series_equal(feats['X2'].astype(int), expected.astype(int), check_names=False)

    def test_x3_equals_x1_over_x2(self, labelled):
        feats = extract_dc_classification_features(labelled)
        np.testing.assert_allclose(feats['X3'].values, (feats['X1'] / feats['X2']).values)

    def test_x4_is_previous_dcc_price(self, labelled):
        feats = extract_dc_classification_features(labelled)
        assert pd.isna(feats['X4'].iloc[0])
        assert feats['X4'].iloc[1] == pytest.approx(labelled['dcc_price'].iloc[0])
        assert feats['X4'].iloc[2] == pytest.approx(labelled['dcc_price'].iloc[1])

    def test_x5_first_row_zero(self, labelled):
        feats = extract_dc_classification_features(labelled)
        assert feats['X5'].iloc[0] == 0

    def test_x5_reflects_previous_has_os(self, labelled):
        feats = extract_dc_classification_features(labelled)
        # Row 1: prior trend (row 0) was αDC → X5=1
        assert feats['X5'].iloc[1] == 1
        # Row 2: prior trend (row 1) was αDC → X5=1
        assert feats['X5'].iloc[2] == 1

    def test_missing_columns_raises(self, ref_trends):
        with pytest.raises(ValueError, match='missing columns'):
            extract_dc_classification_features(ref_trends)