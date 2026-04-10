"""
Tests for parse_dual_dc, label_bbtheta, and extract_tsfdc_features.

Synthetic reference series (theta=0.05):
  Prices with large moves ensure BTheta (0.10) extremes are a subset of STheta extremes.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import (
    extract_tsfdc_features,
    label_bbtheta,
    parse_dc_events,
    parse_dual_dc,
)

STHETA = 0.05
BTHETA = 0.10


def _make_zigzag(n: int = 400, amplitude: float = 0.08, seed: int = 7) -> pd.Series:
    """Synthetic price series with regular cycles larger than BTheta."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    prices = 100.0 * (1 + amplitude * np.sin(2 * np.pi * t / 20))
    noise = rng.normal(0, 0.005, n)
    return pd.Series(prices * (1 + noise))


@pytest.fixture(scope='module')
def zigzag():
    return _make_zigzag()


@pytest.fixture(scope='module')
def dual_dc(zigzag):
    return parse_dual_dc(zigzag, STHETA, BTHETA)


@pytest.fixture(scope='module')
def trends_s(dual_dc):
    return dual_dc[0]


@pytest.fixture(scope='module')
def trends_b(dual_dc):
    return dual_dc[1]


@pytest.fixture(scope='module')
def labelled(trends_s, trends_b):
    return label_bbtheta(trends_s, trends_b)


@pytest.fixture(scope='module')
def features(labelled, trends_b):
    return extract_tsfdc_features(labelled, trends_b, STHETA, BTHETA)


# ---------------------------------------------------------------------------
# parse_dual_dc tests
# ---------------------------------------------------------------------------

class TestParseDualDC:
    def test_returns_tuple_of_two_dataframes(self, dual_dc):
        ts, tb = dual_dc
        assert isinstance(ts, pd.DataFrame)
        assert isinstance(tb, pd.DataFrame)

    def test_stheta_has_more_trends_than_btheta(self, trends_s, trends_b):
        assert len(trends_s) > len(trends_b)

    def test_btheta_trends_non_empty(self, trends_b):
        assert len(trends_b) > 0

    def test_stheta_trends_non_empty(self, trends_s):
        assert len(trends_s) > 0

    def test_both_have_required_columns(self, trends_s, trends_b):
        required = {'direction', 'ext_start_price', 'ext_end_price', 'dcc_price',
                    'ext_end_pos', 'ext_start_pos', 'dcc_pos', 'tmv', 't', 'r'}
        for col in required:
            assert col in trends_s.columns, f"trends_s missing {col}"
            assert col in trends_b.columns, f"trends_b missing {col}"

    def test_raises_if_btheta_lte_stheta(self, zigzag):
        with pytest.raises(ValueError):
            parse_dual_dc(zigzag, 0.05, 0.03)

    def test_raises_if_btheta_equals_stheta(self, zigzag):
        with pytest.raises(ValueError):
            parse_dual_dc(zigzag, 0.05, 0.05)

    def test_btheta_ext_end_pos_subset_of_stheta(self, trends_s, trends_b):
        """Every BTheta extreme bar must also be a STheta extreme bar."""
        stheta_bars = set(trends_s['ext_end_pos'].values)
        for bar in trends_b['ext_end_pos'].values:
            assert bar in stheta_bars, f"BTheta extreme bar {bar} not in STheta extremes"

    def test_stheta_dcc_pos_monotone(self, trends_s):
        assert (np.diff(trends_s['dcc_pos'].values) > 0).all()

    def test_btheta_dcc_pos_monotone(self, trends_b):
        assert (np.diff(trends_b['dcc_pos'].values) > 0).all()


# ---------------------------------------------------------------------------
# label_bbtheta tests
# ---------------------------------------------------------------------------

class TestLabelBBTheta:
    def test_bbtheta_column_added(self, labelled):
        assert 'bbtheta' in labelled.columns

    def test_bbtheta_is_bool(self, labelled):
        assert labelled['bbtheta'].dtype == bool or labelled['bbtheta'].dtype == object
        # No NaN values
        assert not labelled['bbtheta'].isna().any()

    def test_original_columns_preserved(self, trends_s, labelled):
        for col in trends_s.columns:
            pd.testing.assert_series_equal(labelled[col], trends_s[col], check_names=False)

    def test_both_classes_present(self, labelled):
        # With synthetic data, some trends reach BTheta and some don't
        assert labelled['bbtheta'].any()
        assert (~labelled['bbtheta']).any()

    def test_true_labels_are_btheta_extremes(self, labelled, trends_b):
        """Every True label must correspond to a known BTheta extreme bar."""
        btheta_bars = set(trends_b['ext_end_pos'].values)
        true_bars = labelled.loc[labelled['bbtheta'] == True, 'ext_end_pos'].values
        for bar in true_bars:
            assert bar in btheta_bars

    def test_missing_column_raises_trends_s(self, trends_s, trends_b):
        bad = trends_s.drop(columns=['ext_end_pos'])
        with pytest.raises(ValueError, match='trends_s missing column'):
            label_bbtheta(bad, trends_b)

    def test_missing_column_raises_trends_b(self, trends_s, trends_b):
        bad = trends_b.drop(columns=['ext_end_pos'])
        with pytest.raises(ValueError, match='trends_b missing column'):
            label_bbtheta(trends_s, bad)

    def test_empty_btheta_gives_all_false(self, trends_s):
        empty_b = pd.DataFrame(columns=['ext_end_pos'])
        result = label_bbtheta(trends_s, empty_b)
        assert (result['bbtheta'] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# extract_tsfdc_features tests
# ---------------------------------------------------------------------------

class TestExtractTSFDCFeatures:
    def test_returns_dataframe_with_required_columns(self, features):
        for col in ('TMV', 'T', 'OSV', 'COP'):
            assert col in features.columns

    def test_same_index_as_trends_s(self, labelled, features):
        pd.testing.assert_index_equal(features.index, labelled.index)

    def test_tmv_equals_trends_s_tmv(self, labelled, features):
        np.testing.assert_array_equal(features['TMV'].values, labelled['tmv'].values)

    def test_t_equals_trends_s_t(self, labelled, features):
        np.testing.assert_array_equal(features['T'].values, labelled['t'].values)

    def test_tmv_always_positive(self, features):
        assert (features['TMV'] > 0).all()

    def test_t_always_positive(self, features):
        assert (features['T'] > 0).all()

    def test_osv_first_row_nan(self, features):
        assert pd.isna(features['OSV'].iloc[0])

    def test_osv_subsequent_rows_finite(self, features):
        osv_rest = features['OSV'].iloc[1:]
        assert osv_rest.notna().any()

    def test_cop_nan_before_first_btheta(self, zigzag, trends_b):
        """Rows before the first BTheta DCC should have COP = NaN."""
        first_b_dcc = trends_b['dcc_pos'].iloc[0]
        ts = parse_dc_events(zigzag, STHETA)
        feats = extract_tsfdc_features(ts, trends_b, STHETA, BTHETA)
        before_first = feats.loc[ts['dcc_pos'] < first_b_dcc, 'COP']
        assert before_first.isna().all()

    def test_cop_finite_after_first_btheta(self, features):
        """At least some COP values should be finite (after first BTheta DCC)."""
        assert features['COP'].notna().any()

    def test_cop_with_empty_btheta_all_nan(self, labelled):
        empty_b = pd.DataFrame(columns=['dcc_price', 'dcc_pos'])
        feats = extract_tsfdc_features(labelled, empty_b, STHETA, BTHETA)
        assert feats['COP'].isna().all()

    def test_missing_columns_raises_trends_s(self, labelled, trends_b):
        bad = labelled.drop(columns=['tmv'])
        with pytest.raises(ValueError, match='trends_s missing columns'):
            extract_tsfdc_features(bad, trends_b, STHETA, BTHETA)

    def test_missing_columns_raises_trends_b(self, labelled, trends_b):
        bad = trends_b.drop(columns=['dcc_price'])
        with pytest.raises(ValueError, match='trends_b missing columns'):
            extract_tsfdc_features(labelled, bad, STHETA, BTHETA)

    def test_osv_formula(self, labelled, features):
        """Verify OSV = (dcc_curr - dcc_prev) / (dcc_prev * stheta) at row 1."""
        dcc_0 = labelled['dcc_price'].iloc[0]
        dcc_1 = labelled['dcc_price'].iloc[1]
        expected = (dcc_1 - dcc_0) / (dcc_0 * STHETA)
        assert features['OSV'].iloc[1] == pytest.approx(expected, rel=1e-9)