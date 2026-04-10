"""
Tests for create_lag_features utility.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.utils.lag_features import create_lag_features


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_df():
    return pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0], "vol": [10.0, 20.0, 30.0, 40.0, 50.0]})


@pytest.fixture
def multi_col_df():
    np.random.seed(0)
    return pd.DataFrame(
        {"a": np.arange(10, dtype=float), "b": np.arange(10, 20, dtype=float), "c": np.arange(20, 30, dtype=float)}
    )


# =============================================================================
# Output shape and structure
# =============================================================================

class TestOutputStructure:
    def test_column_count_single_lag_all_columns(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1])
        # 2 original + 2 lagged
        assert result.shape[1] == 4

    def test_column_count_multiple_lags_all_columns(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1, 2, 3])
        # 2 original + 2*3 lagged
        assert result.shape[1] == 8

    def test_column_count_subset_columns(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1, 2], columns=["close"])
        # 2 original + 1*2 lagged
        assert result.shape[1] == 4

    def test_row_count_preserved(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1, 2])
        assert len(result) == len(simple_df)

    def test_original_columns_preserved(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1])
        assert "close" in result.columns
        assert "vol" in result.columns

    def test_lag_column_naming(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1, 3])
        assert "close_lag_1" in result.columns
        assert "close_lag_3" in result.columns
        assert "vol_lag_1" in result.columns
        assert "vol_lag_3" in result.columns

    def test_lag_column_naming_subset(self, simple_df):
        result = create_lag_features(simple_df, n_list=[2], columns=["close"])
        assert "close_lag_2" in result.columns
        assert "vol_lag_2" not in result.columns

    def test_index_preserved(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({"x": range(5)}, index=idx)
        result = create_lag_features(df, n_list=[1])
        assert list(result.index) == list(idx)


# =============================================================================
# Correct lag values
# =============================================================================

class TestLagValues:
    def test_lag_1_shifts_by_one(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1])
        # row 1: close_lag_1 should equal original close at row 0
        assert result["close_lag_1"].iloc[1] == simple_df["close"].iloc[0]
        assert result["close_lag_1"].iloc[2] == simple_df["close"].iloc[1]

    def test_lag_2_shifts_by_two(self, simple_df):
        result = create_lag_features(simple_df, n_list=[2])
        assert result["close_lag_2"].iloc[2] == simple_df["close"].iloc[0]
        assert result["close_lag_2"].iloc[3] == simple_df["close"].iloc[1]

    def test_leading_rows_are_nan(self, simple_df):
        result = create_lag_features(simple_df, n_list=[2])
        assert pd.isna(result["close_lag_2"].iloc[0])
        assert pd.isna(result["close_lag_2"].iloc[1])
        assert not pd.isna(result["close_lag_2"].iloc[2])

    def test_original_values_unchanged(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1, 2])
        pd.testing.assert_series_equal(result["close"], simple_df["close"])
        pd.testing.assert_series_equal(result["vol"], simple_df["vol"])

    def test_multiple_lags_independent(self, multi_col_df):
        result = create_lag_features(multi_col_df, n_list=[1, 3])
        # lag_1 at row 3 == original at row 2
        assert result["a_lag_1"].iloc[3] == multi_col_df["a"].iloc[2]
        # lag_3 at row 3 == original at row 0
        assert result["a_lag_3"].iloc[3] == multi_col_df["a"].iloc[0]

    def test_all_columns_lagged_when_columns_is_none(self, multi_col_df):
        result = create_lag_features(multi_col_df, n_list=[1])
        for col in ["a", "b", "c"]:
            assert f"{col}_lag_1" in result.columns


# =============================================================================
# columns=None behaviour
# =============================================================================

class TestColumnsNone:
    def test_defaults_to_all_columns(self, simple_df):
        result_none = create_lag_features(simple_df, n_list=[1], columns=None)
        result_explicit = create_lag_features(simple_df, n_list=[1], columns=["close", "vol"])
        pd.testing.assert_frame_equal(result_none, result_explicit)

    def test_all_lag_columns_present(self, multi_col_df):
        result = create_lag_features(multi_col_df, n_list=[1, 2])
        for col in multi_col_df.columns:
            for n in [1, 2]:
                assert f"{col}_lag_{n}" in result.columns


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_single_row_all_lags_nan(self):
        df = pd.DataFrame({"x": [99.0]})
        result = create_lag_features(df, n_list=[1])
        assert pd.isna(result["x_lag_1"].iloc[0])

    def test_lag_larger_than_rows(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = create_lag_features(df, n_list=[5])
        assert result["x_lag_5"].isna().all()

    def test_non_sequential_lags(self, simple_df):
        result = create_lag_features(simple_df, n_list=[1, 5, 10])
        assert "close_lag_1" in result.columns
        assert "close_lag_5" in result.columns
        assert "close_lag_10" in result.columns

    def test_unsorted_lags(self, simple_df):
        result = create_lag_features(simple_df, n_list=[3, 1])
        # Columns should appear in the order lags were given
        lag_cols = [c for c in result.columns if "lag" in c]
        assert lag_cols[0] == "close_lag_3"
        assert lag_cols[1] == "vol_lag_3"
        assert lag_cols[2] == "close_lag_1"
        assert lag_cols[3] == "vol_lag_1"

    def test_dataframe_not_mutated(self, simple_df):
        original_cols = list(simple_df.columns)
        create_lag_features(simple_df, n_list=[1, 2])
        assert list(simple_df.columns) == original_cols

    def test_works_with_datetime_index(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="h")
        df = pd.DataFrame({"price": np.arange(6, dtype=float)}, index=idx)
        result = create_lag_features(df, n_list=[1, 2])
        assert result["price_lag_1"].iloc[1] == 0.0
        assert result["price_lag_2"].iloc[2] == 0.0


# =============================================================================
# Validation errors
# =============================================================================

class TestValidation:
    def test_empty_n_list_raises(self, simple_df):
        with pytest.raises(ValueError, match="n_list must contain at least one lag"):
            create_lag_features(simple_df, n_list=[])

    def test_zero_lag_raises(self, simple_df):
        with pytest.raises(ValueError, match="positive integers"):
            create_lag_features(simple_df, n_list=[0])

    def test_negative_lag_raises(self, simple_df):
        with pytest.raises(ValueError, match="positive integers"):
            create_lag_features(simple_df, n_list=[-1])

    def test_mixed_valid_invalid_lag_raises(self, simple_df):
        with pytest.raises(ValueError, match="positive integers"):
            create_lag_features(simple_df, n_list=[1, -2, 3])

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="Columns not found"):
            create_lag_features(simple_df, n_list=[1], columns=["close", "nonexistent"])

    def test_all_missing_columns_raises(self, simple_df):
        with pytest.raises(ValueError, match="Columns not found"):
            create_lag_features(simple_df, n_list=[1], columns=["foo", "bar"])
