"""
Tests for timothymasters.feature_computer — batch computation layer.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.timothymasters import (
    compute_features, list_indicators, list_groups,
    DEFAULT_PARAMS,
)
from okmich_quant_features.timothymasters.utils.single_features_computer import (
    ALL_INDICATORS,
    MOMENTUM_INDICATORS,
    TREND_INDICATORS,
    VARIANCE_INDICATORS,
    VOLUME_INDICATORS,
    INFORMATION_INDICATORS,
    FTI_INDICATORS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ohlcv_df():
    """200-bar OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 200
    close = 100.0 + rng.standard_normal(n).cumsum()
    high  = close + np.abs(rng.standard_normal(n)) * 0.5 + 0.1
    low   = close - np.abs(rng.standard_normal(n)) * 0.5 - 0.1
    open_ = close + rng.standard_normal(n) * 0.3
    volume = rng.integers(1_000, 5_000, n).astype(np.float64)

    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def no_fti_df():
    """Smaller 100-bar DataFrame for fast non-FTI group tests."""
    rng = np.random.default_rng(7)
    n = 100
    close = 50.0 + rng.standard_normal(n).cumsum()
    high  = close + np.abs(rng.standard_normal(n)) + 0.1
    low   = close - np.abs(rng.standard_normal(n)) - 0.1
    open_ = close + rng.standard_normal(n) * 0.2
    volume = rng.integers(500, 3_000, n).astype(np.float64)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": volume}
    )


# ---------------------------------------------------------------------------
# Shape and column name tests
# ---------------------------------------------------------------------------

class TestComputeFeatures:

    def test_all_groups_shape(self, ohlcv_df):
        result = compute_features(ohlcv_df)
        assert result.shape == (200, 38), f"Expected (200, 38), got {result.shape}"

    def test_all_groups_column_names(self, ohlcv_df):
        result = compute_features(ohlcv_df)
        assert all(col.startswith("tm_") for col in result.columns)
        expected = ["tm_" + name for name in ALL_INDICATORS]
        assert list(result.columns) == expected

    def test_momentum_group_count(self, no_fti_df):
        result = compute_features(no_fti_df, groups="momentum")
        assert result.shape[1] == 11
        for name in MOMENTUM_INDICATORS:
            assert f"tm_{name}" in result.columns

    def test_trend_group_count(self, no_fti_df):
        result = compute_features(no_fti_df, groups="trend")
        assert result.shape[1] == 10
        for name in TREND_INDICATORS:
            assert f"tm_{name}" in result.columns

    def test_variance_group_count(self, no_fti_df):
        result = compute_features(no_fti_df, groups="variance")
        assert result.shape[1] == 2

    def test_volume_group_count(self, no_fti_df):
        result = compute_features(no_fti_df, groups="volume")
        assert result.shape[1] == 9

    def test_information_group_count(self, no_fti_df):
        result = compute_features(no_fti_df, groups="information")
        assert result.shape[1] == 2

    def test_fti_group_count(self, ohlcv_df):
        result = compute_features(ohlcv_df, groups="fti")
        assert result.shape[1] == 4
        for name in FTI_INDICATORS:
            assert f"tm_{name}" in result.columns

    def test_individual_indicator_list(self, no_fti_df):
        result = compute_features(no_fti_df, groups=["rsi", "macd", "adx"])
        assert result.shape[1] == 3
        assert list(result.columns) == ["tm_rsi", "tm_macd", "tm_adx"]

    def test_mixed_group_and_indicator(self, no_fti_df):
        result = compute_features(no_fti_df, groups=["momentum", "trend"])
        assert result.shape[1] == 21
        # no duplicates
        assert len(set(result.columns)) == 21

    def test_mixed_list_deduplication(self, no_fti_df):
        # "momentum" + "rsi" should still be 11 columns, not 12
        result = compute_features(no_fti_df, groups=["momentum", "rsi"])
        assert result.shape[1] == 11

    # ---------------------------------------------------------------------------
    # Parameter override
    # ---------------------------------------------------------------------------

    def test_custom_params_rsi_warmup(self, no_fti_df):
        # Default period=14: first 14 bars are NaN, bar 14 is valid
        default_result = compute_features(no_fti_df, groups="rsi")
        assert np.all(np.isnan(default_result["tm_rsi"].iloc[:14].values))
        assert not np.isnan(default_result["tm_rsi"].iloc[14])

        # period=21: first 21 bars are NaN, bar 21 is valid
        custom_result = compute_features(no_fti_df, groups="rsi", params={"rsi": {"period": 21}})
        assert np.all(np.isnan(custom_result["tm_rsi"].iloc[:21].values))
        assert not np.isnan(custom_result["tm_rsi"].iloc[21])

    def test_custom_params_do_not_affect_other_indicators(self, no_fti_df):
        r1 = compute_features(no_fti_df, groups=["rsi", "macd"])
        r2 = compute_features(no_fti_df, groups=["rsi", "macd"], params={"rsi": {"period": 21}})
        # macd values should be identical
        np.testing.assert_array_equal(r1["tm_macd"].values, r2["tm_macd"].values)

    # ---------------------------------------------------------------------------
    # Prefix
    # ---------------------------------------------------------------------------

    def test_empty_prefix(self, no_fti_df):
        result = compute_features(no_fti_df, groups=["rsi", "adx"], prefix="")
        assert list(result.columns) == ["rsi", "adx"]

    def test_custom_prefix(self, no_fti_df):
        result = compute_features(no_fti_df, groups=["rsi"], prefix="feat_")
        assert "feat_rsi" in result.columns

    # ---------------------------------------------------------------------------
    # Index preservation
    # ---------------------------------------------------------------------------

    def test_output_index_matches_input(self, ohlcv_df):
        result = compute_features(ohlcv_df, groups="momentum")
        pd.testing.assert_index_equal(result.index, ohlcv_df.index)

    def test_integer_index_preserved(self):
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame({
            "open":   100 + rng.standard_normal(n).cumsum(),
            "high":   101 + rng.standard_normal(n).cumsum(),
            "low":    99  + rng.standard_normal(n).cumsum(),
            "close":  100 + rng.standard_normal(n).cumsum(),
            "tick_volume": rng.integers(100, 1000, n).astype(float),
        }, index=np.arange(10, 10 + n))
        result = compute_features(df, groups=["rsi", "adx"])
        pd.testing.assert_index_equal(result.index, df.index)

    # ---------------------------------------------------------------------------
    # NaN warmup
    # ---------------------------------------------------------------------------

    def test_warmup_nans_present_rsi(self, no_fti_df):
        result = compute_features(no_fti_df, groups="rsi")
        # period=14 → first 14 bars (indices 0..13) should be NaN
        assert np.all(np.isnan(result["tm_rsi"].values[:14]))

    def test_warmup_nans_present_adx(self, no_fti_df):
        # adx front_bad = 2*period - 1 = 27
        result = compute_features(no_fti_df, groups="adx")
        assert np.all(np.isnan(result["tm_adx"].values[:27]))
        assert not np.isnan(result["tm_adx"].values[27])

    def test_valid_values_after_warmup(self, no_fti_df):
        result = compute_features(no_fti_df, groups="momentum")
        # At least half the rows should have valid (non-NaN) values for rsi
        valid = ~np.isnan(result["tm_rsi"].values)
        assert valid.sum() > 50

    # ---------------------------------------------------------------------------
    # Output dtype
    # ---------------------------------------------------------------------------

    def test_all_float64(self, no_fti_df):
        result = compute_features(no_fti_df, groups=["momentum", "trend", "variance", "volume", "information"])
        for col in result.columns:
            assert result[col].dtype == np.float64, f"Column {col!r} has dtype {result[col].dtype}"

    # ---------------------------------------------------------------------------
    # Column mapping
    # ---------------------------------------------------------------------------

    def test_custom_column_names(self, no_fti_df):
        df = no_fti_df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        })
        result = compute_features(
            df, groups="momentum",
            open_col="Open", high_col="High", low_col="Low",
            close_col="Close", volume_col="Volume",
        )
        assert result.shape[1] == 11

    # ---------------------------------------------------------------------------
    # Error cases
    # ---------------------------------------------------------------------------

    def test_missing_volume_raises_keyerror(self, no_fti_df):
        df_no_vol = no_fti_df.drop(columns=["tick_volume"])
        with pytest.raises(KeyError, match="volume"):
            compute_features(df_no_vol, groups="volume")

    def test_missing_hlc_raises_keyerror(self):
        df = pd.DataFrame({"close": np.arange(50, dtype=float)})
        with pytest.raises(KeyError):
            compute_features(df, groups=["adx"])

    def test_unknown_group_raises_valueerror(self, no_fti_df):
        with pytest.raises(ValueError, match="Unknown group or indicator"):
            compute_features(no_fti_df, groups="not_a_group")

    def test_unknown_indicator_raises_valueerror(self, no_fti_df):
        with pytest.raises(ValueError, match="Unknown group or indicator"):
            compute_features(no_fti_df, groups=["rsi", "nonexistent_indicator"])

    def test_unknown_group_in_list_raises_valueerror(self, no_fti_df):
        with pytest.raises(ValueError):
            compute_features(no_fti_df, groups=["momentum", "bogus"])

    # ---------------------------------------------------------------------------
    # Value correctness spot-checks
    # ---------------------------------------------------------------------------

    def test_rsi_range(self, no_fti_df):
        result = compute_features(no_fti_df, groups="rsi")
        valid = result["tm_rsi"].dropna().values
        assert np.all(valid >= 0.0) and np.all(valid <= 100.0)

    def test_aroon_range(self, no_fti_df):
        result = compute_features(no_fti_df, groups=["aroon_up", "aroon_down"])
        for col in ["tm_aroon_up", "tm_aroon_down"]:
            valid = result[col].dropna().values
            assert np.all(valid >= 0.0) and np.all(valid <= 100.0)

    def test_adx_range(self, no_fti_df):
        result = compute_features(no_fti_df, groups="adx")
        valid = result["tm_adx"].dropna().values
        assert np.all(valid >= 0.0) and np.all(valid <= 100.0)


# ---------------------------------------------------------------------------
# list_indicators / list_groups
# ---------------------------------------------------------------------------

class TestListFunctions:

    def test_list_indicators_all(self):
        names = list_indicators("all")
        assert len(names) == 38
        assert names == ALL_INDICATORS

    def test_list_indicators_momentum(self):
        assert list_indicators("momentum") == MOMENTUM_INDICATORS

    def test_list_indicators_unknown_raises(self):
        with pytest.raises(ValueError):
            list_indicators("bogus")

    def test_list_groups_returns_all_group_names(self):
        groups = list_groups()
        assert set(groups) == {"momentum", "trend", "variance", "volume", "information", "fti", "all"}

    def test_groups_dict_total(self):
        assert len(ALL_INDICATORS) == 38
        # Sum of individual groups equals total (no overlaps)
        from_groups = (
            len(MOMENTUM_INDICATORS)
            + len(TREND_INDICATORS)
            + len(VARIANCE_INDICATORS)
            + len(VOLUME_INDICATORS)
            + len(INFORMATION_INDICATORS)
            + len(FTI_INDICATORS)
        )
        assert from_groups == 38

    def test_default_params_keys_match_all_indicators(self):
        assert set(DEFAULT_PARAMS.keys()) == set(ALL_INDICATORS)
