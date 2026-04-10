"""
Tests for timothymasters/utils/cross_features_computer.py

Mirrors the structure of test_feature_computer.py but for paired-market
(cross-market) indicators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.timothymasters.utils.cross_features_computer import (
    ALL_INDICATORS,
    CORRELATION_INDICATORS,
    DEFAULT_PARAMS,
    DEVIATION_INDICATORS,
    GROUPS,
    PURIFY_INDICATORS,
    TREND_INDICATORS,
    compute_cross_features,
    list_groups,
    list_indicators,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 300  # enough bars for all warmup periods


def _make_ohlc(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + rng.standard_normal(N).cumsum()
    high = close + rng.uniform(0.1, 1.0, N)
    low = close - rng.uniform(0.1, 1.0, N)
    return pd.DataFrame({"high": high, "low": low, "close": close})


@pytest.fixture
def df1() -> pd.DataFrame:
    return _make_ohlc(seed=1)


@pytest.fixture
def df2() -> pd.DataFrame:
    return _make_ohlc(seed=2)


# ---------------------------------------------------------------------------
# Group / indicator constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_all_indicators_count(self):
        assert len(ALL_INDICATORS) == 7

    def test_correlation_indicators(self):
        assert CORRELATION_INDICATORS == ["correlation", "delta_correlation"]

    def test_deviation_indicators(self):
        assert DEVIATION_INDICATORS == ["deviation"]

    def test_trend_indicators(self):
        assert TREND_INDICATORS == ["trend_diff", "cmma_diff"]

    def test_all_is_concatenation(self):
        assert ALL_INDICATORS == CORRELATION_INDICATORS + DEVIATION_INDICATORS + PURIFY_INDICATORS + TREND_INDICATORS

    def test_groups_keys(self):
        assert set(GROUPS.keys()) == {"correlation", "deviation", "purify", "trend", "all"}

    def test_groups_all_value(self):
        assert GROUPS["all"] == ALL_INDICATORS


class TestListHelpers:
    def test_list_all(self):
        assert list_indicators("all") == ALL_INDICATORS

    def test_list_correlation(self):
        assert list_indicators("correlation") == CORRELATION_INDICATORS

    def test_list_deviation(self):
        assert list_indicators("deviation") == DEVIATION_INDICATORS

    def test_list_trend(self):
        assert list_indicators("trend") == TREND_INDICATORS

    def test_list_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown group"):
            list_indicators("volume")

    def test_list_groups(self):
        assert set(list_groups()) == {"correlation", "deviation", "purify", "trend", "all"}


# ---------------------------------------------------------------------------
# compute_cross_features — shape and column naming
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_all_group_gives_7_columns(self, df1, df2):
        result = compute_cross_features(df1, df2)
        assert result.shape == (N, 7)

    def test_correlation_group_gives_2_columns(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="correlation")
        assert result.shape[1] == 2

    def test_deviation_group_gives_1_column(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="deviation")
        assert result.shape[1] == 1

    def test_trend_group_gives_2_columns(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="trend")
        assert result.shape[1] == 2

    def test_list_of_groups_deduplicates(self, df1, df2):
        result = compute_cross_features(df1, df2, groups=["correlation", "correlation"])
        assert result.shape[1] == 2

    def test_mixed_list_of_groups(self, df1, df2):
        result = compute_cross_features(df1, df2, groups=["correlation", "deviation"])
        assert result.shape[1] == 3  # 2 + 1

    def test_individual_indicator_names(self, df1, df2):
        # "delta_correlation" and "trend_diff" are individual indicator names (not group names)
        result = compute_cross_features(df1, df2, groups=["delta_correlation", "trend_diff"])
        assert result.shape[1] == 2

    def test_index_matches_data1(self, df1, df2):
        result = compute_cross_features(df1, df2)
        pd.testing.assert_index_equal(result.index, df1.index)

    def test_all_columns_are_float64(self, df1, df2):
        result = compute_cross_features(df1, df2)
        for col in result.columns:
            assert result[col].dtype == np.float64, f"{col} is not float64"


class TestColumnNaming:
    def test_default_prefix_cm(self, df1, df2):
        result = compute_cross_features(df1, df2)
        assert all(c.startswith("cm_") for c in result.columns)

    def test_custom_prefix(self, df1, df2):
        result = compute_cross_features(df1, df2, prefix="x_")
        assert all(c.startswith("x_") for c in result.columns)

    def test_empty_prefix(self, df1, df2):
        result = compute_cross_features(df1, df2, prefix="")
        assert set(result.columns) == set(ALL_INDICATORS)

    def test_all_indicator_names_present(self, df1, df2):
        result = compute_cross_features(df1, df2)
        expected = {f"cm_{name}" for name in ALL_INDICATORS}
        assert set(result.columns) == expected

    def test_column_order_matches_indicator_order(self, df1, df2):
        result = compute_cross_features(df1, df2, prefix="")
        assert result.columns.tolist() == ALL_INDICATORS


# ---------------------------------------------------------------------------
# Warmup NaNs
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_correlation_has_warmup_nans(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="correlation", prefix="")
        period = DEFAULT_PARAMS["correlation"]["period"]  # 63
        assert result["correlation"].iloc[:period - 1].isna().all()
        assert not result["correlation"].iloc[period:].isna().all()

    def test_delta_correlation_has_warmup_nans(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="correlation", prefix="")
        p = DEFAULT_PARAMS["delta_correlation"]
        warmup = p["period"] + p["delta_period"] - 2
        assert result["delta_correlation"].iloc[:warmup].isna().all()

    def test_deviation_has_warmup_nans(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="deviation", prefix="")
        period = DEFAULT_PARAMS["deviation"]["period"]
        assert result["deviation"].iloc[:period - 1].isna().all()
        assert not result["deviation"].iloc[period:].isna().all()

    def test_trend_diff_has_warmup_nans(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="trend", prefix="")
        p = DEFAULT_PARAMS["trend_diff"]
        first_valid = max(p["period"] - 1, p["atr_period"])
        assert result["trend_diff"].iloc[:first_valid].isna().all()
        assert not result["trend_diff"].iloc[first_valid:].isna().all()

    def test_cmma_diff_has_warmup_nans(self, df1, df2):
        result = compute_cross_features(df1, df2, groups="trend", prefix="")
        p = DEFAULT_PARAMS["cmma_diff"]
        first_valid = max(p["period"] - 1, p["atr_period"])
        assert result["cmma_diff"].iloc[:first_valid].isna().all()
        assert not result["cmma_diff"].iloc[first_valid:].isna().all()


# ---------------------------------------------------------------------------
# Custom params
# ---------------------------------------------------------------------------

class TestCustomParams:
    def test_shorter_period_reduces_warmup(self, df1, df2):
        default = compute_cross_features(df1, df2, groups="correlation", prefix="")
        custom = compute_cross_features(
            df1, df2, groups="correlation", params={"correlation": {"period": 20}}, prefix=""
        )
        # With period=20, first valid is bar 19; with period=63, first valid is bar 62
        assert np.isnan(default["correlation"].iloc[30])
        assert not np.isnan(custom["correlation"].iloc[30])

    def test_params_for_unselected_indicators_are_ignored(self, df1, df2):
        # trend_diff not in groups; its params should have no effect
        result = compute_cross_features(
            df1, df2,
            groups="correlation",
            params={"trend_diff": {"period": 999}},
            prefix="",
        )
        assert "correlation" in result.columns
        assert "trend_diff" not in result.columns


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_different_lengths_raises_value_error(self, df1):
        df2_short = _make_ohlc(seed=2).iloc[:200]
        with pytest.raises(ValueError, match="same length"):
            compute_cross_features(df1, df2_short)

    def test_unknown_group_raises_value_error(self, df1, df2):
        with pytest.raises(ValueError, match="Unknown group or indicator"):
            compute_cross_features(df1, df2, groups="volume")

    def test_unknown_indicator_name_raises_value_error(self, df1, df2):
        with pytest.raises(ValueError, match="Unknown group or indicator"):
            compute_cross_features(df1, df2, groups="nonexistent_indicator")

    def test_missing_close_col_data1_raises_key_error(self, df2):
        df_no_close = _make_ohlc(seed=1).drop(columns=["close"])
        with pytest.raises(KeyError, match="close"):
            compute_cross_features(df_no_close, df2, groups="correlation")

    def test_missing_close_col_data2_raises_key_error(self, df1):
        df_no_close = _make_ohlc(seed=2).drop(columns=["close"])
        with pytest.raises(KeyError, match="close"):
            compute_cross_features(df1, df_no_close, groups="correlation")

    def test_missing_high_col_raises_key_error(self, df1, df2):
        df_no_high = df2.drop(columns=["high"])
        with pytest.raises(KeyError, match="high"):
            compute_cross_features(df1, df_no_high, groups="trend")

    def test_missing_low_col_raises_key_error(self, df1, df2):
        df_no_low = df2.drop(columns=["low"])
        with pytest.raises(KeyError, match="low"):
            compute_cross_features(df1, df_no_low, groups="trend")

    def test_custom_col_name_works(self, df1, df2):
        df1_renamed = df1.rename(columns={"close": "Close"})
        df2_renamed = df2.rename(columns={"close": "Close"})
        result = compute_cross_features(
            df1_renamed, df2_renamed, groups="correlation", close_col="Close"
        )
        assert result.shape[1] == 2

    def test_no_inf_in_output(self, df1, df2):
        result = compute_cross_features(df1, df2)
        assert not np.isinf(result.values[~np.isnan(result.values)]).any()


# ---------------------------------------------------------------------------
# Public API importable from top-level timothymasters
# ---------------------------------------------------------------------------

class TestTopLevelImport:
    def test_compute_cross_features_importable_from_timothymasters(self):
        from okmich_quant_features.timothymasters import compute_cross_features as ccf  # noqa: F401
        assert callable(ccf)

    def test_compute_cross_features_importable_from_utils(self):
        from okmich_quant_features.timothymasters.utils import compute_cross_features as ccf  # noqa: F401
        assert callable(ccf)
