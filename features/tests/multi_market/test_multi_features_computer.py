"""
Tests for timothymasters/utils/multi_features_computer.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.timothymasters.utils.multi_features_computer import (
    ALL_INDICATORS,
    CMMA_STATS_INDICATORS,
    DEFAULT_PARAMS,
    GROUPS,
    JANUS_INDICATORS,
    RISK_INDICATORS,
    TREND_STATS_INDICATORS,
    compute_multi_features,
    list_groups,
    list_indicators,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300
N_MARKETS = 5


def _make_ohlc(seed: int = 0, n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = np.maximum(100.0 + np.cumsum(rng.standard_normal(n)), 1.0)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    return pd.DataFrame({"high": high, "low": low, "close": close})


@pytest.fixture
def markets():
    return [_make_ohlc(seed=i * 5 + 1) for i in range(N_MARKETS)]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_all_indicators_count(self):
        assert len(ALL_INDICATORS) == 40

    def test_trend_stats_count(self):
        assert len(TREND_STATS_INDICATORS) == 5

    def test_cmma_stats_count(self):
        assert len(CMMA_STATS_INDICATORS) == 5

    def test_risk_count(self):
        assert len(RISK_INDICATORS) == 5

    def test_all_is_concatenation(self):
        assert ALL_INDICATORS == TREND_STATS_INDICATORS + CMMA_STATS_INDICATORS + RISK_INDICATORS + JANUS_INDICATORS

    def test_groups_keys(self):
        assert set(GROUPS.keys()) == {"trend_stats", "cmma_stats", "risk", "janus", "all"}

    def test_groups_all_value(self):
        assert GROUPS["all"] == ALL_INDICATORS

    def test_default_params_covers_all_indicators(self):
        for name in ALL_INDICATORS:
            assert name in DEFAULT_PARAMS, f"Missing default params for {name!r}"


# ---------------------------------------------------------------------------
# list_indicators / list_groups
# ---------------------------------------------------------------------------

class TestListHelpers:
    def test_list_all(self):
        assert list_indicators("all") == ALL_INDICATORS

    def test_list_trend_stats(self):
        assert list_indicators("trend_stats") == TREND_STATS_INDICATORS

    def test_list_cmma_stats(self):
        assert list_indicators("cmma_stats") == CMMA_STATS_INDICATORS

    def test_list_risk(self):
        assert list_indicators("risk") == RISK_INDICATORS

    def test_list_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown group"):
            list_indicators("volume")

    def test_list_groups(self):
        assert set(list_groups()) == {"trend_stats", "cmma_stats", "risk", "janus", "all"}


# ---------------------------------------------------------------------------
# compute_multi_features — output shape and column naming
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_all_group_gives_40_columns(self, markets):
        result = compute_multi_features(markets)
        assert result.shape == (N, 40)

    def test_trend_stats_group_gives_5_columns(self, markets):
        result = compute_multi_features(markets, groups="trend_stats")
        assert result.shape[1] == 5

    def test_cmma_stats_group_gives_5_columns(self, markets):
        result = compute_multi_features(markets, groups="cmma_stats")
        assert result.shape[1] == 5

    def test_risk_group_gives_5_columns(self, markets):
        result = compute_multi_features(markets, groups="risk")
        assert result.shape[1] == 5

    def test_list_of_groups_deduplicates(self, markets):
        result = compute_multi_features(markets, groups=["trend_stats", "trend_stats"])
        assert result.shape[1] == 5

    def test_mixed_list_of_groups(self, markets):
        result = compute_multi_features(markets, groups=["trend_stats", "risk"])
        assert result.shape[1] == 10

    def test_individual_indicator_name(self, markets):
        result = compute_multi_features(markets, groups="mahal")
        assert result.shape[1] == 1

    def test_index_matches_markets_0(self, markets):
        result = compute_multi_features(markets)
        pd.testing.assert_index_equal(result.index, markets[0].index)

    def test_all_columns_float64(self, markets):
        result = compute_multi_features(markets)
        for col in result.columns:
            assert result[col].dtype == np.float64, f"{col} is not float64"


class TestColumnNaming:
    def test_default_prefix_mm(self, markets):
        result = compute_multi_features(markets)
        assert all(c.startswith("mm_") for c in result.columns)

    def test_custom_prefix(self, markets):
        result = compute_multi_features(markets, prefix="x_")
        assert all(c.startswith("x_") for c in result.columns)

    def test_empty_prefix(self, markets):
        result = compute_multi_features(markets, prefix="")
        assert set(result.columns) == set(ALL_INDICATORS)

    def test_all_indicator_names_present(self, markets):
        result = compute_multi_features(markets)
        expected = {f"mm_{name}" for name in ALL_INDICATORS}
        assert set(result.columns) == expected

    def test_column_order_matches_indicator_order(self, markets):
        result = compute_multi_features(markets, prefix="")
        assert result.columns.tolist() == ALL_INDICATORS


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_length_mismatch_raises_value_error(self, markets):
        short = _make_ohlc(seed=99, n=200)
        bad_markets = [short] + markets[1:]
        with pytest.raises(ValueError, match="same length"):
            compute_multi_features(bad_markets)

    def test_single_market_raises_value_error(self, markets):
        with pytest.raises(ValueError, match="At least 2 markets"):
            compute_multi_features([markets[0]])

    def test_unknown_group_raises_value_error(self, markets):
        with pytest.raises(ValueError, match="Unknown group or indicator"):
            compute_multi_features(markets, groups="volume")

    def test_unknown_indicator_name_raises_value_error(self, markets):
        with pytest.raises(ValueError, match="Unknown group or indicator"):
            compute_multi_features(markets, groups="nonexistent_indicator")

    def test_missing_close_col_raises_key_error(self, markets):
        bad = [markets[0].drop(columns=["close"])] + markets[1:]
        with pytest.raises(KeyError, match="close"):
            compute_multi_features(bad, groups="risk")

    def test_missing_high_col_raises_key_error(self, markets):
        bad = [markets[0].drop(columns=["high"])] + markets[1:]
        with pytest.raises(KeyError, match="high"):
            compute_multi_features(bad, groups="trend_stats")

    def test_missing_low_col_raises_key_error(self, markets):
        bad = [markets[0].drop(columns=["low"])] + markets[1:]
        with pytest.raises(KeyError, match="low"):
            compute_multi_features(bad, groups="trend_stats")

    def test_no_inf_in_output(self, markets):
        result = compute_multi_features(markets)
        values = result.values
        assert not np.isinf(values[~np.isnan(values)]).any()


# ---------------------------------------------------------------------------
# Custom params
# ---------------------------------------------------------------------------

class TestCustomParams:
    def test_shorter_lookback_reduces_warmup(self, markets):
        default = compute_multi_features(markets, groups="risk", prefix="")
        custom = compute_multi_features(
            markets, groups="coherence",
            params={"coherence": {"lookback": 20}}, prefix=""
        )
        # With lookback=20, first valid is bar 19; default=120, first valid=119
        assert np.isnan(default["coherence"].iloc[50])
        assert not np.isnan(custom["coherence"].iloc[50])

    def test_params_for_unselected_indicators_ignored(self, markets):
        result = compute_multi_features(
            markets, groups="risk",
            params={"trend_rank": {"period": 999}},
            prefix="",
        )
        assert "coherence" in result.columns
        assert "trend_rank" not in result.columns

    def test_prefix_override(self, markets):
        result = compute_multi_features(markets, groups="coherence", prefix="test_")
        assert "test_coherence" in result.columns


# ---------------------------------------------------------------------------
# Public API importable from top-level
# ---------------------------------------------------------------------------

class TestTopLevelImport:
    def test_compute_multi_features_importable_from_timothymasters(self):
        from okmich_quant_features.timothymasters import compute_multi_features as cmf  # noqa: F401
        assert callable(cmf)

    def test_compute_multi_features_importable_from_utils(self):
        from okmich_quant_features.timothymasters.utils import compute_multi_features as cmf  # noqa: F401
        assert callable(cmf)

    def test_multi_market_subpackage_importable(self):
        from okmich_quant_features.timothymasters import multi_market  # noqa: F401
        assert hasattr(multi_market, "trend_rank")
        assert hasattr(multi_market, "mahal")

    def test_trend_rank_importable_directly(self):
        from okmich_quant_features.timothymasters.multi_market import trend_rank  # noqa: F401
        assert callable(trend_rank)

    def test_mahal_importable_directly(self):
        from okmich_quant_features.timothymasters.multi_market import mahal  # noqa: F401
        assert callable(mahal)


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes
# ---------------------------------------------------------------------------

class TestJanusParamRoutingFix:
    """
    Tests for Fix #4: JANUS indicators grouped by unique core param tuple.

    Before the fix, all JANUS indicators in a single compute_multi_features
    call used the first indicator's lookback/spread_tail/min_cma/max_cma.
    Now each distinct combination creates its own Janus object.
    """

    @pytest.fixture
    def markets_5(self):
        return [_make_ohlc(seed=i * 5 + 1) for i in range(5)]

    def test_same_params_gives_correct_output(self, markets_5):
        """Single-param-group JANUS request should work as before."""
        result = compute_multi_features(markets_5, groups="janus",
                                        params={"janus_rs": {"market": 0, "lookback": 60}})
        assert f"mm_janus_rs" in result.columns
        valid = result["mm_janus_rs"].dropna()
        assert len(valid) > 0

    def test_distinct_lookbacks_give_distinct_outputs(self, markets_5):
        """Two JANUS indicators with different lookbacks must be computed independently."""
        result = compute_multi_features(
            markets_5,
            groups=["janus_rs", "janus_market_index"],
            params={
                "janus_rs": {"market": 0, "lookback": 60},
                "janus_market_index": {"lookback": 100},
            },
        )
        col_rs = result["mm_janus_rs"].dropna()
        col_mi = result["mm_janus_market_index"].dropna()
        assert len(col_rs) > 0
        assert len(col_mi) > 0
        # Different lookbacks → different warmup lengths → different valid ranges
        # Both should have valid values; can't be identical since params differ
        assert not col_rs.equals(col_mi)

    def test_janus_rs_with_custom_lookback_matches_direct_call(self, markets_5):
        """compute_multi_features with JANUS should match the direct wrapper."""
        from okmich_quant_features.timothymasters.multi_market.janus import janus_rs
        lookback = 80
        closes = [m["close"].to_numpy() for m in markets_5]

        direct = janus_rs(closes, market=0, lookback=lookback)
        result = compute_multi_features(
            markets_5,
            groups=["janus_rs"],
            params={"janus_rs": {"market": 0, "lookback": lookback}},
        )
        batch = result["mm_janus_rs"].to_numpy()
        np.testing.assert_array_equal(direct, batch)
