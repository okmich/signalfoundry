"""Tests for ruptures-based labeling module."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.diagnostics.ruptures import (
    CostModel,
    Algorithm,
    LabelMethod,
    RupturesConfig,
    ruptures_segment,
    ruptures_trend_labels,
    ruptures_volatility_labels,
    ruptures_multivariate_labels,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_trending_prices():
    """Create synthetic price series with clear trend changes."""
    np.random.seed(42)
    n = 500

    # Segment 1: Uptrend (0-150)
    p1 = 100 + np.cumsum(np.random.randn(150) * 0.5 + 0.3)

    # Segment 2: Downtrend (150-300)
    p2 = p1[-1] + np.cumsum(np.random.randn(150) * 0.5 - 0.3)

    # Segment 3: Sideways (300-400)
    p3 = p2[-1] + np.cumsum(np.random.randn(100) * 0.3)

    # Segment 4: Uptrend (400-500)
    p4 = p3[-1] + np.cumsum(np.random.randn(100) * 0.5 + 0.4)

    prices = np.concatenate([p1, p2, p3, p4])
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(prices, index=index, name="close")


@pytest.fixture
def synthetic_volatility_prices():
    """Create synthetic prices with volatility regime changes."""
    np.random.seed(42)
    n = 600

    # Low volatility (0-200)
    p1 = 100 + np.cumsum(np.random.randn(200) * 0.2)

    # High volatility (200-400)
    p2 = p1[-1] + np.cumsum(np.random.randn(200) * 1.5)

    # Medium volatility (400-600)
    p3 = p2[-1] + np.cumsum(np.random.randn(200) * 0.7)

    prices = np.concatenate([p1, p2, p3])
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(prices, index=index, name="close")


@pytest.fixture
def synthetic_multivariate_df():
    """Create DataFrame with multiple features for multivariate detection."""
    np.random.seed(42)
    n = 500
    index = pd.date_range("2024-01-01", periods=n, freq="h")

    # Create features with regime changes
    # Regime 1 (0-150): high returns, low vol, high volume
    # Regime 2 (150-300): low returns, high vol, low volume
    # Regime 3 (300-500): medium returns, medium vol, medium volume

    returns = np.concatenate([
        np.random.randn(150) * 0.5 + 0.3,
        np.random.randn(150) * 1.2 - 0.2,
        np.random.randn(200) * 0.7 + 0.05,
    ])

    volatility = np.concatenate([
        np.random.exponential(0.3, 150),
        np.random.exponential(1.2, 150),
        np.random.exponential(0.6, 200),
    ])

    volume = np.concatenate([
        np.random.exponential(1000, 150) + 500,
        np.random.exponential(400, 150) + 100,
        np.random.exponential(700, 200) + 300,
    ])

    return pd.DataFrame({
        "returns": returns,
        "volatility": volatility,
        "volume": volume,
    }, index=index)


# ============================================================================
# Test RupturesConfig
# ============================================================================

class TestRupturesConfig:
    def test_default_config(self):
        config = RupturesConfig()
        assert config.model == CostModel.RBF
        assert config.algo == Algorithm.PELT
        assert config.penalty == 10.0
        assert config.min_size == 5

    def test_custom_config(self):
        config = RupturesConfig(
            model=CostModel.LINEAR,
            algo=Algorithm.BINSEG,
            penalty=20.0,
            min_size=10,
        )
        assert config.model == CostModel.LINEAR
        assert config.algo == Algorithm.BINSEG
        assert config.penalty == 20.0

    def test_dynp_requires_n_bkps(self):
        with pytest.raises(ValueError, match="DYNP requires n_bkps"):
            RupturesConfig(algo=Algorithm.DYNP)

    def test_dynp_with_n_bkps(self):
        config = RupturesConfig(algo=Algorithm.DYNP, n_bkps=5)
        assert config.n_bkps == 5


# ============================================================================
# Test ruptures_segment
# ============================================================================

class TestRupturesSegment:
    def test_basic_segmentation(self, synthetic_trending_prices):
        breakpoints = ruptures_segment(synthetic_trending_prices)

        # Should return at least one breakpoint (the end)
        assert len(breakpoints) >= 1
        # Last breakpoint should be length of series
        assert breakpoints[-1] == len(synthetic_trending_prices)

    def test_segmentation_with_different_models(self, synthetic_trending_prices):
        for model in [CostModel.L1, CostModel.L2, CostModel.RBF, CostModel.LINEAR]:
            config = RupturesConfig(model=model, penalty=5.0)
            breakpoints = ruptures_segment(synthetic_trending_prices, config)
            assert len(breakpoints) >= 1
            assert breakpoints[-1] == len(synthetic_trending_prices)

    def test_segmentation_with_different_algos(self, synthetic_trending_prices):
        for algo in [Algorithm.PELT, Algorithm.BINSEG, Algorithm.BOTTOMUP]:
            config = RupturesConfig(algo=algo, penalty=5.0)
            breakpoints = ruptures_segment(synthetic_trending_prices, config)
            assert len(breakpoints) >= 1

    def test_dynp_segmentation(self, synthetic_trending_prices):
        config = RupturesConfig(algo=Algorithm.DYNP, n_bkps=4)
        breakpoints = ruptures_segment(synthetic_trending_prices, config)
        # DYNP should return exactly n_bkps + 1 breakpoints (including end)
        assert len(breakpoints) == 5

    def test_high_penalty_fewer_breakpoints(self, synthetic_trending_prices):
        low_pen = RupturesConfig(penalty=1.0)
        high_pen = RupturesConfig(penalty=100.0)

        bp_low = ruptures_segment(synthetic_trending_prices, low_pen)
        bp_high = ruptures_segment(synthetic_trending_prices, high_pen)

        # Higher penalty should result in fewer or equal breakpoints
        assert len(bp_high) <= len(bp_low)


# ============================================================================
# Test ruptures_trend_labels
# ============================================================================

class TestRupturesTrendLabels:
    def test_direction_labels(self, synthetic_trending_prices):
        labels, segments = ruptures_trend_labels(
            synthetic_trending_prices,
            label_method=LabelMethod.DIRECTION,
        )

        assert len(labels) == len(synthetic_trending_prices)
        assert len(segments) == len(synthetic_trending_prices)
        assert labels.name == "trend_label"
        assert segments.name == "segment_id"

        # Labels should be in {-1, 0, 1}
        unique_labels = set(labels.unique())
        assert unique_labels.issubset({-1, 0, 1})

    def test_slope_labels(self, synthetic_trending_prices):
        labels, segments = ruptures_trend_labels(
            synthetic_trending_prices,
            label_method=LabelMethod.SLOPE,
        )

        # Slope labels should be continuous
        assert labels.dtype == float

    def test_magnitude_labels(self, synthetic_trending_prices):
        labels, segments = ruptures_trend_labels(
            synthetic_trending_prices,
            label_method=LabelMethod.MAGNITUDE,
        )

        # Magnitude labels should be in {-2, -1, 0, 1, 2}
        unique_labels = set(labels.unique())
        assert unique_labels.issubset({-2, -1, 0, 1, 2})

    def test_neutral_threshold(self, synthetic_trending_prices):
        # With high threshold, more labels should be 0
        labels_low, _ = ruptures_trend_labels(
            synthetic_trending_prices,
            label_method=LabelMethod.DIRECTION,
            neutral_threshold=0.0,
        )
        labels_high, _ = ruptures_trend_labels(
            synthetic_trending_prices,
            label_method=LabelMethod.DIRECTION,
            neutral_threshold=0.01,
        )

        neutral_count_low = (labels_low == 0).sum()
        neutral_count_high = (labels_high == 0).sum()

        # Higher threshold should result in more neutral labels
        assert neutral_count_high >= neutral_count_low

    def test_index_preserved(self, synthetic_trending_prices):
        labels, segments = ruptures_trend_labels(synthetic_trending_prices)

        assert labels.index.equals(synthetic_trending_prices.index)
        assert segments.index.equals(synthetic_trending_prices.index)


# ============================================================================
# Test ruptures_volatility_labels
# ============================================================================

class TestRupturesVolatilityLabels:
    def test_basic_volatility_labels(self, synthetic_volatility_prices):
        labels, segments = ruptures_volatility_labels(
            synthetic_volatility_prices,
            n_regimes=3,
        )

        assert len(labels) == len(synthetic_volatility_prices)
        assert labels.name == "vol_regime"

        # Labels should be in {0, 1, 2} for 3 regimes
        unique_labels = set(labels.unique())
        assert unique_labels.issubset({0, 1, 2})

    def test_different_n_regimes(self, synthetic_volatility_prices):
        for n in [2, 3, 4, 5]:
            labels, _ = ruptures_volatility_labels(
                synthetic_volatility_prices,
                n_regimes=n,
            )
            unique_labels = set(labels.unique())
            assert all(0 <= lbl < n for lbl in unique_labels)

    def test_vol_window_parameter(self, synthetic_volatility_prices):
        labels_short, _ = ruptures_volatility_labels(
            synthetic_volatility_prices,
            vol_window=10,
        )
        labels_long, _ = ruptures_volatility_labels(
            synthetic_volatility_prices,
            vol_window=50,
        )

        # Both should have valid labels
        assert len(labels_short) == len(synthetic_volatility_prices)
        assert len(labels_long) == len(synthetic_volatility_prices)


# ============================================================================
# Test ruptures_multivariate_labels
# ============================================================================

class TestRupturesMultivariateLabels:
    def test_basic_multivariate(self, synthetic_multivariate_df):
        labels, segments, stats = ruptures_multivariate_labels(
            synthetic_multivariate_df,
            feature_cols=["returns", "volatility", "volume"],
            n_regimes=3,
        )

        assert len(labels) == len(synthetic_multivariate_df)
        assert labels.name == "regime_label"
        assert isinstance(stats, dict)

    def test_regime_stats_structure(self, synthetic_multivariate_df):
        labels, segments, stats = ruptures_multivariate_labels(
            synthetic_multivariate_df,
            feature_cols=["returns", "volatility"],
            n_regimes=3,
        )

        # Stats should have keys for each regime that exists
        for regime_id, regime_stats in stats.items():
            assert "returns" in regime_stats
            assert "volatility" in regime_stats
            assert "count" in regime_stats
            assert regime_stats["count"] > 0

    def test_missing_feature_col_raises(self, synthetic_multivariate_df):
        with pytest.raises(ValueError, match="Missing feature columns"):
            ruptures_multivariate_labels(
                synthetic_multivariate_df,
                feature_cols=["returns", "nonexistent_column"],
            )

    def test_single_feature(self, synthetic_multivariate_df):
        # Should work with single feature
        labels, segments, stats = ruptures_multivariate_labels(
            synthetic_multivariate_df,
            feature_cols=["returns"],
            n_regimes=3,
        )
        assert len(labels) == len(synthetic_multivariate_df)


# ============================================================================
# Test with NaN handling
# ============================================================================

class TestNaNHandling:
    def test_segment_with_nans(self):
        prices = pd.Series([100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109])
        breakpoints = ruptures_segment(prices)
        assert len(breakpoints) >= 1

    def test_trend_labels_with_nans(self):
        prices = pd.Series([100, 101, np.nan, 103, 104, np.nan, 106, 107, 108, 109])
        labels, _ = ruptures_trend_labels(prices)
        assert len(labels) == len(prices)
        assert not labels.isna().any()


# ============================================================================
# Test Enum values
# ============================================================================

class TestEnums:
    def test_cost_model_values(self):
        assert CostModel.L1.value == "l1"
        assert CostModel.L2.value == "l2"
        assert CostModel.RBF.value == "rbf"
        assert CostModel.LINEAR.value == "linear"
        assert CostModel.NORMAL.value == "normal"

    def test_algorithm_values(self):
        assert Algorithm.PELT.value == "pelt"
        assert Algorithm.BINSEG.value == "binseg"
        assert Algorithm.DYNP.value == "dynp"

    def test_label_method_values(self):
        assert LabelMethod.SLOPE.value == "slope"
        assert LabelMethod.DIRECTION.value == "direction"
        assert LabelMethod.MAGNITUDE.value == "magnitude"
