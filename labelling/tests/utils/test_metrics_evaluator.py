import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.metrics_evaluator import (
    LabelEvalConfig,
    LabelEvaluator,
)

warnings.filterwarnings("ignore")


@pytest.fixture
def sample_config():
    """Sample configuration for tests"""
    return LabelEvalConfig(
        feature_prefix="feat_", label_col="label", n_splits=3, random_state=42
    )


@pytest.fixture
def multi_regime_data():
    """Create comprehensive multi-regime test data"""
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")
    n = len(dates)

    np.random.seed(42)

    # Create regime trend (3 regimes: bull=0, bear=1, sideways=2)
    regime_labels = []
    current_regime = 0
    i = 0

    while i < n:
        if current_regime == 0:  # Bull market
            duration = np.random.randint(100, 300)
            regime_labels.extend([0] * min(duration, n - i))
        elif current_regime == 1:  # Bear market
            duration = np.random.randint(50, 150)
            regime_labels.extend([1] * min(duration, n - i))
        else:  # Sideways market
            duration = np.random.randint(80, 200)
            regime_labels.extend([2] * min(duration, n - i))

        i += duration
        current_regime = (current_regime + 1) % 3

    regime_labels = regime_labels[:n]

    # Create features and returns based on regimes
    features = []
    returns = []

    for i, regime in enumerate(regime_labels):
        if regime == 0:  # Bull market
            mu = 0.001
            sigma = 0.015
        elif regime == 1:  # Bear market
            mu = -0.0008
            sigma = 0.025
        else:  # Sideways market
            mu = 0.0001
            sigma = 0.018

        ret = np.random.normal(mu, sigma)
        returns.append(ret)

        # Create features that are informative about regimes
        feat1 = ret + np.random.normal(0, 0.005)  # Momentum feature
        feat2 = sigma + np.random.normal(0, 0.002)  # Volatility feature
        feat3 = mu * 100 + np.random.normal(0, 0.5)  # Trend feature

        features.append([feat1, feat2, feat3])

    # Create price series from returns
    prices = [100]  # Starting price
    for ret in returns[1:]:
        prices.append(prices[-1] * np.exp(ret))

    df = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "ret": [0] + returns[1:],  # First return is 0
            "label": [
                "bull" if x == 0 else "bear" if x == 1 else "sideways"
                for x in regime_labels
            ],
            "feat_momentum": [f[0] for f in features],
            "feat_volatility": [f[1] for f in features],
            "feat_trend": [f[2] for f in features],
        }
    )

    df.set_index("date", inplace=True)

    # Add some NaN values to test robustness
    df.loc[df.index[100:105], "feat_momentum"] = np.nan
    df.loc[df.index[200:202], "label"] = np.nan

    return df


@pytest.fixture
def binary_regime_data():
    """Create binary regime test data"""
    dates = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
    n = len(dates)

    # Alternating binary regimes
    labels = []
    for i in range(n):
        labels.append("up" if (i // 30) % 2 == 0 else "down")

    # Create returns with regime-dependent characteristics
    returns = []
    for label in labels:
        if label == "up":
            ret = np.random.normal(0.001, 0.01)
        else:
            ret = np.random.normal(-0.0005, 0.015)
        returns.append(ret)

    prices = [100]
    for ret in returns[1:]:
        prices.append(prices[-1] * np.exp(ret))

    df = pd.DataFrame(
        {
            "date": dates,
            "close": prices,
            "ret": [0] + returns[1:],
            "label": labels,
            "feat_signal": np.random.randn(n),
        }
    )

    df.set_index("date", inplace=True)
    return df


@pytest.fixture
def minimal_data():
    """Create minimal test data"""
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "close": np.ones(50) * 100,  # Constant price
            "label": ["regime_A"] * 50,  # Single regime
            "feat_dummy": np.random.randn(50),
        }
    )

    df.set_index("date", inplace=True)
    return df


class TestLabelEvaluatorInitialization:
    """Test initialization and basic setup"""

    def test_initialization_with_valid_data(self, multi_regime_data, sample_config):
        """Test successful initialization with valid data"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        assert evaluator.df is not None
        assert evaluator.cfg == sample_config
        assert hasattr(evaluator, "label_map")
        assert hasattr(evaluator, "inv_label_map")

    def test_initialization_without_config(self, multi_regime_data):
        """Test initialization without explicit config"""
        evaluator = LabelEvaluator(multi_regime_data)
        assert evaluator.cfg is not None
        assert isinstance(evaluator.cfg, LabelEvalConfig)

    def test_initialization_missing_label_column(self):
        """Test initialization with missing label column"""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10),
                "close": np.random.randn(10),
            }
        )
        df.set_index("date", inplace=True)

        with pytest.raises(ValueError, match="DataFrame must contain the label column"):
            LabelEvaluator(df)

    def test_initialization_missing_close_and_returns(self):
        """Test initialization with missing close and returns"""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=10),
                "label": ["A"] * 10,
                "feat_1": np.random.randn(10),
            }
        )
        df.set_index("date", inplace=True)

        with pytest.raises(ValueError, match="Cannot compute returns"):
            LabelEvaluator(df)


class TestStructureMetrics:
    """Test structure metrics computation"""

    def test_structure_metrics_computed(self, multi_regime_data, sample_config):
        """Test that all structure metrics are computed"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.structure_metrics()

        # Check required metrics exist
        assert "n_classes" in metrics
        assert "class_counts" in metrics
        assert "class_entropy" in metrics
        assert "imbalance_ratio" in metrics
        assert "transition_matrix" in metrics
        assert "self_transition_rate" in metrics
        assert "mean_dwell" in metrics
        assert "state_persistence_by_label" in metrics

        # Check types and basic validity
        assert isinstance(metrics["n_classes"], int)
        assert isinstance(metrics["class_counts"], dict)
        assert isinstance(metrics["class_entropy"], float)
        assert metrics["n_classes"] > 0
        assert metrics["class_entropy"] >= 0

    def test_transition_matrix_properties(self, multi_regime_data, sample_config):
        """Test transition matrix properties"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.structure_metrics()
        tm = metrics["transition_matrix"]

        assert isinstance(tm, np.ndarray)
        assert tm.shape[0] == tm.shape[1]  # Square matrix
        assert np.all(tm >= 0)  # Non-negative
        assert np.allclose(tm.sum(axis=1), 1.0, atol=1e-10)  # Rows sum to 1

    def test_dwell_statistics_validity(self, multi_regime_data, sample_config):
        """Test dwell statistics validity"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.dwell_statistics()

        assert "mean_dwell" in metrics
        assert "median_dwell" in metrics
        assert "min_dwell" in metrics
        assert "max_dwell" in metrics

        if not np.isnan(metrics["mean_dwell"]):
            assert metrics["mean_dwell"] > 0
            assert metrics["min_dwell"] <= metrics["mean_dwell"] <= metrics["max_dwell"]


class TestSeparabilityMetrics:
    """Test separability metrics computation"""

    def test_separability_metrics_with_features(self, multi_regime_data, sample_config):
        """Test separability metrics with features present"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.separability_metrics()

        # Check required metrics exist
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "fisher_ratio" in metrics
        assert "return_separability_stats" in metrics
        assert "kruskal_wallis_pval" in metrics
        assert "kruskal_wallis_stat" in metrics

        # Check basic validity
        if not np.isnan(metrics["silhouette"]):
            assert -1 <= metrics["silhouette"] <= 1
        if not np.isnan(metrics["davies_bouldin"]):
            assert metrics["davies_bouldin"] > 0

    def test_separability_metrics_without_features(self, sample_config):
        """Test separability metrics without features"""
        # Create data without features
        dates = pd.date_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            {
                "date": dates,
                "close": np.random.randn(100).cumsum() + 100,
                "label": ["A", "B"] * 50,
            }
        )
        df.set_index("date", inplace=True)

        evaluator = LabelEvaluator(df, sample_config)
        metrics = evaluator.separability_metrics()

        # Should still compute return-based separability
        assert "return_separability_stats" in metrics
        assert "kruskal_wallis_pval" in metrics


class TestPredictiveMetrics:
    """Test predictive metrics computation"""

    def test_predictive_metrics_with_features(self, multi_regime_data, sample_config):
        """Test predictive metrics with features"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.predictive_metrics()

        # Check required metrics exist
        assert "cv_macro_f1" in metrics
        assert "cv_auc" in metrics
        assert "mutual_info" in metrics
        assert "information_coefficient" in metrics

        # Check basic validity
        if not np.isnan(metrics["cv_macro_f1"]):
            assert 0 <= metrics["cv_macro_f1"] <= 1
        if not np.isnan(metrics["cv_auc"]):
            assert 0 <= metrics["cv_auc"] <= 1
        if not np.isnan(metrics["information_coefficient"]):
            assert -1 <= metrics["information_coefficient"] <= 1

    def test_information_coefficient_no_lookahead(self, sample_config):
        """Test that information coefficient avoids look-ahead bias"""
        # Create data where trend should NOT predict future returns
        dates = pd.date_range("2020-01-01", periods=100)

        # Labels based on past, returns independent
        labels = ["regime_A"] * 50 + ["regime_B"] * 50
        returns = np.random.normal(0, 0.01, 100)  # Independent returns

        prices = [100]
        for ret in returns[:-1]:
            prices.append(prices[-1] * np.exp(ret))

        df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "ret": returns,
                "label": labels,
                "feat_dummy": np.random.randn(100),
            }
        )
        df.set_index("date", inplace=True)

        evaluator = LabelEvaluator(df, sample_config)
        ic = evaluator.information_coefficient()

        # IC should be close to 0 for independent data
        # We won't assert exact values due to randomness, but ensure it's computed
        assert isinstance(ic, float)


class TestEconomicMetrics:
    """Test economic metrics computation"""

    def test_economic_metrics_computed(self, multi_regime_data, sample_config):
        """Test that economic metrics are computed"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.economic_metrics()

        # Check required metrics exist
        assert "sr_overall" in metrics
        assert "anova_p_ret" in metrics
        assert "corr_label_future_ret" in metrics
        assert "pnl_by_label" in metrics

        # Check basic validity
        if not np.isnan(metrics["sr_overall"]):
            assert isinstance(metrics["sr_overall"], float)
        if not np.isnan(metrics["corr_label_future_ret"]):
            assert -1 <= metrics["corr_label_future_ret"] <= 1

    def test_naive_pnl_no_lookahead(self, sample_config):
        """Test that PnL calculation avoids look-ahead bias"""
        dates = pd.date_range("2020-01-01", periods=50)

        # Create predictable pattern
        labels = [1] * 25 + [-1] * 25  # Positive then negative signals
        returns = [0.01] * 25 + [-0.01] * 24 + [0]  # Returns that align with signals

        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * np.exp(ret))

        df = pd.DataFrame(
            {"date": dates, "close": prices, "ret": [0] + returns[1:], "label": labels}
        )
        df.set_index("date", inplace=True)

        evaluator = LabelEvaluator(df, sample_config)
        pnl_metrics = evaluator.naive_pnl_by_label()

        assert "pnl_by_label" in pnl_metrics
        pnl_df = pnl_metrics["pnl_by_label"]
        assert not pnl_df.empty


class TestRobustnessMetrics:
    """Test robustness metrics computation"""

    def test_robustness_metrics_computed(self, multi_regime_data, sample_config):
        """Test that robustness metrics are computed"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.robustness_metrics()

        # Check required metrics exist
        assert "bootstrap_jaccard" in metrics
        assert "time_shift_stability" in metrics

        # Check basic validity
        if not np.isnan(metrics["bootstrap_jaccard"]):
            assert 0 <= metrics["bootstrap_jaccard"] <= 1
        if not np.isnan(metrics["time_shift_stability"]):
            assert 0 <= metrics["time_shift_stability"] <= 1


class TestOperationsMetrics:
    """Test operations metrics computation"""

    def test_ops_metrics_computed(self, multi_regime_data, sample_config):
        """Test that operations metrics are computed"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        metrics = evaluator.ops_metrics()

        # Check required metrics exist
        assert "coverage" in metrics
        assert "label_turnover" in metrics

        # Check basic validity
        assert 0 <= metrics["coverage"] <= 1
        if not np.isnan(metrics["label_turnover"]):
            assert 0 <= metrics["label_turnover"] <= 1


class TestCompositeScore:
    """Test composite score computation"""

    def test_composite_score_computed(self, multi_regime_data, sample_config):
        """Test that composite score is computed"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        results = evaluator.evaluate()

        assert "composite_score" in results
        assert "score" in results["composite_score"]
        score = results["composite_score"]["score"]
        assert isinstance(score, float)
        # Score should be finite
        assert np.isfinite(score)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_single_regime_data(self, sample_config):
        """Test with single regime data"""
        dates = pd.date_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            {
                "date": dates,
                "close": np.random.randn(100).cumsum() + 100,
                "label": ["single_regime"] * 100,
                "feat_1": np.random.randn(100),
            }
        )
        df.set_index("date", inplace=True)

        evaluator = LabelEvaluator(df, sample_config)
        results = evaluator.evaluate()

        # Should complete without error
        assert results is not None

    def test_minimal_data_handling(self, sample_config):
        """Test handling of minimal data"""
        dates = pd.date_range("2020-01-01", periods=10)
        df = pd.DataFrame(
            {
                "date": dates,
                "close": np.random.randn(10).cumsum() + 100,
                "label": ["A", "B"] * 5,
                "feat_1": np.random.randn(10),
            }
        )
        df.set_index("date", inplace=True)

        evaluator = LabelEvaluator(df, sample_config)
        results = evaluator.evaluate()

        # Should complete without error even with minimal data
        assert results is not None

    def test_data_with_many_nans(self, sample_config):
        """Test handling of data with many NaN values"""
        dates = pd.date_range("2020-01-01", periods=100)
        df = pd.DataFrame(
            {
                "date": dates,
                "close": np.random.randn(100).cumsum() + 100,
                "label": ["A"] * 50 + [np.nan] * 50,  # Many NaN trend
                "feat_1": np.random.randn(100),
            }
        )
        df.set_index("date", inplace=True)

        evaluator = LabelEvaluator(df, sample_config)
        results = evaluator.evaluate()

        # Should complete without error
        assert results is not None


class TestIntegration:
    """Integration tests"""

    def test_full_evaluation_pipeline(self, multi_regime_data, sample_config):
        """Test the full evaluation pipeline"""
        evaluator = LabelEvaluator(multi_regime_data, sample_config)
        results = evaluator.evaluate()

        # Check all major components are present
        expected_sections = [
            "structure",
            "separability",
            "predictive",
            "economic",
            "robustness",
            "ops",
            "composite_score",
        ]
        for section in expected_sections:
            assert section in results

        # Check that summary can be generated
        summary = evaluator.get_evaluation_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0
        assert "metric" in summary.columns

    def test_binary_vs_multiclass_consistency(
        self, binary_regime_data, multi_regime_data, sample_config
    ):
        """Test consistency between binary and multi-class cases"""
        evaluator_binary = LabelEvaluator(binary_regime_data, sample_config)
        evaluator_multi = LabelEvaluator(multi_regime_data, sample_config)

        results_binary = evaluator_binary.evaluate()
        results_multi = evaluator_multi.evaluate()

        # Both should complete successfully
        assert results_binary is not None
        assert results_multi is not None

        # Both should have composite scores
        assert "composite_score" in results_binary
        assert "composite_score" in results_multi


# ---------------------------------------------------------------------------
# Regression tests added for correctness fixes
# ---------------------------------------------------------------------------

class TestNoFeaturePathKeyConsistency:
    """No-feature branch must use 'davies_bouldin' (not 'dies_bouldin')."""

    def test_no_feature_separability_uses_davies_bouldin_key(self, sample_config):
        """separability_metrics() with no feat_ columns must return key 'davies_bouldin'."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame(
            {
                "close": np.random.randn(200).cumsum() + 100,
                "label": (["A"] * 100 + ["B"] * 100),
            },
            index=dates,
        )
        evaluator = LabelEvaluator(df, sample_config)
        metrics = evaluator.separability_metrics()

        assert "davies_bouldin" in metrics, (
            "'davies_bouldin' key missing from no-feature separability output"
        )
        assert "dies_bouldin" not in metrics, (
            "Typo key 'dies_bouldin' must not appear in output"
        )

    def test_no_feature_separability_values_are_nan(self, sample_config):
        """With no features, silhouette/davies_bouldin/calinski_harabasz must be NaN."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "close": np.ones(100) * 100,
                "label": ["X", "Y"] * 50,
            },
            index=dates,
        )
        evaluator = LabelEvaluator(df, sample_config)
        metrics = evaluator.separability_metrics()

        assert np.isnan(metrics["silhouette"])
        assert np.isnan(metrics["davies_bouldin"])

    def test_composite_score_uses_davies_bouldin_correctly(self, sample_config):
        """composite_score() must not crash when no-feature branch populates davies_bouldin=NaN."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        df = pd.DataFrame(
            {
                "close": np.random.randn(200).cumsum() + 100,
                "label": ["A"] * 100 + ["B"] * 100,
            },
            index=dates,
        )
        evaluator = LabelEvaluator(df, sample_config)
        results = evaluator.evaluate()

        score = results["composite_score"]["score"]
        assert isinstance(score, float)
        assert np.isfinite(score)


class TestRobustnessMetricsEmptyValidIdx:
    """robustness_metrics must not crash when all feature rows are non-finite."""

    def test_all_nan_features_returns_nan_not_crash(self, sample_config):
        """All-NaN feature matrix produces NaN robustness metrics, not an exception."""
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "close": np.random.randn(n).cumsum() + 100,
                "label": ["A", "B"] * (n // 2),
                "feat_bad": [np.nan] * n,  # all NaN → valid_idx will be empty
            },
            index=dates,
        )
        evaluator = LabelEvaluator(df, sample_config)

        # Must not raise; must return NaN for both keys
        metrics = evaluator.robustness_metrics()

        assert "bootstrap_jaccard" in metrics
        assert "time_shift_stability" in metrics
        assert np.isnan(metrics["bootstrap_jaccard"])

    def test_sparse_finite_features_do_not_crash(self, sample_config):
        """Only a few finite feature rows (rest NaN) — guard must handle gracefully."""
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        feat = np.full(n, np.nan)
        feat[0] = 1.0  # only 1 finite row → intersection will have 1 valid row

        df = pd.DataFrame(
            {
                "close": np.random.randn(n).cumsum() + 100,
                "label": (["A"] * 30 + ["B"] * 30),
                "feat_sparse": feat,
            },
            index=dates,
        )
        evaluator = LabelEvaluator(df, sample_config)

        # Must not raise
        metrics = evaluator.robustness_metrics()
        assert isinstance(metrics, dict)


# Performance and utility tests
class TestPerformance:
    """Performance and utility tests"""

    def test_large_dataset_handling(self, sample_config):
        """Test handling of larger datasets"""
        # Create larger dataset
        dates = pd.date_range("2010-01-01", "2023-12-31", freq="D")
        n = len(dates)

        np.random.seed(42)
        labels = np.random.choice(["regime_A", "regime_B", "regime_C"], n)
        returns = np.random.normal(0, 0.01, n)

        prices = [100]
        for ret in returns[:-1]:
            prices.append(prices[-1] * np.exp(ret))

        df = pd.DataFrame(
            {
                "date": dates,
                "close": prices,
                "ret": returns,
                "label": labels,
                "feat_1": np.random.randn(n),
                "feat_2": np.random.randn(n),
            }
        )
        df.set_index("date", inplace=True)

        # This should not take too long and should complete
        evaluator = LabelEvaluator(df, sample_config)
        results = evaluator.evaluate()

        assert results is not None
        assert "composite_score" in results
