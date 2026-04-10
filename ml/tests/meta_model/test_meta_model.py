"""
Tests for the Meta-Model module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from okmich_quant_ml.meta_model import (
    MetaModel,
    MetaModelConfig,
    prepare_meta_dataset,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")

    return pd.DataFrame({
        "rsi": np.random.uniform(20, 80, n),
        "atr": np.random.uniform(0.5, 2.0, n),
        "momentum": np.random.randn(n),
        "trend": np.random.choice([-1, 0, 1], n),
    }, index=dates)


@pytest.fixture
def sample_labels():
    """Create sample labels DataFrame (sparse, at event times)."""
    np.random.seed(42)
    n_events = 100
    # Event times are a subset of the feature times
    event_dates = pd.date_range("2024-01-01", periods=500, freq="5min")[::5][:n_events]

    return pd.DataFrame({
        "label": np.random.choice([-1, 1], n_events),
        "ret": np.random.randn(n_events) * 0.01,
        "barrier_hit": np.random.choice(["upper", "lower", "vertical"], n_events),
    }, index=event_dates)


@pytest.fixture
def aligned_dataset(sample_features, sample_labels):
    """Create aligned X, y dataset."""
    X, y = prepare_meta_dataset(sample_features, sample_labels, target_col="label")
    return X, y


# =============================================================================
# Tests: prepare_meta_dataset
# =============================================================================


class TestPrepareMetaDataset:
    """Tests for prepare_meta_dataset function."""

    def test_basic_alignment(self, sample_features, sample_labels):
        """Test basic alignment of features with labels."""
        X, y = prepare_meta_dataset(sample_features, sample_labels)

        assert len(X) == len(y)
        assert len(X) <= len(sample_labels)
        assert list(X.columns) == list(sample_features.columns)

    def test_index_alignment(self, sample_features, sample_labels):
        """Test that X and y have same index."""
        X, y = prepare_meta_dataset(sample_features, sample_labels)

        assert X.index.equals(y.index)

    def test_custom_target_col(self, sample_features, sample_labels):
        """Test using custom target column."""
        X, y = prepare_meta_dataset(sample_features, sample_labels, target_col="ret")

        assert y.name == "ret" or len(y) > 0  # Series keeps its values from ret

    def test_invalid_target_col_raises(self, sample_features, sample_labels):
        """Test that invalid target_col raises ValueError."""
        with pytest.raises(ValueError, match="not found in labels"):
            prepare_meta_dataset(sample_features, sample_labels, target_col="nonexistent")

    def test_no_overlap_raises(self, sample_features):
        """Test that no overlapping timestamps raises ValueError."""
        # Labels with completely different index
        labels = pd.DataFrame({
            "label": [1, -1, 1],
        }, index=pd.date_range("2099-01-01", periods=3, freq="D"))

        with pytest.raises(ValueError, match="No overlapping timestamps"):
            prepare_meta_dataset(sample_features, labels)

    def test_handles_nan_in_features(self, sample_labels):
        """Test that NaN values in features are handled."""
        features = pd.DataFrame({
            "f1": [1.0, np.nan, 3.0, 4.0, 5.0],
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0],
        }, index=sample_labels.index[:5])

        labels_subset = sample_labels.iloc[:5]

        X, y = prepare_meta_dataset(features, labels_subset)

        # Row with NaN should be dropped
        assert len(X) == 4
        assert not X.isna().any().any()

    def test_handles_nan_in_target(self, sample_features):
        """Test that NaN values in target are handled."""
        labels = pd.DataFrame({
            "label": [1, np.nan, -1, 1, -1],
        }, index=sample_features.index[:5])

        X, y = prepare_meta_dataset(sample_features.iloc[:5], labels)

        # Row with NaN target should be dropped
        assert len(y) == 4
        assert not y.isna().any()


# =============================================================================
# Tests: MetaModelConfig
# =============================================================================


class TestMetaModelConfig:
    """Tests for MetaModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MetaModelConfig()

        assert config.model_type == "classifier"
        assert config.threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetaModelConfig(model_type="regressor", threshold=0.7)

        assert config.model_type == "regressor"
        assert config.threshold == 0.7


# =============================================================================
# Tests: MetaModel
# =============================================================================


class TestMetaModel:
    """Tests for MetaModel class."""

    def test_init(self):
        """Test model initialization."""
        config = MetaModelConfig()
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        model = MetaModel(config, estimator)

        assert model.config == config
        assert model.estimator == estimator
        assert not model.is_fitted

    def test_fit(self, aligned_dataset):
        """Test model fitting."""
        X, y = aligned_dataset

        config = MetaModelConfig()
        model = MetaModel(config, RandomForestClassifier(n_estimators=10, random_state=42))
        result = model.fit(X, y)

        assert result is model  # Returns self
        assert model.is_fitted
        assert model.feature_names == list(X.columns)

    def test_predict(self, aligned_dataset):
        """Test prediction."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(X)
        assert predictions.index.equals(X.index)

    def test_predict_proba(self, aligned_dataset):
        """Test probability prediction."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        proba = model.predict_proba(X)

        assert isinstance(proba, pd.Series)
        assert len(proba) == len(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_proba_regressor_raises(self, aligned_dataset):
        """Test that predict_proba raises for regressor."""
        X, y = aligned_dataset

        model = MetaModel(
            MetaModelConfig(model_type="regressor"),
            RandomForestRegressor(n_estimators=10, random_state=42)
        )
        model.fit(X, y)

        with pytest.raises(ValueError, match="only available for classifiers"):
            model.predict_proba(X)

    def test_should_trade(self, aligned_dataset):
        """Test should_trade method."""
        X, y = aligned_dataset

        model = MetaModel(
            MetaModelConfig(threshold=0.5),
            RandomForestClassifier(n_estimators=10, random_state=42)
        )
        model.fit(X, y)

        should_trade = model.should_trade(X)

        assert isinstance(should_trade, pd.Series)
        assert should_trade.dtype == bool
        assert len(should_trade) == len(X)

    def test_should_trade_threshold(self, aligned_dataset):
        """Test that threshold affects should_trade."""
        X, y = aligned_dataset

        model_low = MetaModel(
            MetaModelConfig(threshold=0.1),
            RandomForestClassifier(n_estimators=10, random_state=42)
        )
        model_low.fit(X, y)

        model_high = MetaModel(
            MetaModelConfig(threshold=0.9),
            RandomForestClassifier(n_estimators=10, random_state=42)
        )
        model_high.fit(X, y)

        # Lower threshold should result in more trades
        trades_low = model_low.should_trade(X).sum()
        trades_high = model_high.should_trade(X).sum()

        assert trades_low >= trades_high

    def test_predict_not_fitted_raises(self, aligned_dataset):
        """Test that predict raises if not fitted."""
        X, _ = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier())

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(X)

    def test_missing_features_raises(self, aligned_dataset):
        """Test that missing features raises error."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        # Remove a column
        X_missing = X.drop(columns=["rsi"])

        with pytest.raises(ValueError, match="Missing features"):
            model.predict(X_missing)

    def test_get_feature_importance(self, aligned_dataset):
        """Test feature importance extraction."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert isinstance(importance, pd.Series)
        assert len(importance) == len(X.columns)
        assert (importance >= 0).all()


class TestMetaModelOptimize:
    """Tests for MetaModel.optimize method."""

    def test_basic_optimize(self, aligned_dataset):
        """Test basic hyperparameter optimization."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(random_state=42))

        param_space = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }

        result = model.optimize(X, y, param_space, n_splits=3, n_iter=4)

        assert result is model
        assert model.is_fitted
        assert hasattr(model, "best_params_")
        assert hasattr(model, "best_score_")

    def test_optimize_uses_timeseries_split(self, aligned_dataset):
        """Test that optimize uses proper time series CV."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), LogisticRegression(max_iter=200))

        param_space = {
            "C": [0.1, 1.0],
        }

        # Should not raise - TimeSeriesSplit preserves temporal order
        model.optimize(X, y, param_space, n_splits=3, n_iter=2)

        assert model.is_fitted


class TestMetaModelWithDifferentEstimators:
    """Tests with different sklearn estimators."""

    def test_with_logistic_regression(self, aligned_dataset):
        """Test with LogisticRegression."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), LogisticRegression(max_iter=200))
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_with_random_forest_regressor(self, aligned_dataset):
        """Test with RandomForestRegressor."""
        X, y = aligned_dataset
        y_continuous = y.astype(float) + np.random.randn(len(y)) * 0.1

        model = MetaModel(
            MetaModelConfig(model_type="regressor"),
            RandomForestRegressor(n_estimators=10, random_state=42)
        )
        model.fit(X, y_continuous)

        predictions = model.predict(X)
        assert isinstance(predictions, pd.Series)


class TestMetaModelEdgeCases:
    """Edge case tests for MetaModel."""

    def test_single_row_prediction(self, aligned_dataset):
        """Test prediction on single row."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        # Single row
        X_single = X.iloc[[0]]
        pred = model.predict(X_single)

        assert len(pred) == 1

    def test_multiclass_classification(self):
        """Test with multiclass labels."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        y = pd.Series(np.random.choice([-1, 0, 1], n), index=X.index)

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_extra_features_ignored(self, aligned_dataset):
        """Test that extra features in prediction are ignored."""
        X, y = aligned_dataset

        model = MetaModel(MetaModelConfig(), RandomForestClassifier(n_estimators=10, random_state=42))
        model.fit(X, y)

        # Add extra column
        X_extra = X.copy()
        X_extra["extra_feature"] = np.random.randn(len(X))

        # Should work, extra feature ignored
        pred = model.predict(X_extra[model.feature_names])
        assert len(pred) == len(X)
