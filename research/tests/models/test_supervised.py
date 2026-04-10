"""
Tests for supervised learning components.

Covers:
- SupervisedTrainer (sklearn and keras walk-forward training)
- SupervisedEvaluator (classification and regression evaluation)
- ConfigParser (supervised config parsing)
- Integration with ExperimentRunner
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from okmich_quant_research.models.supervised_trainer import (
    SupervisedTrainer,
    SupervisedTrainedModel,
    SupervisedModelMetadata,
    WalkForwardResult,
)
from okmich_quant_research.models.supervised_evaluator import (
    SupervisedEvaluator,
    SupervisedEvaluationResult,
)
from okmich_quant_research.models.utils.config_parser import ConfigParser


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_features():
    """Generate sample features DataFrame."""
    np.random.seed(42)
    n_samples = 2000
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="5min")

    return pd.DataFrame(
        {
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "feature_5": np.random.randn(n_samples),
        },
        index=dates,
    )


@pytest.fixture
def classification_target(sample_features):
    """Generate binary classification target."""
    np.random.seed(42)
    return pd.Series(
        np.random.randint(0, 2, len(sample_features)),
        index=sample_features.index,
        name="target",
    )


@pytest.fixture
def regression_target(sample_features):
    """Generate regression target."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(len(sample_features)) * 0.01,
        index=sample_features.index,
        name="target",
    )


@pytest.fixture
def sklearn_model_configs():
    """Sample sklearn model configurations."""
    return [
        {
            "name": "random_forest",
            "algorithm": "RandomForestClassifier",
            "hyperparameters": {"n_estimators": [10, 20], "max_depth": [3, 5]},
        },
        {
            "name": "logistic",
            "algorithm": "LogisticRegression",
            "hyperparameters": {"C": [0.1, 1.0], "max_iter": [100]},
        },
    ]


@pytest.fixture
def sklearn_regressor_configs():
    """Sample sklearn regressor configurations."""
    return [
        {
            "name": "ridge",
            "algorithm": "Ridge",
            "hyperparameters": {"alpha": [0.1, 1.0]},
        },
    ]


@pytest.fixture
def supervised_classification_config():
    """Sample supervised classification YAML config."""
    return {
        "experiment_name": "test_supervised_classification",
        "research_type": "supervised",
        "data": {"symbol": "TEST", "timeframe": "5min"},
        "feature_engineering": {
            "external_function": {
                "module": "test_module",
                "function": "test_features",
                "params": {"lookback": 20},
            },
            "version": "test_v1",
        },
        "target_engineering": {
            "external_function": {
                "module": "test_module",
                "function": "create_target",
                "params": {"horizon": 10},
            },
            "target_column": "direction",
            "task_type": "classification",
        },
        "model": {
            "type": "supervised_classification",
            "supervised": {
                "framework": "sklearn",
                "sklearn_models": [
                    {
                        "name": "rf",
                        "algorithm": "RandomForestClassifier",
                        "hyperparameters": {"n_estimators": [10]},
                    }
                ],
            },
        },
        "walk_forward": {
            "train_period": 500,
            "test_period": 100,
            "step_period": 50,
            "anchored": False,
        },
        "objectives": [
            {"name": "accuracy", "target": "maximize", "weight": 0.5},
            {"name": "f1_score", "target": "maximize", "weight": 0.5},
        ],
        "output": {"folder": None},
    }


@pytest.fixture
def keras_classification_config():
    """Sample Keras classification config."""
    return {
        "experiment_name": "test_keras_classification",
        "research_type": "supervised",
        "data": {"symbol": "TEST", "timeframe": "5min"},
        "feature_engineering": {
            "external_function": {
                "module": "test_module",
                "function": "test_features",
            },
        },
        "target_engineering": {
            "external_function": {
                "module": "test_module",
                "function": "create_target",
            },
            "target_column": "direction",
            "task_type": "classification",
        },
        "model": {
            "type": "supervised_classification",
            "supervised": {
                "framework": "keras",
                "sequence_length": 32,
                "keras_model_builder": {
                    "module": "test_module",
                    "function": "build_model",
                    "params": {
                        "num_features": "__num_features__",
                        "num_classes": "__num_classes__",
                    },
                },
                "tuner_params": {"objective": "val_accuracy", "max_epochs": 5},
                "training_params": {"epochs": 5, "batch_size": 32},
            },
        },
        "walk_forward": {"train_period": 500, "test_period": 100, "step_period": 50},
        "objectives": [{"name": "accuracy", "target": "maximize", "weight": 1.0}],
    }


# =============================================================================
# CONFIG PARSER TESTS
# =============================================================================


class TestConfigParserSupervised:
    """Tests for supervised config parsing."""

    def test_supervised_validation(self, supervised_classification_config):
        """Test that supervised config validates correctly."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        assert parser.get_model_type() == "supervised_classification"

    def test_missing_target_engineering_raises(self):
        """Test that missing target_engineering raises for supervised."""
        config = {
            "experiment_name": "test",
            "data": {"symbol": "TEST", "timeframe": "5min"},
            "model": {"type": "supervised_classification"},
            "objectives": [{"name": "accuracy", "weight": 1.0}],
            # Missing target_engineering section
        }
        with pytest.raises(ValueError, match="target_engineering"):
            ConfigParser(config_dict=config)

    def test_get_target_engineering_config(self, supervised_classification_config):
        """Test target engineering config retrieval."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        te_config = parser.get_target_engineering_config()

        assert "external_function" in te_config
        assert te_config["target_column"] == "direction"
        assert te_config["task_type"] == "classification"

    def test_get_target_external_function_config(self, supervised_classification_config):
        """Test target external function config."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        fn_config = parser.get_target_external_function_config()

        assert fn_config["module"] == "test_module"
        assert fn_config["function"] == "create_target"
        assert fn_config["params"]["horizon"] == 10

    def test_get_walk_forward_config(self, supervised_classification_config):
        """Test walk-forward config retrieval."""
        parser = ConfigParser(config_dict=supervised_classification_config)

        assert parser.get_train_period() == 500
        assert parser.get_test_period() == 100
        assert parser.get_step_period() == 50
        assert parser.is_anchored_walk_forward() is False

    def test_get_model_framework(self, supervised_classification_config):
        """Test model framework retrieval."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        assert parser.get_model_framework() == "sklearn"

    def test_get_sklearn_models(self, supervised_classification_config):
        """Test sklearn models config retrieval."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        models = parser.get_sklearn_models()

        assert len(models) == 1
        assert models[0]["name"] == "rf"
        assert models[0]["algorithm"] == "RandomForestClassifier"

    def test_get_sequence_length(self, keras_classification_config):
        """Test sequence length retrieval for Keras."""
        parser = ConfigParser(config_dict=keras_classification_config)
        assert parser.get_sequence_length() == 32

    def test_get_keras_model_builder(self, keras_classification_config):
        """Test Keras model builder config."""
        parser = ConfigParser(config_dict=keras_classification_config)
        builder = parser.get_keras_model_builder()

        assert builder["module"] == "test_module"
        assert builder["function"] == "build_model"
        assert builder["params"]["num_features"] == "__num_features__"

    def test_get_supervised_metrics_classification(self, supervised_classification_config):
        """Test default classification metrics."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        metrics = parser.get_supervised_metrics()

        assert "accuracy" in metrics
        assert "f1_score" in metrics

    def test_get_task_type(self, supervised_classification_config):
        """Test task type retrieval."""
        parser = ConfigParser(config_dict=supervised_classification_config)
        assert parser.get_task_type() == "classification"


# =============================================================================
# SUPERVISED TRAINER TESTS
# =============================================================================


class TestSupervisedTrainer:
    """Tests for SupervisedTrainer class."""

    def test_trainer_initialization(self):
        """Test trainer initializes correctly."""
        trainer = SupervisedTrainer(random_state=42, scale_features=True)
        assert trainer.random_state == 42
        assert trainer.scale_features is True

    def test_calculate_windows_rolling(self):
        """Test window calculation for rolling walk-forward."""
        trainer = SupervisedTrainer()
        windows = trainer._calculate_windows(
            total_length=1000,
            train_period=200,
            test_period=50,
            step_period=50,
            anchored=False,
        )

        assert len(windows) > 0
        # First window
        assert windows[0] == (0, 200, 200, 250)
        # Windows should not exceed total length
        for _, _, _, test_end in windows:
            assert test_end <= 1000

    def test_calculate_windows_anchored(self):
        """Test window calculation for anchored walk-forward."""
        trainer = SupervisedTrainer()
        windows = trainer._calculate_windows(
            total_length=1000,
            train_period=200,
            test_period=50,
            step_period=50,
            anchored=True,
        )

        assert len(windows) > 0
        # All windows should start at 0
        for train_start, _, _, _ in windows:
            assert train_start == 0
        # Train end should grow
        assert windows[1][1] > windows[0][1]

    def test_sklearn_classification_walk_forward(
        self, sample_features, classification_target, sklearn_model_configs
    ):
        """Test sklearn classification with walk-forward."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=sklearn_model_configs,
            task_type="classification",
            framework="sklearn",
        )

        assert len(trained_models) == 2
        for model in trained_models:
            assert isinstance(model, SupervisedTrainedModel)
            assert model.metadata.task_type == "classification"
            assert len(model.walk_forward_results) > 0
            assert "accuracy" in model.aggregate_metrics

    def test_sklearn_regression_walk_forward(
        self, sample_features, regression_target, sklearn_regressor_configs
    ):
        """Test sklearn regression with walk-forward."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=regression_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=sklearn_regressor_configs,
            task_type="regression",
            framework="sklearn",
        )

        assert len(trained_models) == 1
        model = trained_models[0]
        assert model.metadata.task_type == "regression"
        assert "mse" in model.aggregate_metrics
        assert "r2" in model.aggregate_metrics

    def test_create_sequences(self, sample_features, classification_target):
        """Test sequence creation for RNN."""
        trainer = SupervisedTrainer()
        features = sample_features.values[:100]
        targets = classification_target.values[:100]

        X, y = trainer._create_sequences(features, targets, sequence_length=10)

        assert X.shape == (90, 10, 5)  # n_sequences, seq_len, n_features
        assert y.shape == (90,)

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        trainer = SupervisedTrainer()
        features = np.random.randn(5, 3)
        targets = np.random.randint(0, 2, 5)

        with pytest.raises(ValueError, match="Not enough samples"):
            trainer._create_sequences(features, targets, sequence_length=10)


class TestSupervisedTrainedModel:
    """Tests for SupervisedTrainedModel class."""

    def test_model_save_and_load(
        self, sample_features, classification_target, sklearn_model_configs
    ):
        """Test model save and load functionality."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=500,  # Single window for speed
            model_configs=[sklearn_model_configs[0]],
            task_type="classification",
            framework="sklearn",
        )

        model = trained_models[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            model.save(tmpdir)

            # Check files exist
            assert Path(tmpdir, f"{model.metadata.model_name}.joblib").exists()
            assert Path(tmpdir, f"{model.metadata.model_name}_metadata.json").exists()

            # Load
            loaded = SupervisedTrainedModel.load(model.metadata.model_name, tmpdir)

            assert loaded.metadata.model_name == model.metadata.model_name
            assert loaded.metadata.algorithm == model.metadata.algorithm

    def test_predict(self, sample_features, classification_target, sklearn_model_configs):
        """Test prediction method."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=500,
            model_configs=[sklearn_model_configs[0]],
            task_type="classification",
            framework="sklearn",
        )

        model = trained_models[0]
        X_test = sample_features.iloc[-10:].values

        predictions = model.predict(X_test)
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)


class TestWalkForwardResult:
    """Tests for WalkForwardResult class."""

    def test_to_dict(self):
        """Test WalkForwardResult serialization."""
        result = WalkForwardResult(
            window_idx=0,
            train_start="2024-01-01",
            train_end="2024-01-31",
            test_start="2024-02-01",
            test_end="2024-02-10",
            model_name="test_model",
            model=None,
            scaler=None,
            predictions=np.array([0, 1, 1, 0]),
            true_labels=np.array([0, 1, 0, 0]),
            probabilities=None,
            metrics={"accuracy": 0.75, "f1_score": 0.67},
            hyperparameters={"n_estimators": 100},
            feature_names=["f1", "f2"],
            n_train_samples=100,
            n_test_samples=20,
        )

        d = result.to_dict()
        assert d["window_idx"] == 0
        assert d["model_name"] == "test_model"
        assert d["metrics"]["accuracy"] == 0.75
        assert "model" not in d  # Model object should not be serialized


# =============================================================================
# SUPERVISED EVALUATOR TESTS
# =============================================================================


class TestSupervisedEvaluator:
    """Tests for SupervisedEvaluator class."""

    def test_evaluate_classification(
        self, sample_features, classification_target, sklearn_model_configs
    ):
        """Test classification evaluation."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=[sklearn_model_configs[0]],
            task_type="classification",
            framework="sklearn",
        )

        evaluator = SupervisedEvaluator()
        result = evaluator.evaluate(trained_models, task_type="classification")

        assert isinstance(result, SupervisedEvaluationResult)
        assert len(result.metrics_summary) == 1
        assert "accuracy" in result.metrics_summary["random_forest"]
        assert "f1_score" in result.metrics_summary["random_forest"]
        assert result.confusion_matrices is not None

    def test_evaluate_regression(
        self, sample_features, regression_target, sklearn_regressor_configs
    ):
        """Test regression evaluation."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=regression_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=sklearn_regressor_configs,
            task_type="regression",
            framework="sklearn",
        )

        evaluator = SupervisedEvaluator()
        result = evaluator.evaluate(trained_models, task_type="regression")

        assert isinstance(result, SupervisedEvaluationResult)
        assert "mse" in result.metrics_summary["ridge"]
        assert "rmse" in result.metrics_summary["ridge"]
        assert "r2" in result.metrics_summary["ridge"]

    def test_compare_models(
        self, sample_features, classification_target, sklearn_model_configs
    ):
        """Test model comparison."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=sklearn_model_configs,
            task_type="classification",
            framework="sklearn",
        )

        evaluator = SupervisedEvaluator()
        comparison_df = evaluator.compare_models(trained_models, task_type="classification")

        assert len(comparison_df) == 2
        assert "model" in comparison_df.columns
        assert "f1_score" in comparison_df.columns


class TestSupervisedEvaluationResult:
    """Tests for SupervisedEvaluationResult class."""

    def test_save(
        self, sample_features, classification_target, sklearn_model_configs
    ):
        """Test evaluation result saving."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=[sklearn_model_configs[0]],
            task_type="classification",
            framework="sklearn",
        )

        evaluator = SupervisedEvaluator()
        result = evaluator.evaluate(trained_models, task_type="classification")

        with tempfile.TemporaryDirectory() as tmpdir:
            result.save(tmpdir)

            assert Path(tmpdir, "metrics_summary.json").exists()
            assert Path(tmpdir, "per_window_metrics.parquet").exists()
            assert Path(tmpdir, "confusion_matrices.json").exists()

            # Verify JSON is valid
            with open(Path(tmpdir, "metrics_summary.json")) as f:
                data = json.load(f)
                assert "random_forest" in data

    def test_get_model_metrics(self):
        """Test getting metrics for specific model."""
        result = SupervisedEvaluationResult(
            metrics_summary={
                "model_a": {"accuracy": 0.85, "f1_score": 0.82},
                "model_b": {"accuracy": 0.80, "f1_score": 0.78},
            },
            per_window_metrics=pd.DataFrame(),
        )

        metrics = result.get_model_metrics("model_a")
        assert metrics["accuracy"] == 0.85
        assert metrics["f1_score"] == 0.82

        # Non-existent model returns empty dict
        assert result.get_model_metrics("model_c") == {}


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSupervisedIntegration:
    """Integration tests for supervised learning workflow."""

    def test_full_classification_workflow(
        self, sample_features, classification_target, sklearn_model_configs
    ):
        """Test complete classification workflow."""
        # Train
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=classification_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=sklearn_model_configs,
            task_type="classification",
            framework="sklearn",
        )

        # Evaluate
        evaluator = SupervisedEvaluator()
        result = evaluator.evaluate(trained_models, task_type="classification")

        # Verify end-to-end
        assert len(trained_models) == 2
        assert len(result.metrics_summary) == 2
        assert not result.per_window_metrics.empty

        # Save and verify
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_models(trained_models, tmpdir)
            result.save(tmpdir)

            # Verify all artifacts exist
            assert Path(tmpdir, "models_summary.json").exists()
            assert Path(tmpdir, "metrics_summary.json").exists()

    def test_full_regression_workflow(
        self, sample_features, regression_target, sklearn_regressor_configs
    ):
        """Test complete regression workflow."""
        trainer = SupervisedTrainer(random_state=42)
        feature_cols = list(sample_features.columns)

        trained_models = trainer.walk_forward_train(
            features_df=sample_features,
            target_series=regression_target,
            feature_cols=feature_cols,
            train_period=500,
            test_period=100,
            step_period=200,
            model_configs=sklearn_regressor_configs,
            task_type="regression",
            framework="sklearn",
        )

        evaluator = SupervisedEvaluator()
        result = evaluator.evaluate(trained_models, task_type="regression")

        assert len(trained_models) == 1
        assert "mse" in result.metrics_summary["ridge"]
        assert result.confusion_matrices is None  # No confusion matrix for regression
