import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from tensorflow import keras

from okmich_quant_research.backtesting.keras_models_wfa_optimizer import (
    ModelWalkForwardAnalysisOptimizer,
    WindowResult,
    to_serializable,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    raw_data = pd.DataFrame(
        {
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "feature3": np.random.randn(200),
        },
        index=dates,
    )
    label_data = pd.Series(np.random.randint(0, 2, 200), index=dates)
    return raw_data, label_data


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_feature_engineering_fn():
    """Simple feature engineering function for testing."""

    def feature_fn(train_raw, test_raw, train_labels, test_labels):
        train_features = train_raw.values
        test_features = test_raw.values
        return train_features, test_features, train_labels, test_labels

    return feature_fn


@pytest.fixture
def simple_model_builder_fn():
    """Simple model builder function for testing."""

    def model_fn(hp):
        model = keras.Sequential(
            [
                keras.layers.Dense(
                    hp.Int("units", 8, 32, step=8), activation="relu", input_shape=(3,)
                ),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    return model_fn


@pytest.fixture
def simple_postprocess_fn():
    """Simple postprocessing function for testing."""

    def postprocess_fn(predictions, features, labels, threshold=0.5):
        return (predictions > threshold).astype(float)

    return postprocess_fn


# ============================================================================
# Utility Functions Tests
# ============================================================================


class TestToSerializable:
    """Test to_serializable utility function"""

    def test_numpy_float32_conversion(self):
        result = to_serializable(np.float32(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_numpy_float64_conversion(self):
        result = to_serializable(np.float64(2.718))
        assert isinstance(result, float)
        assert result == pytest.approx(2.718)

    def test_numpy_int32_conversion(self):
        result = to_serializable(np.int32(42))
        assert isinstance(result, int)
        assert result == 42

    def test_numpy_int64_conversion(self):
        result = to_serializable(np.int64(100))
        assert isinstance(result, int)
        assert result == 100

    def test_numpy_array_conversion(self):
        arr = np.array([1, 2, 3])
        result = to_serializable(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_regular_python_types(self):
        assert to_serializable("string") == "string"
        assert to_serializable(42) == 42
        assert to_serializable(3.14) == 3.14
        assert to_serializable(True) == True


# ============================================================================
# WindowResult Tests
# ============================================================================


class TestWindowResult:
    """Test WindowResult dataclass"""

    def test_window_result_creation_minimal(self):
        result = WindowResult(
            window_idx=1,
            train_start="2020-01-01",
            train_end="2020-03-31",
            test_start="2020-04-01",
            test_end="2020-04-30",
            best_model_hp={"units": 16},
            best_postprocess_params={"threshold": 0.5},
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            auc_roc=0.82,
        )
        assert result.window_idx == 1
        assert result.accuracy == 0.85
        assert result.ensemble_size == 1  # Default value
        assert result.feature_importance is None
        assert result.early_stopped is False

    def test_window_result_with_all_fields(self):
        result = WindowResult(
            window_idx=2,
            train_start="2020-01-01",
            train_end="2020-03-31",
            test_start="2020-04-01",
            test_end="2020-04-30",
            best_model_hp={"units": 32, "dropout": 0.2},
            best_postprocess_params={"threshold": 0.6},
            accuracy=0.88,
            precision=0.85,
            recall=0.82,
            f1_score=0.84,
            auc_roc=0.87,
            ensemble_size=3,
            feature_importance={"f1": 0.5, "f2": 0.3, "f3": 0.2},
            early_stopped=True,
            convergence_epoch=15,
            metric_trend="improving",
        )
        assert result.ensemble_size == 3
        assert result.early_stopped is True
        assert result.convergence_epoch == 15
        assert result.metric_trend == "improving"
        assert len(result.feature_importance) == 3


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Test ModelWalkForwardAnalysisOptimizer initialization"""

    def test_successful_initialization(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )
        assert optimizer.train_period == 100
        assert optimizer.test_period == 20
        assert optimizer.step_period == 10
        assert optimizer.anchored is False
        assert optimizer.ensemble_size == 1
        assert optimizer.best_model_selection == "most_recent"

    def test_initialization_with_ensemble(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            ensemble_size=3,
            ensemble_method="average",
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )
        assert optimizer.ensemble_size == 3
        assert optimizer.ensemble_method == "average"

    def test_invalid_raw_data_index(
        self,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data = pd.DataFrame({"a": [1, 2, 3]})  # No DatetimeIndex
        label_data = pd.Series([0, 1, 0])
        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            ModelWalkForwardAnalysisOptimizer(
                raw_data=raw_data,
                label_data=label_data,
                train_period=2,
                test_period=1,
                step_period=1,
                feature_engineering_fn=simple_feature_engineering_fn,
                model_builder_fn=simple_model_builder_fn,
                checkpoint_dir=temp_checkpoint_dir,
                verbose=0,
            )

    def test_invalid_label_data_index(
        self,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        raw_data = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=dates)
        label_data = pd.Series([0, 1, 0, 1, 0])  # No DatetimeIndex
        with pytest.raises(ValueError, match="must have DatetimeIndex"):
            ModelWalkForwardAnalysisOptimizer(
                raw_data=raw_data,
                label_data=label_data,
                train_period=3,
                test_period=1,
                step_period=1,
                feature_engineering_fn=simple_feature_engineering_fn,
                model_builder_fn=simple_model_builder_fn,
                checkpoint_dir=temp_checkpoint_dir,
                verbose=0,
            )

    def test_mismatched_indices(
        self,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        dates1 = pd.date_range("2020-01-01", periods=5, freq="D")
        dates2 = pd.date_range("2020-01-02", periods=5, freq="D")
        raw_data = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=dates1)
        label_data = pd.Series([0, 1, 0, 1, 0], index=dates2)
        with pytest.raises(ValueError, match="indices must match"):
            ModelWalkForwardAnalysisOptimizer(
                raw_data=raw_data,
                label_data=label_data,
                train_period=3,
                test_period=1,
                step_period=1,
                feature_engineering_fn=simple_feature_engineering_fn,
                model_builder_fn=simple_model_builder_fn,
                checkpoint_dir=temp_checkpoint_dir,
                verbose=0,
            )

    def test_insufficient_data(
        self,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        raw_data = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=dates)
        label_data = pd.Series([0, 1, 0, 1, 0], index=dates)
        with pytest.raises(ValueError, match="Insufficient data"):
            ModelWalkForwardAnalysisOptimizer(
                raw_data=raw_data,
                label_data=label_data,
                train_period=10,
                test_period=5,
                step_period=1,
                feature_engineering_fn=simple_feature_engineering_fn,
                model_builder_fn=simple_model_builder_fn,
                checkpoint_dir=temp_checkpoint_dir,
                verbose=0,
            )


# ============================================================================
# Window Calculation Tests
# ============================================================================


class TestWindowCalculation:
    """Test window calculation methods"""

    def test_rolling_windows(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            anchored=False,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )
        windows = optimizer._calculate_windows()
        assert len(windows) > 0

        # Check first window
        train_start, train_end, test_start, test_end = windows[0]
        assert train_start == 0
        assert train_end == 100
        assert test_start == 100
        assert test_end == 120

    def test_anchored_windows(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            anchored=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )
        windows = optimizer._calculate_windows()
        assert len(windows) > 0

        # All windows should start from index 0
        for train_start, _, _, _ in windows:
            assert train_start == 0

    def test_window_no_overlap(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            anchored=False,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )
        windows = optimizer._calculate_windows()

        # Check that train and test don't overlap
        for train_start, train_end, test_start, test_end in windows:
            assert test_start == train_end
            assert test_end <= len(raw_data)


# ============================================================================
# Metrics Calculation Tests
# ============================================================================


class TestMetricsCalculation:
    """Test metrics calculation methods"""

    def test_calculate_metrics_perfect_prediction(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])
        y_pred_proba = np.array([0.9, 0.8, 0.1, 0.2, 0.95])

        metrics = optimizer._calculate_metrics(y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert "auc_roc" in metrics

    def test_calculate_metrics_without_proba(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])

        metrics = optimizer._calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert np.isnan(metrics["auc_roc"])

    def test_calculate_metrics_all_same_class(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        y_true = np.ones(10)
        y_pred = np.ones(10)

        metrics = optimizer._calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0


# ============================================================================
# Post-processing Tests
# ============================================================================


class TestPostprocessing:
    """Test post-processing functionality"""

    def test_generate_postprocess_param_combinations_empty(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        combinations = optimizer._generate_postprocess_param_combinations()
        assert len(combinations) == 1
        assert combinations[0] == {}

    def test_generate_postprocess_param_combinations_with_grid(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            postprocess_param_grid={"threshold": [0.3, 0.5, 0.7], "window": [5, 10]},
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        combinations = optimizer._generate_postprocess_param_combinations()
        assert len(combinations) == 6  # 3 * 2

    def test_generate_postprocess_multiple_params(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        grid = {
            "threshold": [0.3, 0.5, 0.7],
            "window_size": [5, 10],
            "min_samples": [2, 3],
        }

        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            postprocess_param_grid=grid,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        combinations = optimizer._generate_postprocess_param_combinations()
        # 3 * 2 * 2 = 12 combinations
        assert len(combinations) == 12


# ============================================================================
# Data Leakage Detection Tests
# ============================================================================


class TestDataLeakageDetection:
    """Test data leakage detection"""

    def test_detect_no_leakage(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            detect_data_leakage=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        train_dates = pd.date_range("2020-01-01", periods=100, freq="D")
        test_dates = pd.date_range("2020-04-11", periods=20, freq="D")
        train_features = pd.DataFrame(np.random.randn(100, 3), index=train_dates)
        test_features = pd.DataFrame(np.random.randn(20, 3), index=test_dates)
        train_raw = pd.DataFrame(np.random.randn(100, 3), index=train_dates)
        test_raw = pd.DataFrame(np.random.randn(20, 3), index=test_dates)

        # Should not raise any errors
        optimizer._detect_data_leakage(
            train_features, test_features, train_raw, test_raw
        )

    def test_detect_leakage_disabled(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            detect_data_leakage=False,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Should return immediately without checks
        optimizer._detect_data_leakage(None, None, None, None)


# ============================================================================
# Early Stopping Tests
# ============================================================================


class TestEarlyStopping:
    """Test early stopping functionality"""

    def test_early_stopping_callback_enabled(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            enable_early_stopping=True,
            early_stopping_patience=5,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        callback = optimizer._get_early_stopping_callback()
        assert callback is not None
        assert isinstance(callback, keras.callbacks.EarlyStopping)
        assert callback.patience == 5

    def test_early_stopping_callback_disabled(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            enable_early_stopping=False,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        callback = optimizer._get_early_stopping_callback()
        assert callback is None


# ============================================================================
# Ensemble Methods Tests
# ============================================================================


class TestEnsembleMethods:
    """Test ensemble prediction methods"""

    def test_ensemble_predict_single_model(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([[0.7], [0.3], [0.8]]))

        features = np.random.randn(3, 3)
        predictions = optimizer._ensemble_predict([mock_model], features)

        assert predictions.shape == (3, 1)
        mock_model.predict.assert_called_once()

    def test_ensemble_predict_multiple_models_average(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            ensemble_size=3,
            ensemble_method="average",
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        mock_models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.predict = Mock(
                return_value=np.array([[0.6 + i * 0.1], [0.4 - i * 0.1]])
            )
            mock_models.append(mock_model)

        features = np.random.randn(2, 3)
        predictions = optimizer._ensemble_predict(mock_models, features)

        assert predictions.shape == (2, 1)
        # Average of [0.6, 0.7, 0.8] = 0.7
        assert predictions[0, 0] == pytest.approx(0.7)

    def test_ensemble_predict_vote(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            ensemble_size=3,
            ensemble_method="vote",
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        mock_models = []
        predictions_list = [
            np.array([[0.7], [0.3]]),  # [1, 0]
            np.array([[0.6], [0.4]]),  # [1, 0]
            np.array([[0.4], [0.6]]),  # [0, 1]
        ]
        for pred in predictions_list:
            mock_model = Mock()
            mock_model.predict = Mock(return_value=pred)
            mock_models.append(mock_model)

        features = np.random.randn(2, 3)
        predictions = optimizer._ensemble_predict(mock_models, features)

        assert predictions.shape == (2, 1)


# ============================================================================
# Convergence Tracking Tests
# ============================================================================


class TestConvergenceTracking:
    """Test convergence tracking functionality"""

    def test_check_convergence_insufficient_data(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            convergence_window_size=3,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Only 1 window
        optimizer.window_results.append(
            WindowResult(
                window_idx=1,
                train_start="2020-01-01",
                train_end="2020-03-31",
                test_start="2020-04-01",
                test_end="2020-04-30",
                best_model_hp={},
                best_postprocess_params={},
                accuracy=0.8,
                precision=0.75,
                recall=0.7,
                f1_score=0.72,
                auc_roc=0.8,
            )
        )

        trend = optimizer._check_convergence()
        assert trend is None

    def test_check_convergence_improving(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            convergence_window_size=3,
            convergence_threshold=0.05,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Add windows with improving F1 scores
        for i, f1 in enumerate([0.5, 0.6, 0.7]):
            optimizer.window_results.append(
                WindowResult(
                    window_idx=i + 1,
                    train_start="2020-01-01",
                    train_end="2020-03-31",
                    test_start="2020-04-01",
                    test_end="2020-04-30",
                    best_model_hp={},
                    best_postprocess_params={},
                    accuracy=0.8,
                    precision=0.75,
                    recall=0.7,
                    f1_score=f1,
                    auc_roc=0.8,
                )
            )

        trend = optimizer._check_convergence()
        assert trend == "improving"

    def test_check_convergence_degrading(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            convergence_window_size=3,
            convergence_threshold=0.05,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Add windows with degrading F1 scores
        for i, f1 in enumerate([0.7, 0.6, 0.5]):
            optimizer.window_results.append(
                WindowResult(
                    window_idx=i + 1,
                    train_start="2020-01-01",
                    train_end="2020-03-31",
                    test_start="2020-04-01",
                    test_end="2020-04-30",
                    best_model_hp={},
                    best_postprocess_params={},
                    accuracy=0.8,
                    precision=0.75,
                    recall=0.7,
                    f1_score=f1,
                    auc_roc=0.8,
                )
            )

        trend = optimizer._check_convergence()
        assert trend == "degrading"

    def test_check_convergence_stable(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            convergence_window_size=3,
            convergence_threshold=0.05,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Add windows with stable F1 scores
        for i in range(3):
            optimizer.window_results.append(
                WindowResult(
                    window_idx=i + 1,
                    train_start="2020-01-01",
                    train_end="2020-03-31",
                    test_start="2020-04-01",
                    test_end="2020-04-30",
                    best_model_hp={},
                    best_postprocess_params={},
                    accuracy=0.8,
                    precision=0.75,
                    recall=0.7,
                    f1_score=0.65,
                    auc_roc=0.8,
                )
            )

        trend = optimizer._check_convergence()
        assert trend == "stable"


# ============================================================================
# Feature Importance Tests
# ============================================================================


class TestFeatureImportance:
    """Test feature importance tracking"""

    def test_extract_feature_importance_disabled(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            track_feature_importance=False,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        mock_model = Mock()
        result = optimizer._extract_feature_importance(mock_model)
        assert result is None

    def test_get_feature_importance_summary_empty(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            track_feature_importance=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        result = optimizer.get_feature_importance_summary()
        assert result is None

    def test_get_feature_importance_summary_with_data(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            track_feature_importance=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Add mock feature importance data
        optimizer.feature_importance_history = [
            {"window_idx": 1, "importance": {"f1": 0.5, "f2": 0.3}},
            {"window_idx": 2, "importance": {"f1": 0.6, "f2": 0.4}},
        ]

        summary = optimizer.get_feature_importance_summary()
        assert summary is not None
        assert len(summary) == 2
        assert "feature" in summary.columns
        assert "avg_importance" in summary.columns
        assert "std_importance" in summary.columns


# ============================================================================
# Checkpointing Tests
# ============================================================================


class TestCheckpointing:
    """Test checkpointing functionality"""

    def test_save_checkpoint(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        optimizer.window_results.append(
            WindowResult(
                window_idx=1,
                train_start="2020-01-01",
                train_end="2020-03-31",
                test_start="2020-04-01",
                test_end="2020-04-30",
                best_model_hp={"units": 16},
                best_postprocess_params={},
                accuracy=0.8,
                precision=0.75,
                recall=0.7,
                f1_score=0.72,
                auc_roc=0.8,
            )
        )

        optimizer._save_checkpoint(1)

        checkpoint_file = Path(temp_checkpoint_dir) / "checkpoint_1.json"
        assert checkpoint_file.exists()

        with open(checkpoint_file, "r") as f:
            data = json.load(f)

        assert data["window_idx"] == 1
        assert data["completed"] is True
        assert len(data["window_results"]) == 1

    def test_load_checkpoint_none_exists(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        result = optimizer._load_checkpoint()
        assert result is None

    def test_load_checkpoint_exists(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Create a checkpoint manually
        checkpoint_data = {
            "window_idx": 2,
            "window_results": [
                {
                    "window_idx": 1,
                    "train_start": "2020-01-01",
                    "train_end": "2020-03-31",
                    "test_start": "2020-04-01",
                    "test_end": "2020-04-30",
                    "best_model_hp": {},
                    "best_postprocess_params": {},
                    "accuracy": 0.8,
                    "precision": 0.75,
                    "recall": 0.7,
                    "f1_score": 0.72,
                    "auc_roc": 0.8,
                    "ensemble_size": 1,
                    "feature_importance": None,
                    "early_stopped": False,
                    "convergence_epoch": None,
                    "metric_trend": None,
                }
            ],
            "best_window_idx": None,
            "best_metric_value": None,
            "completed": True,
        }

        checkpoint_file = Path(temp_checkpoint_dir) / "checkpoint_2.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        last_window = optimizer._load_checkpoint()
        assert last_window == 2
        assert len(optimizer.window_results) == 1


# ============================================================================
# Memory Management Tests
# ============================================================================


class TestMemoryManagement:
    """Test memory management"""

    @patch(
        "okmich_quant_research.backtesting.keras_models_wfa_optimizer.keras.backend.clear_session"
    )
    @patch("okmich_quant_research.backtesting.keras_models_wfa_optimizer.gc.collect")
    def test_clear_memory_enabled(
        self,
        mock_gc,
        mock_clear_session,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            enable_memory_management=True,
            clear_session_between_windows=True,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        optimizer._clear_memory()

        mock_clear_session.assert_called_once()
        mock_gc.assert_called_once()

    @patch(
        "okmich_quant_research.backtesting.keras_models_wfa_optimizer.keras.backend.clear_session"
    )
    @patch("okmich_quant_research.backtesting.keras_models_wfa_optimizer.gc.collect")
    def test_clear_memory_disabled(
        self,
        mock_gc,
        mock_clear_session,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            enable_memory_management=False,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        optimizer._clear_memory()

        mock_clear_session.assert_not_called()
        mock_gc.assert_not_called()


# ============================================================================
# Best Model Selection Tests
# ============================================================================


class TestBestModelSelection:
    """Test best model selection strategies"""

    def test_most_recent_strategy_always_updates(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            best_model_selection="most_recent",
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        window_result = WindowResult(
            window_idx=1,
            train_start="2020-01-01",
            train_end="2020-03-31",
            test_start="2020-04-01",
            test_end="2020-04-30",
            best_model_hp={},
            best_postprocess_params={},
            accuracy=0.6,
            precision=0.55,
            recall=0.5,
            f1_score=0.52,
            auc_roc=0.6,
        )

        result = optimizer._should_update_best_models(1, window_result)
        assert result is True

    def test_best_single_strategy_only_updates_when_better(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            best_model_selection="best_single",
            best_model_metric="f1_score",
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        optimizer.best_metric_value = 0.75

        # Better performance
        window_result_better = WindowResult(
            window_idx=1,
            train_start="2020-01-01",
            train_end="2020-03-31",
            test_start="2020-04-01",
            test_end="2020-04-30",
            best_model_hp={},
            best_postprocess_params={},
            accuracy=0.85,
            precision=0.82,
            recall=0.8,
            f1_score=0.81,
            auc_roc=0.85,
        )

        result = optimizer._should_update_best_models(1, window_result_better)
        assert result is True

        # Worse performance
        window_result_worse = WindowResult(
            window_idx=2,
            train_start="2020-01-01",
            train_end="2020-03-31",
            test_start="2020-04-01",
            test_end="2020-04-30",
            best_model_hp={},
            best_postprocess_params={},
            accuracy=0.65,
            precision=0.62,
            recall=0.6,
            f1_score=0.61,
            auc_roc=0.65,
        )

        result = optimizer._should_update_best_models(2, window_result_worse)
        assert result is False


# ============================================================================
# Aggregate Results Tests
# ============================================================================


class TestAggregateResults:
    """Test aggregate results functionality"""

    def test_aggregate_results_empty(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        results_df, predictions, true_labels = optimizer._aggregate_results()

        assert len(results_df) == 0
        assert len(predictions) == 0
        assert len(true_labels) == 0

    def test_aggregate_results_with_data(
        self,
        sample_data,
        temp_checkpoint_dir,
        simple_feature_engineering_fn,
        simple_model_builder_fn,
    ):
        raw_data, label_data = sample_data
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=raw_data,
            label_data=label_data,
            train_period=100,
            test_period=20,
            step_period=10,
            feature_engineering_fn=simple_feature_engineering_fn,
            model_builder_fn=simple_model_builder_fn,
            checkpoint_dir=temp_checkpoint_dir,
            verbose=0,
        )

        # Add window results
        test_dates = pd.date_range("2020-04-01", periods=20, freq="D")
        optimizer.window_results.append(
            WindowResult(
                window_idx=1,
                train_start="2020-01-01",
                train_end="2020-03-31",
                test_start="2020-04-01",
                test_end="2020-04-20",
                best_model_hp={},
                best_postprocess_params={},
                accuracy=0.8,
                precision=0.75,
                recall=0.7,
                f1_score=0.72,
                auc_roc=0.8,
            )
        )

        optimizer.all_predictions = pd.Series(np.random.rand(20), index=test_dates)
        optimizer.all_true_labels = pd.Series(
            np.random.randint(0, 2, 20), index=test_dates
        )

        results_df, predictions, true_labels = optimizer._aggregate_results()

        assert len(results_df) == 1
        assert len(predictions) == 20
        assert len(true_labels) == 20
