import gc
import json
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, Any

import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from tensorflow import keras

from .wfa_plot_utils import VisualizationMixin
from .wfa_utils import to_serializable, CheckpointingMixin, WindowGenerationMixin, MetricsTrackingMixin, \
    ResultsAggregationMixin
from okmich_quant_research.env_utils import UniversalLogger, EnvironmentDetector


@dataclass
class WindowResult:
    """Stores results from a single window."""

    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_model_hp: Dict
    best_postprocess_params: Dict
    task_type: str = "classification"  # "classification" or "regression"

    # Classification metrics (Optional for regression)
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None

    # Regression metrics (Optional for classification)
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None

    ensemble_size: int = 1
    feature_importance: Optional[Dict] = None
    early_stopped: bool = False
    convergence_epoch: Optional[int] = None
    metric_trend: Optional[str] = None


class ModelWalkForwardAnalysisOptimizer(CheckpointingMixin, WindowGenerationMixin,
                                        MetricsTrackingMixin, ResultsAggregationMixin, VisualizationMixin):
    """
    Walk-Forward Analysis Optimizer for Time Series Classification Models.

    This class implements a robust walk-forward validation framework for evaluating and optimizing
    machine learning models on time series data. It combines hyperparameter tuning, post-processing
    optimization, and comprehensive performance tracking across multiple time windows.

    Key Features:
        - Walk-forward validation with rolling or anchored windows
        - Automated hyperparameter optimization using Keras Tuner
        - Optional post-processing parameter optimization
        - Ensemble learning support (multiple models with voting/averaging)
        - Multiple "best model" selection strategies (most_recent, best_single, best_average_recent)
        - Early stopping and convergence detection
        - Feature importance tracking across windows
        - Data leakage detection
        - Transfer learning between windows
        - Automatic model saving and export for production deployment
        - Automatic checkpointing and resume capability
        - Memory management for long-running experiments

    Typical Usage:
        ```python
        # Define your feature engineering function
        def feature_engineering_fn(train_raw, test_raw, train_labels, test_labels):
            # Transform raw data into features
            train_features = transform(train_raw)
            test_features = transform(test_raw)
            return train_features, test_features, train_labels, test_labels

        # Define your model builder function
        def model_builder_fn(hp):
            model = keras.Sequential([
                keras.layers.Dense(
                    hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu'
                ),
                keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='sparse_categoricalcrossentropy',
                         metrics=['accuracy'])
            return model

        # Initialize optimizer for live trading
        optimizer = ModelWalkForwardAnalysisOptimizer(
            raw_data=df,                        # DataFrame with DatetimeIndex
            label_data=labels,                  # Series with DatetimeIndex
            train_period=252,                   # Training window size (e.g., 252 days)
            test_period=63,                     # Test window size (e.g., 63 days)
            step_period=21,                     # Step size between windows
            feature_engineering_fn=feature_engineering_fn,
            model_builder_fn=model_builder_fn,
            anchored=False,                     # Use rolling windows
            ensemble_size=3,                    # Train ensemble of 3 models
            best_model_selection='most_recent', # Use most recent for live trading
            enable_early_stopping=True,
            track_feature_importance=True
        )

        # Run walk-forward analysis
        results_df, predictions, true_labels = optimizer.run()

        # Get best models for production
        best_models = optimizer.get_best_models()
        best_params = optimizer.get_best_hyperparameters()

        # Export for deployment
        optimizer.export_best_models('./production_models')

        # Visualize results
        optimizer.plot_results()
        ```

    Best Model Selection Strategies:
        - 'most_recent': Always use the most recent window (recommended for live trading)
        - 'best_single': Use window with highest single performance metric
        - 'best_average_recent': Use window with best average over last N windows

    Args:
        raw_data: DataFrame with DatetimeIndex containing raw features
        label_data: Series with DatetimeIndex containing binary labels
        train_period: Number of time steps in each training window
        test_period: Number of time steps in each test window
        step_period: Number of time steps to move forward between windows
        feature_engineering_fn: Function to transform raw data into model features
        model_builder_fn: Function that builds and returns a Keras model (with hp parameter)
        postprocess_fn: Optional function to post-process model predictions
        postprocess_param_grid: Dict of parameter lists for post-processing optimization
        tuner_params: Dict of parameters for Keras Tuner (e.g., max_trials, objective)
        postprocess_optimization_metric: Metric to optimize post-processing ('f1', 'accuracy', etc.)
        anchored: If True, use expanding window; if False, use rolling window
        checkpoint_dir: Directory for saving checkpoints and results
        transfer_learning: If True, use previous window's best model as initialization
        tuning_epochs: Number of epochs for hyperparameter tuning
        tuning_val_split: Validation split fraction during tuning
        verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
        ensemble_size: Number of models in ensemble (1=single model)
        ensemble_method: How to combine ensemble predictions ('best', 'average', 'vote')
        best_model_selection: Strategy for selecting best models ('most_recent', 'best_single', 'best_average_recent')
        best_model_metric: Metric to use for best model selection ('f1_score', 'accuracy', etc.)
        best_model_lookback: Number of windows to average for 'best_average_recent' strategy
        early_stopping_patience: Epochs to wait before early stopping
        early_stopping_min_delta: Minimum improvement threshold for early stopping
        track_feature_importance: Track and save feature importance across windows
        enable_memory_management: Enable aggressive memory management
        clear_session_between_windows: Clear Keras session between windows
        detect_data_leakage: Perform data leakage checks
        convergence_window_size: Number of recent windows to check for convergence
        convergence_threshold: Threshold for detecting performance trends

    Returns:
        Tuple of (results_df, all_predictions, all_true_labels) where:
            - results_df: DataFrame with per-window results and metrics
            - all_predictions: Series of out-of-sample predictions
            - all_true_labels: Series of true labels for predictions

    Model Retrieval Methods:
        - get_best_models(): Load best ensemble models from disk
        - get_best_hyperparameters(): Get hyperparameters and metadata from best window
        - get_best_window_info(): Get information about which window produced best models
        - predict_with_best_models(): Make predictions using best models
        - export_best_models(): Export models to directory for production deployment

    Checkpointing:
        The optimizer automatically saves checkpoints after each window, allowing you to resume
        interrupted runs with optimizer.run(resume=True). Best models and results are saved to checkpoint_dir.

    Notes:
        - All data must have DatetimeIndex with matching indices
        - feature_engineering_fn must accept (train_raw, test_raw, train_labels, test_labels) and return (train_features, test_features, train_labels, test_labels)
        - model_builder_fn receives a keras_tuner HyperParameters object
        - Post-processing is optional but can improve performance
        - For live trading systems, use best_model_selection='most_recent'
        - Models are saved to disk automatically, not kept in memory
    """

    def __init__(self, raw_data: pd.DataFrame, label_data: pd.Series, train_period: int, test_period: int, step_period: int,
                 feature_engineering_fn: Callable, model_builder_fn: Callable, task_type: str = "classification",
                 postprocess_fn: Optional[Callable] = None, postprocess_param_grid: Optional[Dict[str, List[Any]]] = None,
                 tuner_params: Dict = None, postprocess_optimization_metric: str = "f1", anchored: bool = False,
                 checkpoint_dir: Optional[str] = None, transfer_learning: bool = False,
                 tuning_epochs: int = 10, tuning_val_split: float = 0.2, verbose: int = 1,
                 ensemble_size: int = 1, ensemble_method: str = "best", best_model_selection: str = "most_recent",
                 best_model_metric: str = None, best_model_lookback: int = 3, early_stopping_patience: int = 3,
                 early_stopping_min_delta: float = 0.001, track_feature_importance: bool = False, enable_memory_management: bool = True,
                 clear_session_between_windows: bool = True, detect_data_leakage: bool = True,
                 convergence_window_size: int = 3, convergence_threshold: float = 0.05):

        self.log = UniversalLogger(verbose)
        self.env = EnvironmentDetector()

        # Validate task_type
        if task_type not in ("classification", "regression"):
            raise ValueError(f"task_type must be 'classification' or 'regression', got '{task_type}'")

        self.task_type = task_type
        self.raw_data = raw_data
        self.label_data = label_data
        self.train_period = train_period
        self.test_period = test_period
        self.step_period = step_period
        self.feature_engineering_fn = feature_engineering_fn
        self.model_builder_fn = model_builder_fn
        self.postprocess_fn = postprocess_fn
        self.postprocess_param_grid = postprocess_param_grid or {}
        self.tuner_params = tuner_params or {
            "max_trials": 5,
            "objective": "val_accuracy",
        }
        self.postprocess_optimization_metric = postprocess_optimization_metric
        self.anchored = anchored
        self.transfer_learning = transfer_learning
        self.tuning_epochs = tuning_epochs
        self.tuning_val_split = tuning_val_split or 0.2

        self._validate_inputs(raw_data, label_data)

        if checkpoint_dir is None:
            checkpoint_dir = self.env.get_default_checkpoint_dir(checkpoint_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.window_results: List[WindowResult] = []
        self.all_predictions = pd.Series(dtype=float)  # Stores class labels (0 or 1)
        self.all_predictions_proba = pd.Series(dtype=float)  # Stores probabilities
        self.all_true_labels = pd.Series(dtype=float)
        self.previous_best_model = None

        self.ensemble_size = ensemble_size
        self.ensemble_method = ensemble_method
        self.best_model_selection = best_model_selection

        # Auto-set best_model_metric if not provided
        if best_model_metric is None:
            self.best_model_metric = "f1_score" if task_type == "classification" else "r2"
        else:
            self.best_model_metric = best_model_metric

        self.best_model_lookback = best_model_lookback
        self.best_window_idx = None
        self.best_metric_value = -np.inf

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.track_feature_importance = track_feature_importance
        self.enable_memory_management = enable_memory_management
        self.clear_session_between_windows = clear_session_between_windows
        self.detect_data_leakage = detect_data_leakage
        self.convergence_window_size = convergence_window_size
        self.convergence_threshold = convergence_threshold

        self.feature_importance_history: List[Dict] = []

        self.log.info("ModelWalkForwardAnalysisOptimizer initialized", "✓")
        self.log.info(
            f"Environment: {'Colab' if self.env.is_colab() else 'Jupyter' if self.env.is_jupyter() else 'CLI'}"
        )
        self.log.info(f"Task type: {self.task_type}")

        if self.ensemble_size > 1:
            self.log.info(f"Ensemble mode: {ensemble_method} with {ensemble_size} models")

        self.log.info(f"Best model strategy: {best_model_selection}")

        if self.early_stopping_patience > 1:
            self.log.info(f"Early stopping enabled (patience={self.early_stopping_patience})")
        if self.track_feature_importance:
            self.log.info("Feature importance tracking enabled")
        if self.detect_data_leakage:
            self.log.info("Data leakage detection enabled")

    def _create_window_result(self, **kwargs) -> WindowResult:
        """Create a WindowResult instance from keyword arguments.

        This method is required by CheckpointingMixin for loading checkpoints.

        Args:
            **kwargs: Keyword arguments to pass to WindowResult constructor

        Returns:
            WindowResult instance
        """
        return WindowResult(**kwargs)

    def _validate_inputs(self, raw_data: pd.DataFrame, label_data: pd.Series) -> None:
        """Validates input data."""
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            raise ValueError("raw_data must have DatetimeIndex")
        if not isinstance(label_data.index, pd.DatetimeIndex):
            raise ValueError("label_data must have DatetimeIndex")
        if not raw_data.index.equals(label_data.index):
            raise ValueError("raw_data and label_data indices must match")
        if len(raw_data) < self.train_period + self.test_period:
            raise ValueError("Insufficient data for specified train and test periods")

    def _generate_postprocess_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of post-processing parameters."""
        if not self.postprocess_param_grid:
            return [{}]
        keys = self.postprocess_param_grid.keys()
        values = self.postprocess_param_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        return combinations

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        }

        if y_pred_proba is not None:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics["auc_roc"] = np.nan
        else:
            metrics["auc_roc"] = np.nan
        return metrics

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics - follows SupervisedEvaluator pattern.

        Args:
            y_true: True continuous values
            y_pred: Predicted continuous values

        Returns:
            Dict with mse, rmse, mae, r2, mape metrics
        """
        # Flatten arrays if needed
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
        }

    def _extract_predictions_and_probabilities(self, predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract class labels and probabilities from model predictions.
        Handles both softmax outputs (multi-class) and sigmoid outputs (binary).

        Args:
            predictions: Raw model predictions (can be 1D, 2D with 1 column, or 2D with multiple columns)

        Returns:
            Tuple of (class_labels, probabilities) where:
                - class_labels: Predicted class (0 or 1) for each sample
                - probabilities: Confidence probability for the prediction

        Examples:
            - Softmax [0.2, 0.8] -> class_label=1, probability=0.8
            - Softmax [0.9, 0.1] -> class_label=0, probability=0.9
            - Sigmoid [0.7] -> class_label=1, probability=0.7
            - Sigmoid [0.3] -> class_label=0, probability=0.3
        """
        # Handle softmax outputs (multi-class with 2+ output nodes)
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Softmax: use argmax for class labels and max for probability
            class_labels = np.argmax(predictions, axis=1)
            probabilities = np.max(predictions, axis=1)

        # Handle 2D array with single column (flatten it)
        elif predictions.ndim > 1:
            # Single column output, flatten and apply threshold
            predictions_flat = predictions.flatten()
            class_labels = (predictions_flat > 0.5).astype(int)
            probabilities = predictions_flat
        # Handle 1D array (sigmoid output)
        else:
            class_labels = (predictions > 0.5).astype(int)
            probabilities = predictions

        return class_labels, probabilities

    def _detect_data_leakage(self, train_features: Any, test_features: Any, train_data_raw: pd.DataFrame,
                             test_data_raw: pd.DataFrame) -> None:
        """Detect potential data leakage between train and test sets."""
        if not self.detect_data_leakage:
            return

        warnings_found = []

        # Check 1: Index overlap
        if isinstance(train_features, pd.DataFrame) and isinstance(test_features, pd.DataFrame):
            overlap = train_features.index.intersection(test_features.index)
            if len(overlap) > 0:
                warnings_found.append(f"Index overlap detected: {len(overlap)} samples")

        # Check 2: Identical distributions
        if isinstance(train_features, (pd.DataFrame, np.ndarray)) and isinstance(test_features, (pd.DataFrame, np.ndarray)):
            if isinstance(train_features, pd.DataFrame):
                train_arr = train_features.values
                test_arr = test_features.values
            else:
                train_arr = train_features.reshape(train_features.shape[0], -1)
                test_arr = test_features.reshape(test_features.shape[0], -1)

            train_means = train_arr.mean(axis=0)
            test_means = test_arr.mean(axis=0)
            train_stds = train_arr.std(axis=0)
            test_stds = test_arr.std(axis=0)

            mean_diff = np.abs(train_means - test_means).mean()
            std_diff = np.abs(train_stds - test_stds).mean()

            if mean_diff < 0.01 and std_diff < 0.01:
                warnings_found.append(
                    "Suspiciously similar distributions between train/test"
                )

        # Check 3: Test data dates are after training dates
        if test_data_raw.index.min() <= train_data_raw.index.max():
            warnings_found.append(
                "Test data starts before or during training data ends"
            )

        if warnings_found:
            self.log.warning("⚠️  Potential data leakage detected:")
            for warning in warnings_found:
                self.log.warning(f"   - {warning}")
        else:
            self.log.debug("✓ Data leakage check passed")

    def _get_early_stopping_callback(self) -> Optional[keras.callbacks.EarlyStopping]:
        """Create early stopping callback if enabled."""
        if self.early_stopping_patience < 1:
            return None
        return keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.early_stopping_patience,
                                             min_delta=self.early_stopping_min_delta, restore_best_weights=True,
                                             verbose=0)

    def _clear_memory(self) -> None:
        """Clear memory between windows."""
        if not self.enable_memory_management:
            return

        if self.clear_session_between_windows:
            keras.backend.clear_session()

        gc.collect()
        self.log.debug("Memory cleared")

    def _extract_feature_importance(self, model: keras.Model, feature_names: Optional[List[str]] = None) -> Optional[Dict]:
        """Extract feature importance from trained model."""
        if not self.track_feature_importance:
            return None
        try:
            first_layer = model.layers[0]
            if hasattr(first_layer, "get_weights"):
                weights = first_layer.get_weights()[0]

                if len(weights.shape) == 2:
                    importance = np.abs(weights).mean(axis=1)
                else:
                    importance = np.abs(weights).mean()

                if feature_names is not None:
                    feature_importance = dict(zip(feature_names, importance))
                else:
                    feature_importance = {
                        f"feature_{i}": imp for i, imp in enumerate(importance)
                    }

                return feature_importance
        except Exception as e:
            self.log.debug(f"Could not extract feature importance: {e}")

        return None

    def _train_ensemble(self, tuner: kt.Tuner, train_features: Any, train_labels: Any,
                        callbacks: List = None) -> Tuple[List[keras.Model], List[Dict]]:
        if self.ensemble_size <= 1:
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0].values
            return [best_model], [best_hp]

        self.log.info(f"Training ensemble of {self.ensemble_size} models...", "🎯")

        models = []
        hps = []
        if self.ensemble_method == "best":
            models = tuner.get_best_models(num_models=self.ensemble_size)
            hps = [
                tuner.get_best_hyperparameters(num_trials=1)[0].values
                for _ in range(self.ensemble_size)
            ]
        elif self.ensemble_method in ["average", "vote"]:
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            for i in range(self.ensemble_size):
                model = self.model_builder_fn(best_hp)
                model.fit(train_features, train_labels,
                    epochs=self.tuning_epochs, validation_split=self.tuning_val_split,
                    callbacks=callbacks or [], verbose=0)
                models.append(model)
                hps.append(best_hp.values)
                self.log.debug(f"  Ensemble model {i + 1}/{self.ensemble_size} trained")
        return models, hps

    def _ensemble_predict(self, models: List[keras.Model], features: Any) -> np.ndarray:
        """Generate ensemble predictions."""
        if len(models) == 1:
            return models[0].predict(features, verbose=0)

        all_predictions = []
        for model in models:
            pred = model.predict(features, verbose=0)
            all_predictions.append(pred)

        if self.ensemble_method == "vote":
            # For softmax: get argmax (class labels) for voting
            if all_predictions[0].ndim > 1 and all_predictions[0].shape[1] > 1:
                class_preds = [np.argmax(pred, axis=1) for pred in all_predictions]
                # Vote across models, then convert back to probabilities
                from scipy import stats

                ensemble_classes = stats.mode(class_preds, axis=0)[0].flatten()
                # Create probability array with 1.0 for voted class
                ensemble_pred = np.zeros_like(all_predictions[0])
                ensemble_pred[np.arange(len(ensemble_classes)), ensemble_classes] = 1.0
            else:
                binary_preds = [(pred > 0.5).astype(int) for pred in all_predictions]
                ensemble_pred = np.mean(binary_preds, axis=0)
        else:
            # Average probabilities across models
            ensemble_pred = np.mean(all_predictions, axis=0)
        return ensemble_pred

    def _should_update_best_models(self, window_idx: int, window_result: WindowResult) -> bool:
        """Determine if current window's models should be saved as best."""
        if self.best_model_selection == "most_recent":
            return True
        elif self.best_model_selection == "best_single":
            current_metric = getattr(window_result, self.best_model_metric)
            return current_metric > self.best_metric_value
        elif self.best_model_selection == "best_average_recent":
            if len(self.window_results) < self.best_model_lookback:
                return True

            recent_windows = self.window_results[-self.best_model_lookback:] + [window_result]
            avg_metric = np.mean([getattr(w, self.best_model_metric) for w in recent_windows])
            return avg_metric > self.best_metric_value
        return False

    def _save_best_models(self, models: List[keras.Model], window_result: WindowResult) -> None:
        """Save the best performing models and their parameters to disk."""
        try:
            best_models_dir = self.checkpoint_dir / "best_models"
            best_models_dir.mkdir(parents=True, exist_ok=True)

            # Remove old best models
            for old_model in best_models_dir.glob("model_*.keras"):
                old_model.unlink()

            # Save each model in the ensemble
            for i, model in enumerate(models):
                model_path = best_models_dir / f"model_{i}.keras"
                model.save(model_path)

            # Save metadata with appropriate metrics
            metrics_dict = {}
            if window_result.task_type == "classification":
                metrics_dict = {
                    "accuracy": window_result.accuracy,
                    "precision": window_result.precision,
                    "recall": window_result.recall,
                    "f1_score": window_result.f1_score,
                    "auc_roc": window_result.auc_roc,
                }
            else:  # regression
                metrics_dict = {
                    "mse": window_result.mse,
                    "rmse": window_result.rmse,
                    "mae": window_result.mae,
                    "r2": window_result.r2,
                    "mape": window_result.mape,
                }

            metadata = {
                "window_idx": window_result.window_idx,
                "train_start": window_result.train_start,
                "train_end": window_result.train_end,
                "test_start": window_result.test_start,
                "test_end": window_result.test_end,
                "task_type": window_result.task_type,
                "ensemble_size": len(models),
                "model_hyperparameters": window_result.best_model_hp,
                "postprocess_parameters": window_result.best_postprocess_params,
                "metrics": metrics_dict,
                "best_metric": self.best_model_metric,
                "best_metric_value": float(
                    getattr(window_result, self.best_model_metric)
                ),
                "selection_strategy": self.best_model_selection,
            }

            metadata_path = best_models_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=to_serializable)

            self.log.debug(f"Best models saved to {best_models_dir}", "💾")
        except Exception as e:
            self.log.warning(f"Could not save best models: {e}")

    def _optimize_postprocess_parameters(self, predictions: np.ndarray, val_features: pd.DataFrame,
                                         val_labels: pd.Series) -> Dict[str, Any]:
        """Optimize post-processing parameters."""
        if not self.postprocess_fn:
            return {}

        param_combinations = self._generate_postprocess_param_combinations()
        if len(param_combinations) == 1 and not param_combinations[0]:
            return {}

        self.log.debug(f"Testing {len(param_combinations)} post-processing parameter combinations...")
        best_params = None
        best_metric = -np.inf

        for params in param_combinations:
            try:
                processed_preds = self.postprocess_fn(predictions, val_features, val_labels, **params)
                # Extract class labels and probabilities using helper method
                processed_preds_binary, processed_preds_proba = (
                    self._extract_predictions_and_probabilities(processed_preds)
                )

                metrics = self._calculate_metrics(val_labels.values, processed_preds_binary, processed_preds_proba)
                metric = metrics.get(self.postprocess_optimization_metric, -np.inf)
                if metric > best_metric:
                    best_metric = metric
                    best_params = params
            except Exception as e:
                self.log.debug(f"Failed params {params}: {e}")
                continue
        self.log.debug(f"Best post-processing params: {best_params} (metric: {best_metric:.4f})")
        return best_params or {}

    def _run_window(self, window_idx: int, train_start_idx: int, train_end_idx: int,
                    test_start_idx: int, test_end_idx: int) -> WindowResult:
        print(f"\n{'=' * 60}")
        self.log.info(f"Window {window_idx}", "🔄")
        print(f"{'=' * 60}")

        # Slice data
        train_data_raw = self.raw_data.iloc[train_start_idx:train_end_idx]
        train_labels_raw = self.label_data.iloc[train_start_idx:train_end_idx]
        test_data_raw = self.raw_data.iloc[test_start_idx:test_end_idx]
        test_labels_raw = self.label_data.iloc[test_start_idx:test_end_idx]

        print(
            f"Training: {train_data_raw.index[0].date()} to "
            f"{train_data_raw.index[-1].date()} ({len(train_data_raw)} samples), "
            f"Testing:  {test_data_raw.index[0].date()} to {test_data_raw.index[-1].date()} ({len(test_data_raw)} "
            f"samples)"
        )

        # Feature engineering
        train_features, test_features, train_labels, test_labels = (
            self.feature_engineering_fn(
                train_data_raw, test_data_raw, train_labels_raw, test_labels_raw
            )
        )
        self.log.debug(f"Transformed shapes - Train: {train_features.shape}, Test: {test_features.shape}")

        # Data leakage detection
        self._detect_data_leakage(train_features, test_features, train_data_raw, test_data_raw)

        # Prepare callbacks
        callbacks = []
        early_stop_callback = self._get_early_stopping_callback()
        if early_stop_callback:
            callbacks.append(early_stop_callback)

        # Step 1: Optimize model hyperparameters
        self.log.info("Step 1: Optimizing model hyperparameters...", "🔍")
        tuner = kt.BayesianOptimization(self.model_builder_fn, **self.tuner_params,
                                        directory=str(self.checkpoint_dir / "tuning"),
                                        project_name=f"window_{window_idx}", overwrite=True)

        if isinstance(train_labels, pd.Series):
            train_labels_array = train_labels.values
        else:
            train_labels_array = train_labels

        tuner.search(train_features, train_labels_array, epochs=self.tuning_epochs,
                     validation_split=self.tuning_val_split, callbacks=callbacks, verbose=0)

        ensemble_models, ensemble_hps = self._train_ensemble(tuner, train_features, train_labels_array, callbacks)
        best_model = ensemble_models[0]
        best_model_hp = ensemble_hps[0]

        self.log.info(f"Best model params: {best_model_hp}", "✓")
        if len(ensemble_models) > 1:
            self.log.info(f"Ensemble of {len(ensemble_models)} models trained", "✓")

        feature_names = None
        if isinstance(train_features, pd.DataFrame):
            feature_names = train_features.columns.tolist()

        feature_importance = self._extract_feature_importance(best_model, feature_names)
        if feature_importance:
            self.log.debug(f"Feature importance extracted: {len(feature_importance)} features")
            self.feature_importance_history.append({"window_idx": window_idx, "importance": feature_importance})

        early_stopped = False
        convergence_epoch = None
        if early_stop_callback and hasattr(early_stop_callback, "stopped_epoch"):
            if early_stop_callback.stopped_epoch > 0:
                early_stopped = True
                convergence_epoch = early_stop_callback.stopped_epoch
                self.log.info(f"Early stopped at epoch {convergence_epoch}", "⏸")

        if self.transfer_learning:
            self.previous_best_model = best_model

        # Step 2: Optimize post-processing parameters
        best_postprocess_params = {}
        if self.postprocess_fn:
            val_split_idx = int(len(train_features) * 0.8)
            val_features = train_features[val_split_idx:]

            if isinstance(train_labels, pd.Series):
                val_labels = train_labels.iloc[val_split_idx:]
            else:
                val_labels = pd.Series(train_labels[val_split_idx:])

            val_predictions = self._ensemble_predict(ensemble_models, val_features)

            self.log.info("Step 2: Optimizing post-processing parameters...", "🎯")
            best_postprocess_params = self._optimize_postprocess_parameters(
                val_predictions, val_features, val_labels)
            self.log.info(f"Best post-processing params: {best_postprocess_params}", "✓")

        # Step 3: Generate predictions on test set
        test_predictions_raw = self._ensemble_predict(ensemble_models, test_features)

        # Step 4: Apply post-processing
        if self.postprocess_fn and best_postprocess_params:
            test_predictions = self.postprocess_fn(
                test_predictions_raw,
                test_features,
                test_labels,
                **best_postprocess_params)
        else:
            test_predictions = test_predictions_raw

        test_predictions_binary, test_predictions_proba, test_predictions_continuous = None, None, None
        # Step 5: Calculate metrics based on task type
        if self.task_type == "classification":
            # Convert to binary predictions and probabilities using helper method
            test_predictions_binary, test_predictions_proba = self._extract_predictions_and_probabilities(test_predictions)
            metrics = self._calculate_metrics(test_labels, test_predictions_binary, test_predictions_proba)
        else:  # regression
            # For regression, predictions are already continuous values
            if test_predictions.ndim > 1:
                test_predictions_continuous = test_predictions.flatten()
            else:
                test_predictions_continuous = test_predictions
            metrics = self._calculate_regression_metrics(test_labels, test_predictions_continuous)

        # Store results
        if isinstance(test_labels, pd.Series):
            test_index = test_labels.index
            test_labels_series = test_labels
        else:
            n_samples = len(test_labels)
            test_index = test_data_raw.index[-n_samples:]
            test_labels_series = pd.Series(test_labels, index=test_index)

        # Store predictions based on task type
        if self.task_type == "classification":
            # Store both binary predictions and probabilities
            self.all_predictions = pd.concat(
                [self.all_predictions, pd.Series(test_predictions_binary, index=test_index)])
            self.all_predictions_proba = pd.concat(
                [
                    self.all_predictions_proba,
                    pd.Series(test_predictions_proba, index=test_index),
                ])
        else:  # regression
            # Store continuous predictions directly
            self.all_predictions = pd.concat(
                [self.all_predictions, pd.Series(test_predictions_continuous, index=test_index)]
            )

        self.all_true_labels = pd.concat([self.all_true_labels, test_labels_series])

        # Check convergence based on appropriate metric
        convergence_metric = "f1_score" if self.task_type == "classification" else "r2"
        metric_trend = self._check_convergence(convergence_metric)
        if metric_trend:
            self.log.info(f"Performance trend: {metric_trend}", "📈" if metric_trend == "improving" else "📉")

        # Create window result with appropriate metrics
        window_result = WindowResult(
            window_idx=window_idx,
            train_start=str(train_data_raw.index[0].date()),
            train_end=str(train_data_raw.index[-1].date()),
            test_start=str(test_data_raw.index[0].date()),
            test_end=str(test_data_raw.index[-1].date()),
            best_model_hp=best_model_hp,
            best_postprocess_params=best_postprocess_params,
            task_type=self.task_type,
            **metrics,  # Unpack classification or regression metrics
            ensemble_size=len(ensemble_models),
            feature_importance=feature_importance,
            early_stopped=early_stopped,
            convergence_epoch=convergence_epoch,
            metric_trend=metric_trend,
        )

        self.window_results.append(window_result)

        print(f"\nWindow {window_idx} Results:")
        if self.task_type == "classification":
            print(
                f"  Accuracy:   {window_result.accuracy:.4f},  Precision:  {window_result.precision:.4f},"
                f"  Recall:     {window_result.recall:.4f},  F1 Score:   {window_result.f1_score:.4f},"
                f"  AUC-ROC:    {window_result.auc_roc:.4f}")
        else:  # regression
            print(
                f"  MSE:        {window_result.mse:.4f},  RMSE:       {window_result.rmse:.4f},"
                f"  MAE:        {window_result.mae:.4f},  R²:         {window_result.r2:.4f},"
                f"  MAPE:       {window_result.mape:.2f}%")

        if window_result.ensemble_size > 1:
            print(f"  Ensemble:   {window_result.ensemble_size} models")
        if window_result.early_stopped:
            print(f"  Early stop: epoch {window_result.convergence_epoch}")
        if window_result.metric_trend:
            print(f"  Trend:      {window_result.metric_trend}")

        # Check if this window should be saved as best
        if self._should_update_best_models(window_idx, window_result):
            current_metric = getattr(window_result, self.best_model_metric)
            self.best_metric_value = current_metric
            self.best_window_idx = window_idx

            self._save_best_models(ensemble_models, window_result)
            self.log.info(f"New best models saved! {self.best_model_metric}={current_metric:.4f}","🏆")

        self._save_checkpoint(window_idx)

        # Clean up models from memory
        if self.enable_memory_management:
            del ensemble_models
            del best_model
            keras.backend.clear_session()
            gc.collect()

        self._clear_memory()
        return window_result

    def run(self, resume: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Execute full Walk-Forward Analysis."""
        print("\n" + "=" * 60)
        self.log.info("STARTING WALK-FORWARD ANALYSIS", "🚀")
        print("=" * 60)

        start_window = 0
        if resume:
            last_completed = self._load_checkpoint()
            if last_completed is not None:
                start_window = last_completed + 1
                self.log.info(f"Resuming from window {start_window}", "⏩")
        windows = self._calculate_windows()

        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
            if idx < start_window:
                continue
            self._run_window(idx, train_start, train_end, test_start, test_end)

        return self._aggregate_results()

    def _aggregate_results(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Aggregate and display final results using ResultsAggregationMixin."""
        self._print_results_header()

        if not self.window_results:
            self.log.warning("No windows completed")
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

        results_df = self._create_results_dataframe()

        # Print classification statistics
        self._print_classification_statistics(results_df)

        # Print ensemble and early stopping statistics
        self._print_ensemble_statistics(results_df)
        self._print_early_stopping_statistics(results_df)

        # Print best model info
        self._print_best_model_info()

        # Print feature importance
        self._print_feature_importance_summary()

        # Calculate overall out-of-sample metrics
        if len(self.all_predictions) > 0 and len(self.all_true_labels) > 0:
            if self.task_type == "classification":
                # self.all_predictions already contains class labels (0 or 1)
                # self.all_predictions_proba contains probabilities
                overall_metrics = self._calculate_metrics(
                    self.all_true_labels.values,
                    self.all_predictions.values.astype(int),
                    self.all_predictions_proba.values)

                print("\nOverall Out-of-Sample Performance:")
                print(
                    f"  Accuracy:    {overall_metrics['accuracy']:.4f}, "
                    f"Precision:  {overall_metrics['precision']:.4f}, "
                    f"Recall:      {overall_metrics['recall']:.4f}, "
                    f"F1-Score: {overall_metrics['f1_score']:.4f}, "
                    f"AUC-ROC:     {overall_metrics['auc_roc']:.4f}")
            else:  # regression
                # For regression, predictions are continuous values
                overall_metrics = self._calculate_regression_metrics(
                    self.all_true_labels.values,
                    self.all_predictions.values,
                )

                print("\nOverall Out-of-Sample Performance:")
                print(
                    f"  MSE:      {overall_metrics['mse']:.4f}, "
                    f"RMSE:     {overall_metrics['rmse']:.4f}, "
                    f"MAE:      {overall_metrics['mae']:.4f}, "
                    f"R²:       {overall_metrics['r2']:.4f}, "
                    f"MAPE:     {overall_metrics['mape']:.2f}%"
                )
        # Save results
        self._save_results_files(results_df, save_predictions=True, save_returns=False, save_signals=False)
        return results_df, self.all_predictions, self.all_true_labels

    def _summarize_feature_importance(self, top_n: int=15) -> None:
        """Summarize feature importance across windows."""
        if not self.feature_importance_history:
            return

        all_features = set()
        for window_fi in self.feature_importance_history:
            all_features.update(window_fi["importance"].keys())

        avg_importance = {}
        for feature in all_features:
            importances = [
                window_fi["importance"].get(feature, 0.0)
                for window_fi in self.feature_importance_history
            ]
            avg_importance[feature] = np.mean(importances)
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

        print(f"  Top 10 features (avg importance):")
        for feature, importance in top_features:
            print(f"    {feature}: {importance:.4f}")

    def get_best_models(self) -> List[keras.Model]:
        best_models_dir = self.checkpoint_dir / "best_models"

        if not best_models_dir.exists():
            self.log.warning("No best models found. Run analysis first.")
            return []

        models = []
        i = 0
        while (best_models_dir / f"model_{i}.keras").exists():
            model_path = best_models_dir / f"model_{i}.keras"
            model = keras.models.load_model(model_path)
            models.append(model)
            i += 1

        if models:
            self.log.info(f"Loaded {len(models)} best model(s) from window {self.best_window_idx}", "✓")

        return models

    def get_best_hyperparameters(self) -> Optional[Dict]:
        best_models_dir = self.checkpoint_dir / "best_models"
        metadata_path = best_models_dir / "metadata.json"

        if not metadata_path.exists():
            self.log.warning("No best model metadata found. Run analysis first.")
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            self.log.warning(f"Could not load metadata: {e}")
            return None

    def get_best_window_info(self) -> Optional[Dict]:
        metadata = self.get_best_hyperparameters()
        if not metadata:
            return None
        return {
            "window_idx": metadata["window_idx"],
            "train_period": f"{metadata['train_start']} to {metadata['train_end']}",
            "test_period": f"{metadata['test_start']} to {metadata['test_end']}",
            "metrics": metadata["metrics"],
            "selection_strategy": metadata.get("selection_strategy", "unknown"),
        }

    def predict_with_best_models(self, new_features: Any) -> np.ndarray:
        models = self.get_best_models()

        if not models:
            raise ValueError("No best models available. Run analysis first.")

        return self._ensemble_predict(models, new_features)

    def export_best_models(self, export_dir: str) -> None:
        """
        Export best models and parameters to a specified directory for production use.

        Args:
            export_dir: Directory to export models and metadata
        """
        best_models_dir = self.checkpoint_dir / "best_models"

        if not best_models_dir.exists():
            self.log.warning("No best models to export. Run analysis first.")
            return

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy all model files
        model_count = 0
        for model_file in best_models_dir.glob("model_*.keras"):
            shutil.copy2(model_file, export_path / model_file.name)
            model_count += 1

        # Copy metadata
        metadata_file = best_models_dir / "metadata.json"
        if metadata_file.exists():
            shutil.copy2(metadata_file, export_path / "metadata.json")

        # Create a README for production use
        readme_content = f"""# Production Models Export

        ## Model Information
        - Number of models: {model_count}
        - Selection strategy: {self.best_model_selection}
        - Best metric: {self.best_model_metric}
        - Window index: {self.best_window_idx}

        ## Usage
        ```python
        import keras
        import numpy as np

        # Load models
        models = []
        for i in range({model_count}):
            model = keras.models.load_model(f'model_{{i}}.keras')
            models.append(model)

        # Make predictions (ensemble)
        def ensemble_predict(models, features):
            predictions = [model.predict(features) for model in models]
            return np.mean(predictions, axis=0)

        # Use in production
        predictions = ensemble_predict(models, new_features)
        binary_predictions = (predictions > 0.5).astype(int)
        ```

        ## Metadata
        See metadata.json for full hyperparameters and performance metrics.
        """

        readme_path = export_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)

        self.log.info(f"Best models exported to: {export_path}", "📦")
        self.log.info(f"  - {model_count} model file(s)", "")
        self.log.info(f"  - metadata.json", "")
        self.log.info(f"  - README.md", "")

    def get_all_hyperparameters(self) -> pd.DataFrame:
        """
        Get hyperparameters from all windows.

        Returns:
            DataFrame with window indices and their hyperparameters,
            or empty DataFrame if no windows completed
        """
        if not self.window_results:
            self.log.warning("No windows completed yet")
            return pd.DataFrame()

        data = []
        for window in self.window_results:
            row = {"window_idx": window.window_idx}
            row.update(window.best_model_hp)
            data.append(row)

        return pd.DataFrame(data)

    def get_feature_importance_summary(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance summary across all windows.

        Returns:
            DataFrame with features and their average importance,
            or None if feature importance tracking is disabled
        """
        if not self.feature_importance_history:
            return None

        all_features = set()
        for window_fi in self.feature_importance_history:
            all_features.update(window_fi["importance"].keys())

        summary_data = []
        for feature in all_features:
            importances = [
                window_fi["importance"].get(feature, 0.0)
                for window_fi in self.feature_importance_history
            ]
            summary_data.append(
                {
                    "feature": feature,
                    "avg_importance": np.mean(importances),
                    "std_importance": np.std(importances),
                    "min_importance": np.min(importances),
                    "max_importance": np.max(importances),
                }
            )

        df = pd.DataFrame(summary_data)
        df = df.sort_values("avg_importance", ascending=False)
        return df
