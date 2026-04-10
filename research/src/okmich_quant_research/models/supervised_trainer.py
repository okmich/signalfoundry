"""
Supervised Learning Trainer for Walk-Forward Validation.

This module provides training infrastructure for supervised learning models (classification and regression)
with walk-forward validation support.
"""

import importlib
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from keras import backend, callbacks
import keras_tuner as kt

# Sklearn model registry
SKLEARN_CLASSIFIERS = {
    "RandomForestClassifier": "sklearn.ensemble",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "LogisticRegression": "sklearn.linear_model",
    "SVC": "sklearn.svm",
    "XGBClassifier": "xgboost",
    "LGBMClassifier": "lightgbm",
}

SKLEARN_REGRESSORS = {
    "RandomForestRegressor": "sklearn.ensemble",
    "GradientBoostingRegressor": "sklearn.ensemble",
    "Ridge": "sklearn.linear_model",
    "Lasso": "sklearn.linear_model",
    "ElasticNet": "sklearn.linear_model",
    "SVR": "sklearn.svm",
    "XGBRegressor": "xgboost",
    "LGBMRegressor": "lightgbm",
}


def _convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    return obj


@dataclass
class SupervisedModelMetadata:
    """Metadata for a trained supervised model."""

    model_name: str
    model_type: str  # 'sklearn' or 'keras'
    algorithm: str  # 'RandomForestClassifier', 'stacked_rnn', etc.
    task_type: str  # 'classification' or 'regression'
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    n_train_samples: int
    n_classes: Optional[int] = None  # For classification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "algorithm": self.algorithm,
            "task_type": self.task_type,
            "hyperparameters": _convert_to_json_serializable(self.hyperparameters),
            "feature_names": self.feature_names,
            "n_train_samples": int(self.n_train_samples),
            "n_classes": int(self.n_classes) if self.n_classes else None,
        }


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""

    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model_name: str
    model: Any  # The trained model
    scaler: Optional[Any]  # StandardScaler if used
    predictions: np.ndarray
    true_labels: np.ndarray
    probabilities: Optional[np.ndarray]  # For classification
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    feature_names: List[str]
    n_train_samples: int
    n_test_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without model object)."""
        return {
            "window_idx": self.window_idx,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "model_name": self.model_name,
            "metrics": _convert_to_json_serializable(self.metrics),
            "hyperparameters": _convert_to_json_serializable(self.hyperparameters),
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
        }


@dataclass
class SupervisedTrainedModel:
    """Container for a trained supervised model with metadata."""

    model: Any
    metadata: SupervisedModelMetadata
    scaler: Optional[StandardScaler] = None
    walk_forward_results: List[WalkForwardResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on new data."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (classification only)."""
        if self.metadata.task_type != "classification":
            return None
        if self.scaler is not None:
            X = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def save(self, output_dir: str):
        """Save model, metadata, and results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.metadata.model_name

        # Save model
        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)

        # Save scaler if exists
        if self.scaler is not None:
            scaler_path = output_dir / f"{model_name}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata_path = output_dir / f"{model_name}_metadata.json"
        metadata_dict = self.metadata.to_dict()
        metadata_dict["aggregate_metrics"] = _convert_to_json_serializable(
            self.aggregate_metrics
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        # Save walk-forward results summary
        if self.walk_forward_results:
            results_path = output_dir / f"{model_name}_wf_results.json"
            results_data = [r.to_dict() for r in self.walk_forward_results]
            with open(results_path, "w") as f:
                json.dump(results_data, f, indent=2)

    @classmethod
    def load(cls, model_name: str, model_dir: str) -> "SupervisedTrainedModel":
        """Load a trained model."""
        model_dir = Path(model_dir)

        # Load model
        model_path = model_dir / f"{model_name}.joblib"
        model = joblib.load(model_path)

        # Load scaler if exists
        scaler = None
        scaler_path = model_dir / f"{model_name}_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        # Load metadata
        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)

        aggregate_metrics = metadata_dict.pop("aggregate_metrics", {})
        metadata = SupervisedModelMetadata(**metadata_dict)

        return cls(
            model=model,
            metadata=metadata,
            scaler=scaler,
            aggregate_metrics=aggregate_metrics,
        )


class SupervisedTrainer:
    """
    Trains supervised learning models with walk-forward validation.

    Supports both sklearn and Keras models for classification and regression tasks.

    Examples
    --------
    >>> trainer = SupervisedTrainer()
    >>>
    >>> # Train sklearn models with walk-forward validation
    >>> results = trainer.walk_forward_train(
    ...     features_df=features,
    ...     target_series=targets,
    ...     feature_cols=['rsi', 'macd', 'volatility'],
    ...     train_period=5000,
    ...     test_period=1000,
    ...     step_period=500,
    ...     model_configs=[
    ...         {'name': 'rf', 'algorithm': 'RandomForestClassifier',
    ...          'hyperparameters': {'n_estimators': [100, 200], 'max_depth': [5, 10]}}
    ...     ],
    ...     task_type='classification',
    ...     framework='sklearn'
    ... )
    """

    def __init__(self, random_state: int = 42, scale_features: bool = True):
        """
        Initialize supervised trainer.

        Parameters
        ----------
        random_state : int, default=42
            Random state for reproducibility
        scale_features : bool, default=True
            Whether to scale features using StandardScaler
        """
        self.random_state = random_state
        self.scale_features = scale_features

    def walk_forward_train(self, features_df: pd.DataFrame, target_series: pd.Series, feature_cols: List[str],
                           train_period: int, test_period: int, step_period: int, model_configs: List[Dict[str, Any]],
                           model_builder_fn: Optional[Callable] = None, tuner_params: Optional[Dict[str, Any]] = None,
                           keras_training_params: Optional[Dict[str, Any]] = None, task_type: str = "classification",
                           framework: str = "sklearn", anchored: bool = False, sequence_length: Optional[int] = None,
                           max_train_bars: Optional[int] = None, embargo_bars: int = 0) -> List[SupervisedTrainedModel]:
        """
        Perform walk-forward training and validation.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with features (must have DatetimeIndex)
        target_series : pd.Series
            Target variable (aligned with features_df)
        feature_cols : list of str
            Column names to use as features
        train_period : int
            Number of samples for training window
        test_period : int
            Number of samples for test window
        step_period : int
            Number of samples to step forward between windows
        model_configs : list of dict
            Model configurations (for sklearn)
        model_builder_fn : callable, optional
            Keras model builder function (for keras)
        tuner_params : dict, optional
            Keras Tuner parameters
        keras_training_params : dict, optional
            Keras training parameters (epochs, batch_size, etc.)
        task_type : str
            'classification' or 'regression'
        framework : str
            'sklearn' or 'keras'
        anchored : bool
            Whether to use anchored (expanding) windows
        sequence_length : int, optional
            For Keras RNN models, the sequence length (timesteps).
            If provided, features will be transformed into 3D sequences.

        Returns
        -------
        list of SupervisedTrainedModel
            Trained models with walk-forward results
        """
        print("=" * 80)
        print("SUPERVISED WALK-FORWARD TRAINING")
        print("=" * 80)

        # Validate inputs
        if len(features_df) != len(target_series):
            raise ValueError(
                f"features_df and target_series must have the same length "
                f"(got {len(features_df)} vs {len(target_series)})"
            )

        # Calculate windows
        windows = self._calculate_windows(
            len(features_df), train_period, test_period, step_period, anchored,
            max_train_bars=max_train_bars, embargo_bars=embargo_bars,
        )
        print(f"\nCalculated {len(windows)} walk-forward windows")
        print(f"Train period: {train_period}, Test period: {test_period}")
        print(f"Step period: {step_period}, Anchored: {anchored}")
        if max_train_bars:
            print(f"Max train bars: {max_train_bars} (capped-expanding)")
        if embargo_bars:
            print(f"Embargo: {embargo_bars} bars")

        if not windows:
            min_required = train_period + embargo_bars + test_period
            raise ValueError(
                f"No walk-forward windows could be generated. "
                f"Dataset has {len(features_df)} bars but needs at least "
                f"{min_required} (train={train_period} + embargo={embargo_bars} "
                f"+ test={test_period}). Reduce window sizes or provide more data."
            )

        if framework == "sklearn":
            return self._train_sklearn_walk_forward(
                features_df=features_df,
                target_series=target_series,
                feature_cols=feature_cols,
                windows=windows,
                model_configs=model_configs,
                task_type=task_type,
            )
        elif framework == "keras":
            return self._train_keras_walk_forward(
                features_df=features_df,
                target_series=target_series,
                feature_cols=feature_cols,
                windows=windows,
                model_builder_fn=model_builder_fn,
                tuner_params=tuner_params or {},
                keras_training_params=keras_training_params or {},
                task_type=task_type,
                sequence_length=sequence_length,
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")

    def _calculate_windows(
            self,
            total_length: int,
            train_period: int,
            test_period: int,
            step_period: int,
            anchored: bool,
            max_train_bars: Optional[int] = None,
            embargo_bars: int = 0,
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate window boundaries for walk-forward analysis.

        Supports three window strategies:
        - Rolling (anchored=False):          fixed-size training window slides forward.
        - Expanding (anchored=True):         training window grows from the origin.
        - Capped-expanding (anchored=True,   expands until max_train_bars, then rolls.
          max_train_bars set):

        Parameters
        ----------
        embargo_bars : int
            Bars to skip between train end and test start (prevents label leakage
            at fold boundaries when labels have a warmup window).
        """
        windows = []

        if anchored:
            n_windows = (total_length - train_period - test_period - embargo_bars) // step_period + 1
        else:
            n_windows = (total_length - train_period - test_period - embargo_bars) // step_period + 1

        for i in range(n_windows):
            if anchored:
                uncapped_train_start = 0
                uncapped_train_end = train_period + (i * step_period)
                # Capped-expanding: once cap reached, roll the start forward
                if max_train_bars is not None and (uncapped_train_end - uncapped_train_start) > max_train_bars:
                    train_end_idx = uncapped_train_end
                    train_start_idx = train_end_idx - max_train_bars
                else:
                    train_start_idx = uncapped_train_start
                    train_end_idx = uncapped_train_end
            else:
                train_start_idx = i * step_period
                train_end_idx = train_start_idx + train_period

            test_start_idx = train_end_idx + embargo_bars
            test_end_idx = test_start_idx + test_period

            if test_end_idx > total_length:
                break

            windows.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))

        return windows

    def _train_sklearn_walk_forward(
            self,
            features_df: pd.DataFrame,
            target_series: pd.Series,
            feature_cols: List[str],
            windows: List[Tuple[int, int, int, int]],
            model_configs: List[Dict[str, Any]],
            task_type: str,
    ) -> List[SupervisedTrainedModel]:
        """Train sklearn models with walk-forward validation."""

        trained_models = []

        for model_config in model_configs:
            model_name = model_config["name"]
            algorithm = model_config["algorithm"]
            hyperparams_raw = model_config.get("hyperparameters", {})
            # ParameterGrid requires all values to be lists; wrap scalars
            hyperparams_grid = {
                k: v if isinstance(v, list) else [v]
                for k, v in hyperparams_raw.items()
            }

            print(f"\n[{model_name}] Training with {len(windows)} windows...")

            # Generate all hyperparameter combinations
            if hyperparams_grid:
                param_grid = list(ParameterGrid(hyperparams_grid))
            else:
                param_grid = [{}]

            # Track best model across all windows and hyperparams
            # For both classification and regression, higher score = better
            # (regression uses R2 or negated MSE)
            best_score = -np.inf
            best_model = None
            best_scaler = None
            best_hyperparams = {}
            all_window_results = []

            for params in param_grid:
                window_results = []

                for window_idx, (
                        train_start,
                        train_end,
                        test_start,
                        test_end,
                ) in enumerate(windows):
                    # Get train/test data
                    X_train = features_df.iloc[train_start:train_end][feature_cols].values
                    y_train = target_series.iloc[train_start:train_end].values
                    X_test = features_df.iloc[test_start:test_end][feature_cols].values
                    y_test = target_series.iloc[test_start:test_end].values

                    # Scale features
                    scaler = None
                    if self.scale_features:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    # Create and train model
                    model = self._create_sklearn_model(
                        algorithm, task_type, params, self.random_state
                    )
                    model.fit(X_train, y_train)

                    # Predict
                    predictions = model.predict(X_test)
                    probabilities = None
                    if task_type == "classification" and hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_test)

                    # Calculate metrics
                    if task_type == "classification":
                        metrics = self._calculate_classification_metrics(
                            y_test, predictions, probabilities
                        )
                    else:
                        metrics = self._calculate_regression_metrics(y_test, predictions)

                    # Create window result
                    train_start_date = str(features_df.index[train_start])
                    train_end_date = str(features_df.index[train_end - 1])
                    test_start_date = str(features_df.index[test_start])
                    test_end_date = str(features_df.index[test_end - 1])

                    result = WalkForwardResult(
                        window_idx=window_idx,
                        train_start=train_start_date,
                        train_end=train_end_date,
                        test_start=test_start_date,
                        test_end=test_end_date,
                        model_name=f"{model_name}_{window_idx}",
                        model=model,
                        scaler=scaler,
                        predictions=predictions,
                        true_labels=y_test,
                        probabilities=probabilities[:, 1] if probabilities is not None and probabilities.shape[
                            1] == 2 else probabilities,
                        metrics=metrics,
                        hyperparameters=params,
                        feature_names=feature_cols,
                        n_train_samples=len(y_train),
                        n_test_samples=len(y_test),
                    )
                    window_results.append(result)

                # Aggregate metrics for this hyperparameter set
                avg_metrics = self._aggregate_metrics(window_results)

                # Check if this is the best
                if task_type == "classification":
                    score = avg_metrics.get("f1_score", avg_metrics.get("accuracy", 0))
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_scaler = scaler
                        best_hyperparams = params
                        all_window_results = window_results
                else:
                    score = avg_metrics.get("r2", -avg_metrics.get("mse", np.inf))
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_scaler = scaler
                        best_hyperparams = params
                        all_window_results = window_results

            # Create trained model container
            n_classes = None
            if task_type == "classification":
                n_classes = len(np.unique(target_series.dropna()))

            metadata = SupervisedModelMetadata(
                model_name=model_name,
                model_type="sklearn",
                algorithm=algorithm,
                task_type=task_type,
                hyperparameters=best_hyperparams,
                feature_names=feature_cols,
                n_train_samples=windows[-1][1] - windows[-1][0] if windows else 0,
                n_classes=n_classes,
            )

            aggregate_metrics = self._aggregate_metrics(all_window_results)

            trained_model = SupervisedTrainedModel(
                model=best_model,
                metadata=metadata,
                scaler=best_scaler,
                walk_forward_results=all_window_results,
                aggregate_metrics=aggregate_metrics,
            )
            trained_models.append(trained_model)

            print(f"   [OK] {model_name}: Best params={best_hyperparams}")
            for metric_name, metric_value in aggregate_metrics.items():
                print(f"        {metric_name}: {metric_value:.4f}")

        print("\n" + "=" * 80)
        print(f"TRAINING COMPLETE: {len(trained_models)} models trained")
        print("=" * 80)

        return trained_models

    def _create_sequences(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for RNN models.

        Parameters
        ----------
        features : np.ndarray
            2D array of shape (n_samples, n_features)
        targets : np.ndarray
            1D array of shape (n_samples,)
        sequence_length : int
            Number of timesteps per sequence

        Returns
        -------
        X : np.ndarray
            3D array of shape (n_sequences, sequence_length, n_features)
        y : np.ndarray
            1D array of shape (n_sequences,)
        """
        n_samples = len(features)
        n_sequences = n_samples - sequence_length

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for sequence_length ({sequence_length})"
            )

        X = np.zeros((n_sequences, sequence_length, features.shape[1]))
        y = np.zeros(n_sequences)

        for i in range(n_sequences):
            X[i] = features[i: i + sequence_length]
            y[i] = targets[i + sequence_length]

        return X, y

    def _train_keras_walk_forward(
            self,
            features_df: pd.DataFrame,
            target_series: pd.Series,
            feature_cols: List[str],
            windows: List[Tuple[int, int, int, int]],
            model_builder_fn: Optional[Callable],
            tuner_params: Dict[str, Any],
            keras_training_params: Dict[str, Any],
            task_type: str,
            sequence_length: Optional[int] = None) -> List[SupervisedTrainedModel]:
        """Train Keras models with walk-forward validation."""
        if model_builder_fn is None:
            raise ValueError("model_builder_fn is required for Keras training")

        print("\n[keras] Training with walk-forward validation...")
        if sequence_length:
            print(f"   Using sequence_length={sequence_length} for RNN")

        epochs = keras_training_params.get("epochs", 50)
        batch_size = keras_training_params.get("batch_size", 32)
        validation_split = keras_training_params.get("validation_split", 0.2)

        all_window_results = []

        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"   Window {window_idx + 1}/{len(windows)}...")

            # Get train/test data
            X_train_raw = features_df.iloc[train_start:train_end][feature_cols].values
            y_train_raw = target_series.iloc[train_start:train_end].values
            X_test_raw = features_df.iloc[test_start:test_end][feature_cols].values
            y_test_raw = target_series.iloc[test_start:test_end].values

            # Scale features (before creating sequences)
            scaler = None
            if self.scale_features:
                scaler = StandardScaler()
                X_train_raw = scaler.fit_transform(X_train_raw)
                X_test_raw = scaler.transform(X_test_raw)

            # Create sequences for RNN if sequence_length is provided
            if sequence_length:
                X_train, y_train = self._create_sequences(
                    X_train_raw, y_train_raw, sequence_length
                )
                X_test, y_test = self._create_sequences(
                    X_test_raw, y_test_raw, sequence_length
                )
            else:
                X_train, y_train = X_train_raw, y_train_raw
                X_test, y_test = X_test_raw, y_test_raw

            # Use Keras Tuner for hyperparameter optimization
            default_tuner_objective = "val_loss" if task_type == "regression" else "val_accuracy"
            tuner = kt.Hyperband(
                model_builder_fn,
                objective=tuner_params.get("objective", default_tuner_objective),
                max_epochs=tuner_params.get("max_epochs", epochs),
                factor=tuner_params.get("factor", 3),
                directory=tuner_params.get("directory", "keras_tuner"),
                project_name=f"window_{window_idx}",
                overwrite=True,
            )

            # Early stopping callback
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            # Search for best hyperparameters
            tuner.search(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=0,
            )

            # Get best model
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0].values

            # Predict
            predictions_raw = best_model.predict(X_test, verbose=0)
            if task_type == "classification":
                if predictions_raw.shape[-1] > 1:
                    predictions = np.argmax(predictions_raw, axis=-1)
                    probabilities = predictions_raw
                else:
                    predictions = (predictions_raw > 0.5).astype(int).flatten()
                    probabilities = predictions_raw.flatten()
            else:
                predictions = predictions_raw.flatten()
                probabilities = None

            # Calculate metrics
            if task_type == "classification":
                metrics = self._calculate_classification_metrics(
                    y_test, predictions, probabilities
                )
            else:
                metrics = self._calculate_regression_metrics(y_test, predictions)

            # Create window result
            train_start_date = str(features_df.index[train_start])
            train_end_date = str(features_df.index[train_end - 1])
            test_start_date = str(features_df.index[test_start])
            test_end_date = str(features_df.index[test_end - 1])

            result = WalkForwardResult(
                window_idx=window_idx,
                train_start=train_start_date,
                train_end=train_end_date,
                test_start=test_start_date,
                test_end=test_end_date,
                model_name=f"keras_{window_idx}",
                model=best_model,
                scaler=scaler,
                predictions=predictions,
                true_labels=y_test,
                probabilities=probabilities,
                metrics=metrics,
                hyperparameters=best_hp,
                feature_names=feature_cols,
                n_train_samples=len(y_train),
                n_test_samples=len(y_test),
            )
            all_window_results.append(result)

            # Cleanup
            backend.clear_session()

        # Create final trained model (use last window's model)
        n_classes = None
        if task_type == "classification":
            n_classes = len(np.unique(target_series.dropna()))

        metadata = SupervisedModelMetadata(
            model_name="keras_model",
            model_type="keras",
            algorithm="keras_tuner",
            task_type=task_type,
            hyperparameters=all_window_results[-1].hyperparameters if all_window_results else {},
            feature_names=feature_cols,
            n_train_samples=windows[-1][1] - windows[-1][0] if windows else 0,
            n_classes=n_classes,
        )

        aggregate_metrics = self._aggregate_metrics(all_window_results)

        trained_model = SupervisedTrainedModel(
            model=all_window_results[-1].model if all_window_results else None,
            metadata=metadata,
            scaler=all_window_results[-1].scaler if all_window_results else None,
            walk_forward_results=all_window_results,
            aggregate_metrics=aggregate_metrics,
        )

        print(f"   [OK] Keras model trained")
        for metric_name, metric_value in aggregate_metrics.items():
            print(f"        {metric_name}: {metric_value:.4f}")

        return [trained_model]

    def _create_sklearn_model(self, algorithm: str, task_type: str, params: Dict[str, Any], random_state: int) -> Any:
        """Create a sklearn model instance."""
        if task_type == "classification":
            registry = SKLEARN_CLASSIFIERS
        else:
            registry = SKLEARN_REGRESSORS

        if algorithm not in registry:
            raise ValueError(
                f"Unknown algorithm '{algorithm}' for {task_type}. "
                f"Available: {list(registry.keys())}"
            )

        module_name = registry[algorithm]
        module = importlib.import_module(module_name)
        model_class = getattr(module, algorithm)

        # Add random_state if supported
        model_params = params.copy()
        try:
            # Check if the model accepts random_state=
            sig = inspect.signature(model_class)
            if "random_state" in sig.parameters:
                model_params["random_state"] = random_state
        except Exception:
            pass

        return model_class(**model_params)

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          y_proba: Optional[np.ndarray] = None, average: str = "weighted") -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average=average, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average=average, zero_division=0)
            ),
        }

        # AUC-ROC (requires probabilities)
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    if y_proba.ndim == 2:
                        proba = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, 0]
                    else:
                        proba = y_proba
                    metrics["auc_roc"] = float(roc_auc_score(y_true, proba))
                else:
                    # Multiclass
                    metrics["auc_roc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                    )
            except ValueError:
                metrics["auc_roc"] = np.nan

        return metrics

    def _calculate_regression_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)

        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        }

    def _aggregate_metrics(
            self, window_results: List[WalkForwardResult]
    ) -> Dict[str, float]:
        """Aggregate metrics across all windows."""
        if not window_results:
            return {}

        # Collect all metrics
        all_metrics = {}
        for result in window_results:
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                if not np.isnan(metric_value):
                    all_metrics[metric_name].append(metric_value)

        # Calculate mean for each metric
        aggregate = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregate[metric_name] = float(np.mean(values))
                aggregate[f"{metric_name}_std"] = float(np.std(values))
            else:
                aggregate[metric_name] = np.nan
                aggregate[f"{metric_name}_std"] = np.nan

        return aggregate

    def save_models(self, models: List[SupervisedTrainedModel], output_dir: str):
        """Save all trained models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving {len(models)} models to: {output_dir}")

        for model in models:
            model.save(output_dir)
            print(f"   [OK] {model.metadata.model_name}")

        # Save summary
        summary = {
            "n_models": len(models),
            "models": [
                {
                    **m.metadata.to_dict(),
                    "aggregate_metrics": _convert_to_json_serializable(m.aggregate_metrics),
                }
                for m in models
            ],
            "timestamp": datetime.now().isoformat(),
        }

        summary_path = output_dir / "models_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"   [OK] models_summary.json")
