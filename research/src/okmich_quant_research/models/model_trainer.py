import json
import logging
import os
import joblib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from ..backtesting.hmm_clustering_comparison_backtesting_pipeline import LabelClusterPipelineConfig, \
    LabelAndClusterTestingAndBacktesterPipeline, CLUSTERING_ALGOS


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
    elif hasattr(obj, "__dict__") and hasattr(obj, "__class__"):
        # Handle RandomState and other complex objects
        return str(obj)
    return obj


_logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    model_name: str
    model_type: str  # 'hmm' or 'clustering'
    algorithm: str  # 'hmm_mm_learn', 'kmeans', etc.
    n_states: int
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, float]
    feature_names: List[str]
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "algorithm": self.algorithm,
            "n_states": int(self.n_states),
            "hyperparameters": _convert_to_json_serializable(self.hyperparameters),
            "training_metrics": _convert_to_json_serializable(self.training_metrics),
            "feature_names": self.feature_names,
            "n_samples": int(self.n_samples),
        }


class TrainedModel:
    """Container for a trained model with metadata."""

    def __init__(self, model: Any, metadata: ModelMetadata, labels: np.ndarray):
        self.model = model
        self.metadata = metadata
        self.labels = labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new data."""
        if self.metadata.model_type == "hmm":
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def save(self, output_dir: str):
        """Save model, metadata, and labels."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.metadata.model_name

        # Save model
        model_path = output_dir / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata_path = output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Save labels
        labels_path = output_dir / f"{model_name}_labels.npy"
        np.save(labels_path, self.labels)

    @classmethod
    def load(cls, model_name: str, model_dir: str) -> "TrainedModel":
        """Load a trained model."""
        model_dir = Path(model_dir)

        # Load model
        model_path = model_dir / f"{model_name}.joblib"
        model = joblib.load(model_path)

        # Load metadata
        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
            metadata = ModelMetadata(**metadata_dict)

        # Load labels
        labels_path = model_dir / f"{model_name}_labels.npy"
        labels = np.load(labels_path)

        return cls(model=model, metadata=metadata, labels=labels)


class ModelTrainer:
    """
    Trains HMM and clustering models using the pipeline infrastructure.

    This class leverages LabelAndClusterTestingAndBacktesterPipeline to automatically
    support all current and future algorithms. When new algorithms are added to the
    pipeline, they become immediately available through this trainer.

    Supported HMM variants:
        - hmm_learn: Gaussian HMM with normal emissions
        - hmm_mm_learn: Gaussian HMM with GMM emissions
        - hmm_pmgnt: Pomegranate HMM with normal distribution
        - hmm_lambda: Pomegranate HMM with lambda distribution
        - hmm_student: Pomegranate HMM with Student-t distribution

    Supported clustering algorithms:
        - kmeans, mbkmeans, swkmeans, swmbkmeans, meanshift, dbscan, agglo, ward, birch
        - hdbscan, umap_hdbscan, spectral, gmm
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    @staticmethod
    def get_supported_algorithms() -> List[str]:
        return CLUSTERING_ALGOS.copy()

    def train_clustering(self, features_df: pd.DataFrame, feature_cols: List[str], algorithms: List[str] = None,
                         n_clusters_range: List[int] = None) -> List[TrainedModel]:
        print("=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        print(f"\nTraining data: {features_df.shape[0]} samples, {len(feature_cols)} features")
        print(f"Algorithms: {algorithms}")
        print(f"State/cluster range: {n_clusters_range}")

        trained_models = []
        for n_states in n_clusters_range:
            try:
                trained_models.extend(
                    self._train_single_clustering_state_via_pipeline(
                        features_df=features_df,
                        feature_cols=feature_cols,
                        algorithms=algorithms,
                        n_states=n_states,
                    )
                )
            except Exception as e:
                print(f"   [FAIL] {algorithms}_{n_states}: {e}")
        print(f"\n[OK] Successfully trained {len(trained_models)}/{len(algorithms) * len(n_clusters_range)} models")
        return trained_models

    def _train_single_clustering_state_via_pipeline(self, features_df: pd.DataFrame, feature_cols: List[str],
                                                    algorithms: List[str], n_states: int) -> List[TrainedModel]:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save features to parquet (pipeline expects to read from files)
            temp_symbol = "temp_model_training"
            temp_file = os.path.join(temp_dir, f"{temp_symbol}.parquet")
            features_df[feature_cols].to_parquet(temp_file)

            # Create minimal pipeline config
            class MinimalPipelineConfig(LabelClusterPipelineConfig):
                def __init__(self):
                    super().__init__(
                        input_dir=temp_dir,
                        output_dir=temp_dir,
                        symbols=[temp_symbol],
                        clustering_algos=algorithms,
                        columns_to_scale=feature_cols,
                        columns_scaling_exclude=[],
                        should_resample=False,
                        should_scale=False,  # Data is already prepared
                        should_dim_reduce=len(feature_cols) >= 12,
                        should_fit_cluster=True,
                        should_save_output_df=False,
                        skip_backtesting=True,
                    )

                def create_features(self, df):
                    return df

            # Instantiate pipeline
            pipeline_config = MinimalPipelineConfig()
            pipeline = LabelAndClusterTestingAndBacktesterPipeline(
                pipeline_config=pipeline_config, default_cluster=n_states
            )

            # Get the trained model from the pipeline
            models_output_df = (
                pipeline.run()
            )  # models_output_df will contain each a column named lbl_{algorithm} for each algorithm
            label_columns = [
                c for c in models_output_df.columns if c.startswith("lbl_")
            ]

            training_models = []

            for label_column in label_columns:
                algo = label_column[4:]
                metrics = {
                    "silhouette_score": float(pipeline.algo_silhouette_scores[algo]),
                    "n_unique_labels": int(
                        len(np.unique(models_output_df[label_column]))
                    ),
                }

                labels = models_output_df[label_column].values
                X = models_output_df[[c for c in feature_cols if c != "close"]].values
                try:
                    metrics["calinski_harabasz_score"] = float(
                        calinski_harabasz_score(X, labels)
                    )
                except Exception as e:
                    _logger.warning(
                        "calinski_harabasz_score failed for algo=%s n_states=%s: %s: %s",
                        algo, n_states, type(e).__name__, e,
                    )

                try:
                    metrics["davies_bouldin_score"] = float(
                        davies_bouldin_score(X, labels)
                    )
                except Exception as e:
                    _logger.warning(
                        "davies_bouldin_score failed for algo=%s n_states=%s: %s: %s",
                        algo, n_states, type(e).__name__, e,
                    )

                model = pipeline.get_model(algo)
                # Compute HMM-specific metrics
                if algo.startswith("hmm"):
                    try:
                        if hasattr(model, "get_aic_bic"):
                            metrics["aic"], metrics["bic"] = model.get_aic_bic(X)
                    except Exception as e:
                        _logger.warning(
                            "get_aic_bic failed for algo=%s n_states=%s: %s: %s",
                            algo, n_states, type(e).__name__, e,
                        )

                # Create metadata
                metadata = ModelMetadata(
                    model_name=f"{algo}_{n_states}",
                    model_type="clustering",
                    algorithm=algo,
                    n_states=n_states,
                    hyperparameters=self._extract_hyperparameters(model, algo),
                    training_metrics=metrics,
                    feature_names=feature_cols,
                    n_samples=X.shape[0],
                )
                training_models.append(TrainedModel(model=model, metadata=metadata, labels=labels))
            return training_models

    def _extract_hyperparameters(self, model: Any, algorithm: str) -> Dict[str, Any]:
        """Extract hyperparameters from a trained model."""
        hyperparams = {}

        # Common parameters
        if hasattr(model, "random_state"):
            hyperparams["random_state"] = model.random_state

        if algorithm.startswith("hmm"):
            if hasattr(model, "n_states"):
                hyperparams["n_states"] = model.n_states
            if hasattr(model, "covariance_type"):
                hyperparams["covariance_type"] = model.covariance_type
            if hasattr(model, "n_mix"):
                hyperparams["n_mix"] = model.n_mix
            if hasattr(model, "distribution_type"):
                hyperparams["distribution_type"] = model.distribution_type
            if hasattr(model, "hmm_learn_type"):
                hyperparams["hmm_learn_type"] = model.hmm_learn_type
        else:
            if hasattr(model, "n_clusters"):
                hyperparams["n_clusters"] = model.n_clusters
            if hasattr(model, "n_components"):
                hyperparams["n_components"] = model.n_components
            if hasattr(model, "covariance_type"):
                hyperparams["covariance_type"] = model.covariance_type
            if hasattr(model, "n_init"):
                hyperparams["n_init"] = model.n_init

        return hyperparams

    def save_models(self, models: List[TrainedModel], output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving {len(models)} models to: {output_dir}")

        for model in models:
            model.save(output_dir)
            print(f"   [OK] {model.metadata.model_name}")

        # Save summary
        summary = {
            "n_models": len(models),
            "models": [m.metadata.to_dict() for m in models],
        }
        summary_path = output_dir / "models_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"   [OK] models_summary.json")
