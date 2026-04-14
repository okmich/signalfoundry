import json
from enum import StrEnum
from pathlib import Path
from typing import Union, Dict, Optional, Any

import joblib
import numpy as np
import pandas as pd

from .prophet import ProphetFeatureGenerationService
from .posterior_inference.inferers import ArgmaxInferer
from .posterior_inference.pipeline import PosteriorPipeline
from .posterior_inference.protocols import PosteriorInferer, PosteriorTransformer


# ---------------------------------------------------------------------------
# Module-level artifact helpers (shared by all wrappers)
# ---------------------------------------------------------------------------

def _load_required_joblib(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file: {path}")
    return joblib.load(path)


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata.json must contain a JSON object, got: {type(metadata).__name__}")
    return metadata


def _normalize_state_mapping(state_mapping: Optional[dict]) -> dict:
    if not state_mapping:
        return {}
    return {int(k): v for k, v in state_mapping.items()}


class InferenceModelWrapper:
    def __init__(self, model_dict: Dict[str, str]):
        """
        model dict should be of the form
        {
            'type': 'sklearn or prophet',
            'model_path': '/file-path-to-model.pkl',
        }

        For sklearn: model_path contains artifacts dict with 'model'

        Parameters
        ----------
        model_dict : Dict[str, str]
            Model configuration dictionary
        """
        self.model_type = model_dict.get("type", "sklearn")

        raw_model_path = model_dict.get("model_path")
        if not raw_model_path:
            raise ValueError("model_dict must include 'model_path' for InferenceModelWrapper")

        self.transform_pipeline = joblib.load(model_dict["pipeline_path"]) if model_dict.get("pipeline_path") else None
        if self.model_type == "prophet":
            self.model = ProphetFeatureGenerationService(raw_model_path)
        elif self.model_type == "sklearn":
            artifacts = joblib.load(raw_model_path)
            self.model = artifacts["model"]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> Any:
        # Prophet operates on DataFrames and returns features, not class probabilities.
        # Handle it before the numpy conversion so it receives the original data intact.
        if self.model_type == "prophet":
            return self.model.get_features(data)

        _features = data if isinstance(data, np.ndarray) else data.values

        # Apply pipeline transformation if available
        transformed_features = self.transform_pipeline.transform(_features) if self.transform_pipeline else _features
        probs = self.model.predict_proba(transformed_features)
        return probs, np.argmax(probs, axis=1)


class HmmModelWrapper:
    """HMM wrapper for three-file export format used by posterior-first workflow."""

    MODEL_FILENAME = "hmm_model.joblib"
    PIPELINE_FILENAME = "transform_pipeline.joblib"
    METADATA_FILENAME = "metadata.json"

    def __init__(self, model_dict: Dict[str, str], use_fixed_lag_posterior: bool = False, fixed_lag: Optional[int] = None,
                 posterior_transformers: Optional[list[PosteriorTransformer]] = None,
                 posterior_inferer: Optional[PosteriorInferer] = None):
        """
        model_dict should be of the form:
        {
            'type': 'hmm',
            'model_path': '/path/to/export_folder_or_hmm_model.joblib',
        }

        Export folder content:
        - hmm_model.joblib
        - transform_pipeline.joblib
        - metadata.json

        Posterior pipeline:
        - optional transformer chain
        - inferer defaults to ArgmaxInferer
        """
        self.model_type = model_dict.get("type", "hmm")
        if self.model_type != "hmm":
            raise ValueError(f"HmmModelWrapper only supports model_type='hmm', got: {self.model_type}")

        raw_model_path = model_dict.get("model_path")
        if not raw_model_path:
            raise ValueError("model_dict must include 'model_path' for HmmModelWrapper")

        self.artifact_dir = self._resolve_artifact_dir(raw_model_path)
        self.model = _load_required_joblib(self.artifact_dir / self.MODEL_FILENAME, "hmm model")
        self.transform_pipeline = _load_required_joblib(
            self.artifact_dir / self.PIPELINE_FILENAME, "transform pipeline"
        )
        self.metadata = _load_metadata(self.artifact_dir / self.METADATA_FILENAME)
        self.state_mapping = _normalize_state_mapping(self.metadata.get("state_mapping"))
        self.use_fixed_lag_posterior = use_fixed_lag_posterior
        self.fixed_lag = int(fixed_lag) if fixed_lag is not None else int(
            self.metadata.get("target_lag") or self.metadata.get("lag", 0)
        )
        if self.fixed_lag < 0:
            raise ValueError(f"fixed_lag must be >= 0, got {self.fixed_lag}")
        self.posterior_transformers = list(posterior_transformers) if posterior_transformers is not None else []
        self.posterior_inferer = posterior_inferer if posterior_inferer is not None else ArgmaxInferer()
        self.posterior_pipeline = PosteriorPipeline(transformers=self.posterior_transformers,
                                                    inferer=self.posterior_inferer)

    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> tuple[np.ndarray, Any]:
        _features = data if isinstance(data, np.ndarray) else data.values
        transformed_features = self.transform_pipeline.transform(_features)

        if self.use_fixed_lag_posterior:
            full_probs = np.asarray(self.model.predict_proba_fixed_lag(transformed_features, lag=self.fixed_lag), dtype=float)
            minimum_rows = self.fixed_lag + 1
            if full_probs.shape[0] < minimum_rows:
                raise ValueError(
                    f"Not enough rows for fixed-lag matured posterior: need at least {minimum_rows}, "
                    f"got {full_probs.shape[0]}."
                )
            # Strict as-of output: emit only the matured row at index [-(L+1)].
            probs = full_probs[[-minimum_rows]]
        else:
            probs = np.asarray(self.model.predict_proba(transformed_features), dtype=float)

        inferred = self.posterior_pipeline.run(probs)
        return probs, inferred

    @staticmethod
    def _resolve_artifact_dir(model_path: str) -> Path:
        path = Path(model_path)
        if path.is_dir():
            return path
        if path.is_file() and path.name == HmmModelWrapper.MODEL_FILENAME:
            return path.parent
        raise FileNotFoundError(
            f"model_path must point to artifact directory or '{HmmModelWrapper.MODEL_FILENAME}'. Got: {path}"
        )


class KerasModelWrapper:
    """Keras wrapper for three-file export format used by posterior-first workflow."""

    MODEL_FILENAME = "model.keras"
    PIPELINE_FILENAME = "transform_pipeline.joblib"
    LEGACY_PIPELINE_FILENAME = "scaler.joblib"
    METADATA_FILENAME = "metadata.json"

    class KerasTaskType(StrEnum):
        REGRESSION = "regression"
        CLASSIFICATION = "classification"
        CLASSIFICATION_WITH_POSTERIOR = "classification_with_posterior"

    def __init__(self, model_dict: Dict[str, str], posterior_transformers: Optional[list[PosteriorTransformer]] = None,
                 posterior_inferer: Optional[PosteriorInferer] = None):
        """
        model_dict should be of the form:
        {
            'type': 'keras',
            'model_path': '/path/to/export_folder_or_model.keras',
        }

        Export folder content:
        - model.keras
        - transform_pipeline.joblib
        - metadata.json

        Posterior pipeline:
        - optional transformer chain
        - inferer defaults to ArgmaxInferer
        """
        self.model_type = model_dict.get("type", "keras")
        if self.model_type != "keras":
            raise ValueError(f"KerasModelWrapper only supports model_type='keras', got: {self.model_type}")

        raw_model_path = model_dict.get("model_path")
        if not raw_model_path:
            raise ValueError("model_dict must include 'model_path' for KerasModelWrapper")

        self.artifact_dir = self._resolve_artifact_dir(raw_model_path)
        self.metadata = _load_metadata(self.artifact_dir / self.METADATA_FILENAME)
        self.state_mapping = _normalize_state_mapping(self.metadata.get("state_mapping"))

        task_type_value = self.metadata.get("task_type")
        if not isinstance(task_type_value, str):
            raise ValueError("metadata.json must include 'task_type' as a string for KerasModelWrapper")
        try:
            self.task_type = self.KerasTaskType(task_type_value)
        except ValueError as error:
            valid = ", ".join(task.value for task in self.KerasTaskType)
            raise ValueError(f"Invalid keras task_type '{task_type_value}'. Expected one of: {valid}") from error

        self.transform_pipeline = self._load_transform_pipeline(self.artifact_dir)

        from tensorflow.keras.models import load_model
        self.model = load_model(self.artifact_dir / self.MODEL_FILENAME)

        self.posterior_transformers = list(posterior_transformers) if posterior_transformers is not None else []
        self.posterior_inferer = posterior_inferer if posterior_inferer is not None else ArgmaxInferer()
        self.posterior_pipeline = PosteriorPipeline(transformers=self.posterior_transformers,
                                                    inferer=self.posterior_inferer)

        sequence_length_value = self.metadata.get("sequence_length")
        self.sequence_length = int(sequence_length_value) if sequence_length_value is not None else 0
        is_sequence_meta = self.metadata.get("is_sequence_model")
        if is_sequence_meta is None:
            self.is_sequence_model = self.sequence_length > 1
        else:
            self.is_sequence_model = self._coerce_bool(is_sequence_meta, "is_sequence_model")

        if self.is_sequence_model and self.sequence_length < 1:
            raise ValueError("sequence model requires metadata 'sequence_length' >= 1")

        self._tab_to_seq = None
        if self.is_sequence_model and self.transform_pipeline is not None:
            try:
                from okmich_quant_neural_net.tab_2_seq import transform_to_sequence
            except Exception as error:
                raise ImportError(
                    "Failed to import okmich_quant_neural_net.tab_2_seq.transform_to_sequence required for sequence models"
                ) from error
            self._tab_to_seq = transform_to_sequence

    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> tuple[np.ndarray, Any]:
        _features = data if isinstance(data, np.ndarray) else data.values
        _features = np.asarray(_features)
        if _features.ndim != 2:
            raise ValueError(f"Expected 2D tabular input (n_samples, n_features), got shape {_features.shape}")

        model_input = self._prepare_model_input(_features)
        prediction = np.asarray(self.model.predict(model_input, verbose=0), dtype=float)

        if self.task_type == self.KerasTaskType.REGRESSION:
            return prediction, prediction
        probs = self._as_class_probability_matrix(prediction)
        if self.task_type == self.KerasTaskType.CLASSIFICATION:
            labels = np.argmax(probs, axis=-1)
            return probs, labels

        inferred = self.posterior_pipeline.run(probs)
        return probs, inferred

    def _prepare_model_input(self, features: np.ndarray) -> np.ndarray:
        if not self.is_sequence_model:
            return self.transform_pipeline.transform(features) if self.transform_pipeline else features

        if features.shape[0] <= self.sequence_length:
            raise ValueError(
                f"Sequence model requires more rows than sequence_length. Got rows={features.shape[0]}, "
                f"sequence_length={self.sequence_length}"
            )

        if self._tab_to_seq is not None:
            return self._tab_to_seq(features, self.sequence_length, self.transform_pipeline)

        # No scaler/pipeline available: construct rolling windows directly.
        n_samples = features.shape[0] - self.sequence_length
        indices = np.arange(self.sequence_length)[None, :] + np.arange(n_samples)[:, None]
        return features[indices]

    @staticmethod
    def _resolve_artifact_dir(model_path: str) -> Path:
        path = Path(model_path)
        if path.is_dir():
            return path
        if path.is_file() and path.name == KerasModelWrapper.MODEL_FILENAME:
            return path.parent
        raise FileNotFoundError(
            f"model_path must point to artifact directory or '{KerasModelWrapper.MODEL_FILENAME}'. Got: {path}"
        )

    @classmethod
    def _load_transform_pipeline(cls, artifact_dir: Path):
        primary = artifact_dir / cls.PIPELINE_FILENAME
        if primary.exists():
            return joblib.load(primary)

        legacy = artifact_dir / cls.LEGACY_PIPELINE_FILENAME
        if legacy.exists():
            return joblib.load(legacy)

        raise FileNotFoundError(
            f"Missing transform pipeline. Expected '{cls.PIPELINE_FILENAME}' or '{cls.LEGACY_PIPELINE_FILENAME}' in {artifact_dir}"
        )

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "f", "no", "n", "off"}:
                return False
        raise ValueError(f"metadata field '{field_name}' must be boolean-like, got {value!r}")

    @staticmethod
    def _as_class_probability_matrix(prediction: np.ndarray) -> np.ndarray:
        probs = np.asarray(prediction, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        if probs.ndim != 2:
            raise ValueError(f"Expected classification predictions with shape (n_samples, n_classes), got {probs.shape}")
        if probs.shape[1] == 1:
            positive = np.clip(probs[:, 0], 0.0, 1.0)
            probs = np.column_stack([1.0 - positive, positive])
        return probs
