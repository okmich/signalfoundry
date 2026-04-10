from typing import Union, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from .prophet import ProphetFeatureGenerationService
from .hmm import InferenceCache


class InferenceModelWrapper:
    def __init__(self, model_dict: Dict[str, str], use_inference_cache: bool = False, cache_capacity: int = 10):
        """
        model dict should be of the form
        {
            'type': 'hmm or keras or sklearn',
            'pipeline_path': '/optional-file-path-to-pipeline.pkl',  # For keras
            'model_path': '/file-path-to-model.pkl',  # For keras, or artifacts.pkl for hmm/sklearn
        }

        For keras: model and pipeline are separate files
        For hmm/sklearn: model_path contains artifacts dict with 'model' and 'pipeline'

        Parameters
        ----------
        model_dict : Dict[str, str]
            Model configuration dictionary
        use_inference_cache : bool, default=False
            If True and model_type is 'hmm', maintains an immutable cache of
            filter inferences for live trading. Enables exact backtest/live parity.
        cache_capacity : int, default=10000
            Maximum number of inferences to cache (FIFO expiration when exceeded)
        """
        self.pipeline = None
        self.state_mapping = None
        self.inference_cache = None
        self.model_type = model_dict.get("type", "sklearn")

        self.transform_pipeline = joblib.load(model_dict["pipeline_path"]) if model_dict.get("pipeline_path") else None
        if self.model_type == "keras":
            from tensorflow.keras.models import load_model  # lazy import of the keras/tensorflow libraries
            self.model = load_model(model_dict["model_path"])
        elif self.model_type == "prophet":
            self.model = ProphetFeatureGenerationService(model_dict["model_path"])
        elif self.model_type == "hmm":
            artifacts = joblib.load(model_dict["model_path"])
            self.model = artifacts["model"]
            self.state_mapping = artifacts.get("state_mapping")

            # Initialize InferenceCache if requested
            if use_inference_cache:
                self.inference_cache = InferenceCache(capacity=cache_capacity)
        elif self.model_type == "sklearn":
            artifacts = joblib.load(model_dict["model_path"])
            self.model = artifacts["model"]
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, data: Union[np.ndarray, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
        # Prophet operates on DataFrames and returns features, not class probabilities.
        # Handle it before the numpy conversion so it receives the original data intact.
        if self.model_type == "prophet":
            return self.model.get_features(data)

        _features = data if isinstance(data, np.ndarray) else data.values

        # Apply pipeline transformation if available
        transformed_features = self.transform_pipeline.transform(_features) if self.transform_pipeline else _features
        if self.model_type == "keras":
            probs = self.model.predict(transformed_features, verbose=0)
        elif self.model_type == "sklearn":
            probs = self.model.predict_proba(transformed_features)
        elif self.model_type == "hmm":
            probs = self.model.predict_proba(transformed_features)

            # Cache inferences if enabled
            if self.inference_cache is not None:
                if self.inference_cache.is_empty:
                    self.inference_cache.append_all(probs)
                else:
                    self.inference_cache.append(
                        probs[-1]
                    )  # Ongoing: only cache the latest (most recent) inference
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return probs, np.argmax(probs, axis=1)

    # InferenceCache utility methods (HMM only)
    def get_latest_inference(self) -> Optional[tuple[float, ...]]:
        if self.inference_cache is None:
            return None
        try:
            return self.inference_cache.latest()
        except IndexError:
            return None

    def get_inference_history(self, k: Optional[int] = None) -> Optional[list[tuple[float, ...]]]:
        if self.inference_cache is None:
            return None
        if k is None:
            k = len(self.inference_cache)
        return self.inference_cache.window(k)

    def format_cache_contents(self, max_items: Optional[int] = None, precision: int = 4) -> Optional[str]:
        if self.inference_cache is None:
            return None
        return self.inference_cache.format_contents(max_items=max_items, precision=precision)

    @property
    def cache_size(self) -> int:
        """Return the current number of cached inferences (HMM only)."""
        return len(self.inference_cache) if self.inference_cache is not None else 0

    @property
    def is_cache_enabled(self) -> bool:
        """Check if inference caching is enabled."""
        return self.inference_cache is not None
