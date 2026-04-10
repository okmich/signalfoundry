import logging
from typing import Any

import numpy as np

StatefulModel = Any

logger = logging.getLogger(__name__)


def _reset_layer_state(layer) -> None:
    """
    Reset state on a single stateful layer with Keras-version compatibility.

    Keras 3 and TF 2.x expose ``reset_states()`` (plural) on recurrent layers.
    Some custom or third-party layers may only expose ``reset_state()``
    (singular).  This helper tries both names, warning once if neither exists.
    """
    if hasattr(layer, 'reset_states'):
        layer.reset_states()
    elif hasattr(layer, 'reset_state'):
        layer.reset_state()
    else:
        logger.warning(
            f"Layer '{layer.name}' has no reset_states() or reset_state() method. "
            "State was not reset — this may cause incorrect inference behaviour."
        )


class ModelManager:
    """
    Manages the lifecycle and state of a stateful RNN for live trading.

    Attributes:
        model: The underlying stateful Keras model instance.
        stability_period: Number of bars to wait after reset before trusting predictions.
        is_warmed_up: Flag indicating if the model has been initialized.
        bars_since_reset: Counter for bars processed since the last reset.
    """

    def __init__(self, model: StatefulModel, stability_period: int = 0):
        """
        Initializes the StatefulModelManager.

        Args:
            model: A pre-trained, stateful Keras model instance.
            stability_period: Number of bars to wait after a reset before predictions are considered valid. Default is 0 (no wait).
        """
        self.model = model
        self.stability_period = stability_period
        self.is_warmed_up = False
        self.bars_since_reset = 0

        # Collect stateful layers for Keras 3 compatibility
        # In Keras 3, Functional models don't have reset_states() method
        self._stateful_layers = [
            layer for layer in model.layers if hasattr(layer, 'stateful') and getattr(layer, 'stateful', False)
        ]
        if not self._stateful_layers:
            logger.warning(
                "No stateful layers found in model. Ensure model was built with stateful=True."
            )

    # --- Pillar 1: State Initialization (Warm-up) ---
    def initialize_state(self, warmup_data: np.ndarray):
        """
        Performs batch warm-up to initialize the model's hidden state.

        Feeds the entire warmup sequence in one pass to build a meaningful initial hidden state.
        This is much faster than feeding one bar at a time.

        Args:
            warmup_data: Array of shape (warmup_length, num_features) containing historical data to initialize the model's hidden state.

        Raises:
            RuntimeError: If model is already warmed up (call reset_state first).
            ValueError: If warmup_data is empty or None.
        """
        if self.is_warmed_up:
            raise RuntimeError("Model is already warmed up. Call reset_state() before re-initializing.")

        if warmup_data is None or warmup_data.size == 0:
            raise ValueError("Warm-up data cannot be empty.")

        logger.info(f"Starting state warm-up with {len(warmup_data)} records...")

        # Batch warmup: feed entire sequence at once (batch_size=1, timesteps=N, features)
        # Use direct call instead of predict() to avoid any internal state resets
        warmup_batch = np.expand_dims(warmup_data, axis=0)
        _ = self.model(warmup_batch, training=False)

        self.is_warmed_up = True
        self.bars_since_reset = 0
        logger.info("Warm-up complete. Model is ready.")

    # --- Pillar 2: State Resetting (Amnesia) ---

    def reset_state(self):
        """
        Resets the model's hidden state and manager flags.

        Call this on market session boundaries (e.g., daily for stocks, weekly for FX) to prevent state drift.
        After reset, call initialize_state() with fresh warmup data before predicting.
        """
        # Reset state for each stateful layer (Keras-version compatible)
        for layer in self._stateful_layers:
            _reset_layer_state(layer)

        self.is_warmed_up = False
        self.bars_since_reset = 0
        logger.info("Model state has been reset.")

    # --- Pillar 3: Session Synchronization (Patience) ---

    def predict(self, current_bar_data: np.ndarray) -> np.ndarray:
        """
        Generates a prediction for the current bar.

        IMPORTANT: Always returns a prediction, but check `is_stable` property
        before acting on the result. During the stability period, predictions
        may be unreliable as the model's hidden state has not fully converged.

        Args:
            current_bar_data: Feature data for the current bar.
                Accepted shapes: ``(num_features,)`` or ``(1, num_features)``.
                Multi-row 2D arrays are rejected — use the underlying model
                directly for multi-timestep inference.

        Returns:
            Model prediction array. Check `manager.is_stable` before using.

        Raises:
            RuntimeError: If model has not been warmed up.
            ValueError: If input shape is not ``(num_features,)`` or ``(1, num_features)``.

        Example:
            >>> pred = manager.predict(bar)
            >>> if manager.is_stable:
            ...     trade(pred)
            ... else:
            ...     logger.warning("Skipping unstable prediction")
        """
        if not self.is_warmed_up:
            raise RuntimeError("Model must be warmed up before prediction. Call initialize_state().")

        self.bars_since_reset += 1

        # Loud warning during instability
        if not self.is_stable:
            logger.warning(
                f"UNSTABLE PREDICTION: bar {self.bars_since_reset}/{self.stability_period} "
                f"since reset. Model state may not be fully converged. Use with caution."
            )

        # Reshape for single-timestep prediction: (batch_size=1, timesteps=1, features).
        # Accept 1D (features,) or 2D (1, features) only.
        # A 2D array with more than one row would process multiple timesteps and
        # update state N times silently — reject it explicitly.
        if current_bar_data.ndim == 1:
            single_bar = current_bar_data.reshape(1, 1, -1)
        elif current_bar_data.ndim == 2:
            if current_bar_data.shape[0] != 1:
                raise ValueError(
                    f"predict() is designed for single-bar inference. "
                    f"Expected input shape (num_features,) or (1, num_features), "
                    f"got {current_bar_data.shape}. "
                    "For multi-timestep inference use model() directly."
                )
            single_bar = np.expand_dims(current_bar_data, axis=0)  # (1, 1, features)
        else:
            raise ValueError(
                f"predict() expects 1D or 2D input, got ndim={current_bar_data.ndim}."
            )

        # Use direct call to preserve state across predictions
        # model.predict() may reset state internally in some Keras versions
        prediction = self.model(single_bar, training=False)
        return prediction.numpy()

    @property
    def is_stable(self) -> bool:
        """Returns True if model is warmed up and past the stability period."""
        return self.is_warmed_up and self.bars_since_reset > self.stability_period
