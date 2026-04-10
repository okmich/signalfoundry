from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numba import njit

from .base import BasePostProcessor


# =============================================================================
# Numba-optimized core functions
# =============================================================================

@njit(cache=True)
def _normalize_distribution(dist: np.ndarray) -> np.ndarray:
    """Normalize probability distribution."""
    total = np.sum(dist)
    if total > 0:
        return dist / total
    else:
        # Uniform if all zero
        return np.ones_like(dist) / len(dist)


@njit(cache=True)
def _diffuse_belief(belief: np.ndarray, process_noise: float) -> np.ndarray:
    """
    Diffuse belief distribution according to process noise.

    Parameters
    ----------
    belief : np.ndarray, shape (K,)
        Current belief distribution over K states
    process_noise : float
        Process noise parameter (0 = no diffusion, 1 = uniform)

    Returns
    -------
    np.ndarray, shape (K,)
        Diffused belief
    """
    num_states = len(belief)

    # Mix current belief with uniform distribution
    uniform = np.ones(num_states) / num_states
    diffused = (1 - process_noise) * belief + process_noise * uniform

    return _normalize_distribution(diffused)


@njit(cache=True)
def _update_belief(belief: np.ndarray, observation: int, posterior: np.ndarray, measurement_noise: float) -> np.ndarray:
    """
    Update belief with new observation.

    Parameters
    ----------
    belief : np.ndarray, shape (K,)
        Prior belief distribution
    observation : int
        Observed state
    posterior : np.ndarray, shape (K,)
        HMM posterior probabilities
    measurement_noise : float
        Measurement uncertainty

    Returns
    -------
    np.ndarray, shape (K,)
        Updated belief
    """
    # Construct measurement likelihood
    # Weight HMM posteriors by inverse of measurement noise
    measurement_weight = 1.0 / (measurement_noise + 1e-10)

    # Likelihood from HMM posteriors
    likelihood = posterior**measurement_weight

    # Bayesian update: posterior ∝ prior × likelihood
    updated = belief * likelihood

    return _normalize_distribution(updated)


@njit(cache=True)
def _adapt_noise(process_noise: float, measurement_noise: float, prediction_error: float,
                 adaptation_rate: float) -> tuple[float, float]:
    """
    Adapt noise parameters based on prediction error.

    Parameters
    ----------
    process_noise : float
        Current process noise
    measurement_noise : float
        Current measurement noise
    prediction_error : float
        Recent prediction error
    adaptation_rate : float
        Learning rate for adaptation

    Returns
    -------
    tuple of (process_noise, measurement_noise)
        Adapted noise parameters
    """
    # If prediction error is high, increase measurement noise
    # If prediction error is low, decrease it (trust measurements more)

    # Simple exponential moving average adaptation
    target_measurement_noise = min(1.0, max(0.01, prediction_error))
    new_measurement_noise = (
        1 - adaptation_rate
    ) * measurement_noise + adaptation_rate * target_measurement_noise

    # Keep process noise relatively stable
    new_process_noise = process_noise

    return new_process_noise, new_measurement_noise


@njit(cache=True)
def _kalman_style_smoother_core(states: np.ndarray, posteriors: np.ndarray, process_noise: float,
                                measurement_noise: float, adaptation_rate: float, num_states: int) -> tuple[np.ndarray, float, float]:
    """
    Apply adaptive Kalman-style smoothing.

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Input state sequence
    posteriors : np.ndarray, shape (T, K)
        Posterior probabilities
    process_noise : float
        Initial process noise
    measurement_noise : float
        Initial measurement noise
    adaptation_rate : float
        Noise adaptation rate
    num_states : int
        Number of discrete states

    Returns
    -------
    tuple of (smoothed, final_process_noise, final_measurement_noise)
        Smoothed state sequence and final noise parameters
    """
    n = len(states)
    smoothed = np.zeros(n, dtype=states.dtype)

    # Initialize belief to uniform
    belief = np.ones(num_states) / num_states

    # Track prediction errors for adaptation
    recent_errors = np.zeros(10)
    error_idx = 0

    for i in range(n):
        # Prediction step: diffuse belief
        belief = _diffuse_belief(belief, process_noise)

        # Calculate prediction error BEFORE update (true prediction vs observation)
        predicted_state = np.argmax(belief)
        prediction_error = 1.0 if predicted_state != states[i] else 0.0

        # Update step: incorporate measurement
        belief = _update_belief(belief, states[i], posteriors[i], measurement_noise)

        # Output: most likely state
        smoothed[i] = np.argmax(belief)

        # Store error
        recent_errors[error_idx % 10] = prediction_error
        error_idx += 1

        # Adapt noise parameters
        if error_idx > 10:
            mean_error = np.mean(recent_errors)
            process_noise, measurement_noise = _adapt_noise(
                process_noise, measurement_noise, mean_error, adaptation_rate
            )

    return smoothed, process_noise, measurement_noise


# =============================================================================
# AdaptiveKalmanStyleSmoother
# =============================================================================


class AdaptiveKalmanStyleSmoother(BasePostProcessor):
    """
    Treat state sequence as signal with adaptive noise estimation.

    This processor maintains a belief distribution over states and
    updates it using Kalman-filter-like principles adapted for discrete
    states. It adapts noise parameters based on prediction errors.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - process_noise : float, default=0.1
            Expected natural transition rate (0-1)
        - measurement_noise : float, default=0.2
            Expected HMM uncertainty (0-1)
        - adaptation_rate : float, default=0.05
            Learning rate for noise adaptation
        - num_states : int, required
            Number of discrete states

    Examples
    --------
    >>> processor = AdaptiveKalmanStyleSmoother({
    ...     'process_noise': 0.1,
    ...     'measurement_noise': 0.2,
    ...     'adaptation_rate': 0.05,
    ...     'num_states': 3
    ... })
    >>> smoothed = processor.process(states, posteriors=posteriors)

    Notes
    -----
    For trading applications:
    - Lower process_noise: Assumes regimes are persistent
    - Higher measurement_noise: HMM is uncertain
    - adaptation_rate: How quickly to adapt (0.01-0.1 typical)

    This processor is particularly useful when:
    - Regime dynamics change over time
    - HMM uncertainty varies across different market conditions
    - You want automatic adaptation without manual tuning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AdaptiveKalmanStyleSmoother."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("process_noise", 0.1)
        config.setdefault("measurement_noise", 0.2)
        config.setdefault("adaptation_rate", 0.05)

        if "num_states" not in config:
            raise ValueError("num_states is required for AdaptiveKalmanStyleSmoother")

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["num_states"], int):
            raise ValueError("num_states must be an integer")
        if self.config["num_states"] < 2:
            raise ValueError("num_states must be >= 2")

        for param in ["process_noise", "measurement_noise", "adaptation_rate"]:
            if not isinstance(self.config[param], (int, float)):
                raise ValueError(f"{param} must be a number")
            if not (0 <= self.config[param] <= 1):
                raise ValueError(f"{param} must be between 0 and 1")

    def process(self, states: Union[np.ndarray, pd.Series],
                posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        """
        Process state sequence (offline batch mode).
        Raises
        ------
        ValueError
            If posteriors are not provided
        """
        if posteriors is None:
            raise ValueError("AdaptiveKalmanStyleSmoother requires posteriors")

        # Extract arrays and index
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        if isinstance(posteriors, pd.DataFrame):
            posteriors_arr = posteriors.values
        else:
            posteriors_arr = posteriors

        # Validate shapes
        if states_arr.shape[0] != posteriors_arr.shape[0]:
            raise ValueError("states and posteriors must have same length")

        # Apply smoother
        smoothed_arr, final_process, final_measurement = _kalman_style_smoother_core(
            states_arr,
            posteriors_arr,
            self.config["process_noise"],
            self.config["measurement_noise"],
            self.config["adaptation_rate"],
            self.config["num_states"],
        )

        # Store final adapted parameters
        self._final_process_noise = final_process
        self._final_measurement_noise = final_measurement

        # Return in same format as input
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        """
        Process single state observation (online streaming mode).
        Raises
        ------
        ValueError
            If posterior is not provided
        """
        if posterior is None:
            raise ValueError("AdaptiveKalmanStyleSmoother requires posteriors")

        # Initialize state on first call
        if self._online_state is None:
            self._online_state = {
                "belief": np.ones(self.config["num_states"])
                / self.config["num_states"],
                "process_noise": self.config["process_noise"],
                "measurement_noise": self.config["measurement_noise"],
                "recent_errors": np.zeros(10),
                "error_idx": 0,
            }

        # Prediction step
        self._online_state["belief"] = _diffuse_belief(self._online_state["belief"], self._online_state["process_noise"])

        # Calculate prediction error BEFORE update (true prediction vs observation)
        predicted_state = int(np.argmax(self._online_state["belief"]))
        prediction_error = 1.0 if predicted_state != state else 0.0

        # Update step
        self._online_state["belief"] = _update_belief(self._online_state["belief"], state, posterior, self._online_state["measurement_noise"])

        # Output
        smoothed_state = int(np.argmax(self._online_state["belief"]))
        error_idx = self._online_state["error_idx"]
        self._online_state["recent_errors"][error_idx % 10] = prediction_error
        self._online_state["error_idx"] += 1

        # Adapt noise parameters
        if error_idx > 10:
            mean_error = np.mean(self._online_state["recent_errors"])
            new_process, new_measurement = _adapt_noise(
                self._online_state["process_noise"],
                self._online_state["measurement_noise"],
                mean_error,
                self.config["adaptation_rate"],
            )
            self._online_state["process_noise"] = new_process
            self._online_state["measurement_noise"] = new_measurement

        return smoothed_state

    def get_adapted_parameters(self) -> Optional[Dict[str, float]]:
        """
        Get final adapted noise parameters.

        Returns
        -------
        dict or None
            Dictionary with 'process_noise' and 'measurement_noise',
            or None if not yet processed
        """
        if hasattr(self, "_final_process_noise"):
            return {
                "process_noise": self._final_process_noise,
                "measurement_noise": self._final_measurement_noise,
            }
        elif self._online_state is not None:
            return {
                "process_noise": self._online_state["process_noise"],
                "measurement_noise": self._online_state["measurement_noise"],
            }
        else:
            return None
