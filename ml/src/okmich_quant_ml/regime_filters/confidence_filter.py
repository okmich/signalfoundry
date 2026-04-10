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
def _mode_filter_with_confidence_core(states: np.ndarray, posteriors: np.ndarray, window_size: int,
                                      confidence_weight: float, min_score_threshold: float, causal: bool) -> np.ndarray:
    """
    Apply confidence-weighted mode filter to state sequence.

    For each window, calculate score for each state:
    score(state) = count(state) * mean(posteriors[state])^confidence_weight

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Input state sequence
    posteriors : np.ndarray, shape (T, K)
        Posterior probabilities for K states
    window_size : int
        Window size for mode calculation
    confidence_weight : float
        Exponent for confidence weighting (0=ignore, 1=linear, >1=emphasize)
    min_score_threshold : float
        Minimum score required to accept state
    causal : bool
        If True, only use past observations

    Returns
    -------
    np.ndarray, shape (T,)
        Filtered state sequence
    """
    n = len(states)
    num_states = posteriors.shape[1]

    if n == 0:
        return states.copy()

    smoothed = np.zeros(n, dtype=states.dtype)

    for i in range(n):
        # Define window
        if causal:
            start = max(0, i - window_size + 1)
            end = i + 1
        else:
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

        # Extract window
        window_states = states[start:end]
        window_posteriors = posteriors[start:end, :]

        # Calculate scores for each state
        best_score = -np.inf
        best_state = states[i]

        for state in range(num_states):
            # Count occurrences
            count = np.sum(window_states == state)

            if count > 0:
                # Get posteriors for this state in the window
                state_posteriors = window_posteriors[window_states == state, state]
                mean_posterior = np.mean(state_posteriors)

                # Calculate weighted score
                if confidence_weight == 0.0:
                    score = float(count)
                else:
                    score = float(count) * (mean_posterior**confidence_weight)

                # Update best if score is higher and meets threshold
                if score > best_score and score >= min_score_threshold:
                    best_score = score
                    best_state = state

        smoothed[i] = best_state

    return smoothed


@njit(cache=True)
def _calculate_state_scores(window_states: np.ndarray, window_posteriors: np.ndarray, num_states: int,
                            confidence_weight: float) -> np.ndarray:
    """
    Calculate scores for each state in a window.

    Parameters
    ----------
    window_states : np.ndarray
        States in the window
    window_posteriors : np.ndarray, shape (W, K)
        Posteriors in the window
    num_states : int
        Total number of states
    confidence_weight : float
        Confidence weighting exponent

    Returns
    -------
    np.ndarray, shape (K,)
        Scores for each state
    """
    scores = np.zeros(num_states, dtype=np.float64)

    for state in range(num_states):
        count = np.sum(window_states == state)

        if count > 0:
            # Get posteriors for this state
            state_posteriors = window_posteriors[window_states == state, state]
            mean_posterior = np.mean(state_posteriors)

            # Calculate weighted score
            if confidence_weight == 0.0:
                scores[state] = float(count)
            else:
                scores[state] = float(count) * (mean_posterior**confidence_weight)

    return scores


# =============================================================================
# ModeFilterWithConfidence
# =============================================================================


class ModeFilterWithConfidence(BasePostProcessor):
    """
    Voting-based smoothing weighted by posterior probabilities.

    This filter combines frequency-based voting (mode) with HMM confidence
    (posteriors) to make more informed smoothing decisions. States with
    higher posterior probabilities have more weight in the voting.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - window_size : int, default=7
            Window size for mode calculation
        - confidence_weight : float, default=1.0
            Exponent for confidence weighting
            - 0: Ignore confidence, pure count-based
            - 1: Linear weighting by confidence
            - >1: Emphasize high-confidence observations
        - tie_breaking : str, default='confidence'
            How to break ties: 'temporal', 'confidence', 'random'
        - min_score_threshold : float, default=0.0
            Minimum score required to accept state
        - causal : bool, default=True
            If True, only use past observations (default for safe live trading)

    Warning
    -------
    Setting causal=False creates look-ahead bias (uses future observations).
    This is ONLY suitable for offline analysis, NOT for backtesting or live trading.
    Non-causal mode is deprecated and will be removed in a future version.

    Examples
    --------
    >>> # Heavy weighting by confidence
    >>> processor = ModeFilterWithConfidence({
    ...     'window_size': 7,
    ...     'confidence_weight': 2.0,  # Square the posteriors
    ...     'min_score_threshold': 1.5
    ... })
    >>> smoothed = processor.process(states, posteriors=posteriors)

    Notes
    -----
    This processor REQUIRES posteriors to function. If posteriors are not
    available, use MedianFilter instead.

    For trading applications:
    - Use higher confidence_weight when HMM is well-calibrated
    - Use lower confidence_weight when posteriors are unreliable
    - min_score_threshold helps reject low-confidence regime assignments
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ModeFilterWithConfidence."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("window_size", 7)
        config.setdefault("confidence_weight", 1.0)
        config.setdefault("tie_breaking", "confidence")
        config.setdefault("min_score_threshold", 0.0)
        config.setdefault("causal", True)

        # Warn if non-causal mode is explicitly requested
        if not config.get("causal", True):
            import warnings
            warnings.warn(
                "ModeFilterWithConfidence with causal=False uses future data (look-ahead bias). "
                "This is NOT suitable for live trading and will inflate backtesting results. "
                "causal=False is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["window_size"], int):
            raise ValueError("window_size must be an integer")
        if self.config["window_size"] < 1:
            raise ValueError("window_size must be >= 1")
        # Only validate odd window size for non-causal (deprecated) mode
        if not self.config["causal"] and self.config["window_size"] % 2 == 0:
            raise ValueError(
                "window_size must be odd for symmetric window (or use causal=True)"
            )

        if not isinstance(self.config["confidence_weight"], (int, float)):
            raise ValueError("confidence_weight must be a number")
        if self.config["confidence_weight"] < 0:
            raise ValueError("confidence_weight must be >= 0")

        if self.config["tie_breaking"] not in ["temporal", "confidence", "random"]:
            raise ValueError(
                "tie_breaking must be 'temporal', 'confidence', or 'random'"
            )

        if not isinstance(self.config["min_score_threshold"], (int, float)):
            raise ValueError("min_score_threshold must be a number")

    def process(self, states: Union[np.ndarray, pd.Series], posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        """
        Process state sequence (offline batch mode).
        Raises
        ------
        ValueError
            If posteriors are not provided
        """
        if posteriors is None:
            raise ValueError("ModeFilterWithConfidence requires posteriors")

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

        # Apply filter
        smoothed_arr = _mode_filter_with_confidence_core(
            states_arr,
            posteriors_arr,
            self.config["window_size"],
            self.config["confidence_weight"],
            self.config["min_score_threshold"],
            self.config["causal"],
        )

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
        # Online mode requires causal=True (cannot look ahead in streaming mode)
        if not self.config["causal"]:
            raise ValueError(
                "process_online() requires causal=True. "
                "Online streaming mode cannot use future observations. "
                "Set causal=True in config."
            )

        if posterior is None:
            raise ValueError("ModeFilterWithConfidence requires posteriors")

        # Initialize state on first call
        if self._online_state is None:
            num_states = len(posterior)
            self._online_state = {
                "state_buffer": np.zeros(self.config["window_size"], dtype=np.int32),
                "posterior_buffer": np.zeros(
                    (self.config["window_size"], num_states), dtype=np.float64
                ),
                "index": 0,
                "count": 0,
                "num_states": num_states,
            }

        # Add new observation to circular buffers
        buffer_idx = self._online_state["index"] % self.config["window_size"]
        self._online_state["state_buffer"][buffer_idx] = state
        self._online_state["posterior_buffer"][buffer_idx, :] = posterior
        self._online_state["index"] += 1
        self._online_state["count"] = min(
            self._online_state["count"] + 1, self.config["window_size"]
        )

        # Get active portions of buffers
        active_count = self._online_state["count"]
        if active_count < self.config["window_size"]:
            active_states = self._online_state["state_buffer"][:active_count]
            active_posteriors = self._online_state["posterior_buffer"][:active_count, :]
        else:
            # Reorder circular buffer to linear
            idx = self._online_state["index"]
            active_states = np.concatenate(
                [
                    self._online_state["state_buffer"][
                        idx % self.config["window_size"] :
                    ],
                    self._online_state["state_buffer"][
                        : idx % self.config["window_size"]
                    ],
                ]
            )
            active_posteriors = np.vstack(
                [
                    self._online_state["posterior_buffer"][
                        idx % self.config["window_size"] :, :
                    ],
                    self._online_state["posterior_buffer"][
                        : idx % self.config["window_size"], :
                    ],
                ]
            )

        # Calculate scores
        scores = _calculate_state_scores(
            active_states,
            active_posteriors,
            self._online_state["num_states"],
            self.config["confidence_weight"],
        )

        # Find best state meeting threshold
        best_score = -np.inf
        best_state = state

        for s in range(self._online_state["num_states"]):
            if (
                scores[s] > best_score
                and scores[s] >= self.config["min_score_threshold"]
            ):
                best_score = scores[s]
                best_state = s

        return int(best_state)
