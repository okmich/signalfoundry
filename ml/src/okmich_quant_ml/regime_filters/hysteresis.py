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
def _hysteresis_count_based_core(states: np.ndarray, entry_threshold: int, exit_threshold: int) -> np.ndarray:
    """
    Apply count-based hysteresis to state sequence.

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Input state sequence
    entry_threshold : int
        Consecutive observations required to enter regime
    exit_threshold : int
        Consecutive violations required to exit regime

    Returns
    -------
    np.ndarray, shape (T,)
        Filtered state sequence
    """
    n = len(states)
    if n == 0:
        return states.copy()

    smoothed = np.zeros(n, dtype=states.dtype)
    current_regime = states[0]
    entry_count = 1
    exit_count = 0

    smoothed[0] = current_regime

    for i in range(1, n):
        if states[i] == current_regime:
            # Same as current regime - reset exit counter
            exit_count = 0
            smoothed[i] = current_regime
        else:
            # Different from current regime
            exit_count += 1

            if exit_count >= exit_threshold:
                # Exit threshold met, try to enter new regime
                # Check if we have enough consecutive observations for entry
                entry_count = 1
                for j in range(i - 1, max(-1, i - entry_threshold), -1):
                    if states[j] == states[i]:
                        entry_count += 1
                    else:
                        break

                if entry_count >= entry_threshold:
                    # Enter new regime
                    current_regime = states[i]
                    exit_count = 0
                    smoothed[i] = current_regime
                else:
                    # Not enough for entry, stay in current
                    smoothed[i] = current_regime
            else:
                # Haven't reached exit threshold yet
                smoothed[i] = current_regime

    return smoothed


@njit(cache=True)
def _hysteresis_confidence_based_core(states: np.ndarray, posteriors: np.ndarray,
                                      entry_threshold: float, exit_threshold: float) -> np.ndarray:
    """
    Apply confidence-based hysteresis to state sequence.

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Input state sequence
    posteriors : np.ndarray, shape (T, K)
        Posterior probabilities for K states
    entry_threshold : float
        Cumulative confidence required to enter regime
    exit_threshold : float
        Cumulative confidence loss required to exit regime

    Returns
    -------
    np.ndarray, shape (T,)
        Filtered state sequence
    """
    n = len(states)
    if n == 0:
        return states.copy()

    smoothed = np.zeros(n, dtype=states.dtype)
    current_regime = states[0]
    entry_confidence = posteriors[0, states[0]]
    exit_confidence = 0.0

    smoothed[0] = current_regime

    for i in range(1, n):
        regime_prob = posteriors[i, current_regime]

        if states[i] == current_regime:
            # Same regime - reset exit confidence
            exit_confidence = 0.0
            smoothed[i] = current_regime
        else:
            # Different regime
            # Accumulate exit confidence (1 - current_regime_prob)
            exit_confidence += 1.0 - regime_prob

            if exit_confidence >= exit_threshold:
                # Consider entering new regime
                # Accumulate entry confidence for new regime
                new_regime = states[i]
                entry_conf = 0.0

                for j in range(i, -1, -1):
                    if states[j] == new_regime:
                        entry_conf += posteriors[j, new_regime]
                        if entry_conf >= entry_threshold:
                            # Enter new regime
                            current_regime = new_regime
                            exit_confidence = 0.0
                            entry_confidence = entry_conf
                            break
                    else:
                        break

            smoothed[i] = current_regime

    return smoothed


# =============================================================================
# HysteresisProcessor
# =============================================================================


class HysteresisProcessor(BasePostProcessor):
    """
    Create 'sticky' regimes with different entry/exit thresholds.

    This processor models asymmetric market behavior (e.g., "stairs up,
    elevator down") by requiring different levels of evidence to enter
    vs. exit a regime.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - entry_threshold : int or float, default=10
            Periods or confidence required to enter regime
        - exit_threshold : int or float, default=3
            Periods or confidence required to exit regime
        - use_confidence : bool, default=False
            If True, use posteriors; if False, use counts
        - per_state_params : dict, optional
            Regime-specific thresholds mapping state -> {'entry': x, 'exit': y}

    Examples
    --------
    >>> # Asymmetric thresholds for bull/bear regimes
    >>> config = {
    ...     'per_state_params': {
    ...         0: {'entry': 15, 'exit': 5},   # Bull: slow entry, fast exit
    ...         1: {'entry': 5, 'exit': 10},   # Bear: fast entry, slow exit
    ...     }
    ... }
    >>> processor = HysteresisProcessor(config)

    Notes
    -----
    For trading applications:
    - Bull markets: Higher entry threshold (resist entering too early),
      lower exit threshold (exit quickly on weakness)
    - Bear markets: Lower entry threshold (detect crashes quickly),
      higher exit threshold (avoid whipsaws in recovery)
    - Crisis: Very low entry threshold, very high exit threshold
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize HysteresisProcessor."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("entry_threshold", 10)
        config.setdefault("exit_threshold", 3)
        config.setdefault("use_confidence", False)
        config.setdefault("per_state_params", None)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["use_confidence"], bool):
            raise ValueError("use_confidence must be a boolean")

        # Validate thresholds
        entry = self.config["entry_threshold"]
        exit_val = self.config["exit_threshold"]

        if self.config["use_confidence"]:
            if not isinstance(entry, (int, float)) or entry <= 0:
                raise ValueError("entry_threshold must be positive number")
            if not isinstance(exit_val, (int, float)) or exit_val <= 0:
                raise ValueError("exit_threshold must be positive number")
        else:
            if not isinstance(entry, int) or entry < 1:
                raise ValueError("entry_threshold must be positive integer")
            if not isinstance(exit_val, int) or exit_val < 1:
                raise ValueError("exit_threshold must be positive integer")

        # Validate per_state_params if provided
        if self.config["per_state_params"] is not None:
            if not isinstance(self.config["per_state_params"], dict):
                raise ValueError("per_state_params must be a dict")

    def _get_thresholds_for_state(self, state: int) -> tuple[Union[int, float], Union[int, float]]:
        """Get entry/exit thresholds for a specific state."""
        if self.config["per_state_params"] is not None:
            state_params = self.config["per_state_params"].get(state)
            if state_params is not None:
                return state_params["entry"], state_params["exit"]

        return self.config["entry_threshold"], self.config["exit_threshold"]

    def _count_based_per_state(self, states_arr: np.ndarray) -> np.ndarray:
        """Count-based hysteresis with per-state entry/exit thresholds."""
        n = len(states_arr)
        if n == 0:
            return states_arr.copy()

        smoothed = np.zeros(n, dtype=states_arr.dtype)
        current_regime = states_arr[0]
        exit_count = 0
        smoothed[0] = current_regime

        for i in range(1, n):
            if states_arr[i] == current_regime:
                exit_count = 0
                smoothed[i] = current_regime
            else:
                _, exit_thresh = self._get_thresholds_for_state(current_regime)
                exit_count += 1

                if exit_count >= exit_thresh:
                    proposed = states_arr[i]
                    entry_thresh, _ = self._get_thresholds_for_state(proposed)
                    entry_count = 1
                    for j in range(i - 1, max(-1, i - entry_thresh), -1):
                        if states_arr[j] == proposed:
                            entry_count += 1
                        else:
                            break
                    if entry_count >= entry_thresh:
                        current_regime = proposed
                        exit_count = 0
                        smoothed[i] = current_regime
                    else:
                        smoothed[i] = current_regime
                else:
                    smoothed[i] = current_regime

        return smoothed

    def _confidence_based_per_state(self, states_arr: np.ndarray, posteriors_arr: np.ndarray) -> np.ndarray:
        """Confidence-based hysteresis with per-state entry/exit thresholds."""
        n = len(states_arr)
        if n == 0:
            return states_arr.copy()

        smoothed = np.zeros(n, dtype=states_arr.dtype)
        current_regime = states_arr[0]
        exit_confidence = 0.0
        smoothed[0] = current_regime

        for i in range(1, n):
            regime_prob = posteriors_arr[i, current_regime]

            if states_arr[i] == current_regime:
                exit_confidence = 0.0
                smoothed[i] = current_regime
            else:
                _, exit_thresh = self._get_thresholds_for_state(current_regime)
                exit_confidence += 1.0 - regime_prob

                if exit_confidence >= exit_thresh:
                    new_regime = states_arr[i]
                    entry_thresh, _ = self._get_thresholds_for_state(new_regime)
                    entry_conf = 0.0
                    for j in range(i, -1, -1):
                        if states_arr[j] == new_regime:
                            entry_conf += posteriors_arr[j, new_regime]
                            if entry_conf >= entry_thresh:
                                current_regime = new_regime
                                exit_confidence = 0.0
                                break
                        else:
                            break

                smoothed[i] = current_regime

        return smoothed

    def process(self, states: Union[np.ndarray, pd.Series], posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None, returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        # Extract array and index
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        has_per_state = self.config["per_state_params"] is not None

        # Check if using confidence mode
        if self.config["use_confidence"]:
            if posteriors is None:
                raise ValueError("posteriors required when use_confidence=True")

            if isinstance(posteriors, pd.DataFrame):
                posteriors_arr = posteriors.values
            else:
                posteriors_arr = posteriors

            if has_per_state:
                smoothed_arr = self._confidence_based_per_state(states_arr, posteriors_arr)
            else:
                smoothed_arr = _hysteresis_confidence_based_core(states_arr, posteriors_arr, self.config["entry_threshold"], self.config["exit_threshold"])
        else:
            # Count-based mode
            if has_per_state:
                smoothed_arr = self._count_based_per_state(states_arr)
            else:
                smoothed_arr = _hysteresis_count_based_core(states_arr, self.config["entry_threshold"], self.config["exit_threshold"])

        # Return in same format as input
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        # Initialize state on first call
        if self._online_state is None:
            self._online_state = {
                "current_regime": state,
                "entry_count": 1,
                "exit_count": 0,
                "entry_confidence": posterior[state] if posterior is not None else 1.0,
                "exit_confidence": 0.0,
                "recent_states": [],
                "recent_posteriors": [],
            }
            return state

        current_regime = self._online_state["current_regime"]
        entry_thresh, _ = self._get_thresholds_for_state(state)
        _, exit_thresh = self._get_thresholds_for_state(current_regime)

        if self.config["use_confidence"]:
            if posterior is None:
                raise ValueError("posterior required when use_confidence=True")

            regime_prob = posterior[current_regime]

            if state == current_regime:
                self._online_state["exit_confidence"] = 0.0
                return current_regime
            else:
                # Accumulate exit confidence
                self._online_state["exit_confidence"] += 1.0 - regime_prob

                if self._online_state["exit_confidence"] >= exit_thresh:
                    # Check entry confidence for new regime
                    # Need to track recent observations
                    self._online_state["recent_states"].append(state)
                    self._online_state["recent_posteriors"].append(posterior)

                    # Calculate entry confidence
                    entry_conf = sum(
                        post[state]
                        for s, post in zip(
                            self._online_state["recent_states"],
                            self._online_state["recent_posteriors"],
                        )
                        if s == state
                    )

                    if entry_conf >= entry_thresh:
                        # Enter new regime
                        self._online_state["current_regime"] = state
                        self._online_state["exit_confidence"] = 0.0
                        self._online_state["entry_confidence"] = entry_conf
                        self._online_state["recent_states"] = []
                        self._online_state["recent_posteriors"] = []
                        return state

                return current_regime
        else:
            # Count-based mode
            if state == current_regime:
                self._online_state["exit_count"] = 0
                return current_regime
            else:
                self._online_state["exit_count"] += 1

                if self._online_state["exit_count"] >= exit_thresh:
                    # Track recent states for entry check
                    self._online_state["recent_states"].append(state)

                    # Keep only recent window
                    max_lookback = int(entry_thresh * 2)
                    if len(self._online_state["recent_states"]) > max_lookback:
                        self._online_state["recent_states"] = self._online_state[
                            "recent_states"
                        ][-max_lookback:]

                    # Count consecutive observations of new state
                    entry_count = 0
                    for s in reversed(self._online_state["recent_states"]):
                        if s == state:
                            entry_count += 1
                        else:
                            break

                    if entry_count >= entry_thresh:
                        # Enter new regime
                        self._online_state["current_regime"] = state
                        self._online_state["exit_count"] = 0
                        self._online_state["entry_count"] = entry_count
                        self._online_state["recent_states"] = []
                        return state

                return current_regime
