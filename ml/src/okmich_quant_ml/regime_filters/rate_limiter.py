from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit

from .base import BasePostProcessor


# =============================================================================
# Numba-optimized core functions
# =============================================================================


@njit(cache=True)
def _transition_rate_limiter_core(states: np.ndarray, max_transitions: int, window_size: int, penalty_duration: int) -> np.ndarray:
    """
    Apply transition rate limiting to state sequence.

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Input state sequence
    max_transitions : int
        Maximum transitions allowed per window
    window_size : int
        Size of sliding window
    penalty_duration : int
        Periods to hold regime after limit hit

    Returns
    -------
    np.ndarray, shape (T,)
        Rate-limited state sequence
    """
    n = len(states)
    if n == 0:
        return states.copy()

    smoothed = np.zeros(n, dtype=states.dtype)
    transition_positions = np.zeros(max_transitions * 2, dtype=np.int32)
    transition_count = 0
    penalty_counter = 0
    current_regime = states[0]

    smoothed[0] = states[0]

    for i in range(1, n):
        # Decrement penalty counter
        if penalty_counter > 0:
            penalty_counter -= 1
            smoothed[i] = current_regime
            continue

        # Check if state is changing
        if states[i] != current_regime:
            # Count transitions in current window
            window_start = max(0, i - window_size)
            active_transitions = 0

            for j in range(transition_count):
                if transition_positions[j] >= window_start:
                    active_transitions += 1

            # Check if we can allow transition
            if active_transitions < max_transitions:
                # Allow transition
                current_regime = states[i]
                smoothed[i] = current_regime

                # Record transition position
                if transition_count < len(transition_positions):
                    transition_positions[transition_count] = i
                    transition_count += 1
                else:
                    # Shift buffer
                    for k in range(len(transition_positions) - 1):
                        transition_positions[k] = transition_positions[k + 1]
                    transition_positions[-1] = i
            else:
                # Reject transition - hit limit
                smoothed[i] = current_regime
                penalty_counter = penalty_duration
        else:
            # No change requested
            smoothed[i] = current_regime

    return smoothed


@njit(cache=True)
def _count_transitions_in_window(transition_positions: np.ndarray, transition_count: int, window_start: int) -> int:
    """
    Count how many transitions fall within the window.

    Parameters
    ----------
    transition_positions : np.ndarray
        Array of transition positions
    transition_count : int
        Number of valid transitions in array
    window_start : int
        Start position of window

    Returns
    -------
    int
        Number of transitions in window
    """
    count = 0
    for i in range(transition_count):
        if transition_positions[i] >= window_start:
            count += 1
    return count


# =============================================================================
# TransitionRateLimiter
# =============================================================================


class TransitionRateLimiter(BasePostProcessor):
    """
    Constrain maximum frequency of regime transitions.

    This processor limits how often regime changes can occur, which is critical for controlling trading costs and portfolio turnover.
    When the transition limit is reached, the processor enforces a penalty period where the current regime is held.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - max_transitions_per_window : int, default=2
            Maximum regime changes allowed in window
        - window_size : int, default=20
            Size of sliding window (e.g., 20 trading days ≈ 1 month)
        - penalty_duration : int, default=10
            Periods to hold regime after limit hit
        - sliding : bool, default=True
            If False, use non-overlapping windows (not yet implemented)
        - cost_aware : bool, default=False
            Weight transitions by cost (not yet implemented)

    Examples
    --------
    >>> # Limit to 2 transitions per month (20 trading days)
    >>> processor = TransitionRateLimiter({
    ...     'max_transitions_per_window': 2,
    ...     'window_size': 20,
    ...     'penalty_duration': 10
    ... })
    >>> smoothed = processor.process(noisy_regimes)

    Notes
    -----
    For trading applications:
    - Use window_size matching your rebalancing frequency
    - Set max_transitions based on acceptable turnover costs
    - penalty_duration prevents gaming the system by waiting just
      outside the window
    - Particularly useful for strategies with capacity constraints
      or high transaction costs

    Use Cases
    -----
    - **Cost-aware regime following**: Prevent excessive rebalancing
    - **Capacity constraints**: Limit switching for large AUM strategies
    - **Regulatory requirements**: Maintain minimum holding periods
    - **Risk management**: Prevent reactive over-trading
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TransitionRateLimiter."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("max_transitions_per_window", 2)
        config.setdefault("window_size", 20)
        config.setdefault("penalty_duration", 10)
        config.setdefault("sliding", True)
        config.setdefault("cost_aware", False)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["max_transitions_per_window"], int):
            raise ValueError("max_transitions_per_window must be an integer")
        if self.config["max_transitions_per_window"] < 0:
            raise ValueError("max_transitions_per_window must be >= 0")

        if not isinstance(self.config["window_size"], int):
            raise ValueError("window_size must be an integer")
        if self.config["window_size"] < 1:
            raise ValueError("window_size must be >= 1")

        if not isinstance(self.config["penalty_duration"], int):
            raise ValueError("penalty_duration must be an integer")
        if self.config["penalty_duration"] < 0:
            raise ValueError("penalty_duration must be >= 0")

        if not isinstance(self.config["sliding"], bool):
            raise ValueError("sliding must be a boolean")

        if self.config["cost_aware"]:
            # TODO: Implement cost-aware mode
            raise NotImplementedError("cost_aware mode not yet implemented")

        if not self.config["sliding"]:
            # TODO: Implement non-overlapping windows
            raise NotImplementedError("non-overlapping windows not yet implemented")

    def process(self, states: Union[np.ndarray, pd.Series],
                posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        # Apply rate limiting
        smoothed_arr = _transition_rate_limiter_core(
            states_arr,
            self.config["max_transitions_per_window"],
            self.config["window_size"],
            self.config["penalty_duration"],
        )

        # Return in same format as input
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        # Initialize state on first call
        if self._online_state is None:
            buffer_size = max(
                self.config["max_transitions_per_window"] * 2,
                100,  # Minimum buffer size
            )
            self._online_state = {
                "current_regime": state,
                "transition_positions": np.zeros(buffer_size, dtype=np.int32),
                "transition_count": 0,
                "penalty_counter": 0,
                "position": 0,
                "buffer_size": buffer_size,
            }
            return state

        # Increment position counter
        self._online_state["position"] += 1
        current_pos = self._online_state["position"]

        # Decrement penalty counter
        if self._online_state["penalty_counter"] > 0:
            self._online_state["penalty_counter"] -= 1
            return int(self._online_state["current_regime"])

        # Check if state is changing
        if state != self._online_state["current_regime"]:
            # Count transitions in current window
            window_start = max(0, current_pos - self.config["window_size"])
            active_transitions = _count_transitions_in_window(
                self._online_state["transition_positions"],
                self._online_state["transition_count"],
                window_start,
            )

            # Check if we can allow transition
            if active_transitions < self.config["max_transitions_per_window"]:
                # Allow transition
                self._online_state["current_regime"] = state

                # Record transition position
                idx = self._online_state["transition_count"]
                if idx < self._online_state["buffer_size"]:
                    self._online_state["transition_positions"][idx] = current_pos
                    self._online_state["transition_count"] += 1
                else:
                    # Shift buffer (remove oldest)
                    self._online_state["transition_positions"][:-1] = (
                        self._online_state["transition_positions"][1:]
                    )
                    self._online_state["transition_positions"][-1] = current_pos

                return int(state)
            else:
                # Reject transition - hit limit
                self._online_state["penalty_counter"] = self.config["penalty_duration"]
                return int(self._online_state["current_regime"])
        else:
            # No change requested
            return int(self._online_state["current_regime"])

    def get_transition_history(self) -> Optional[np.ndarray]:
        """
        Get transition history from online processing.

        Returns
        -------
        np.ndarray or None
            Array of transition positions, or None if not initialized
        """
        if self._online_state is None:
            return None

        count = self._online_state["transition_count"]
        return self._online_state["transition_positions"][:count].copy()
