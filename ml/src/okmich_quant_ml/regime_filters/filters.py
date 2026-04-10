from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numba import njit

from .base import BasePostProcessor


@njit(cache=True)
def _minimum_duration_filter_core(
    states: np.ndarray, min_duration: int, hold_previous: bool) -> np.ndarray:
    n = len(states)
    if n == 0:
        return states.copy()

    smoothed = np.zeros(n, dtype=states.dtype)
    current_state = states[0]
    count = 1
    last_confirmed = states[0]

    # First element
    smoothed[0] = states[0]

    for i in range(1, n):
        if states[i] == current_state:
            count += 1
        else:
            # State changed
            current_state = states[i]
            count = 1

        # Check if we've met minimum duration
        if count >= min_duration:
            last_confirmed = current_state
            smoothed[i] = current_state
        else:
            if hold_previous:
                smoothed[i] = last_confirmed
            else:
                smoothed[i] = current_state
    return smoothed


@njit(cache=True)
def _mode_filter_core(states: np.ndarray, window_size: int, causal: bool) -> np.ndarray:
    n = len(states)
    if n == 0:
        return states.copy()

    smoothed = np.zeros(n, dtype=states.dtype)
    for i in range(n):
        if causal:
            # Only look backward
            start = max(0, i - window_size + 1)
            end = i + 1
        else:
            # Symmetric window
            half_window = window_size // 2
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

        # Extract window
        window = states[start:end]
        # Find mode (most frequent state)
        # Simple implementation: count each unique value
        unique_vals = np.unique(window)
        max_count = 0
        mode_val = window[0]

        for val in unique_vals:
            count = np.sum(window == val)
            if count > max_count:
                max_count = count
                mode_val = val

        smoothed[i] = mode_val

    return smoothed


# =============================================================================
# MinimumDurationFilter
# =============================================================================
class MinimumDurationFilter(BasePostProcessor):
    """
    Eliminate brief regime flickers by requiring minimum occupancy duration.

    This filter removes short-lived regime transitions that are likely noise, particularly useful for
    eliminating single-day regime spikes during news events or earnings announcements.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - min_duration : int, default=5
            Minimum consecutive periods required
        - hold_previous : bool, default=True
            If True, hold previous regime; if False, use mode in window
        - unit : str, default='periods'
            'periods' or 'time' for calendar-aware processing (future)

    Examples
    --------
    >>> processor = MinimumDurationFilter({'min_duration': 5})
    >>> raw_states = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> smoothed = processor.process(raw_states)
    >>> # Single-period spike at index 2 will be removed
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("min_duration", 5)
        config.setdefault("hold_previous", True)
        config.setdefault("unit", "periods")
        super().__init__(config)

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["min_duration"], int):
            raise ValueError("min_duration must be an integer")
        if self.config["min_duration"] < 1:
            raise ValueError("min_duration must be >= 1")
        if not isinstance(self.config["hold_previous"], bool):
            raise ValueError("hold_previous must be a boolean")
        if self.config["unit"] not in ["periods", "time"]:
            raise ValueError("unit must be 'periods' or 'time'")

    def process(self, states: Union[np.ndarray, pd.Series], posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        # Apply filter
        smoothed_arr = _minimum_duration_filter_core(
            states_arr, self.config["min_duration"], self.config["hold_previous"]
        )
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        # Initialize state on first call
        if self._online_state is None:
            self._online_state = {
                "current_state": state,
                "count": 1,
                "last_confirmed": state,
            }
            return state

        # Check if state changed
        if state == self._online_state["current_state"]:
            self._online_state["count"] += 1
        else:
            self._online_state["current_state"] = state
            self._online_state["count"] = 1

        # Check if we've met minimum duration
        if self._online_state["count"] >= self.config["min_duration"]:
            self._online_state["last_confirmed"] = state
            return state
        else:
            if self.config["hold_previous"]:
                return self._online_state["last_confirmed"]
            else:
                return state


# =============================================================================
# MedianFilter
# =============================================================================

class MedianFilter(BasePostProcessor):
    """
    Remove salt-and-pepper noise using sliding window mode filter.
    This filter replaces each state with the most frequent state in a sliding window, effectively removing isolated single-frame errors.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - window_size : int, default=5
            Window size (must be odd for symmetric window)
        - mode : str, default='reflect'
            Edge handling (currently only 'reflect' supported)
        - causal : bool, default=True
            If True, only use past observations (default for safe live trading)

    Warning
    -------
    Setting causal=False creates look-ahead bias (uses future observations).
    This is ONLY suitable for offline analysis, NOT for backtesting or live trading.
    Non-causal mode is deprecated and will be removed in a future version.

    Examples
    --------
    >>> processor = MedianFilter({'window_size': 5})
    >>> raw_states = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1])
    >>> smoothed = processor.process(raw_states)
    >>> # Isolated state=1 at index 2 will be replaced with 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            config = {}
        # Set defaults
        config.setdefault("window_size", 5)
        config.setdefault("mode", "reflect")
        config.setdefault("causal", True)

        # Warn if non-causal mode is explicitly requested
        if not config.get("causal", True):
            import warnings
            warnings.warn(
                "MedianFilter with causal=False uses future data (look-ahead bias). "
                "This is NOT suitable for live trading and will inflate backtesting results. "
                "causal=False is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2
            )

        super().__init__(config)

    def validate_config(self) -> None:
        if not isinstance(self.config["window_size"], int):
            raise ValueError("window_size must be an integer")
        if self.config["window_size"] < 1:
            raise ValueError("window_size must be >= 1")
        # Only validate odd window size for non-causal (deprecated) mode
        if not self.config["causal"] and self.config["window_size"] % 2 == 0:
            raise ValueError(
                "window_size must be odd for symmetric window (or use causal=True)"
            )
        if self.config["mode"] not in ["reflect", "constant", "wrap"]:
            raise ValueError("mode must be 'reflect', 'constant', or 'wrap'")

    def process(self, states: Union[np.ndarray, pd.Series], posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        # Extract array and index
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        # Apply filter
        smoothed_arr = _mode_filter_core(
            states_arr, self.config["window_size"], self.config["causal"]
        )
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        # Online mode requires causal=True (cannot look ahead in streaming mode)
        if not self.config["causal"]:
            raise ValueError(
                "process_online() requires causal=True. "
                "Online streaming mode cannot use future observations. "
                "Set causal=True in config."
            )

        # Initialize state on first call
        if self._online_state is None:
            self._online_state = {
                "buffer": np.zeros(self.config["window_size"], dtype=np.int32),
                "index": 0,
                "count": 0,
            }

        # Add new state to circular buffer
        buffer_idx = self._online_state["index"] % self.config["window_size"]
        self._online_state["buffer"][buffer_idx] = state
        self._online_state["index"] += 1
        self._online_state["count"] = min(
            self._online_state["count"] + 1, self.config["window_size"]
        )

        # Calculate mode of buffer
        active_buffer = self._online_state["buffer"][: self._online_state["count"]]
        unique_vals = np.unique(active_buffer)
        max_count = 0
        mode_val = state

        for val in unique_vals:
            count = np.sum(active_buffer == val)
            if count > max_count:
                max_count = count
                mode_val = val

        return int(mode_val)
