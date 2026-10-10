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


# =============================================================================
# HysteresisProcessor
# =============================================================================


class HysteresisProcessor(BasePostProcessor):
    """
    Create 'sticky' regimes with different entry/exit thresholds (count-based).

    Operates purely on label sequences. The incumbent regime is held until a challenger qualifies; at bar ``t`` the
    label switches from incumbent ``a`` to the observed state ``b`` only when both hold:

    * exit:  ``a`` has been absent for at least ``exit(a)`` consecutive bars, and
    * entry: ``b`` has been observed for at least ``entry(b)`` consecutive bars ending at ``t``.

    There is no exit to a neutral state: until a challenger qualifies, the incumbent is held. With two states the
    absence run and the challenger run are the same run, so the switch delay is ``max(exit(a), entry(b))`` and an exit
    threshold at or below the challenger's entry threshold has no effect. ``process_online`` applies the same rule bar
    by bar, so streaming a sequence returns exactly what ``process`` returns for it.

    This is a timer on labels. For a probability band on the posterior (hold the incumbent until a challenger's
    posterior clears an entry bar or the incumbent's collapses below an exit bar) use
    ``okmich_quant_ml.posterior_inference.DeadbandHysteresisInferer``; for cumulative-confidence thresholds use
    ``okmich_quant_ml.posterior_inference.ConfidenceHysteresisInferer``.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - entry_threshold : int, default=10
            Consecutive observations of a challenger required before it can take over
        - exit_threshold : int, default=3
            Consecutive bars without the incumbent required before any switch
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
    The delay to leave ``a`` for ``b`` is ``max(exit(a), entry(b))`` in the two-state case, so in the example above
    bull -> bear takes 5 bars (max(5, 5)) and bear -> bull takes 15 (max(10, 15)). "Fast exit" only shortens a
    switch when the challenger's entry threshold is also short.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize HysteresisProcessor."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("entry_threshold", 10)
        config.setdefault("exit_threshold", 3)
        config.setdefault("per_state_params", None)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        entry = self.config["entry_threshold"]
        exit_val = self.config["exit_threshold"]

        if not isinstance(entry, int) or entry < 1:
            raise ValueError("entry_threshold must be positive integer")
        if not isinstance(exit_val, int) or exit_val < 1:
            raise ValueError("exit_threshold must be positive integer")

        # Validate per_state_params strictly so misconfigurations fail at construction time
        # rather than producing undefined behavior at process() time.
        per_state = self.config["per_state_params"]
        if per_state is not None:
            if not isinstance(per_state, dict):
                raise ValueError("per_state_params must be a dict")
            for state_key, state_cfg in per_state.items():
                if not isinstance(state_cfg, dict):
                    raise ValueError(f"per_state_params[{state_key!r}] must be a dict with 'entry' and 'exit' keys.")
                missing = {"entry", "exit"} - set(state_cfg.keys())
                if missing:
                    raise ValueError(
                        f"per_state_params[{state_key!r}] is missing required keys: {sorted(missing)}."
                    )
                state_entry = state_cfg["entry"]
                state_exit = state_cfg["exit"]
                if not isinstance(state_entry, int) or state_entry < 1:
                    raise ValueError(
                        f"per_state_params[{state_key!r}]['entry'] must be a positive integer, got {state_entry!r}."
                    )
                if not isinstance(state_exit, int) or state_exit < 1:
                    raise ValueError(
                        f"per_state_params[{state_key!r}]['exit'] must be a positive integer, got {state_exit!r}."
                    )

    def _get_thresholds_for_state(self, state: int) -> tuple[int, int]:
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

    def process(self, states: Union[np.ndarray, pd.Series],
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        # Extract array and index
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        has_per_state = self.config["per_state_params"] is not None

        if has_per_state:
            smoothed_arr = self._count_based_per_state(states_arr)
        else:
            smoothed_arr = _hysteresis_count_based_core(
                states_arr, self.config["entry_threshold"], self.config["exit_threshold"],
            )

        # Return in same format as input
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        """Streaming counterpart of ``process``: same rule, same output, one state per call."""
        online = self._online_state
        if online is None:
            self._online_state = {"current_regime": state, "exit_count": 0, "run_state": state, "run_length": 1}
            return state

        # Length of the run of identical observations ending now (the entry evidence for `state`).
        if state == online["run_state"]:
            online["run_length"] += 1
        else:
            online["run_state"] = state
            online["run_length"] = 1

        current_regime = online["current_regime"]
        if state == current_regime:
            online["exit_count"] = 0
            return current_regime

        online["exit_count"] += 1
        entry_thresh, _ = self._get_thresholds_for_state(state)
        _, exit_thresh = self._get_thresholds_for_state(current_regime)
        if online["exit_count"] >= exit_thresh and online["run_length"] >= entry_thresh:
            online["current_regime"] = state
            online["exit_count"] = 0
        return online["current_regime"]
