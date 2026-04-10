from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class BasePostProcessor(ABC):
    """
    Abstract base class for HMM post-inference processors.

    All processors should:
    - Accept state sequences as input (1D integer arrays)
    - Optionally accept posterior probabilities (2D arrays)
    - Support pandas Series/DataFrame with datetime index
    - Return smoothed state sequences preserving index
    - Support BOTH offline batch processing AND online streaming (low-latency)

    Parameters
    ----------
    config : dict
        Processor-specific configuration parameters

    Attributes
    ----------
    config : dict
        Configuration parameters
    _online_state : dict or None
        Internal state for online processing
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize processor with configuration parameters.

        Parameters
        ----------
        config : dict
            Processor-specific configuration parameters
        """
        self.config = config
        self.validate_config()
        self._online_state: Optional[Dict[str, Any]] = None

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises
        ------
        ValueError
            If configuration parameters are invalid
        """
        ...

    @abstractmethod
    def process(
        self,
        states: Union[np.ndarray, pd.Series],
        posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        returns: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Union[np.ndarray, pd.Series]:
        """
        Process state sequence (offline batch mode).

        Use for backtesting, research, and bulk processing.

        Parameters
        ----------
        states : np.ndarray or pd.Series, shape (T,)
            Input state sequence of length T
            If pd.Series, datetime index is preserved
        posteriors : np.ndarray or pd.DataFrame, shape (T, K), optional
            Posterior probabilities for K states at each timestep
        returns : np.ndarray or pd.Series, shape (T,), optional
            Asset returns for regime-aware processing (trading-specific)

        Returns
        -------
        np.ndarray or pd.Series, shape (T,)
            Smoothed state sequence (preserves input type/index)
        """
        ...

    @abstractmethod
    def process_online(
        self,
        state: int,
        posterior: Optional[np.ndarray] = None,
        return_value: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> int:
        """
        Process single state observation (online streaming mode - LOW LATENCY).

        Use for live trading where sub-millisecond latency is required.
        Maintains internal state between calls.

        Parameters
        ----------
        state : int
            Current state observation
        posterior : np.ndarray, shape (K,), optional
            Posterior probabilities for current observation
        return_value : float, optional
            Current asset return (trading-specific)
        timestamp : pd.Timestamp, optional
            Timestamp of observation (for time-aware processing)

        Returns
        -------
        int
            Smoothed state output

        Notes
        -----
        Must call reset() before starting a new sequence.
        Implementation should be optimized for speed:
        - Minimize allocations
        - Use circular buffers for windows
        - Pre-compute where possible
        - Target <100 microseconds per call
        """
        ...

    def reset(self) -> BasePostProcessor:
        """
        Reset internal state for online processing.

        Must be called before processing a new sequence in online mode.

        Returns
        -------
        self
            Returns self for method chaining
        """
        self._online_state = None
        return self

    def get_regime_statistics(
        self, states: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Calculate trading-relevant regime statistics.

        Parameters
        ----------
        states : np.ndarray or pd.Series
            State sequence to analyze

        Returns
        -------
        dict
            Statistics including:
            - mean_duration: Average regime duration
            - transition_frequency: Transitions per period
            - regime_stability: Measure of label consistency
            - regime_counts: Count of each regime
        """
        states_arr = states.values if isinstance(states, pd.Series) else states

        # Calculate transitions
        transitions = np.sum(states_arr[1:] != states_arr[:-1])
        transition_frequency = (
            transitions / len(states_arr) if len(states_arr) > 0 else 0
        )

        # Calculate durations
        durations = []
        if len(states_arr) > 0:
            current_state = states_arr[0]
            current_duration = 1

            for i in range(1, len(states_arr)):
                if states_arr[i] == current_state:
                    current_duration += 1
                else:
                    durations.append(current_duration)
                    current_state = states_arr[i]
                    current_duration = 1
            durations.append(current_duration)

        mean_duration = np.mean(durations) if durations else 0

        # Regime counts
        unique_states, counts = np.unique(states_arr, return_counts=True)
        regime_counts = dict(zip(unique_states.tolist(), counts.tolist()))

        # Stability measure (inverse of transition rate)
        regime_stability = 1.0 / (transition_frequency + 1e-10)

        return {
            "mean_duration": mean_duration,
            "transition_frequency": transition_frequency,
            "regime_stability": regime_stability,
            "regime_counts": regime_counts,
            "num_transitions": transitions,
        }
