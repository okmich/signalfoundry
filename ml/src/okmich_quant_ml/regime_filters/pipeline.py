from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import BasePostProcessor


class ProcessorPipeline:
    """
    Chain multiple processors in sequence.

    This class allows you to apply multiple post-processing filters in
    succession, with the output of each processor feeding into the next.

    Parameters
    ----------
    processors : list of BasePostProcessor
        List of processors to apply in sequence

    Examples
    --------
    >>> from okmich_quant_ml.hmm.postprocessor import MinimumDurationFilter, HysteresisProcessor, MedianFilter, \
    ...     ProcessorPipeline
    >>>
    >>> pipeline = ProcessorPipeline([
    ...     MinimumDurationFilter({'min_duration': 3}),
    ...     HysteresisProcessor({'entry_threshold': 5, 'exit_threshold': 2}),
    ...     MedianFilter({'window_size': 5})
    ... ])
    >>>
    >>> smoothed = pipeline.process(states, posteriors)

    Notes
    -----
    When using online mode, the pipeline maintains state across all processors. Call reset() before processing a new sequence.
    """

    def __init__(self, processors: List[BasePostProcessor]) -> None:
        """
        Initialize the processor pipeline.

        Parameters
        ----------
        processors : list of BasePostProcessor
            List of processors to apply in sequence
        """
        if not processors:
            raise ValueError("At least one processor is required")

        if not all(isinstance(p, BasePostProcessor) for p in processors):
            raise TypeError("All processors must inherit from BasePostProcessor")

        self.processors = processors

    def process(self, states: Union[np.ndarray, pd.Series],
                posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        current_states = states
        for processor in self.processors:
            current_states = processor.process(current_states, posteriors=posteriors, returns=returns)
        return current_states

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        current_state = state

        for processor in self.processors:
            current_state = processor.process_online(current_state, posterior=posterior, return_value=return_value,
                                                     timestamp=timestamp)
        return current_state

    def reset(self) -> ProcessorPipeline:
        """
        Reset all processors in the pipeline.

        Call this before processing a new sequence in online mode.

        Returns
        -------
        self
            Returns self for method chaining
        """
        for processor in self.processors:
            processor.reset()
        return self

    def get_regime_statistics(self, states: Union[np.ndarray, pd.Series]) -> dict:
        """
        Calculate regime statistics for the input sequence.

        This uses the last processor in the pipeline to calculate statistics.

        Parameters
        ----------
        states : np.ndarray or pd.Series
            State sequence to analyze

        Returns
        -------
        dict
            Regime statistics
        """
        if not self.processors:
            raise ValueError("Pipeline has no processors")
        return self.processors[-1].get_regime_statistics(states)

    def __len__(self) -> int:
        """Return number of processors in pipeline."""
        return len(self.processors)

    def __getitem__(self, index: int) -> BasePostProcessor:
        """Get processor at index."""
        return self.processors[index]

    def __repr__(self) -> str:
        """String representation of pipeline."""
        processor_names = [p.__class__.__name__ for p in self.processors]
        return f"ProcessorPipeline({processor_names})"
