from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


class BaseSignal:
    """
    Base class for trading signal generation.
    """

    def __init__(self, **params):
        self.params = params

    def generate(self, data: pd.DataFrame, *args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate entry/exit signals for given data.

        This method maintains a simple interface for compatibility with vectorbt and backtesting frameworks.
        It returns binary signals (0/1) as numpy arrays.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Tuple of (entries_long, exits_long, entries_short, exits_short)
            Each array contains 0 (no signal) or 1 (signal) for each bar
            :param *args:
        """
        raise NotImplementedError("Subclasses must implement generate() method")

    def get_signal_context(self) -> Optional[Dict[str, Any]]:
        """
        Get rich context information for last generated signal.

        This method is optional and allows signal implementations to provide additional metadata, confidence scores,
        indicator values, and reasoning for the signals generated. Useful for live trading, logging, and analysis.

        The return format is intentionally flexible - implementations can return whatever context is relevant
        (simple dict, nested structures, DataFrame, etc.)

        Common suggested keys (conventions, not required):
            - 'strength': float (0.0-1.0) - signal strength/magnitude
            - 'confidence': float (0.0-1.0) - confidence in the signal
            - 'indicators': dict - indicator values that triggered signal
            - 'reasons': list[str] - human-readable reasons for signal
            - 'timeframe': str - timeframe of the signal
            - 'metadata': dict - any additional implementation-specific data

        Returns:
            Dict with signal context, or None if:
            - No signal at the specified index
            - Implementation doesn't provide context
            - Context not available/applicable
        """
        return None
