"""
Abstract base for RegimeLabelGenerator.

Defines the interface that every labelling strategy must satisfy.
All strategies share the same generate_labels() contract so that the walk-forward training loop can treat them polymorphically.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class RegimeLabelGenerator(ABC):
    """
    Abstract base for regime label generators.

    Each subclass implements one labelling strategy (A, B, B2, or C) and is responsible for the label-generation step of the regime classification
    pipeline.  The downstream supervised classifier (or the HMM itself for
    Strategy A) is trained on the returned labels.

    The common interface is::

        labels = generator.generate_labels(df, X, price_col, return_col)

    Subclasses declare leaks_future as a class-level bool to make the
    look-ahead status visible without instantiation.

    Attributes
    ----------
    leaks_future : bool
        True if label generation uses look-ahead (future) data.
        Strategies B and B2 set this to True and require
        acknowledges_lookahead=True in their constructors.
    """

    leaks_future: bool  # set as a class attribute in each concrete subclass

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Minimum bars required before the first valid label is produced.

        Use this as embargo_bars in walk-forward loops to prevent label
        leakage at fold boundaries.
        """
        ...

    @abstractmethod
    def generate_labels(
        self,
        df: pd.DataFrame,
        X: Optional[np.ndarray] = None,
        price_col: str = "close",
        return_col: str = "returns",
    ) -> pd.Series:
        """
        Generate regime labels for a training fold.

        Parameters
        ----------
        df : pd.DataFrame
            Training fold OHLCV DataFrame.  Must contain price_col.
            For strategies that map HMM states to yardstick labels (A, B2),
            must also contain return_col.
        X : np.ndarray, optional
            Feature matrix aligned to df.index.
            Required by HMM-based strategies (A, B2); ignored by B and C.
        price_col : str, default='close'
            Column name for close prices.
        return_col : str, default='returns'
            Column name for log returns, used for yardstick state mapping.

        Returns
        -------
        pd.Series
            Regime labels aligned to df.index.  The value range depends on
            the yardstick and strategy:
            - Strategy C (CausalRegimeLabeler): always {-1, 0, 1}.
            - Strategies A/B2 after yardstick mapping: depends on
              map_label_to_* output (e.g. {-1, 0, 1} for trend_direction).
            First warmup_bars entries may be NaN.
        """
        ...

    @staticmethod
    def _check_lookahead(acknowledges_lookahead: bool, class_name: str) -> None:
        """Raise ValueError if look-ahead use is not explicitly acknowledged."""
        if not acknowledges_lookahead:
            raise ValueError(
                f"{class_name} generates labels with look-ahead (leaks_future=True). "
                "Pass acknowledges_lookahead=True to confirm this is intentional "
                "and understood."
            )
