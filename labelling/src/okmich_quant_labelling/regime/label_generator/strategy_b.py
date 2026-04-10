"""
Strategy B — Oracle distillation.

An oracle labeller (look-ahead, leaks_future=True) generates labels over the
training fold.  A supervised classifier is then trained on these labels.
At inference, only the classifier runs — no oracle access.

The oracle is expected to expose::

    label(df, price_col='close') -> pd.Series

Any labeller in okmich_quant_labelling.diagnostics.regime satisfies this
interface (AmplitudeBasedLabeler, CTL, optimal_trend, auto_label).

leaks_future = True
-------------------
The oracle uses future price data to segment or score past bars.
This is acknowledged explicitly via acknowledges_lookahead=True, which is
required in the constructor — the framework raises an error otherwise.
"""

from typing import Optional

import numpy as np
import pandas as pd

from ._base import RegimeLabelGenerator


class OracleDistillationStrategy(RegimeLabelGenerator):
    """
    Strategy B — Oracle distillation.

    Wraps any oracle labeller that exposes
    ``label(df, price_col='close') -> pd.Series``.

    The oracle's output is used directly as training targets for a supervised
    classifier.  No yardstick mapping is applied here — the oracle is expected
    to produce labels that are already in the semantic space of the chosen
    yardstick (e.g. {-1, 0, 1} for direction-type oracles).

    Parameters
    ----------
    oracle : object
        Any labeller with a ``label(df, price_col='close') -> pd.Series``
        method.  Examples: AmplitudeBasedLabeler, bi_ctl_label (wrapped),
        bi_oracle_label (wrapped).
    acknowledges_lookahead : bool
        Must be True.  Raises ValueError otherwise.  Explicit acknowledgement
        that look-ahead labelling is being used is required by the framework.
    warmup_bars_override : int, default=0
        If the oracle labeller has a known warm-up period (e.g. a minimum
        amplitude window), set it here so walk-forward loops can apply the
        correct embargo at fold boundaries.

    Attributes
    ----------
    leaks_future : bool
        Always True.
    """

    leaks_future: bool = True

    def __init__(
        self,
        oracle,
        acknowledges_lookahead: bool,
        warmup_bars_override: int = 0,
    ):
        self._check_lookahead(acknowledges_lookahead, self.__class__.__name__)
        self.oracle = oracle
        self._warmup_bars = warmup_bars_override

    # ------------------------------------------------------------------
    # RegimeLabelGenerator interface
    # ------------------------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def generate_labels(
        self,
        df: pd.DataFrame,
        X: Optional[np.ndarray] = None,
        price_col: str = "close",
        return_col: str = "returns",
    ) -> pd.Series:
        """
        Run the oracle labeller on the training fold.

        Parameters
        ----------
        df : pd.DataFrame
            Training fold DataFrame.  The oracle labeller may require OHLCV
            columns depending on its implementation.
        X : np.ndarray, optional
            Not used by this strategy.
        price_col : str, default='close'
            Forwarded to oracle.label().
        return_col : str, default='returns'
            Not used by this strategy.

        Returns
        -------
        pd.Series
            Labels produced by the oracle, aligned to df.index.
        """
        labels = self.oracle.label(df, price_col=price_col)
        labels.name = "regime_label"
        return labels

    def __repr__(self) -> str:
        return (
            f"OracleDistillationStrategy("
            f"oracle={self.oracle.__class__.__name__})"
        )