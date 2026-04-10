"""
Strategy C — Causal regime labelling.

CausalRegimeLabeler produces labels using only rolling causal metrics: no
future data, no HMM, no oracle.  The labels are noisy and transition-focused,
but fully valid for live deployment.

leaks_future = False — guaranteed.
acknowledges_lookahead not required (not applicable).
"""

from typing import Optional

import numpy as np
import pandas as pd

from okmich_quant_labelling.regime.causal_regime_labeler import CausalRegimeLabeler

from ._base import RegimeLabelGenerator


class CausalStrategy(RegimeLabelGenerator):
    """
    Strategy C — Causal regime labelling via CausalRegimeLabeler.

    Wraps a CausalRegimeLabeler for use in the RegimeLabelGenerator
    interface.  No HMM, no oracle, no look-ahead.  The labeller uses rolling
    metrics only: slope/R², log-momentum, ATR/rolling-std, or efficiency ratio.

    The labels produced are always in {-1, 0, 1} (plus NaN for warmup bars),
    aligned to df.index.

    Behavioural note
    ----------------
    CausalRegimeLabeler is a regime-change detector, not a state classifier.
    Labels fire when the rolling metric is extreme relative to recent history.
    Once the new regime fills the lookback window, labels normalize toward
    neutral.  A supervised classifier trained on these labels learns to
    generalise beyond transition moments — which is the intended design.

    Parameters
    ----------
    labeler : CausalRegimeLabeler
        A configured (but not necessarily fitted) CausalRegimeLabeler instance.

    Attributes
    ----------
    leaks_future : bool
        Always False.
    """

    leaks_future: bool = False

    def __init__(self, labeler: CausalRegimeLabeler):
        if not isinstance(labeler, CausalRegimeLabeler):
            raise TypeError(
                f"Expected CausalRegimeLabeler, got {type(labeler).__name__}. "
                "Construct a CausalRegimeLabeler and pass it as `labeler`."
            )
        self.labeler = labeler

    # ------------------------------------------------------------------
    # RegimeLabelGenerator interface
    # ------------------------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """Delegates to CausalRegimeLabeler.warmup_bars."""
        return self.labeler.warmup_bars

    def generate_labels(
        self,
        df: pd.DataFrame,
        X: Optional[np.ndarray] = None,
        price_col: str = "close",
        return_col: str = "returns",
    ) -> pd.Series:
        """
        Apply CausalRegimeLabeler to the training fold.

        Parameters
        ----------
        df : pd.DataFrame
            Training fold OHLCV DataFrame.  Must contain price_col.
            For volatility with use_atr=True, must also contain 'high' and 'low'.
        X : np.ndarray, optional
            Not used by this strategy.
        price_col : str, default='close'
            Forwarded to CausalRegimeLabeler.label().
        return_col : str, default='returns'
            Not used by this strategy.

        Returns
        -------
        pd.Series
            Regime labels in {-1, 0, 1} aligned to df.index.
            First warmup_bars entries are NaN.
        """
        return self.labeler.label(df, price_col=price_col)

    def __repr__(self) -> str:
        return f"CausalStrategy(labeler={self.labeler!r})"