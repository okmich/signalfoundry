"""
Strategy B2 — HMM Viterbi distillation.

An HMM is fitted on training features.  Its Viterbi sequence over the training
window is mapped to yardstick labels, which become the training targets for a
supervised classifier.  At inference, only the classifier runs — the HMM is
NOT deployed.

This produces stable, semantically meaningful labels (always {-1,0,1} for
trend_direction after mapping) that do not depend on raw HMM state indices,
which can shift across retraining folds.

leaks_future = True
-------------------
Viterbi over the full training sequence is non-causal within that sequence:
bar i's label is influenced by bars i+1 … T.  The classifier is therefore
trained on labels that embed look-ahead within the training fold.

Requires acknowledges_lookahead=True.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from okmich_quant_labelling.regime.threshold_optimizer import MarketPropertyType

from ._base import RegimeLabelGenerator
from ._mapping import apply_yardstick_mapping


class HmmViterbiDistillationStrategy(RegimeLabelGenerator):
    """
    Strategy B2 — HMM Viterbi distillation.

    The HMM is used only as a label generator; it is not deployed at inference.
    A supervised classifier is trained on the mapped Viterbi labels and is the
    sole component used at live inference time.

    Parameters
    ----------
    hmm : BasePomegranateHMM
        An unfitted PomegranateHMM or PomegranateMixtureHMM instance.
        ``generate_labels`` will mutate this object (fit in-place).
    yardstick : str or MarketPropertyType
        Semantic dimension to map Viterbi states into.
    acknowledges_lookahead : bool
        Must be True.  Raises ValueError otherwise.
    mapping_kwargs : dict, optional
        Extra keyword arguments forwarded to the appropriate map_label_to_*
        function (e.g. ``return_col``, ``method``, ``min_samples``).

    Attributes
    ----------
    leaks_future : bool
        Always True.
    """

    leaks_future: bool = True

    def __init__(
        self,
        hmm,
        yardstick: Union[str, MarketPropertyType],
        acknowledges_lookahead: bool,
        mapping_kwargs: Optional[dict] = None,
    ):
        self._check_lookahead(acknowledges_lookahead, self.__class__.__name__)
        self._hmm = hmm
        self.yardstick = MarketPropertyType(yardstick)
        self.mapping_kwargs = mapping_kwargs or {}

    # ------------------------------------------------------------------
    # RegimeLabelGenerator interface
    # ------------------------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        """HMM has no fixed warmup period; returns 0."""
        return 0

    def generate_labels(
        self,
        df: pd.DataFrame,
        X: Optional[np.ndarray] = None,
        price_col: str = "close",
        return_col: str = "returns",
    ) -> pd.Series:
        """
        Fit the HMM, run Viterbi, and map states to yardstick labels.

        Parameters
        ----------
        df : pd.DataFrame
            Training fold OHLCV DataFrame.  Must contain return_col for
            the state-to-yardstick mapping step.
        X : np.ndarray
            Feature matrix aligned to df.index.  Required.
        price_col : str, default='close'
            Unused by this strategy; kept for interface compatibility.
        return_col : str, default='returns'
            Column in df used to compute the state→yardstick mapping.

        Returns
        -------
        pd.Series
            Yardstick labels aligned to df.index.
        """
        if X is None:
            raise ValueError(
                "HmmViterbiDistillationStrategy requires a feature matrix X. "
                "Pass X=<np.ndarray> to generate_labels()."
            )
        if return_col not in df.columns:
            raise KeyError(
                f"return_col '{return_col}' not found in df. "
                f"Available columns: {df.columns.tolist()}"
            )

        # ---- Fit HMM then run Viterbi ----
        from okmich_quant_ml.hmm.util import InferenceMode  # local import

        original_mode = self._hmm.inference_mode
        self._hmm.inference_mode = InferenceMode.VITERBI
        try:
            self._hmm.fit(X)
            states_arr = self._hmm.predict(X)
        finally:
            self._hmm.inference_mode = original_mode

        states = pd.Series(states_arr, index=df.index, name="_state")

        return apply_yardstick_mapping(
            df,
            states,
            self.yardstick,
            return_col=return_col,
            **self.mapping_kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"HmmViterbiDistillationStrategy("
            f"yardstick={self.yardstick.value!r}, "
            f"n_states={self._hmm.n_states})"
        )
