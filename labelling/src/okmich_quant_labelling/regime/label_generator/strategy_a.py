"""
Strategy A — Pure HMM.

The HMM is both the label generator and the deployed model.  No separate
supervised classifier is trained.

generate_labels() fits the HMM on the training-fold features, runs Viterbi
decoding over the fold to produce state assignments, then maps states to the
chosen yardstick using the appropriate map_label_to_* function.  The fitted
HMM is accessible via the fitted_hmm property for live inference.

leaks_future = False
--------------------
At deployment the HMM uses InferenceMode.FILTERING (causal).  Viterbi is run
only over the training fold during label generation — this is look-ahead
*within* the training window, which is acceptable because the HMM itself is
the model, not a separate classifier being trained on those labels.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from okmich_quant_labelling.regime.threshold_optimizer import MarketPropertyType

from ._base import RegimeLabelGenerator
from ._mapping import apply_yardstick_mapping


class HmmDirectStrategy(RegimeLabelGenerator):
    """
    Strategy A — Pure HMM regime labeller.

    The HMM is fitted on training features and its Viterbi state sequence is
    mapped to the chosen yardstick.  The fitted HMM is also the model deployed
    at inference (using InferenceMode.FILTERING for online, causal prediction).

    Parameters
    ----------
    hmm : BasePomegranateHMM
        An unfitted PomegranateHMM or PomegranateMixtureHMM instance.
        ``generate_labels`` will mutate this object (fit in-place).
    yardstick : str or MarketPropertyType
        Semantic dimension through which HMM states are interpreted.
    mapping_kwargs : dict, optional
        Extra keyword arguments forwarded to the appropriate map_label_to_*
        function (e.g. ``return_col``, ``method``, ``min_samples``).

    Attributes
    ----------
    leaks_future : bool
        Always False.
    fitted_hmm
        The trained HMM instance after generate_labels() has been called.
    """

    leaks_future: bool = False

    def __init__(
        self,
        hmm,
        yardstick: Union[str, MarketPropertyType],
        mapping_kwargs: Optional[dict] = None,
    ):
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

    @property
    def fitted_hmm(self):
        """The trained HMM instance for live inference.

        Switch its inference_mode to InferenceMode.FILTERING before deploying
        to guarantee causal (online) state prediction.
        """
        return self._hmm

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
            the yardstick state-mapping step.
        X : np.ndarray
            Feature matrix aligned to df.index.  Required — the HMM is
            fitted on this array.
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
                "HmmDirectStrategy requires a feature matrix X. "
                "Pass X=<np.ndarray> to generate_labels()."
            )
        if return_col not in df.columns:
            raise KeyError(
                f"return_col '{return_col}' not found in df. "
                f"Available columns: {df.columns.tolist()}"
            )

        # ---- fit HMM ----
        self._hmm.fit(X)

        # ---- Viterbi over training fold (for state-to-yardstick mapping) ----
        from okmich_quant_ml.hmm.util import InferenceMode  # local import to avoid hard dep at module level

        original_mode = self._hmm.inference_mode
        self._hmm.inference_mode = InferenceMode.VITERBI
        try:
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
            f"HmmDirectStrategy("
            f"yardstick={self.yardstick.value!r}, "
            f"n_states={self._hmm.n_states})"
        )
