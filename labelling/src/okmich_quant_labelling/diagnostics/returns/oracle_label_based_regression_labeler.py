from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from ._target_type import RegressionTargetType, REGRESSION_TARGET_TYPE_IDS
from ._numba_kernels import _compute_oracle_targets_nb
from ._shared_utils import extract_segments
from .utils.normalizers import normalize_by_volatility, clip_by_percentile


class OracleLabelBasedRegressionLabeler:
    """
    Generates continuous regression targets from pre-computed discrete labels.

    Unlike AmplitudeBasedRegressionLabeler and AutoLabelRegression — which internally run their own segment detectors —
    this labeler accepts a DataFrame that already contains discrete labels (-1, 0, +1) in a named column and computes
    continuous regression targets directly from those segments.

    This makes it detector-agnostic: the labels can come from any source (AmplitudeBasedLabeler, auto_label, CTL, HMM Viterbi,
    manual annotation, etc.).  Because all segment-based detectors are non-causal, the resulting targets are valid for:
        - Oracle benchmarking and upper-bound measurement
        - HMM state alignment / diagnostics
        - Visualisation

    They are NOT valid as supervised-ML training targets for live trading.

    Parameters
    ----------
    label_col : str
        Column name in the input DataFrame that contains discrete labels.
        Expected values: -1 (downtrend), 0 (neutral/no trend), +1 (uptrend).
    target_type : RegressionTargetType, default=FORWARD_RETURN
        Type of continuous target to compute per bar within each segment.
    normalize : bool, default=True
        Whether to normalize targets by global price volatility (std of log
        returns).  Recommended for multi-asset comparisons.
    clip_percentile : float, optional
        If provided, clip extreme values to the [100-p, p] percentile range
        after normalization.
    use_log_returns : bool, default=True
        Use log returns for all return-based target types.
    """

    def __init__(self, label_col: str, target_type: RegressionTargetType = RegressionTargetType.FORWARD_RETURN,
                 normalize: bool = True, clip_percentile: float = None, use_log_returns: bool = True):
        if not isinstance(target_type, RegressionTargetType):
            raise TypeError(f"target_type must be RegressionTargetType, got {type(target_type)}")
        self.label_col = label_col
        self.target_type = target_type
        self.normalize = normalize
        self.clip_percentile = clip_percentile
        self.use_log_returns = use_log_returns

    def label(self, df: pd.DataFrame, price_col: str = "close", return_raw_labels: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate continuous regression targets from pre-existing labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing price data and the label column.
        price_col : str, default="close"
            Column name containing prices.
        return_raw_labels : bool, default=False
            If True, also return the raw discrete labels as a Series.

        Returns
        -------
        pd.Series
            Continuous regression targets aligned to df.index.
            Bars with label 0 (neutral) receive a target of 0.0.

        If return_raw_labels=True:
            Tuple[pd.Series, pd.Series] — (targets, raw_labels).
        """
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}")
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}")

        prices = df[price_col].values.astype(np.float64)
        raw_labels = df[self.label_col].values.astype(np.float64)

        segments = extract_segments(raw_labels, prices)

        target_type_id = REGRESSION_TARGET_TYPE_IDS[self.target_type]
        segments_array = _build_segments_array(segments)
        targets = _compute_oracle_targets_nb(prices, segments_array, target_type_id, self.use_log_returns)

        targets_series = pd.Series(targets, index=df.index, dtype=float)

        if self.normalize:
            targets_series = normalize_by_volatility(
                targets_series, df[price_col], use_log_returns=self.use_log_returns
            )

        if self.clip_percentile is not None:
            targets_series = clip_by_percentile(targets_series, self.clip_percentile)

        if return_raw_labels:
            return targets_series, pd.Series(raw_labels, index=df.index, dtype=float)

        return targets_series


def _build_segments_array(segments: list[dict]) -> np.ndarray:
    """Convert segment list to numpy array for Numba, shape (n_segments, 5)."""
    n_segs = len(segments)
    arr = np.empty((n_segs, 5), dtype=np.float64)
    for i, seg in enumerate(segments):
        arr[i, 0] = seg["start_idx"]
        arr[i, 1] = seg["end_idx"]
        arr[i, 2] = seg["direction"]
        arr[i, 3] = seg["start_price"]
        arr[i, 4] = seg["extreme_price"]
    return arr
