from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from okmich_quant_labelling.diagnostics.regime.auto_label import _auto_labeling
from ._target_type import RegressionTargetType, REGRESSION_TARGET_TYPE_IDS
from .utils.normalizers import normalize_by_volatility, clip_by_percentile
from ._numba_kernels import _compute_auto_targets_nb
from ._shared_utils import extract_segments, ols_slope

# Supported target types for this labeler
_SUPPORTED = {
    RegressionTargetType.PERCENTAGE_FROM_EXTREME,
    RegressionTargetType.CUMULATIVE_RETURN,
    RegressionTargetType.MOMENTUM,
    RegressionTargetType.SLOPE,
    RegressionTargetType.RETURN_TO_EXTREME,
}


class AutoLabelRegression:
    """
    Generates continuous regression targets for trend movements using the same percentage-threshold detection logic as auto_label().
    Instead of discrete labels (-1, 0, +1), outputs continuous values representing the strength/magnitude of identified trends.

    Reuses the same percentage-threshold detection logic (_auto_labeling) from okmich_quant_labelling.trend.auto_label

    Parameters
    ----------
    omega : float
        Percentage threshold for trend reversal detection (e.g., 0.02 for 2%).
        Same as the classification version.
    target_type : RegressionTargetType, default=MOMENTUM
        Type of regression target to generate.
    normalize : bool, default=True
        Whether to normalize targets by global price volatility (std of log returns).
        Recommended for multi-asset scenarios.
    clip_percentile : float, optional
        If provided, clip extreme values to [100-p, p] percentile bounds.
        Applied after normalization.
    use_log_returns : bool, default=True
        Use log returns instead of simple returns for CUMULATIVE_RETURN,
        MOMENTUM, SLOPE, and RETURN_TO_EXTREME calculations.
    """

    def __init__(self, omega: float, target_type: RegressionTargetType = RegressionTargetType.MOMENTUM,
                 normalize: bool = True, clip_percentile: float = None, use_log_returns: bool = True):
        if not isinstance(target_type, RegressionTargetType):
            raise TypeError(f"target_type must be RegressionTargetType, got {type(target_type)}")
        if target_type not in _SUPPORTED:
            raise ValueError(
                f"{target_type} is not supported by AutoLabelRegression. "
                f"Supported: {sorted(t.value for t in _SUPPORTED)}"
            )
        self.omega = omega
        self.target_type = target_type
        self.normalize = normalize
        self.clip_percentile = clip_percentile
        self.use_log_returns = use_log_returns

    def label(self, series: pd.Series, return_raw_labels: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate continuous regression targets.

        Parameters
        ----------
        series : pd.Series
            Price series with any index (datetime recommended).
        return_raw_labels : bool, default=False
            If True, also return the raw discrete labels (-1, 0, +1) produced by the underlying detection algorithm.

        Returns
        -------
        pd.Series
            Continuous regression targets aligned to the input index.
            Positive values → upward trend strength.
            Negative values → downward trend strength.
            Zero → neutral (no significant trend detected yet).

        If return_raw_labels=True:
            Tuple[pd.Series, pd.Series] — (regression_targets, raw_labels).
        """
        prices = series.values.astype(float)
        timestamps = series.index.tolist()

        # Phase 1: trend detection (identical to classification version)
        raw_labels, _ = _auto_labeling(prices, timestamps, self.omega)

        # Phase 2: extract segment boundaries from the raw label array
        segments = extract_segments(raw_labels, prices)

        # Phase 3: compute continuous target for each bar
        targets = self._compute_targets(prices, raw_labels, segments)
        targets_series = pd.Series(targets, index=series.index, dtype=float)

        # Phase 4: optional normalization and clipping
        if self.normalize:
            targets_series = normalize_by_volatility(targets_series, series, use_log_returns=self.use_log_returns)

        if self.clip_percentile is not None:
            targets_series = clip_by_percentile(targets_series, self.clip_percentile)

        if return_raw_labels:
            raw_series = pd.Series(raw_labels, index=series.index, dtype=float)
            return targets_series, raw_series

        return targets_series

    def _compute_targets(self, prices: np.ndarray, raw_labels: np.ndarray, segments: list[dict]) -> np.ndarray:
        target_type_id = REGRESSION_TARGET_TYPE_IDS[self.target_type]
        segments_array = self._build_segments_array(segments)
        return _compute_auto_targets_nb(prices, segments_array, target_type_id, self.use_log_returns)

    def _compute_targets_py(self, prices: np.ndarray, raw_labels: np.ndarray, segments: list[dict]) -> np.ndarray:
        targets = np.zeros(len(prices), dtype=float)

        for seg in segments:
            start = seg["start_idx"]
            end = seg["end_idx"]
            direction = seg["direction"]
            start_price = seg["start_price"]
            extreme_price = seg["extreme_price"]
            for i in range(start, end + 1):
                targets[i] = self._target_for_bar(
                    prices, i, start, direction, start_price, extreme_price
                )

        return targets

    @staticmethod
    def _build_segments_array(segments: list[dict]) -> np.ndarray:
        """
        Convert segment list to numpy array for Numba.

        Returns array of shape (n_segments, 5) with columns:
        [start_idx, end_idx, direction, start_price, extreme_price].
        """
        n_segs = len(segments)
        arr = np.empty((n_segs, 5), dtype=float)
        for i, seg in enumerate(segments):
            arr[i, 0] = seg["start_idx"]
            arr[i, 1] = seg["end_idx"]
            arr[i, 2] = seg["direction"]
            arr[i, 3] = seg["start_price"]
            arr[i, 4] = seg["extreme_price"]
        return arr

    def _target_for_bar(self, prices: np.ndarray, i: int, start: int, direction: int, start_price: float,
                        extreme_price: float) -> float:
        p_i = prices[i]
        p_s = start_price
        if p_s == 0 or np.isnan(p_i) or np.isnan(p_s):
            return 0.0

        tt = self.target_type
        if tt == RegressionTargetType.PERCENTAGE_FROM_EXTREME:
            return direction * (p_i - p_s) / p_s
        elif tt == RegressionTargetType.CUMULATIVE_RETURN:
            if self.use_log_returns:
                return direction * np.log(p_i / p_s)
            return direction * (p_i - p_s) / p_s
        elif tt == RegressionTargetType.MOMENTUM:
            elapsed = i - start + 1
            if self.use_log_returns:
                cum_ret = np.log(p_i / p_s)
            else:
                cum_ret = (p_i - p_s) / p_s
            return direction * cum_ret / elapsed
        elif tt == RegressionTargetType.SLOPE:
            return ols_slope(prices, start, i, direction)
        elif tt == RegressionTargetType.RETURN_TO_EXTREME:
            if extreme_price == 0 or np.isnan(extreme_price):
                return 0.0
            if direction > 0:
                return np.log(extreme_price / p_i) if self.use_log_returns else (extreme_price - p_i) / p_i
            else:
                return -np.log(p_i / extreme_price) if self.use_log_returns else -(p_i - extreme_price) / extreme_price

        return 0.0
