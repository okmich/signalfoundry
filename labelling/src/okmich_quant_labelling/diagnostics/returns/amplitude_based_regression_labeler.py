from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

from okmich_quant_labelling.diagnostics.regime.amplitude_based_labeler import AmplitudeBasedLabeler, PriceType
from ._target_type import RegressionTargetType, REGRESSION_TARGET_TYPE_IDS
from .utils.normalizers import normalize_by_volatility, clip_by_percentile
from ._numba_kernels import _compute_amp_targets_nb
from ._shared_utils import extract_segments, ols_slope

# Supported target types for this labeler
_SUPPORTED = {
    RegressionTargetType.SLOPE,
    RegressionTargetType.MOMENTUM,
    RegressionTargetType.CUMULATIVE_RETURN,
    RegressionTargetType.AMPLITUDE_PER_BAR,
    RegressionTargetType.FORWARD_RETURN,
    RegressionTargetType.FORWARD_RETURN_PER_BAR,
    RegressionTargetType.RETURN_TO_EXTREME,
}


class AmplitudeBasedRegressionLabeler:
    """
    Generates continuous regression targets for trend movements using the same amplitude-based detection logic as AmplitudeBasedLabeler.
    Instead of discrete labels (-1, 0, +1), outputs continuous values representing the strength/magnitude of identified trends.

    Parameters
    ----------
    minamp : float
        Minimum amplitude of move (usually in bps). Same as classification version.
    Tinactive : int
        Maximum inactive period where no new high (low) is achieved (unit: # of samples).
        Same as classification version.
    target_type : RegressionTargetType, default=SLOPE
        Type of regression target to generate.
    normalize : bool, default=True
        Whether to normalize targets by global price volatility (std of log returns).
        Recommended for multi-asset scenarios.
    clip_percentile : float, optional
        If provided, clip extreme values to [100-p, p] percentile bounds.
        Applied after normalization.
    use_log_returns : bool, default=True
        Use log returns for CUMULATIVE_RETURN, MOMENTUM, SLOPE, FORWARD_RETURN,
        FORWARD_RETURN_PER_BAR, and RETURN_TO_EXTREME calculations.
    """

    def __init__(self, minamp: float, Tinactive: int, target_type: RegressionTargetType = RegressionTargetType.SLOPE,
                 normalize: bool = True, clip_percentile: float = None, use_log_returns: bool = True):
        if not isinstance(target_type, RegressionTargetType):
            raise TypeError(f"target_type must be RegressionTargetType, got {type(target_type)}")
        if target_type not in _SUPPORTED:
            raise ValueError(
                f"{target_type} is not supported by AmplitudeBasedRegressionLabeler. "
                f"Supported: {sorted(t.value for t in _SUPPORTED)}"
            )
        self.minamp = minamp
        self.Tinactive = Tinactive
        self.target_type = target_type
        self.normalize = normalize
        self.clip_percentile = clip_percentile
        self.use_log_returns = use_log_returns

        # Internal labeler reused for detection only
        self._detector = AmplitudeBasedLabeler(minamp=minamp, Tinactive=Tinactive)

    def label(self, df: pd.DataFrame, price_col: str = "close", scale: float = 1e4, return_raw_labels: bool = False) \
            -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate continuous regression targets.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing price data.
        price_col : str, default="close"
            Column name containing prices.
        scale : float, default=1e4
            Basis-points scale used internally for cumr computation.
            Same as the classification version.
        return_raw_labels : bool, default=False
            If True, also return the raw discrete labels (-1, 0, +1) for
            comparison / validation.

        Returns
        -------
        pd.Series
            Continuous regression targets aligned to df.index.
            Positive values → upward trend strength.
            Negative values → downward trend strength.
            Zero → neutral (no significant trend).

        If return_raw_labels=True:
            Tuple[pd.Series, pd.Series] — (regression_targets, raw_labels).
        """
        prices = df[price_col]
        cumr = PriceType.PRICE.toBps(prices, scale=scale)
        n = len(cumr)

        # Phase 1 & 2: detection — identical to classification version
        raw_labels = np.zeros(n, dtype=np.double)
        self._detector._pass1(cumr, raw_labels)
        self._detector._filter(cumr, raw_labels)

        # Phase 3: extract segment boundaries
        prices_arr = prices.values.astype(float)
        cumr_arr = cumr if isinstance(cumr, np.ndarray) else np.array(cumr)
        segments = extract_segments(raw_labels, prices_arr)

        # Phase 4: compute continuous target for each bar
        targets = self._compute_targets(prices_arr, cumr_arr, raw_labels, segments)
        targets_series = pd.Series(targets, index=df.index, dtype=float)

        # Phase 5: optional normalization and clipping
        if self.normalize:
            targets_series = normalize_by_volatility(
                targets_series, prices, use_log_returns=self.use_log_returns
            )

        if self.clip_percentile is not None:
            targets_series = clip_by_percentile(targets_series, self.clip_percentile)

        if return_raw_labels:
            raw_series = pd.Series(raw_labels, index=df.index, dtype=float)
            return targets_series, raw_series

        return targets_series

    def _compute_targets(self, prices: np.ndarray, cumr: np.ndarray, raw_labels: np.ndarray,
                         segments: list[dict]) -> np.ndarray:
        """Compute targets using Numba-optimized kernel."""
        target_type_id = REGRESSION_TARGET_TYPE_IDS[self.target_type]
        segments_array = self._build_segments_array(segments)
        return _compute_amp_targets_nb(prices, cumr, segments_array, target_type_id, self.use_log_returns)

    def _compute_targets_py(self, prices: np.ndarray, cumr: np.ndarray, raw_labels: np.ndarray,
                            segments: list[dict]) -> np.ndarray:
        """Pure Python version for accuracy verification."""
        targets = np.zeros(len(prices), dtype=float)

        for seg in segments:
            start = seg["start_idx"]
            end = seg["end_idx"]
            direction = seg["direction"]
            start_price = seg["start_price"]
            extreme_price = seg["extreme_price"]
            seg_cumr = cumr[start:end + 1]

            # Segment-level scalars (used by forward-looking types)
            duration = end - start + 1
            total_amplitude = cumr[end] - cumr[start]
            amplitude_per_bar = total_amplitude / duration if duration > 0 else 0.0

            if self.use_log_returns and start_price > 0 and prices[end] > 0:
                forward_return_full = np.log(prices[end] / start_price)
            elif start_price > 0:
                forward_return_full = (prices[end] - start_price) / start_price
            else:
                forward_return_full = 0.0

            for i in range(start, end + 1):
                targets[i] = self._target_for_bar(
                    prices, cumr, i, start, end, direction,
                    start_price, extreme_price,
                    amplitude_per_bar, forward_return_full, duration,
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

    def _target_for_bar(self, prices: np.ndarray, cumr: np.ndarray, i: int, start: int, end: int, direction: int,
                        start_price: float, extreme_price: float, amplitude_per_bar: float, forward_return_full: float,
                        duration: int) -> float:
        p_i = prices[i]
        p_s = start_price
        if p_s == 0 or np.isnan(p_i) or np.isnan(p_s):
            return 0.0

        tt = self.target_type
        if tt == RegressionTargetType.SLOPE:
            return ols_slope(prices, start, i, direction)
        elif tt == RegressionTargetType.CUMULATIVE_RETURN:
            if self.use_log_returns:
                return direction * np.log(p_i / p_s)
            return direction * (p_i - p_s) / p_s
        elif tt == RegressionTargetType.MOMENTUM:
            elapsed = i - start + 1
            cum_ret = np.log(p_i / p_s) if self.use_log_returns else (p_i - p_s) / p_s
            return direction * cum_ret / elapsed
        elif tt == RegressionTargetType.AMPLITUDE_PER_BAR:
            # Constant across the whole segment (forward-looking)
            return direction * amplitude_per_bar
        elif tt == RegressionTargetType.FORWARD_RETURN:
            # Return from current bar to segment end (forward-looking)
            p_end = prices[end]
            if p_end == 0 or np.isnan(p_end):
                return 0.0
            if self.use_log_returns:
                return direction * np.log(p_end / p_i)
            return direction * (p_end - p_i) / p_i
        elif tt == RegressionTargetType.FORWARD_RETURN_PER_BAR:
            # Forward return normalized by remaining bars (forward-looking)
            p_end = prices[end]
            remaining = end - i + 1
            if p_end == 0 or np.isnan(p_end) or remaining == 0:
                return 0.0
            if self.use_log_returns:
                fwd = np.log(p_end / p_i)
            else:
                fwd = (p_end - p_i) / p_i
            return direction * fwd / remaining
        elif tt == RegressionTargetType.RETURN_TO_EXTREME:
            if extreme_price == 0 or np.isnan(extreme_price):
                return 0.0
            if direction > 0:
                return np.log(extreme_price / p_i) if self.use_log_returns else (extreme_price - p_i) / p_i
            else:
                return -np.log(p_i / extreme_price) if self.use_log_returns else -(p_i - extreme_price) / extreme_price
        return 0.0
