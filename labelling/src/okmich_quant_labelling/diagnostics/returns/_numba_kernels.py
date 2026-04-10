"""
Numba-optimized kernels for regression target computation.

All functions are JIT-compiled with @njit for performance.
Must stay in sync with integer IDs in _target_type.py.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Import integer constants for dispatch
from ._target_type import (
    TT_PERCENTAGE_FROM_EXTREME,
    TT_CUMULATIVE_RETURN,
    TT_MOMENTUM,
    TT_SLOPE,
    TT_AMPLITUDE_PER_BAR,
    TT_FORWARD_RETURN,
    TT_FORWARD_RETURN_PER_BAR,
    TT_RETURN_TO_EXTREME,
)


@njit(cache=True)
def _ols_slope_nb(prices: np.ndarray, start: int, current: int, direction: int) -> float:
    """OLS slope of log-price over time indices [0, …, current-start]."""
    n = current - start + 1
    if n < 2:
        return 0.0

    X = np.arange(n, dtype=np.float64)
    Y = np.log(prices[start : current + 1])

    X_mean = X.mean()
    Y_mean = Y.mean()
    dX = X - X_mean

    denom = np.dot(dX, dX)
    if denom == 0.0:
        return 0.0

    beta = np.dot(dX, Y - Y_mean) / denom
    return direction * beta


@njit(cache=True)
def _compute_auto_targets_nb(
    prices: np.ndarray,
    segments: np.ndarray,
    target_type_id: int,
    use_log_returns: bool,
) -> np.ndarray:
    """
    Compute targets for AutoLabelRegression using Numba.

    Parameters
    ----------
    prices : np.ndarray
        Price array of shape (n_bars,).
    segments : np.ndarray
        Segment array of shape (n_segments, 5) with columns:
        [start_idx, end_idx, direction, start_price, extreme_price].
    target_type_id : int
        Integer ID of the target type (see _target_type.py).
    use_log_returns : bool
        Whether to use log returns.

    Returns
    -------
    np.ndarray
        Target values of shape (n_bars,).
    """
    n = len(prices)
    targets = np.zeros(n, dtype=np.float64)

    for seg_idx in range(segments.shape[0]):
        start = int(segments[seg_idx, 0])
        end = int(segments[seg_idx, 1])
        direction = int(segments[seg_idx, 2])
        start_price = segments[seg_idx, 3]
        extreme_price = segments[seg_idx, 4]

        for i in range(start, end + 1):
            p_i = prices[i]
            p_s = start_price

            if p_s == 0.0 or np.isnan(p_i) or np.isnan(p_s):
                targets[i] = 0.0
                continue

            # Dispatch by target_type_id
            if target_type_id == TT_PERCENTAGE_FROM_EXTREME:
                targets[i] = direction * (p_i - p_s) / p_s

            elif target_type_id == TT_CUMULATIVE_RETURN:
                if use_log_returns:
                    targets[i] = direction * np.log(p_i / p_s)
                else:
                    targets[i] = direction * (p_i - p_s) / p_s

            elif target_type_id == TT_MOMENTUM:
                elapsed = i - start + 1
                if use_log_returns:
                    cum_ret = np.log(p_i / p_s)
                else:
                    cum_ret = (p_i - p_s) / p_s
                targets[i] = direction * cum_ret / elapsed

            elif target_type_id == TT_SLOPE:
                targets[i] = _ols_slope_nb(prices, start, i, direction)

            elif target_type_id == TT_RETURN_TO_EXTREME:
                if extreme_price == 0.0 or np.isnan(extreme_price):
                    targets[i] = 0.0
                else:
                    if direction > 0:
                        if use_log_returns:
                            targets[i] = np.log(extreme_price / p_i)
                        else:
                            targets[i] = (extreme_price - p_i) / p_i
                    else:
                        if use_log_returns:
                            targets[i] = -np.log(p_i / extreme_price)
                        else:
                            targets[i] = -(p_i - extreme_price) / extreme_price

            else:
                targets[i] = 0.0

    return targets


@njit(cache=True)
def _compute_oracle_targets_nb(
    prices: np.ndarray,
    segments: np.ndarray,
    target_type_id: int,
    use_log_returns: bool,
) -> np.ndarray:
    """
    Compute targets for OracleLabelBasedRegressionLabeler using Numba.

    Handles all 8 RegressionTargetType values from prices alone (no cumr
    needed). AMPLITUDE_PER_BAR is expressed as the full-segment log/simple
    return divided by segment duration.

    Parameters
    ----------
    prices : np.ndarray
        Price array of shape (n_bars,).
    segments : np.ndarray
        Segment array of shape (n_segments, 5) with columns:
        [start_idx, end_idx, direction, start_price, extreme_price].
    target_type_id : int
        Integer ID of the target type (see _target_type.py).
    use_log_returns : bool
        Whether to use log returns.

    Returns
    -------
    np.ndarray
        Target values of shape (n_bars,).
    """
    n = len(prices)
    targets = np.zeros(n, dtype=np.float64)

    for seg_idx in range(segments.shape[0]):
        start = int(segments[seg_idx, 0])
        end = int(segments[seg_idx, 1])
        direction = int(segments[seg_idx, 2])
        start_price = segments[seg_idx, 3]
        extreme_price = segments[seg_idx, 4]

        # Segment-level scalars for forward-looking types
        duration = end - start + 1
        p_end = prices[end]

        if start_price > 0.0 and p_end > 0.0:
            if use_log_returns:
                segment_return = np.log(p_end / start_price)
            else:
                segment_return = (p_end - start_price) / start_price
        else:
            segment_return = 0.0

        amplitude_per_bar = segment_return / duration if duration > 0 else 0.0

        for i in range(start, end + 1):
            p_i = prices[i]
            p_s = start_price

            if p_s == 0.0 or np.isnan(p_i) or np.isnan(p_s):
                targets[i] = 0.0
                continue

            if target_type_id == TT_PERCENTAGE_FROM_EXTREME:
                targets[i] = direction * (p_i - p_s) / p_s

            elif target_type_id == TT_CUMULATIVE_RETURN:
                if use_log_returns:
                    targets[i] = direction * np.log(p_i / p_s)
                else:
                    targets[i] = direction * (p_i - p_s) / p_s

            elif target_type_id == TT_MOMENTUM:
                elapsed = i - start + 1
                if use_log_returns:
                    cum_ret = np.log(p_i / p_s)
                else:
                    cum_ret = (p_i - p_s) / p_s
                targets[i] = direction * cum_ret / elapsed

            elif target_type_id == TT_SLOPE:
                targets[i] = _ols_slope_nb(prices, start, i, direction)

            elif target_type_id == TT_AMPLITUDE_PER_BAR:
                targets[i] = direction * amplitude_per_bar

            elif target_type_id == TT_FORWARD_RETURN:
                if p_end == 0.0 or np.isnan(p_end):
                    targets[i] = 0.0
                elif use_log_returns:
                    targets[i] = direction * np.log(p_end / p_i)
                else:
                    targets[i] = direction * (p_end - p_i) / p_i

            elif target_type_id == TT_FORWARD_RETURN_PER_BAR:
                remaining = end - i + 1
                if p_end == 0.0 or np.isnan(p_end) or remaining == 0:
                    targets[i] = 0.0
                else:
                    if use_log_returns:
                        fwd = np.log(p_end / p_i)
                    else:
                        fwd = (p_end - p_i) / p_i
                    targets[i] = direction * fwd / remaining

            elif target_type_id == TT_RETURN_TO_EXTREME:
                if extreme_price == 0.0 or np.isnan(extreme_price):
                    targets[i] = 0.0
                else:
                    if direction > 0:
                        if use_log_returns:
                            targets[i] = np.log(extreme_price / p_i)
                        else:
                            targets[i] = (extreme_price - p_i) / p_i
                    else:
                        if use_log_returns:
                            targets[i] = -np.log(p_i / extreme_price)
                        else:
                            targets[i] = -(p_i - extreme_price) / extreme_price

            else:
                targets[i] = 0.0

    return targets


@njit(cache=True)
def _compute_amp_targets_nb(
    prices: np.ndarray,
    cumr: np.ndarray,
    segments: np.ndarray,
    target_type_id: int,
    use_log_returns: bool,
) -> np.ndarray:
    """
    Compute targets for AmplitudeBasedRegressionLabeler using Numba.

    Parameters
    ----------
    prices : np.ndarray
        Price array of shape (n_bars,).
    cumr : np.ndarray
        Cumulative return array (in bps) of shape (n_bars,).
    segments : np.ndarray
        Segment array of shape (n_segments, 5) with columns:
        [start_idx, end_idx, direction, start_price, extreme_price].
    target_type_id : int
        Integer ID of the target type (see _target_type.py).
    use_log_returns : bool
        Whether to use log returns.

    Returns
    -------
    np.ndarray
        Target values of shape (n_bars,).
    """
    n = len(prices)
    targets = np.zeros(n, dtype=np.float64)

    for seg_idx in range(segments.shape[0]):
        start = int(segments[seg_idx, 0])
        end = int(segments[seg_idx, 1])
        direction = int(segments[seg_idx, 2])
        start_price = segments[seg_idx, 3]
        extreme_price = segments[seg_idx, 4]

        # Segment-level scalars (used by forward-looking types)
        duration = end - start + 1
        total_amplitude = cumr[end] - cumr[start]
        amplitude_per_bar = total_amplitude / duration if duration > 0 else 0.0

        if use_log_returns and start_price > 0.0 and prices[end] > 0.0:
            forward_return_full = np.log(prices[end] / start_price)
        elif start_price > 0.0:
            forward_return_full = (prices[end] - start_price) / start_price
        else:
            forward_return_full = 0.0

        for i in range(start, end + 1):
            p_i = prices[i]
            p_s = start_price

            if p_s == 0.0 or np.isnan(p_i) or np.isnan(p_s):
                targets[i] = 0.0
                continue

            # Dispatch by target_type_id
            if target_type_id == TT_SLOPE:
                targets[i] = _ols_slope_nb(prices, start, i, direction)

            elif target_type_id == TT_CUMULATIVE_RETURN:
                if use_log_returns:
                    targets[i] = direction * np.log(p_i / p_s)
                else:
                    targets[i] = direction * (p_i - p_s) / p_s

            elif target_type_id == TT_MOMENTUM:
                elapsed = i - start + 1
                if use_log_returns:
                    cum_ret = np.log(p_i / p_s)
                else:
                    cum_ret = (p_i - p_s) / p_s
                targets[i] = direction * cum_ret / elapsed

            elif target_type_id == TT_AMPLITUDE_PER_BAR:
                targets[i] = direction * amplitude_per_bar

            elif target_type_id == TT_FORWARD_RETURN:
                p_end = prices[end]
                if p_end == 0.0 or np.isnan(p_end):
                    targets[i] = 0.0
                else:
                    if use_log_returns:
                        targets[i] = direction * np.log(p_end / p_i)
                    else:
                        targets[i] = direction * (p_end - p_i) / p_i

            elif target_type_id == TT_FORWARD_RETURN_PER_BAR:
                p_end = prices[end]
                remaining = end - i + 1
                if p_end == 0.0 or np.isnan(p_end) or remaining == 0:
                    targets[i] = 0.0
                else:
                    if use_log_returns:
                        fwd = np.log(p_end / p_i)
                    else:
                        fwd = (p_end - p_i) / p_i
                    targets[i] = direction * fwd / remaining

            elif target_type_id == TT_RETURN_TO_EXTREME:
                if extreme_price == 0.0 or np.isnan(extreme_price):
                    targets[i] = 0.0
                else:
                    if direction > 0:
                        if use_log_returns:
                            targets[i] = np.log(extreme_price / p_i)
                        else:
                            targets[i] = (extreme_price - p_i) / p_i
                    else:
                        if use_log_returns:
                            targets[i] = -np.log(p_i / extreme_price)
                        else:
                            targets[i] = -(p_i - extreme_price) / extreme_price

            else:
                targets[i] = 0.0

    return targets
