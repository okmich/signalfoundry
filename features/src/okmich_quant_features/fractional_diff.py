from typing import Union, Optional

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller


@njit(cache=True)
def get_weights(d: float, window_size: int) -> np.ndarray:
    """Calculate binomial weights for fractional differentiation."""
    weights = np.ones(window_size)
    for k in range(1, window_size):
        weights[k] = -weights[k - 1] * (d - k + 1) / k
    return weights


def _infer_threshold(value: float, margin: int = 3) -> float:
    """Infer threshold from value's decimal precision, with margin orders of magnitude smaller."""
    if value == 0 or not np.isfinite(value):
        return 1e-8
    s = f"{abs(value):.15g}"
    if '.' in s:
        decimals = len(s.split('.')[1].rstrip('0'))
    else:
        decimals = 0
    return 10 ** -(decimals + margin)


@njit(cache=True)
def _apply_frac_diff(series_array: np.ndarray, weights: np.ndarray, window_size: int) -> np.ndarray:
    """Apply fractional differentiation weights to a series."""
    n = len(series_array)
    result = np.empty(n)
    result[:window_size - 1] = np.nan
    for i in range(window_size - 1, n):
        acc = 0.0
        for j in range(window_size):
            acc += weights[j] * series_array[i - j]
        result[i] = acc
    return result


def _auto_determine_window_size(d: float, threshold: float) -> int:
    """Determine optimal window size based on weight convergence."""
    window_size = 100
    max_window = 1000

    while window_size < max_window:
        weights = get_weights(d, window_size)
        if np.abs(weights[-1]) < threshold:
            break
        window_size += 50

    return min(window_size, max_window)


def fractional_differentiate_series(series: Union[pd.Series, np.ndarray], d: float = 0.5,
                                    window_size: Optional[int] = None) -> tuple:
    """Apply fractional differentiation. Threshold auto-inferred from data precision."""
    series_array = np.asarray(series)
    if window_size is None:
        threshold = _infer_threshold(series_array[-1])
        window_size = _auto_determine_window_size(d, threshold)

    weights = get_weights(d, window_size)
    result = _apply_frac_diff(series_array, weights, window_size)
    return result, window_size


def get_optimal_fractional_differentiation_order(series: Union[pd.Series, np.ndarray],
                                                 max_d: float = 1.0, min_d: float = 0.0, step: float = 0.05,
                                                 adf_threshold: float = 0.01,) -> tuple:
    """
    Find the optimal fractional differentiation order that makes the series stationary.

    Parameters:
    -----------
    series : Union[pd.Series, np.ndarray]
        Input time series
    max_d : float, optional
        Maximum d value to test (default: 1.0)
    min_d : float, optional
        Minimum d value to test (default: 0.0)
    step : float, optional
        Step size for d values (default: 0.05)
    adf_threshold : float, optional
        ADF test p-value threshold for stationarity (default: 0.01)

    Returns:
    --------
    tuple
        (optimal_d, optimal_window, differentiated_series, adf_results)
    """

    d_values = np.arange(min_d, max_d + step, step)
    adf_results = {}

    window_size = None
    for d in d_values:
        diff_series, window_size = fractional_differentiate_series(series, d=d)
        clean_series = diff_series[~np.isnan(diff_series)]
        if len(clean_series) > 10:
            adf_result = adfuller(clean_series)
            adf_results[d] = adf_result[1]  # p-value

    # Find d with p-value below threshold
    stationary_ds = [d for d, pval in adf_results.items() if pval < adf_threshold]

    if stationary_ds:
        optimal_d = min(stationary_ds)  # Choose minimal d that achieves stationarity
    else:
        optimal_d = d_values[-1]  # Fallback to maximum d

    # Recompute window for the chosen d — do NOT reuse the last loop's window_size
    # since different d values produce different convergence windows.
    optimal_series, optimal_window = fractional_differentiate_series(series, d=optimal_d)
    return optimal_d, optimal_window, optimal_series, adf_results


class FractionalDifferentiator:
    """Fractional differentiator with weight caching."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._weights_cache = {}

    def differentiate(self, series: Union[pd.Series, np.ndarray], d: float, window_size: Optional[int] = None) -> np.ndarray:
        """Apply fractional differentiation with cached weights."""
        if window_size is None:
            window_size = self.window_size

        cache_key = (d, window_size)
        if cache_key not in self._weights_cache:
            self._weights_cache[cache_key] = get_weights(d, window_size)

        return _apply_frac_diff(np.asarray(series), self._weights_cache[cache_key], window_size)

    def clear_cache(self):
        """Clear the weights cache."""
        self._weights_cache.clear()
