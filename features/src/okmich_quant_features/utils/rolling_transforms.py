"""
Rolling / windowed feature transformations.

All functions operate on a single time-series (numpy array or pandas Series)
and return the same type with the same index.  The first (window - 1) values
are NaN where a full window is required.

Functions
---------
rolling_zscore          -- (x - rolling_mean) / rolling_std
rolling_percentile_rank -- fraction of window values below current value
rolling_volatility_scale -- x / rolling_std  (vol-normalised level)
rolling_slope           -- OLS slope over a trailing window
rolling_persistence     -- fraction of window where x > threshold
tanh_compress           -- soft-clip to (-1, 1) via tanh scaling
sigmoid_compress        -- soft-clip to (0, 1) via sigmoid scaling

Notes
-----
- If any element in a window is NaN the output for that bar is NaN.
  Pre-clean inputs with forward-fill / dropna if partial-window NaN
  propagation is undesirable.
- All rolling kernels use population std (ddof=0) for consistency.
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Union

_Series = Union[np.ndarray, pd.Series]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: _Series) -> np.ndarray:
    """Convert to a C-contiguous float64 numpy array."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(x, dtype=np.float64)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def _wrap(result: np.ndarray, template: _Series, name_suffix: str = "") -> _Series:
    if isinstance(template, pd.Series):
        tname = template.name
        if tname is None:
            name = name_suffix or None
        else:
            name = str(tname) + name_suffix
        return pd.Series(result, index=template.index, name=name)
    return result


def _validate_window(window, n: int) -> int:
    """Validate and coerce window to int. Returns the coerced value."""
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > n:
        raise ValueError(f"window ({window}) exceeds series length ({n})")
    return window


# ---------------------------------------------------------------------------
# Numba kernels (all produce float64 output arrays with leading NaNs)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rolling_zscore_kernel(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        mu = np.mean(w)
        sigma = np.std(w)
        if np.isnan(sigma):
            out[i] = np.nan
        elif sigma <= 0.0:
            out[i] = 0.0
        else:
            out[i] = (x[i] - mu) / sigma
    return out


@njit(cache=True)
def _rolling_percentile_rank_kernel(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        # Propagate NaN: any NaN in the window produces NaN output.
        has_nan = False
        for v in w:
            if np.isnan(v):
                has_nan = True
                break
        if has_nan:
            out[i] = np.nan
            continue
        count = 0
        for v in w:
            if v < x[i]:
                count += 1
        out[i] = count / window
    return out


@njit(cache=True)
def _rolling_volatility_scale_kernel(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        sigma = np.std(w)
        if np.isnan(sigma):
            out[i] = np.nan
        elif sigma <= 0.0:
            out[i] = 0.0
        else:
            out[i] = x[i] / sigma
    return out


@njit(cache=True)
def _rolling_slope_kernel(x: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of x regressed on integer time indices [0, 1, ..., window-1].

    Returns the slope in units of x per bar. NaN in any window element
    propagates to NaN output via the mean/sum arithmetic.
    """
    n = len(x)
    out = np.full(n, np.nan)
    t = np.arange(window, dtype=np.float64)
    t_mean = (window - 1) / 2.0
    t_var = np.sum((t - t_mean) ** 2)  # constant across windows
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        w_mean = np.mean(w)
        cov = np.sum((t - t_mean) * (w - w_mean))
        # NaN in w propagates through w_mean and cov automatically.
        out[i] = cov / t_var if t_var > 0.0 else 0.0
    return out


@njit(cache=True)
def _rolling_persistence_kernel(x: np.ndarray, window: int, threshold: float, above: bool) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = x[i - window + 1: i + 1]
        # Propagate NaN: check entire window for any NaN.
        has_nan = False
        for v in w:
            if np.isnan(v):
                has_nan = True
                break
        if has_nan:
            out[i] = np.nan
            continue
        count = 0
        for v in w:
            if above:
                if v > threshold:
                    count += 1
            else:
                if v < threshold:
                    count += 1
        out[i] = count / window
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_zscore(x: _Series, window: int) -> _Series:
    """Rolling z-score: (x - mean) / std over trailing window.

    Parameters
    ----------
    x : array-like
    window : int
        Look-back period (>= 1).

    Returns
    -------
    Same type as input. Returns 0.0 where std == 0, NaN where any window
    element is NaN.
    """
    arr = _to_numpy(x)
    window = _validate_window(window, len(arr))
    result = _rolling_zscore_kernel(arr, window)
    return _wrap(result, x, "_zscore")


def rolling_percentile_rank(x: _Series, window: int) -> _Series:
    """Fraction of window values strictly below the current value.

    Returns values in [0, 1).  Equivalent to a rolling empirical CDF evaluated
    at the current observation.  When all window values are equal the rank is 0.

    Parameters
    ----------
    x : array-like
    window : int

    Returns
    -------
    Same type as input. NaN when x[i] is NaN.
    """
    arr = _to_numpy(x)
    window = _validate_window(window, len(arr))
    result = _rolling_percentile_rank_kernel(arr, window)
    return _wrap(result, x, "_pct_rank")


def rolling_volatility_scale(x: _Series, window: int) -> _Series:
    """Volatility-scaled series: x / rolling_std.

    Useful for normalising a momentum or signal series by recent realised
    volatility so that output magnitude is comparable across regimes.

    Parameters
    ----------
    x : array-like
    window : int

    Returns
    -------
    Same type as input. Returns 0.0 where std == 0, NaN where any window
    element is NaN.
    """
    arr = _to_numpy(x)
    window = _validate_window(window, len(arr))
    result = _rolling_volatility_scale_kernel(arr, window)
    return _wrap(result, x, "_volscale")


def rolling_slope(x: _Series, window: int) -> _Series:
    """OLS slope of x over a trailing window, in units of x per bar.

    Computed by regressing x on evenly-spaced time indices [0, 1, ..., window-1].
    A positive value means the series is trending up within the window; a negative
    value means it is trending down.

    Parameters
    ----------
    x : array-like
    window : int
        Must be >= 2 for the slope to be well-defined.

    Returns
    -------
    Same type as input. NaN where any window element is NaN.
    """
    arr = _to_numpy(x)
    window = _validate_window(window, len(arr))
    if window < 2:
        raise ValueError(f"window must be >= 2 for rolling_slope, got {window}")
    result = _rolling_slope_kernel(arr, window)
    return _wrap(result, x, "_slope")


def rolling_persistence(x: _Series, window: int, threshold: float = 0.0, above: bool = True) -> _Series:
    """Fraction of bars in the trailing window where x is above (or below) threshold.

    Parameters
    ----------
    x : array-like
    window : int
    threshold : float, default 0.0
        The level to compare against.
    above : bool, default True
        If True, count bars where x > threshold.
        If False, count bars where x < threshold.

    Returns
    -------
    Same type as input, values in [0, 1]. NaN where any window element is NaN.
    """
    arr = _to_numpy(x)
    window = _validate_window(window, len(arr))
    result = _rolling_persistence_kernel(arr, window, float(threshold), above)
    return _wrap(result, x, "_persistence")


def tanh_compress(x: _Series, scale: float = 1.0) -> _Series:
    """Soft-clip to (-1, 1) via tanh(x / scale).

    Parameters
    ----------
    x : array-like
    scale : float, default 1.0
        Controls the linear region width.  Values |x| << scale behave nearly
        linearly; values |x| >> scale saturate near ±1.  Must be > 0.

    Returns
    -------
    Same type as input, values in (-1, 1).
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    arr = _to_numpy(x)
    result = np.tanh(arr / scale)
    return _wrap(result, x, "_tanh")


def sigmoid_compress(x: _Series, scale: float = 1.0) -> _Series:
    """Soft-clip to (0, 1) via sigmoid(x / scale).

    Parameters
    ----------
    x : array-like
    scale : float, default 1.0
        Controls the linear region width.  Must be > 0.

    Returns
    -------
    Same type as input, values in (0, 1).
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    arr = _to_numpy(x)
    result = 1.0 / (1.0 + np.exp(-arr / scale))
    return _wrap(result, x, "_sigmoid")