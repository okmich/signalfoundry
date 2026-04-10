from typing import Any, Dict, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from numba import njit, prange


# ---------------------------------------------------------------------------
# Low-level Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _zigzag_reversals(arr: np.ndarray, threshold: float):
    """
    Count thresholded direction flips and record indices where flips occur.

    Returns (count, reversal_indices).
    """
    n = arr.size
    if n < 2:
        return 0, np.empty(0, dtype=np.int64)

    prices = np.empty(n, dtype=np.float64)
    prices[0] = 1.0
    for i in range(1, n):
        prices[i] = prices[i - 1] * np.exp(arr[i])

    last_pivot = prices[0]
    last_direction = 0
    reversals = 0
    reversal_idx = np.empty(n, dtype=np.int64)
    pos = 0

    for i in range(1, n):
        p = prices[i]
        if last_pivot == 0:
            continue
        rel = (p - last_pivot) / last_pivot
        if abs(rel) >= threshold:
            new_direction = 1 if rel > 0 else -1
            if last_direction != 0 and new_direction != last_direction:
                reversals += 1
                reversal_idx[pos] = i
                pos += 1
            last_direction = new_direction
            last_pivot = p

    return reversals, reversal_idx[:pos]


@njit(cache=True)
def _count_reversals_kernel(arr: np.ndarray, threshold: float) -> int:
    """Count thresholded direction flips from a log-return array.

    Numba JIT equivalent of ``_count_reversals_from_returns``.
    Uses an incremental price reconstruction to avoid allocating a full
    cumsum array on every call.
    """
    n = arr.shape[0]
    if n < 2:
        return 0
    price = np.exp(arr[0])
    last_pivot = price
    last_direction = 0
    reversals = 0
    for i in range(1, n):
        price *= np.exp(arr[i])
        if last_pivot == 0.0:
            continue
        rel = (price - last_pivot) / last_pivot
        if rel >= threshold or rel <= -threshold:
            new_dir = 1 if rel > 0.0 else -1
            if last_direction != 0 and new_dir != last_direction:
                reversals += 1
            last_direction = new_dir
            last_pivot = price
    return reversals


@njit(cache=True, parallel=True)
def _threshold_scan_global(arr: np.ndarray, thr_grid: np.ndarray) -> np.ndarray:
    """Evaluate all thresholds on a single global log-return array.

    Parameters
    ----------
    arr : float64 array of log returns (no NaNs)
    thr_grid : thresholds to evaluate

    Returns
    -------
    counts : (n_thr,) -- reversal count for each threshold
    """
    n_thr = thr_grid.shape[0]
    counts = np.zeros(n_thr, dtype=np.float64)
    for i in prange(n_thr):          # embarrassingly parallel -- no shared writes
        counts[i] = float(_count_reversals_kernel(arr, thr_grid[i]))
    return counts


@njit(cache=True, parallel=True)
def _threshold_scan_windowed(arr: np.ndarray, thr_grid: np.ndarray, window: int, valid_end: np.ndarray):
    """Evaluate all thresholds across rolling windows of a log-return array.

    Parameters
    ----------
    arr : float64 array of log returns (may contain NaN; guarded by valid_end)
    thr_grid : thresholds to evaluate
    window : rolling window length in bars
    valid_end : bool array, True at index ``end`` iff window [end-window+1, end]
                contains no NaN -- computed once in Python before calling this kernel

    Returns
    -------
    max_revs : (n_thr,) -- maximum reversal count across valid windows, per threshold
    mean_dens : (n_thr,) -- mean density (reversals/window) across valid windows, NaN
                if no valid windows exist
    valid_count : int -- number of valid windows processed (same for all thresholds)
    """
    n = arr.shape[0]
    n_thr = thr_grid.shape[0]

    max_revs = np.zeros(n_thr, dtype=np.float64)
    mean_dens = np.empty(n_thr, dtype=np.float64)
    for i in range(n_thr):
        mean_dens[i] = np.nan

    # Count valid windows once (scalar, read-only inside prange -- no race condition)
    valid_count = 0
    for end in range(window - 1, n):
        if valid_end[end]:
            valid_count += 1

    for i in prange(n_thr):          # parallel over thresholds; each thread owns index i
        thr = thr_grid[i]
        mr = 0.0
        sd = 0.0
        for end in range(window - 1, n):
            if not valid_end[end]:
                continue
            start = end - window + 1
            cnt = float(_count_reversals_kernel(arr[start : end + 1], thr))
            if cnt > mr:
                mr = cnt
            sd += cnt
        max_revs[i] = mr
        if valid_count > 0:
            mean_dens[i] = sd / (window * valid_count)

    return max_revs, mean_dens, valid_count


# ---------------------------------------------------------------------------
# Python reference (kept for backward compatibility and test verification)
# ---------------------------------------------------------------------------

def _count_reversals_from_returns(arr: np.ndarray, threshold: float) -> int:
    """Count thresholded direction flips from a returns array (pure Python)."""
    if arr is None or arr.size < 2:
        return 0
    prices = np.exp(np.cumsum(arr))
    last_pivot = prices[0]
    last_direction = 0
    reversals = 0
    for p in prices[1:]:
        if last_pivot == 0:
            continue
        rel = (p - last_pivot) / last_pivot
        if abs(rel) >= threshold:
            new_dir = 1 if rel > 0 else -1
            if last_direction != 0 and new_dir != last_direction:
                reversals += 1
            last_direction = new_dir
            last_pivot = p
    return reversals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@overload
def zigzag_density(prices: Union[pd.Series, np.ndarray], threshold: float = 0.02, window: None = None,
                   align: Literal["causal", "current"] = "causal") -> Tuple[float, np.ndarray]:
    ...


@overload
def zigzag_density(prices: Union[pd.Series, np.ndarray], threshold: float = 0.02, window: int = ...,
                   align: Literal["causal", "current"] = "causal") -> Tuple[pd.Series, Tuple[np.ndarray, np.ndarray]]:
    ...


def zigzag_density(prices: Union[pd.Series, np.ndarray], threshold: float = 0.02, window: Optional[int] = None,
                   align: Literal["causal", "current"] = "causal"
                   ) -> Union[Tuple[float, np.ndarray], Tuple[pd.Series, Tuple[np.ndarray, np.ndarray]]]:
    """
    Calculate thresholded pivot-flip density and reversal indices.

    Note
    ----
    This is a thresholded direction-flip counter on a reconstructed path from
    log returns. It is not a full canonical ZigZag extrema algorithm.

    Parameters
    ----------
    prices : array-like or pd.Series
        Price series.
    threshold : float
        Relative price move required to count a swing (e.g. 0.02 for 2%).
        Must be > 0.
    window : int or None
        Rolling window size. If None, return scalar density and global reversal indices.
    align : {'causal', 'current'}
        - 'causal': density at t uses data up to t-1 (safe for backtesting).
        - 'current': density at t uses data up to t.

    Returns
    -------
    If window is None:
        (float, np.ndarray)
        density and global reversal indices.
    Else:
        (pd.Series, (np.ndarray, np.ndarray))
        density series and `(reversal_points, reversal_counts)` where:
        - reversal_points has shape (n, window), padded with -1
        - reversal_counts has shape (n,) with valid count per row
    """
    if isinstance(prices, pd.Series):
        data = prices.copy()
    else:
        data = pd.Series(np.asarray(prices))

    if align not in ("causal", "current"):
        raise ValueError(f"align must be 'causal' or 'current', got {align!r}")
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    rs = pd.Series(np.log(data / data.shift(1)))
    n = len(rs)

    if window is None:
        if n < 2:
            return np.nan, np.empty(0, dtype=np.int64)
        rs_valid = rs.dropna()
        if len(rs_valid) < 2:
            return np.nan, np.empty(0, dtype=np.int64)
        rev_count, rev_idx = _zigzag_reversals(rs_valid.values, threshold)
        density = rev_count / len(rs_valid)
        return density, rev_idx

    if not (isinstance(window, int) and window >= 2):
        raise ValueError("window must be an integer >= 2 when provided")

    result = pd.Series(np.nan, index=rs.index, dtype=float)
    reversal_points = np.full((n, window), -1, dtype=np.int64)
    reversal_counts = np.zeros(n, dtype=np.int64)

    if n < 2 or window > n:
        return result, (reversal_points, reversal_counts)

    for end in range(window - 1, n):
        start = end - window + 1
        window_slice = rs.iloc[start : end + 1]
        if window_slice.isna().any():
            continue

        rev_count, rev_idx = _zigzag_reversals(window_slice.values, threshold)
        result.iloc[end] = rev_count / window

        k = rev_idx.size
        if k > 0:
            reversal_points[end, :k] = rev_idx + start
        reversal_counts[end] = k

    if align == "causal":
        result = result.shift(1)
        rp_shifted = np.full_like(reversal_points, -1)
        rc_shifted = np.zeros_like(reversal_counts)
        rp_shifted[1:] = reversal_points[:-1]
        rc_shifted[1:] = reversal_counts[:-1]
        reversal_points, reversal_counts = rp_shifted, rc_shifted

    return result, (reversal_points, reversal_counts)


def find_threshold_for_reversals(
    log_returns,
    *,
    window: Optional[int] = None,
    thr_grid: Optional[np.ndarray] = None,
    thr_min: float = 1e-4,
    thr_max: float = 0.2,
    num: int = 100,
    min_reversals: int = 1,
    target_density: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Find the threshold that produces a desired reversal density or count.

    Uses Numba JIT kernels parallelised over thresholds for speed.

    Parameters
    ----------
    log_returns : array-like or pd.Series
        Log-return series to analyse.
    window : int or None
        Rolling window size.  None evaluates the whole series as one block.
    thr_grid : array-like or None
        Explicit threshold grid.  If None, ``np.linspace(thr_min, thr_max, num)``
        is used.
    thr_min, thr_max : float
        Bounds for the auto-generated grid (ignored when ``thr_grid`` is given).
    num : int
        Number of points in the auto-generated grid.
    min_reversals : int
        Minimum reversal count to accept a threshold (used when
        ``target_density`` is None).  The *smallest* qualifying threshold is
        returned so as to be as sensitive as possible while meeting the bar.
    target_density : float or None
        If given, return the threshold whose mean density is closest to this
        value, regardless of ``min_reversals``.
    verbose : bool
        Print diagnostic arrays to stdout.

    Returns
    -------
    chosen : float or None
        Selected threshold, or None if no qualifying threshold was found.
    diagnostics : dict
        ``thr_grid``, ``max_revs``, ``mean_dens``, and (windowed mode)
        ``valid_windows_count`` / ``window`` / ``min_reversals``.
    """
    if isinstance(log_returns, pd.Series):
        rs = log_returns.copy()
    else:
        rs = pd.Series(np.asarray(log_returns))
    n = len(rs)

    # --- build / validate threshold grid ---
    if thr_grid is None:
        if thr_min <= 0 or thr_max <= 0:
            raise ValueError("thr_min and thr_max must be > 0")
        if thr_min >= thr_max:
            raise ValueError("thr_min must be < thr_max")
        if num < 1:
            raise ValueError("num must be >= 1")
        thr_grid = np.linspace(thr_min, thr_max, num)

    thr_grid = np.asarray(thr_grid, dtype=np.float64)
    if thr_grid.size == 0:
        raise ValueError("thr_grid must not be empty")
    if np.any(thr_grid <= 0):
        raise ValueError("all thresholds must be > 0")
    if min_reversals < 0:
        raise ValueError("min_reversals must be >= 0")
    if target_density is not None and not (0 <= target_density <= 1):
        raise ValueError("target_density must be between 0 and 1")

    # default return arrays (overwritten below)
    max_revs = np.zeros(thr_grid.shape, dtype=np.float64)
    mean_dens = np.full(thr_grid.shape, np.nan, dtype=np.float64)

    if n < 2:
        return None, {"thr_grid": thr_grid, "max_revs": max_revs, "mean_dens": mean_dens}

    # ------------------------------------------------------------------
    # Global mode -- evaluate whole series once per threshold
    # ------------------------------------------------------------------
    if window is None:
        rs_valid = rs.dropna()
        n_valid = len(rs_valid)
        if n_valid < 2:
            return None, {"thr_grid": thr_grid, "max_revs": max_revs, "mean_dens": mean_dens}

        counts = _threshold_scan_global(rs_valid.values, thr_grid)
        max_revs = counts.copy()
        mean_dens = counts / n_valid

        chosen = _select_threshold(thr_grid, counts, mean_dens, min_reversals, target_density)

        diag: Dict[str, Any] = {
            "thr_grid": thr_grid,
            "counts": counts,
            "max_revs": max_revs,
            "mean_dens": mean_dens,
        }
        if verbose:
            print("thresholds evaluated:", thr_grid)
            print("max_revs:", max_revs)
            print("mean_dens:", mean_dens)
        return chosen, diag

    # ------------------------------------------------------------------
    # Windowed mode -- evaluate rolling windows across all thresholds
    # ------------------------------------------------------------------
    if not (isinstance(window, int) and window >= 2):
        raise ValueError("window must be an integer >= 2 when provided")

    # Build the valid-window boolean mask in O(n) using a cumsum trick --
    # avoids the original O(n x window) per-window NaN scan.
    rs_arr = rs.values
    nan_mask = np.isnan(rs_arr).astype(np.int64)
    nan_cs = np.concatenate(([0], np.cumsum(nan_mask)))
    valid_end = np.zeros(n, dtype=np.bool_)
    if window <= n:
        valid_end[window - 1 :] = (nan_cs[window:] - nan_cs[: n - window + 1]) == 0

    max_revs, mean_dens, valid_windows_count = _threshold_scan_windowed(
        rs_arr, thr_grid, window, valid_end
    )

    chosen = _select_threshold_windowed(thr_grid, max_revs, mean_dens, min_reversals, target_density)

    diagnostics: Dict[str, Any] = {
        "thr_grid": thr_grid,
        "max_revs": max_revs,
        "mean_dens": mean_dens,
        "valid_windows_count": valid_windows_count,
        "window": window,
        "min_reversals": min_reversals,
    }
    if verbose:
        print("thresholds evaluated:", thr_grid)
        print("max_revs:", max_revs)
        print("mean_dens:", mean_dens)

    return chosen, diagnostics


# ---------------------------------------------------------------------------
# Private selection helpers (pure Python -- called once, not in tight loop)
# ---------------------------------------------------------------------------

def _select_threshold(
    thr_grid: np.ndarray,
    counts: np.ndarray,
    mean_dens: np.ndarray,
    min_reversals: int,
    target_density: Optional[float]) -> Optional[float]:
    """Choose threshold for the global (no-window) case."""
    if target_density is not None:
        valid = ~np.isnan(mean_dens)
        if not valid.any():
            return None
        idx = np.argmin(np.abs(mean_dens[valid] - target_density))
        return float(thr_grid[valid][idx])
    ok = np.where(counts >= min_reversals)[0]
    return float(thr_grid[ok].min()) if ok.size > 0 else None


def _select_threshold_windowed(
    thr_grid: np.ndarray,
    max_revs: np.ndarray,
    mean_dens: np.ndarray,
    min_reversals: int,
    target_density: Optional[float],
) -> Optional[float]:
    """Choose threshold for the windowed case."""
    if target_density is not None:
        valid = ~np.isnan(mean_dens)
        if not valid.any():
            return None
        idx = np.argmin(np.abs(mean_dens[valid] - target_density))
        return float(thr_grid[valid][idx])
    ok = np.where(max_revs >= min_reversals)[0]
    if ok.size > 0:
        return float(thr_grid[ok].min())
    # fallback: best available
    if not np.all(np.isnan(max_revs)):
        best = int(np.nanargmax(max_revs))
        if max_revs[best] > 0:
            return float(thr_grid[best])
    return None
