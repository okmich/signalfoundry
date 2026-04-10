"""
Numba-accelerated kernels for ROC analysis.

These kernels replace the two innermost Python loops in ``ROCAnalyzer``:

1. ``_opt_thresh_core`` — the O(n) incremental threshold scan
   (replicates Masters' ``opt_thresh``).
2. ``mcpt_kernel`` — the outer 1 000-permutation MCPT loop that calls
   ``_opt_thresh_core`` on each shuffle (replicates Masters' ``opt_MCPT``).

Keeping 1 000 × n Python-level loop iterations in compiled Numba code produces a 20–50x speedup vs the pure-Python
fallback — critical for batch screening of many indicators.

Usage
-----
These are private helpers consumed by ``ROCAnalyzer._mcpt_numba()``.
Do NOT call them directly from user code.

Source / Attribution
--------------------
The algorithmic logic — incremental win/loss accumulation, tie-skip, Fisher-Yates
shuffle, conservative ``>=`` comparison for MCPT counters — faithfully replicates:

    Timothy Masters, "Statistically Sound Indicators For Financial Market
    Prediction", Apress, 2013.  C++ source: ROC.CPP.
"""

import numpy as np
from numba import njit


# --------------------------------------------------------------------------- #
# Stationarity / structural-break kernels  (BREAK_MEAN.CPP port)             #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _mannwhitney_z(hist: np.ndarray, rec: np.ndarray) -> float:
    """
    Standardised z-score of the Mann-Whitney U statistic.

    Uses the rank-based formula with average-rank tie correction —
    replicates what ``scipy.stats.mannwhitneyu`` computes internally,
    but runs inside Numba without the Python interpreter overhead.

    Parameters
    ----------
    hist : 1-D float64 array  — historical window
    rec  : 1-D float64 array  — recent window
    """
    n1 = len(hist)
    n2 = len(rec)
    n = n1 + n2

    # Build combined array: historical first, recent second
    combined = np.empty(n, dtype=np.float64)
    for i in range(n1):
        combined[i] = hist[i]
    for i in range(n2):
        combined[n1 + i] = rec[i]

    sort_idx = np.argsort(combined)

    # Assign average ranks (ties get equal ranks)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and combined[sort_idx[j]] == combined[sort_idx[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0      # average of ranks (i+1) .. j
        for k in range(i, j):
            ranks[sort_idx[k]] = avg_rank
        i = j

    # Rank sum for historical group (indices 0 .. n1-1 in combined)
    r1 = 0.0
    for i in range(n1):
        r1 += ranks[i]

    u1 = r1 - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    var_u = n1 * n2 * (n1 + n2 + 1) / 12.0
    return (u1 - mean_u) / (var_u ** 0.5)


@njit(cache=True)
def _max_u_stat_kernel(
    data: np.ndarray,
    min_recent: int,
    max_recent: int,
    comparisons: int,
    record_location: bool,
) -> tuple:
    """
    Sliding-window Mann-Whitney search (Masters' ``compute_break_mean``).

    Returns (max_abs_z, break_idx, recent_size).
    """
    n = len(data)
    ntot = n - comparisons + 1

    max_stat = 0.0            # 0.0 sentinel: any |z| > 0 will update
    best_break_idx = 0
    best_recent_size = min_recent

    for recent_size in range(min_recent, max_recent + 1):
        n1 = ntot - recent_size
        if n1 < 2:
            continue

        for icomp in range(comparisons - 1, -1, -1):
            istart1 = icomp
            istart2 = istart1 + n1

            if istart2 + recent_size > n:
                continue

            z = _mannwhitney_z(
                data[istart1 : istart1 + n1],
                data[istart2 : istart2 + recent_size],
            )

            if abs(z) > abs(max_stat):
                max_stat = z
                if record_location:
                    best_break_idx = istart2
                    best_recent_size = recent_size

            # For observed data: only test most-recent position
            if record_location:
                break

    return max_stat, best_break_idx, best_recent_size


@njit(cache=True)
def stationarity_mcpt_kernel(data: np.ndarray, min_recent: int, max_recent: int, comparisons: int,
                             nperms: int, seed: int) -> tuple:
    """
    Full MCPT for the break-in-mean test.

    Returns (obs_stat, break_idx, recent_size, p_value).
    """
    n = len(data)

    # Observed statistic
    obs_stat, break_idx, recent_size = _max_u_stat_kernel(data, min_recent, max_recent, comparisons, True)

    np.random.seed(seed)
    count_exceed = 1          # original counts as first permutation
    shuffled = data.copy()

    for _ in range(nperms - 1):
        # Fisher-Yates in-place shuffle
        for j in range(n - 1, 0, -1):
            k = np.random.randint(0, j + 1)
            tmp = shuffled[j]
            shuffled[j] = shuffled[k]
            shuffled[k] = tmp

        perm_stat, _, _ = _max_u_stat_kernel(
            shuffled, min_recent, max_recent, comparisons, False
        )
        if abs(perm_stat) >= abs(obs_stat):
            count_exceed += 1

    return obs_stat, break_idx, recent_size, count_exceed / nperms


# --------------------------------------------------------------------------- #
# Core threshold scan (inner loop of opt_thresh)                              #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _opt_thresh_core(sorted_signals: np.ndarray, sorted_returns: np.ndarray, min_kept: int, eps: float) -> tuple:
    """
    O(n) incremental optimal-threshold search on pre-sorted arrays.

    Parameters
    ----------
    sorted_signals : 1-D float64 array
        Signal values sorted in *descending* order.
    sorted_returns : 1-D float64 array
        Forward returns in the same order as *sorted_signals*.
    min_kept : int
        Minimum number of observations that must remain in a candidate set.
    eps : float
        Small constant added to loss denominators to avoid division by zero.

    Returns
    -------
    tuple of (pf_all, long_pf, long_idx, short_pf, short_idx)
        pf_all     : float  — grand profit factor (all data)
        long_pf    : float  — best long profit factor
        long_idx   : int    — sorted-array split index for long threshold
        short_pf   : float  — best short profit factor
        short_idx  : int    — sorted-array split index for short threshold
    """
    n = len(sorted_signals)

    # Initialise: all data in "above" set
    win_above = 0.0
    lose_above = 0.0
    for i in range(n):
        r = sorted_returns[i]
        if r > 0.0:
            win_above += r
        elif r < 0.0:
            lose_above -= r      # store absolute value

    win_below = 0.0
    lose_below = 0.0

    pf_all = win_above / (lose_above + eps)
    best_long_pf = pf_all
    best_long_idx = 0
    best_short_pf = -1.0
    best_short_idx = n - 1

    for i in range(n - 1):
        ret = sorted_returns[i]
        if ret > 0.0:
            win_above -= ret
            lose_below += ret
        elif ret < 0.0:
            lose_above += ret    # ret < 0 → decrements lose_above
            win_below -= ret     # -ret > 0 → increments win_below

        # Skip tied signal values (Masters' tie-skip pattern)
        if sorted_signals[i + 1] == sorted_signals[i]:
            continue

        n_above = n - i - 1
        if n_above >= min_kept:
            pf_above = win_above / (lose_above + eps)
            if pf_above > best_long_pf:
                best_long_pf = pf_above
                best_long_idx = i + 1

        n_below = i + 1
        if n_below >= min_kept:
            pf_below = win_below / (lose_below + eps)
            if pf_below > best_short_pf:
                best_short_pf = pf_below
                best_short_idx = i + 1

    return pf_all, best_long_pf, best_long_idx, best_short_pf, best_short_idx


# --------------------------------------------------------------------------- #
# Full MCPT loop                                                               #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def mcpt_kernel(sorted_signals: np.ndarray, sorted_returns: np.ndarray, min_kept: int, nreps: int, seed: int) -> tuple:
    """
    Monte Carlo Permutation Testing — full loop in compiled Numba code.

    Parameters
    ----------
    sorted_signals : 1-D float64 array
        Signal values sorted in *descending* order (fixed across permutations).
    sorted_returns : 1-D float64 array
        Forward returns in the same sort order as *sorted_signals*.
    min_kept : int
        Minimum set size for threshold candidates.
    nreps : int
        Total permutation count (including the original as rep #1).
    seed : int
        Random seed for Fisher-Yates shuffle.

    Returns
    -------
    tuple of 10 values:
        pf_all       : float — grand profit factor
        long_pf      : float — best long profit factor (observed)
        long_idx     : int   — sorted index for long threshold
        n_long       : int   — number of long trades
        short_pf     : float — best short profit factor (observed)
        short_idx    : int   — sorted index for short threshold
        n_short      : int   — number of short trades
        pval_long    : float — MCPT p-value for long strategy
        pval_short   : float — MCPT p-value for short strategy
        pval_best    : float — MCPT p-value for best-of-two strategy
    """
    n = len(sorted_signals)
    eps = 1e-30

    # --- Observed result -------------------------------------------------- #
    pf_all, orig_long_pf, orig_long_idx, orig_short_pf, orig_short_idx = (
        _opt_thresh_core(sorted_signals, sorted_returns, min_kept, eps)
    )
    orig_best_pf = max(orig_long_pf, orig_short_pf)

    # Conservative counters start at 1 (original counts as first permutation)
    long_count = 1
    short_count = 1
    best_count = 1

    # --- Permutation loop ------------------------------------------------- #
    # Shuffling the already-sorted returns is equivalent to shuffling the
    # original returns and re-applying the sort key (uniform permutation).
    np.random.seed(seed)
    shuffled = sorted_returns.copy()

    for _ in range(nreps - 1):
        # Fisher-Yates in-place shuffle
        for j in range(n - 1, 0, -1):
            k = np.random.randint(0, j + 1)    # inclusive [0, j]
            tmp = shuffled[j]
            shuffled[j] = shuffled[k]
            shuffled[k] = tmp

        _, perm_long_pf, _, perm_short_pf, _ = _opt_thresh_core(
            sorted_signals, shuffled, min_kept, eps
        )

        if perm_long_pf >= orig_long_pf:
            long_count += 1
        if perm_short_pf >= orig_short_pf:
            short_count += 1
        if max(perm_long_pf, perm_short_pf) >= orig_best_pf:
            best_count += 1

    return (
        pf_all,
        orig_long_pf,
        orig_long_idx,
        n - orig_long_idx,          # n_long
        orig_short_pf,
        orig_short_idx,
        orig_short_idx,             # n_short
        long_count / nreps,
        short_count / nreps,
        best_count / nreps,
    )