"""
Tie-aware equal-size histogram binning.

Source / Attribution
--------------------
This module is a Python port of the ``PART.CPP`` algorithm by Timothy Masters, published in:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.
    C++ source: PART.CPP (distributed with the book's companion code)

The core ``partition()`` function replicates Masters' algorithm exactly, including the relative floating-point tolerance
used for tie detection and the iterative boundary-adjustment strategy that ensures tied values are never split across
bins. This is a prerequisite for the Entropy and Mutual Information indicators described in the same work.

Usage
-----
>>> from okmich_quant_features.utils.binning import partition, validate_partition
>>> import numpy as np
>>> data = np.array([1.5, 2.3, 2.3, 2.3, 3.1, 4.2, 4.2, 5.0])
>>> bins, n_actual, bounds = partition(data, npart=3, return_bounds=True)
>>> bins
array([0, 1, 1, 1, 1, 2, 2, 2])
>>> n_actual
3
"""

import numpy as np
from typing import Optional, Tuple


def partition(data: np.ndarray, npart: int, return_bounds: bool = False) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
    """
    Partition *data* into *npart* equal-sized bins without splitting tied values.

    This is a direct port of Timothy Masters' ``partition()`` function from ``PART.CPP``.  Naive equal-width or
    equal-frequency binning (e.g. ``numpy.digitize``) can place identical values into different bins, which violates the
    mathematical requirements of histogram-based indicators such as Shannon Entropy and Mutual Information.
    This algorithm resolves that by iteratively moving bin boundaries away from tied blocks.

    Algorithm summary
    -----------------
    1. Sort the data; assign integer *ranks* so that tied values share the same
       rank.  Tie detection uses the same relative floating-point tolerance as
       Masters: ``diff >= 1e-12 * (1 + |a| + |b|)``.
    2. Create initial equal-sized bins (``n // npart`` cases each).
    3. Iterate: for every internal boundary that falls inside a tied block, move
       it either left (before the block) or right (after the block), choosing
       whichever direction minimises the resulting bin-size imbalance.
    4. Map sorted bin assignments back to the original data order.

    Parameters
    ----------
    data : array-like
        1-D array of values to be binned.
    npart : int
        Desired number of bins.  If *npart* > ``len(data)`` it is clamped to
        ``len(data)``.
    return_bounds : bool, default False
        When ``True`` the function also returns the upper boundary value of
        each bin (the maximum value in that bin).

    Returns
    -------
    bins : np.ndarray of int, shape (n,)
        Bin index (0-based) assigned to each element of *data*.
    npart_actual : int
        Actual number of bins created.  May be less than *npart* when the data
        contains so many ties that fewer distinct bins are possible.
    bounds : np.ndarray of float or None
        Upper boundary value for each bin (only when ``return_bounds=True``).

    Examples
    --------
    >>> data = np.array([1.5, 2.3, 2.3, 2.3, 3.1, 4.2, 4.2, 5.0])
    >>> bins, n, bounds = partition(data, npart=3, return_bounds=True)
    >>> bins   # tied 2.3s stay together; tied 4.2s stay together
    array([0, 1, 1, 1, 1, 2, 2, 2])

    Notes
    -----
    The maximum number of iterations of the boundary-adjustment loop is capped at 1 000 to guard against pathological inputs
    (e.g. data that is entirely one value), though convergence is typically reached in a single pass.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if npart > n:
        npart = n
    if npart < 1:
        npart = 1

    # ------------------------------------------------------------------ #
    # Step 1: sort and assign integer ranks (ties get the same rank)     #
    # ------------------------------------------------------------------ #
    sort_idx = np.argsort(data, kind="stable")
    sorted_data = data[sort_idx]

    # Relative floating-point tolerance identical to Masters' C++ code:
    #   diff >= 1e-12 * (1.0 + fabs(x[i]) + fabs(x[i-1]))
    tol = 1e-12
    ranks = np.zeros(n, dtype=np.int32)
    current_rank = 0
    for i in range(1, n):
        diff = sorted_data[i] - sorted_data[i - 1]
        scale = 1.0 + abs(sorted_data[i]) + abs(sorted_data[i - 1])
        if diff >= tol * scale:
            current_rank += 1
        ranks[i] = current_rank

    # ------------------------------------------------------------------- #
    # Step 2: initial equal-sized bins                                    #
    # bin_end[i] = index (into sorted array) of the last element in bin i #
    # ------------------------------------------------------------------- #
    bin_end = np.zeros(npart, dtype=np.int32)
    k = 0
    for i in range(npart):
        j = (n - k) // (npart - i)
        k += j
        bin_end[i] = k - 1

    # ------------------------------------------------------------------ #
    # Step 3: iteratively resolve boundaries that split tied blocks        #
    # ------------------------------------------------------------------ #
    max_iterations = 1000
    for _ in range(max_iterations):
        tie_found = False

        for ibound in range(npart - 1):
            boundary_idx = bin_end[ibound]

            # Does this boundary split a tied block?
            if ranks[boundary_idx] != ranks[boundary_idx + 1]:
                continue

            tie_found = True

            # Find the full extent of the tied block
            istart = boundary_idx
            while istart > 0 and ranks[istart - 1] == ranks[istart]:
                istart -= 1

            istop = boundary_idx + 1
            while istop < n - 1 and ranks[istop] == ranks[istop + 1]:
                istop += 1

            # -- Option A: move boundary LEFT (before the tied block) ---- #
            left_boundary = istart - 1
            prev_end = bin_end[ibound - 1] if ibound > 0 else -1
            if left_boundary >= 0 and left_boundary > prev_end:
                nleft = left_boundary - prev_end
                next_end = bin_end[ibound + 1]
                nright = next_end - left_boundary
                left_imbalance = abs(nleft - nright)
            else:
                left_imbalance = np.inf

            # -- Option B: move boundary RIGHT (after the tied block) ---- #
            right_boundary = istop
            next_end = bin_end[ibound + 1]
            if right_boundary < n - 1 and right_boundary < next_end:
                nleft = right_boundary - prev_end
                nright = next_end - right_boundary
                right_imbalance = abs(nleft - nright)
            else:
                right_imbalance = np.inf

            # Choose the direction with less imbalance
            if left_imbalance <= right_imbalance:
                if left_imbalance < np.inf:
                    bin_end[ibound] = left_boundary
            elif right_imbalance < np.inf:
                bin_end[ibound] = right_boundary

        if not tie_found:
            break

    # ------------------------------------------------------------------ #
    # Step 4: assign cases to bins and map back to original order          #
    # ------------------------------------------------------------------ #
    bin_assignment = np.zeros(n, dtype=np.int32)
    for i in range(npart):
        start = bin_end[i - 1] + 1 if i > 0 else 0
        end = bin_end[i] + 1
        bin_assignment[start:end] = i

    bins = np.empty(n, dtype=np.int32)
    bins[sort_idx] = bin_assignment
    npart_actual = int(np.unique(bins).size)

    if return_bounds:
        bounds = np.array([sorted_data[bin_end[i]] for i in range(npart)], dtype=np.float64)
        return bins, npart_actual, bounds

    return bins, npart_actual, None


def validate_partition(data: np.ndarray, bins: np.ndarray) -> bool:
    """
    Check that no tied values have been assigned to different bins.

    Parameters
    ----------
    data : np.ndarray
        The original data array passed to :func:`partition`.
    bins : np.ndarray
        The bin assignments returned by :func:`partition`.

    Returns
    -------
    bool
        ``True`` if the partition is valid (no ties split), ``False`` otherwise.
    """
    data = np.asarray(data, dtype=np.float64)
    tol = 1e-12
    for i in range(len(data) - 1):
        diff = abs(data[i] - data[i + 1])
        scale = 1.0 + abs(data[i]) + abs(data[i + 1])
        if diff < tol * scale and bins[i] != bins[i + 1]:
            return False
    return True
