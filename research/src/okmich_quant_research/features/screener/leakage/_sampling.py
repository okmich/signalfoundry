"""
Row subsampling for the interaction-SHAP probe.

The probe is O(n_rows × n_features²) for the interaction tensor. A 5k-row stratified subsample is plenty for diagnosing
whether a pair carries interaction signal — but the subsample is biased toward label tails because that is where leakage
most often manifests. The resulting interaction magnitudes are therefore a *stress test*, not unbiased population estimates.
The report labels them as such so analysts don't over-interpret.

Caller contract: ``y`` must be NaN-free. The interaction probe enforces this upstream by dropping NaN-label rows before invoking the sampler.
"""
from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class SamplingTask(StrEnum):
    REGIME = "regime"
    RETURN = "return"


def stratified_row_sample(y: pd.Series, task: SamplingTask | str, n_rows: int,
                          tail_oversample: float = 2.0,
                          n_quantile_bins: int = 10,
                          random_state: int = 42) -> np.ndarray:
    """
    Return integer row positions for a label-stratified subsample.

    Parameters
    ----------
    y : pd.Series
        The label series the rows are stratified against. Must be NaN-free (caller's responsibility). Index is ignored;
        positions in ``y`` are returned.
    task : SamplingTask or str
        ``"regime"`` → stratify on raw class labels.
        ``"return"`` → quantile-bin into ``n_quantile_bins`` and stratify
        on the bins. Top and bottom bins receive ``tail_oversample`` weight.
    n_rows : int
        Target number of rows. If ``len(y) <= n_rows`` returns all positions.
        For the return task the returned count is exactly ``n_rows``; for
        the regime task it's within rounding of ``n_rows``.
    tail_oversample : float
        Multiplier on the top and bottom quantile bins (return task only —
        ignored for regime).  ``1.0`` = no oversampling. Default ``2.0``.
    n_quantile_bins : int
        Number of quantile bins for the return task. Default 10 (deciles).
    random_state : int
        For reproducibility.

    Returns
    -------
    np.ndarray
        Integer positions (0..len(y)-1) of the sampled rows. Sorted ascending
        so the caller can slice X by `.iloc[positions]` and preserve order.
    """
    n = len(y)
    if n_rows <= 0 or n <= n_rows:
        return np.arange(n)

    task = SamplingTask(task)

    if task == SamplingTask.REGIME:
        positions = _stratified_regime(y, n_rows, random_state)
    else:
        positions = _stratified_return(y, n_rows, tail_oversample, n_quantile_bins, random_state)

    positions.sort()
    return positions


def _stratified_regime(y: pd.Series, n_rows: int, random_state: int) -> np.ndarray:
    """Stratified subsample on discrete labels via StratifiedShuffleSplit."""
    y_arr = np.asarray(y)
    test_size = n_rows / len(y_arr)

    # StratifiedShuffleSplit needs at least 2 members per class. Fold any
    # singleton class in by sampling it whole and stratifying the rest.
    unique, counts = np.unique(y_arr, return_counts=True)
    singletons = unique[counts < 2]
    if len(singletons) > 0:
        keep_mask = np.isin(y_arr, singletons)
        forced_positions = np.flatnonzero(keep_mask)
        rest_positions = np.flatnonzero(~keep_mask)
        remaining = max(1, n_rows - len(forced_positions))
        if remaining < len(rest_positions):
            sub_test_size = remaining / len(rest_positions)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sub_test_size, random_state=random_state)
            _, sampled = next(sss.split(rest_positions.reshape(-1, 1), y_arr[rest_positions]))
            return np.concatenate([forced_positions, rest_positions[sampled]])
        return np.concatenate([forced_positions, rest_positions])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, sampled = next(sss.split(np.zeros(len(y_arr)), y_arr))
    return sampled


def _stratified_return(y: pd.Series, n_rows: int, tail_oversample: float,
                       n_quantile_bins: int, random_state: int) -> np.ndarray:
    """
    Quantile-stratified subsample with tail oversampling.

    Uses pd.qcut with ``duplicates='drop'`` so degenerate distributions (many
    tied values) collapse to fewer bins rather than raising. Returns exactly
    ``n_rows`` rows by trimming the over-rounded bin (largest bin) at the end.
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    all_positions = np.arange(n)

    try:
        bins = pd.qcut(y_arr, q=n_quantile_bins, labels=False, duplicates="drop")
    except ValueError:
        # Single-valued series — uniform sample
        return rng.choice(all_positions, size=min(n_rows, n), replace=False)

    bins = np.asarray(bins)
    unique_bins = np.unique(bins[~pd.isna(bins)])
    if len(unique_bins) < 2:
        return rng.choice(all_positions, size=min(n_rows, n), replace=False)

    # Compute per-bin positions and counts in one pass
    bin_positions = {b: all_positions[bins == b] for b in unique_bins}
    bin_counts = {b: len(bin_positions[b]) for b in unique_bins}

    # Per-bin weight — tails get tail_oversample, middle gets 1.0
    bin_min, bin_max = unique_bins.min(), unique_bins.max()
    weights = {b: (tail_oversample if b in (bin_min, bin_max) else 1.0) for b in unique_bins}
    total_weight = sum(weights[b] * bin_counts[b] for b in unique_bins)

    # First-pass per-bin target (clamped to bin capacity)
    targets: dict = {}
    for b in unique_bins:
        raw = n_rows * weights[b] * bin_counts[b] / total_weight
        targets[b] = max(1, min(int(round(raw)), bin_counts[b]))

    # Reconcile to exactly n_rows. Trimming the largest target avoids
    # disproportionately gutting small (often tail) bins.
    total = sum(targets.values())
    while total > n_rows:
        donor = max(targets, key=lambda k: targets[k])
        if targets[donor] <= 1:
            break
        targets[donor] -= 1
        total -= 1
    while total < n_rows:
        # Award an extra row to the bin with the most spare capacity.
        receiver = max(unique_bins, key=lambda k: bin_counts[k] - targets[k])
        if targets[receiver] >= bin_counts[receiver]:
            break
        targets[receiver] += 1
        total += 1

    sampled: list[np.ndarray] = []
    for b in unique_bins:
        chosen = rng.choice(bin_positions[b], size=targets[b], replace=False)
        sampled.append(chosen)
    return np.concatenate(sampled)
