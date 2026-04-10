import numpy as np
import pandas as pd


def bin_targets_by_quantile(targets, quantiles=None, labels=None):
    """
    Convert continuous targets to classes using quantile-based binning (qcut).

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Continuous regression targets
    quantiles : list of float, optional
        Quantile thresholds (e.g., [0.25, 0.5, 0.75] for quartiles)
        If None, defaults to [0.33, 0.67] (tertiles)
    labels : list, optional
        Labels for bins. If None, uses integer indices.

    Returns
    -------
    pd.Series
        Discrete class labels
    """
    if quantiles is None:
        quantiles = [0.33, 0.67]  # Default: 3 classes

    # Convert to Series if needed
    if isinstance(targets, np.ndarray):
        targets = pd.Series(targets)

    # Use qcut to bin by quantiles
    try:
        binned = pd.qcut(targets, q=quantiles, labels=labels, duplicates="drop")
    except ValueError as e:
        # Handle case where quantiles create duplicate bin edges
        print(f"Warning: {e}. Using fewer bins.")
        # Fall back to simpler binning
        n_bins = len(quantiles) + 1
        binned = pd.qcut(targets, q=n_bins, labels=labels, duplicates="drop")

    return binned


def bin_targets_by_threshold(targets, thresholds=None, labels=None):
    """
    Convert continuous targets to classes using fixed thresholds (cut).

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Continuous regression targets
    thresholds : list of float, optional
        Threshold values (e.g., [-0.5, 0.5] creates 3 classes)
        If None, defaults to [-0.5, 0.5]
    labels : list, optional
        Labels for bins. If None, uses integer indices.

    Returns
    -------
    pd.Series
        Discrete class labels
    """
    if thresholds is None:
        thresholds = [-0.5, 0.5]  # Default: 3 classes (negative, neutral, positive)

    # Convert to Series if needed
    if isinstance(targets, np.ndarray):
        targets = pd.Series(targets)

    # Add -inf and +inf to create complete bins
    bins = [-np.inf] + sorted(thresholds) + [np.inf]

    # Use cut to bin by fixed thresholds
    binned = pd.cut(targets, bins=bins, labels=labels)
    return binned


def create_balanced_bins(targets, n_bins=5):
    """
    Create bins with approximately balanced sample counts.

    Uses quantile-based binning to ensure each bin has roughly equal number of samples.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Continuous regression targets
    n_bins : int, default=5
        Number of bins to create

    Returns
    -------
    tuple of (pd.Series, list)
        - Binned classes
        - Bin boundaries (edges)
    """
    # Convert to Series if needed
    if isinstance(targets, np.ndarray):
        targets_series = pd.Series(targets)
    else:
        targets_series = targets

    # Create quantile bins
    binned, bin_edges = pd.qcut(targets_series, q=n_bins, labels=False, retbins=True, duplicates="drop")
    return binned, bin_edges.tolist()


def classify_by_sign(targets, neutral_threshold=0.1):
    """
    Classify targets into -1, 0, +1 based on sign and neutral threshold.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Continuous regression targets
    neutral_threshold : float, default=0.1
        Absolute value below which targets are considered neutral (0)

    Returns
    -------
    pd.Series or np.ndarray
        Classification labels: -1 (negative), 0 (neutral), +1 (positive)
    """
    if isinstance(targets, pd.Series):
        labels = pd.Series(index=targets.index, dtype=int)
        labels[targets > neutral_threshold] = 1
        labels[targets < -neutral_threshold] = -1
        labels[targets.abs() <= neutral_threshold] = 0
    else:
        labels = np.zeros_like(targets, dtype=int)
        labels[targets > neutral_threshold] = 1
        labels[targets < -neutral_threshold] = -1
    return labels


def create_bins_by_std(targets, n_std_per_bin=0.5):
    """
    Create bins based on standard deviation units.

    Parameters
    ----------
    targets : pd.Series or np.ndarray
        Continuous regression targets
    n_std_per_bin : float, default=0.5
        Number of standard deviations per bin

    Returns
    -------
    pd.Series
        Binned classes
    """
    # Calculate mean and std
    if isinstance(targets, pd.Series):
        mean = targets.mean()
        std = targets.std()
    else:
        mean = np.nanmean(targets)
        std = np.nanstd(targets)

    # Create bin edges based on std
    max_std = 3  # Cover ±3 std
    n_bins = int(2 * max_std / n_std_per_bin)

    bin_edges = [mean + (i - n_bins / 2) * n_std_per_bin * std for i in range(n_bins + 1)]
    bin_edges = [-np.inf] + bin_edges + [np.inf]

    # Convert to Series if needed
    if isinstance(targets, np.ndarray):
        targets = pd.Series(targets)

    binned = pd.cut(targets, bins=bin_edges, labels=False)
    return binned
