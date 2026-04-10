"""
Shared Parametric Labeling Methods

This module implements threshold-based labeling methods that use statistical parameters (percentiles, z-scores) to
create categorical labels from continuous forward-looking features.

These methods are UNIVARIATE only - they accept a single pd.Series as input.
For multivariate labeling, use the non-parametric methods (clustering, HMM).

This is a shared utility module used by both:
- okmich_quant_features.path_structure.labelling
- okmich_quant_features.volatility.labelling
"""

from typing import Tuple

import numpy as np
import pandas as pd


def percentile_labels(
        feature: pd.Series,
        n_states: int = 3,
        percentiles: Tuple[float, ...] = None,
        labels: Tuple[int, ...] = None,
) -> pd.Series:
    """
    Create categorical labels using percentile-based thresholds (adaptive/dynamic method).

    This method divides the feature distribution into n_states regions using percentiles.
    It adapts to the data distribution, making it suitable for non-stationary markets.

    Parameters
    ----------
    feature : pd.Series
        Univariate forward-looking feature (e.g., directional efficiency, forward volatility).
        Should be continuous values.

    n_states : int, default=3
        Number of categorical states to create.
        Default is 3 states.

    percentiles : tuple of float, optional
        Custom percentile boundaries for state transitions.
        For n_states=3, default is (25, 75) meaning:
            - Bottom 25%: State 0
            - Middle 50%: State 1
            - Top 25%: State 2

        For n_states=5, default is (20, 40, 60, 80).

        Must have exactly (n_states - 1) values.

    labels : tuple of int, optional
        Custom label values for each state.
        Default is range(n_states): [0, 1, 2, ...]
        Must have exactly n_states values.

    Returns
    -------
    pd.Series
        Categorical labels with the same index as the input feature.
        NaN values in the input will result in NaN labels.

    Examples
    --------
    >>> import pandas as pd
    >>> from okmich_quant_features.labelling import percentile_labels
    >>>
    >>> # For path structure (directional efficiency)
    >>> eff = directional_efficiency(prices, lookahead=24)
    >>> labels = percentile_labels(eff, n_states=3)
    >>> # Label 0: High choppiness (low efficiency)
    >>> # Label 2: Low choppiness (high efficiency)
    >>>
    >>> # For volatility (forward realized vol)
    >>> fwd_vol = forward_realized_volatility(prices, lookahead=24)
    >>> labels = percentile_labels(fwd_vol, n_states=3)
    >>> # Label 0: Low volatility
    >>> # Label 2: High volatility

    Notes
    -----
    - This method is adaptive to the data distribution
    - Works well with non-stationary markets where regimes shift over time
    - Percentile thresholds are calculated on non-NaN values only
    - Interpretation depends on the feature:
        * For directional efficiency: Low label = choppy, High label = smooth
        * For volatility: Low label = quiet, High label = volatile
    """
    # Validate inputs
    if not isinstance(feature, pd.Series):
        raise TypeError("feature must be a pandas Series (univariate input)")

    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    # Set default percentiles if not provided
    if percentiles is None:
        percentiles = tuple(np.linspace(0, 100, n_states + 1)[1:-1])
    else:
        if len(percentiles) != n_states - 1:
            raise ValueError(
                f"percentiles must have exactly {n_states - 1} values for {n_states} states"
            )

    # Set default labels if not provided
    if labels is None:
        labels = tuple(range(n_states))
    else:
        if len(labels) != n_states:
            raise ValueError(f"labels must have exactly {n_states} values")

    # Calculate percentile thresholds on non-NaN values
    valid_data = feature.dropna()
    if len(valid_data) == 0:
        raise ValueError("feature contains no valid (non-NaN) values")

    thresholds = np.percentile(valid_data, percentiles)

    # Create labels using np.digitize
    result = pd.Series(np.nan, index=feature.index, dtype="Int64")

    # Only label non-NaN values
    valid_mask = feature.notna()
    result.loc[valid_mask] = pd.Series(
        np.digitize(feature.loc[valid_mask], thresholds),
        index=feature.loc[valid_mask].index,
    )

    # Map bin indices to custom labels if provided
    if labels != tuple(range(n_states)):
        mapping = {i: labels[i] for i in range(n_states)}
        result = result.map(mapping)

    result.name = (
        f"{feature.name}_percentile_labels" if feature.name else "percentile_labels"
    )
    return result


def zscore_labels(
        feature: pd.Series,
        n_states: int = 3,
        z_threshold: float = None,
        labels: Tuple[int, ...] = None,
) -> pd.Series:
    """
    Create categorical labels using z-score-based thresholds (outlier-sensitive method).

    This method defines thresholds based on standard deviations from the mean.
    It's more sensitive to outliers and highlights extreme values.

    Parameters
    ----------
    feature : pd.Series
        Univariate forward-looking feature (e.g., directional efficiency, forward volatility).
        Should be continuous values.

    n_states : int, default=3
        Number of categorical states to create.
        Currently only supports n_states=3.

    z_threshold : float, optional
        Number of standard deviations from mean to use as threshold.
        For n_states=3:
            - State 0: feature <= (μ - z*σ)
            - State 1: (μ - z*σ) < feature < (μ + z*σ)
            - State 2: feature >= (μ + z*σ)

        Default values:
            - z_threshold=1.0 creates roughly 68% in middle state
            - z_threshold=1.5 creates roughly 87% in middle state

        Suggested starting value: 1.0 or 1.5

    labels : tuple of int, optional
        Custom label values for each state.
        Default is (0, 1, 2) for n_states=3.

    Returns
    -------
    pd.Series
        Categorical labels with the same index as the input feature.
        NaN values in the input will result in NaN labels.

    Examples
    --------
    >>> import pandas as pd
    >>> from okmich_quant_features.labelling import zscore_labels
    >>>
    >>> # For volatility
    >>> fwd_vol = forward_realized_volatility(prices, lookahead=24)
    >>> labels = zscore_labels(fwd_vol, z_threshold=1.0)
    >>> # Label 0: Extremely low volatility (< μ - 1σ)
    >>> # Label 1: Normal volatility
    >>> # Label 2: Extremely high volatility (> μ + 1σ)

    Notes
    -----
    - This method highlights extreme values (outliers)
    - More sensitive to the tails of the distribution than percentile method
    - Interpretation depends on feature type
    - Consider using z_threshold=1.0 for balanced distribution
    - Use z_threshold=1.5 or 2.0 for more conservative extreme detection
    - Note: Volatility distributions often have fat tails, so z-score may
      not perfectly align with percentiles
    """
    # Validate inputs
    if not isinstance(feature, pd.Series):
        raise TypeError("feature must be a pandas Series (univariate input)")

    if n_states != 3:
        raise ValueError("zscore_labels currently only supports n_states=3")

    # Set default z_threshold
    if z_threshold is None:
        z_threshold = 1.0

    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")

    # Set default labels
    if labels is None:
        labels = (0, 1, 2)
    else:
        if len(labels) != n_states:
            raise ValueError(f"labels must have exactly {n_states} values")

    # Calculate mean and std on non-NaN values
    valid_data = feature.dropna()
    if len(valid_data) == 0:
        raise ValueError("feature contains no valid (non-NaN) values")

    if len(valid_data) < 2:
        raise ValueError("feature must have at least 2 non-NaN values to calculate std")

    mu = valid_data.mean()
    sigma = valid_data.std()

    if sigma == 0:
        raise ValueError(
            "feature has zero standard deviation (all values are identical)"
        )

    # Calculate thresholds
    lower_threshold = mu - z_threshold * sigma
    upper_threshold = mu + z_threshold * sigma

    # Create labels
    result = pd.Series(np.nan, index=feature.index, dtype="Int64")

    # Only label non-NaN values
    valid_mask = feature.notna()

    # Assign labels based on thresholds
    result.loc[valid_mask & (feature <= lower_threshold)] = labels[0]  # Low
    result.loc[
        valid_mask & (feature > lower_threshold) & (feature < upper_threshold)
        ] = labels[
        1
    ]  # Medium
    result.loc[valid_mask & (feature >= upper_threshold)] = labels[2]  # High

    result.name = f"{feature.name}_zscore_labels" if feature.name else "zscore_labels"
    return result


def get_label_distribution(labels: pd.Series) -> pd.DataFrame:
    """
    Get the distribution of labels to check balance.

    Parameters
    ----------
    labels : pd.Series
        Categorical labels

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: label, count, percentage
        Sorted by label value
    """
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a pandas Series")

    # Count excluding NaN
    value_counts = labels.value_counts().sort_index()
    total = value_counts.sum()

    result = pd.DataFrame(
        {
            "label": value_counts.index,
            "count": value_counts.values,
            "percentage": (value_counts.values / total * 100).round(2),
        }
    )

    return result
