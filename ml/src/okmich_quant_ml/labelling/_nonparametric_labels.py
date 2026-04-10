"""
Shared Non-Parametric Labeling Methods

This module implements unsupervised learning-based labeling methods that discover patterns in the data without relying on predefined thresholds.

These methods support BOTH univariate and multivariate inputs:
- Univariate: Single pd.Series (e.g., directional efficiency or forward volatility)
- Multivariate: pd.DataFrame with multiple forward-looking features

This is a shared utility module used by both:
- okmich.quant.features.path_structure.labelling
- okmich.quant.features.volatility.labelling
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from okmich_quant_ml.hmm import DistType, PomegranateHMM
from okmich_quant_ml.clustering import SlicedWassersteinKMeans


def kmeans_labels(features: Union[pd.Series, pd.DataFrame], n_states: int = 3, random_state: int = 42,
                  scale: bool = True, **kwargs) -> pd.Series:
    """
    Create categorical labels using K-Means clustering.
    This method partitions the feature space into n_states clusters and assigns labels based on the
    mean feature value(s) of each cluster.

    Parameters
    ----------
    features : pd.Series or pd.DataFrame
        Forward-looking feature(s) to cluster.
        - pd.Series: Univariate (e.g., directional efficiency only)
        - pd.DataFrame: Multivariate (e.g., efficiency + volatility)

    n_states : int, default=3
        Number of clusters/states to create.

    random_state : int, default=42
        Random seed for reproducibility.

    scale : bool, default=True
        Whether to standardize features before clustering.
        Recommended when using multivariate features with different scales.

    **kwargs
        Additional keyword arguments passed to sklearn.cluster.KMeans.

    Returns
    -------
    pd.Series
        Categorical labels ordered by cluster centroid values.
        For ascending features (volatility):
            - Label 0: Cluster with lowest mean (low volatility)
            - Label 1: Cluster with middle mean (medium volatility)
            - Label 2: Cluster with highest mean (high volatility)

    Examples
    --------
    >>> # Univariate: Volatility only
    >>> fwd_vol = forward_realized_volatility(prices, lookahead=24)
    >>> labels = kmeans_labels(fwd_vol, n_states=3)
    >>>
    >>> # Multivariate: Volatility + ATR
    >>> features = pd.DataFrame({'fwd_vol': fwd_vol, 'atr': atr_series})
    >>> labels = kmeans_labels(features, n_states=3)

    Notes
    -----
    - Clustering discovers natural groupings in the data
    - No predefined thresholds needed
    - Labels are ordered by the mean of the primary feature (first column if DataFrame)
    - For multivariate data, scaling prevents dominance by high-variance features
    """
    # Handle both Series and DataFrame inputs
    if isinstance(features, pd.Series):
        X = features.dropna().values.reshape(-1, 1)
        index = features.dropna().index
        original_index = features.index
        feature_name = features.name if features.name else "feature"
    elif isinstance(features, pd.DataFrame):
        X = features.dropna().values
        index = features.dropna().index
        original_index = features.index
        feature_name = features.columns[0]
    else:
        raise TypeError("features must be either pd.Series or pd.DataFrame")

    if len(X) == 0:
        raise ValueError("No valid data after removing NaN values")

    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    # Scale if requested
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Fit K-Means
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, **kwargs)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Get centroids in original scale
    if scale:
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        centroids = kmeans.cluster_centers_

    # Order clusters by the first feature (ascending)
    centroid_order = np.argsort(centroids[:, 0])
    label_mapping = {
        old_label: new_label for new_label, old_label in enumerate(centroid_order)
    }

    # Apply mapping
    mapped_labels = np.array([label_mapping[label] for label in cluster_labels])

    # Create result Series with NaN for missing indices
    result = pd.Series(np.nan, index=original_index, dtype="Int64")
    result.loc[index] = mapped_labels

    result.name = f"{feature_name}_kmeans_labels"
    return result


def hmm_labels(features: Union[pd.Series, pd.DataFrame], n_states: int = 3, distribution: str = "normal",
               random_state: int = 42, scale: bool = True, **kwargs) -> pd.Series:
    """
    Create categorical labels using Hidden Markov Model (HMM).

    HMMs learn both the state definitions and transition probabilities,
    making them powerful regime filters that account for temporal dependencies.

    Parameters
    ----------
    features : pd.Series or pd.DataFrame
        Forward-looking feature(s) to model.
        - pd.Series: Univariate
        - pd.DataFrame: Multivariate

    n_states : int, default=3
        Number of hidden states.

    distribution : str, default='normal'
        Distribution type for pomegranate HMM:
        - 'normal': Gaussian distribution (most common)
        - 'lambda': Lambda distribution (for bounded data)

    random_state : int, default=42
        Random seed for reproducibility.

    scale : bool, default=True
        Whether to standardize features before training.

    **kwargs
        Additional keyword arguments passed to the HMM constructor.

    Returns
    -------
    pd.Series
        Categorical labels from Viterbi decoding, ordered by mean feature values.
        For ascending features (volatility):
            - Label 0: State with lowest mean (low volatility)
            - Label 1: State with middle mean (medium volatility)
            - Label 2: State with highest mean (high volatility)

    Examples
    --------
    >>> # Univariate HMM
    >>> fwd_vol = forward_realized_volatility(prices, lookahead=24)
    >>> labels = hmm_labels(fwd_vol, n_states=3, hmm_type='pomegranate')
    >>>
    >>> # Multivariate HMM: Volatility + Jump component
    >>> features = pd.DataFrame({'fwd_vol': fwd_vol, 'jump_var': jv})
    >>> labels = hmm_labels(features, n_states=3)

    Notes
    -----
    - HMMs capture regime persistence through transition probabilities
    - More sophisticated than clustering - accounts for temporal structure
    - Viterbi algorithm finds most likely state sequence
    - Labels are mapped to ensure ordering by mean feature value
    - For volatility:
        * Label 0 = Low volatility regime
        * Label 1 = Medium volatility regime
        * Label 2 = High volatility regime
    """
    # Handle both Series and DataFrame inputs
    if isinstance(features, pd.Series):
        X = features.dropna().values.reshape(-1, 1)
        index = features.dropna().index
        original_index = features.index
        feature_name = features.name if features.name else "feature"
    elif isinstance(features, pd.DataFrame):
        X = features.dropna().values
        index = features.dropna().index
        original_index = features.index
        feature_name = features.columns[0]
    else:
        raise TypeError("features must be either pd.Series or pd.DataFrame")

    if len(X) == 0:
        raise ValueError("No valid data after removing NaN values")

    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    # Scale if requested
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    _ALLOWED_DISTRIBUTIONS = {"lambda", "normal"}
    if distribution not in _ALLOWED_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Allowed values: {sorted(_ALLOWED_DISTRIBUTIONS)}"
        )
    dist_type = DistType.LAMDA if distribution == "lambda" else DistType.NORMAL
    model = PomegranateHMM(distribution_type=dist_type, n_states=n_states, random_state=random_state, **kwargs)

    # Fit and predict
    state_sequence = model.fit_predict(X_scaled).astype(np.int32)

    # Calculate mean feature value for each state (using original scale)
    state_means = []
    for state in range(n_states):
        state_mask = state_sequence == state
        if state_mask.sum() > 0:
            state_mean = X[state_mask, 0].mean()  # Use first feature for ordering
        else:
            state_mean = np.nan
        state_means.append(state_mean)

    # Order states by mean value (ascending)
    state_order = np.argsort(state_means)
    label_mapping = {old_state: new_label for new_label, old_state in enumerate(state_order)}

    # Apply mapping
    mapped_labels = np.array([label_mapping[state] for state in state_sequence])

    # Create result Series with NaN for missing indices
    result = pd.Series(np.nan, index=original_index, dtype="Int64")
    result.loc[index] = mapped_labels

    result.name = f"{feature_name}_hmm_labels"
    return result


def swkmeans_labels(
    features: Union[pd.Series, pd.DataFrame],
    n_states: int = 3,
    window_size: int = 50,
    lifting_size: int = 20,
    n_projections: int = 16,
    chunk_size: int = 2500,
    random_state: int = 42,
    scale: bool = True,
    **kwargs,
) -> pd.Series:
    """
    Create categorical labels using Sliced-Wasserstein K-Means clustering.

    This method uses time-series-aware clustering based on Sliced Wasserstein distances,
    which compares distributions of windowed sequences rather than point-by-point values.
    This makes it particularly effective for time series regime detection.

    Parameters
    ----------
    features : pd.Series or pd.DataFrame
        Forward-looking feature(s) to cluster.
        - pd.Series: Univariate time series
        - pd.DataFrame: Multivariate time series

    n_states : int, default=3
        Number of clusters/regimes to identify.

    window_size : int, default=50
        Size of the sliding window for creating time series subsequences.
        Larger values capture longer-term patterns.
        Should be larger than lifting_size.

    lifting_size : int, default=20
        Sliding window offset/stride parameter.
        Smaller values increase overlap between windows (smoother labels).
        Must be > 0 and < window_size.

    n_projections : int, default=16
        Number of random projection directions for sliced Wasserstein distance.
        More projections improve accuracy but increase computation time.
        Recommended range: 10-50.

    chunk_size : int, default=2500
        Number of sequences to process in each batch.
        Affects memory usage and computation speed.
        Larger = faster but more memory.

    random_state : int, default=42
        Random seed for reproducibility.

    scale : bool, default=True
        Whether to standardize features before clustering.
        Recommended for multivariate features with different scales.

    **kwargs
        Additional keyword arguments passed to SlicedWassersteinKMeans.

    Returns
    -------
    pd.Series
        Categorical labels ordered by cluster centroid values.
        For ascending features (volatility):
            - Label 0: Cluster with lowest mean (low volatility regime)
            - Label 1: Cluster with middle mean (medium volatility regime)
            - Label 2: Cluster with highest mean (high volatility regime)

    Examples
    --------
    >>> # Univariate: Volatility regime detection
    >>> fwd_vol = forward_realized_volatility(prices, lookahead=24)
    >>> labels = swkmeans_labels(fwd_vol, n_states=3, window_size=50)
    >>>
    >>> # Multivariate: Volatility + Jump component
    >>> features = pd.DataFrame({'fwd_vol': fwd_vol, 'jump_var': jv})
    >>> labels = swkmeans_labels(features, n_states=3, window_size=50)
    >>>
    >>> # Custom parameters for high-frequency data
    >>> labels = swkmeans_labels(fwd_vol, n_states=4,
    ...                          window_size=100, lifting_size=10,
    ...                          n_projections=32)

    Notes
    -----
    - Sliced-Wasserstein distance compares distributions of windowed sequences
    - More sophisticated than standard K-Means for time series
    - Captures temporal structure and patterns within windows
    - Particularly effective for:
        * Regime detection (volatility clustering, momentum regimes)
        * Pattern recognition in financial time series
        * Robust to noise and outliers (especially with p=1)
    - Window parameters:
        * window_size: Controls pattern length to capture
        * lifting_size: Controls label smoothness (smaller = smoother)
    - For univariate data, automatically reshapes to (n_samples, 1)
    - Labels are ordered by mean feature value for consistency

    References
    ----------
    Paper: https://www.aimspress.com/article/doi/10.3934/DSFE.2025016
    "Sliced-Wasserstein K-Means for Multidimensional Time Series"

    See Also
    --------
    kmeans_labels : Standard K-Means clustering (simpler, faster)
    hmm_labels : Hidden Markov Model with transition probabilities
    """
    # Handle both Series and DataFrame inputs
    if isinstance(features, pd.Series):
        X = features.dropna().values.reshape(-1, 1)
        index = features.dropna().index
        original_index = features.index
        feature_name = features.name if features.name else "feature"
    elif isinstance(features, pd.DataFrame):
        X = features.dropna().values
        index = features.dropna().index
        original_index = features.index
        feature_name = features.columns[0]
    else:
        raise TypeError("features must be either pd.Series or pd.DataFrame")

    if len(X) == 0:
        raise ValueError("No valid data after removing NaN values")

    if n_states < 2:
        raise ValueError("n_states must be at least 2")

    if window_size <= lifting_size:
        raise ValueError(
            f"window_size ({window_size}) must be greater than lifting_size ({lifting_size})"
        )

    if lifting_size < 1:
        raise ValueError("lifting_size must be at least 1")

    # Check if we have enough data for windowing
    min_required = window_size + (n_states - 1) * lifting_size
    if len(X) < min_required:
        raise ValueError(
            f"Insufficient data: need at least {min_required} samples for "
            f"window_size={window_size}, lifting_size={lifting_size}, n_states={n_states}. "
            f"Got {len(X)} samples."
        )

    # SlicedWassersteinKMeans requires POSITIVE values (computes log returns internally)
    # So we shift data to be positive rather than using StandardScaler
    if scale:
        # Min-max scaling to [1, 2] range (all positive, avoids log(0))
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(1.0, 2.0))
        X_scaled = scaler.fit_transform(X)
    else:
        # Still need to ensure positive values
        X_min = X.min(axis=0)
        if np.any(X_min <= 0):
            # Shift to make all values positive
            X_scaled = X - X_min + 1.0
        else:
            X_scaled = X

    # Fit Sliced-Wasserstein K-Means
    swk = SlicedWassersteinKMeans(n_clusters=n_states, window_size=window_size, lifting_size=lifting_size,
                                  chunk_size=chunk_size, n_projections=n_projections, random_state=random_state,
                                  n_init=3,  # Multiple initializations for better clustering
                                **kwargs)
    # SlicedWassersteinKMeans expects price/value data (not returns)
    # It will compute returns internally
    # The fit_predict returns labels for time series (same length as returns = n-1)
    try:
        timeseries_labels = swk.fit_predict(X_scaled)
    except RuntimeError as e:
        # If all initializations failed, fall back to standard kmeans
        raise ValueError(
            f"SWK-Means clustering failed: {str(e)}. "
            f"This may happen with insufficient data or poor initialization. "
            f"Try reducing n_states, window_size, or using kmeans_labels instead."
        ) from e

    # timeseries_labels has length = len(X_scaled) - 1 (one per return/time point after first)
    # The model's internal logic maps sequence labels to time series labels
    expected_len = len(X_scaled) - 1  # Returns have n-1 length
    if len(timeseries_labels) != expected_len:
        raise RuntimeError(
            f"Unexpected number of labels: got {len(timeseries_labels)}, "
            f"expected {expected_len} (n_samples - 1 for returns)"
        )

    # Calculate mean feature value for each cluster (using original scale)
    # Note: features X correspond to time points, labels correspond to returns (shifted by 1)
    # So we align: label[i] describes the transition from X[i] to X[i+1]
    # We'll assign label[i] to time point i+1 (the destination point)
    cluster_means = []
    for cluster in range(n_states):
        cluster_mask = timeseries_labels == cluster
        if cluster_mask.sum() > 0:
            # Get the feature values at positions i+1 for all labels[i] == cluster
            # This corresponds to X[1:][cluster_mask]
            cluster_mean = X[1:][
                cluster_mask, 0
            ].mean()  # Use first feature for ordering
        else:
            cluster_mean = np.nan
        cluster_means.append(cluster_mean)

    # Order clusters by mean value (ascending)
    cluster_order = np.argsort(cluster_means)
    label_mapping = {
        old_cluster: new_label for new_label, old_cluster in enumerate(cluster_order)
    }

    # Apply mapping to reorder labels
    mapped_labels = np.array([label_mapping[label] for label in timeseries_labels])

    # Create result Series with NaN for missing indices
    result = pd.Series(np.nan, index=original_index, dtype="Int64")

    # Assign labels to positions [1:] (skip first point as no return defined)
    # Labels describe the state at each time point based on recent return patterns
    result.loc[index[1:]] = mapped_labels.astype(np.int64)

    result.name = f"{feature_name}_swkmeans_labels"
    return result

def get_cluster_statistics(features: Union[pd.Series, pd.DataFrame], labels: pd.Series) -> pd.DataFrame:
    """
    Calculate descriptive statistics for each cluster/state.

    Parameters
    ----------
    features : pd.Series or pd.DataFrame
        Original features used for clustering
    labels : pd.Series
        Cluster/state labels

    Returns
    -------
    pd.DataFrame
        Statistics for each label including:
        - count: number of observations
        - mean: mean feature value(s)
        - std: standard deviation
        - min/max: range
    """
    # Combine features and labels
    if isinstance(features, pd.Series):
        df = pd.DataFrame({"feature": features, "label": labels})
        group_cols = ["feature"]
    else:
        df = features.copy()
        df["label"] = labels
        group_cols = features.columns.tolist()

    # Drop NaN labels
    df = df.dropna(subset=["label"])

    if len(df) == 0:
        raise ValueError("No valid data after removing NaN labels")

    # Calculate statistics per label
    stats_list = []
    for label in sorted(df["label"].unique()):
        label_df = df[df["label"] == label]

        stats = {"label": int(label), "count": len(label_df)}

        # Add stats for each feature
        for col in group_cols:
            stats[f"{col}_mean"] = label_df[col].mean()
            stats[f"{col}_std"] = label_df[col].std()
            stats[f"{col}_min"] = label_df[col].min()
            stats[f"{col}_max"] = label_df[col].max()

        stats_list.append(stats)

    return pd.DataFrame(stats_list)
