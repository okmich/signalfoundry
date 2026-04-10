"""
Label Comparison and Evaluation Utilities

This module provides functions to compare different labeling methods and evaluate which produces the most predictable,
well-separated, and economically meaningful labels.
"""

from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def compare_label_distributions(
        label_dict: Dict[str, pd.Series], feature: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compare the distribution of labels across different labeling methods.

    Parameters
    ----------
    label_dict : dict
        Dictionary mapping method names to label Series.
        Example: {'percentile': labels1, 'zscore': labels2, 'kmeans': labels3}

    feature : pd.Series, optional
        The underlying feature used for labeling (e.g., directional efficiency).
        If provided, will include feature statistics per label.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - method: labeling method name
        - label: label value (0, 1, 2, etc., or 'NA')
        - count: number of observations
        - percentage: percentage of total observations
        - feature_mean (if feature provided): mean feature value for this label
        - feature_std (if feature provided): std of feature value

    Examples
    --------
    >>> from okmich_quant_features.path_structure.labelling import *
    >>> prices = pd.Series(range(100, 200))
    >>> eff = directional_efficiency(prices, lookahead=10)
    >>>
    >>> labels_pct = percentile_labels(eff, n_states=3)
    >>> labels_z = zscore_labels(eff, z_threshold=1.0)
    >>> labels_km = kmeans_labels(eff, n_states=3)
    >>>
    >>> comparison = compare_label_distributions(
    ...     {'percentile': labels_pct, 'zscore': labels_z, 'kmeans': labels_km},
    ...     feature=eff
    ... )
    >>> print(comparison)
    """
    results = []

    for method_name, labels in label_dict.items():
        # Get value counts (excludes NA by default)
        value_counts = labels.value_counts().sort_index()
        total = len(labels)  # Total includes NA

        # Add rows for each label value
        for label_val, count in value_counts.items():
            row = {
                "method": method_name,
                "label": int(label_val),
                "count": count,
                "percentage": round(count / total * 100, 2),
            }

            # Add feature statistics if provided
            if feature is not None:
                label_mask = labels == label_val
                label_features = feature[label_mask].dropna()

                if len(label_features) > 0:
                    row["feature_mean"] = round(label_features.mean(), 4)
                    row["feature_std"] = round(label_features.std(), 4)
                    row["feature_min"] = round(label_features.min(), 4)
                    row["feature_max"] = round(label_features.max(), 4)

            results.append(row)

        # Add NA row if present
        na_count = labels.isna().sum()
        if na_count > 0:
            row = {
                "method": method_name,
                "label": "NA",
                "count": na_count,
                "percentage": round(na_count / total * 100, 2),
            }

            # Add feature statistics for NA rows
            if feature is not None:
                na_mask = labels.isna()
                na_features = feature[na_mask].dropna()

                if len(na_features) > 0:
                    row["feature_mean"] = round(na_features.mean(), 4)
                    row["feature_std"] = round(na_features.std(), 4)
                    row["feature_min"] = round(na_features.min(), 4)
                    row["feature_max"] = round(na_features.max(), 4)

            results.append(row)

    return pd.DataFrame(results)


def evaluate_label_separation(
        feature: Union[pd.Series, pd.DataFrame],
        labels: pd.Series,
        method_name: str = "unknown",
        sample_size: Optional[int] = 10000,
) -> Dict[str, float]:
    """
    Evaluate how well-separated the labels are using clustering metrics.

    Parameters
    ----------
    feature : pd.Series or pd.DataFrame
        Feature(s) used for labeling
    labels : pd.Series
        Generated labels
    method_name : str
        Name of the labeling method (for reporting)
    sample_size : int, optional
        Maximum number of samples to use for silhouette score calculation.
        Silhouette has O(n²) complexity, so we sample for large datasets.
        Set to None to use all data (slow for >10k samples).
        Default: 10000

    Returns
    -------
    dict
        Dictionary containing separation metrics:
        - silhouette_score: Higher is better (range: -1 to 1, >0.5 is good)
        - davies_bouldin_score: Lower is better (>0, closer to 0 is better)
        - calinski_harabasz_score: Higher is better (>0)
        - intra_class_variance: Lower is better (measures compactness within labels)
        - inter_class_distance: Higher is better (measures separation between labels)

    Notes
    -----
    These metrics help identify which labeling method produces the most distinct states:
    - High silhouette score (>0.5): Labels are well-separated
    - Low Davies-Bouldin (<1.0): Labels are compact and well-separated
    - High Calinski-Harabasz: Labels have high between-cluster vs within-cluster variance ratio

    Performance:
    - For large datasets (>10k samples), silhouette_score is computed on a random sample
    - Davies-Bouldin and Calinski-Harabasz are fast (O(n)) and use all data
    - Custom metrics (intra/inter) are fast and use all data
    """
    # Prepare data
    if isinstance(feature, pd.Series):
        X = feature.values.reshape(-1, 1)
    else:
        X = feature.values

    # Align and remove NaN
    valid_mask = (
        labels.notna() & feature.notna().all(axis=1)
        if isinstance(feature, pd.DataFrame)
        else labels.notna() & feature.notna()
    )
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask].values

    if len(X_valid) == 0:
        raise ValueError("No valid data after removing NaN values")

    n_labels = len(np.unique(labels_valid))
    if n_labels < 2:
        raise ValueError(f"Need at least 2 unique labels, got {n_labels}")

    # Calculate clustering metrics
    metrics = {"method": method_name, "n_labels": n_labels, "n_samples": len(X_valid)}

    # Silhouette score - SAMPLE if dataset is large (O(n²) complexity!)
    try:
        if sample_size is not None and len(X_valid) > sample_size:
            # Stratified sampling to preserve label distribution
            np.random.seed(42)
            sample_indices = []

            for label in np.unique(labels_valid):
                label_indices = np.where(labels_valid == label)[0]
                n_per_label = int(sample_size * len(label_indices) / len(X_valid))
                n_per_label = max(
                    min(n_per_label, len(label_indices)), 1
                )  # At least 1 per label

                sampled = np.random.choice(
                    label_indices, size=n_per_label, replace=False
                )
                sample_indices.extend(sampled)

            sample_indices = np.array(sample_indices)
            X_sample = X_valid[sample_indices]
            labels_sample = labels_valid[sample_indices]

            metrics["silhouette_score"] = silhouette_score(X_sample, labels_sample)
            metrics["silhouette_sampled"] = True
            metrics["silhouette_sample_size"] = len(sample_indices)
        else:
            metrics["silhouette_score"] = silhouette_score(X_valid, labels_valid)
            metrics["silhouette_sampled"] = False
            metrics["silhouette_sample_size"] = len(X_valid)
    except Exception as e:
        metrics["silhouette_score"] = np.nan
        metrics["silhouette_sampled"] = False
        metrics["silhouette_sample_size"] = 0

    # Davies-Bouldin - Fast O(n), use all data
    try:
        metrics["davies_bouldin_score"] = davies_bouldin_score(X_valid, labels_valid)
    except Exception as e:
        metrics["davies_bouldin_score"] = np.nan

    # Calinski-Harabasz - Fast O(n), use all data
    try:
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(
            X_valid, labels_valid
        )
    except Exception as e:
        metrics["calinski_harabasz_score"] = np.nan

    # Calculate intra-class variance (lower is better) - Fast, use all data
    intra_variance = []
    for label in np.unique(labels_valid):
        label_mask = labels_valid == label
        label_data = X_valid[label_mask]
        if len(label_data) > 1:
            intra_variance.append(np.var(label_data))

    metrics["intra_class_variance"] = (
        np.mean(intra_variance) if intra_variance else np.nan
    )

    # Calculate inter-class distance (higher is better) - Fast, use all data
    label_means = []
    for label in np.unique(labels_valid):
        label_mask = labels_valid == label
        label_data = X_valid[label_mask]
        label_means.append(np.mean(label_data, axis=0))

    if len(label_means) > 1:
        label_means = np.array(label_means)
        # Calculate pairwise distances between centroids
        from scipy.spatial.distance import pdist

        distances = pdist(label_means)
        metrics["inter_class_distance"] = np.mean(distances)
    else:
        metrics["inter_class_distance"] = np.nan

    return metrics


def compare_all_methods(
        feature: Union[pd.Series, pd.DataFrame],
        label_dict: Dict[str, pd.Series],
        sample_size: Optional[int] = 10000,
) -> pd.DataFrame:
    """
    Compare separation quality across all labeling methods.

    Parameters
    ----------
    feature : pd.Series or pd.DataFrame
        Feature(s) used for labeling
    label_dict : dict
        Dictionary mapping method names to label Series
    sample_size : int, optional
        Maximum number of samples for silhouette score calculation.
        Default: 10000. Set to None to use all data (slow for >10k samples).

    Returns
    -------
    pd.DataFrame
        Comparison of all methods with separation metrics

    Examples
    --------
    >>> eff = directional_efficiency(prices, lookahead=10)
    >>> labels = {
    ...     'percentile': percentile_labels(eff, n_states=3),
    ...     'zscore': zscore_labels(eff, z_threshold=1.0),
    ...     'kmeans': kmeans_labels(eff, n_states=3)
    ... }
    >>> comparison = compare_all_methods(eff, labels)
    >>> print(comparison.sort_values('silhouette_score', ascending=False))

    Notes
    -----
    For large datasets (>10k samples), silhouette scores are computed on a stratified
    random sample to avoid O(n²) computational complexity. This provides a good
    approximation in a fraction of the time.
    """
    results = []

    for method_name, labels in label_dict.items():
        try:
            metrics = evaluate_label_separation(
                feature, labels, method_name, sample_size=sample_size
            )
            results.append(metrics)
        except Exception as e:
            print(f"Warning: Could not evaluate {method_name}: {e}")

    df = pd.DataFrame(results)

    # Add interpretation column for silhouette score
    if "silhouette_score" in df.columns:
        df["silhouette_interpretation"] = pd.cut(
            df["silhouette_score"],
            bins=[-1, 0.25, 0.5, 0.7, 1],
            labels=["Poor", "Fair", "Good", "Excellent"],
        )

    return df


def plot_label_distributions(
        label_dict: Dict[str, pd.Series],
        feature: Optional[pd.Series] = None,
        figsize: tuple = (15, 10),
):
    """
    Visualize label distributions across different methods.

    Parameters
    ----------
    label_dict : dict
        Dictionary mapping method names to label Series
    feature : pd.Series, optional
        If provided, will plot histograms colored by labels
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    n_methods = len(label_dict)

    if feature is not None:
        # Create subplots: bar charts + histograms
        fig, axes = plt.subplots(2, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
    else:
        # Just bar charts
        fig, axes = plt.subplots(1, n_methods, figsize=(figsize[0], figsize[1] // 2))
        if n_methods == 1:
            axes = [axes]

    for idx, (method_name, labels) in enumerate(label_dict.items()):
        # Bar chart of label counts
        ax_bar = axes[0, idx] if feature is not None else axes[idx]

        # Get value counts including NA
        value_counts = labels.value_counts().sort_index()
        na_count = labels.isna().sum()
        total = len(labels)

        # Calculate percentages
        percentages = value_counts / total * 100

        # Create bar positions and labels
        bar_labels = [str(int(x)) for x in value_counts.index]
        bar_values = percentages.values

        # Add NA bar if present
        if na_count > 0:
            bar_labels.append("NA")
            bar_values = np.append(bar_values, na_count / total * 100)

        # Create bars
        colors = ["C0"] * len(value_counts)
        if na_count > 0:
            colors.append("gray")

        ax_bar.bar(
            bar_labels, bar_values, color=colors, edgecolor="black", linewidth=0.5
        )
        ax_bar.set_xlabel("Label")
        ax_bar.set_ylabel("Percentage (%)")
        ax_bar.set_title(f"{method_name}\nLabel Distribution")
        ax_bar.grid(axis="y", alpha=0.3)

        # Add percentage labels on bars
        for i, (label, pct) in enumerate(zip(bar_labels, bar_values)):
            ax_bar.text(i, pct, f"{pct:.1f}%", ha="center", va="bottom")

        # Histogram colored by labels
        if feature is not None:
            ax_hist = axes[1, idx]

            # Get unique labels, including NaN/NA as a separate category
            unique_labels = []
            has_na = labels.isna().any()

            for val in labels.unique():
                if pd.notna(val):
                    unique_labels.append(val)

            # Sort numeric labels
            unique_labels = sorted(unique_labels)

            # Plot each label category
            for label in unique_labels:
                label_mask = labels == label
                label_features = feature[label_mask].dropna()

                if len(label_features) > 0:
                    ax_hist.hist(
                        label_features, bins=30, alpha=0.5, label=f"Label {int(label)}"
                    )

            # Plot NA category if it exists
            if has_na:
                na_mask = labels.isna()
                na_features = feature[na_mask].dropna()

                if len(na_features) > 0:
                    ax_hist.hist(
                        na_features,
                        bins=30,
                        alpha=0.5,
                        label="Label NA (unlabeled)",
                        color="gray",
                        edgecolor="black",
                        linewidth=0.5,
                    )

            ax_hist.set_xlabel("Feature Value")
            ax_hist.set_ylabel("Frequency")
            ax_hist.set_title(f"{method_name}\nFeature Distribution by Label")
            ax_hist.legend()
            ax_hist.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def calculate_label_agreement(label_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    Calculate pairwise agreement between different labeling methods.

    Uses Chi-squared test to assess if label assignments are independent.

    Parameters
    ----------
    label_dict : dict
        Dictionary mapping method names to label Series

    Returns
    -------
    pd.DataFrame
        Pairwise agreement matrix with p-values from chi-squared tests.
        Low p-values (<0.05) indicate significant association between methods.
    """
    methods = list(label_dict.keys())
    n_methods = len(methods)

    agreement_matrix = np.zeros((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                agreement_matrix[i, j] = 1.0  # Perfect agreement with itself
            elif i < j:
                # Create contingency table
                df = pd.DataFrame(
                    {"method1": label_dict[method1], "method2": label_dict[method2]}
                ).dropna()

                if len(df) > 0:
                    contingency = pd.crosstab(df["method1"], df["method2"])
                    chi2, p_value, dof, expected = chi2_contingency(contingency)

                    # Use Cramér's V as agreement measure (0 to 1)
                    n = contingency.sum().sum()
                    min_dim = min(contingency.shape) - 1
                    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

                    agreement_matrix[i, j] = cramers_v
                    agreement_matrix[j, i] = cramers_v
                else:
                    agreement_matrix[i, j] = np.nan
                    agreement_matrix[j, i] = np.nan

    return pd.DataFrame(agreement_matrix, index=methods, columns=methods)
