"""
Stage 3 — Redundancy Reduction (Hierarchical Clustering)
=========================================================
Among correlated features, keep only the most informative representative.
This prevents multicollinearity from distorting importance scores in later stages.

Method: Agglomerative clustering on 1 - |Spearman correlation|.
From each cluster, keep the feature with the highest IC-IR score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ._result import StageReport


def stage3_redundancy(X: pd.DataFrame, icir_scores: dict[str, float], corr_threshold: float = 0.75,
                      verbose: bool = True) -> tuple[pd.DataFrame, StageReport, dict]:
    """
    Hierarchical clustering redundancy filter.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    icir_scores : dict[str, float]
        IC-IR score per feature (from Stage 2). Used to pick the cluster representative. Falls back to 0.0 for any feature not in the dict.
    corr_threshold : float
        Spearman |correlation| above which two features are considered redundant. Default 0.75.
    verbose : bool

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    cluster_info : dict ``{"assignments": {cluster_id: [feature_names]}, "representatives": {cluster_id: surviving_feature}}``.
        Needed by downstream leakage diagnostics to trace which features were absorbed into which cluster representative.
    """
    n_before = X.shape[1]

    if n_before == 0:
        report = StageReport("Stage3_Redundancy", 0, 0, [])
        return X, report, {"assignments": {}, "representatives": {}}

    if n_before == 1:
        only = X.columns[0]
        report = StageReport("Stage3_Redundancy", 1, 1, [])
        return X, report, {"assignments": {1: [only]}, "representatives": {1: only}}

    # Compute Spearman correlation on filled data
    X_filled = X.fillna(X.median())
    corr = X_filled.corr(method="spearman").abs().clip(0, 1)

    # Convert correlation to distance; ensure symmetry and zero diagonal
    dist = (1.0 - corr).clip(0)
    np.fill_diagonal(dist.values, 0.0)

    # Condensed distance vector for scipy
    condensed = squareform(dist.values, checks=False)
    condensed = np.clip(condensed, 0, None)  # fix any floating-point negatives

    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")

    # From each cluster, select the feature with the highest IC-IR
    n_clusters = labels.max()
    kept, removed = [], []
    assignments: dict[int, list[str]] = {}
    representatives: dict[int, str] = {}

    for cluster_id in range(1, n_clusters + 1):
        cluster_cols = [col for col, lbl in zip(X.columns, labels) if lbl == cluster_id]
        assignments[int(cluster_id)] = cluster_cols
        if len(cluster_cols) == 1:
            kept.append(cluster_cols[0])
            representatives[int(cluster_id)] = cluster_cols[0]
        else:
            best = max(cluster_cols, key=lambda c: icir_scores.get(c, 0.0))
            kept.append(best)
            representatives[int(cluster_id)] = best
            removed.extend([c for c in cluster_cols if c != best])

    if verbose:
        print(f"  Stage 3 clustering (corr>={corr_threshold}): "
              f"{n_before} -> {len(kept)} features, {n_clusters} clusters, "
              f"{len(removed)} redundant removed")

    report = StageReport(stage="Stage3_Redundancy", n_before=n_before, n_after=len(kept), removed=removed)
    cluster_info = {"assignments": assignments, "representatives": representatives}
    return X[kept], report, cluster_info
