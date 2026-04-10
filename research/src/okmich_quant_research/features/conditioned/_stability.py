"""Stability scores and sub-period IC checks."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_stability_scores(ic_matrix: pd.DataFrame) -> pd.Series:
    """
    Compute stability_score = 1 - std(IC) / mean(|IC|) per feature across conditions.

    Clipped to [-1, 1]. NaN if mean(|IC|) == 0 or fewer than 2 conditions
    have computed (non-NaN) IC.

    Parameters
    ----------
    ic_matrix : pd.DataFrame
        Features x conditions, float IC values.

    Returns
    -------
    pd.Series
        Per-feature stability score.
    """
    scores = {}
    for feat in ic_matrix.index:
        row = ic_matrix.loc[feat].dropna()
        if len(row) < 2:
            scores[feat] = np.nan
            continue

        mean_abs = row.abs().mean()
        if mean_abs == 0:
            scores[feat] = np.nan
            continue

        score = 1.0 - row.std() / mean_abs
        scores[feat] = float(np.clip(score, -1.0, 1.0))

    return pd.Series(scores, name="stability_scores")


def compute_stability_threshold(stability_scores: pd.Series) -> float:
    """
    Adaptive stability threshold via Jenks natural breaks (k=2).

    Finds the value that maximises between-group variance in the stability
    score distribution.

    Fallback:
    - Clamped to [0.2, 0.8] if the break is extreme.
    - Falls back to 0.5 if fewer than 10 features have non-NaN scores.

    Parameters
    ----------
    stability_scores : pd.Series
        Per-feature stability scores.

    Returns
    -------
    float
        The adaptive stability threshold.
    """
    valid = stability_scores.dropna().values
    if len(valid) < 10:
        return 0.5

    # Jenks natural breaks (k=2) — find the break that maximises
    # between-group variance (equivalent to Otsu's method for k=2)
    sorted_vals = np.sort(valid)
    best_break = 0.5
    best_variance = -1.0

    # Test candidate breaks between consecutive sorted values
    for i in range(1, len(sorted_vals)):
        group1 = sorted_vals[:i]
        group2 = sorted_vals[i:]

        n1, n2 = len(group1), len(group2)
        mean1, mean2 = group1.mean(), group2.mean()
        overall_mean = sorted_vals.mean()

        # Between-group variance
        bgv = (n1 * (mean1 - overall_mean) ** 2 + n2 * (mean2 - overall_mean) ** 2)

        if bgv > best_variance:
            best_variance = bgv
            # Break is midpoint between the two groups' boundary values
            best_break = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0

    # Clamp to [0.2, 0.8]
    threshold = float(np.clip(best_break, 0.2, 0.8))
    return threshold


def compute_subperiod_stability(features_df: pd.DataFrame, target: pd.Series, condition_labels: pd.Series, n_subperiods: int = 3, min_observations: int = 1000) -> pd.DataFrame:
    """
    For each (feature, condition), split into n_subperiods non-overlapping
    time chunks and compute IC within each. Return std of IC across chunks.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Prediction target.
    condition_labels : pd.Series
        Condition labels.
    n_subperiods : int
        Number of non-overlapping time chunks.
    min_observations : int
        Minimum rows per chunk for reliable IC.

    Returns
    -------
    pd.DataFrame
        Features x conditions, std of IC across sub-periods.
        NaN where insufficient data per chunk.
    """
    conditions = sorted(condition_labels.dropna().unique())
    features = features_df.columns.tolist()
    result = np.full((len(features), len(conditions)), np.nan)

    min_per_chunk = max(1, min_observations // n_subperiods)

    for j, cond in enumerate(conditions):
        mask = condition_labels == cond
        indices = features_df.index[mask]

        if len(indices) < min_per_chunk * n_subperiods:
            continue

        # Split into n_subperiods chunks by position (temporal order)
        chunks = np.array_split(indices, n_subperiods)

        for i, feat in enumerate(features):
            chunk_ics = []
            skip_feature = False

            for chunk_idx in chunks:
                feat_chunk = features_df.loc[chunk_idx, feat]
                target_chunk = target.loc[chunk_idx]
                valid = feat_chunk.notna() & target_chunk.notna()
                n_valid = valid.sum()

                if n_valid < min_per_chunk:
                    skip_feature = True
                    break

                x = feat_chunk.loc[valid].values
                y = target_chunk.loc[valid].values

                if np.std(x) == 0 or np.std(y) == 0:
                    skip_feature = True
                    break

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(x, y)

                chunk_ics.append(corr)

            if skip_feature or len(chunk_ics) < n_subperiods:
                continue

            result[i, j] = float(np.std(chunk_ics))

    return pd.DataFrame(result, index=features, columns=conditions)