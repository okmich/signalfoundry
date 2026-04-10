"""IC computation, permutation-based threshold, and diagnostic checks."""

from __future__ import annotations

import logging
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_conditional_ic(features_df: pd.DataFrame, target: pd.Series, condition_labels: pd.Series, min_observations: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman IC and p-value for each (feature, condition) pair.

    NaN handling is per-feature: for each (feature, condition), drop rows
    where that feature or the target is NaN before computing correlation.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix (rows = bars, cols = features).
    target : pd.Series
        Prediction target, aligned index with features_df.
    condition_labels : pd.Series
        Discrete condition labels, aligned index with features_df.
    min_observations : int
        Minimum rows per (feature, condition) after NaN removal.

    Returns
    -------
    ic_matrix : pd.DataFrame
        Features x conditions, float IC values. NaN where insufficient data.
    pvalue_matrix : pd.DataFrame
        Features x conditions, float p-values from spearmanr.
    obs_matrix : pd.DataFrame
        Features x conditions, int observation counts after NaN removal.
    """
    conditions = sorted(condition_labels.dropna().unique())
    features = features_df.columns.tolist()

    ic_vals = np.full((len(features), len(conditions)), np.nan)
    pv_vals = np.full((len(features), len(conditions)), np.nan)
    ob_vals = np.zeros((len(features), len(conditions)), dtype=int)

    for j, cond in enumerate(conditions):
        mask = condition_labels == cond
        target_cond = target.loc[mask]

        for i, feat in enumerate(features):
            feat_cond = features_df.loc[mask, feat]
            # Per-feature NaN removal
            valid = feat_cond.notna() & target_cond.notna()
            n_valid = valid.sum()
            ob_vals[i, j] = n_valid

            if n_valid < min_observations:
                continue

            x = feat_cond.loc[valid].values
            y = target_cond.loc[valid].values

            # Skip if constant
            if np.std(x) == 0 or np.std(y) == 0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, pval = spearmanr(x, y)

            ic_vals[i, j] = corr
            pv_vals[i, j] = pval

    ic_matrix = pd.DataFrame(ic_vals, index=features, columns=conditions)
    pvalue_matrix = pd.DataFrame(pv_vals, index=features, columns=conditions)
    obs_matrix = pd.DataFrame(ob_vals, index=features, columns=conditions)

    return ic_matrix, pvalue_matrix, obs_matrix


def compute_global_ic(features_df: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Unconditional Spearman IC per feature (full dataset)."""
    results = {}
    for feat in features_df.columns:
        valid = features_df[feat].notna() & target.notna()
        x = features_df.loc[valid, feat].values
        y = target.loc[valid].values

        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            results[feat] = np.nan
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(x, y)

        results[feat] = corr

    return pd.Series(results, name="global_ic")


def compute_ic_threshold(features_df: pd.DataFrame, target: pd.Series, condition_labels: pd.Series, n_permutations: int = 100, ic_percentile: float = 95.0, min_observations: int = 1000, feature_subsample_ratio: float = 0.3) -> float:
    """
    Permutation-based adaptive IC threshold.

    Shuffles the target n_permutations times (preserving condition label
    alignment), computes |IC| for a random subsample of features across
    all conditions per shuffle, and returns the ic_percentile-th percentile
    of the pooled |null IC| distribution.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix.
    target : pd.Series
        Prediction target.
    condition_labels : pd.Series
        Condition labels.
    n_permutations : int
        Number of target shuffles.
    ic_percentile : float
        Percentile of null |IC| for the threshold.
    min_observations : int
        Minimum rows per (feature, condition).
    feature_subsample_ratio : float
        Fraction of features to subsample per permutation.

    Returns
    -------
    float
        The adaptive IC threshold (noise floor).
    """
    rng = np.random.default_rng(42)
    all_features = features_df.columns.tolist()
    n_subsample = max(1, int(len(all_features) * feature_subsample_ratio))

    conditions = sorted(condition_labels.dropna().unique())
    null_ics: list[float] = []

    for _ in range(n_permutations):
        # Shuffle target, preserving index alignment with condition labels
        shuffled_target = target.sample(frac=1.0, random_state=rng.integers(0, 2**31)).values
        shuffled_target = pd.Series(shuffled_target, index=target.index, name=target.name)

        # Subsample features
        sampled_features = rng.choice(all_features, size=n_subsample, replace=False).tolist()

        for cond in conditions:
            mask = condition_labels == cond
            target_cond = shuffled_target.loc[mask]

            for feat in sampled_features:
                feat_cond = features_df.loc[mask, feat]
                valid = feat_cond.notna() & target_cond.notna()
                n_valid = valid.sum()

                if n_valid < min_observations:
                    continue

                x = feat_cond.loc[valid].values
                y = target_cond.loc[valid].values

                if np.std(x) == 0 or np.std(y) == 0:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(x, y)

                if not np.isnan(corr):
                    null_ics.append(abs(corr))

    if len(null_ics) == 0:
        logger.warning("No null IC values computed — falling back to ic_threshold=0.03")
        return 0.03

    threshold = float(np.percentile(null_ics, ic_percentile))
    return threshold


def check_ic_threshold_diagnostic(ic_threshold: float, ic_matrix: pd.DataFrame) -> None:
    """
    Log a warning if ic_threshold exceeds the median of the real |IC| distribution.

    Called by the analyzer after both ic_matrix and ic_threshold are available,
    to avoid recomputing the IC matrix.
    """
    real_abs_ic = ic_matrix.abs().values.flatten()
    real_abs_ic = real_abs_ic[~np.isnan(real_abs_ic)]

    if len(real_abs_ic) > 0 and ic_threshold > np.median(real_abs_ic):
        logger.warning(
            "Adaptive ic_threshold (%.4f) exceeds median |real IC| (%.4f). "
            "The feature set may lack signal entirely, not just conditional structure.",
            ic_threshold, np.median(real_abs_ic),
        )