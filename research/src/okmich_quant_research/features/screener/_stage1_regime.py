"""
Stage 1 (Regime) — Hygiene Filter: MI + KS Test
=================================================
For regime classification, two complementary tests are used:
  - Mutual Information: any nonlinear statistical dependence on the regime label
  - KS Test: whether the feature distribution differs across regime classes

A feature passes if it clears at least ``min_passes`` out of 2 tests.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from ._result import StageReport


def _compute_mi(X: pd.DataFrame, y: pd.Series, n_neighbors: int = 5) -> pd.Series:
    """Compute mutual information between each feature and regime labels."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Fill NaN with column median for MI computation (MI can't handle NaN)
    X_filled = X.fillna(X.median())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(
            X_filled.values, y_enc, discrete_features=False,
            n_neighbors=n_neighbors, random_state=42,
        )
    return pd.Series(mi, index=X.columns)


def _compute_ks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    For each feature, compute the max KS statistic across all pairs of
    regime classes. Returns the max KS stat (higher = more discriminative).
    """
    y_arr = np.asarray(y)
    classes = pd.unique(y_arr)
    ks_stats = {}

    for col in X.columns:
        feat = X[col]
        valid = feat.notna().to_numpy()
        feat_vals = feat.to_numpy()[valid]
        aligned_y = y_arr[valid]

        max_stat = 0.0
        for cls_a, cls_b in combinations(classes, 2):
            group_a = feat_vals[aligned_y == cls_a]
            group_b = feat_vals[aligned_y == cls_b]
            if len(group_a) < 2 or len(group_b) < 2:
                continue
            stat, _ = ks_2samp(group_a, group_b)
            max_stat = max(max_stat, stat)

        ks_stats[col] = max_stat

    return pd.Series(ks_stats)


def stage1_regime(X: pd.DataFrame, y: pd.Series, mi_threshold: float = 0.02, ks_threshold: float = 0.10,
                  mi_pct: float | None = None, ks_pct: float | None = None,
                  min_passes: int = 2, verbose: bool = True) -> tuple[pd.DataFrame, StageReport, dict]:
    """
    Hygiene filter for regime classification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = bars).
    y : pd.Series
        Regime labels (integer or string).
    mi_threshold : float
        Minimum MI score to pass the MI test. Ignored when ``mi_pct`` is set.
    ks_threshold : float
        Minimum max KS statistic to pass the KS test. Ignored when ``ks_pct`` is set.
    mi_pct : float or None
        If set (0–1), use this percentile of the observed MI score distribution as
        the threshold instead of ``mi_threshold``. E.g. 0.30 keeps features above
        the 30th percentile of MI scores.
    ks_pct : float or None
        Same as ``mi_pct`` but for KS scores.
    min_passes : int
        Minimum number of tests a feature must pass (max 2).
    verbose : bool

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    scores : dict
        ``{"mi": mi_series, "ks": ks_series}`` for inspection.
    """
    import numpy as np

    n_before = X.shape[1]

    mi_scores = _compute_mi(X, y)
    ks_scores = _compute_ks(X, y)

    if verbose:
        print(f"    MI  scores — min:{mi_scores.min():.5f}  med:{mi_scores.median():.5f}  max:{mi_scores.max():.5f}")
        print(f"    KS  scores — min:{ks_scores.min():.5f}  med:{ks_scores.median():.5f}  max:{ks_scores.max():.5f}")

    # ── Determine which features pass each test (rank-based when pct is set) ─
    if mi_pct is not None:
        n_keep = max(1, round(n_before * (1.0 - mi_pct)))
        mi_pass = set(mi_scores.nlargest(n_keep).index)
        mi_label = f"MI top-{100*(1-mi_pct):.0f}%({n_keep}/{n_before})"
    else:
        mi_pass = set(mi_scores.index[mi_scores.values >= mi_threshold])
        mi_label = f"MI>{mi_threshold:.4f}(abs)"

    if ks_pct is not None:
        n_keep = max(1, round(n_before * (1.0 - ks_pct)))
        ks_pass = set(ks_scores.nlargest(n_keep).index)
        ks_label = f"KS top-{100*(1-ks_pct):.0f}%({n_keep}/{n_before})"
    else:
        ks_pass = set(ks_scores.index[ks_scores.values >= ks_threshold])
        ks_label = f"KS>{ks_threshold:.4f}(abs)"

    kept, removed = [], []
    for col in X.columns:
        passes = int(col in mi_pass) + int(col in ks_pass)
        if passes >= min_passes:
            kept.append(col)
        else:
            removed.append(col)

    if verbose:
        print(f"  Stage 1 (regime) {mi_label} | {ks_label} | min_passes={min_passes}: "
              f"{n_before} -> {len(kept)} features ({len(removed)} removed)")

    report = StageReport(stage="Stage1_Hygiene_Regime", n_before=n_before, n_after=len(kept), removed=removed)
    scores = {"mi": mi_scores, "ks": ks_scores}
    return X[kept], report, scores
