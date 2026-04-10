"""
Stage 1 (Return) — Hygiene Filter: MI + Distance Correlation
=============================================================
For return prediction, two complementary tests are used:
  - Mutual Information (regression): any nonlinear dependence on forward returns
  - Distance Correlation (dcor): detects all forms of statistical dependence,
    superior to Pearson for nonlinear relationships and continuous targets

A feature passes if it clears at least ``min_passes`` out of 2 tests.

Percentile mode (mi_pct / dcor_pct)
------------------------------------
When a percentile parameter is provided, the filter uses **rank-based selection**
rather than a fixed threshold: the top ``(1 - pct)`` fraction of features by score
are marked as passing.  This is strictly more robust than converting the percentile
to a scalar threshold because it avoids floating-point tie-breaking issues and
guarantees a minimum survival count regardless of score distributions.
"""
from __future__ import annotations

import math
import warnings
from typing import List

import dcor
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from ._result import StageReport


def _compute_mi_regression(X: pd.DataFrame, y: pd.Series, n_neighbors: int = 5) -> pd.Series:
    """Compute mutual information between each feature and continuous target."""
    common = X.index.intersection(y.index)
    X_aligned = X.loc[common].fillna(X.loc[common].median())
    y_aligned = y.loc[common]

    # Drop rows where y is NaN (trailing NaN from shift / warmup)
    mask = y_aligned.notna()
    X_clean = X_aligned[mask]
    y_clean = y_aligned[mask]

    if len(y_clean) < 10:
        return pd.Series(0.0, index=X.columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_regression(X_clean.values, y_clean.values, n_neighbors=n_neighbors, random_state=42)
    return pd.Series(mi, index=X.columns)


def _compute_dcor(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute unbiased distance correlation between each feature and the target.
    dcor ∈ [0, 1]; 0 = independent, 1 = perfectly dependent.
    """
    common = X.index.intersection(y.index)
    y_aligned = y.loc[common]
    mask = y_aligned.notna()
    y_clean = y_aligned[mask].values

    dcor_scores = {}
    for col in X.columns:
        vals = X.loc[common, col][mask]
        valid = vals.notna()
        if valid.sum() < 10:
            dcor_scores[col] = 0.0
            continue
        x_clean = vals[valid].values
        y_sub = y_clean[valid.values]
        try:
            score = dcor.u_distance_correlation_sqr(x_clean, y_sub)
            v = float(score)
            dcor_scores[col] = 0.0 if (math.isnan(v) or math.isinf(v)) else max(0.0, v)
        except Exception:
            dcor_scores[col] = 0.0

    return pd.Series(dcor_scores)


def stage1_return(X: pd.DataFrame, y: pd.Series, mi_threshold: float = 0.02, dcor_threshold: float = 0.05,
                  mi_pct: float | None = None, dcor_pct: float | None = None,
                  min_passes: int = 2, verbose: bool = True) -> tuple[pd.DataFrame, StageReport, dict]:
    """
    Hygiene filter for return prediction.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = bars).
    y : pd.Series
        Forward log-returns (e.g. ``log(close.shift(-H) / close)``).
        NaN at the tail is handled automatically.
    mi_threshold : float
        Minimum MI score to pass the MI test. Ignored when ``mi_pct`` is set.
    dcor_threshold : float
        Minimum distance correlation to pass the dcor test. Ignored when ``dcor_pct`` is set.
    mi_pct : float or None
        If set (0–1), use **rank-based** selection: keep the top ``(1 - mi_pct)``
        fraction of features by MI score.  E.g. ``mi_pct=0.25`` keeps the top-75%.
        Always guarantees at least one feature survives.
    dcor_pct : float or None
        Same as ``mi_pct`` but for dcor scores.
    min_passes : int
        Minimum tests to pass (max 2).  Default 2 (AND).  Use 1 for OR logic.
    verbose : bool

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    scores : dict
        ``{"mi": mi_series, "dcor": dcor_series}`` for inspection.
    """
    n_before = X.shape[1]

    mi_scores = _compute_mi_regression(X, y)
    dcor_scores = _compute_dcor(X, y)

    if verbose:
        print(f"    MI   scores — min:{mi_scores.min():.5f}  med:{mi_scores.median():.5f}  max:{mi_scores.max():.5f}")
        print(f"    dcor scores — min:{dcor_scores.min():.5f}  med:{dcor_scores.median():.5f}  max:{dcor_scores.max():.5f}")

    # ── Determine which features pass each test ──────────────────────────────
    if mi_pct is not None:
        # Rank-based: keep top (1 - mi_pct) fraction; always at least 1
        n_keep = max(1, round(n_before * (1.0 - mi_pct)))
        mi_pass = set(mi_scores.nlargest(n_keep).index)
        mi_label = f"MI top-{100*(1-mi_pct):.0f}%({n_keep}/{n_before})"
    else:
        mi_pass = set(mi_scores.index[mi_scores.values >= mi_threshold])
        mi_label = f"MI>{mi_threshold:.4f}(abs)"

    if dcor_pct is not None:
        n_keep = max(1, round(n_before * (1.0 - dcor_pct)))
        dcor_pass = set(dcor_scores.nlargest(n_keep).index)
        dcor_label = f"dcor top-{100*(1-dcor_pct):.0f}%({n_keep}/{n_before})"
    else:
        dcor_pass = set(dcor_scores.index[dcor_scores.values >= dcor_threshold])
        dcor_label = f"dcor>{dcor_threshold:.4f}(abs)"

    kept, removed = [], []
    for col in X.columns:
        passes = int(col in mi_pass) + int(col in dcor_pass)
        if passes >= min_passes:
            kept.append(col)
        else:
            removed.append(col)

    if verbose:
        print(f"  Stage 1 (return) {mi_label} | {dcor_label} | min_passes={min_passes}: "
              f"{n_before} -> {len(kept)} features ({len(removed)} removed)")

    report = StageReport(stage="Stage1_Hygiene_Return", n_before=n_before, n_after=len(kept), removed=removed)
    scores = {"mi": mi_scores, "dcor": dcor_scores}
    return X[kept], report, scores