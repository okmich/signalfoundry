"""
Stage 0 — Near-Zero Variance Filter
====================================
Removes features that are effectively constant and carry no information.
Must run first because constant features crash MI estimation and bias clustering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._result import StageReport


def stage0_variance_filter(X: pd.DataFrame, cv_threshold: float = 1e-4, const_pct_threshold: float = 0.95,
                           verbose: bool = True) -> tuple[pd.DataFrame, StageReport]:
    """
    Remove near-constant and constant-dominated features.

    A feature is removed if EITHER:
      - Its coefficient of variation (std / |mean|) is below ``cv_threshold``, OR
      - More than ``const_pct_threshold`` fraction of non-NaN values are identical.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = bars, cols = features). May contain NaN.
    cv_threshold : float
        Minimum acceptable CV. Default 1e-4.
    const_pct_threshold : float
        Maximum fraction of bars sharing the modal value. Default 0.95.
    verbose : bool
        Print removed features.

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    """
    n_before = X.shape[1]
    kept, removed = [], []

    for col in X.columns:
        vals = X[col].dropna()
        if len(vals) < 2:
            removed.append(col)
            continue

        # Check constant-dominant fraction
        mode_frac = (vals == vals.mode().iloc[0]).mean()
        if mode_frac >= const_pct_threshold:
            removed.append(col)
            continue

        # Check coefficient of variation
        abs_mean = vals.abs().mean()
        if abs_mean < 1e-12:
            # Near-zero mean: fall back to pure std check
            if vals.std() < cv_threshold:
                removed.append(col)
                continue
        else:
            cv = vals.std() / abs_mean
            if cv < cv_threshold:
                removed.append(col)
                continue
        kept.append(col)

    if verbose and removed:
        print(f"  Stage 0 removed {len(removed)} near-constant features: {removed}")

    report = StageReport(stage="Stage0_Variance", n_before=n_before, n_after=len(kept), removed=removed)
    return X[kept], report