"""
Stage 4 — Boruta
================
Binary classification of features into confirmed / tentative / rejected.

Boruta extends Random Forest importance by creating "shadow" (shuffled) copies of each feature and testing whether the
real feature outperforms its own shadow consistently across many iterations. Features that consistently outperform all
shadow features are confirmed; those that cannot be decided are tentative.

Uses RandomForest for both tasks (Boruta was designed for tree importances).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from ._result import StageReport


def stage4_boruta(X: pd.DataFrame, y: pd.Series, task: str = "regime", max_iter: int = 100,
                  n_estimators: int | str = "auto", perc: int = 100, alpha: float = 0.05,
                  verbose: bool = True) -> tuple[pd.DataFrame, StageReport, dict[str, list[str]]]:
    """
    Boruta feature selection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no NaN — fill before calling).
    y : pd.Series
        Target — regime labels (task='regime') or returns (task='return').
    task : str
        ``"regime"`` or ``"return"``.
    max_iter : int
        Maximum Boruta iterations. More = more reliable but slower.
    n_estimators : int or "auto"
        Number of trees. "auto" lets Boruta choose.
    perc : int
        Percentile of shadow importance to beat (100 = max shadow). Lower
        values (e.g. 90) are more lenient and confirm more features.
    alpha : float
        Two-tailed p-value threshold for the binomial test. Default 0.05.
    verbose : bool

    Returns
    -------
    X_confirmed : pd.DataFrame
        Confirmed + tentative features (conservative: include tentative
        for further SHAP ranking; reject only explicitly rejected ones).
    report : StageReport
    boruta_groups : dict
        Keys: "confirmed", "tentative", "rejected".
    """
    n_before = X.shape[1]

    # Fill NaN
    X_filled = X.fillna(X.median())

    if task == "regime":
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        estimator = RandomForestClassifier(
            n_jobs=-1, class_weight="balanced", random_state=42,
        )
    else:
        y_enc = y.values.astype(float)
        estimator = RandomForestRegressor(n_jobs=-1, random_state=42)

    selector = BorutaPy(estimator=estimator, n_estimators=n_estimators, max_iter=max_iter, perc=perc,
                        alpha=alpha, random_state=42, verbose=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X_filled.values, y_enc)

    confirmed_mask = selector.support_
    tentative_mask = selector.support_weak_
    rejected_mask  = ~confirmed_mask & ~tentative_mask

    confirmed = X.columns[confirmed_mask].tolist()
    tentative = X.columns[tentative_mask].tolist()
    rejected  = X.columns[rejected_mask].tolist()

    # Pass confirmed + tentative to Stage 5; only drop rejected
    keep = confirmed + tentative

    if verbose:
        print(f"  Stage 4 Boruta (iter={max_iter}, perc={perc}): "
              f"confirmed={len(confirmed)}, tentative={len(tentative)}, "
              f"rejected={len(rejected)}")

    report = StageReport(stage="Stage4_Boruta", n_before=n_before, n_after=len(keep), removed=rejected)
    boruta_groups = {
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected":  rejected,
    }
    return X[keep], report, boruta_groups