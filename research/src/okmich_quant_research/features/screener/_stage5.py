"""
Stage 5 — Model-Based Ranking (SHAP + Permutation Importance)
==============================================================
Final ranking of confirmed features by their actual marginal contribution inside a predictive model. This is a ranking step, not a filter.

Two methods are used for robustness:
  - SHAP (TreeExplainer on XGBoost): directional, interaction-aware
  - Permutation Importance (MDA): model-agnostic, collinearity-robust

For return prediction with overlapping labels, a simple PurgedKFold is used to prevent train/test contamination.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier, XGBRegressor


# ── Purged K-Fold ─────────────────────────────────────────────────────────────

class _PurgedKFold:
    """
    Simple purged K-Fold for time-series with overlapping labels.

    For each fold, removes samples within ``horizon`` bars of the test boundary
    from the training set (purge) plus an optional embargo gap after the test set.

    Parameters
    ----------
    n_splits : int
    horizon : int
        Number of forward bars used to compute the label. Determines purge width.
    embargo_pct : float
        Fraction of total samples to embargo after the test set. Default 0.01.
    """

    def __init__(self, n_splits: int = 5, horizon: int = 1, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // self.n_splits
        embargo_bars = max(1, int(n * self.embargo_pct))

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = test_start + fold_size if k < self.n_splits - 1 else n

            # Purge: remove training samples within horizon of the test boundary
            purge_start = max(0, test_start - self.horizon)
            embargo_end = min(n, test_end + embargo_bars)

            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(embargo_end, n),
            ])
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


# ── SHAP ranking ──────────────────────────────────────────────────────────────

def _shap_ranking(X: pd.DataFrame, y: np.ndarray, task: str, cv) -> pd.Series:
    """
    Compute mean |SHAP| values averaged across CV folds.
    Uses XGBoost if available, falls back to RandomForest.
    """
    shap_accum = np.zeros(X.shape[1])
    n_folds = 0

    for train_idx, val_idx in cv.split(X, y if task == "regime" else None):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]

        X_tr_filled = X_tr.fillna(X_tr.median())
        X_val_filled = X_val.fillna(X_tr.median())  # use train median

        if task == "regime":
            n_classes = len(np.unique(y_tr))
            obj = "multi:softprob" if n_classes > 2 else "binary:logistic"
            model = XGBClassifier(objective=obj, n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  eval_metric="logloss", random_state=42, verbosity=0)
        else:
            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42,
                                 verbosity=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_filled.values, y_tr)

        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_val_filled.values)
            # For multiclass, sv is a list — take mean across classes
            if isinstance(sv, list):
                sv = np.mean([np.abs(s) for s in sv], axis=0)
            shap_accum += np.abs(sv).mean(axis=0)
            n_folds += 1
        except Exception:
            # Fall back to feature importance
            if hasattr(model, "feature_importances_"):
                shap_accum += model.feature_importances_
                n_folds += 1

    if n_folds == 0:
        return pd.Series(0.0, index=X.columns)

    return pd.Series(shap_accum / n_folds, index=X.columns).sort_values(ascending=False)


# ── Permutation importance (MDA) ─────────────────────────────────────────────

def _mda_ranking(X: pd.DataFrame, y: np.ndarray, task: str, cv) -> pd.Series:
    """
    Compute mean permutation importance averaged across CV folds.
    """
    mda_accum = np.zeros(X.shape[1])
    n_folds = 0

    for train_idx, val_idx in cv.split(X, y if task == "regime" else None):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        X_tr_filled = X_tr.fillna(X_tr.median())
        X_val_filled = X_val.fillna(X_tr.median())

        if task == "regime":
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr_filled.values, y_tr)
            result = permutation_importance(model, X_val_filled.values, y_val, n_repeats=10, random_state=42, n_jobs=-1)

        mda_accum += result.importances_mean
        n_folds += 1

    if n_folds == 0:
        return pd.Series(0.0, index=X.columns)

    return pd.Series(mda_accum / n_folds, index=X.columns).sort_values(ascending=False)


# ── Public entry point ────────────────────────────────────────────────────────

def stage5_model_ranking(X: pd.DataFrame, y: pd.Series, task: str = "regime", n_splits: int = 5, horizon: int = 1,
                         embargo_pct: float = 0.01, verbose: bool = True) -> tuple[pd.Series, pd.Series]:
    """
    Rank confirmed features by SHAP and permutation importance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (confirmed + tentative features from Stage 4).
    y : pd.Series
        Target.
    task : str
        ``"regime"`` uses StratifiedKFold; ``"return"`` uses PurgedKFold.
    n_splits : int
        Number of CV folds.
    horizon : int
        Forward label horizon (bars). Used for purge width in return task.
    embargo_pct : float
        Embargo fraction of total samples after the test fold.
    verbose : bool

    Returns
    -------
    shap_rank : pd.Series
        Mean |SHAP| importance, descending.
    mda_rank : pd.Series
        Mean permutation importance, descending.
    """
    # Align index and drop trailing NaN from forward returns
    common = X.index.intersection(y.index)
    X_aligned = X.loc[common]
    y_aligned = y.loc[common]

    mask = y_aligned.notna()
    X_clean = X_aligned[mask]
    y_clean = y_aligned[mask]

    if task == "regime":
        le = LabelEncoder()
        y_enc = le.fit_transform(y_clean)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    else:
        y_enc = y_clean.values.astype(float)
        cv = _PurgedKFold(n_splits=n_splits, horizon=horizon, embargo_pct=embargo_pct)

    if verbose:
        method_name = "StratifiedKFold" if task == "regime" else f"PurgedKFold(horizon={horizon})"
        print(f"  Stage 5 SHAP + MDA ranking ({method_name}, {n_splits} folds, 'XGBoost')...")

    shap_rank = _shap_ranking(X_clean, y_enc, task, cv)
    mda_rank  = _mda_ranking(X_clean, y_enc, task, cv)

    if verbose:
        top5 = shap_rank.head(5)
        print(f"    Top 5 by SHAP: {list(zip(top5.index, top5.round(4).values))}")

    return shap_rank, mda_rank
