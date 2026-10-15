"""
Stage 5 — Model-Based Ranking (SHAP + Permutation Importance)
==============================================================
Final ranking of confirmed features by their actual marginal contribution inside a predictive model. This is a ranking step, not a filter.

Two methods are used for robustness:
  - SHAP (TreeExplainer on XGBoost): directional, interaction-aware
  - Permutation Importance (MDA): model-agnostic, collinearity-robust

Both tasks use the same purged, embargoed contiguous K-fold. Return labels overlap their neighbours by ``horizon``
bars; regime labels are persistent, so a shuffled or interleaved split (the regime task once used an unpurged
StratifiedKFold) puts near-identical neighbouring bars on both sides of a fold edge. Training folds that lack a
regime class are skipped with a warning (XGBoost needs every class in training). SHAP never falls back silently:
a fold where it fails uses XGBoost's own feature importances, with a warning, and the per-fold method is recorded
in the detail dict (``return_detail=True``).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier, XGBRegressor


# ── Purged K-Fold ─────────────────────────────────────────────────────────────

class _PurgedKFold:
    """
    Simple purged K-Fold for time-series with overlapping labels.

    For each fold, removes samples within ``horizon`` bars of the test boundary
    from the training set (purge) plus an embargo gap after the test set.

    The embargo is at least ``horizon`` bars: the labels of the last test bars read up to ``horizon`` bars past the
    test fold, so a shorter embargo would let them see training data.

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

    def embargo_bars(self, n: int) -> int:
        return max(self.horizon, int(n * self.embargo_pct), 1)

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // self.n_splits
        embargo_bars = self.embargo_bars(n)

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


def _usable_folds(cv: _PurgedKFold, X: pd.DataFrame, y: np.ndarray, task: str) -> tuple[list, int]:
    """Materialise the folds once; on the regime task drop folds whose training side lacks a class."""
    folds = list(cv.split(X))
    if task != "regime":
        return folds, 0
    n_classes = len(np.unique(y))
    usable = [(tr, te) for tr, te in folds if len(np.unique(y[tr])) == n_classes]
    n_skipped = len(folds) - len(usable)
    if n_skipped:
        warnings.warn(f"Stage 5: skipped {n_skipped} of {len(folds)} folds whose training side lacks a regime class "
                      f"(contiguous folds on persistent labels); ranking uses the remaining {len(usable)}.",
                      stacklevel=3)
    return usable, n_skipped


# ── SHAP ranking ──────────────────────────────────────────────────────────────

def _mean_abs_shap(sv, n_features: int) -> np.ndarray:
    """Mean |SHAP| per feature from any shape shap returns: a per-class list (older shap), (n, f) for regression /
    binary, or (n, f, K) for multiclass (shap >= 0.45)."""
    if isinstance(sv, list):
        return np.mean([np.abs(np.asarray(s)).mean(axis=0) for s in sv], axis=0)
    sv = np.abs(np.asarray(sv))
    if sv.ndim == 2 and sv.shape[1] == n_features:
        return sv.mean(axis=0)
    if sv.ndim == 3 and sv.shape[1] == n_features:
        return sv.mean(axis=(0, 2))
    raise ValueError(f"unexpected SHAP value shape {sv.shape} for {n_features} features")


def _shap_ranking(X: pd.DataFrame, y: np.ndarray, task: str, folds: list) -> tuple[pd.Series, int]:
    """
    Compute mean |SHAP| values averaged across CV folds (XGBoost).

    Returns the ranking and the number of folds where SHAP failed and XGBoost's feature importances were used
    instead (each one is also warned about).
    """
    shap_accum = np.zeros(X.shape[1])
    n_folds = 0
    n_fallback = 0
    n_classes = len(np.unique(y)) if task == "regime" else 0

    for train_idx, val_idx in folds:
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]

        X_tr_filled = X_tr.fillna(X_tr.median())
        X_val_filled = X_val.fillna(X_tr.median())  # use train median

        if task == "regime":
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
            shap_accum += _mean_abs_shap(sv, X.shape[1])
            n_folds += 1
        except Exception as ex:
            warnings.warn(f"Stage 5: SHAP failed on a fold ({type(ex).__name__}: {ex}); using XGBoost feature "
                          f"importances for that fold, which are on a different scale.", stacklevel=2)
            shap_accum += model.feature_importances_
            n_folds += 1
            n_fallback += 1

    if n_folds == 0:
        return pd.Series(0.0, index=X.columns), n_fallback

    return pd.Series(shap_accum / n_folds, index=X.columns).sort_values(ascending=False), n_fallback


# ── Permutation importance (MDA) ─────────────────────────────────────────────

def _mda_ranking(X: pd.DataFrame, y: np.ndarray, task: str, folds: list) -> pd.Series:
    """
    Compute mean permutation importance averaged across CV folds.
    """
    mda_accum = np.zeros(X.shape[1])
    n_folds = 0

    for train_idx, val_idx in folds:
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
                         embargo_pct: float = 0.01, verbose: bool = True, return_detail: bool = False):
    """
    Rank confirmed features by SHAP and permutation importance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (confirmed + tentative features from Stage 4).
    y : pd.Series
        Target.
    task : str
        ``"regime"`` or ``"return"``. Both use the purged, embargoed contiguous K-fold.
    n_splits : int
        Number of CV folds.
    horizon : int
        Forward label horizon (bars): the purge width before each test fold, and the floor of the embargo after
        it. For the regime task, the look-ahead of the labeler (0 for a causal label).
    embargo_pct : float
        Embargo fraction of total samples after the test fold (at least ``horizon`` bars, at least 1).
    verbose : bool
    return_detail : bool
        If True, also return a dict describing what ran: ``cv``, ``n_splits``, ``horizon``, ``embargo_bars``,
        ``n_folds_used``, ``skipped_folds`` (regime folds whose training side lacked a class), ``shap_method``
        (``"tree_shap"``, ``"mixed"`` or ``"xgb_feature_importances"``) and ``shap_fallback_folds``.

    Returns
    -------
    shap_rank : pd.Series
        Mean |SHAP| importance, descending.
    mda_rank : pd.Series
        Mean permutation importance, descending.
    detail : dict
        Only when ``return_detail=True``.
    """
    # Align index and drop trailing NaN from forward returns
    common = X.index.intersection(y.index)
    X_aligned = X.loc[common]
    y_aligned = y.loc[common]

    mask = y_aligned.notna()
    X_clean = X_aligned[mask]
    y_clean = y_aligned[mask]

    if task == "regime":
        y_enc = LabelEncoder().fit_transform(y_clean)
    else:
        y_enc = y_clean.values.astype(float)
    cv = _PurgedKFold(n_splits=n_splits, horizon=horizon, embargo_pct=embargo_pct)
    embargo_bars = cv.embargo_bars(len(X_clean))

    if verbose:
        print(f"  Stage 5 SHAP + MDA ranking (PurgedKFold(horizon={horizon}, embargo={embargo_bars}), "
              f"{n_splits} folds, 'XGBoost')...")

    folds, n_skipped = _usable_folds(cv, X_clean, y_enc, task)
    shap_rank, n_fallback = _shap_ranking(X_clean, y_enc, task, folds)
    mda_rank = _mda_ranking(X_clean, y_enc, task, folds)

    if verbose:
        top5 = shap_rank.head(5)
        print(f"    Top 5 by SHAP: {list(zip(top5.index, top5.round(4).values))}")

    if not return_detail:
        return shap_rank, mda_rank
    if n_fallback == 0:
        shap_method = "tree_shap"
    else:
        shap_method = "xgb_feature_importances" if n_fallback == len(folds) else "mixed"
    detail = {"cv": "PurgedKFold", "n_splits": n_splits, "horizon": horizon, "embargo_bars": embargo_bars,
              "n_folds_used": len(folds), "skipped_folds": n_skipped, "shap_method": shap_method,
              "shap_fallback_folds": n_fallback}
    return shap_rank, mda_rank, detail
