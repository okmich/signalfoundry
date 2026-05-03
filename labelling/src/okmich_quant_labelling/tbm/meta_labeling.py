"""Meta-labelling: secondary classifier that decides whether to act on a primary signal.

The meta-model answers: given that the primary model fired direction X,
should we actually place the bet? Output probability is used as a position size
scalar at inference (`predict_proba(X)[:, 1]`).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

logger = logging.getLogger(__name__)

DEFAULT_N_ESTIMATORS = 100

# Splitters that DO NOT respect time order — leak across folds for financial
# labels. Allowed only when caller passes allow_unsafe_splitters=True.
_UNSAFE_SPLITTERS = (
    KFold, StratifiedKFold,
    RepeatedKFold, RepeatedStratifiedKFold,
    ShuffleSplit, StratifiedShuffleSplit,
    GroupKFold,
)


def get_meta_labels(triple_barrier_labels: pd.DataFrame, primary_predictions: pd.Series) -> pd.Series:
    """Build binary meta-labels: 1 if primary was correct, 0 otherwise.

    Vertical-expiry events (tb_label == 0) become meta-label 0 unconditionally —
    no clean win, do not bet. Primary direction of 0 (no signal) also yields 0.
    """
    if "label" not in triple_barrier_labels.columns:
        raise ValueError("triple_barrier_labels must contain 'label' column")
    if not triple_barrier_labels.index.is_unique:
        raise ValueError("triple_barrier_labels.index must be unique")
    if not primary_predictions.index.is_unique:
        raise ValueError("primary_predictions.index must be unique")

    missing = primary_predictions.index.difference(triple_barrier_labels.index)
    if len(missing) > 0:
        raise KeyError(f"{len(missing)} primary_predictions index entries are absent from triple_barrier_labels")

    tb_label = triple_barrier_labels.loc[primary_predictions.index, "label"]
    primary_sign = np.sign(primary_predictions.to_numpy())
    tb_sign = np.sign(tb_label.to_numpy())
    meta = np.where((tb_sign != 0) & (primary_sign == tb_sign), 1, 0)
    return pd.Series(meta, index=primary_predictions.index, name="meta_label", dtype=int)


def build_meta_features(events: pd.DatetimeIndex, close: pd.Series, volatility: pd.Series,
                        primary_scores: pd.Series, additional_features: Optional[pd.DataFrame] = None,
                        vol_ratio_window: int = 20) -> pd.DataFrame:
    """Construct minimum feature set for the meta-model.

    `vol_ratio` denominator uses `.shift(1)` so the rolling-mean comparison
    relies strictly on information available before t0.

    `vol_ratio_window` is in the same cadence as `volatility` — caller must
    set explicitly to match data.
    """
    if vol_ratio_window < 2:
        raise ValueError(f"vol_ratio_window must be >= 2, got {vol_ratio_window}")

    if len(events) == 0:
        return pd.DataFrame(columns=["primary_score", "vol_at_signal", "vol_ratio", "ret_1", "ret_5", "ret_20"])

    log_close = np.log(close)
    ret_1 = log_close.diff(1)
    ret_5 = log_close.diff(5)
    ret_20 = log_close.diff(20)
    vol_ratio = volatility / volatility.rolling(vol_ratio_window, min_periods=vol_ratio_window).mean().shift(1)

    df = pd.DataFrame(index=events)
    df["primary_score"] = primary_scores.reindex(events)
    df["vol_at_signal"] = volatility.reindex(events)
    df["vol_ratio"] = vol_ratio.reindex(events)
    df["ret_1"] = ret_1.reindex(events)
    df["ret_5"] = ret_5.reindex(events)
    df["ret_20"] = ret_20.reindex(events)

    if additional_features is not None:
        before = len(df)
        df = df.join(additional_features, how="inner")
        dropped_pct = (before - len(df)) / before if before else 0.0
        if dropped_pct > 0.05:
            logger.warning("additional_features inner join dropped %.1f%% of events", dropped_pct * 100)

    nan_mask = df.isna().any(axis=1)
    if nan_mask.any():
        logger.warning("dropping %d events with NaN features", int(nan_mask.sum()))
        df = df.loc[~nan_mask]
    return df


def train_meta_model(features: pd.DataFrame, meta_labels: pd.Series, cv_splitter,
                     model=None, sample_weight: Optional[pd.Series] = None,
                     allow_unsafe_splitters: bool = False,
                     strict_timestamp_resolution: bool = True) -> Tuple[object, Dict]:
    """Train meta-model with cross-validated metrics, then refit on full data.

    The passed `model` is NOT mutated — both per-fold models and the final
    refit model are clones. The returned model is a fresh clone fit on the
    full (features, meta_labels).

    Splitter safety
    ---------------
    By default, partition-style sklearn splitters (KFold, StratifiedKFold,
    ShuffleSplit, etc.) are REJECTED — they leak across time for financial
    labels. Allowed:
      - iterables of (train, test) pairs (e.g., from `purged_walk_forward_cv`)
      - `TimeSeriesSplit` (sklearn time-ordered splitter)
    Set `allow_unsafe_splitters=True` to override (e.g., for non-temporal data).

    Strict timestamp resolution
    ---------------------------
    When CV folds are timestamp-indexed and `strict_timestamp_resolution=True`
    (default), any fold timestamp not in `features.index` raises. Set False to
    silently drop with a warning (legacy behaviour).
    """
    if cv_splitter is None:
        raise ValueError("cv_splitter is required (no default — pass a purged splitter)")

    _validate_splitter_safety(cv_splitter, allow_unsafe_splitters)

    aligned = features.join(meta_labels.rename("__y__"), how="inner")
    if aligned.empty:
        raise ValueError("features and meta_labels share no common index")

    feat_index = aligned.index
    X = aligned.drop(columns=["__y__"]).to_numpy()
    y = aligned["__y__"].to_numpy().astype(int)

    if len(np.unique(y)) < 2:
        raise ValueError(f"meta_labels must have at least 2 distinct classes, got {np.unique(y).tolist()}")

    if model is None:
        model = RandomForestClassifier(n_estimators=DEFAULT_N_ESTIMATORS, class_weight="balanced", n_jobs=-1)

    weights = None
    if sample_weight is not None:
        weights = sample_weight.reindex(feat_index).to_numpy()
        n_nan = int(np.isnan(weights).sum())
        if n_nan > 0:
            raise ValueError(
                f"sample_weight has {n_nan} NaN values after reindex to features.index; "
                f"ensure sample_weight covers every feature index entry"
            )
        if not np.isfinite(weights).all():
            raise ValueError("sample_weight contains non-finite values after reindex")

    splits = _materialize_splits(cv_splitter, X, y, feat_index, strict=strict_timestamp_resolution)

    proba_buf = np.full(len(y), np.nan, dtype=float)
    test_mask = np.zeros(len(y), dtype=bool)
    n_folds = 0
    for train_idx, test_idx in splits:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) < 2:
            logger.warning("fold has only one class in train set; skipping")
            continue
        fold_model = clone(model)
        fold_fit = {"sample_weight": weights[train_idx]} if weights is not None else {}
        fold_model.fit(X[train_idx], y[train_idx], **fold_fit)
        proba_buf[test_idx] = fold_model.predict_proba(X[test_idx])[:, 1]
        test_mask[test_idx] = True
        n_folds += 1

    if n_folds == 0 or not test_mask.any():
        raise ValueError("cv_splitter produced no usable folds")

    y_test = y[test_mask]
    y_proba_test = proba_buf[test_mask]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba_test)) if len(np.unique(y_test)) > 1 else float("nan")
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "roc_auc": roc_auc,
        "brier_score": float(brier_score_loss(y_test, y_proba_test)),
        "n_train": int(len(y)),
        "n_evaluated": int(test_mask.sum()),
        "label_balance_train": float(y.mean()),
        "label_balance_test": float(y_test.mean()),
        "cv_folds": n_folds,
    }

    final_model = clone(model)
    final_fit = {"sample_weight": weights} if weights is not None else {}
    final_model.fit(X, y, **final_fit)
    return final_model, metrics


def _validate_splitter_safety(cv_splitter, allow_unsafe: bool) -> None:
    if allow_unsafe:
        return
    # Only sklearn-style splitter instances are checkable; iterables are assumed safe.
    if hasattr(cv_splitter, "split") and isinstance(cv_splitter, _UNSAFE_SPLITTERS):
        cls = type(cv_splitter).__name__
        raise ValueError(
            f"{cls} is a partition-style splitter that does not respect time order; "
            f"using it on financial labels leaks across folds. Pass an explicit "
            f"purged walk-forward splitter or set allow_unsafe_splitters=True."
        )


def _materialize_splits(cv_splitter, X: np.ndarray, y: np.ndarray, feat_index: pd.Index,
                        strict: bool) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Normalize splitter output to integer-position pairs."""
    if hasattr(cv_splitter, "split"):
        return list(cv_splitter.split(X, y))

    label_to_pos = {ts: i for i, ts in enumerate(feat_index)}
    out = []
    for train, test in cv_splitter:
        out.append((_to_positions(train, label_to_pos, "train", strict),
                    _to_positions(test, label_to_pos, "test", strict)))
    return out


def _to_positions(idx: Union[np.ndarray, pd.Index], label_to_pos: Dict, name: str,
                  strict: bool) -> np.ndarray:
    """Convert an index-or-array into integer positions in the feature matrix.

    Detection order:
      1. numpy array of integer dtype -> use as-is
      2. pandas Index of integer dtype -> use values
      3. pandas Index of timestamps -> map via label_to_pos (strict-aware)
      4. fallback: cast to numpy array, repeat dtype check
    """
    if isinstance(idx, np.ndarray) and idx.dtype.kind in ("i", "u"):
        return idx.astype(np.int64)
    if isinstance(idx, pd.Index) and idx.dtype.kind in ("i", "u"):
        return idx.to_numpy().astype(np.int64)
    if isinstance(idx, pd.Index):
        return _resolve_timestamp_positions(idx, label_to_pos, name, strict)
    arr = np.asarray(idx)
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.int64)
    return _resolve_timestamp_positions(arr, label_to_pos, name, strict)


def _resolve_timestamp_positions(idx, label_to_pos: Dict, name: str, strict: bool) -> np.ndarray:
    positions = []
    missing = []
    for t in idx:
        if t in label_to_pos:
            positions.append(label_to_pos[t])
        else:
            missing.append(t)
    if missing:
        if strict:
            raise KeyError(
                f"{name} split: {len(missing)}/{len(idx)} timestamps not in feature index "
                f"(first missing: {missing[0]}). Pass strict_timestamp_resolution=False to drop silently."
            )
        logger.warning("%s split: %d/%d timestamps not in feature index; dropped",
                       name, len(missing), len(idx))
    return np.asarray(positions, dtype=np.int64)
