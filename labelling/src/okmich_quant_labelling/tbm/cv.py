"""Purged + embargoed cross-validation for triple-barrier labels.

Standard purged walk-forward CV per Lopez de Prado Ch. 7 — NOT Combinatorial
Purged CV (CPCV, Ch. 12). CPCV requires k test groups, n paths, and path
recombination logic which is out of scope here.

`embargo_train_labels` is exposed for callers who manually compose K-fold-style
splits where train events can fall AFTER a test fold. In `purged_walk_forward_cv`
itself the train set always precedes the test set, so embargo would be a no-op
and is not applied there.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Generator, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_TRAIN_EVENTS = 30


class FoldSplitMode(str, Enum):
    EVENTS = "events"  # equal event-count splits (default; matches Lopez de Prado)
    TIME = "time"      # equal time-span splits (folds reflect calendar duration)


def purge_train_labels(train_index: pd.DatetimeIndex, test_index: pd.DatetimeIndex,
                       labels: pd.DataFrame) -> pd.DatetimeIndex:
    """Remove train events whose label window (`t1`) overlaps the test window.

    A train event t0 is purged if `labels.loc[t0, 't1'] >= test_index[0]`.
    """
    if len(test_index) == 0 or len(train_index) == 0:
        return train_index
    if "t1" not in labels.columns:
        raise ValueError("labels must contain 't1' column")
    unknown = train_index.difference(labels.index)
    if len(unknown) > 0:
        raise ValueError(
            f"train_index has {len(unknown)} timestamps not in labels.index "
            f"(first: {unknown[0]}). Pass a subset of labels.index."
        )
    test_start = test_index[0]
    t1_for_train = labels.loc[train_index, "t1"]
    keep_mask = t1_for_train < test_start
    return pd.DatetimeIndex(t1_for_train.index[keep_mask])


def embargo_train_labels(train_index: pd.DatetimeIndex, test_end: pd.Timestamp, embargo_bars: int,
                         close: pd.Series) -> pd.DatetimeIndex:
    """Drop train events whose iloc position falls within `embargo_bars` after test_end.

    Useful when train data can come AFTER a test fold (K-fold, CPCV). For pure
    walk-forward CV this is a no-op since train always precedes test.
    """
    if embargo_bars <= 0 or len(train_index) == 0:
        return train_index

    if test_end not in close.index:
        end_iloc = close.index.searchsorted(test_end, side="right") - 1
    else:
        end_iloc = close.index.get_loc(test_end)

    embargo_end_iloc = min(end_iloc + embargo_bars, len(close.index) - 1)
    embargo_end_ts = close.index[embargo_end_iloc]
    return train_index[(train_index <= test_end) | (train_index > embargo_end_ts)]


def purged_walk_forward_cv(labels: pd.DataFrame, close: pd.Series, n_splits: int = 5,
                           split_by: str = FoldSplitMode.EVENTS.value
                           ) -> Generator[Tuple[pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
    """Yield (train_index, test_index) pairs with overlap purging applied.

    Splits `labels.index` into `n_splits` contiguous time-ordered folds. Each
    fold serves as the test set with all earlier folds as the training set
    (walk-forward). Train events whose `t1` extends into the test window are
    purged. Folds with fewer than MIN_TRAIN_EVENTS train events after purging
    are skipped with a warning.

    Embargo is intentionally NOT applied here — train always precedes test.
    Callers needing K-fold-style splits should compose folds manually and use
    `embargo_train_labels` directly.

    Parameters
    ----------
    split_by : "events" | "time"
        - "events" (default): equal event-count slices (~len(labels)/n_splits per fold).
          Folds may reflect very uneven calendar durations under uneven event sampling
          (e.g. CUSUM in changing volatility regimes).
        - "time": equal calendar-duration slices spanning labels.index. Each fold
          covers ~total_duration/n_splits in time. Folds may be heavily uneven
          in event count.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if "t1" not in labels.columns:
        raise ValueError("labels must contain 't1' column")
    if len(labels) < n_splits:
        raise ValueError(f"need at least n_splits={n_splits} labels, got {len(labels)}")
    if len(close.index) == 0:
        raise ValueError("close must be non-empty")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close.index must be monotonic increasing")
    try:
        mode = FoldSplitMode(split_by)
    except ValueError:
        valid = [m.value for m in FoldSplitMode]
        raise ValueError(f"split_by must be one of {valid}, got {split_by!r}")

    sorted_labels = labels.sort_index()
    sorted_index = sorted_labels.index
    n = len(sorted_labels)

    if mode == FoldSplitMode.EVENTS:
        fold_positions = np.array_split(np.arange(n), n_splits)
    else:
        # Time-based: split [first, last] timestamp into n equal sub-intervals.
        first_ts = sorted_index[0]
        last_ts = sorted_index[-1]
        edges = pd.date_range(start=first_ts, end=last_ts, periods=n_splits + 1)
        fold_positions = []
        for i in range(n_splits):
            lo = edges[i]
            hi = edges[i + 1]
            if i == n_splits - 1:
                mask = (sorted_index >= lo) & (sorted_index <= hi)
            else:
                mask = (sorted_index >= lo) & (sorted_index < hi)
            fold_positions.append(np.where(mask)[0])

    for fold_i in range(1, n_splits):
        test_pos = fold_positions[fold_i]
        train_pos = np.concatenate(fold_positions[:fold_i]) if fold_positions[:fold_i] else np.empty(0, dtype=int)
        if len(test_pos) == 0 or len(train_pos) == 0:
            logger.warning("fold %d has empty train or test (%d / %d); skipping",
                           fold_i, len(train_pos), len(test_pos))
            continue
        test_index = sorted_index[test_pos]
        train_index = sorted_index[train_pos]

        purged = purge_train_labels(train_index, test_index, sorted_labels)

        if len(purged) < MIN_TRAIN_EVENTS:
            logger.warning("fold %d has %d train events after purging (min=%d); skipping",
                           fold_i, len(purged), MIN_TRAIN_EVENTS)
            continue

        yield purged, test_index
