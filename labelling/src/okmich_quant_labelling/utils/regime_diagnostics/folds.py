"""Walk-forward folds: calendar test folds after ``first_test_start``, each with an expanding or rolling train
window."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    number: int
    test_start: int  # first test bar (position)
    test_end: int  # one past the last test bar
    train_start: int  # first train bar; train runs up to test_start, before the per-horizon purge

    def train_stop(self, horizon: int) -> int:
        """One past the last train decision bar whose forward window ends before the test fold (the purge)."""
        return max(self.train_start, self.test_start - horizon)


def walk_forward_folds(index: pd.DatetimeIndex, first_test_start, step: str = "QS", train: str = "expanding",
                       min_train: str = "365D") -> list[Fold]:
    """Test folds start at ``first_test_start`` and then at every ``step`` boundary; train precedes each fold.

    ``train`` is ``"expanding"`` (all earlier bars) or a rolling length such as ``"730D"``. Folds whose train window
    spans less than ``min_train`` are skipped.
    """
    first = pd.Timestamp(first_test_start)
    first = first.tz_localize(index.tz) if first.tz is None else first.tz_convert(index.tz)
    if first <= index[0] or first > index[-1]:
        raise ValueError("first_test_start must fall inside the data, after its first bar.")

    later = pd.date_range(first, index[-1], freq=step)
    edges = [first] + [edge for edge in later if edge > first] + [index[-1] + pd.Timedelta(1, "ns")]
    positions = index.searchsorted(pd.DatetimeIndex(edges))
    min_span = pd.Timedelta(min_train)

    spans = [(int(start), int(end)) for start, end in zip(positions[:-1], positions[1:]) if end > start]
    spans = _merge_stubs(spans)

    folds = []
    for start, end in spans:
        test_time = index[start]
        if train == "expanding":
            train_start = 0
        else:
            train_start = int(index.searchsorted(test_time - pd.Timedelta(train)))
        if test_time - index[train_start] < min_span:
            continue
        folds.append(Fold(len(folds), start, end, train_start))
    if not folds:
        raise ValueError("No walk-forward fold has enough train data; move first_test_start later or relax min_train.")
    return folds


def _merge_stubs(spans: list[tuple[int, int]], min_share: float = 0.25) -> list[tuple[int, int]]:
    """Fold a test span shorter than ``min_share`` of the median span into its neighbour (the previous one when it
    exists, so the merged fold keeps the earlier, more conservative train boundary). A partial last quarter would
    otherwise be scored as a fold of its own on a handful of bars."""
    if len(spans) < 2:
        return spans
    floor = min_share * float(sorted(end - start for start, end in spans)[len(spans) // 2])
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and end - start < floor:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < floor:
        merged[:2] = [(merged[0][0], merged[1][1])]
    return merged
