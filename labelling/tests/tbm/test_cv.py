"""Tests for labelling.tbm.cv — purging, embargo, walk-forward."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.tbm.cv import (
    embargo_train_labels,
    purge_train_labels,
    purged_walk_forward_cv,
)


def _build_labels(n=200, freq="1h"):
    idx = pd.date_range("2026-01-01", periods=n, freq=freq)
    t1 = idx + pd.Timedelta(hours=5)  # each label has a 5-bar window
    return pd.DataFrame({"t1": t1, "ret": np.zeros(n), "label": np.zeros(n, dtype=int),
                         "barrier": ["vertical"] * n}, index=idx)


class TestPurgeTrainLabels:
    def test_overlap_purged(self):
        labels = _build_labels(20)
        train = labels.index[:10]
        test = labels.index[10:]
        purged = purge_train_labels(train, test, labels)
        # train events 5..9 have t1 within test window -> purged
        assert len(purged) == 5
        assert purged[-1] == labels.index[4]

    def test_empty_inputs(self):
        labels = _build_labels(10)
        empty = pd.DatetimeIndex([])
        assert len(purge_train_labels(empty, labels.index, labels)) == 0
        assert len(purge_train_labels(labels.index, empty, labels)) == len(labels)

    def test_missing_t1_column_raises(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        labels = pd.DataFrame({"label": [0] * 5}, index=idx)
        with pytest.raises(ValueError):
            purge_train_labels(idx[:2], idx[2:], labels)

    def test_train_index_not_subset_raises_with_clear_message(self):
        labels = _build_labels(10)
        # train_index has a timestamp not in labels.index
        bad_train = labels.index[:5].append(pd.DatetimeIndex(["2099-01-01"]))
        with pytest.raises(ValueError, match="not in labels.index"):
            purge_train_labels(bad_train, labels.index[5:], labels)


class TestEmbargoTrainLabels:
    def test_post_test_buffer_dropped(self):
        labels = _build_labels(50)
        close = pd.Series(np.zeros(len(labels)), index=labels.index)
        # train spans full range; test_end at index 30; embargo 5 bars
        train = labels.index
        test_end = labels.index[30]
        out = embargo_train_labels(train, test_end, embargo_bars=5, close=close)
        # train events at iloc 31..35 should be removed
        kept = set(out)
        for i in range(31, 36):
            assert labels.index[i] not in kept

    def test_zero_embargo_returns_input(self):
        labels = _build_labels(20)
        close = pd.Series(np.zeros(20), index=labels.index)
        out = embargo_train_labels(labels.index, labels.index[10], 0, close)
        assert out.equals(labels.index)


class TestPurgedWalkForwardCV:
    def test_yields_chronological_folds(self):
        labels = _build_labels(200)
        close = pd.Series(np.zeros(len(labels)), index=labels.index)
        folds = list(purged_walk_forward_cv(labels, close, n_splits=5))
        assert len(folds) == 4  # n_splits - 1 (first fold is reserved for training only)
        prev_test_end = pd.Timestamp.min.tz_localize(None)
        for train_idx, test_idx in folds:
            assert train_idx.max() < test_idx.min()
            assert test_idx[-1] > prev_test_end
            prev_test_end = test_idx[-1]

    def test_low_train_count_skipped(self, caplog):
        # n=80, 5 splits => fold sizes ~16. After purging the t1 overlap, fold 1 has
        # < 30 train events and is skipped; later folds yield normally.
        labels = _build_labels(80)
        close = pd.Series(np.zeros(len(labels)), index=labels.index)
        with caplog.at_level("WARNING"):
            folds = list(purged_walk_forward_cv(labels, close, n_splits=5))
        assert any("skipping" in m for m in caplog.messages)
        assert len(folds) >= 1

    def test_small_n_splits_raises(self):
        labels = _build_labels(20)
        close = pd.Series(np.zeros(20), index=labels.index)
        with pytest.raises(ValueError):
            list(purged_walk_forward_cv(labels, close, n_splits=1))

    def test_invalid_split_by_raises(self):
        labels = _build_labels(200)
        close = pd.Series(np.zeros(len(labels)), index=labels.index)
        with pytest.raises(ValueError, match="split_by"):
            list(purged_walk_forward_cv(labels, close, n_splits=5, split_by="bogus"))

    def test_split_by_time_yields_time_bounded_folds(self):
        # Build labels with uneven event density:
        #   first 100 labels packed in 10 days, next 100 spread over 90 days.
        idx_a = pd.date_range("2026-01-01", periods=100, freq="2h")
        idx_b = pd.date_range("2026-01-12", periods=100, freq="1D")
        idx = idx_a.append(idx_b)
        t1 = idx + pd.Timedelta(hours=5)
        labels = pd.DataFrame({"t1": t1, "ret": np.zeros(200), "label": np.zeros(200, dtype=int),
                               "barrier": ["vertical"] * 200}, index=idx)
        close = pd.Series(np.zeros(len(labels)), index=labels.index)
        folds = list(purged_walk_forward_cv(labels, close, n_splits=5, split_by="time"))
        # Time-mode folds should sweep evenly through calendar time, NOT event count.
        # The dense-front + sparse-back labelling makes event counts very uneven.
        for train_idx, test_idx in folds:
            assert train_idx.max() < test_idx.min()
        if folds:
            # The last fold must reach the end of the series in time-mode
            assert folds[-1][1][-1] == labels.index[-1]
