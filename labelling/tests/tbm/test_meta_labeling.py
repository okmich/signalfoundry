"""Tests for labelling.tbm.meta_labeling."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from okmich_quant_labelling.tbm.meta_labeling import (
    build_meta_features,
    get_meta_labels,
    train_meta_model,
)


class TestGetMetaLabels:
    def test_correct_primary_marked_one(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1h")
        tb = pd.DataFrame({"label": [1, -1, 0, 1, -1]}, index=idx)
        primary = pd.Series([1, -1, 1, -1, -1], index=idx)
        out = get_meta_labels(tb, primary)
        assert list(out) == [1, 1, 0, 0, 1]

    def test_vertical_always_zero(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="1h")
        tb = pd.DataFrame({"label": [0, 0, 0]}, index=idx)
        primary = pd.Series([1, -1, 1], index=idx)
        out = get_meta_labels(tb, primary)
        assert list(out) == [0, 0, 0]

    def test_index_contract_missing_raises(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="1h")
        tb = pd.DataFrame({"label": [1, -1, 0]}, index=idx)
        primary_extra = pd.Series([1], index=pd.DatetimeIndex(["2099-01-01"]))
        with pytest.raises(KeyError):
            get_meta_labels(tb, primary_extra)

    def test_non_unique_tb_index_rejected(self):
        idx = pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-02"])
        tb = pd.DataFrame({"label": [1, -1, 0]}, index=idx)
        primary = pd.Series([1, -1, 1], index=pd.date_range("2026-01-01", periods=3, freq="1h"))
        with pytest.raises(ValueError, match="unique"):
            get_meta_labels(tb, primary)

    def test_non_unique_primary_index_rejected(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="1h")
        tb = pd.DataFrame({"label": [1, -1, 0]}, index=idx)
        primary = pd.Series([1, -1, 1], index=pd.DatetimeIndex(["2026-01-01", "2026-01-01", "2026-01-02"]))
        with pytest.raises(ValueError, match="unique"):
            get_meta_labels(tb, primary)


class TestBuildMetaFeatures:
    def test_no_lookahead_in_vol_ratio(self, gbm_close):
        vol = gbm_close.pct_change().rolling(20).std()
        primary = pd.Series(0.6, index=gbm_close.index)
        events = gbm_close.index[100:200]
        feats = build_meta_features(events, gbm_close, vol, primary)

        t0 = events[5]
        denom_window = vol.rolling(20, min_periods=20).mean().shift(1)
        expected = vol.loc[t0] / denom_window.loc[t0]
        assert feats.loc[t0, "vol_ratio"] == pytest.approx(expected, nan_ok=True)

    def test_minimum_feature_set_present(self, gbm_close):
        vol = gbm_close.pct_change().rolling(20).std()
        primary = pd.Series(0.6, index=gbm_close.index)
        events = gbm_close.index[100:200]
        feats = build_meta_features(events, gbm_close, vol, primary)
        for col in ["primary_score", "vol_at_signal", "vol_ratio", "ret_1", "ret_5", "ret_20"]:
            assert col in feats.columns

    def test_drops_nan_rows(self, gbm_close):
        vol = gbm_close.pct_change().rolling(20).std()
        primary = pd.Series(0.6, index=gbm_close.index)
        events = gbm_close.index[:50]  # too early; ret_20 will be NaN at the start
        feats = build_meta_features(events, gbm_close, vol, primary)
        assert not feats.isna().any().any()

    def test_custom_vol_ratio_window(self, gbm_close):
        vol = gbm_close.pct_change().rolling(20).std()
        primary = pd.Series(0.6, index=gbm_close.index)
        events = gbm_close.index[200:300]
        feats = build_meta_features(events, gbm_close, vol, primary, vol_ratio_window=50)
        # Sanity: ratio at event[5] uses 50-bar mean shifted by 1
        t0 = events[5]
        denom = vol.rolling(50, min_periods=50).mean().shift(1).loc[t0]
        expected = vol.loc[t0] / denom
        assert feats.loc[t0, "vol_ratio"] == pytest.approx(expected, nan_ok=True)

    def test_invalid_vol_ratio_window(self, gbm_close):
        vol = gbm_close.pct_change().rolling(20).std()
        primary = pd.Series(0.6, index=gbm_close.index)
        events = gbm_close.index[100:110]
        with pytest.raises(ValueError):
            build_meta_features(events, gbm_close, vol, primary, vol_ratio_window=1)


class TestTrainMetaModel:
    def test_requires_cv_splitter(self):
        idx = pd.date_range("2026-01-01", periods=50, freq="1h")
        feats = pd.DataFrame(np.random.randn(50, 4), columns=list("abcd"), index=idx)
        y = pd.Series([0, 1] * 25, index=idx)
        with pytest.raises(ValueError, match="cv_splitter is required"):
            train_meta_model(feats, y, cv_splitter=None)

    def test_returns_model_and_metrics_with_stratified(self):
        rng = np.random.default_rng(0)
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n, p=[0.6, 0.4]), index=idx)
        model, metrics = train_meta_model(feats, y, cv_splitter=StratifiedKFold(n_splits=5), allow_unsafe_splitters=True)
        for k in ["precision", "recall", "f1", "roc_auc", "brier_score", "n_train",
                  "n_evaluated", "label_balance_train", "label_balance_test", "cv_folds"]:
            assert k in metrics
        assert metrics["n_train"] == n
        proba = model.predict_proba(feats.values)
        assert proba.shape == (n, 2)

    def test_does_not_mutate_caller_model(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.validation import check_is_fitted, NotFittedError
        rng = np.random.default_rng(11)
        n = 100
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        caller_model = RandomForestClassifier(n_estimators=10, random_state=0)
        fitted, _ = train_meta_model(feats, y, cv_splitter=StratifiedKFold(n_splits=3), allow_unsafe_splitters=True,
                                     model=caller_model)
        # Caller's model should still be unfitted
        with pytest.raises(NotFittedError):
            check_is_fitted(caller_model)
        # Returned model is a different object and IS fitted
        assert fitted is not caller_model
        check_is_fitted(fitted)

    def test_integer_pd_index_split(self):
        # Regression: integer pd.Index used to fall through to timestamp lookup -> empty.
        rng = np.random.default_rng(13)
        n = 60
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 2)), columns=list("ab"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        splitter = [(pd.Index(np.arange(40)), pd.Index(np.arange(40, 60)))]
        model, metrics = train_meta_model(feats, y, cv_splitter=splitter)
        assert metrics["n_evaluated"] == 20

    def test_accepts_timestamp_pair_iterable(self):
        rng = np.random.default_rng(1)
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n, p=[0.5, 0.5]), index=idx)
        # Walk-forward style: first 60% train, next 40% test (timestamps)
        train_ts = idx[:120]
        test_ts = idx[120:]
        splitter = [(train_ts, test_ts)]
        model, metrics = train_meta_model(feats, y, cv_splitter=splitter)
        assert metrics["cv_folds"] == 1
        assert metrics["n_evaluated"] == 80

    def test_accepts_integer_pair_iterable(self):
        rng = np.random.default_rng(2)
        n = 100
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        splitter = [(np.arange(60), np.arange(60, 100))]
        model, metrics = train_meta_model(feats, y, cv_splitter=splitter)
        assert metrics["cv_folds"] == 1

    def test_rejects_single_class_labels(self):
        idx = pd.date_range("2026-01-01", periods=50, freq="1h")
        feats = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"), index=idx)
        y = pd.Series([1] * 50, index=idx)
        with pytest.raises(ValueError, match="at least 2 distinct classes"):
            train_meta_model(feats, y, cv_splitter=StratifiedKFold(n_splits=5), allow_unsafe_splitters=True)

    def test_skips_single_class_train_folds(self, caplog):
        rng = np.random.default_rng(3)
        n = 100
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        # First half all 0s, second half all 1s -> contiguous train fold has only one class
        y = pd.Series([0] * 50 + [1] * 50, index=idx)
        train_a = np.arange(0, 50)        # all 0s -> single class, must skip
        test_a = np.arange(50, 75)        # all 1s
        train_b = np.arange(0, 75)        # mixed -> usable
        test_b = np.arange(75, 100)       # all 1s
        splitter = [(train_a, test_a), (train_b, test_b)]
        with caplog.at_level("WARNING"):
            try:
                model, metrics = train_meta_model(feats, y, cv_splitter=splitter)
            except ValueError:
                # Fold-b train_b has y in [0,1] so it's mixed; fold-a should skip,
                # then fold-b's test is all-1 so roc_auc returns NaN but model still trains.
                pytest.fail("expected at least one usable fold (fold-b)")
        assert any("only one class" in m for m in caplog.messages)

    def test_no_overlap_raises(self):
        idx_a = pd.date_range("2026-01-01", periods=10, freq="1h")
        idx_b = pd.date_range("2099-01-01", periods=10, freq="1h")
        feats = pd.DataFrame(np.zeros((10, 2)), columns=list("ab"), index=idx_a)
        y = pd.Series([0, 1] * 5, index=idx_b)
        with pytest.raises(ValueError):
            train_meta_model(feats, y, cv_splitter=StratifiedKFold(n_splits=2), allow_unsafe_splitters=True)

    def test_rejects_unsafe_splitter_by_default(self):
        rng = np.random.default_rng(7)
        n = 100
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        with pytest.raises(ValueError, match="partition-style"):
            train_meta_model(feats, y, cv_splitter=StratifiedKFold(n_splits=3))

    def test_accepts_time_series_split(self):
        from sklearn.model_selection import TimeSeriesSplit
        rng = np.random.default_rng(8)
        n = 100
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        # Should not raise (TimeSeriesSplit respects time order)
        model, metrics = train_meta_model(feats, y, cv_splitter=TimeSeriesSplit(n_splits=3))
        assert metrics["cv_folds"] >= 1

    def test_strict_timestamp_resolution_raises_on_missing(self):
        rng = np.random.default_rng(9)
        n = 50
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        # CV fold contains a timestamp NOT in feats.index
        bad_ts = pd.Timestamp("2099-01-01")
        train = pd.DatetimeIndex(list(idx[:30]) + [bad_ts])
        test = idx[30:]
        with pytest.raises(KeyError, match="not in feature index"):
            train_meta_model(feats, y, cv_splitter=[(train, test)])

    def test_strict_false_drops_silently(self, caplog):
        rng = np.random.default_rng(10)
        n = 50
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        bad_ts = pd.Timestamp("2099-01-01")
        train = pd.DatetimeIndex(list(idx[:30]) + [bad_ts])
        test = idx[30:]
        with caplog.at_level("WARNING"):
            model, metrics = train_meta_model(feats, y, cv_splitter=[(train, test)],
                                              strict_timestamp_resolution=False)
        assert any("not in feature index" in m for m in caplog.messages)
        assert metrics["cv_folds"] == 1

    def test_rejects_nan_sample_weights(self):
        rng = np.random.default_rng(12)
        n = 60
        idx = pd.date_range("2026-01-01", periods=n, freq="1h")
        feats = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"), index=idx)
        y = pd.Series(rng.choice([0, 1], size=n), index=idx)
        # Weights cover only first 50; reindex produces NaN for last 10
        partial_weights = pd.Series(np.ones(50), index=idx[:50])
        with pytest.raises(ValueError, match="sample_weight"):
            train_meta_model(feats, y, cv_splitter=[(np.arange(40), np.arange(40, 60))],
                             sample_weight=partial_weights)
