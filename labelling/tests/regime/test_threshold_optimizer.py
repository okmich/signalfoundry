import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from okmich_quant_labelling.regime import (
    ClipMethod,
    DirectFeatureThresholdOptimizer,
    MarketPropertyType,
    ObjectiveType,
    ThresholdMethod,
)


def _make_feature_series(n: int = 1200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.Series(rng.normal(0.0, 1.0, n), index=idx, name="feature")


def _turnover(labels: pd.Series) -> float:
    clean = labels.dropna()
    if len(clean) < 2:
        return np.nan
    return float((clean != clean.shift(1)).dropna().mean())


class TestDirectFeatureThresholdOptimizerInitialization:
    def test_rejects_invalid_n_classes(self):
        with pytest.raises(ValueError, match="n_classes"):
            DirectFeatureThresholdOptimizer(n_classes=1)

    def test_rejects_invalid_min_persistence(self):
        with pytest.raises(ValueError, match="min_persistence"):
            DirectFeatureThresholdOptimizer(min_persistence=0)

    def test_repr_contains_key_params(self):
        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.VOLATILITY,
            n_classes=4,
            threshold_method=ThresholdMethod.KMEANS_1D,
            clip_method=ClipMethod.PERCENTILE,
        )
        r = repr(opt)
        assert "DirectFeatureThresholdOptimizer" in r
        assert "volatility" in r
        assert "kmeans_1d" in r
        assert "percentile" in r

    def test_directional_supervised_requires_edge_objective(self):
        with pytest.raises(ValueError, match="must optimize ObjectiveType.EDGE"):
            DirectFeatureThresholdOptimizer(
                market_property_type=MarketPropertyType.DIRECTION,
                threshold_method=ThresholdMethod.SUPERVISED_GRID,
                objective_type=ObjectiveType.SEPARATION,
            )


class TestQuantileFitPredict:
    def test_fit_predict_quantile_k_classes(self):
        feature = _make_feature_series(n=1500, seed=1)
        opt = DirectFeatureThresholdOptimizer(
            n_classes=4,
            threshold_method=ThresholdMethod.QUANTILE,
            min_class_support=120,
            max_class_imbalance=2.5,
        )
        labels = opt.fit_predict(feature)

        assert opt.is_fitted_ is True
        assert opt.thresholds_.shape == (3,)
        assert np.all(np.diff(opt.thresholds_) > 0)
        assert set(labels.dropna().astype(int).unique()).issubset({0, 1, 2, 3})

    def test_predict_preserves_nan_positions(self):
        feature = _make_feature_series(n=800, seed=2)
        feature.iloc[10:20] = np.nan
        opt = DirectFeatureThresholdOptimizer(n_classes=3, min_class_support=80)
        opt.fit(feature.ffill())
        labels = opt.predict(feature)

        assert labels.iloc[10:20].isna().all()
        assert labels.iloc[30:].notna().any()

    def test_constant_series_fails_fast(self):
        idx = pd.date_range("2024-01-01", periods=500, freq="5min")
        feature = pd.Series(3.14, index=idx)
        opt = DirectFeatureThresholdOptimizer(n_classes=3, threshold_method=ThresholdMethod.QUANTILE, min_class_support=50)
        with pytest.raises(ValueError, match="Thresholds are not strictly increasing"):
            opt.fit(feature)

    def test_insufficient_data_length_raises(self):
        feature = _make_feature_series(n=450, seed=3)
        opt = DirectFeatureThresholdOptimizer(
            n_classes=4,
            threshold_method=ThresholdMethod.QUANTILE,
            min_class_support=180,
        )
        with pytest.raises(ValueError, match="Insufficient data length"):
            opt.fit(feature)

    def test_failed_refit_clears_previous_fitted_state(self):
        train = _make_feature_series(n=1400, seed=31)
        bad = pd.Series(3.14, index=pd.date_range("2024-02-01", periods=800, freq="5min"))
        opt = DirectFeatureThresholdOptimizer(n_classes=3, threshold_method=ThresholdMethod.QUANTILE, min_class_support=80)
        opt.fit(train)

        with pytest.raises(ValueError, match="Thresholds are not strictly increasing"):
            opt.fit(bad)

        assert opt.is_fitted_ is False
        assert opt.thresholds_ is None
        assert opt.class_signals_ is None
        with pytest.raises(RuntimeError, match="Optimizer is not fitted"):
            opt.predict(train)


class TestKMeansMethod:
    def test_kmeans_fit_is_reproducible(self):
        feature = _make_feature_series(n=1600, seed=4)
        opt1 = DirectFeatureThresholdOptimizer(
            n_classes=3,
            threshold_method=ThresholdMethod.KMEANS_1D,
            random_state=11,
            min_class_support=120,
        )
        opt2 = DirectFeatureThresholdOptimizer(
            n_classes=3,
            threshold_method=ThresholdMethod.KMEANS_1D,
            random_state=11,
            min_class_support=120,
        )
        opt1.fit(feature)
        opt2.fit(feature)
        np.testing.assert_allclose(opt1.thresholds_, opt2.thresholds_)


class TestClipping:
    def test_percentile_clipping_flags_outliers(self):
        feature = _make_feature_series(n=1000, seed=5)
        feature.iloc[100] = 200.0
        feature.iloc[200] = -220.0

        opt = DirectFeatureThresholdOptimizer(
            n_classes=3,
            clip_method=ClipMethod.PERCENTILE,
            clip_lower_pct=0.01,
            clip_upper_pct=0.99,
            min_class_support=80,
        )
        opt.fit(feature)
        labels, diag = opt.predict(feature, return_diagnostics=True)

        assert labels.notna().sum() > 0
        assert bool(diag.loc[feature.index[100], "is_clipped_high"]) is True
        assert bool(diag.loc[feature.index[200], "is_clipped_low"]) is True

    def test_clip_bounds_remain_frozen_on_predict(self):
        train = _make_feature_series(n=900, seed=6)
        test = _make_feature_series(n=300, seed=7)
        test.iloc[-1] = 999.0

        opt = DirectFeatureThresholdOptimizer(
            n_classes=3,
            clip_method=ClipMethod.IQR,
            min_class_support=80,
        )
        opt.fit(train)
        bounds_before = tuple(opt.clip_bounds_)
        _ = opt.predict(test)
        assert tuple(opt.clip_bounds_) == bounds_before


class TestSmoothing:
    def test_min_persistence_reduces_turnover(self):
        train = pd.Series(np.linspace(-2.0, 2.0, 1000), index=pd.date_range("2024-01-01", periods=1000, freq="5min"))
        test_values = [-0.4, -0.35, -0.3, -0.25] + ([-0.05, 0.05] * 298)
        test = pd.Series(test_values, index=pd.date_range("2024-03-01", periods=600, freq="5min"))

        fast = DirectFeatureThresholdOptimizer(n_classes=2, min_persistence=1, min_class_support=200)
        slow = DirectFeatureThresholdOptimizer(n_classes=2, min_persistence=4, min_class_support=200)
        fast.fit(train)
        slow.fit(train)

        fast_labels = fast.predict(test)
        slow_labels = slow.predict(test)
        assert _turnover(slow_labels) < _turnover(fast_labels)

    def test_hysteresis_reduces_boundary_flips(self):
        train = _make_feature_series(n=1500, seed=8)
        no_hys = DirectFeatureThresholdOptimizer(n_classes=2, min_persistence=1, hysteresis=0.0, min_class_support=250)
        hys = DirectFeatureThresholdOptimizer(n_classes=2, min_persistence=1, hysteresis=0.03, min_class_support=250)
        no_hys.fit(train)
        hys.fit(train)

        boundary = float(no_hys.thresholds_[0])
        around_boundary = pd.Series(
            boundary + np.tile([-0.005, 0.005, -0.007, 0.007], 150),
            index=pd.date_range("2024-06-01", periods=600, freq="5min"),
        )

        labels_no = no_hys.predict(around_boundary)
        labels_hy = hys.predict(around_boundary)
        assert _turnover(labels_hy) < _turnover(labels_no)


class TestSupervisedMethods:
    def test_supervised_grid_fits_with_forward_returns(self):
        feature = _make_feature_series(n=1400, seed=9)
        rng = np.random.default_rng(99)
        forward_returns = 0.001 * np.sign(feature) + rng.normal(0.0, 0.0005, len(feature))
        forward_returns = pd.Series(forward_returns, index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.SUPERVISED_GRID,
            objective_type=ObjectiveType.EDGE,
            supervised_grid_size=19,
            min_class_support=120,
            turnover_penalty=0.1,
        )
        opt.fit(feature, forward_returns=forward_returns)
        labels = opt.predict(feature)

        assert opt.thresholds_.shape == (2,)
        assert labels.notna().sum() > 0

    def test_supervised_methods_require_forward_returns(self):
        feature = _make_feature_series(n=1000, seed=10)
        opt = DirectFeatureThresholdOptimizer(
            n_classes=3,
            threshold_method=ThresholdMethod.SUPERVISED_GRID,
            min_class_support=100,
        )
        with pytest.raises(ValueError, match="forward_returns are required"):
            opt.fit(feature)

    def test_supervised_grid_accepts_array_forward_returns(self):
        feature = _make_feature_series(n=1400, seed=91)
        rng = np.random.default_rng(92)
        forward_returns = 0.001 * np.sign(feature.to_numpy(dtype=float)) + rng.normal(0.0, 0.0005, len(feature))

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.SUPERVISED_GRID,
            objective_type=ObjectiveType.EDGE,
            supervised_grid_size=17,
            min_class_support=120,
        )
        opt.fit(feature, forward_returns=forward_returns)
        labels = opt.predict(feature)

        assert opt.thresholds_.shape == (2,)
        assert labels.notna().sum() > 0


class TestForwardBlockDiagnostics:
    def test_evaluate_forward_blocks_outputs_expected_columns(self):
        feature = _make_feature_series(n=1800, seed=11)
        rng = np.random.default_rng(11)
        forward_returns = pd.Series(rng.normal(0.0, 0.0008, len(feature)), index=feature.index)

        opt = DirectFeatureThresholdOptimizer(n_classes=3, min_class_support=120)
        opt.fit(feature)
        thresholds_before = opt.thresholds_.copy()

        blocks = opt.evaluate_forward_blocks(feature=feature, forward_returns=forward_returns, block_size=300)
        np.testing.assert_allclose(opt.thresholds_, thresholds_before)

        required = {
            "block_id",
            "n_samples",
            "coverage",
            "turnover",
            "mean_dwell",
            "clip_rate",
            "class_0_count",
            "class_1_count",
            "class_2_count",
            "edge",
            "edge_bps",
            "hit_rate",
        }
        assert required.issubset(set(blocks.columns))
        assert len(blocks) >= 2


class TestDriftAndAcceptance:
    def test_evaluate_drift_detects_large_shift(self):
        train = _make_feature_series(n=2000, seed=21)
        shifted = pd.Series(
            np.random.default_rng(22).normal(2.0, 1.2, len(train)),
            index=train.index,
            name="feature",
        )
        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            min_class_support=150,
            clip_method=ClipMethod.PERCENTILE,
        )
        opt.fit(train)
        drift = opt.evaluate_drift(shifted)

        assert "psi" in drift
        assert "alert_trigger" in drift
        assert drift["psi"] >= 0.0
        assert drift["alert_trigger"] is True

    def test_evaluate_acceptance_directional_contract(self):
        feature = _make_feature_series(n=2200, seed=23)
        rng = np.random.default_rng(24)
        forward_returns = pd.Series(0.0008 * np.sign(feature) + rng.normal(0.0, 0.0004, len(feature)), index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.SUPERVISED_GRID,
            objective_type=ObjectiveType.EDGE,
            min_class_support=120,
            supervised_grid_size=17,
        )
        opt.fit(feature, forward_returns=forward_returns)
        result = opt.evaluate_acceptance(feature=feature, forward_returns=forward_returns, block_size=500, n_perm=80)

        assert "pass" in result
        assert "violated_gates" in result
        assert "summary" in result
        assert "edge" in result["summary"]
        assert "perm_pvalue" in result["summary"]

    def test_evaluate_acceptance_nondirectional_contract(self):
        feature = _make_feature_series(n=2200, seed=25)
        rng = np.random.default_rng(26)
        forward_returns = pd.Series(rng.normal(0.0, 0.0007, len(feature)), index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.VOLATILITY,
            n_classes=3,
            threshold_method=ThresholdMethod.QUANTILE,
            objective_type=ObjectiveType.SEPARATION,
            min_class_support=120,
        )
        opt.fit(feature, forward_returns=forward_returns)
        result = opt.evaluate_acceptance(feature=feature, forward_returns=forward_returns, block_size=500, n_perm=80)

        assert "pass" in result
        assert "separation_score" in result["summary"]
        assert "kruskal_pvalue" in result["summary"]

    def test_evaluate_acceptance_directional_fails_on_non_finite_metrics(self):
        feature = _make_feature_series(n=1800, seed=33)
        forward_returns = pd.Series(np.nan, index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.QUANTILE,
            min_class_support=120,
        )
        opt.fit(feature)
        result = opt.evaluate_acceptance(feature=feature, forward_returns=forward_returns, block_size=300, n_perm=20)

        assert result["pass"] is False
        assert "edge_non_finite" in result["violated_gates"]
        assert "perm_pvalue_non_finite" in result["violated_gates"]

    def test_evaluate_acceptance_uses_requested_block_size_once(self, monkeypatch):
        feature = _make_feature_series(n=2200, seed=34)
        rng = np.random.default_rng(35)
        forward_returns = pd.Series(0.0008 * np.sign(feature) + rng.normal(0.0, 0.0004, len(feature)), index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.SUPERVISED_GRID,
            objective_type=ObjectiveType.EDGE,
            min_class_support=120,
            supervised_grid_size=17,
        )
        opt.fit(feature, forward_returns=forward_returns)

        block_calls: list[int] = []
        original = opt.evaluate_forward_blocks

        def wrapped(*args, **kwargs):
            block_calls.append(kwargs.get("block_size"))
            return original(*args, **kwargs)

        monkeypatch.setattr(opt, "evaluate_forward_blocks", wrapped)
        _ = opt.evaluate_acceptance(feature=feature, forward_returns=forward_returns, block_size=250, n_perm=20)

        assert block_calls == [250]


class TestPersistence:
    def test_save_load_round_trip_predictions_identical(self):
        feature = _make_feature_series(n=1200, seed=12)
        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.PATH_STRUCTURE,
            n_classes=3,
            clip_method=ClipMethod.MAD_ZSCORE,
            threshold_method=ThresholdMethod.KMEANS_1D,
            min_class_support=100,
            random_state=123,
        )
        opt.fit(feature)
        expected = opt.predict(feature)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=str(Path.cwd())) as handle:
            path = Path(handle.name)
        try:
            opt.save(path)
            loaded = DirectFeatureThresholdOptimizer.load(path)
            actual = loaded.predict(feature)
        finally:
            if path.exists():
                path.unlink()

        pd.testing.assert_series_equal(expected, actual)

    def test_save_load_preserves_drift_diagnostics(self):
        train = _make_feature_series(n=1600, seed=55)
        shifted = pd.Series(
            np.random.default_rng(56).normal(1.5, 1.1, len(train)),
            index=train.index,
            name="feature",
        )
        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            min_class_support=120,
            clip_method=ClipMethod.PERCENTILE,
        )
        opt.fit(train)
        expected = opt.evaluate_drift(shifted)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, dir=str(Path.cwd())) as handle:
            path = Path(handle.name)
        try:
            opt.save(path)
            loaded = DirectFeatureThresholdOptimizer.load(path)
            actual = loaded.evaluate_drift(shifted)
        finally:
            if path.exists():
                path.unlink()

        assert actual["psi"] == pytest.approx(expected["psi"], rel=1e-6, abs=1e-9)
        assert actual["ks_stat"] == pytest.approx(expected["ks_stat"], rel=1e-6, abs=1e-9)
        assert actual["current_clip_rate"] == pytest.approx(expected["current_clip_rate"], rel=1e-6, abs=1e-9)
        assert actual["alert_trigger"] == expected["alert_trigger"]
        assert actual["refit_trigger"] == expected["refit_trigger"]
        assert actual["trigger_breakdown"] == expected["trigger_breakdown"]


class TestDriftTriggerBreakdown:
    def test_mean_shift_fires_psi_and_ks_triggers(self):
        train = _make_feature_series(n=2000, seed=61)
        shifted = pd.Series(
            np.random.default_rng(62).normal(3.0, 0.8, len(train)),
            index=train.index,
            name="feature",
        )
        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            min_class_support=150,
            clip_method=ClipMethod.PERCENTILE,
        )
        opt.fit(train)
        drift = opt.evaluate_drift(shifted)

        breakdown = drift["trigger_breakdown"]
        assert breakdown["psi_alert_trigger"] is True
        assert breakdown["ks_trigger"] is True
        assert drift["alert_trigger"] is True

    def test_stable_distribution_does_not_fire_triggers(self):
        rng = np.random.default_rng(71)
        idx = pd.date_range("2024-01-01", periods=2200, freq="5min")
        draws = rng.normal(0.0, 1.0, len(idx))
        train = pd.Series(draws[:1800], index=idx[:1800], name="feature")
        test = pd.Series(draws[1800:], index=idx[1800:], name="feature")
        opt = DirectFeatureThresholdOptimizer(n_classes=3, min_class_support=150)
        opt.fit(train)
        drift = opt.evaluate_drift(test)

        breakdown = drift["trigger_breakdown"]
        assert breakdown["psi_refit_trigger"] is False
        assert drift["refit_trigger"] is False


class TestNonDirectionalBlockOutput:
    def test_evaluate_forward_blocks_skips_edge_for_nondirectional(self):
        feature = _make_feature_series(n=1800, seed=81)
        rng = np.random.default_rng(82)
        forward_returns = pd.Series(rng.normal(0.0, 0.0008, len(feature)), index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.VOLATILITY,
            n_classes=3,
            threshold_method=ThresholdMethod.QUANTILE,
            min_class_support=120,
        )
        opt.fit(feature)
        blocks = opt.evaluate_forward_blocks(feature=feature, forward_returns=forward_returns, block_size=300)

        assert "edge" not in blocks.columns
        assert "edge_bps" not in blocks.columns
        assert "hit_rate" not in blocks.columns
        # Structural columns are still present.
        for col in ("block_id", "n_samples", "coverage", "turnover", "mean_dwell", "clip_rate"):
            assert col in blocks.columns

    def test_evaluate_forward_blocks_includes_edge_for_directional(self):
        feature = _make_feature_series(n=1800, seed=83)
        rng = np.random.default_rng(84)
        forward_returns = pd.Series(rng.normal(0.0, 0.0008, len(feature)), index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.QUANTILE,
            min_class_support=120,
        )
        opt.fit(feature)
        blocks = opt.evaluate_forward_blocks(feature=feature, forward_returns=forward_returns, block_size=300)

        for col in ("edge", "edge_bps", "hit_rate"):
            assert col in blocks.columns


class TestHysteresisSkipTransition:
    def test_hysteresis_gates_on_boundary_adjacent_to_target_class(self):
        train = pd.Series(
            np.linspace(-3.0, 3.0, 1500),
            index=pd.date_range("2024-01-01", periods=1500, freq="5min"),
        )
        opt = DirectFeatureThresholdOptimizer(
            n_classes=3,
            threshold_method=ThresholdMethod.QUANTILE,
            min_persistence=1,
            hysteresis=0.2,
            min_class_support=200,
        )
        opt.fit(train)
        t0, t1 = float(opt.thresholds_[0]), float(opt.thresholds_[1])
        assert (t1 - t0) > 0.4  # Sanity: skip gap is wider than hysteresis band.

        # Seed the state deep in class 0, then jump to a value that clears
        # threshold t1 (raw label = 2) but sits INSIDE the hysteresis band
        # above t1 (value < t1 + h). The correct gate for a 0 -> 2 transition
        # is "value >= t1 + h", so the transition must be rejected.
        below_t0 = t0 - 1.0
        inside_band_above_t1 = t1 + 0.05
        test_values = [below_t0] * 5 + [inside_band_above_t1] * 5
        test = pd.Series(
            test_values,
            index=pd.date_range("2024-03-01", periods=len(test_values), freq="5min"),
        )
        labels = opt.predict(test)

        assert int(labels.iloc[-1]) == 0, "skip transition 0->2 should not fire while value is inside hysteresis band of t1"


class TestStateMachineNaNGap:
    def test_nan_gap_resets_candidate_count(self):
        train = _make_feature_series(n=1500, seed=91)
        opt = DirectFeatureThresholdOptimizer(
            n_classes=2,
            threshold_method=ThresholdMethod.QUANTILE,
            min_persistence=3,
            hysteresis=0.0,
            min_class_support=250,
        )
        opt.fit(train)
        t0 = float(opt.thresholds_[0])
        above = t0 + 0.5
        below = t0 - 0.5

        # Build: long run in class 0, then two consecutive "above" ticks
        # (count reaches 2 under persistence=3, so current stays 0), then a
        # NaN gap, then one more "above" tick.
        #   - Without reset: count carries through NaN and becomes 3 on the
        #     post-gap tick, transitioning current to 1.
        #   - With reset: count restarts at 1 post-gap, current stays at 0.
        vals = [below] * 5 + [above, above, np.nan, above]
        test = pd.Series(
            vals,
            index=pd.date_range("2024-04-01", periods=len(vals), freq="5min"),
        )
        labels = opt.predict(test)

        assert int(labels.iloc[-1]) == 0


class TestSupervisedDE:
    def test_supervised_de_fits_with_forward_returns(self):
        feature = _make_feature_series(n=1400, seed=101)
        rng = np.random.default_rng(102)
        forward_returns = 0.001 * np.sign(feature) + rng.normal(0.0, 0.0005, len(feature))
        forward_returns = pd.Series(forward_returns, index=feature.index)

        opt = DirectFeatureThresholdOptimizer(
            market_property_type=MarketPropertyType.DIRECTION,
            n_classes=3,
            threshold_method=ThresholdMethod.SUPERVISED_DE,
            objective_type=ObjectiveType.EDGE,
            min_class_support=120,
            de_max_iter=30,
            de_popsize=10,
        )
        opt.fit(feature, forward_returns=forward_returns)
        labels = opt.predict(feature)

        assert opt.thresholds_.shape == (2,)
        assert np.all(np.diff(opt.thresholds_) > 0)
        assert labels.notna().sum() > 0
