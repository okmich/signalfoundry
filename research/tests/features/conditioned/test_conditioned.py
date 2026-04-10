"""
Tests for okmich_quant_research.features.conditioned

Strategy: synthetic datasets with known conditional structure.
  - 3 conditions (regimes), ~1200 rows each (above min_observations=1000)
  - Features with known IC patterns:
    - feat_global: predictive in all conditions (global_stable candidate)
    - feat_regime_a: predictive only in condition A (condition_specific candidate)
    - feat_flip: positive IC in A, negative IC in B (sign-flipping)
    - feat_noise: pure noise (unclassified candidate)
    - feat_multi: predictive in A and B but not C (conditional_ensemble candidate)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.conditioned._enums import ConditionPass, FeatureBucket, FeatureStatus
from okmich_quant_research.features.conditioned._ic_analysis import compute_conditional_ic, compute_global_ic, compute_ic_threshold
from okmich_quant_research.features.conditioned._partition import partition_features
from okmich_quant_research.features.conditioned._result import FeatureConditionMap, assign_status
from okmich_quant_research.features.conditioned._stability import compute_stability_scores, compute_stability_threshold, compute_subperiod_stability
from okmich_quant_research.features.conditioned.analyzer import ConditionalFeatureAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data():
    """Dataset with 3 conditions, 5 features with known IC structure."""
    np.random.seed(42)
    n_per_condition = 1200
    n = n_per_condition * 3

    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    conditions = np.array(["A"] * n_per_condition + ["B"] * n_per_condition + ["C"] * n_per_condition)

    target = np.random.randn(n)

    features = pd.DataFrame(index=idx)
    condition_labels = pd.Series(conditions, index=idx, name="regime")

    # feat_global: correlated with target in all conditions
    features["feat_global"] = target * 0.3 + np.random.randn(n) * 0.7

    # feat_regime_a: correlated only in condition A
    feat_a = np.random.randn(n)
    feat_a[:n_per_condition] = target[:n_per_condition] * 0.4 + np.random.randn(n_per_condition) * 0.6
    features["feat_regime_a"] = feat_a

    # feat_flip: positive in A, negative in B, noise in C
    feat_flip = np.random.randn(n)
    feat_flip[:n_per_condition] = target[:n_per_condition] * 0.3 + np.random.randn(n_per_condition) * 0.7
    feat_flip[n_per_condition:2*n_per_condition] = -target[n_per_condition:2*n_per_condition] * 0.3 + np.random.randn(n_per_condition) * 0.7
    features["feat_flip"] = feat_flip

    # feat_noise: pure noise
    features["feat_noise"] = np.random.randn(n)

    # feat_multi: predictive in A and B, not C
    feat_multi = np.random.randn(n)
    feat_multi[:n_per_condition] = target[:n_per_condition] * 0.25 + np.random.randn(n_per_condition) * 0.75
    feat_multi[n_per_condition:2*n_per_condition] = target[n_per_condition:2*n_per_condition] * 0.25 + np.random.randn(n_per_condition) * 0.75
    features["feat_multi"] = feat_multi

    target_series = pd.Series(target, index=idx, name="target")

    return features, target_series, condition_labels


@pytest.fixture(scope="module")
def time_labels(synthetic_data):
    """Time labels cycling through 3 sessions."""
    features, _, _ = synthetic_data
    n = len(features)
    sessions = np.tile(["session_1", "session_2", "session_3"], n // 3 + 1)[:n]
    return pd.Series(sessions, index=features.index, name="session")


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: _ic_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestICAnalysis:
    def test_compute_conditional_ic_shape(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)

        assert ic.shape == (5, 3)
        assert pv.shape == (5, 3)
        assert obs.shape == (5, 3)
        assert list(ic.index) == list(features.columns)
        assert set(ic.columns) == {"A", "B", "C"}

    def test_global_feature_has_ic_everywhere(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, _, _ = compute_conditional_ic(features, target, conditions, min_observations=100)

        # feat_global should have positive IC in all conditions
        for cond in ["A", "B", "C"]:
            assert ic.loc["feat_global", cond] > 0.05

    def test_noise_feature_has_low_ic(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, _, _ = compute_conditional_ic(features, target, conditions, min_observations=100)

        # feat_noise should have |IC| near 0 everywhere
        for cond in ["A", "B", "C"]:
            assert abs(ic.loc["feat_noise", cond]) < 0.1

    def test_insufficient_data_produces_nan(self, synthetic_data):
        features, target, conditions = synthetic_data
        # With min_observations very high, everything should be NaN
        ic, _, obs = compute_conditional_ic(features, target, conditions, min_observations=100000)
        assert ic.isna().all().all()

    def test_compute_global_ic(self, synthetic_data):
        features, target, _ = synthetic_data
        global_ic = compute_global_ic(features, target)

        assert len(global_ic) == 5
        assert global_ic["feat_global"] > 0.1
        assert abs(global_ic["feat_noise"]) < 0.1

    def test_compute_ic_threshold_returns_positive(self, synthetic_data):
        features, target, conditions = synthetic_data
        threshold = compute_ic_threshold(features, target, conditions, n_permutations=10, min_observations=100)
        assert threshold > 0


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: _stability
# ─────────────────────────────────────────────────────────────────────────────

class TestStability:
    def test_stability_scores_range(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, _, _ = compute_conditional_ic(features, target, conditions, min_observations=100)
        scores = compute_stability_scores(ic)

        assert len(scores) == 5
        for val in scores.dropna():
            assert -1.0 <= val <= 1.0

    def test_global_feature_has_high_stability(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, _, _ = compute_conditional_ic(features, target, conditions, min_observations=100)
        scores = compute_stability_scores(ic)

        # feat_global should be more stable than feat_flip
        assert scores["feat_global"] > scores["feat_flip"]

    def test_stability_threshold_in_range(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, _, _ = compute_conditional_ic(features, target, conditions, min_observations=100)
        scores = compute_stability_scores(ic)
        threshold = compute_stability_threshold(scores)

        assert 0.2 <= threshold <= 0.8

    def test_stability_threshold_fallback_few_features(self):
        # Fewer than 10 features -> fallback to 0.5
        scores = pd.Series([0.3, 0.7, 0.5])
        assert compute_stability_threshold(scores) == 0.5

    def test_subperiod_stability_shape(self, synthetic_data):
        features, target, conditions = synthetic_data
        sp = compute_subperiod_stability(features, target, conditions, n_subperiods=3, min_observations=100)
        assert sp.shape == (5, 3)

    def test_nan_if_insufficient_per_chunk(self, synthetic_data):
        features, target, conditions = synthetic_data
        # Very high min_observations per chunk
        sp = compute_subperiod_stability(features, target, conditions, n_subperiods=3, min_observations=100000)
        assert sp.isna().all().all()


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: assign_status
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignStatus:
    def test_status_types(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)
        status = assign_status(ic, pv, obs, ic_threshold=0.05, max_pvalue=0.05, min_observations=100)

        for val in status.values.flatten():
            assert isinstance(val, FeatureStatus)

    def test_insufficient_data_status(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)
        # Override obs to be low
        obs_low = obs * 0
        status = assign_status(ic, pv, obs_low, ic_threshold=0.05, min_observations=100)

        for val in status.values.flatten():
            assert val == FeatureStatus.INSUFFICIENT_DATA

    def test_active_and_negative_assignment(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)
        status = assign_status(ic, pv, obs, ic_threshold=0.05, max_pvalue=0.05, min_observations=100)

        # feat_global in condition A should be ACTIVE (positive IC)
        assert status.loc["feat_global", "A"] == FeatureStatus.ACTIVE

        # feat_flip in condition B should be NEGATIVE (negative IC)
        assert status.loc["feat_flip", "B"] == FeatureStatus.NEGATIVE


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests: _partition
# ─────────────────────────────────────────────────────────────────────────────

class TestPartition:
    def test_all_features_classified(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)
        status = assign_status(ic, pv, obs, ic_threshold=0.05, min_observations=100)
        scores = compute_stability_scores(ic)
        sp = compute_subperiod_stability(features, target, conditions, n_subperiods=3, min_observations=100)

        gs, cs, ce, uc, detected = partition_features(ic, status, scores, sp, ic_threshold=0.05, stability_threshold=0.5)

        all_classified = set(gs) | set(f for fs in cs.values() for f in fs) | set(ce) | set(uc)
        assert all_classified == set(features.columns)

    def test_no_feature_in_multiple_buckets(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)
        status = assign_status(ic, pv, obs, ic_threshold=0.05, min_observations=100)
        scores = compute_stability_scores(ic)
        sp = compute_subperiod_stability(features, target, conditions, n_subperiods=3, min_observations=100)

        gs, cs, ce, uc, _ = partition_features(ic, status, scores, sp, ic_threshold=0.05, stability_threshold=0.5)

        gs_set = set(gs)
        cs_set = set(f for fs in cs.values() for f in fs)
        ce_set = set(ce)
        uc_set = set(uc)

        assert len(gs_set & cs_set) == 0
        assert len(gs_set & ce_set) == 0
        assert len(gs_set & uc_set) == 0
        assert len(cs_set & ce_set) == 0
        assert len(cs_set & uc_set) == 0
        assert len(ce_set & uc_set) == 0

    def test_noise_is_unclassified(self, synthetic_data):
        features, target, conditions = synthetic_data
        ic, pv, obs = compute_conditional_ic(features, target, conditions, min_observations=100)
        status = assign_status(ic, pv, obs, ic_threshold=0.05, min_observations=100)
        scores = compute_stability_scores(ic)
        sp = compute_subperiod_stability(features, target, conditions, n_subperiods=3, min_observations=100)

        gs, cs, ce, uc, _ = partition_features(ic, status, scores, sp, ic_threshold=0.05, stability_threshold=0.5)

        assert "feat_noise" in uc


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test: ConditionalFeatureAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzer:
    def test_analyze_by_regime(self, synthetic_data):
        features, target, conditions = synthetic_data
        analyzer = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=10, verbose=False)
        result = analyzer.analyze_by_regime(features, target, conditions)

        assert isinstance(result, FeatureConditionMap)
        assert result.condition_pass == ConditionPass.REGIME
        assert len(result.ic_matrix.columns) == 3
        assert len(result.global_stable) + sum(len(v) for v in result.condition_specific.values()) + len(result.conditional_ensemble) + len(result.unclassified) == 5

    def test_full_analysis_regime_only(self, synthetic_data):
        features, target, conditions = synthetic_data
        analyzer = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=10, verbose=False)
        results = analyzer.full_analysis(features, target, regime_labels=conditions)

        assert ConditionPass.REGIME in results
        assert ConditionPass.TEMPORAL not in results

    def test_full_analysis_both(self, synthetic_data, time_labels):
        features, target, conditions = synthetic_data
        analyzer = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=10, verbose=False)
        results = analyzer.full_analysis(features, target, regime_labels=conditions, time_labels=time_labels)

        assert ConditionPass.REGIME in results
        assert ConditionPass.TEMPORAL in results

    def test_full_analysis_no_labels_raises(self, synthetic_data):
        features, target, _ = synthetic_data
        analyzer = ConditionalFeatureAnalyzer(verbose=False)
        with pytest.raises(ValueError, match="At least one"):
            analyzer.full_analysis(features, target)

    def test_hierarchical_validates_separator(self, synthetic_data):
        features, target, conditions = synthetic_data
        bad_labels = conditions.map(lambda x: f"{x}|bad")
        analyzer = ConditionalFeatureAnalyzer(verbose=False)
        with pytest.raises(ValueError, match="reserved separator"):
            analyzer.analyze_hierarchical(features, target, bad_labels, conditions)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test: FeatureConditionMap methods
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureConditionMap:
    @pytest.fixture
    def fcm(self, synthetic_data):
        features, target, conditions = synthetic_data
        analyzer = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=10, verbose=False)
        return analyzer.analyze_by_regime(features, target, conditions)

    def test_active_features_for(self, fcm):
        active = fcm.active_features_for("A")
        assert isinstance(active, list)
        assert "feat_global" in active

    def test_inverted_features_for(self, fcm):
        inverted = fcm.inverted_features_for("B")
        assert isinstance(inverted, list)

    def test_sign_flipping_features(self, fcm):
        flippers = fcm.sign_flipping_features()
        assert isinstance(flippers, list)
        # feat_flip should be a sign flipper
        assert "feat_flip" in flippers

    def test_feature_profile(self, fcm):
        profile = fcm.feature_profile("feat_global")
        assert "condition" in profile.columns
        assert "ic" in profile.columns
        assert "status" in profile.columns
        assert len(profile) == 3

    def test_condition_profile(self, fcm):
        profile = fcm.condition_profile("A")
        assert "feature" in profile.columns
        assert len(profile) == 5

    def test_summary(self, fcm):
        s = fcm.summary()
        assert "metric" in s.columns
        assert "value" in s.columns
        assert len(s) > 0

    def test_to_mask(self, fcm):
        mask = fcm.to_mask()
        assert mask.shape == fcm.status_matrix.shape
        assert set(mask.values.flatten()) <= {0, 1}

    def test_to_signed_mask(self, fcm):
        mask = fcm.to_signed_mask()
        assert mask.shape == fcm.status_matrix.shape
        assert set(mask.values.flatten()) <= {-1, 0, 1}

    def test_unknown_condition_raises(self, fcm):
        with pytest.raises(KeyError):
            fcm.active_features_for("UNKNOWN")

    def test_unknown_feature_raises(self, fcm):
        with pytest.raises(KeyError):
            fcm.feature_profile("UNKNOWN")

    def test_repr(self, fcm):
        r = repr(fcm)
        assert "FeatureConditionMap" in r
        assert "regime" in r


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_load_roundtrip(self, synthetic_data):
        features, target, conditions = synthetic_data
        analyzer = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=10, verbose=False)
        original = analyzer.analyze_by_regime(features, target, conditions)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = original.save(str(Path(tmpdir) / "test_map"))
            loaded = FeatureConditionMap.load(str(save_path))

            # Check core data
            pd.testing.assert_frame_equal(original.ic_matrix, loaded.ic_matrix)
            pd.testing.assert_frame_equal(original.pvalue_matrix, loaded.pvalue_matrix)
            pd.testing.assert_frame_equal(original.subperiod_ic_std, loaded.subperiod_ic_std, check_dtype=False)

            # Check partition
            assert original.global_stable == loaded.global_stable
            assert original.condition_specific == loaded.condition_specific
            assert original.conditional_ensemble == loaded.conditional_ensemble
            assert original.unclassified == loaded.unclassified

            # Check scalars
            assert original.ic_threshold == loaded.ic_threshold
            assert original.stability_threshold == loaded.stability_threshold
            assert original.condition_pass == loaded.condition_pass
            assert original.conditional_structure_detected == loaded.conditional_structure_detected

    def test_load_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            FeatureConditionMap.load("/nonexistent/path")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TestEnums:
    def test_feature_status_values(self):
        assert FeatureStatus.ACTIVE == "active"
        assert FeatureStatus.NEGATIVE == "negative"

    def test_feature_bucket_values(self):
        assert FeatureBucket.GLOBAL_STABLE == "global_stable"

    def test_condition_pass_values(self):
        assert ConditionPass.REGIME == "regime"
        assert ConditionPass.HIERARCHICAL == "hierarchical"