"""
Regression tests for the 2026-10-15 conditioned-package fixes:
  - create_report() crashed (passed show= to plot methods that did not accept it)
  - save() failed on integer condition labels (parquet column names, JSON keys)
  - partition: dominant condition could be a non-significant one; sign-flippers could be condition_specific
  - permutation null: iid shuffle too narrow on persistent series -> circular-shift default
"""

import inspect
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from okmich_quant_research.features.conditioned import ConditionalFeatureAnalyzer, ConditionedFeatureVisualizer, \
    FeatureConditionMap, NullMethod
from okmich_quant_research.features.conditioned._enums import FeatureStatus
from okmich_quant_research.features.conditioned._ic_analysis import compute_conditional_ic, compute_ic_threshold
from okmich_quant_research.features.conditioned._partition import partition_features
from okmich_quant_research.features.conditioned._result import assign_status

A, I, N = FeatureStatus.ACTIVE, FeatureStatus.INACTIVE, FeatureStatus.NEGATIVE


def _make_data(labels: list, n_per_condition: int = 600, seed: int = 3):
    """Features with known structure over len(labels) contiguous condition blocks."""
    rng = np.random.default_rng(seed)
    k = len(labels)
    n = n_per_condition * k
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    target = rng.standard_normal(n)
    first = slice(0, n_per_condition)

    feats = pd.DataFrame(index=idx)
    feats["feat_global"] = target * 0.3 + rng.standard_normal(n) * 0.7
    feat_first = rng.standard_normal(n)
    feat_first[first] = target[first] * 0.5 + rng.standard_normal(n_per_condition) * 0.5
    feats["feat_first_only"] = feat_first
    feats["feat_noise"] = rng.standard_normal(n)

    cond = pd.Series(np.repeat(np.asarray(labels), n_per_condition), index=idx, name="regime")
    return feats, pd.Series(target, index=idx, name="target"), cond


# ─────────────────────────────────────────────────────────────────────────────
# create_report / show plumbing
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateReport:
    def test_every_plot_method_accepts_show(self):
        plot_methods = [m for name, m in inspect.getmembers(ConditionedFeatureVisualizer) if name.startswith("plot_")]
        assert len(plot_methods) == 10
        for method in plot_methods:
            assert "show" in inspect.signature(method).parameters, method.__name__

    def test_create_report_writes_all_files(self):
        feats, target, cond = _make_data(["A", "B", "C"])
        fcm = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=5, verbose=False).analyze_by_regime(
            feats, target, cond)

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ConditionedFeatureVisualizer.create_report(fcm, output_dir=tmpdir, show=False)
            assert set(paths) == {"ic_heatmap", "status_grid", "partition_summary", "stability_distribution",
                                  "subperiod_stability", "condition_counts", "summary"}
            for path in paths.values():
                assert Path(path).is_file(), path


# ─────────────────────────────────────────────────────────────────────────────
# Persistence with integer labels
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegerLabelPersistence:
    @pytest.fixture(scope="class")
    def int_fcm(self):
        feats, target, cond = _make_data([0, 1, 2])
        return ConditionalFeatureAnalyzer(min_observations=100, n_permutations=5, verbose=False).analyze_by_regime(
            feats, target, cond)

    def test_round_trip_restores_int_labels(self, int_fcm):
        assert int_fcm.condition_specific, "fixture should put feat_first_only in condition_specific"

        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = FeatureConditionMap.load(str(int_fcm.save(str(Path(tmpdir) / "int_map"))))

        pd.testing.assert_frame_equal(int_fcm.ic_matrix, loaded.ic_matrix)
        pd.testing.assert_frame_equal(int_fcm.pvalue_matrix, loaded.pvalue_matrix)
        pd.testing.assert_frame_equal(int_fcm.subperiod_ic_std, loaded.subperiod_ic_std, check_dtype=False)
        assert (int_fcm.status_matrix == loaded.status_matrix).all().all()
        assert list(loaded.status_matrix.columns) == [0, 1, 2]
        assert all(isinstance(c, int) for c in loaded.ic_matrix.columns)

        assert loaded.condition_specific == int_fcm.condition_specific
        assert all(isinstance(k, int) for k in loaded.condition_specific)
        assert loaded.skipped_conditions == int_fcm.skipped_conditions
        assert list(loaded.condition_sample_counts.index) == [0, 1, 2]
        assert loaded.null_method == NullMethod.CIRCULAR_SHIFT and loaded.null_min_shift == 500

        # Lookups by the original integer label keep working after a reload
        assert loaded.active_features_for(0) == int_fcm.active_features_for(0)

    def test_loads_map_saved_before_label_kind_existed(self):
        feats, target, cond = _make_data(["A", "B", "C"])
        fcm = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=5, verbose=False).analyze_by_regime(
            feats, target, cond)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = fcm.save(str(Path(tmpdir) / "old_map"))
            meta_path = out / "metadata.json"
            meta = json.loads(meta_path.read_text())
            for key in ("condition_label_kind", "null_method", "null_min_shift"):
                meta.pop(key)
            meta_path.write_text(json.dumps(meta))

            loaded = FeatureConditionMap.load(str(out))

        pd.testing.assert_frame_equal(fcm.ic_matrix, loaded.ic_matrix)
        assert loaded.condition_specific == fcm.condition_specific
        assert loaded.null_method is None


# ─────────────────────────────────────────────────────────────────────────────
# Partition: dominant condition must be significant; sign flips are not condition_specific
# ─────────────────────────────────────────────────────────────────────────────

def _partition(ics: dict, statuses: dict, conditions=("A", "B", "C")):
    ic = pd.DataFrame(ics, index=list(conditions)).T
    status = pd.DataFrame(statuses, index=list(conditions)).T
    stability = pd.Series(np.nan, index=ic.index)  # NaN -> never global_stable, so step 3 is exercised directly
    subperiod = pd.DataFrame(np.nan, index=ic.index, columns=ic.columns)
    return partition_features(ic, status, stability, subperiod, ic_threshold=0.05, stability_threshold=0.5)


class TestPartitionFixes:
    def test_dominant_condition_is_never_a_non_significant_one(self):
        # Largest |IC| sits in A, which failed the p-value test; B is the only significant condition
        _, cs, ce, uc, _ = _partition({"f": [0.30, 0.20, 0.01]}, {"f": [I, A, I]})
        assert "f" not in cs.get("A", [])
        assert "f" in uc  # B does not dominate A (0.20 < 2 * 0.30): no robust bucket

    def test_significant_dominant_condition_is_kept(self):
        _, cs, _, _, _ = _partition({"f": [0.05, 0.30, 0.02]}, {"f": [I, A, I]})
        assert cs == {"B": ["f"]}

    def test_sign_flipper_goes_to_conditional_ensemble(self):
        # +0.30 in A passes 3x dominance over -0.10 in B, but B is significant with the opposite sign
        _, cs, ce, _, _ = _partition({"f": [0.30, -0.10, 0.0]}, {"f": [A, N, I]})
        assert cs == {}
        assert ce == ["f"]

    def test_same_sign_second_condition_still_condition_specific(self):
        _, cs, ce, _, _ = _partition({"f": [0.30, 0.10, 0.0]}, {"f": [A, A, I]})
        assert cs == {"A": ["f"]} and ce == []

    def test_single_computed_and_zero_others_unchanged(self):
        ins = FeatureStatus.INSUFFICIENT_DATA
        _, cs, _, _, _ = _partition({"single": [np.nan, 0.25, np.nan], "zeros": [0.0, 0.0, -0.4]},
                                    {"single": [ins, A, ins], "zeros": [I, I, N]})
        assert cs == {"B": ["single"], "C": ["zeros"]}


# ─────────────────────────────────────────────────────────────────────────────
# Permutation null under serial dependence
# ─────────────────────────────────────────────────────────────────────────────

class TestCircularShiftNull:
    @pytest.fixture(scope="class")
    def ar1_data(self):
        rng = np.random.default_rng(7)
        n, phi, k = 20000, 0.99, 10

        def ar1():
            return lfilter([1.0], [1.0, -phi], rng.standard_normal(n))

        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        feats = pd.DataFrame({f"f{i}": ar1() for i in range(k)}, index=idx)  # all independent of the target
        target = pd.Series(ar1(), index=idx, name="target")
        cond = pd.Series(np.where((np.arange(n) // 2500) % 2 == 0, 0, 1), index=idx, name="regime")
        return feats, target, cond

    @staticmethod
    def _n_false_passes(feats, target, cond, threshold):
        ic, pv, obs = compute_conditional_ic(feats, target, cond, min_observations=500)
        status = assign_status(ic, pv, obs, threshold, max_pvalue=0.05, min_observations=500)
        return int(((status == A) | (status == N)).any(axis=1).sum())

    def test_circular_null_prices_in_persistence(self, ar1_data):
        feats, target, cond = ar1_data
        kwargs = dict(n_permutations=10, min_observations=500, feature_subsample_ratio=1.0)
        thr_iid = compute_ic_threshold(feats, target, cond, null_method=NullMethod.IID_SHUFFLE, **kwargs)
        thr_circ = compute_ic_threshold(feats, target, cond, null_method=NullMethod.CIRCULAR_SHIFT, **kwargs)

        assert thr_circ > 3 * thr_iid
        assert self._n_false_passes(feats, target, cond, thr_iid) >= 7  # iid: most independent features "work"
        assert self._n_false_passes(feats, target, cond, thr_circ) <= 4

    def test_null_method_validation(self, ar1_data):
        feats, target, cond = ar1_data
        small = dict(n_permutations=1, min_observations=500, feature_subsample_ratio=0.1)
        assert compute_ic_threshold(feats, target, cond, null_method="circular_shift", **small) > 0
        with pytest.raises(ValueError):
            compute_ic_threshold(feats, target, cond, null_method="bogus", **small)
        with pytest.raises(ValueError):
            compute_ic_threshold(feats, target, cond, min_shift=0, **small)

    def test_analyzer_records_null_provenance(self):
        feats, target, cond = _make_data(["A", "B", "C"])
        fcm = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=5, verbose=False).analyze_by_regime(
            feats, target, cond)
        assert fcm.null_method == NullMethod.CIRCULAR_SHIFT and fcm.null_min_shift == 500

        iid = ConditionalFeatureAnalyzer(min_observations=100, n_permutations=5, verbose=False,
                                         null_method=NullMethod.IID_SHUFFLE).analyze_by_regime(feats, target, cond)
        assert iid.null_method == NullMethod.IID_SHUFFLE and iid.null_min_shift is None

        fixed = ConditionalFeatureAnalyzer(min_observations=100, ic_threshold_override=0.05,
                                           verbose=False).analyze_by_regime(feats, target, cond)
        assert fixed.null_method is None and fixed.null_min_shift is None
