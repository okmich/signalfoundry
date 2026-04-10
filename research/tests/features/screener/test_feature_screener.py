"""
Tests for okmich_quant_research.features.screener.FeatureScreener

Strategy: use small synthetic datasets so tests run quickly.
  - n=300 bars, 8 features
  - feat_0, feat_1: genuinely predictive (correlated with labels)
  - feat_2, feat_3: pure noise
  - feat_4: near-constant (Stage 0 target)
  - feat_5: duplicate of feat_0 (Stage 3 redundancy target)
  - feat_6, feat_7: mild signal

Stages are tested in isolation (unit) and end-to-end (integration).
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.screener import (
    FeatureScreener,
    ScreenerResult,
    StageReport,
)
from okmich_quant_research.features.screener._stage0 import stage0_variance_filter
from okmich_quant_research.features.screener._stage1_regime import stage1_regime
from okmich_quant_research.features.screener._stage1_return import stage1_return
from okmich_quant_research.features.screener._stage2 import stage2_temporal_stability
from okmich_quant_research.features.screener._stage3 import stage3_redundancy
from okmich_quant_research.features.screener._stage4 import stage4_boruta
from okmich_quant_research.features.screener._stage5 import stage5_model_ranking
from okmich_quant_research.features.screener._result import ScreenerResult, StageReport


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_regime_data():
    """Small dataset: 300 bars, 3 regimes, 8 features."""
    np.random.seed(0)
    n = 300
    regime = pd.Series(np.random.choice([0, 1, 2], size=n))
    X = pd.DataFrame({
        "feat_0":  regime * 0.8 + np.random.randn(n) * 0.2,   # strong signal
        "feat_1": -regime * 0.5 + np.random.randn(n) * 0.3,   # moderate signal
        "feat_2":  np.random.randn(n),                          # pure noise
        "feat_3":  np.random.randn(n),                          # pure noise
        "feat_4":  1.0 + np.random.randn(n) * 1e-7,           # near-constant
        "feat_5":  regime * 0.8 + np.random.randn(n) * 0.2,   # duplicate of feat_0
        "feat_6":  regime * 0.2 + np.random.randn(n) * 0.8,   # weak signal
        "feat_7":  np.random.randn(n),                          # noise
    })
    return X, regime


@pytest.fixture(scope="module")
def synthetic_return_data():
    """Small dataset: 300 bars, continuous forward returns, 6 features."""
    np.random.seed(1)
    n = 300
    X = pd.DataFrame({
        "mom":     np.random.randn(n).cumsum() * 0.01,
        "vol":     np.abs(np.random.randn(n)) + 0.5,
        "noise_a": np.random.randn(n),
        "noise_b": np.random.randn(n),
        "const":   0.5 + np.random.randn(n) * 1e-7,
        "signal":  np.random.randn(n),
    })
    # Forward returns correlated with "signal" feature
    forward_returns = pd.Series(
        0.4 * X["signal"].shift(1).fillna(0).values + np.random.randn(n) * 0.1
    )
    return X, forward_returns


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Variance filter
# ─────────────────────────────────────────────────────────────────────────────

class TestStage0:
    def test_removes_near_constant(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        X_out, report = stage0_variance_filter(X, verbose=False)
        assert "feat_4" not in X_out.columns

    def test_keeps_signal_features(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        X_out, _ = stage0_variance_filter(X, verbose=False)
        assert "feat_0" in X_out.columns
        assert "feat_1" in X_out.columns

    def test_report_counts(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        X_out, report = stage0_variance_filter(X, verbose=False)
        assert report.n_before == X.shape[1]
        assert report.n_after == X_out.shape[1]
        assert report.n_removed == report.n_before - report.n_after
        assert "feat_4" in report.removed

    def test_all_constant_removed(self):
        X = pd.DataFrame({"a": [1.0] * 50, "b": np.random.randn(50)})
        X_out, report = stage0_variance_filter(X, verbose=False)
        assert "a" not in X_out.columns
        assert "b" in X_out.columns

    def test_report_stage_name(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        _, report = stage0_variance_filter(X, verbose=False)
        assert report.stage == "Stage0_Variance"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Hygiene (regime)
# ─────────────────────────────────────────────────────────────────────────────

class TestStage1Regime:
    def test_keeps_predictive_features(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_out, report, scores = stage1_regime(X, regime, verbose=False)
        assert "feat_0" in X_out.columns

    def test_removes_noise(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        # Use strict thresholds to force noise removal
        X_out, report, scores = stage1_regime(
            X, regime, mi_threshold=0.05, ks_threshold=0.15, verbose=False
        )
        removed = set(report.removed)
        # At least one noise feature should be removed
        assert len(removed) > 0

    def test_scores_contain_all_features(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        _, _, scores = stage1_regime(X, regime, verbose=False)
        assert set(scores["mi"].index) == set(X.columns)
        assert set(scores["ks"].index) == set(X.columns)

    def test_mi_higher_for_signal(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        _, _, scores = stage1_regime(X, regime, verbose=False)
        assert scores["mi"]["feat_0"] > scores["mi"]["feat_2"]

    def test_ks_higher_for_signal(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        _, _, scores = stage1_regime(X, regime, verbose=False)
        assert scores["ks"]["feat_0"] > scores["ks"]["feat_2"]

    def test_report_stage_name(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        _, report, _ = stage1_regime(X, regime, verbose=False)
        assert report.stage == "Stage1_Hygiene_Regime"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Hygiene (return)
# ─────────────────────────────────────────────────────────────────────────────

class TestStage1Return:
    def test_runs_without_error(self, synthetic_return_data):
        X, y = synthetic_return_data
        X_out, report, scores = stage1_return(X, y, verbose=False)
        assert isinstance(X_out, pd.DataFrame)

    def test_scores_contain_all_features(self, synthetic_return_data):
        X, y = synthetic_return_data
        _, _, scores = stage1_return(X, y, verbose=False)
        assert set(scores["mi"].index) == set(X.columns)
        assert set(scores["dcor"].index) == set(X.columns)

    def test_dcor_scores_non_negative(self, synthetic_return_data):
        X, y = synthetic_return_data
        _, _, scores = stage1_return(X, y, verbose=False)
        assert (scores["dcor"] >= 0).all()

    def test_dcor_bounded_at_one(self, synthetic_return_data):
        X, y = synthetic_return_data
        _, _, scores = stage1_return(X, y, verbose=False)
        assert (scores["dcor"] <= 1.0 + 1e-8).all()

    def test_report_stage_name(self, synthetic_return_data):
        X, y = synthetic_return_data
        _, report, _ = stage1_return(X, y, verbose=False)
        assert report.stage == "Stage1_Hygiene_Return"

    def test_handles_nan_in_y(self, synthetic_return_data):
        X, y = synthetic_return_data
        y_with_nan = y.copy()
        y_with_nan.iloc[-20:] = np.nan   # simulate shift() tail NaN
        X_out, report, scores = stage1_return(X, y_with_nan, verbose=False)
        assert isinstance(X_out, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Temporal stability (IC-IR)
# ─────────────────────────────────────────────────────────────────────────────

class TestStage2:
    def test_icir_higher_for_signal(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        _, _, icir = stage2_temporal_stability(
            X[["feat_0", "feat_2"]], regime, task="regime",
            window=50, min_icir=0.0, verbose=False
        )
        assert icir["feat_0"] > icir["feat_2"]

    def test_removes_low_icir(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_2"]]
        _, _, icir_all = stage2_temporal_stability(
            X_in, regime, task="regime", window=50, min_icir=0.0, verbose=False
        )
        # Set threshold just above feat_2's IC-IR to force its removal
        threshold = icir_all["feat_2"] + 0.1
        X_out, report, _ = stage2_temporal_stability(
            X_in, regime, task="regime", window=50, min_icir=threshold, verbose=False
        )
        assert "feat_2" not in X_out.columns

    def test_icir_dict_covers_all_inputs(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1", "feat_2"]]
        _, _, icir = stage2_temporal_stability(
            X_in, regime, task="regime", window=50, min_icir=0.0, verbose=False
        )
        assert set(icir.keys()) == set(X_in.columns)

    def test_report_stage_name(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        _, report, _ = stage2_temporal_stability(
            X, regime, task="regime", window=50, verbose=False
        )
        assert report.stage == "Stage2_TemporalStability"

    def test_return_task_runs(self, synthetic_return_data):
        X, y = synthetic_return_data
        X_in = X[["signal", "noise_a"]]
        X_out, report, icir = stage2_temporal_stability(
            X_in, y, task="return", window=50, min_icir=0.0, verbose=False
        )
        assert isinstance(icir, dict)
        assert "signal" in icir

    def test_walk_forward_pct_zero_disables_check(self, synthetic_regime_data):
        """walk_forward_pct=0.0 disables the hit-rate filter (backward compat)."""
        X, regime = synthetic_regime_data
        # With min_icir=0.0 and walk_forward_pct=0.0, all features pass
        X_out, report, _ = stage2_temporal_stability(
            X[["feat_0", "feat_2"]], regime, task="regime",
            window=50, min_icir=0.0, walk_forward_pct=0.0, verbose=False
        )
        assert report.n_after == 2

    def test_walk_forward_pct_removes_inconsistent(self):
        """Strict hit-rate threshold removes an anti-predictive feature (return task)."""
        np.random.seed(42)
        n = 500
        # 'good': consistently positively correlated with forward returns
        # 'anti': consistently negatively correlated with forward returns
        base = np.random.randn(n)
        y = pd.Series(0.6 * base + np.random.randn(n) * 0.1)
        X = pd.DataFrame({
            "good": base,               # IC > 0 in most windows
            "anti": -base,              # IC < 0 in most windows (anti-predictive)
        })
        # With walk_forward_pct=0.70: 'anti' fails because <70% of IC windows > 0
        X_strict, report, _ = stage2_temporal_stability(
            X, y, task="return", window=60, min_icir=0.0,
            walk_forward_pct=0.70, verbose=False,
        )
        assert "good" in X_strict.columns
        assert "anti" not in X_strict.columns


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Redundancy reduction
# ─────────────────────────────────────────────────────────────────────────────

class TestStage3:
    def test_removes_duplicate(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        # feat_0 and feat_5 are near-identical
        X_in = X[["feat_0", "feat_1", "feat_5"]]
        icir = {"feat_0": 2.0, "feat_1": 1.0, "feat_5": 1.5}
        X_out, report = stage3_redundancy(X_in, icir, corr_threshold=0.80, verbose=False)
        # feat_5 should be absorbed into feat_0's cluster
        assert "feat_5" not in X_out.columns

    def test_keeps_best_icir(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        X_in = X[["feat_0", "feat_5"]]  # near-identical
        # feat_5 has higher IC-IR — it should be kept
        icir = {"feat_0": 1.0, "feat_5": 3.0}
        X_out, report = stage3_redundancy(X_in, icir, corr_threshold=0.80, verbose=False)
        assert "feat_5" in X_out.columns
        assert "feat_0" not in X_out.columns

    def test_single_feature_passthrough(self):
        X = pd.DataFrame({"a": np.random.randn(100)})
        X_out, report = stage3_redundancy(X, {"a": 1.0}, verbose=False)
        assert "a" in X_out.columns
        assert report.n_removed == 0

    def test_uncorrelated_features_all_kept(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "a": np.random.randn(200),
            "b": np.random.randn(200),
        })
        X_out, report = stage3_redundancy(X, {"a": 1.0, "b": 1.0},
                                           corr_threshold=0.75, verbose=False)
        assert "a" in X_out.columns
        assert "b" in X_out.columns

    def test_report_stage_name(self, synthetic_regime_data):
        X, _ = synthetic_regime_data
        X_in = X[["feat_0", "feat_1"]]
        _, report = stage3_redundancy(X_in, {"feat_0": 1.0, "feat_1": 1.0},
                                      verbose=False)
        assert report.stage == "Stage3_Redundancy"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Boruta
# ─────────────────────────────────────────────────────────────────────────────

class TestStage4:
    def test_returns_boruta_groups(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1", "feat_2"]]
        _, report, groups = stage4_boruta(
            X_in, regime, task="regime", max_iter=20, verbose=False
        )
        assert "confirmed" in groups
        assert "tentative" in groups
        assert "rejected" in groups

    def test_all_features_classified(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1", "feat_2"]]
        _, _, groups = stage4_boruta(
            X_in, regime, task="regime", max_iter=20, verbose=False
        )
        all_classified = set(groups["confirmed"]) | set(groups["tentative"]) | set(groups["rejected"])
        assert all_classified == set(X_in.columns)

    def test_strong_signal_confirmed(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        # feat_0 has very strong signal; should almost always be confirmed
        X_in = X[["feat_0", "feat_2"]]
        _, _, groups = stage4_boruta(
            X_in, regime, task="regime", max_iter=30, verbose=False
        )
        assert "feat_0" in groups["confirmed"] or "feat_0" in groups["tentative"]

    def test_report_stage_name(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1"]]
        _, report, _ = stage4_boruta(X_in, regime, task="regime", max_iter=15, verbose=False)
        assert report.stage == "Stage4_Boruta"

    def test_return_task_runs(self, synthetic_return_data):
        X, y = synthetic_return_data
        X_in = X[["signal", "noise_a"]].fillna(0)
        y_clean = y.fillna(0)
        _, _, groups = stage4_boruta(X_in, y_clean, task="return", max_iter=15, verbose=False)
        assert isinstance(groups["confirmed"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Model ranking
# ─────────────────────────────────────────────────────────────────────────────

class TestStage5:
    def test_returns_two_series(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1", "feat_2"]]
        shap_rank, mda_rank = stage5_model_ranking(
            X_in, regime, task="regime", n_splits=2, verbose=False
        )
        assert isinstance(shap_rank, pd.Series)
        assert isinstance(mda_rank, pd.Series)

    def test_shap_covers_all_features(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1", "feat_2"]]
        shap_rank, _ = stage5_model_ranking(
            X_in, regime, task="regime", n_splits=2, verbose=False
        )
        assert set(shap_rank.index) == set(X_in.columns)

    def test_shap_non_negative(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1"]]
        shap_rank, _ = stage5_model_ranking(
            X_in, regime, task="regime", n_splits=2, verbose=False
        )
        assert (shap_rank >= 0).all()

    def test_shap_descending_order(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_1", "feat_2"]]
        shap_rank, _ = stage5_model_ranking(
            X_in, regime, task="regime", n_splits=2, verbose=False
        )
        assert list(shap_rank.values) == sorted(shap_rank.values, reverse=True)

    def test_signal_ranks_above_noise(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        X_in = X[["feat_0", "feat_2"]]
        shap_rank, _ = stage5_model_ranking(
            X_in, regime, task="regime", n_splits=2, verbose=False
        )
        assert shap_rank["feat_0"] > shap_rank["feat_2"]

    def test_return_task_uses_purged_kfold(self, synthetic_return_data):
        X, y = synthetic_return_data
        X_in = X[["signal", "noise_a"]]
        shap_rank, mda_rank = stage5_model_ranking(
            X_in, y, task="return", n_splits=2, horizon=5, verbose=False
        )
        assert set(shap_rank.index) == set(X_in.columns)


# ─────────────────────────────────────────────────────────────────────────────
# ScreenerResult
# ─────────────────────────────────────────────────────────────────────────────

class TestScreenerResult:
    @pytest.fixture
    def sample_result(self):
        return ScreenerResult(
            confirmed=["feat_0", "feat_1"],
            tentative=["feat_6"],
            rejected=["feat_2", "feat_3", "feat_4", "feat_5", "feat_7"],
            shap_rank=pd.Series({"feat_0": 0.7, "feat_1": 0.2, "feat_6": 0.1}),
            mda_rank=pd.Series({"feat_0": 0.5, "feat_1": 0.3, "feat_6": 0.05}),
            stage_reports=[
                StageReport("Stage0_Variance", 8, 7, ["feat_4"]),
                StageReport("Stage1_Hygiene_Regime", 7, 4, ["feat_2", "feat_3", "feat_7"]),
                StageReport("Stage2_TemporalStability", 4, 4, []),
                StageReport("Stage3_Redundancy", 4, 3, ["feat_5"]),
                StageReport("Stage4_Boruta", 3, 3, []),
            ],
            icir_scores={"feat_0": 5.2, "feat_1": 3.1, "feat_6": 0.8},
        )

    def test_repr(self, sample_result):
        r = repr(sample_result)
        assert "confirmed=2" in r
        assert "tentative=1" in r

    def test_summary_shape(self, sample_result):
        df = sample_result.summary()
        assert len(df) == 5
        assert "stage" in df.columns
        assert "n_removed" in df.columns

    def test_summary_n_removed_correct(self, sample_result):
        df = sample_result.summary()
        assert df.loc[df.stage == "Stage0_Variance", "n_removed"].iloc[0] == 1

    def test_top_features_returns_dataframe(self, sample_result):
        df = sample_result.top_features()
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "shap" in df.columns
        assert "boruta_status" in df.columns

    def test_top_features_sorted_by_shap(self, sample_result):
        df = sample_result.top_features()
        assert df.iloc[0]["feature"] == "feat_0"

    def test_top_features_n_limit(self, sample_result):
        df = sample_result.top_features(n=2)
        assert len(df) <= 2

    def test_boruta_status_correct(self, sample_result):
        df = sample_result.top_features()
        confirmed_rows = df[df.feature.isin(["feat_0", "feat_1"])]
        assert (confirmed_rows.boruta_status == "confirmed").all()
        tentative_rows = df[df.feature == "feat_6"]
        assert (tentative_rows.boruta_status == "tentative").all()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    @pytest.fixture(scope="class")
    def regime_result(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        screener = FeatureScreener(
            ic_window=50,
            boruta_max_iter=20,
            n_cv_splits=3,
            verbose=False,
        )
        return screener.screen_for_regimes(X, regime)

    def test_returns_screener_result(self, regime_result):
        assert isinstance(regime_result, ScreenerResult)

    def test_confirmed_is_list(self, regime_result):
        assert isinstance(regime_result.confirmed, list)

    def test_tentative_is_list(self, regime_result):
        assert isinstance(regime_result.tentative, list)

    def test_shap_rank_is_series(self, regime_result):
        assert isinstance(regime_result.shap_rank, pd.Series)

    def test_mda_rank_is_series(self, regime_result):
        assert isinstance(regime_result.mda_rank, pd.Series)

    def test_stage_reports_present(self, regime_result):
        # Stage0_PrefixDedup + Stage0_Variance + Stage1 + Stage2 + Stage3 + Stage4 = 6
        assert len(regime_result.stage_reports) == 6

    def test_feat4_not_in_confirmed(self, regime_result):
        assert "feat_4" not in regime_result.confirmed

    def test_signal_features_in_confirmed_or_tentative(self, regime_result):
        survivors = set(regime_result.confirmed) | set(regime_result.tentative)
        # feat_0 and feat_5 are duplicates — at least one should survive clustering
        # feat_1 also has strong signal; at minimum one predictive feature survives
        predictive = {"feat_0", "feat_1", "feat_5", "feat_6"}
        assert len(survivors & predictive) >= 1

    def test_summary_dataframe(self, regime_result):
        df = regime_result.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 4

    def test_top_features_includes_signal(self, regime_result):
        tf = regime_result.top_features()
        predictive = {"feat_0", "feat_1", "feat_5", "feat_6"}
        assert len(set(tf.feature.values) & predictive) >= 1

    def test_starting_features_filter(self, synthetic_regime_data):
        X, regime = synthetic_regime_data
        screener = FeatureScreener(ic_window=50, boruta_max_iter=15,
                                   n_cv_splits=2, verbose=False)
        result = screener.screen_for_regimes(
            X, regime,
            starting_features=["feat_0", "feat_1", "feat_4"],
        )
        # feat_4 is near-constant — should be removed
        assert "feat_4" not in result.confirmed

    def test_return_pipeline_runs(self, synthetic_return_data):
        X, y = synthetic_return_data
        screener = FeatureScreener(
            ic_window=50, boruta_max_iter=15,
            n_cv_splits=2, verbose=False,
        )
        result = screener.screen_for_returns(X, y, horizon=5)
        assert isinstance(result, ScreenerResult)
        assert isinstance(result.shap_rank, pd.Series)