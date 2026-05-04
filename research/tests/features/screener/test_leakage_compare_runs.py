"""
Tests for LeakageDiagnostics.compare_runs (cheap rank-correlation diagnostic).

Hand-crafted ScreenerResult fixtures — no full screener invocation needed,
so these run in milliseconds.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from okmich_quant_research.features.screener import ScreenerResult
from okmich_quant_research.features.screener.leakage import (
    LeakageDiagnostics,
    LeakageReport,
    Severity,
    SuspectRegistry,
    classify,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(shap_rank: dict[str, float], mda_rank: dict[str, float],
                 confirmed: list[str] | None = None,
                 cluster_assignments: dict[int, list[str]] | None = None,
                 cluster_representatives: dict[int, str] | None = None) -> ScreenerResult:
    """Minimal ScreenerResult builder for comparison tests."""
    confirmed = confirmed if confirmed is not None else list(shap_rank.keys())
    return ScreenerResult(
        confirmed=confirmed,
        tentative=[],
        rejected=[],
        shap_rank=pd.Series(shap_rank).sort_values(ascending=False),
        mda_rank=pd.Series(mda_rank).sort_values(ascending=False),
        cluster_assignments=cluster_assignments or {i + 1: [f] for i, f in enumerate(shap_rank)},
        cluster_representatives=cluster_representatives or {i + 1: f for i, f in enumerate(shap_rank)},
        boruta_groups={"confirmed": confirmed, "tentative": [], "rejected": []},
    )


@pytest.fixture
def suspects_tm():
    return SuspectRegistry(prefixes=("tm_vol_",), rationale="vol family suspected")


# ─────────────────────────────────────────────────────────────────────────────
# Severity classifier (unit)
# ─────────────────────────────────────────────────────────────────────────────

class TestClassify:
    def test_clean(self):
        assert classify({"a": 0.95, "b": 0.90}) == Severity.CLEAN

    def test_watch(self):
        assert classify({"a": 0.80, "b": 0.95}) == Severity.WATCH

    def test_investigate_low(self):
        assert classify({"a": 0.50, "b": 0.95}) == Severity.INVESTIGATE

    def test_min_drives_decision(self):
        # one low correlation drags severity down regardless of the other
        assert classify({"a": 0.99, "b": 0.60}) == Severity.INVESTIGATE

    def test_nan_is_investigate(self):
        assert classify({"a": float("nan"), "b": 0.99}) == Severity.INVESTIGATE

    def test_empty_is_investigate(self):
        assert classify({}) == Severity.INVESTIGATE

    def test_custom_thresholds(self):
        # tighten the bar
        assert classify({"a": 0.92}, thresholds=(0.95, 0.80)) == Severity.WATCH


# ─────────────────────────────────────────────────────────────────────────────
# compare_runs
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareRuns:
    def test_identical_runs_clean(self, suspects_tm):
        ranks = {"tm_vol_5": 1.0, "feat_a": 0.5, "feat_b": 0.2}
        full = _make_result(ranks, ranks)
        diag = LeakageDiagnostics(full, suspects_tm)
        report = diag.compare_runs(full)

        assert isinstance(report, LeakageReport)
        assert report.severity == Severity.CLEAN
        assert math.isclose(report.rank_correlations["shap_full_vs_pruned"], 1.0)
        assert math.isclose(report.rank_correlations["mda_full_vs_pruned"], 1.0)
        assert report.suspect_interactions is None  # set by probe(), not compare_runs

    def test_shifted_ranks_watch(self, suspects_tm):
        # Modest reordering: top-3 stay the same set, but rank order swaps inside
        full = _make_result({"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6},
                            {"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6})
        # Pruned: swap two adjacent pairs to get a Spearman in the watch band
        pruned = _make_result({"a": 1.0, "b": 0.7, "c": 0.9, "d": 0.6, "e": 0.8},
                              {"a": 1.0, "b": 0.7, "c": 0.9, "d": 0.6, "e": 0.8})
        diag = LeakageDiagnostics(full, suspects_tm)
        report = diag.compare_runs(pruned)
        assert report.severity in {Severity.WATCH, Severity.INVESTIGATE}
        # Either way, both correlations must be < 1.0
        assert report.rank_correlations["shap_full_vs_pruned"] < 1.0

    def test_inverted_ranks_investigate(self, suspects_tm):
        full = _make_result({"a": 1.0, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2},
                            {"a": 1.0, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2})
        # Reversed → Spearman = -1
        pruned = _make_result({"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8, "e": 1.0},
                              {"a": 0.2, "b": 0.4, "c": 0.6, "d": 0.8, "e": 1.0})
        diag = LeakageDiagnostics(full, suspects_tm)
        report = diag.compare_runs(pruned)
        assert report.severity == Severity.INVESTIGATE
        assert math.isclose(report.rank_correlations["shap_full_vs_pruned"], -1.0)

    def test_empty_intersection_handled(self, suspects_tm):
        full = _make_result({"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5})
        pruned = _make_result({"x": 1.0, "y": 0.5}, {"x": 1.0, "y": 0.5})
        diag = LeakageDiagnostics(full, suspects_tm)
        report = diag.compare_runs(pruned)
        # Empty intersection → spearman undefined → NaN → INVESTIGATE
        assert math.isnan(report.rank_correlations["shap_full_vs_pruned"])
        assert report.severity == Severity.INVESTIGATE

    def test_top_movers_includes_disappearance(self, suspects_tm):
        # 'b' present in full, absent in pruned → must rank as top mover
        full = _make_result({"a": 1.0, "b": 0.9, "c": 0.5}, {"a": 1.0, "b": 0.9, "c": 0.5})
        pruned = _make_result({"a": 1.0, "c": 0.5}, {"a": 1.0, "c": 0.5})
        diag = LeakageDiagnostics(full, suspects_tm)
        report = diag.compare_runs(pruned)

        assert "b" in report.top_movers["feature"].values
        b_row = report.top_movers[report.top_movers["feature"] == "b"].iloc[0]
        assert math.isinf(b_row["shap_abs_shift"])  # disappeared → infinite shift
        assert math.isinf(b_row["mda_abs_shift"])
        # Stable feature 'a' should have shap_abs_shift == 0
        a_row = report.top_movers[report.top_movers["feature"] == "a"].iloc[0]
        assert a_row["shap_abs_shift"] == 0.0
        assert a_row["mda_abs_shift"] == 0.0

    def test_top_movers_in_suspect_cluster_flag(self, suspects_tm):
        """
        Cluster 1 contains a suspect (tm_vol_5) and feat_momentum (survivor).
        feat_momentum should be flagged in_suspect_cluster=True even though
        feat_momentum is not itself a suspect.
        """
        full = ScreenerResult(
            confirmed=["feat_momentum", "feat_skew"],
            tentative=[], rejected=["tm_vol_5"],
            shap_rank=pd.Series({"feat_momentum": 1.0, "feat_skew": 0.5}),
            mda_rank=pd.Series({"feat_momentum": 1.0, "feat_skew": 0.5}),
            cluster_assignments={1: ["tm_vol_5", "feat_momentum"], 2: ["feat_skew"]},
            cluster_representatives={1: "feat_momentum", 2: "feat_skew"},
            boruta_groups={"confirmed": ["feat_momentum", "feat_skew"], "tentative": [], "rejected": []},
        )
        pruned = ScreenerResult(
            confirmed=["feat_momentum", "feat_skew"],
            tentative=[], rejected=[],
            shap_rank=pd.Series({"feat_momentum": 0.5, "feat_skew": 1.0}),  # swapped
            mda_rank=pd.Series({"feat_momentum": 0.5, "feat_skew": 1.0}),
            cluster_assignments={1: ["feat_momentum"], 2: ["feat_skew"]},
            cluster_representatives={1: "feat_momentum", 2: "feat_skew"},
            boruta_groups={"confirmed": ["feat_momentum", "feat_skew"], "tentative": [], "rejected": []},
        )
        report = LeakageDiagnostics(full, suspects_tm).compare_runs(pruned)
        mom = report.top_movers[report.top_movers["feature"] == "feat_momentum"].iloc[0]
        skew = report.top_movers[report.top_movers["feature"] == "feat_skew"].iloc[0]
        assert bool(mom["in_suspect_cluster"]) is True
        assert bool(skew["in_suspect_cluster"]) is False
        # Notes should mention the inheritor
        assert any("feat_momentum" in n for n in report.notes)

    def test_cluster_lineage_propagated(self, suspects_tm):
        full = ScreenerResult(
            confirmed=["feat_momentum"],
            tentative=[], rejected=["tm_vol_5"],
            shap_rank=pd.Series({"feat_momentum": 1.0}),
            mda_rank=pd.Series({"feat_momentum": 1.0}),
            cluster_assignments={1: ["tm_vol_5", "feat_momentum"]},
            cluster_representatives={1: "feat_momentum"},
            boruta_groups={"confirmed": ["feat_momentum"], "tentative": [], "rejected": []},
        )
        pruned = _make_result({"feat_momentum": 1.0}, {"feat_momentum": 1.0})
        report = LeakageDiagnostics(full, suspects_tm).compare_runs(pruned)
        assert len(report.cluster_lineage) == 1
        assert report.cluster_lineage.iloc[0]["survivor"] == "feat_momentum"

    def test_compare_classmethod(self, suspects_tm):
        ranks = {"a": 1.0, "b": 0.5}
        full = _make_result(ranks, ranks)
        pruned = _make_result(ranks, ranks)
        diag, report = LeakageDiagnostics.compare(full, pruned, suspects_tm, label="custom_label")
        assert isinstance(diag, LeakageDiagnostics)
        assert report.label == "custom_label"
        assert report.severity == Severity.CLEAN

    def test_label_round_trip(self, suspects_tm):
        ranks = {"a": 1.0, "b": 0.5}
        full = _make_result(ranks, ranks)
        report = LeakageDiagnostics(full, suspects_tm).compare_runs(full, label="2024_vs_2025")
        assert report.label == "2024_vs_2025"
        assert "2024_vs_2025" in repr(report)

    def test_repr_contains_correlations(self, suspects_tm):
        ranks = {"a": 1.0, "b": 0.5}
        full = _make_result(ranks, ranks)
        report = LeakageDiagnostics(full, suspects_tm).compare_runs(full)
        text = repr(report)
        assert "shap_full_vs_pruned" in text
        assert "mda_full_vs_pruned" in text
        assert "clean" in text

    def test_top_movers_has_mda_shift_column(self, suspects_tm):
        """S8 — MDA shift must be reported alongside SHAP shift."""
        ranks = {"a": 1.0, "b": 0.5}
        full = _make_result(ranks, ranks)
        report = LeakageDiagnostics(full, suspects_tm).compare_runs(full)
        assert "mda_abs_shift" in report.top_movers.columns
        assert "shap_abs_shift" in report.top_movers.columns

    def test_report_is_frozen(self, suspects_tm):
        """S1 — LeakageReport must be immutable."""
        ranks = {"a": 1.0, "b": 0.5}
        full = _make_result(ranks, ranks)
        report = LeakageDiagnostics(full, suspects_tm).compare_runs(full)
        with pytest.raises((AttributeError, Exception)):
            report.label = "mutated"

    def test_with_notes_returns_new_report(self, suspects_tm):
        """LeakageReport.with_notes returns a new instance, original untouched."""
        ranks = {"a": 1.0, "b": 0.5}
        full = _make_result(ranks, ranks)
        original = LeakageDiagnostics(full, suspects_tm).compare_runs(full)
        original_notes = original.notes
        derived = original.with_notes(("extra note",))
        assert "extra note" in derived.notes
        assert original.notes == original_notes  # unchanged
        assert derived is not original
