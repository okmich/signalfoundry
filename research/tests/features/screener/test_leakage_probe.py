"""
Tests for LeakageDiagnostics.probe and the stratified row sampler.

The probe is expensive — interaction-SHAP on an XGBoost fit. Tests use small
synthetic datasets (a few hundred rows × ~6 features) and a tiny
``interaction_rows`` budget to keep runtime tractable.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.screener import ScreenerResult
from okmich_quant_research.features.screener.leakage import (
    LeakageDiagnostics,
    SamplingTask,
    Severity,
    SuspectRegistry,
    stratified_row_sample,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sampler tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStratifiedRowSample:
    def test_returns_all_when_n_rows_exceeds_length(self):
        y = pd.Series(np.random.randn(50))
        positions = stratified_row_sample(y, task=SamplingTask.RETURN, n_rows=200)
        assert len(positions) == 50
        assert (positions == np.arange(50)).all()

    def test_zero_n_rows_returns_all(self):
        y = pd.Series(np.random.randn(50))
        positions = stratified_row_sample(y, task=SamplingTask.RETURN, n_rows=0)
        assert len(positions) == 50

    def test_regime_stratification_preserves_classes(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2]))
        positions = stratified_row_sample(y, task=SamplingTask.REGIME, n_rows=300)
        sampled = y.iloc[positions]
        # Every class should appear in the subsample
        assert set(sampled.unique()) == {0, 1, 2}
        # Sample size should land near target
        assert 250 <= len(sampled) <= 350

    def test_return_tail_oversampling(self):
        """Top and bottom deciles must be over-represented relative to flat sampling."""
        rng = np.random.default_rng(1)
        y = pd.Series(rng.standard_normal(2000))
        positions = stratified_row_sample(y, task=SamplingTask.RETURN, n_rows=400, tail_oversample=3.0)
        sampled = y.iloc[positions]
        # Compare tail mass: |z| > 1.5 should be relatively richer than in y
        full_tail_pct = (y.abs() > 1.5).mean()
        sample_tail_pct = (sampled.abs() > 1.5).mean()
        assert sample_tail_pct > full_tail_pct

    def test_return_with_nan_y(self):
        y = pd.Series(np.r_[np.random.randn(100), np.full(20, np.nan)])
        positions = stratified_row_sample(y, task=SamplingTask.RETURN, n_rows=50)
        # No sampled position should land on a NaN
        assert not y.iloc[positions].isna().any()

    def test_positions_are_sorted(self):
        y = pd.Series(np.random.randn(500))
        positions = stratified_row_sample(y, task=SamplingTask.RETURN, n_rows=100)
        assert (np.diff(positions) >= 0).all()

    def test_return_returns_exactly_n_rows(self):
        """B2 — sum of per-bin samples must equal the requested budget."""
        y = pd.Series(np.random.randn(2000))
        for n_rows in (100, 333, 1234):
            positions = stratified_row_sample(y, task=SamplingTask.RETURN, n_rows=n_rows)
            assert len(positions) == n_rows, f"expected {n_rows}, got {len(positions)}"


# ─────────────────────────────────────────────────────────────────────────────
# Probe fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def leak_synthetic():
    """
    Synthetic regression setup with one engineered leak path.

    Suspect family: ``tm_vol_*``.
    True signal: ``tm_vol_5 * gate``  (an interaction-mediated leak).
    Confirmed partners: ``feat_momentum`` (genuine signal),
                        ``gate``         (leak partner — encodes the interaction),
                        ``feat_noise``   (noise).

    Probe should pick up (tm_vol_5, gate) as the highest-interaction pair.
    """
    rng = np.random.default_rng(42)
    n = 800

    tm_vol_5 = rng.standard_normal(n)
    gate = rng.standard_normal(n)
    feat_momentum = rng.standard_normal(n)
    feat_noise = rng.standard_normal(n)

    # Forward returns: independent contributions from momentum + interaction signal
    y = 0.5 * feat_momentum + 1.5 * tm_vol_5 * gate + 0.3 * rng.standard_normal(n)

    X = pd.DataFrame({
        "tm_vol_5":      tm_vol_5,
        "gate":          gate,
        "feat_momentum": feat_momentum,
        "feat_noise":    feat_noise,
    })
    y_series = pd.Series(y, name="forward_return")
    return X, y_series


@pytest.fixture
def leak_full_run(leak_synthetic):
    """
    A ScreenerResult mimicking what FeatureScreener would have produced on
    leak_synthetic: tm_vol_5, gate, and feat_momentum confirmed; feat_noise
    rejected. Each in its own singleton cluster so the partner-side cluster
    metadata is well-defined.
    """
    X, _ = leak_synthetic
    confirmed = ["feat_momentum", "tm_vol_5", "gate"]
    shap_rank = pd.Series({"feat_momentum": 0.5, "tm_vol_5": 0.3, "gate": 0.2})
    mda_rank = pd.Series({"feat_momentum": 0.5, "tm_vol_5": 0.3, "gate": 0.2})
    return ScreenerResult(
        confirmed=confirmed, tentative=[], rejected=["feat_noise"],
        shap_rank=shap_rank, mda_rank=mda_rank,
        cluster_assignments={1: ["tm_vol_5"], 2: ["gate"], 3: ["feat_momentum"], 4: ["feat_noise"]},
        cluster_representatives={1: "tm_vol_5", 2: "gate", 3: "feat_momentum", 4: "feat_noise"},
        boruta_groups={"confirmed": confirmed, "tentative": [], "rejected": ["feat_noise"]},
    )


@pytest.fixture
def leak_pruned_run(leak_synthetic):
    """Mimics screener output after dropping the tm_vol_ family."""
    confirmed = ["feat_momentum", "gate"]
    shap_rank = pd.Series({"feat_momentum": 0.6, "gate": 0.2})
    mda_rank = pd.Series({"feat_momentum": 0.6, "gate": 0.2})
    return ScreenerResult(
        confirmed=confirmed, tentative=[], rejected=["feat_noise"],
        shap_rank=shap_rank, mda_rank=mda_rank,
        cluster_assignments={1: ["gate"], 2: ["feat_momentum"], 3: ["feat_noise"]},
        cluster_representatives={1: "gate", 2: "feat_momentum", 3: "feat_noise"},
        boruta_groups={"confirmed": confirmed, "tentative": [], "rejected": ["feat_noise"]},
    )


@pytest.fixture
def suspects_tm():
    return SuspectRegistry(prefixes=("tm_vol_",), rationale="vol family suspected leak source")


# ─────────────────────────────────────────────────────────────────────────────
# Probe behavioural tests
# ─────────────────────────────────────────────────────────────────────────────

class TestProbe:
    def test_interactions_disabled_returns_compare_only(self, leak_synthetic, leak_full_run, leak_pruned_run, suspects_tm):
        X, y = leak_synthetic
        diag = LeakageDiagnostics(leak_full_run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=leak_pruned_run, interaction_rows=0)
        assert report.suspect_interactions is None
        assert any("interaction_rows=0" in n for n in report.notes)
        # Compare diagnostic still ran
        assert "shap_full_vs_pruned" in report.rank_correlations

    def test_clean_severity_skips_interactions_without_force(self, leak_synthetic, leak_full_run, suspects_tm):
        # full == pruned → identical → CLEAN → interactions skipped
        X, y = leak_synthetic
        diag = LeakageDiagnostics(leak_full_run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=leak_full_run, interaction_rows=500)
        assert report.severity == Severity.CLEAN
        assert report.suspect_interactions is None
        assert any("CLEAN" in n or "force_interactions" in n for n in report.notes)

    def test_force_interactions_overrides_clean(self, leak_synthetic, leak_full_run, suspects_tm):
        X, y = leak_synthetic
        diag = LeakageDiagnostics(leak_full_run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=leak_full_run,
                            interaction_rows=400, force_interactions=True)
        assert report.suspect_interactions is not None
        assert len(report.suspect_interactions) > 0

    def test_interaction_detects_known_leak_partner(self, leak_synthetic, leak_full_run, leak_pruned_run, suspects_tm):
        """
        The synthetic fixture has y = ... + 1.5 * tm_vol_5 * gate + ...
        The probe should rank (tm_vol_5, gate) as the top interaction pair.
        """
        X, y = leak_synthetic
        diag = LeakageDiagnostics(leak_full_run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=leak_pruned_run,
                            interaction_rows=600, force_interactions=True, top_k_pairs=10)

        assert report.suspect_interactions is not None
        assert len(report.suspect_interactions) > 0
        top = report.suspect_interactions.iloc[0]
        assert top["suspect"] == "tm_vol_5"
        assert top["partner"] == "gate"
        # And gate's interaction must dominate any other partner
        gate_interaction = report.suspect_interactions[
            report.suspect_interactions["partner"] == "gate"
        ]["mean_abs_interaction"].iloc[0]
        non_gate = report.suspect_interactions[
            report.suspect_interactions["partner"] != "gate"
        ]
        if not non_gate.empty:
            assert gate_interaction > non_gate["mean_abs_interaction"].max()

    def test_partner_metadata_columns_present(self, leak_synthetic, leak_full_run, leak_pruned_run, suspects_tm):
        X, y = leak_synthetic
        diag = LeakageDiagnostics(leak_full_run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=leak_pruned_run,
                            interaction_rows=400, force_interactions=True)
        cols = set(report.suspect_interactions.columns)
        required = {
            "suspect", "partner", "mean_abs_interaction",
            "partner_marginal_shap_rank", "partner_corr_with_suspect",
            "partner_corr_with_label", "partner_cluster_id", "partner_cluster_size",
        }
        assert required.issubset(cols)

    def test_no_pruned_run_runs_with_force(self, leak_synthetic, leak_full_run, suspects_tm):
        X, y = leak_synthetic
        diag = LeakageDiagnostics(leak_full_run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=None,
                            interaction_rows=400, force_interactions=True)
        # No comparison ⇒ severity falls through as INVESTIGATE; interactions still run
        assert report.severity == Severity.INVESTIGATE
        assert report.suspect_interactions is not None
        assert any("No pruned_run" in n for n in report.notes)

    def test_no_suspects_in_universe_handled(self, leak_synthetic, leak_full_run):
        """A registry that matches nothing should short-circuit the interaction probe gracefully."""
        X, y = leak_synthetic
        suspects = SuspectRegistry(prefixes=("nonexistent_",), rationale="dry run")
        diag = LeakageDiagnostics(leak_full_run, suspects)
        report = diag.probe(X, y, task="return", pruned_run=leak_full_run,
                            interaction_rows=400, force_interactions=True)
        assert report.suspect_interactions is None
        assert any("No suspects" in n for n in report.notes)

    def test_no_confirmed_partners_handled(self, leak_synthetic, suspects_tm):
        """If every confirmed feature is a suspect, there are no partners to test against."""
        X, y = leak_synthetic
        run = ScreenerResult(
            confirmed=["tm_vol_5"], tentative=[], rejected=[],
            shap_rank=pd.Series({"tm_vol_5": 1.0}),
            mda_rank=pd.Series({"tm_vol_5": 1.0}),
            cluster_assignments={1: ["tm_vol_5"]},
            cluster_representatives={1: "tm_vol_5"},
            boruta_groups={"confirmed": ["tm_vol_5"], "tentative": [], "rejected": []},
        )
        diag = LeakageDiagnostics(run, suspects_tm)
        report = diag.probe(X, y, task="return", pruned_run=None,
                            interaction_rows=400, force_interactions=True)
        assert report.suspect_interactions is None
        assert any("No confirmed non-suspect partners" in n for n in report.notes)
