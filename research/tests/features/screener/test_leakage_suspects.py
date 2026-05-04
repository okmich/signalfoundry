"""
Tests for okmich_quant_research.features.screener.leakage._suspects

Resolution-only — these tests construct ScreenerResult fixtures by hand
to exercise the suspect-resolver in isolation, without paying the cost
of running the full screener pipeline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from okmich_quant_research.features.screener import ScreenerResult
from okmich_quant_research.features.screener.leakage import (
    ResolvedSuspectSet,
    SuspectRegistry,
    resolve,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_series():
    return pd.Series(dtype=float)


@pytest.fixture
def screener_result_basic(empty_series):
    """
    Three clusters:
      cluster 1: tm_vol_5, tm_vol_20  (vol family — survivor: tm_vol_5)
      cluster 2: feat_momentum, tm_vol_60  (mixed — survivor: feat_momentum)
      cluster 3: feat_skew  (singleton)

    Stage 4: feat_momentum confirmed; tm_vol_5 tentative; feat_skew rejected.
    """
    return ScreenerResult(
        confirmed=["feat_momentum"],
        tentative=["tm_vol_5"],
        rejected=["tm_vol_20", "tm_vol_60", "feat_skew"],
        shap_rank=empty_series,
        mda_rank=empty_series,
        cluster_assignments={
            1: ["tm_vol_5", "tm_vol_20"],
            2: ["feat_momentum", "tm_vol_60"],
            3: ["feat_skew"],
        },
        cluster_representatives={
            1: "tm_vol_5",
            2: "feat_momentum",
            3: "feat_skew",
        },
        boruta_groups={
            "confirmed": ["feat_momentum"],
            "tentative": ["tm_vol_5"],
            "rejected":  ["feat_skew"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# SuspectRegistry construction
# ─────────────────────────────────────────────────────────────────────────────

class TestSuspectRegistry:
    def test_requires_at_least_one_match_source(self):
        with pytest.raises(ValueError):
            SuspectRegistry()

    def test_prefix_only(self):
        r = SuspectRegistry(prefixes=("tm_vol_",))
        assert r.matches("tm_vol_5")
        assert not r.matches("feat_momentum")

    def test_explicit_only(self):
        r = SuspectRegistry(explicit_features=("feat_skew",))
        assert r.matches("feat_skew")
        assert not r.matches("feat_skewness")  # exact match required

    def test_union_semantics(self):
        r = SuspectRegistry(prefixes=("tm_vol_",), explicit_features=("feat_skew",))
        assert r.matches("tm_vol_5")
        assert r.matches("feat_skew")
        assert not r.matches("feat_momentum")

    def test_rationale_round_trip(self):
        r = SuspectRegistry(prefixes=("tm_",), rationale="all tm_ features derive from realised vol")
        assert "realised vol" in r.rationale


# ─────────────────────────────────────────────────────────────────────────────
# resolve()
# ─────────────────────────────────────────────────────────────────────────────

class TestResolve:
    def test_direct_suspects_includes_rejected(self, screener_result_basic):
        """A suspect dropped at any stage still appears in direct_suspects."""
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, screener_result_basic)
        assert set(out.direct_suspects) == {"tm_vol_5", "tm_vol_20", "tm_vol_60"}

    def test_cluster_inheritance_detected(self, screener_result_basic):
        """
        cluster 2 contains tm_vol_60 (suspect) but the survivor is
        feat_momentum (NOT a suspect) — feat_momentum has inherited the
        suspect's correlated signal.
        """
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, screener_result_basic)
        assert "feat_momentum" in out.cluster_inherited_suspects
        # tm_vol_5 is the survivor of cluster 1 AND a direct suspect — it does
        # not count as cluster-inherited (the survivor is the suspect itself).
        assert "tm_vol_5" not in out.cluster_inherited_suspects

    def test_confirmed_suspects(self, screener_result_basic):
        """tm_vol_5 is tentative, not confirmed → confirmed_suspects empty."""
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, screener_result_basic)
        assert out.confirmed_suspects == []

    def test_confirmed_suspects_non_empty(self, empty_series):
        """When a suspect IS in confirmed, it surfaces here."""
        result = ScreenerResult(
            confirmed=["tm_vol_5"], tentative=[], rejected=[],
            shap_rank=empty_series, mda_rank=empty_series,
            cluster_assignments={1: ["tm_vol_5"]},
            cluster_representatives={1: "tm_vol_5"},
            boruta_groups={"confirmed": ["tm_vol_5"], "tentative": [], "rejected": []},
        )
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, result)
        assert out.confirmed_suspects == ["tm_vol_5"]

    def test_cluster_lineage_shape(self, screener_result_basic):
        """One row per cluster containing >=1 suspect; cluster 3 (no suspect) excluded."""
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, screener_result_basic)
        assert len(out.cluster_lineage) == 2
        assert set(out.cluster_lineage["cluster_id"]) == {1, 2}
        assert set(out.cluster_lineage.columns) == {
            "cluster_id", "members", "suspects_in_cluster", "survivor", "survivor_is_suspect",
        }

    def test_cluster_lineage_survivor_is_suspect_flag(self, screener_result_basic):
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, screener_result_basic)
        row1 = out.cluster_lineage[out.cluster_lineage["cluster_id"] == 1].iloc[0]
        row2 = out.cluster_lineage[out.cluster_lineage["cluster_id"] == 2].iloc[0]
        assert bool(row1["survivor_is_suspect"]) is True   # tm_vol_5 survived its own cluster
        assert bool(row2["survivor_is_suspect"]) is False  # feat_momentum survived a mixed cluster

    def test_clean_run_no_matches(self, screener_result_basic):
        """When no feature matches the registry, every output collection is empty."""
        registry = SuspectRegistry(prefixes=("nonexistent_",))
        out = resolve(registry, screener_result_basic)
        assert out.direct_suspects == []
        assert out.cluster_inherited_suspects == []
        assert out.confirmed_suspects == []
        assert out.cluster_lineage.empty

    def test_input_features_finds_pre_stage3_rejects(self, empty_series):
        """
        A suspect rejected before Stage 3 (e.g. by variance filter) won't appear
        in any cluster. Without input_features it would be missed; with
        input_features it shows up as a direct_suspect.
        """
        result = ScreenerResult(
            confirmed=["feat_momentum"], tentative=[], rejected=["tm_vol_5"],
            shap_rank=empty_series, mda_rank=empty_series,
            cluster_assignments={1: ["feat_momentum"]},  # tm_vol_5 never reached Stage 3
            cluster_representatives={1: "feat_momentum"},
            boruta_groups={"confirmed": ["feat_momentum"], "tentative": [], "rejected": []},
        )
        registry = SuspectRegistry(prefixes=("tm_vol_",))

        # Without input_features: still finds it via the rejected list
        out = resolve(registry, result)
        assert "tm_vol_5" in out.direct_suspects

        # With explicit input_features list including a suspect not in any other field
        out_explicit = resolve(registry, result, input_features=["feat_momentum", "tm_vol_5", "tm_vol_99"])
        assert set(out_explicit.direct_suspects) == {"tm_vol_5", "tm_vol_99"}

    def test_repr(self, screener_result_basic):
        registry = SuspectRegistry(prefixes=("tm_vol_",))
        out = resolve(registry, screener_result_basic)
        text = repr(out)
        assert "direct=3" in text
        assert "cluster_inherited=1" in text
        assert "confirmed=0" in text
