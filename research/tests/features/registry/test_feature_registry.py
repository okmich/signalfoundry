"""
Tests for okmich_quant_research.features.registry.FeatureRegistry
"""
import pytest
import pandas as pd

from okmich_quant_research.features.registry import (
    FeatureRegistry,
    FeatureEntry,
    CATALOG,
    CRITICAL, HIGH, MEDIUM, LOW, NONE,
    H_INTRADAY, H_SHORT, H_MEDIUM, H_LONG, H_ANY,
    R_TRENDING, R_RANGING, R_VOLATILE, R_LOW_VOL, R_CRISIS,
    SIGNAL_TYPES, RELEVANCE_LEVELS, HORIZONS, MARKET_REGIMES,
)


@pytest.fixture(scope="module")
def reg():
    return FeatureRegistry()


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────
class TestSchema:
    def test_catalog_not_empty(self):
        assert len(CATALOG) > 0

    def test_all_entries_are_feature_entry(self):
        for e in CATALOG:
            assert isinstance(e, FeatureEntry)

    def test_all_signal_types_valid(self):
        for e in CATALOG:
            assert e.signal_type in SIGNAL_TYPES, (
                f"{e.qualified_name} has unknown signal_type {e.signal_type!r}"
            )

    def test_all_relevance_levels_valid(self):
        for e in CATALOG:
            for attr in ("regime_relevance", "return_relevance", "direction_relevance"):
                val = getattr(e, attr)
                assert val in RELEVANCE_LEVELS, (
                    f"{e.qualified_name}.{attr} = {val!r} is not a valid level"
                )

    def test_all_horizons_valid(self):
        for e in CATALOG:
            assert e.horizon in HORIZONS, (
                f"{e.qualified_name} has unknown horizon {e.horizon!r}"
            )

    def test_all_works_best_in_valid(self):
        for e in CATALOG:
            for r in e.works_best_in:
                assert r in MARKET_REGIMES, (
                    f"{e.qualified_name}.works_best_in contains unknown regime {r!r}"
                )

    def test_all_output_types_valid(self):
        valid = {"series", "dataframe", "scalar", "array"}
        for e in CATALOG:
            assert e.output_type in valid, (
                f"{e.qualified_name} has unknown output_type {e.output_type!r}"
            )

    def test_no_empty_names(self):
        for e in CATALOG:
            assert e.name.strip(), f"Entry with empty name in module {e.module!r}"

    def test_no_empty_descriptions(self):
        for e in CATALOG:
            assert e.description.strip(), (
                f"{e.qualified_name} has an empty description"
            )

    def test_no_empty_modules(self):
        for e in CATALOG:
            assert e.module.strip(), f"Entry {e.name!r} has an empty module path"

    def test_qualified_name_format(self):
        for e in CATALOG:
            qn = e.qualified_name
            assert "." in qn, f"qualified_name {qn!r} does not contain a dot"
            assert qn.endswith(e.name), (
                f"qualified_name {qn!r} does not end with function name {e.name!r}"
            )

    def test_directional_implies_meaningful_direction_relevance(self):
        """Features marked directional should not have direction_relevance=NONE."""
        for e in CATALOG:
            if e.directional:
                assert e.direction_relevance != NONE, (
                    f"{e.qualified_name} is directional but direction_relevance=NONE"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Registry size and completeness
# ─────────────────────────────────────────────────────────────────────────────
class TestRegistrySize:
    def test_total_count(self, reg):
        assert len(reg) >= 200, f"Expected 200+ features, got {len(reg)}"

    def test_all_modules_represented(self, reg):
        modules = {e.module.split(".")[0] for e in reg}
        expected = {
            "microstructure", "timothymasters", "volatility",
            "path_structure", "momentum", "volume", "trend",
        }
        missing = expected - modules
        assert not missing, f"Modules missing from registry: {missing}"

    def test_microstructure_count(self, reg):
        ms = reg.by_module("microstructure")
        assert len(ms) >= 60

    def test_timothymasters_count(self, reg):
        tm = reg.by_module("timothymasters")
        assert len(tm) >= 85  # 38 single-market + 7 cross-market + 40 multi-market

    def test_volatility_count(self, reg):
        vol = reg.by_module("volatility")
        assert len(vol) >= 20

    def test_path_structure_count(self, reg):
        ps = reg.by_module("path_structure")
        assert len(ps) >= 10

    def test_all_signal_types_present(self, reg):
        found = {e.signal_type for e in reg}
        for st in SIGNAL_TYPES:
            assert st in found, f"Signal type {st!r} has no entries in the registry"

    def test_critical_features_exist(self, reg):
        critical = reg.candidates_for("regime", min_relevance=CRITICAL)
        assert len(critical) >= 10


# ─────────────────────────────────────────────────────────────────────────────
# get() — lookup by name
# ─────────────────────────────────────────────────────────────────────────────
class TestGet:
    def test_get_by_qualified_name(self, reg):
        e = reg.get("microstructure.order_flow.vpin")
        assert e.name == "vpin"
        assert e.signal_type == "toxicity"

    def test_get_by_short_name_unique(self, reg):
        e = reg.get("hurst_exponent")
        assert e.name == "hurst_exponent"

    def test_get_raises_for_missing(self, reg):
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent_feature_xyz")

    def test_get_raises_for_ambiguous(self, reg):
        # 'adx' exists in both timothymasters and momentum
        with pytest.raises(ValueError, match="Ambiguous"):
            reg.get("adx")

    def test_get_vpin_metadata(self, reg):
        e = reg.get("vpin")
        assert e.regime_relevance == CRITICAL
        assert e.return_relevance == HIGH
        assert e.needs_volume is True
        assert e.causal is True

    def test_get_hurst_metadata(self, reg):
        e = reg.get("hurst_exponent")
        assert e.regime_relevance == CRITICAL
        assert e.horizon == H_LONG
        assert e.needs_volume is False
        assert e.needs_spread is False


# ─────────────────────────────────────────────────────────────────────────────
# candidates_for()
# ─────────────────────────────────────────────────────────────────────────────
class TestCandidatesFor:
    def test_regime_critical(self, reg):
        result = reg.candidates_for("regime", min_relevance=CRITICAL)
        assert len(result) >= 10
        for e in result:
            assert e.regime_relevance == CRITICAL

    def test_regime_high(self, reg):
        result = reg.candidates_for("regime", min_relevance=HIGH)
        assert len(result) > reg.candidates_for("regime", min_relevance=CRITICAL).__len__()
        for e in result:
            assert e.regime_relevance in (CRITICAL, HIGH)

    def test_return_high(self, reg):
        result = reg.candidates_for("return", min_relevance=HIGH)
        assert len(result) >= 50
        for e in result:
            assert e.return_relevance in (CRITICAL, HIGH)

    def test_direction_high(self, reg):
        result = reg.candidates_for("direction", min_relevance=HIGH)
        assert len(result) >= 30
        for e in result:
            assert e.direction_relevance in (CRITICAL, HIGH)

    def test_invalid_task_raises(self, reg):
        with pytest.raises(AssertionError):
            reg.candidates_for("invalid_task")

    def test_invalid_relevance_raises(self, reg):
        with pytest.raises(AssertionError):
            reg.candidates_for("regime", min_relevance="VERY_HIGH")

    def test_medium_superset_of_high(self, reg):
        high = reg.candidates_for("regime", min_relevance=HIGH)
        medium = reg.candidates_for("regime", min_relevance=MEDIUM)
        assert len(medium) >= len(high)


# ─────────────────────────────────────────────────────────────────────────────
# Filter methods
# ─────────────────────────────────────────────────────────────────────────────
class TestFilters:
    def test_by_signal_type(self, reg):
        toxicity = reg.by_signal_type("toxicity")
        assert len(toxicity) >= 5
        for e in toxicity:
            assert e.signal_type == "toxicity"

    def test_by_signal_type_invalid_raises(self, reg):
        with pytest.raises(AssertionError):
            reg.by_signal_type("not_a_type")

    def test_by_module_partial_match(self, reg):
        ms = reg.by_module("microstructure")
        assert len(ms) >= 60
        for e in ms:
            assert "microstructure" in e.module

    def test_by_horizon_intraday(self, reg):
        result = reg.by_horizon(H_INTRADAY)
        assert len(result) >= 5
        for e in result:
            assert e.horizon == H_INTRADAY

    def test_by_horizon_invalid_raises(self, reg):
        with pytest.raises(AssertionError):
            reg.by_horizon("weekly")

    def test_works_in(self, reg):
        crisis = reg.works_in(R_CRISIS)
        assert len(crisis) >= 5
        for e in crisis:
            assert R_CRISIS in e.works_best_in

    def test_works_in_invalid_raises(self, reg):
        with pytest.raises(AssertionError):
            reg.works_in("bull_market")

    def test_directional(self, reg):
        result = reg.directional()
        assert len(result) >= 80
        for e in result:
            assert e.directional is True

    def test_causal_only(self, reg):
        result = reg.causal_only()
        # All registered features should be causal
        assert len(result) == len(reg)

    def test_needs_only_price(self, reg):
        result = reg.needs_only_price()
        assert len(result) >= 50
        for e in result:
            assert e.needs_volume is False
            assert e.needs_spread is False
            assert e.needs_benchmark is False

    def test_filter_volume_only(self, reg):
        result = reg.filter(needs_volume=True)
        assert len(result) >= 50
        for e in result:
            assert e.needs_volume is True

    def test_filter_no_spread(self, reg):
        result = reg.filter(needs_spread=False)
        for e in result:
            assert e.needs_spread is False

    def test_filter_series_output(self, reg):
        result = reg.filter(output_type="series")
        assert len(result) > len(reg.filter(output_type="dataframe"))
        for e in result:
            assert e.output_type == "series"


# ─────────────────────────────────────────────────────────────────────────────
# Chaining
# ─────────────────────────────────────────────────────────────────────────────
class TestChaining:
    def test_chain_candidates_directional_price_only(self, reg):
        result = (
            reg.candidates_for("return", min_relevance=HIGH)
            .directional()
            .needs_only_price()
        )
        assert len(result) >= 20
        for e in result:
            assert e.return_relevance in (CRITICAL, HIGH)
            assert e.directional is True
            assert e.needs_volume is False

    def test_chain_regime_crisis_volume(self, reg):
        result = (
            reg.candidates_for("regime", min_relevance=HIGH)
            .works_in(R_CRISIS)
            .filter(needs_volume=True)
        )
        assert len(result) >= 3
        for e in result:
            assert e.regime_relevance in (CRITICAL, HIGH)
            assert R_CRISIS in e.works_best_in
            assert e.needs_volume is True

    def test_chain_toxicity_directional(self, reg):
        result = reg.by_signal_type("toxicity").directional()
        assert len(result) >= 1
        for e in result:
            assert e.signal_type == "toxicity"
            assert e.directional is True

    def test_chain_microstructure_intraday(self, reg):
        result = reg.by_module("microstructure").by_horizon(H_INTRADAY)
        assert len(result) >= 5
        for e in result:
            assert "microstructure" in e.module
            assert e.horizon == H_INTRADAY

    def test_chained_view_is_iterable(self, reg):
        result = reg.candidates_for("regime", min_relevance=HIGH).directional()
        names = [e.name for e in result]
        assert len(names) == len(result)

    def test_chained_view_names(self, reg):
        result = reg.by_signal_type("toxicity")
        names = result.names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_chained_view_qualified_names(self, reg):
        result = reg.by_signal_type("toxicity")
        qnames = result.qualified_names()
        for qn in qnames:
            assert "." in qn


# ─────────────────────────────────────────────────────────────────────────────
# summary() DataFrame
# ─────────────────────────────────────────────────────────────────────────────
class TestSummary:
    def test_returns_dataframe(self, reg):
        df = reg.summary()
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, reg):
        df = reg.summary()
        assert len(df) == len(reg)

    def test_expected_columns(self, reg):
        df = reg.summary()
        expected = {
            "name", "module", "signal_type", "description",
            "regime_relevance", "return_relevance", "direction_relevance",
            "horizon", "directional", "causal", "output_type",
            "needs_volume", "needs_spread", "needs_benchmark",
            "works_best_in", "notes",
        }
        assert expected.issubset(set(df.columns))

    def test_summary_on_view(self, reg):
        view = reg.by_signal_type("toxicity")
        df = view.summary()
        assert len(df) == len(view)
        assert (df.signal_type == "toxicity").all()

    def test_critical_regime_filterable_via_dataframe(self, reg):
        df = reg.summary()
        critical = df[df.regime_relevance == CRITICAL]
        assert len(critical) >= 10

    def test_no_null_required_columns(self, reg):
        df = reg.summary()
        for col in ("name", "module", "signal_type", "description"):
            assert df[col].notna().all(), f"Nulls found in column {col!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Known important features — spot checks
# ─────────────────────────────────────────────────────────────────────────────
class TestSpotChecks:
    @pytest.mark.parametrize("name,module_fragment,signal_type,causal", [
        ("vpin",               "microstructure",    "toxicity",       True),
        ("hurst_exponent",     "path_structure",    "regime",         True),
        ("choppiness_index",   "path_structure",    "regime",         True),
        ("cvd",                "microstructure",    "order_flow",     True),
        ("adx",                "timothymasters",    "regime",         True),
        ("volatility_of_volatility", "volatility",  "regime",         True),
        ("aroon_diff",         "timothymasters",    "trend",          True),
        ("pin_proxy",          "microstructure",    "toxicity",       True),
        ("institutional_footprint_score", "microstructure", "composite", True),
    ])
    def test_known_feature_metadata(self, reg, name, module_fragment, signal_type, causal):
        # Use qualified lookup to avoid ambiguity
        matches = [e for e in reg if e.name == name and module_fragment in e.module]
        assert len(matches) >= 1, f"Feature {name!r} not found in {module_fragment!r}"
        e = matches[0]
        assert e.signal_type == signal_type, (
            f"{e.qualified_name}: expected signal_type={signal_type!r}, got {e.signal_type!r}"
        )
        assert e.causal == causal

    def test_vpin_is_volume_dependent(self, reg):
        e = reg.get("microstructure.order_flow.vpin")
        assert e.needs_volume is True

    def test_hurst_needs_only_price(self, reg):
        e = reg.get("hurst_exponent")
        assert e.needs_volume is False
        assert e.needs_spread is False
        assert e.needs_benchmark is False

    def test_liquidity_commonality_needs_benchmark(self, reg):
        e = reg.get("liquidity_commonality")
        assert e.needs_benchmark is True

    def test_cvd_is_directional(self, reg):
        e = reg.get("microstructure.order_flow.cvd")
        assert e.directional is True
        assert e.direction_relevance == CRITICAL

    def test_regime_fragility_not_directional(self, reg):
        e = reg.get("regime_fragility_index")
        assert e.directional is False

    def test_liquidity_resilience_returns_dataframe(self, reg):
        e = reg.get("liquidity_resilience")
        assert e.output_type == "dataframe"