"""Axis-layer behaviour inside the screener.

Covers the parts of the trend/momentum collapse that only show up end-to-end: the one-sided guard
reaching the output, the per-axis horizon reaching the evaluator, the liquidity data-capability gate,
the sub-cell report, and the removal of the momentum code path.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.hmm_screener import (AXIS_EVALUATORS, HmmFeatureScreener,
                                                         HmmScreenerConfig, OneSidedPolicy,
                                                         ParetoStatus, ScreenStrategy, get_evaluator,
                                                         primary_horizon_for,
                                                         stage0c_collinearity_filter)
from okmich_quant_research.features.registry import Axis

import okmich_quant_features.momentum as mom


def _make_ohlc(T: int = 900, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.001, T)
    log_rets[: T // 2] += 0.0008
    log_rets[T // 2:] -= 0.0008
    close = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame({"close": close,
                         "high": close * (1.0 + rng.uniform(0.0, 0.002, T)),
                         "low": close * (1.0 - rng.uniform(0.0, 0.002, T)),
                         "tick_volume": rng.uniform(50.0, 150.0, T)})


def _di_features(df: pd.DataFrame) -> pd.DataFrame:
    """Emit the one-sided DI pair and its canonical spread under their REGISTRY-qualified names.

    The names matter: the screener resolves a column through ``FeatureRegistry.get``, so an unqualified
    column would silently skip every coherence and one-sided check.
    """
    high, low, close = df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)
    out["momentum.minus_di"] = mom.minus_di(high, low, close, period=14)
    out["momentum.plus_di"] = mom.plus_di(high, low, close, period=14)
    out["momentum.di_spread"] = mom.di_spread(high, low, close, period=14)
    out["momentum.roc"] = mom.roc(close, window=14)
    return out


def _first_unstamped_directional_name() -> str:
    """A catalogued DIRECTIONAL-eligible feature the probe has not covered."""
    from okmich_quant_research.features.registry import FeatureRegistry
    reg = FeatureRegistry()
    return next(e.qualified_name for e in reg.eligible_for(Axis.DIRECTIONAL) if e.invariance is None)


def _screener(features, axis=Axis.DIRECTIONAL, **cfg):
    raw = _make_ohlc()
    config = HmmScreenerConfig(axis=axis, algo="hmm_lambda", n_states=2, data_size=len(raw),
                               random_state=42, **cfg)
    return HmmFeatureScreener(config, raw, _di_features), features


# ── acceptance criterion 2: a lone one-sided feature is surfaced ──────────────────────────────────

def test_lone_one_sided_feature_warns_on_the_subset():
    """``momentum.minus_di`` alone cannot express direction: its low state pools 'the other way' with
    'no move at all'. It won the trend axis on 11 of 14 FX symbols anyway, because nothing looked."""
    screener, feats = _screener(["momentum.minus_di"])
    warns = screener._check_one_sided(("momentum.minus_di",))
    assert len(warns) == 1
    assert "ONE-SIDED" in warns[0]
    assert "momentum.plus_di" in warns[0]


def test_one_sided_feature_with_its_conjugate_present_is_not_flagged():
    """The PAIR spans the axis, so together they are fine — only lone halves are a defect."""
    screener, _ = _screener([])
    assert screener._check_one_sided(("momentum.minus_di", "momentum.plus_di")) == []


def test_a_safe_one_sided_pair_produces_no_warning_from_any_check():
    """Regression: the coherence check used to re-emit ``is_eligible``'s ONE-SIDED advisory.

    ``is_eligible`` sees one feature at a time, so it cannot know whether the conjugate is present and
    always advises "pair it with its conjugate". Emitted per subset, that told a subset holding BOTH
    halves — which is exactly what the advice asks for — that it had a problem. Worse, it put a warning
    on nearly every row, which is how a genuinely loud warning gets ignored.
    """
    screener, _ = _screener([])
    pair = ("momentum.minus_di", "momentum.plus_di")
    assert screener._validate_subset_coherence(pair) == []
    assert screener._check_one_sided(pair) == []


def test_lone_one_sided_feature_is_reported_exactly_once():
    """One finding, one message, from the only check that has subset context."""
    screener, _ = _screener([])
    subset = ("momentum.minus_di",)
    total = screener._validate_subset_coherence(subset) + screener._check_one_sided(subset)
    assert len(total) == 1
    assert total[0].startswith("one_sided: ")


def test_unstamped_advisory_does_not_appear_per_subset():
    """A coverage gap is a property of the POOL, identical on every subset containing the feature.

    67 of the 116 DIRECTIONAL-eligible catalogue entries carry no stamp, so repeating the advisory per
    subset would put a line on nearly every output row.
    """
    screener, _ = _screener([])
    unstamped = _first_unstamped_directional_name()
    assert screener._validate_subset_coherence((unstamped,)) == []


def test_measurement_coverage_is_reported_once_per_screen():
    screener, _ = _screener([])
    unstamped = _first_unstamped_directional_name()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screener._report_measurement_coverage([unstamped, "momentum.roc"])
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "no measured invariance stamp" in messages[0]
    assert unstamped in messages[0]


def test_measurement_coverage_names_unregistered_candidates():
    screener, _ = _screener([])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screener._report_measurement_coverage(["momentum.roc", "not_a_registered_column"])
    messages = [str(w.message) for w in caught]
    assert any("not in the FeatureRegistry" in m and "not_a_registered_column" in m for m in messages)


def test_measurement_coverage_is_silent_for_a_fully_stamped_pool():
    screener, _ = _screener([])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screener._report_measurement_coverage(["momentum.roc", "momentum.di_spread"])
    assert [str(w.message) for w in caught] == []


def test_measurement_coverage_skips_the_liquidity_axis():
    """Reflection cannot classify a volume feature, so an unstamped liquidity candidate is expected,
    not a gap — warning about it would be noise the reader cannot act on."""
    screener, _ = _screener([], axis=Axis.LIQUIDITY, has_real_volume=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        screener._report_measurement_coverage([_first_unstamped_directional_name()])
    assert [str(w.message) for w in caught] == []


def test_canonical_spread_is_never_flagged():
    screener, _ = _screener([])
    assert screener._check_one_sided(("momentum.di_spread",)) == []


def test_one_sided_check_is_directional_axis_only():
    """Reflection says nothing about a volume feature, so the guard must not fire off-axis."""
    screener, _ = _screener([], axis=Axis.VOLATILITY)
    assert screener._check_one_sided(("momentum.minus_di",)) == []


def test_raise_policy_refuses_outright():
    screener, _ = _screener([], one_sided_policy=OneSidedPolicy.RAISE)
    with pytest.raises(ValueError, match="ONE-SIDED"):
        screener._check_one_sided(("momentum.minus_di",))


@pytest.mark.slow
def test_one_sided_warning_survives_into_the_persisted_result():
    """A warning nobody can read is not a warning. It has to reach ``SubsetEvaluation.warnings`` AND
    the results frame that gets written out."""
    screener, _ = _screener([])
    result = screener.screen(["momentum.minus_di", "momentum.roc"], strategy=ScreenStrategy.EXHAUSTIVE,
                             max_subset_size=1)
    lone = [e for e in result.evaluations if e.features == ("momentum.minus_di",)]
    assert lone, "expected the single-feature minus_di subset to be evaluated"
    assert any("ONE-SIDED" in w for w in lone[0].warnings)

    row = result.results_[result.results_.features == "momentum.minus_di"].iloc[0]
    assert "ONE-SIDED" in row.warnings


@pytest.mark.slow
def test_exclude_policy_marks_the_subset_fragile():
    """A lone one-sided feature is a structural defect in what the subset can EXPRESS, which is exactly
    what the Phase-A gate is for — the separation number can look fine while measuring 'moving strongly
    one way' against 'everything else'."""
    screener, _ = _screener([], one_sided_policy=OneSidedPolicy.EXCLUDE)
    result = screener.screen(["momentum.minus_di", "momentum.roc"], strategy=ScreenStrategy.EXHAUSTIVE,
                             max_subset_size=1)
    lone = [e for e in result.evaluations if e.features == ("momentum.minus_di",)][0]
    assert lone.pareto_status is ParetoStatus.FRAGILE


# ── acceptance criterion 7: the per-axis horizon reaches the evaluator ────────────────────────────

def test_primary_horizon_for_directional_is_eighteen():
    assert primary_horizon_for(Axis.DIRECTIONAL) == 18


@pytest.mark.slow
def test_directional_horizon_reaches_the_evaluator_raw_details():
    """The corpus was screened at 12 because ``horizons[0]`` was hard-coded as the primary horizon for
    every axis. Assert the measured per-axis value actually lands in the evaluator's own report."""
    screener, _ = _screener([])
    result = screener.screen(["momentum.roc"], strategy=ScreenStrategy.EXHAUSTIVE, max_subset_size=1)
    ev = [e for e in result.evaluations if e.error is None][0]
    assert ev.raw_details["horizon"] == 18


@pytest.mark.slow
def test_explicit_primary_horizon_overrides_the_axis_default_end_to_end():
    screener, _ = _screener([], primary_horizon=7)
    result = screener.screen(["momentum.roc"], strategy=ScreenStrategy.EXHAUSTIVE, max_subset_size=1)
    ev = [e for e in result.evaluations if e.error is None][0]
    assert ev.raw_details["horizon"] == 7


def test_evaluator_prefers_primary_horizon_over_horizons_tuple():
    """Direct call, no fit: the explicit per-axis horizon must win over the ``horizons[0]`` fallback."""
    raw = _make_ohlc(400)
    labels = np.zeros(len(raw), dtype=int)
    labels[len(raw) // 2:] = 1
    out = get_evaluator(Axis.DIRECTIONAL)(gamma=None, state_labels=labels, raw_data=raw,
                                          horizons=(12, 60), primary_horizon=18)
    assert out.raw_details["horizon"] == 18


def test_evaluator_falls_back_to_horizons_when_no_primary_given():
    raw = _make_ohlc(400)
    labels = np.zeros(len(raw), dtype=int)
    labels[len(raw) // 2:] = 1
    out = get_evaluator(Axis.DIRECTIONAL)(gamma=None, state_labels=labels, raw_data=raw, horizons=(9, 60))
    assert out.raw_details["horizon"] == 9


# ── acceptance criterion 8: the momentum path is gone ─────────────────────────────────────────────

def test_evaluate_momentum_is_not_importable():
    with pytest.raises(ImportError):
        from okmich_quant_research.features.hmm_screener import evaluate_momentum  # noqa: F401


def test_infer_is_directional_is_gone():
    """It existed solely to choose ``evaluate_momentum``'s signed/unsigned branch, and it read the
    hand-DECLARED ``directional`` flag — which the measurement contradicts for the DI and Aroon pairs."""
    assert not hasattr(HmmFeatureScreener, "_infer_is_directional")


def test_dispatch_is_keyed_by_axis_with_no_stubs():
    assert set(AXIS_EVALUATORS) == set(Axis)
    assert AXIS_EVALUATORS[Axis.DIRECTIONAL].__name__ == "evaluate_direction"


def test_get_evaluator_rejects_a_signal_type():
    """A feature tag must not resolve to an evaluator, and the error has to say what to use instead —
    a rejection alone leaves the caller nowhere to go."""
    for signal_type in ("trend", "momentum", "price_structure"):
        with pytest.raises(KeyError, match="signal_types, not axes"):
            get_evaluator(signal_type)


def test_evaluator_error_names_the_corresponding_axis():
    with pytest.raises(KeyError, match="Axis.DIRECTIONAL"):
        get_evaluator("trend")


# ── liquidity data-capability gate ────────────────────────────────────────────────────────────────

def test_liquidity_axis_refuses_an_undeclared_feed():
    """On MT5 feeds ``tick_volume`` is a tick COUNT, so the column being present proves nothing. The
    library has no feed metadata, so the capability is the caller's assertion to make."""
    raw = _make_ohlc()
    config = HmmScreenerConfig(axis=Axis.LIQUIDITY, algo="hmm_lambda", n_states=2, data_size=len(raw))
    with pytest.raises(ValueError, match="tick COUNT"):
        HmmFeatureScreener(config, raw, _di_features)


def test_liquidity_axis_accepted_once_the_capability_is_declared():
    raw = _make_ohlc()
    config = HmmScreenerConfig(axis=Axis.LIQUIDITY, algo="hmm_lambda", n_states=2, data_size=len(raw),
                               has_real_volume=True)
    assert HmmFeatureScreener(config, raw, _di_features) is not None


@pytest.mark.parametrize("axis", [Axis.DIRECTIONAL, Axis.VOLATILITY, Axis.PATH_STRUCTURE])
def test_other_axes_are_unaffected_by_the_volume_gate(axis):
    raw = _make_ohlc()
    config = HmmScreenerConfig(axis=axis, algo="hmm_lambda", n_states=2, data_size=len(raw))
    assert HmmFeatureScreener(config, raw, _di_features) is not None


# ── sub-cell reporting ────────────────────────────────────────────────────────────────────────────

def test_subset_cells_report_the_scale_composition():
    """DIRECTIONAL is ONE axis over two sub-cells. A K=2 normal-emission HMM fitted on a mix can
    partition on move MAGNITUDE rather than direction, and no separation metric would show it — so the
    composition rides along on every row."""
    screener, _ = _screener([])
    cells = screener._subset_cells(("momentum.roc", "momentum.di_spread"))
    assert cells == {"scale-carrying": 1, "scale-free": 1}


def test_subset_cells_flag_unstamped_and_unregistered_columns():
    screener, _ = _screener([])
    cells = screener._subset_cells(("momentum.roc", "not_a_registered_column"))
    assert cells["unregistered"] == 1


@pytest.mark.slow
def test_subset_cells_reach_the_evaluation_raw_details():
    screener, _ = _screener([])
    result = screener.screen(["momentum.roc"], strategy=ScreenStrategy.EXHAUSTIVE, max_subset_size=1)
    ev = [e for e in result.evaluations if e.error is None][0]
    assert ev.raw_details["subset_cells"] == {"scale-carrying": 1}


# ── evaluator warnings are captured rather than dropped ───────────────────────────────────────────

def test_evaluator_warnings_are_captured_onto_the_subset(monkeypatch):
    """``label_util`` signals real problems through ``warnings.warn`` -- ranking fallbacks, monotonicity
    re-ranking, regimes disappearing after a dropna. Those escaped to stderr, where once-per-location
    dedup meant that on a several-hundred-subset screen only the FIRST was ever printed and no output
    row recorded any of them.
    """
    screener, _ = _screener([])

    def noisy_evaluator(**_kwargs):
        warnings.warn("ranking fallback applied", UserWarning)
        from okmich_quant_research.features.hmm_screener import AxisEvaluation
        return AxisEvaluation(0.5, 2.0, "n_significant_states", raw_details={"horizon": 18})

    monkeypatch.setattr("okmich_quant_research.features.hmm_screener.screener.get_evaluator",
                        lambda _axis: noisy_evaluator)
    _evaluation, captured = screener._call_evaluator(None, np.zeros(10, dtype=int), pd.DataFrame())
    assert any("ranking fallback applied" in w for w in captured)


# ── acceptance criterion 5: the union pool dedups at MAX_VIF = 3.0 ────────────────────────────────

def _union_pool(n_families: int = 12, per_family: int = 4, n_independent: int = 13,
                T: int = 4000, seed: int = 11) -> pd.DataFrame:
    """A 61-column pool with the measured redundancy structure of trend+momentum.

    The real 61-candidate pool lives in the lab's recipe builders, not in this library, so this
    reproduces its SHAPE: near-duplicate families (the norm_sma/ema/dema/tema group, the DI pair, the
    Aroon trio) plus weak-but-independent candidates that the filter must NOT remove.
    """
    rng = np.random.default_rng(seed)
    cols = {}
    for f in range(n_families):
        base = pd.Series(rng.normal(size=T)).rolling(20).mean()
        for k in range(per_family):
            cols[f"dup{f}_{k}"] = base + 0.02 * pd.Series(rng.normal(size=T))
    for i in range(n_independent):
        cols[f"indep{i}"] = pd.Series(rng.normal(size=T)).rolling(20).mean()
    return pd.DataFrame(cols).dropna()


def test_union_pool_collapses_to_its_independent_core_at_max_vif_three():
    """MAX_VIF = 3.0 is exactly the ``|r| >= 0.816`` ceiling, since ``VIF = 1/(1 - r**2)`` for a pair.

    NOTE what this does and does not prove. It pins the MECHANISM: every near-duplicate family collapses
    to one survivor and no surviving pair exceeds the ceiling. The measured "61 raw candidates -> 15-20
    survivors" figure needs the lab's real recipe pool and is a lab-side check; the recipes are not in
    this repository.
    """
    pool = _union_pool()
    assert pool.shape[1] == 61

    kept, report = stage0c_collinearity_filter(pool, max_vif=3.0, verbose=False)
    assert kept.shape[1] == 25                       # 12 families -> 12 survivors, + 13 independents
    assert report.n_before == 61

    corr = kept.corr().abs().to_numpy()
    np.fill_diagonal(corr, 0.0)
    assert corr.max() < 0.8165, "a surviving pair still exceeds the MAX_VIF=3.0 correlation ceiling"


def test_dedup_keeps_weak_but_independent_candidates():
    """Orthogonality is not edge, but the filter is a near-duplicate remover, not an orthogonaliser:
    dropping weak-but-independent features that combine would be the opposite of the intent."""
    pool = _union_pool()
    kept, _ = stage0c_collinearity_filter(pool, max_vif=3.0, verbose=False)
    assert sum(c.startswith("indep") for c in kept.columns) == 13


# ── a mapper returning nothing must not read as "did not separate" ────────────────────────────────

def test_diag_guard_checks_the_median_column_not_just_count():
    """``_weighted_separation_stats`` bails when EITHER ``count`` or the median column is absent, and
    returns ``(0.0, 0.0)``. Guarding only ``count`` left the same silent-zero hole open for a
    diagnostics frame that carries counts but not the axis target."""
    from okmich_quant_research.features.hmm_screener._evaluators import _diag_is_empty

    diag = pd.DataFrame({"count": [10, 10], "median": [0.1, -0.1]})
    assert not _diag_is_empty(diag, "median")
    assert _diag_is_empty(diag, "median_vol")          # counts present, target absent
    assert _diag_is_empty(pd.DataFrame(), "median")
    assert _diag_is_empty(None, "median")


@pytest.mark.parametrize("axis,label", [
    (Axis.DIRECTIONAL, "n_significant_states"),
    (Axis.VOLATILITY, "n_distinct_buckets"),
    (Axis.PATH_STRUCTURE, "n_distinct_scores"),
    (Axis.LIQUIDITY, "n_distinct_buckets"),
])
def test_every_evaluator_reports_an_empty_mapper_as_an_error(axis, label, monkeypatch):
    """All four, including LIQUIDITY, which originally had no guard at all: a total mapper failure has
    to surface as an error rather than as a legitimate-looking ``axis_separation == 0.0``."""
    import okmich_quant_research.features.hmm_screener._evaluators as ev

    monkeypatch.setattr(ev, "map_label_to_trend_direction", lambda *a, **k: ({}, pd.DataFrame()))
    monkeypatch.setattr(ev, "map_regime_to_volatility_score", lambda *a, **k: ({}, pd.DataFrame()))
    monkeypatch.setattr(ev, "map_regime_to_path_structure_score", lambda *a, **k: ({}, pd.DataFrame()))

    raw = _make_ohlc(600)
    labels = np.zeros(len(raw), dtype=int)
    labels[len(raw) // 2:] = 1
    out = get_evaluator(axis)(gamma=None, state_labels=labels, raw_data=raw, horizons=(12, 60),
                              primary_horizon=12)
    assert out.axis_separation == 0.0
    assert "no diagnostics" in out.raw_details["error"]
    assert out.secondary_label == label
