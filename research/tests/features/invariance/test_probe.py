"""Classifier behaviour, pinned on the numbers actually measured on the corpus.

``test_transforms.py`` proves the transforms and the round-trip on synthetic bars. This file proves the
classification of the cases synthetic bars cannot produce — chiefly ONE_SIDED, whose signature
(``refl_corr`` mid-band, conjugate at ~1) only arises on real, asymmetric price paths.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.invariance import (PARITY_BAND, SCALE_CARRY, SCALE_FREE, VIF3_R,
                                                       aggregate_stamps, classify_cell, classify_parity,
                                                       classify_scale, cross_correlations, iqr,
                                                       nearest_neighbour_redundancy, probe_invariance,
                                                       stamp_summary, unscored, write_stamps_csv)
from okmich_quant_research.features.registry import Parity, ScaleClass, load_invariance_stamps

# Medians measured over EURUSD.r / GBPUSD.r / USDCAD.r / US500.r, FXPIG-Server M5, 80k bars each.
MEASURED_MINUS_DI_REFL = -0.687
MEASURED_AROON_UP_REFL = -0.436


# ── parity classification ─────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("refl,expected", [
    (-1.0, Parity.ODD),
    (-0.9999, Parity.ODD),
    (-PARITY_BAND, Parity.ODD),
    (1.0, Parity.EVEN),
    (PARITY_BAND, Parity.EVEN),
    (float("nan"), Parity.UNSCORED),
])
def test_parity_of_a_self_matching_feature(refl, expected):
    assert classify_parity(refl, conj_corr=float("nan"), conj_is_self=True) is expected


@pytest.mark.parametrize("refl", [MEASURED_MINUS_DI_REFL, MEASURED_AROON_UP_REFL])
def test_measured_one_sided_signature_classifies_as_one_sided(refl):
    """The exact numbers from the corpus: mid-band reflection, conjugate at ~1.

    This is the case that decides whether a directional screen is measuring direction at all.
    ``momentum.minus_di`` won the trend axis on 11 of 14 FX symbols while being half of an odd pair.
    """
    assert classify_parity(refl, conj_corr=1.0, conj_is_self=False) is Parity.ONE_SIDED


def test_mid_band_without_a_conjugate_is_mixed_not_one_sided():
    """A feature that fails the odd test and has no partner genuinely confounds direction with
    magnitude. That is a defect, and it must not be quietly filed as a one-sided half-pair."""
    assert classify_parity(-0.687, conj_corr=float("nan"), conj_is_self=True) is Parity.MIXED


def test_a_weak_conjugate_does_not_rescue_a_mixed_feature():
    assert classify_parity(-0.5, conj_corr=0.6, conj_is_self=False) is Parity.MIXED


def test_self_match_is_never_one_sided():
    """``conj_is_self`` guards the whole distinction: a feature is not its own conjugate."""
    assert classify_parity(-0.5, conj_corr=1.0, conj_is_self=True) is Parity.MIXED


# ── scale classification ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("exponent,expected", [
    (1.0, ScaleClass.CARRYING),
    (SCALE_CARRY, ScaleClass.CARRYING),
    (0.0, ScaleClass.FREE),
    (SCALE_FREE, ScaleClass.FREE),
    (0.4, ScaleClass.PARTIAL),
    (float("nan"), ScaleClass.UNSCORED),
])
def test_scale_classification_bands(exponent, expected):
    assert classify_scale(exponent) is expected


def test_cells_cover_the_taxonomy():
    assert "signed drift" in classify_cell(Parity.ODD, ScaleClass.CARRYING)
    assert "normalised direction" in classify_cell(Parity.ODD, ScaleClass.FREE)
    assert "volatility" in classify_cell(Parity.EVEN, ScaleClass.CARRYING)
    assert "path shape" in classify_cell(Parity.EVEN, ScaleClass.FREE)
    assert "one-sided" in classify_cell(Parity.ONE_SIDED, ScaleClass.FREE)


def test_one_sided_cell_ignores_scale_class():
    """One-sidedness is a defect in what the feature can express; its scale class is beside the point."""
    for sc in (ScaleClass.CARRYING, ScaleClass.FREE, ScaleClass.UNSCORED):
        assert classify_cell(Parity.ONE_SIDED, sc) == classify_cell(Parity.ONE_SIDED, ScaleClass.FREE)


# ── the IQR degeneracy guard ──────────────────────────────────────────────────────────────────────

def test_iqr_refuses_a_near_constant_column_with_outliers():
    """Measured failure: ``velocity_consistency`` has IQR ~0 and range ~1, which turned a meaningless
    perturbation into a response of 1e11 and put a broken feature at the top of the table. Unscoreable
    is the honest verdict, and it is itself a defect worth seeing."""
    v = np.zeros(1000)
    v[:3] = [1e6, -1e6, 5e5]
    assert np.isnan(iqr(pd.Series(v)))


def test_iqr_refuses_too_few_observations():
    assert np.isnan(iqr(pd.Series(np.arange(10, dtype=float))))


def test_iqr_of_a_healthy_column_is_positive():
    assert iqr(pd.Series(np.random.default_rng(0).normal(size=1000))) > 0


# ── correlation plumbing ──────────────────────────────────────────────────────────────────────────

def test_cross_correlations_do_not_collapse_shared_column_names():
    """The a|/b| prefixing is load-bearing.

    A pool and its own reflection share every column name. Without the prefix pandas aligns the two
    into one column, self-correlation reads 1.0, and EVERY feature classifies as EVEN — a result that
    looks entirely plausible and is entirely wrong.
    """
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"f": rng.normal(size=500)})
    b = pd.DataFrame({"f": -a["f"]})
    assert cross_correlations(a, b).at["f", "f"] == pytest.approx(-1.0)


def test_vif3_r_matches_the_max_vif_of_three_ceiling():
    """``VIF = 1/(1 - r**2)`` for a pair, so MAX_VIF = 3.0 is exactly |r| >= sqrt(2/3)."""
    assert VIF3_R == pytest.approx(0.8165, abs=5e-5)
    assert 1.0 / (1.0 - VIF3_R ** 2) == pytest.approx(3.0)


def test_nearest_neighbour_redundancy_finds_the_duplicate():
    rng = np.random.default_rng(1)
    base = rng.normal(size=800)
    pool_a = pd.DataFrame({"a1": base + 0.01 * rng.normal(size=800), "a2": rng.normal(size=800)})
    pool_b = pd.DataFrame({"b1": base, "b2": rng.normal(size=800)})
    out = nearest_neighbour_redundancy(pool_a, pool_b).set_index("feature")
    assert out.at["a1", "nearest"] == "b1"
    assert out.at["a1", "abs_r"] > VIF3_R
    assert out.at["a2", "abs_r"] < VIF3_R


# ── aggregation across symbols ────────────────────────────────────────────────────────────────────

def _probe_rows(rows):
    return pd.DataFrame(rows, columns=["symbol", "feature", "refl_corr", "parity", "scale_exp",
                                       "scale_class", "cell", "conjugate", "conjugate_corr", "n_valid"])


def test_aggregate_reclassifies_from_raw_responses_not_from_the_parity_column():
    """A parity that is not stable across symbols is not a property of the feature.

    Every row below CLAIMS ``odd`` in its parity column, but the underlying reflections median to
    -0.50 — mid-band. Aggregation must re-derive the verdict from the raw numbers and return MIXED,
    not trust the per-symbol labels. Voting on labels would launder an unstable feature into a clean
    stamp, which is precisely the declared-over-measured failure this layer exists to prevent.
    """
    rows = [("s1", "f", -0.99, "odd", 0.0, "scale-free", "", "", float("nan"), 900),
            ("s2", "f", -0.50, "odd", 0.0, "scale-free", "", "", float("nan"), 900),
            ("s3", "f", -0.10, "odd", 0.0, "scale-free", "", "", float("nan"), 900)]
    assert np.median([-0.99, -0.50, -0.10]) == -0.50
    assert aggregate_stamps(_probe_rows(rows), measured_on="unit-test")["f"].parity is Parity.MIXED


def test_aggregate_keeps_a_stable_odd_verdict():
    rows = [("s1", "f", -0.99, "odd", 1.0, "scale-carrying", "", "", float("nan"), 900),
            ("s2", "f", -0.97, "odd", 1.0, "scale-carrying", "", "", float("nan"), 900)]
    stamps = aggregate_stamps(_probe_rows(rows), measured_on="unit-test")
    assert stamps["f"].parity is Parity.ODD
    assert stamps["f"].scale_class is ScaleClass.CARRYING
    assert stamps["f"].conjugate == ""                 # conjugate is set only for ONE_SIDED


def test_aggregate_records_the_conjugate_for_a_one_sided_feature():
    rows = [("s1", "a", MEASURED_MINUS_DI_REFL, "one-sided", 0.0, "scale-free", "", "b", 1.0, 900),
            ("s2", "a", MEASURED_MINUS_DI_REFL, "one-sided", 0.0, "scale-free", "", "b", 1.0, 900)]
    stamps = aggregate_stamps(_probe_rows(rows), measured_on="unit-test")
    assert stamps["a"].parity is Parity.ONE_SIDED
    assert stamps["a"].conjugate == "b"


def test_aggregate_of_nothing_is_nothing():
    assert aggregate_stamps(_probe_rows([]), measured_on="unit-test") == {}


def test_unscored_lists_only_unmeasurable_stamps():
    rows = [("s1", "good", -0.99, "odd", 1.0, "scale-carrying", "", "", float("nan"), 900),
            ("s1", "bad", float("nan"), "unscored", float("nan"), "unscored", "", "", float("nan"), 900)]
    stamps = aggregate_stamps(_probe_rows(rows), measured_on="unit-test")
    assert unscored(stamps) == ["bad"]


# ── round-trip through the on-disk stamp format ───────────────────────────────────────────────────

def test_stamps_round_trip_through_the_csv_the_registry_reads(tmp_path):
    """``write_stamps_csv`` must emit exactly what ``registry._invariance`` parses back — otherwise a
    re-measurement silently produces a file the registry cannot load, and every feature reverts to
    unstamped, which reads identically to 'the probe was never run'."""
    rows = [("s1", "a", MEASURED_MINUS_DI_REFL, "one-sided", 0.0, "scale-free", "", "b", 1.0, 900),
            ("s1", "b", -0.99, "odd", 1.0, "scale-carrying", "", "", float("nan"), 900)]
    stamps = aggregate_stamps(_probe_rows(rows), measured_on="unit-test corpus, 2026-09-03")
    path = tmp_path / "_invariance.csv"
    write_stamps_csv(stamps, path)

    reloaded = load_invariance_stamps(path)
    assert reloaded == stamps
    assert reloaded["a"].measured_on == "unit-test corpus, 2026-09-03"


def test_stamp_summary_is_one_row_per_feature():
    rows = [("s1", "a", -0.99, "odd", 1.0, "scale-carrying", "", "", float("nan"), 900)]
    summary = stamp_summary(aggregate_stamps(_probe_rows(rows), measured_on="x"))
    assert list(summary["feature"]) == ["a"]
    assert "signed drift" in summary.at[0, "cell"]


# ── probe input contract ──────────────────────────────────────────────────────────────────────────

def test_probe_rejects_a_pool_with_no_common_column():
    """A feature that throws on a transformed path has no measurable parity; reporting one anyway
    would be a fabrication."""
    raw = pd.DataFrame({"open": [1.0] * 400, "high": [1.1] * 400, "low": [0.9] * 400, "close": [1.0] * 400})

    calls = {"n": 0}

    def fickle(df):
        calls["n"] += 1
        return pd.DataFrame({f"col{calls['n']}": np.arange(len(df), dtype=float)}, index=df.index)

    with pytest.raises(ValueError, match="no column present in all three"):
        probe_invariance(raw, fickle)
