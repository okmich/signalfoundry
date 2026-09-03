"""The axis layer: derived eligibility, the shipped invariance stamps, and the tag/axis boundary."""
import pytest

from okmich_quant_research.features.registry import (AXIS_PRIMARY_HORIZON, AXIS_SIGNAL_TYPES, Axis,
                                                     FeatureEntry, FeatureInvariance, FeatureRegistry,
                                                     INVARIANCE_STAMPS, PRICE_PATH_AXES, Parity,
                                                     SIGNAL_TYPES, ScaleClass, UNSTAMPED_MEASUREMENTS,
                                                     is_eligible)


@pytest.fixture(scope="module")
def reg():
    return FeatureRegistry()


def _entry(signal_type, invariance=None, name="probe_feature", module="unit"):
    return FeatureEntry(name=name, module=module, signal_type=signal_type, description="d",
                        invariance=invariance)


# ── the enum itself ───────────────────────────────────────────────────────────────────────────────

def test_axis_has_exactly_four_members():
    """``momentum`` is gone (it was trend at half the lookback) and ``acceleration`` was killed after
    measurement — do not re-add either."""
    assert {a.value for a in Axis} == {"directional", "path_structure", "volatility", "liquidity"}


def test_liquidity_is_not_a_price_path_axis():
    """Deliberate asymmetry: liquidity reads a volume/order-flow substrate, so reflecting the PRICE
    path leaves it invariant and the invariance test carries no information about it. These four are
    not four of a kind, and nobody should later 'fix' that."""
    assert PRICE_PATH_AXES == frozenset({Axis.DIRECTIONAL, Axis.PATH_STRUCTURE, Axis.VOLATILITY})
    assert Axis.LIQUIDITY not in PRICE_PATH_AXES


def test_every_axis_has_a_tag_set_and_a_horizon():
    assert set(AXIS_SIGNAL_TYPES) == set(Axis)
    assert set(AXIS_PRIMARY_HORIZON) == set(Axis)


def test_axis_tag_sets_name_only_real_signal_types():
    for axis, tags in AXIS_SIGNAL_TYPES.items():
        assert not set(tags) - set(SIGNAL_TYPES), f"{axis} names an unknown signal_type"


def test_directional_draws_on_both_former_axes_tags():
    """The collapse in one line: ``trend`` and ``momentum`` are tags of ONE axis now."""
    assert {"trend", "momentum"} <= AXIS_SIGNAL_TYPES[Axis.DIRECTIONAL]


# ── the tag/axis boundary is enforced, not bridged ────────────────────────────────────────────────
#
# There is deliberately no ``signal_type -> Axis`` mapping helper in this package, and no path by which
# a tag can stand in for an axis. The two are different kinds of thing: a tag says what family an
# indicator came from, an axis says what partition a model is being asked to separate. Conflating them
# is the entire bug this layer exists to remove, so the boundary is asserted rather than bridged. What
# the library owes anyone holding output filed under a tag name is a clear statement of which axis it
# corresponds to, and that is carried in the errors below and in test_config / test_axis_screening.

@pytest.mark.parametrize("signal_type", ["trend", "momentum", "price_structure"])
def test_a_signal_type_is_not_accepted_as_an_axis(signal_type):
    """A feature TAG must never be usable where an axis is expected — that conflation is the whole bug."""
    assert signal_type in SIGNAL_TYPES
    with pytest.raises(ValueError):
        Axis(signal_type)


def test_the_two_former_directional_tags_have_exactly_one_axis_between_them():
    """``trend`` and ``momentum`` were two axes and are now one; both tags feed DIRECTIONAL and there is
    no second directional axis for them to disagree about."""
    directional_tags = {"trend", "momentum"} & AXIS_SIGNAL_TYPES[Axis.DIRECTIONAL]
    assert directional_tags == {"trend", "momentum"}
    assert [a for a in Axis if {"trend", "momentum"} & AXIS_SIGNAL_TYPES[a]] == [Axis.DIRECTIONAL]


# ── acceptance criterion 4: eligibility is DERIVED ────────────────────────────────────────────────

def test_momentum_tagged_odd_feature_is_accepted_by_directional():
    """The headline case: namespace says ``momentum``, measurement says ODD, so it is directional."""
    e = _entry("momentum", FeatureInvariance(Parity.ODD, ScaleClass.CARRYING))
    ok, _ = is_eligible(e, Axis.DIRECTIONAL)
    assert ok


def test_even_scale_free_feature_is_rejected_from_directional_and_accepted_by_path_structure():
    """``path_structure.efficiency_ratio`` is tagged ``regime``, which BOTH axes draw on. Only the
    measured stamp separates them — which is the whole point of the invariance gate."""
    e = _entry("regime", FeatureInvariance(Parity.EVEN, ScaleClass.FREE))
    assert "regime" in AXIS_SIGNAL_TYPES[Axis.DIRECTIONAL]        # tag gate alone would admit it
    assert not is_eligible(e, Axis.DIRECTIONAL)[0]
    assert is_eligible(e, Axis.PATH_STRUCTURE)[0]


def test_even_scale_carrying_feature_belongs_to_volatility_not_path_structure():
    e = _entry("regime", FeatureInvariance(Parity.EVEN, ScaleClass.CARRYING))
    assert is_eligible(e, Axis.VOLATILITY)[0]
    assert not is_eligible(e, Axis.PATH_STRUCTURE)[0]


def test_mixed_feature_belongs_to_no_price_path_axis():
    """MIXED confounds direction with magnitude. That is a defect in the feature, not a gap in the
    taxonomy, and it must not be given a home."""
    e = _entry("regime", FeatureInvariance(Parity.MIXED, ScaleClass.FREE))
    for axis in PRICE_PATH_AXES:
        ok, reason = is_eligible(e, axis)
        assert not ok
        assert "MIXED" in reason


def test_wrong_tag_is_rejected_before_invariance_is_consulted():
    e = _entry("liquidity", FeatureInvariance(Parity.ODD, ScaleClass.CARRYING))
    ok, reason = is_eligible(e, Axis.DIRECTIONAL)
    assert not ok
    assert "signal_type" in reason


def test_unstamped_feature_is_admitted_with_a_reason_naming_the_gap():
    """'Never measured' and 'measured and wrong' are different findings and must not collapse into one:
    an unmeasured feature is a coverage gap in the probe, not a defect in the feature."""
    ok, reason = is_eligible(_entry("trend", None), Axis.DIRECTIONAL)
    assert ok
    assert "no invariance stamp" in reason


def test_unscored_stamp_is_admitted_with_a_reason():
    e = _entry("trend", FeatureInvariance(Parity.UNSCORED, ScaleClass.UNSCORED))
    ok, reason = is_eligible(e, Axis.DIRECTIONAL)
    assert ok
    assert "UNSCORED" in reason


def test_liquidity_is_tag_gated_only_whatever_the_parity_says():
    """Reflection cannot classify a volume feature, so it must not be allowed to veto one."""
    for parity in (Parity.ODD, Parity.EVEN, Parity.MIXED):
        e = _entry("liquidity", FeatureInvariance(parity, ScaleClass.FREE))
        assert is_eligible(e, Axis.LIQUIDITY)[0]


def test_one_sided_is_admitted_to_directional_but_carries_a_loud_reason():
    """It belongs on the axis — it is half of an ODD pair — but is not safe alone, and the caller has
    to be told which half it has."""
    e = _entry("trend", FeatureInvariance(Parity.ONE_SIDED, ScaleClass.FREE, conjugate="unit.other"))
    ok, reason = is_eligible(e, Axis.DIRECTIONAL)
    assert ok
    assert "ONE-SIDED" in reason and "unit.other" in reason


def test_no_hand_maintained_membership_list_exists():
    """Eligibility must stay derived. A per-axis list of feature NAMES anywhere would re-open exactly
    the hole this layer closed, so assert the axis module carries no such thing."""
    from okmich_quant_research.features.registry import _axis

    for name, value in vars(_axis).items():
        if name.startswith("__") or not isinstance(value, dict):
            continue
        for key in value:
            assert not (isinstance(key, str) and "." in key), (
                f"_axis.{name} is keyed by qualified feature names -- axis membership must be derived "
                f"from measured invariance, not listed"
            )


# ── acceptance criterion 2: the shipped one-sided stamps ──────────────────────────────────────────

@pytest.mark.parametrize("feature,conjugate", [
    ("momentum.minus_di", "momentum.plus_di"),
    ("momentum.plus_di", "momentum.minus_di"),
    ("timothymasters.trend.aroon_down", "timothymasters.trend.aroon_up"),
    ("timothymasters.trend.aroon_up", "timothymasters.trend.aroon_down"),
])
def test_measured_one_sided_pairs_are_stamped_with_their_conjugates(reg, feature, conjugate):
    """These four are the registry-level defect the work order exists to fix, and ``minus_di`` is the
    feature that won the trend axis on 11 of 14 FX symbols."""
    inv = reg.get(feature).invariance
    assert inv is not None, f"{feature} must carry a measured stamp"
    assert inv.parity is Parity.ONE_SIDED
    assert inv.conjugate == conjugate
    assert inv.measured_on, "a stamp without provenance is just another assertion"


def test_canonical_odd_spreads_are_stamped_odd(reg):
    """The fix for a one-sided pair: the spread, which is odd by construction."""
    for feature in ("momentum.di_spread", "timothymasters.trend.aroon_diff"):
        assert reg.get(feature).invariance.parity is Parity.ODD


def test_di_spread_is_registered_and_directional(reg):
    e = reg.get("momentum.di_spread")
    assert e.signal_type == "trend"
    assert e.directional
    assert is_eligible(e, Axis.DIRECTIONAL)[0]


def test_known_odd_and_even_features_are_stamped_as_measured(reg):
    assert reg.get("momentum.roc").invariance.parity is Parity.ODD
    assert reg.get("momentum.roc").invariance.scale_class is ScaleClass.CARRYING
    assert reg.get("path_structure.efficiency_ratio").invariance.parity is Parity.EVEN
    assert reg.get("path_structure.efficiency_ratio").invariance.scale_class is ScaleClass.FREE


def test_directional_efficiency_index_is_odd_despite_its_name(reg):
    """It reads as an efficiency (unsigned) measure; its own docstring calls it a signed trend-strength
    measure; it tests ODD. Exactly the case a namespace cannot decide."""
    assert reg.get("momentum.directional_efficiency_index").invariance.parity is Parity.ODD


# ── stamp file health ─────────────────────────────────────────────────────────────────────────────

def test_every_shipped_stamp_matches_a_catalog_entry():
    """A stamped name with no catalogue entry is a coverage gap worth seeing rather than an error, but
    right now there are none and a regression should be visible."""
    assert UNSTAMPED_MEASUREMENTS == []


def test_conjugates_are_symmetric_and_resolvable(reg):
    """A one-sided feature's conjugate must itself be catalogued and point back — an unresolvable
    conjugate would make the screener's one-sided guard silently useless."""
    for name, stamp in INVARIANCE_STAMPS.items():
        if stamp.parity is not Parity.ONE_SIDED:
            assert stamp.conjugate == "", f"{name} carries a conjugate without being ONE_SIDED"
            continue
        partner = reg.get(stamp.conjugate)
        assert partner.invariance.parity is Parity.ONE_SIDED
        assert partner.invariance.conjugate == name


def test_registry_query_surface_reflects_measurement(reg):
    assert len(reg.by_parity(Parity.ONE_SIDED)) == 4
    assert len(reg.by_parity(Parity.ODD)) > 40
    assert set(reg.eligible_for(Axis.DIRECTIONAL).qualified_names()) >= {"momentum.roc", "momentum.di_spread"}


def test_summary_exposes_the_measured_columns(reg):
    df = reg.summary()
    assert {"parity", "scale_class", "conjugate"} <= set(df.columns)
    row = df[df.name == "minus_di"].iloc[0]
    assert row.parity == Parity.ONE_SIDED.value
    assert row.conjugate == "momentum.plus_di"
