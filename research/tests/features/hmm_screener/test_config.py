import pytest

from okmich_quant_research.features.hmm_screener import HmmScreenerConfig, OneSidedPolicy, ScreenStrategy
from okmich_quant_research.features.registry import AXIS_SIGNAL_TYPES, Axis


def test_config_minimal_construction_succeeds() -> None:
    c = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=4)
    assert c.axis is Axis.DIRECTIONAL
    assert c.algo == "hmm_lambda"
    assert c.n_states == 4
    assert c.allowed_signal_types is None


def test_config_accepts_axis_as_string() -> None:
    """Ergonomics: the enum's own value works, so callers need not import Axis to pass one."""
    assert HmmScreenerConfig(axis="directional", algo="hmm_lambda", n_states=2).axis is Axis.DIRECTIONAL


@pytest.mark.parametrize("signal_type", ["trend", "momentum", "price_structure"])
def test_config_rejects_a_signal_type_as_axis(signal_type) -> None:
    """A feature tag is not an axis and cannot be used as one — there is no bridge between them."""
    with pytest.raises(ValueError, match="not a valid Axis"):
        HmmScreenerConfig(axis=signal_type, algo="hmm_lambda", n_states=3)


@pytest.mark.parametrize("signal_type,axis_name", [
    ("trend", "Axis.DIRECTIONAL"),
    ("momentum", "Axis.DIRECTIONAL"),
])
def test_config_signal_type_error_names_the_corresponding_axis(signal_type, axis_name) -> None:
    """Rejecting is not enough: anyone holding output filed under a tag name needs to be told which
    axis it corresponds to, and the error is the only place that says so."""
    with pytest.raises(ValueError, match="feature TAG"):
        HmmScreenerConfig(axis=signal_type, algo="hmm_lambda", n_states=3)
    with pytest.raises(ValueError, match=axis_name.replace(".", r"\.")):
        HmmScreenerConfig(axis=signal_type, algo="hmm_lambda", n_states=3)


def test_config_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="not a valid Axis"):
        HmmScreenerConfig(axis="not_a_real_axis", algo="hmm_lambda", n_states=3)


# ── allowed_signal_types: still feature TAGS, now defaulted from the axis ──────────────────────────

def test_effective_allowed_signal_types_defaults_to_the_axis_tag_set() -> None:
    """Was ``frozenset({signal_type})``, which only worked while axis and tag were one string.

    DIRECTIONAL legitimately draws on trend, momentum AND regime — only 9 of the 23 candidates in the
    measured trend pool were tagged ``trend``, so the old default flagged as "off-axis" exactly the
    cross-namespace features that turned out to be on-axis.
    """
    c = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=4)
    assert c.effective_allowed_signal_types == AXIS_SIGNAL_TYPES[Axis.DIRECTIONAL]
    assert {"trend", "momentum", "regime"} == set(c.effective_allowed_signal_types)


def test_effective_allowed_signal_types_explicit_override() -> None:
    c = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=3,
                          allowed_signal_types=frozenset({"trend"}))
    assert c.effective_allowed_signal_types == frozenset({"trend"})


def test_config_rejects_unknown_allowed_signal_type() -> None:
    with pytest.raises(ValueError, match="allowed_signal_types"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=3,
                          allowed_signal_types=frozenset({"trend", "made_up"}))


# ── per-axis primary horizon ───────────────────────────────────────────────────────────────────────

def test_directional_primary_horizon_defaults_to_18() -> None:
    """Measured: the directional label's unconditional separation peaks at H=18 (0.091), not 12 (0.080).

    The entire existing corpus was screened at 12 only because ``horizons[0]`` was hard-coded as the
    primary horizon for every axis.
    """
    assert HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=2).effective_primary_horizon == 18


@pytest.mark.parametrize("axis", [Axis.VOLATILITY, Axis.PATH_STRUCTURE, Axis.LIQUIDITY])
def test_other_axes_keep_the_inherited_horizon_of_12(axis) -> None:
    """12 is inherited, not measured: these axes have never been HMM-screened on this corpus."""
    assert HmmScreenerConfig(axis=axis, algo="hmm_lambda", n_states=2).effective_primary_horizon == 12


def test_explicit_primary_horizon_overrides_the_axis_default() -> None:
    c = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=2, primary_horizon=24)
    assert c.effective_primary_horizon == 24


def test_primary_horizon_is_independent_of_horizons_tuple() -> None:
    """``horizons`` is unchanged; the primary horizon is no longer silently ``horizons[0]``."""
    c = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=2, horizons=(5, 60))
    assert c.horizons == (5, 60)
    assert c.effective_primary_horizon == 18


def test_config_rejects_non_positive_primary_horizon() -> None:
    with pytest.raises(ValueError, match="primary_horizon"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=2, primary_horizon=0)


# ── new capability / policy fields ─────────────────────────────────────────────────────────────────

def test_has_real_volume_defaults_false() -> None:
    """Refusing LIQUIDITY by default is the point: tick_volume is a tick COUNT on MT5 feeds."""
    assert HmmScreenerConfig(axis=Axis.LIQUIDITY, algo="hmm_lambda", n_states=2).has_real_volume is False


def test_one_sided_policy_defaults_to_warn() -> None:
    c = HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=2)
    assert c.one_sided_policy is OneSidedPolicy.WARN


def test_one_sided_policy_has_no_substitute_option() -> None:
    """Substitution would corrupt the beam's dedup bookkeeping and make the reported subset differ
    from the subset actually fitted. Substituting is the caller's job, via the candidate pool."""
    assert {p.value for p in OneSidedPolicy} == {"warn", "exclude", "raise"}


# ── unchanged validation ───────────────────────────────────────────────────────────────────────────

def test_config_rejects_unknown_algo() -> None:
    with pytest.raises(ValueError, match="algo"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_xyz", n_states=3)


def test_config_rejects_n_states_less_than_two() -> None:
    with pytest.raises(ValueError, match="n_states"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=1)


def test_config_rejects_out_of_range_honesty_threshold() -> None:
    with pytest.raises(ValueError, match="honesty_threshold"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=3, honesty_threshold=1.5)


def test_screen_strategy_values() -> None:
    assert ScreenStrategy.ABLATION.value == "ablation"
    assert ScreenStrategy.EXHAUSTIVE.value == "exhaustive"


def test_config_min_significant_states_default_is_two() -> None:
    assert HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=4).min_significant_states == 2


def test_config_max_balance_ratio_default_is_ten() -> None:
    assert HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=4).max_balance_ratio == 10.0


def test_config_rejects_min_significant_states_below_one() -> None:
    with pytest.raises(ValueError, match="min_significant_states"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=4, min_significant_states=0)


def test_config_rejects_min_significant_states_above_n_states() -> None:
    with pytest.raises(ValueError, match="min_significant_states"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=3, min_significant_states=4)


def test_config_rejects_max_balance_ratio_below_one() -> None:
    with pytest.raises(ValueError, match="max_balance_ratio"):
        HmmScreenerConfig(axis=Axis.DIRECTIONAL, algo="hmm_lambda", n_states=4, max_balance_ratio=0.5)
