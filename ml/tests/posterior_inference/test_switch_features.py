"""Regime-switch features: switch_indicator, posterior_shift, last_transition, entry_confidence, typical_dwell, dwell_ratio."""

import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import DwellStatistic, dwell_length, dwell_ratio, entry_confidence, \
    last_transition, posterior_shift, switch_indicator, typical_dwell


def _one_hot_path(states, K: int = 3, top: float = 0.8) -> np.ndarray:
    """Posterior rows with ``top`` on the given state and the rest spread evenly."""
    probs = np.full((len(states), K), (1.0 - top) / (K - 1))
    probs[np.arange(len(states)), states] = top
    return probs


STATES = [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]


def test_switch_indicator_marks_argmax_changes():
    out = switch_indicator(_one_hot_path(STATES))
    np.testing.assert_array_equal(out, [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    assert out.dtype == np.int64


def test_posterior_shift_is_total_variation():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.25, 0.75]])
    np.testing.assert_allclose(posterior_shift(probs), [0.0, 0.0, 1.0, 0.25])


def test_posterior_shift_is_zero_on_static_uncertain_posterior():
    """The overlap form 1 − Σ p_{t-1} p_t would read 0.5 here although nothing moved."""
    np.testing.assert_allclose(posterior_shift(np.full((5, 2), 0.5)), 0.0)


def test_last_transition_tracks_most_recent_switch():
    out = last_transition(_one_hot_path([0, 0, 1, 1, 2, 0]))
    expected = [[-1, -1], [-1, -1], [0, 1], [0, 1], [1, 2], [2, 0]]
    np.testing.assert_array_equal(out, expected)


def test_entry_confidence_carries_the_opening_top_prob():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.45, 0.55], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])
    np.testing.assert_allclose(entry_confidence(probs), [0.9, 0.9, 0.55, 0.55, 0.55, 0.7])


def test_typical_dwell_excludes_censored_first_and_last_runs():
    # runs: 0×2 (left-censored), 1×3, 0×4, 1×2 (right-censored)
    out = typical_dwell(_one_hot_path(STATES))
    np.testing.assert_allclose(out[:2], [4.0, 3.0])
    assert np.isnan(out[2])


def test_typical_dwell_mean_statistic():
    states = [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]  # completed: 1×2, 0×4, 1×4
    np.testing.assert_allclose(typical_dwell(_one_hot_path(states), DwellStatistic.MEAN)[:2], [4.0, 3.0])
    np.testing.assert_allclose(typical_dwell(_one_hot_path(states), "median")[:2], [4.0, 3.0])


def test_dwell_ratio_divides_by_frozen_typical():
    probs = _one_hot_path(STATES)
    typical = np.array([4.0, 3.0, np.nan])
    expected = dwell_length(probs) / typical[np.argmax(probs, axis=1)]
    np.testing.assert_allclose(dwell_ratio(probs, typical), expected)
    assert np.isnan(dwell_ratio(_one_hot_path([2, 2]), typical)).all()


@pytest.mark.parametrize("typical", [np.array([4.0, 3.0]), np.array([4.0, 0.0, 1.0])])
def test_dwell_ratio_validates_typical(typical):
    with pytest.raises(ValueError, match="dwell_ratio"):
        dwell_ratio(_one_hot_path(STATES), typical)


@pytest.mark.parametrize("feature", [switch_indicator, posterior_shift, last_transition, entry_confidence,
                                     lambda p: dwell_ratio(p, np.array([4.0, 3.0, 2.0]))])
def test_switch_features_are_causal(feature):
    """Row t must not change when rows after t change."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3) * 0.5, size=60)
    full = feature(probs)
    for t in (0, 1, 7, 30, 59):
        np.testing.assert_array_equal(feature(probs[: t + 1])[t], full[t])


@pytest.mark.parametrize("feature", [switch_indicator, posterior_shift, last_transition, entry_confidence,
                                     typical_dwell, lambda p: dwell_ratio(p, np.ones(2))])
def test_switch_features_reject_nan(feature):
    bad = np.array([[0.5, 0.5], [np.nan, 0.5]])
    with pytest.raises(ValueError, match="NaN"):
        feature(bad)


@pytest.mark.parametrize("feature", [switch_indicator, posterior_shift, entry_confidence])
def test_switch_features_handle_empty_input(feature):
    assert feature(np.zeros((0, 3))).shape == (0,)


def test_last_transition_and_typical_dwell_on_empty_input():
    assert last_transition(np.zeros((0, 3))).shape == (0, 2)
    assert np.isnan(typical_dwell(np.zeros((0, 3)))).all()
