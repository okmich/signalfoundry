"""API: relabelling, clamped chains, persistence, model selection, validation."""
import numpy as np
import pytest

from okmich_quant_ml.hmm import CoupledHMM as Exported
from okmich_quant_ml.hmm.coupled import ClampedEmission, CoupledHMM, GaussianEmission, LocationScaleEmission

from ._planted import DIR, FULL, NONE, VOL, direction_vol_model, planted


def test_exported_from_the_hmm_package():
    assert Exported is CoupledHMM


def test_sort_states_relabels_without_changing_the_model():
    model = planted()
    _, X = model.sample(500, seed=9)
    before = model.log_likelihood(X)
    joint = model.filter(X)
    model.sort_states(DIR, np.array([2.0, 0.0, 1.0]))      # new order: old 1, old 2, old 0
    assert model.log_likelihood(X) == pytest.approx(before, abs=1e-9)
    d_before = joint.reshape(-1, 3, 2).sum(axis=2)
    np.testing.assert_allclose(model.chain_marginal(model.filter(X), DIR), d_before[:, [1, 2, 0]], atol=1e-12)


def test_clamped_chain_follows_the_observed_states():
    states, X = planted().sample(4_000, seed=11)
    X = np.column_stack([X, states[:, VOL].astype(float)])
    emissions = [LocationScaleEmission(0, location=DIR, scale=VOL), ClampedEmission(2, chain=VOL)]
    model = CoupledHMM((3, 2), emissions, [(DIR, VOL), (VOL,)], observed_chains=(VOL,), n_init=1).fit(X)
    vol = model.chain_marginal(model.filter(X), VOL)
    np.testing.assert_array_equal(vol.argmax(axis=1), states[:, VOL])
    assert np.allclose(vol.max(axis=1), 1.0)
    assert np.allclose(model.transitions[VOL], 0.5)


def test_clamped_column_must_hold_state_indices():
    _, X = planted().sample(200, seed=12)
    X = np.column_stack([X, np.full(200, 2.0)])              # chain VOL has states 0 and 1 only
    emissions = [LocationScaleEmission(0, location=DIR, scale=VOL), ClampedEmission(2, chain=VOL)]
    model = CoupledHMM((3, 2), emissions, [(DIR, VOL), (VOL,)], observed_chains=(VOL,), n_init=1)
    with pytest.raises(ValueError, match="state indices"):
        model.fit(X)


def test_save_and_load_round_trip(tmp_path):
    model = planted()
    _, X = model.sample(300, seed=13)
    path = tmp_path / "chmm.joblib"
    model.save(str(path))
    loaded = CoupledHMM.load(str(path))
    np.testing.assert_allclose(loaded.filter(X), model.filter(X))


def test_stationary_distribution_is_invariant():
    model = planted()
    p = model.stationary_distribution()
    np.testing.assert_allclose(p @ model.transition_matrix(), p, atol=1e-12)
    assert p.sum() == pytest.approx(1.0)


def test_model_selection_counts_coupling_parameters():
    full, none = direction_vol_model(FULL), direction_vol_model(NONE)
    # joint start 5; direction 6 rows x 2 vs 3 x 2; volatility 6 rows x 1 vs 2 x 1; emissions 3 + 2 + 2 + 2
    assert full.n_free_params() == 5 + 12 + 6 + 9
    assert none.n_free_params() == 5 + 6 + 2 + 9
    model = planted()
    _, X = model.sample(1_000, seed=14)
    aic, bic = model.get_aic_bic(X)
    assert bic - aic == pytest.approx(model.n_free_params() * (np.log(len(X)) - 2))


@pytest.mark.parametrize("kwargs, message", [
    ({"n_states": (3, 2), "parents": [(0,)]}, "one entry per chain"),
    ({"n_states": (3, 2), "parents": [(0, 5), (1,)]}, "chain indices"),
    ({"n_states": (3, 2), "observed_chains": (1,)}, "observed_chains"),
    ({"n_states": (16, 17)}, "joint states"),
])
def test_constructor_validation(kwargs, message):
    emissions = [GaussianEmission(0, chain=0), GaussianEmission(1, chain=1)]
    with pytest.raises(ValueError, match=message):
        CoupledHMM(emissions=emissions, **kwargs)


def test_input_validation():
    model = planted()
    with pytest.raises(ValueError, match="2-D"):
        model.filter(np.zeros(10))
    with pytest.raises(ValueError, match="columns"):
        model.filter(np.zeros((10, 1)))
    with pytest.raises(ValueError, match="infinite"):
        model.filter(np.array([[0.0, np.inf]] * 3))
    with pytest.raises(ValueError, match="lengths"):
        model.filter(np.zeros((10, 2)), lengths=[4, 5])
    with pytest.raises(RuntimeError, match="not fitted"):
        direction_vol_model(FULL).filter(np.zeros((10, 2)))
    with pytest.raises(ValueError, match="different chains"):
        LocationScaleEmission(0, location=1, scale=1)
