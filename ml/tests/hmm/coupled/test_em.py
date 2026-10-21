"""EM: monotone objective, recovery of planted coupling, no spurious coupling, warm starts, fixed drifts."""
import numpy as np
import pytest

from okmich_quant_ml.hmm.coupled import CoupledHMM, LocationScaleEmission

from ._planted import DIR, FULL, NONE, VOL, direction_vol_model, planted


@pytest.mark.parametrize("parents", [FULL, NONE])
def test_em_is_monotone(parents):
    _, X = planted().sample(3_000, seed=6)
    model = direction_vol_model(parents, n_init=1, max_iter=60, tol=0.0, transition_prior=0.0).fit(X)
    assert np.all(np.diff(model.fit_report.history) > -1e-8)


def test_recovers_planted_coupling():
    truth = planted()
    _, X = truth.sample(80_000, seed=7)
    model = direction_vol_model(FULL, n_init=2, random_state=1).fit(X)
    model.sort_states(DIR, model.emissions[0].mu).sort_states(VOL, model.emissions[1].mean)
    assert model.log_likelihood(X) >= truth.log_likelihood(X)
    np.testing.assert_allclose(model.emissions[0].mu, truth.emissions[0].mu, atol=0.05)
    np.testing.assert_allclose(model.emissions[0].scale, truth.emissions[0].scale, atol=0.05)
    # switch rates in burst rows are weakly identified (flat x burst ranged 0.056-0.072 over seeds, truth 0.08, with the
    # fit's likelihood above the truth's every time): sampling noise, not an EM fault
    switch = 1 - np.einsum("dvd->dv", model.transitions[0])
    np.testing.assert_allclose(switch, 1 - np.einsum("dvd->dv", truth.transitions[0]), atol=0.03)
    np.testing.assert_allclose(model.transitions[0], truth.transitions[0], atol=0.035)
    np.testing.assert_allclose(model.transitions[1], truth.transitions[1], atol=0.02)
    assert np.all(switch[:, 1] > 3 * switch[:, 0])


def test_coupled_fit_does_not_find_coupling_in_an_uncoupled_world():
    _, X = planted(parents=NONE).sample(80_000, seed=8)
    model = direction_vol_model(FULL, n_init=2, random_state=2).fit(X)
    model.sort_states(DIR, model.emissions[0].mu).sort_states(VOL, model.emissions[1].mean)
    switch = 1 - np.einsum("dvd->dv", model.transitions[0])
    ratio = switch[:, 1] / switch[:, 0]
    assert np.all((ratio > 0.6) & (ratio < 1.6))


def test_warm_start_from_the_factorial_fit_never_ends_below_it():
    _, X = planted().sample(5_000, seed=10)
    none = direction_vol_model(NONE, n_init=1, transition_prior=0.0).fit(X)
    full = direction_vol_model(FULL, n_init=0, transition_prior=0.0).fit(X, warm_start=none)
    assert full.fit_report.history[0] == pytest.approx(none.fit_report.loglik, abs=1e-6)
    assert full.fit_report.loglik >= none.fit_report.loglik - 1e-8
    assert full.fit_report.restart == -1


def test_fixed_drifts_stay_fixed_through_em_and_warm_starts():
    _, X = planted().sample(4_000, seed=11)
    fixed = np.array([-0.3, 0.0, 0.3])

    def model():
        emissions = [LocationScaleEmission(0, location=DIR, scale=VOL, fixed_mu=fixed),
                     direction_vol_model(FULL).emissions[1]]
        return CoupledHMM((3, 2), emissions, FULL, n_init=1)

    first = model().fit(X)
    np.testing.assert_array_equal(first.emissions[0].mu, fixed)
    other = model()
    other.emissions[0].fixed_mu = fixed * 2                    # a later fold with a different drift
    other.fit(X, warm_start=first)
    np.testing.assert_array_equal(other.emissions[0].mu, fixed * 2)
    assert first.emissions[0].n_free_params() == 2
