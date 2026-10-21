"""Exact inference: brute-force path enumeration, missing data, factorisation, sequences, causality."""
import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from okmich_quant_ml.hmm import InferenceMode
from okmich_quant_ml.hmm.coupled import CoupledHMM, GaussianEmission

from ._planted import DIR, NONE, VOL, planted


def brute_force(model: CoupledHMM, X: np.ndarray):
    """log p(x), filtered and smoothed joint posteriors by summing over every joint path."""
    log_b = model._log_emissions(X)
    log_t = np.log(model.transition_matrix())
    log_pi = np.log(model.pi)
    n, j = log_b.shape
    paths = np.array(list(itertools.product(range(j), repeat=n)))
    score = log_pi[paths[:, 0]] + log_b[0, paths[:, 0]]
    prefix = [score.copy()]
    for t in range(1, n):
        score = score + log_t[paths[:, t - 1], paths[:, t]] + log_b[t, paths[:, t]]
        prefix.append(score.copy())
    total = logsumexp(score)
    smoothed = np.array([[np.exp(logsumexp(score[paths[:, t] == k]) - total) for k in range(j)] for t in range(n)])
    filtered = np.array([[np.exp(logsumexp(prefix[t][paths[:, t] == k]) - logsumexp(prefix[t])) for k in range(j)]
                         for t in range(n)])
    return total, filtered, smoothed


def test_inference_matches_path_enumeration():
    model = planted()
    _, X = model.sample(6, seed=3)
    total, filtered, smoothed = brute_force(model, X)
    assert model.log_likelihood(X) == pytest.approx(total, abs=1e-10)
    np.testing.assert_allclose(model.filter(X), filtered, atol=1e-10)
    np.testing.assert_allclose(model.smooth(X), smoothed, atol=1e-10)


def test_missing_observation_contributes_nothing():
    model = planted()
    _, X = model.sample(5, seed=4)
    X[2, 0] = np.nan
    total, filtered, _ = brute_force(model, X)
    assert model.log_likelihood(X) == pytest.approx(total, abs=1e-10)
    np.testing.assert_allclose(model.filter(X), filtered, atol=1e-10)


def test_uncoupled_posterior_factorises_into_chain_filters():
    """With separate emissions per chain and no coupling, the joint filter is the product of single-chain filters."""
    emissions = [GaussianEmission(0, chain=DIR, tied=True), GaussianEmission(1, chain=VOL)]
    model = CoupledHMM((3, 2), emissions, NONE)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    model._init_params(X, rng, 0)
    model.transitions = [np.array([[0.9, 0.06, 0.04], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]]),
                         np.array([[0.95, 0.05], [0.1, 0.9]])]
    model.pi = np.outer([0.2, 0.5, 0.3], [0.6, 0.4]).ravel()
    joint = model.filter(X)

    def single(x, mean, var, trans, pi):
        out = np.empty((len(x), len(mean)))
        p = pi
        for t in range(len(x)):
            if t:
                p = out[t - 1] @ trans
            lik = np.exp(-0.5 * (x[t] - mean) ** 2 / var) / np.sqrt(var)
            out[t] = p * lik / (p * lik).sum()
        return out

    d = single(X[:, 0], emissions[0].mean, emissions[0].var, model.transitions[0], np.array([0.2, 0.5, 0.3]))
    v = single(X[:, 1], emissions[1].mean, emissions[1].var, model.transitions[1], np.array([0.6, 0.4]))
    np.testing.assert_allclose(joint, (d[:, :, None] * v[:, None, :]).reshape(len(X), 6), atol=1e-10)


def test_sequences_restart_from_the_initial_distribution():
    model = planted()
    _, X = model.sample(300, seed=5)
    split = model.log_likelihood(X, lengths=[120, 180])
    assert split == pytest.approx(model.log_likelihood(X[:120]) + model.log_likelihood(X[120:]), abs=1e-9)


def test_filter_is_causal():
    """Appending bars never changes an earlier filtered posterior or predictive density."""
    model = planted()
    _, X = model.sample(500, seed=6)
    full, part = model.filter(X), model.filter(X[:300])
    np.testing.assert_allclose(full[:300], part, atol=1e-12)
    np.testing.assert_allclose(model.compute_per_bar_predictive_loglik(X)[:300],
                               model.compute_per_bar_predictive_loglik(X[:300]), atol=1e-12)
    assert model.compute_per_bar_predictive_loglik(X).sum() == pytest.approx(model.log_likelihood(X), abs=1e-9)


def test_inference_mode_selects_filtering_or_smoothing():
    model = planted()
    _, X = model.sample(200, seed=7)
    np.testing.assert_allclose(model.predict_proba(X, DIR), model.chain_marginal(model.filter(X), DIR))
    model.inference_mode = InferenceMode.SMOOTHING
    np.testing.assert_allclose(model.predict_proba(X, DIR), model.chain_marginal(model.smooth(X), DIR))
    model.inference_mode = InferenceMode.VITERBI
    with pytest.raises(ValueError):
        model.predict_proba(X, DIR)
