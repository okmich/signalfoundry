"""Without inputs the IOHMM is an HMM: same fit as PomegranateHMM, identical inference at identical parameters.

The fit differs only through pomegranate's learned end probabilities (this model is open-ended), so parameters agree
to ~1e-3, not exactly. Inference at identical (row-stochastic) parameters must agree to machine precision.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from okmich_quant_ml.hmm import DistType, InferenceMode, PomegranateHMM, PomegranateIOHMM

from ._synthetic import FIT_KW


def _make_row_stochastic(hmm: PomegranateHMM) -> None:
    """Pomegranate's transition rows sum to 1 - end_i; renormalise them so both models hold the same matrix."""
    with torch.no_grad():
        hmm._model.edges.copy_(torch.log(torch.as_tensor(hmm.transition_prob(), dtype=hmm._model.edges.dtype)))
    hmm._model._reset_cache()


@pytest.fixture(scope="module")
def fitted_pair(homogeneous_data):
    hmm = PomegranateHMM(DistType.NORMAL, 2, **FIT_KW).fit(homogeneous_data)
    io = PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).fit(homogeneous_data)
    return hmm, io


def test_fit_without_inputs_matches_pomegranate(fitted_pair, homogeneous_data):
    hmm, io = fitted_pair
    for p_hmm, p_io in zip(hmm.parameters, io.parameters):
        np.testing.assert_allclose(p_io["means"], p_hmm["means"], atol=5e-3)
        np.testing.assert_allclose(p_io["covs"], p_hmm["covs"], rtol=5e-3)
    np.testing.assert_allclose(io.transition_prob(), hmm.transition_prob(), atol=2e-3)
    agree = np.mean(io.predict(homogeneous_data) == hmm.predict(homogeneous_data))
    assert agree >= 0.999


def test_from_hmm_inference_is_identical(fitted_pair, homogeneous_data):
    hmm = copy.deepcopy(fitted_pair[0])
    _make_row_stochastic(hmm)
    io = PomegranateIOHMM.from_hmm(hmm)
    X = homogeneous_data

    np.testing.assert_allclose(io.predict_proba(X), hmm.predict_proba(X), atol=1e-9)
    np.testing.assert_array_equal(io.predict_causal_viterbi(X), hmm.predict_causal_viterbi(X))
    np.testing.assert_array_equal(io.predict_viterbi(X), hmm.predict_viterbi(X))
    np.testing.assert_allclose(io.compute_per_bar_predictive_loglik(X), hmm.compute_per_bar_predictive_loglik(X),
                               atol=1e-8)
    for lag in (1, 5, 40):
        np.testing.assert_allclose(io.predict_proba_fixed_lag(X, lag), hmm.predict_proba_fixed_lag(X, lag), atol=1e-9)
    windowed_io = io.predict_proba_fixed_lag_sweep(X[:300], [0, 3], window_size=50)
    windowed_hmm = hmm.predict_proba_fixed_lag_sweep(X[:300], [0, 3], window_size=50)
    for lag in (0, 3):
        np.testing.assert_allclose(windowed_io[lag], windowed_hmm[lag], atol=1e-9)
    np.testing.assert_allclose(io.forecast(io.predict_proba(X)[-1], 3), hmm.forecast(hmm.predict_proba(X)[-1], 3),
                               atol=1e-12)
    assert io.expected_duration(0) == pytest.approx(hmm.expected_duration(0))
    np.testing.assert_allclose(io.get_stationary_distribution(), hmm.get_stationary_distribution(), atol=1e-12)


def test_smoothing_mode_equals_open_ended_fixed_lag(fitted_pair, homogeneous_data):
    """Smoothing here is open-ended, so it equals the base class's fixed-lag smoother at lag T-1."""
    io = copy.deepcopy(fitted_pair[1])
    io.inference_mode = InferenceMode.SMOOTHING
    X = homogeneous_data[:500]
    np.testing.assert_allclose(io.predict_proba(X), io.predict_proba_fixed_lag(X, len(X) - 1), atol=1e-12)
    np.testing.assert_allclose(io.state_posteriors(X), io.predict_proba(X), atol=1e-12)


def test_warm_start_from_an_hmm_never_lowers_the_likelihood(fitted_pair, homogeneous_data):
    """A model with inputs started from the fitted HMM nests it: EM can only raise the likelihood from there."""
    hmm = copy.deepcopy(fitted_pair[0])
    _make_row_stochastic(hmm)
    null = PomegranateIOHMM.from_hmm(hmm)
    X = homogeneous_data
    U = np.random.default_rng(4).normal(size=(len(X), 1))
    full = PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).fit(X, U=U, init_model=hmm)
    assert full.log_likelihood(X, U=U) >= null.log_likelihood(X) - 1e-6
