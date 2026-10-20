"""statsmodels' Markov-switching TVTP is an independent implementation of the same model (K = 2, Gaussian emissions).

At statsmodels' parameters our filter and likelihood must reproduce its own; fitted from scratch, our EM must reach
its maximum. Both use the input on row t for the transition from t - 1 into t.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from pomegranate.distributions import Normal
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from okmich_quant_ml.hmm import DistType, PomegranateIOHMM

from ._synthetic import clock_sequence


@pytest.fixture(scope="module")
def tvtp():
    X, U, _ = clock_sequence(3_000, seed=0)
    y, u = X[:, 0], U[:, 0]
    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True,
                           exog_tvtp=np.column_stack([np.ones(len(u)), u]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(disp=False)
    return y, u, mod, res


def _with_statsmodels_parameters(y, u, mod, res) -> PomegranateIOHMM:
    """Our model holding statsmodels' parameters. statsmodels: p[i->0] = logistic(beta_i0 . [1, u])."""
    p = dict(zip(mod.param_names, res.params))
    model = PomegranateIOHMM(DistType.NORMAL, 2, covariance_type="diag", max_iter=1).fit(y, U=u)
    model._distributions = [Normal(means=[p[f"const[{k}]"]], covs=[p[f"sigma2[{k}]"]], covariance_type="diag",
                                   dtype=torch.float64) for k in range(2)]
    model._log_pi = np.log(mod.initial_probabilities(res.params))
    model._u_mean, model._u_std = np.zeros(1), np.ones(1)  # raw inputs = model inputs
    tr = model._transitions
    tr.b[0, 1], tr.W[0, 1, 0] = -p["p[0->0].tvtp0"], -p["p[0->0].tvtp1"]  # leave 0: log-odds = -beta_00 . z
    tr.b[1, 0], tr.W[1, 0, 0] = p["p[1->0].tvtp0"], p["p[1->0].tvtp1"]  # leave 1: log-odds = +beta_10 . z
    return model


def test_filter_and_likelihood_match_statsmodels(tvtp):
    y, u, mod, res = tvtp
    model = _with_statsmodels_parameters(y, u, mod, res)
    sm_filtered = np.asarray(res.filtered_marginal_probabilities)
    np.testing.assert_allclose(model.predict_proba(y, U=u), sm_filtered, atol=1e-9)
    assert model.log_likelihood(y, U=u) == pytest.approx(res.llf, abs=1e-8)
    sm_matrices = np.transpose(mod.regime_transition_matrix(res.params, exog_tvtp=mod.exog_tvtp), (2, 1, 0))
    np.testing.assert_allclose(model.transition_matrices(u)[1:], sm_matrices[1:], atol=1e-12)


def test_em_reaches_the_statsmodels_maximum(tvtp):
    y, u, mod, res = tvtp
    ours = PomegranateIOHMM(DistType.NORMAL, 2, covariance_type="diag", l2_penalty=0.0, tol=1e-8, max_iter=5_000,
                            random_state=7).fit(y, U=u)
    # Our start distribution is free where statsmodels' is the steady state, so ours can only be as high or higher.
    assert ours.log_likelihood(y, U=u) >= res.llf - 1e-3

    p = dict(zip(mod.param_names, res.params))
    sm_var = np.array([p["sigma2[0]"], p["sigma2[1]"]])
    our_var = np.array([float(np.ravel(s["covs"])[0]) for s in ours.parameters])
    perm = np.argsort(our_var)[np.argsort(np.argsort(sm_var))]  # perm[k] = our state matching statsmodels state k
    np.testing.assert_allclose(our_var[perm], sm_var, rtol=0.02)
    b, W = ours.baseline_logits(raw=True), ours.input_coefficients(raw=True)
    assert b[perm[0], perm[1]] == pytest.approx(-p["p[0->0].tvtp0"], abs=0.1)
    assert W[perm[0], perm[1], 0] == pytest.approx(-p["p[0->0].tvtp1"], abs=0.1)
    assert b[perm[1], perm[0]] == pytest.approx(p["p[1->0].tvtp0"], abs=0.1)
    assert W[perm[1], perm[0], 0] == pytest.approx(p["p[1->0].tvtp1"], abs=0.1)
