"""ssm (lindermanlab/ssm) is a second independent implementation: 3 states, 3 diagonal-Gaussian features, 2 inputs.

ssm's input-driven HMM gives each destination one weight vector shared by every origin, the special case
``W[i, j] = Ws[j] - Ws[i]`` of our model, so every ssm model has an exact counterpart in ours. At ssm's fitted
parameters every inference output must match (transition matrices, emissions, filtered, one-step-ahead and smoothed
posteriors, log-likelihood, Viterbi path). Fitted from scratch, our EM must reach at least ssm's maximum (ours nests
it), with the same emissions and transition probabilities within estimation error.

The reference is ``fixtures/ssm_reference.npz``, written by ``oracle/ssm/make_reference.py`` in the container described
by ``oracle/ssm/Dockerfile`` (ssm needs a C++ build, so it never enters the workspace venv). Like statsmodels, ssm uses
the input on row t for the transition from t - 1 into t (our KNOWN_AHEAD).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pomegranate.distributions import Normal
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

from okmich_quant_ml.hmm import DistType, InputTiming, PomegranateIOHMM

FIXTURE = Path(__file__).parent / "fixtures" / "ssm_reference.npz"


@pytest.fixture(scope="module")
def ref() -> dict:
    with np.load(FIXTURE) as data:
        return {key: data[key] for key in data.files}


def _stacked(ref: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    x, u = ref["x"], ref["u"]
    return x.reshape(-1, x.shape[-1]), u.reshape(-1, u.shape[-1]), [x.shape[1]] * x.shape[0]


@pytest.fixture(scope="module")
def at_ssm_parameters(ref) -> PomegranateIOHMM:
    """Our model holding ssm's fitted parameters, inputs in raw units."""
    X, U, lengths = _stacked(ref)
    model = PomegranateIOHMM(DistType.NORMAL, 3, covariance_type="diag", max_iter=1,
                             input_timing=InputTiming.KNOWN_AHEAD).fit(X, lengths, U=U)
    model._distributions = [Normal(means=ref["mus"][k], covs=ref["sigmasq"][k], covariance_type="diag",
                                   dtype=torch.float64) for k in range(3)]
    model._log_pi = np.log(ref["pi0"])
    model._u_mean, model._u_std = np.zeros(2), np.ones(2)
    log_Ps, Ws = ref["log_Ps"], ref["Ws"]
    model._transitions.b = log_Ps - np.diag(log_Ps)[:, None]
    model._transitions.W = Ws[None, :, :] - Ws[:, None, :]  # W[i, j] = Ws[j] - Ws[i]
    return model


def test_inference_matches_ssm(ref, at_ssm_parameters):
    model = at_ssm_parameters
    for s in range(ref["x"].shape[0]):
        x, u = ref["x"][s], ref["u"][s]
        A = model.transition_matrices(u)
        np.testing.assert_allclose(A[1:], ref["Ps"][s], atol=1e-12)
        np.testing.assert_allclose(model._log_emissions(x), ref["log_likes"][s], atol=1e-9)
        filtered = model.predict_proba(x, U=u)
        np.testing.assert_allclose(filtered, ref["filtered"][s], atol=1e-9)
        predicted = np.vstack([np.exp(model._log_pi), np.einsum("ti,tij->tj", filtered[:-1], A[1:])])
        np.testing.assert_allclose(predicted, ref["predicted"][s], atol=1e-9)  # ssm's filter() is one step ahead
        np.testing.assert_allclose(model.state_posteriors(x, U=u), ref["smoothed"][s], atol=1e-9)
        assert model.log_likelihood(x, U=u) == pytest.approx(ref["loglik"][s], abs=1e-8)
        np.testing.assert_array_equal(model.predict_viterbi(x, U=u), ref["viterbi"][s])


def test_em_reaches_the_ssm_maximum(ref):
    X, U, lengths = _stacked(ref)
    ours = PomegranateIOHMM(DistType.NORMAL, 3, covariance_type="diag", l2_penalty=0.0, tol=1e-6, max_iter=5_000,
                            n_restarts=3, random_state=7, input_timing=InputTiming.KNOWN_AHEAD).fit(X, lengths, U=U)
    x, u = ref["x"], ref["u"]
    ll = sum(ours.log_likelihood(x[s], U=u[s]) for s in range(len(x)))
    # ssm's model is a special case of ours (6 fewer free weights), so ours can only be as high or higher
    assert ll >= float(ref["total_loglik"]) - 1e-3

    our_means = np.array([p["means"] for p in ours.parameters])
    rows, cols = linear_sum_assignment(((our_means[:, None, :] - ref["mus"][None]) ** 2).sum(axis=-1))
    perm = rows[np.argsort(cols)]  # perm[k] = our state playing ssm state k
    np.testing.assert_allclose(our_means[perm], ref["mus"], atol=0.01)
    our_vars = np.array([p["covs"] for p in ours.parameters])
    np.testing.assert_allclose(our_vars[perm], ref["sigmasq"], rtol=0.02)

    # Transitions, judged on the scale of estimation error. The data come from ssm's restricted family, so ssm's
    # estimate is the efficient one; ours spends 6 extra weights and may be up to twice as far from the truth
    # (measured: ssm 0.0056, ours 0.0089, ours vs ssm 0.0072 in mean absolute probability per bar).
    rows, cols = linear_sum_assignment(((ref["mus"][:, None, :] - ref["true_mus"][None]) ** 2).sum(axis=-1))
    true_of_ssm = cols[np.argsort(rows)]  # true state played by ssm state k
    true_A = np.concatenate([softmax(np.log(ref["true_P"])[None] + (u[s][1:] @ ref["true_Ws"].T)[:, None, :], axis=2)
                             for s in range(len(x))])[:, true_of_ssm][:, :, true_of_ssm]
    ssm_A = ref["Ps"].reshape(-1, 3, 3)
    ours_A = np.concatenate([ours.transition_matrices(u[s])[1:][:, perm][:, :, perm] for s in range(len(x))])
    ssm_error = np.mean(np.abs(ssm_A - true_A))
    assert np.mean(np.abs(ours_A - true_A)) < 2 * ssm_error
    assert np.mean(np.abs(ours_A - ssm_A)) < 2 * ssm_error
