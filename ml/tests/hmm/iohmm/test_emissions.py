"""The Normal fast path against pomegranate's own ``log_probability`` / ``summarize`` / ``from_summaries``.

Every covariance type, ``min_cov``, a frozen state, more rows than one reduction chunk, and the cases that must take
pomegranate's own path (another family, mixed covariance types, float32).
"""
from __future__ import annotations

import copy

import numba
import numpy as np
import pytest
import torch
from pomegranate.distributions import Exponential, Normal

from okmich_quant_ml.hmm.iohmm.emissions import _normal_kind, log_emissions, update_emissions

N_ROWS, D, K = 10_000, 3, 3


def _states(covariance_type: str, min_cov=None, frozen_first: bool = False, dtype=np.float64) -> list:
    """K distinct states. A sphere state is fitted (pomegranate's constructor takes no scalar variance)."""
    rng = np.random.default_rng(11)
    states = []
    for k in range(K):
        means = rng.normal(size=D).astype(dtype)
        if covariance_type == "sphere":
            state = Normal(covariance_type="sphere", min_cov=min_cov, dtype=torch.float64)
            state.fit(torch.as_tensor(means + rng.uniform(0.5, 1.5) * rng.normal(size=(500, D))))
        else:
            if covariance_type == "full":
                root = rng.normal(size=(D, D))
                covs = root @ root.T + 0.5 * np.eye(D)
            else:
                covs = rng.uniform(0.3, 2.0, size=D)
            state = Normal(means=means, covs=covs.astype(dtype), covariance_type=covariance_type, min_cov=min_cov)
        if frozen_first and k == 0:
            state.frozen = torch.tensor(True)
        states.append(state)
    return states


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(12)
    X = rng.normal(scale=1.5, size=(N_ROWS, D))
    gamma = rng.dirichlet(np.ones(K), size=N_ROWS)
    return X, torch.as_tensor(X, dtype=torch.float64), gamma


@pytest.mark.parametrize("covariance_type", ["full", "diag", "sphere"])
def test_log_densities_match_pomegranate(data, covariance_type):
    X, X_t, _ = data
    states = _states(covariance_type)
    assert _normal_kind(states) == covariance_type
    want = np.column_stack([s.log_probability(X_t).numpy() for s in states])
    np.testing.assert_allclose(log_emissions(states, X, X_t), want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("covariance_type", ["full", "diag", "sphere"])
@pytest.mark.parametrize("min_cov, frozen_first", [(None, False), (0.8, True)])
def test_em_update_matches_pomegranate(data, covariance_type, min_cov, frozen_first):
    X, X_t, gamma = data
    fast = _states(covariance_type, min_cov, frozen_first)
    slow = copy.deepcopy(fast)
    update_emissions(fast, X, X_t, gamma)
    for k, state in enumerate(slow):
        state.summarize(X_t, sample_weight=torch.as_tensor(gamma[:, k:k + 1], dtype=torch.float64))
        state.from_summaries()
    for a, b in zip(fast, slow):
        np.testing.assert_allclose(a.means.detach().numpy(), b.means.detach().numpy(), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(a.covs.detach().numpy(), b.covs.detach().numpy(), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(a._w_sum.numpy(), b._w_sum.numpy())  # both reset to zero for the next round
    if frozen_first:
        np.testing.assert_array_equal(fast[0].means.detach().numpy(), _states(covariance_type)[0].means.numpy())


def test_other_cases_take_pomegranates_path(data):
    X, X_t, gamma = data
    assert _normal_kind([Exponential(scales=[1.0, 2.0, 3.0], dtype=torch.float64)]) is None
    assert _normal_kind(_states("diag")[:2] + _states("full")[:1]) is None
    assert _normal_kind(_states("diag", dtype=np.float32)) is None
    positive = np.abs(X) + 0.1
    states = [Exponential(scales=[1.0 + k] * D, dtype=torch.float64) for k in range(K)]
    want = np.column_stack([s.log_probability(torch.as_tensor(positive)).numpy() for s in states])
    np.testing.assert_allclose(log_emissions(states, positive), want, rtol=1e-12)


def test_moments_do_not_depend_on_the_thread_count(data):
    X, X_t, gamma = data
    threads = numba.get_num_threads()
    results = []
    for n in (threads, 1):
        numba.set_num_threads(n)
        try:
            states = _states("full")
            update_emissions(states, X, X_t, gamma)
            results.append(np.concatenate([np.ravel(s.covs.detach().numpy()) for s in states]))
        finally:
            numba.set_num_threads(threads)
    np.testing.assert_array_equal(results[0], results[1])
