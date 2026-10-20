"""The compiled kernels against the plain NumPy / SciPy code they replaced, on random parameters.

Sequences of mixed lengths (including length 1), time-varying and single transition matrices, emissions with
``-inf`` entries, and an extreme set that pushes some sequences off the scaled path into the log-space fallback.
Agreement is to floating-point rounding: only the summation order differs. The parallel kernels must give the same
bits on one thread as on many.
"""
from __future__ import annotations

import numba
import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp

from okmich_quant_ml.hmm.iohmm.kernels import (_scaled_pass, backward_log, expected_statistics, forward_log,
                                               input_log_matrices, logit_row_objective, logsumexp,
                                               sequences_log_likelihood, windowed_fixed_lag)

K = 3
LENGTHS = [1, 2, 7, 40, 1, 25]


def _log_softmax(a: np.ndarray, axis: int) -> np.ndarray:
    return a - scipy_logsumexp(a, axis=axis, keepdims=True)


@pytest.fixture(params=["time_varying", "single", "extreme"])
def params(request):
    rng = np.random.default_rng(3)
    T = sum(LENGTHS)
    log_pi = _log_softmax(rng.normal(size=K), axis=0)
    T_A = 1 if request.param == "single" else T
    log_A = _log_softmax(rng.normal(scale=2.0, size=(T_A, K, K)), axis=2)
    log_B = rng.normal(scale=3.0, size=(T, K))
    log_B[5, 1] = -np.inf  # a state the emission rules out
    if request.param == "extreme":
        log_B = rng.normal(scale=300.0, size=(T, K))  # emission gaps of hundreds of nats
        log_A[20] = _log_softmax(np.array([[0.0, -400.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), axis=1)
    ends = np.cumsum(LENGTHS)
    return log_pi, log_A, log_B, ends - np.asarray(LENGTHS), ends


def _scaled_ok(log_pi, log_A, log_B, starts, ends) -> list[bool]:
    time_varying = log_A.shape[0] > 1
    return [bool(_scaled_pass(log_pi, log_A[a:z] if time_varying else log_A, log_B[a:z], True)[0])
            for a, z in zip(starts, ends)]


def _is_extreme(params) -> bool:
    log_B = params[2]
    return bool(np.abs(log_B[np.isfinite(log_B)]).max() > 100)


def _rtol(params) -> float:
    """Log-space arithmetic on terms of size M carries ~M * eps of relative error in exp(.): scale the tolerance."""
    return 1e-12 * max(1.0, float(np.abs(params[2][np.isfinite(params[2])]).max()) / 10)


def test_the_fixtures_exercise_both_paths(params):
    ok = _scaled_ok(*params)
    assert any(ok)
    if _is_extreme(params):
        assert not all(ok)


def test_scaled_pass_refuses_what_it_cannot_hold():
    log_A = np.log(np.full((1, 2, 2), 0.5))
    log_B = np.zeros((4, 2))
    assert _scaled_pass(np.log([0.5, 0.5]), log_A, log_B, True)[0]
    tiny_move = np.log(np.array([[[1.0, 1e-120], [0.5, 0.5]]]))
    assert not _scaled_pass(np.log([0.5, 0.5]), tiny_move, log_B, True)[0]
    unlikely_start = np.array([[0.0, -400.0]] + [[0.0, 0.0]] * 3)  # only state 0 fits bar 0, and it starts at 1e-120
    assert not _scaled_pass(np.log([1e-120, 1.0]), log_A, unlikely_start, True)[0]
    impossible = log_B.copy()
    impossible[2] = -np.inf
    assert not _scaled_pass(np.log([0.5, 0.5]), log_A, impossible, True)[0]


def _reference_e_step(log_pi, log_A, log_B, starts, ends):
    """The per-sequence NumPy E-step the kernel replaced."""
    time_varying = log_A.shape[0] > 1
    gamma = np.empty_like(log_B)
    xi_parts, first, loglik = [], np.zeros(K), 0.0
    for a, z in zip(starts, ends):
        lB = log_B[a:z]
        lA = log_A[a:z] if time_varying else log_A
        la = forward_log(log_pi, lA, lB)
        ll = scipy_logsumexp(la[-1])
        loglik += ll
        lb = backward_log(lA, lB, np.zeros(K))
        gamma[a:z] = np.exp(la + lb - ll)
        first += gamma[a]
        if z - a > 1:
            lA_next = lA[1:] if time_varying else lA
            xi = np.exp(la[:-1, :, None] + lA_next + (lB[1:] + lb[1:])[:, None, :] - ll)
            xi_parts.append(xi if time_varying else xi.sum(axis=0, keepdims=True))
    xi = np.concatenate(xi_parts)
    return loglik, gamma, first, xi if time_varying else xi.sum(axis=0, keepdims=True)


def test_expected_statistics_match_the_numpy_e_step(params):
    got = expected_statistics(*params)
    want = _reference_e_step(*params)
    assert got[0] == pytest.approx(want[0], abs=1e-10)
    for g, w in zip(got[1:], want[1:]):
        assert g.shape == w.shape
        np.testing.assert_allclose(g, w, rtol=_rtol(params), atol=1e-14)
    assert sequences_log_likelihood(*params) == pytest.approx(want[0], abs=1e-10)


@pytest.mark.parametrize("lag", [0, 3])
def test_windowed_fixed_lag_matches_the_python_loop(params, lag):
    log_pi, log_A, log_B, _, _ = params
    window = 7
    T = len(log_B)
    time_varying = log_A.shape[0] > 1
    want = np.empty((T, K))
    for t in range(T):
        end = min(t + lag, T - 1)
        start = max(0, end - window + 1)
        lA = log_A[start:end + 1] if time_varying else log_A
        lB = log_B[start:end + 1]
        post = forward_log(log_pi, lA, lB)[t - start] + backward_log(lA, lB, np.zeros(K))[t - start]
        want[t] = np.exp(post - scipy_logsumexp(post))
    np.testing.assert_allclose(windowed_fixed_lag(log_pi, log_A, log_B, lag, window), want, rtol=1e-12, atol=1e-15)


def test_input_log_matrices_match_numpy():
    rng = np.random.default_rng(1)
    d, T = 2, 50
    b = rng.normal(size=(K, K))
    np.fill_diagonal(b, 0.0)
    W = rng.normal(size=(K, K, d))
    W[np.arange(K), np.arange(K)] = 0.0
    U = rng.normal(scale=3.0, size=(T, d))
    eta = b[None] + np.einsum("ijd,td->tij", W, U)
    np.testing.assert_allclose(input_log_matrices(b, W, U), eta - scipy_logsumexp(eta, axis=2, keepdims=True),
                               rtol=1e-12, atol=1e-13)


def _numpy_row_objective(coef, X1, counts, others, total):
    """The NumPy M-step objective the kernel replaced (unpenalised), for origin row with destinations ``others``."""
    n_out = counts.sum(axis=1)
    eta = np.zeros((len(X1), counts.shape[1]))
    eta[:, others] = X1 @ coef.T
    lse = scipy_logsumexp(eta, axis=1)
    nll = float(np.sum(n_out * lse - np.sum(counts * eta, axis=1))) / total
    prob = np.exp(eta - lse[:, None])
    return nll, ((n_out[:, None] * prob - counts)[:, others] / total).T @ X1


def test_logit_row_objective_matches_numpy_and_its_gradient_is_exact():
    rng = np.random.default_rng(2)
    N, d, i = 400, 2, 1
    others = np.array([j for j in range(K) if j != i])
    X1 = np.column_stack([np.ones(N), rng.normal(size=(N, d))])
    counts = rng.dirichlet(np.ones(K), size=N) * rng.uniform(0.0, 1.0, size=(N, 1))  # soft counts out of origin i
    total = float(counts.sum())
    coef = rng.normal(size=(K - 1, d + 1))
    nll, grad = logit_row_objective(coef, X1, np.ascontiguousarray(counts.sum(axis=1)),
                                    np.ascontiguousarray(counts[:, others]), total)
    want_nll, want_grad = _numpy_row_objective(coef, X1, counts, others, total)
    assert nll == pytest.approx(want_nll, rel=1e-12)
    np.testing.assert_allclose(grad, want_grad, rtol=1e-10, atol=1e-14)
    step = 1e-6
    for k in range(K - 1):
        for c in range(d + 1):
            up, down = coef.copy(), coef.copy()
            up[k, c] += step
            down[k, c] -= step
            numeric = (_numpy_row_objective(up, X1, counts, others, total)[0]
                       - _numpy_row_objective(down, X1, counts, others, total)[0]) / (2 * step)
            assert grad[k, c] == pytest.approx(numeric, abs=1e-7)


def test_results_do_not_depend_on_the_thread_count(params):
    rng = np.random.default_rng(5)
    N = 10_000  # several row chunks
    X1 = np.column_stack([np.ones(N), rng.normal(size=(N, 2))])
    counts = rng.dirichlet(np.ones(K), size=N)
    objective_args = (rng.normal(size=(K - 1, 3)), X1, counts.sum(axis=1), np.ascontiguousarray(counts[:, 1:]), N)

    def run():
        return (expected_statistics(*params), sequences_log_likelihood(*params),
                windowed_fixed_lag(*params[:3], 3, 7), logit_row_objective(*objective_args))

    many = run()
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        one = run()
    finally:
        numba.set_num_threads(threads)
    for a, b in zip(np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in _flatten(many)]),
                    np.concatenate([np.ravel(np.asarray(v, dtype=float)) for v in _flatten(one)])):
        assert a == b or (np.isnan(a) and np.isnan(b))


def _flatten(values):
    for v in values:
        if isinstance(v, tuple):
            yield from _flatten(v)
        else:
            yield np.atleast_1d(v)


@pytest.mark.parametrize("axis, keepdims", [(None, False), (0, False), (1, True), (2, False), (-1, True)])
def test_logsumexp_matches_scipy(axis, keepdims):
    rng = np.random.default_rng(0)
    a = rng.normal(scale=50.0, size=(4, 5, 3))
    a[1, 2, :] = -np.inf  # a slice that is all -inf
    a[2, 0, 1] = -np.inf
    np.testing.assert_allclose(logsumexp(a, axis=axis, keepdims=keepdims),
                               scipy_logsumexp(a, axis=axis, keepdims=keepdims), rtol=1e-14)
    assert np.isneginf(logsumexp(np.full(3, -np.inf)))
    assert isinstance(logsumexp(a[0, 0]), float)
