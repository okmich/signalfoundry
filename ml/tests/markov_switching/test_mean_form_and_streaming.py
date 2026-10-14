"""Correctness of the mean-form MS-AR filter/forecasts, MS-OU θ, causal defaults, and GARCH/VAR streaming filters.

The streaming tests use the same yardstick for every family: fit on y[:N], stream y[N:N+M] through ``update()``, and
compare against the batch filter of the *same fitted parameters* run over y[:N+M]. Identical prefixes make the two
filters identical, so any gap is a bug, not noise.
"""

import itertools
import warnings

import numpy as np
import pytest
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from okmich_quant_ml.markov_switching import MarkovSwitchingAR, MarkovSwitchingGARCH, MarkovSwitchingOU, \
    MarkovSwitchingVAR
from okmich_quant_ml.markov_switching.ar import _HamiltonMarkovAutoregression
from okmich_quant_ml.markov_switching.kernels import _companion, _ms_ar_joint_forward_step, _scaled_update


# ─── Simulation ─────────────────────────────────────────────────────────────────

STICKY_P = np.array([[0.98, 0.02], [0.03, 0.97]])


def simulate_mean_form(T: int, mu: np.ndarray, phi: np.ndarray, sigma: np.ndarray, P: np.ndarray,
                       seed: int) -> np.ndarray:
    """y_t − μ_{s_t} = Σ_l φ_{s_t,l} (y_{t−l} − μ_{s_{t−l}}) + σ_{s_t} e_t, burn-in 50 bars."""
    rng = np.random.default_rng(seed)
    p = phi.shape[1]
    s = np.zeros(T + 50, dtype=int)
    y = np.zeros(T + 50)
    for t in range(p, T + 50):
        s[t] = rng.choice(len(mu), p=P[s[t - 1]])
        dev = sum(phi[s[t], j] * (y[t - 1 - j] - mu[s[t - 1 - j]]) for j in range(p))
        y[t] = mu[s[t]] + dev + sigma[s[t]] * rng.normal()
    return y[50:]


@pytest.fixture(scope="module")
def level_series():
    """Order-2 series on levels with well-separated regime means — where mean-vs-intercept matters."""
    return simulate_mean_form(1150, np.array([-2.0, 3.0]), np.array([[0.5, 0.1], [0.8, -0.1]]),
                              np.array([0.3, 0.5]), STICKY_P, seed=0)


@pytest.fixture(scope="module")
def three_regime_series():
    """K=3 needs three real regimes: statsmodels' EM can fail (SVD) fitting a superfluous one."""
    P = np.array([[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.01, 0.03, 0.96]])
    return simulate_mean_form(1150, np.array([-1.0, 0.0, 1.5]), np.array([[0.5], [0.2], [0.7]]),
                              np.array([0.3, 0.2, 0.4]), P, seed=4)


@pytest.fixture(scope="module")
def fitted_level_ar(level_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MarkovSwitchingAR(n_regimes=2, order=2, random_state=1).fit(level_series[:800])


# ─── statsmodels variance keying ────────────────────────────────────────────────

@pytest.mark.parametrize("order", [1, 2])
def test_conditional_variance_keyed_on_current_regime(level_series, order):
    """σ² must follow s_t. Upstream keys it on s_{t-order+1}; the subclass fixes it and matches upstream at order 1."""
    kw = dict(k_regimes=2, order=order, switching_ar=True, switching_variance=True)
    fixed = _HamiltonMarkovAutoregression(level_series[:300], **kw)
    stock = MarkovAutoregression(level_series[:300], **kw)
    names = fixed.param_names
    params = np.array([{"p[0->0]": 0.95, "p[1->0]": 0.05, "const[0]": -2.0, "const[1]": 3.0,
                        "sigma2[0]": 0.09, "sigma2[1]": 0.25}.get(n, 0.3) for n in names])

    resid = fixed._resid(params)
    sigma2 = np.array([0.09, 0.25]).reshape((2,) + (1,) * (order + 1))
    expected = -0.5 * resid ** 2 / sigma2 - 0.5 * np.log(2 * np.pi * sigma2)
    np.testing.assert_allclose(fixed._conditional_loglikelihoods(params), expected, rtol=1e-12)

    if order == 1:
        np.testing.assert_allclose(stock._conditional_loglikelihoods(params), expected, rtol=1e-12)
    else:
        assert not np.allclose(stock._conditional_loglikelihoods(params), expected)


# ─── MS-AR streaming filter ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n_regimes, order", [(2, 1), (2, 2), (2, 3), (3, 1)])
def test_ar_update_reproduces_batch_filter(request, n_regimes, order):
    """Covers the joint-path digit decoding for K^(p+1) = 4, 8, 16 and 9 states."""
    level_series = request.getfixturevalue("level_series" if n_regimes == 2 else "three_regime_series")
    N, M = 800, 300
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingAR(n_regimes=n_regimes, order=order, random_state=1).fit(level_series[:N])

    streamed = []
    for v in level_series[N:N + M]:
        ms.update(v)
        streamed.append(ms.current_regime_proba())

    batch = _HamiltonMarkovAutoregression(level_series[:N + M], k_regimes=n_regimes, order=order, switching_ar=True,
                                          switching_variance=True).filter(ms.fitted_model_.params)
    np.testing.assert_allclose(np.array(streamed), batch.filtered_marginal_probabilities[-M:], atol=1e-10)
    np.testing.assert_allclose(ms._last_forward_joint_, batch.filtered_joint_probabilities[..., -1].reshape(-1),
                               atol=1e-10)


def test_ar_streaming_joint_initialised_from_fit(fitted_level_ar):
    joint = fitted_level_ar._last_forward_joint_
    assert joint.shape == (2 ** (fitted_level_ar.order + 1),)
    np.testing.assert_allclose(joint.reshape(2, -1).sum(axis=1), fitted_level_ar.filtered_probabilities_[-1], atol=1e-12)


def test_ar_regime_parameters_report_mean_and_intercept(fitted_level_ar):
    params = fitted_level_ar.get_regime_parameters().sort_values("mean")
    np.testing.assert_allclose(params["mean"].to_numpy(), [-2.0, 3.0], atol=0.15)
    np.testing.assert_allclose(params["intercept"], params["mean"] * (1.0 - params["persistence"]), rtol=1e-12)


def test_ar_update_rejects_exog_models(level_series):
    exog = np.random.default_rng(0).normal(size=(400, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingAR(n_regimes=2, order=1).fit(level_series[:400], exog=exog)
    with pytest.raises(NotImplementedError, match="exog"):
        ms.update(0.0)



def test_ar_stay_in_regime_rejects_unreachable_regime(fitted_level_ar):
    import copy
    ms = copy.deepcopy(fitted_level_ar)
    ms.transition_matrix_ = np.array([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="unreachable"):
        ms.forecast(steps=3, regime=1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_updates_reject_non_finite_bars_without_touching_state(fitted_level_ar, var_series, garch_returns, bad):
    import copy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms_var = MarkovSwitchingVAR(n_regimes=2, order=1).fit(var_series[:300], num_restarts=1, maxiter=20)
        ms_garch = MarkovSwitchingGARCH(n_regimes=2, order=1).fit(garch_returns[:300], num_restarts=1, maxiter=10)
    cases = [(copy.deepcopy(fitted_level_ar), np.array([0.1, bad])),
             (ms_garch, np.array([0.001, bad])),
             (ms_var, np.array([[0.0, 0.0], [bad, 0.0]]))]
    for model, batch in cases:
        before = model.current_regime_proba()
        with pytest.raises(ValueError, match="finite"):
            model.update(batch)
        np.testing.assert_array_equal(model.current_regime_proba(), before)
        assert model._n_updates_ == 0


def test_joint_step_ignores_impossible_best_path():
    """The best-fitting path (s_t = 1) has zero prior mass. Shifting by the max log-lik over all paths would
    underflow every possible path and return uniform; the log-space shift returns the true posterior."""
    joint_prev = np.array([1.0, 0.0, 0.0, 0.0])            # certain on (s_t, s_{t-1}) = (0, 0)
    out = _ms_ar_joint_forward_step(np.array([[0.5], [0.5]]), np.array([0.0, 100.0]), np.array([1.0, 1.0]),
                                    STICKY_P, joint_prev, np.array([0.0]), 50.0)
    np.testing.assert_allclose(out, [0.98, 0.0, 0.02, 0.0], atol=1e-12)


def test_scaled_update_shifts_over_possible_states():
    posterior, log_norm = _scaled_update(np.array([1.0, 0.0]), np.array([-1000.0, 0.0]))
    np.testing.assert_allclose(posterior, [1.0, 0.0])
    assert log_norm == pytest.approx(-1000.0)
    posterior, log_norm = _scaled_update(np.array([0.5, 0.5]), np.array([np.nan, 0.0]))
    assert np.isnan(posterior).all() and np.isnan(log_norm)


def test_legacy_pickle_is_refused(fitted_level_ar, tmp_path):
    import copy
    import joblib
    legacy = copy.deepcopy(fitted_level_ar)
    del legacy.__dict__["_last_forward_joint_"]
    path = tmp_path / "legacy.joblib"
    joblib.dump(legacy, path)
    with pytest.raises(ValueError, match="v2.1"):
        MarkovSwitchingAR.load(str(path))


def test_refit_window_rejects_non_finite_batch_before_touching_state(fitted_level_ar):
    import copy
    ms = copy.deepcopy(fitted_level_ar)
    before = ms.current_regime_proba()
    with pytest.raises(ValueError, match="finite"):
        ms.refit_window(np.array([0.1, np.nan]), window=200)
    assert not hasattr(ms, "_window_buffer_")
    np.testing.assert_array_equal(ms.current_regime_proba(), before)


def test_mean_only_forecast_matches_full(fitted_level_ar):
    full = fitted_level_ar.forecast(steps=6, return_variance=True)
    np.testing.assert_array_equal(fitted_level_ar.forecast(steps=6), full["mean"])


def test_garch_vectorised_residuals_match_loop(garch_returns):
    y, c, phi = garch_returns, 3e-4, np.array([0.1, -0.05])
    loop = [y[t] - (c + phi[0] * y[t - 1] + phi[1] * y[t - 2]) for t in range(2, len(y))]
    np.testing.assert_array_equal(MarkovSwitchingGARCH(order=2)._compute_ar_residuals(y, c, phi), loop)

# ─── MS-AR exact forecasts (brute-force path enumeration) ───────────────────────

def _enumerated_moments(ms: MarkovSwitchingAR, steps: int, stay_in: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact E[y_{T+h}] and Var[y_{T+h}] by summing over every past (p+1)-path and every future path."""
    K, p = ms.n_regimes, ms.order
    ar, mu, s2 = ms._cached_ar_coeffs_, ms._cached_means_, ms._cached_sigma2s_
    P = np.asarray(ms.transition_matrix_)
    joint = ms._last_forward_joint_.reshape((K,) * (p + 1))
    hist = ms._history_buffer_
    futures = [(stay_in,) * steps] if stay_in is not None else itertools.product(range(K), repeat=steps)

    total_w, ey, ey2 = 0.0, np.zeros(steps), np.zeros(steps)
    for future in futures:
        for past in itertools.product(range(K), repeat=p + 1):
            w = joint[past]
            prev = past[0]
            for s in future:
                w *= P[prev, s]
                prev = s
            z = np.array([hist[p - 1 - lag] - mu[past[lag]] for lag in range(p)])
            cov = np.zeros((p, p))
            for h, s in enumerate(future):
                F = _companion(ar[s])
                z = F @ z
                cov = F @ cov @ F.T
                cov[0, 0] += s2[s]
                ey[h] += w * (mu[s] + z[0])
                ey2[h] += w * ((mu[s] + z[0]) ** 2 + cov[0, 0])
            total_w += w
    mean = ey / total_w
    return mean, ey2 / total_w - mean ** 2


def test_ar_forecast_matches_path_enumeration(fitted_level_ar):
    steps = 4
    fc = fitted_level_ar.forecast(steps=steps, return_variance=True)
    mean, var = _enumerated_moments(fitted_level_ar, steps)
    np.testing.assert_allclose(fc["mean"], mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fc["variance"], var, rtol=1e-9, atol=1e-12)

    mixed = sum(fc["regime_probabilities"][:, r] * fc["regime_forecasts"][r] for r in range(2))
    np.testing.assert_allclose(fc["mean"], mixed, rtol=1e-12)


@pytest.mark.parametrize("regime", [0, 1])
def test_ar_stay_in_regime_forecast_matches_enumeration(fitted_level_ar, regime):
    steps = 4
    fc = fitted_level_ar.forecast(steps=steps, regime=regime, return_variance=True)
    mean, var = _enumerated_moments(fitted_level_ar, steps, stay_in=regime)
    np.testing.assert_allclose(fc["mean"], mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fc["variance"], var, rtol=1e-9, atol=1e-12)


def test_ar_forecast_follows_streamed_state(level_series, fitted_level_ar):
    import copy
    ms = copy.deepcopy(fitted_level_ar)
    ms.update(level_series[800:820])
    mean, var = _enumerated_moments(ms, 3)
    fc = ms.forecast(steps=3, return_variance=True)
    np.testing.assert_allclose(fc["mean"], mean, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(fc["variance"], var, rtol=1e-9, atol=1e-12)


# ─── MS-AR fit reproducibility and train() ──────────────────────────────────────

def test_ar_fit_is_seeded_and_leaves_global_rng_alone(level_series):
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = MarkovSwitchingAR(n_regimes=2, order=1, random_state=7).fit(level_series[:400])
        b = MarkovSwitchingAR(n_regimes=2, order=1, random_state=7).fit(level_series[:400])
    assert np.random.random() == expected_next
    np.testing.assert_array_equal(a.fitted_model_.params, b.fitted_model_.params)


def test_ar_train_adopts_streaming_state(level_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingAR(random_state=1).train(level_series[:500], order_range=(1, 2), n_regimes_range=(2, 2))
        best = MarkovSwitchingAR(n_regimes=ms.n_regimes, order=ms.order, random_state=1).fit(level_series[:500])
    np.testing.assert_allclose(ms.forecast(steps=3), best.forecast(steps=3), rtol=1e-12)
    ms.update(level_series[500])
    assert ms._n_updates_ == 1


# ─── MS-OU θ ────────────────────────────────────────────────────────────────────

def test_ou_theta_is_the_regime_mean():
    """Old code reported θ = μ / (1 − φ): ≈ 5.0 instead of 1.0 for the slow regime here."""
    y = simulate_mean_form(2000, np.array([0.0, 1.0]), np.array([[0.5], [0.8]]), np.array([0.3, 0.3]), STICKY_P, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ou = MarkovSwitchingOU(n_regimes=2).fit(y)
    params = ou.get_ou_parameters().sort_values("phi")
    np.testing.assert_allclose(params["theta"].to_numpy(), [0.0, 1.0], atol=0.15)
    means = ou.get_regime_parameters().sort_values("regime")["mean"].to_numpy()
    np.testing.assert_allclose(ou.get_ou_parameters().sort_values("regime")["theta"].to_numpy(), means, rtol=1e-12)


# ─── MS-GARCH streaming filter ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def garch_returns():
    rng = np.random.default_rng(3)
    T = 500
    y = np.empty(T)
    s2 = 1e-4
    eps_prev = 0.0
    regime = 0
    for t in range(T):
        if rng.random() < 0.03:
            regime = 1 - regime
        omega, alpha, beta = (1e-6, 0.05, 0.90) if regime == 0 else (5e-6, 0.10, 0.85)
        s2 = omega + alpha * eps_prev ** 2 + beta * s2
        eps_prev = np.sqrt(s2) * rng.normal()
        y[t] = 0.0002 + eps_prev
    return y


def test_garch_update_reproduces_batch_filter(garch_returns):
    N = 400
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingGARCH(n_regimes=2, order=1).fit(garch_returns[:N], num_restarts=1, maxiter=30)

    # Stored filter describes the stored parameters
    alpha_fit = ms._e_step(garch_returns[:N], ms._params_)[0]
    np.testing.assert_allclose(ms.filtered_probabilities_, alpha_fit, atol=1e-12)

    M = len(garch_returns) - N
    streamed = []
    for v in garch_returns[N:]:
        ms.update(v)
        streamed.append(ms.current_regime_proba())
    alpha_batch = ms._e_step(garch_returns, ms._params_)[0]
    np.testing.assert_allclose(np.array(streamed), alpha_batch[-M:], atol=1e-10)


# ─── MS-VAR streaming filter ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def var_series():
    rng = np.random.default_rng(5)
    T, n = 600, 2
    A = [np.array([[0.5, 0.1], [0.0, 0.3]]), np.array([[-0.2, 0.0], [0.1, -0.1]])]
    mu = [np.array([0.001, 0.0]), np.array([-0.001, 0.002])]
    L = [np.linalg.cholesky(np.array([[1e-4, 3e-5], [3e-5, 1e-4]])), np.linalg.cholesky(np.eye(n) * 4e-4)]
    Y = np.zeros((T, n))
    regime = 0
    for t in range(1, T):
        if rng.random() < 0.03:
            regime = 1 - regime
        Y[t] = mu[regime] + A[regime] @ Y[t - 1] + L[regime] @ rng.normal(size=n)
    return Y


@pytest.mark.parametrize("order", [1, 2])
def test_var_update_reproduces_batch_filter(var_series, order):
    N = 450
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingVAR(n_regimes=2, order=order).fit(var_series[:N], num_restarts=1, maxiter=50)

    X, Y_eff = ms._build_design_matrix(var_series[:N])
    np.testing.assert_allclose(ms.filtered_probabilities_, ms._e_step(Y_eff, X, ms._params_)[0], atol=1e-12)

    M = len(var_series) - N
    streamed = []
    for row in var_series[N:]:
        ms.update(row)
        streamed.append(ms.current_regime_proba())
    X_full, Y_full = ms._build_design_matrix(var_series)
    alpha_batch = ms._e_step(Y_full, X_full, ms._params_)[0]
    np.testing.assert_allclose(np.array(streamed), alpha_batch[-M:], atol=1e-10)


def test_var_update_batch_equals_sequential(var_series):
    import copy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingVAR(n_regimes=2, order=1).fit(var_series[:450], num_restarts=1, maxiter=50)
    a, b = copy.deepcopy(ms), copy.deepcopy(ms)
    a.update(var_series[450:470])
    for row in var_series[450:470]:
        b.update(row)
    np.testing.assert_allclose(a.current_regime_proba(), b.current_regime_proba(), rtol=1e-12)
    np.testing.assert_array_equal(a._history_buffer_, b._history_buffer_)


def test_var_update_validates_width(var_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = MarkovSwitchingVAR(n_regimes=2, order=1).fit(var_series[:300], num_restarts=1, maxiter=20)
    with pytest.raises(ValueError, match="columns"):
        ms.update(np.zeros(3))


# ─── Causal defaults ────────────────────────────────────────────────────────────

def test_predict_regime_defaults_are_causal(fitted_level_ar, var_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms_var = MarkovSwitchingVAR(n_regimes=2, order=1).fit(var_series[:300], num_restarts=1, maxiter=20)
    for model in (fitted_level_ar, ms_var):
        np.testing.assert_array_equal(model.predict_regime_proba(), model.filtered_probabilities_)
        np.testing.assert_array_equal(model.predict_regime(), np.argmax(model.filtered_probabilities_, axis=1))
