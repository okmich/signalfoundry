"""Tests for MarkovSwitchingVAR."""

import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.markov_switching.var import MarkovSwitchingVAR


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def var_data():
    """Synthetic 2-regime VAR(1) data with 3 assets."""
    rng = np.random.default_rng(42)
    T = 500
    n = 3
    Y = np.zeros((T, n))

    # Regime 0: low vol, moderate persistence, positive cross-correlation
    A0 = np.array([
        [0.5, 0.1, 0.0],
        [0.1, 0.4, 0.1],
        [0.0, 0.1, 0.3],
    ])
    mu0 = np.array([0.001, 0.002, 0.001])
    Sigma0 = np.array([
        [0.01, 0.003, 0.001],
        [0.003, 0.01, 0.002],
        [0.001, 0.002, 0.01],
    ])
    L0 = np.linalg.cholesky(Sigma0)

    # Regime 1: high vol, higher persistence, stronger correlation
    A1 = np.array([
        [0.7, 0.15, 0.05],
        [0.1, 0.6, 0.1],
        [0.05, 0.1, 0.5],
    ])
    mu1 = np.array([-0.002, -0.001, -0.003])
    Sigma1 = np.array([
        [0.04, 0.015, 0.008],
        [0.015, 0.05, 0.012],
        [0.008, 0.012, 0.03],
    ])
    L1 = np.linalg.cholesky(Sigma1)

    As = [A0, A1]
    mus = [mu0, mu1]
    Ls = [L0, L1]

    regime = 0
    Y[0] = rng.multivariate_normal(np.zeros(n), Sigma0)

    for t in range(1, T):
        if rng.random() < 0.05:
            regime = 1 - regime
        eps = Ls[regime] @ rng.standard_normal(n)
        Y[t] = mus[regime] + As[regime] @ Y[t - 1] + eps

    return Y


@pytest.fixture
def fitted_var(var_data):
    ms = MarkovSwitchingVAR(n_regimes=2, order=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(var_data, num_restarts=3, maxiter=80)
    return ms


# ─── Constructor ────────────────────────────────────────────────────────────────

def test_constructor_defaults():
    ms = MarkovSwitchingVAR()
    assert ms.n_regimes == 2
    assert ms.order == 1
    assert not ms.is_fitted


def test_constructor_invalid_n_regimes():
    with pytest.raises(ValueError, match="n_regimes must be at least 2"):
        MarkovSwitchingVAR(n_regimes=1)


def test_constructor_invalid_order():
    with pytest.raises(ValueError, match="order must be at least 1"):
        MarkovSwitchingVAR(order=0)


def test_constructor_invalid_ridge():
    with pytest.raises(ValueError, match="ridge must be non-negative"):
        MarkovSwitchingVAR(ridge=-1.0)


# ─── Fitting ────────────────────────────────────────────────────────────────────

def test_fit_returns_self(var_data):
    ms = MarkovSwitchingVAR(n_regimes=2, order=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ms.fit(var_data, num_restarts=1, maxiter=20)
    assert result is ms


def test_fit_sets_is_fitted(fitted_var):
    assert fitted_var.is_fitted


def test_fit_rejects_1d():
    ms = MarkovSwitchingVAR()
    with pytest.raises(ValueError, match="Y must be 2-D"):
        ms.fit(np.random.randn(100))


def test_fit_rejects_single_asset():
    ms = MarkovSwitchingVAR()
    with pytest.raises(ValueError, match="n_assets must be >= 2"):
        ms.fit(np.random.randn(100, 1))


def test_fit_short_series():
    ms = MarkovSwitchingVAR(n_regimes=2, order=1)
    with pytest.raises(ValueError, match="Time series too short"):
        ms.fit(np.random.randn(10, 3))


def test_fit_stores_data(fitted_var, var_data):
    assert fitted_var.data_ is not None
    assert fitted_var.data_.shape == var_data.shape


def test_fit_n_assets(fitted_var):
    assert fitted_var.n_assets_ == 3


def test_fit_aic_bic_set(fitted_var):
    assert fitted_var.aic is not None
    assert fitted_var.bic is not None
    assert fitted_var.bic >= fitted_var.aic  # BIC penalises more


# ─── Transition matrix ───────────────────────────────────────────────────────────

def test_transition_matrix_shape(fitted_var):
    P = fitted_var.get_transition_matrix()
    assert P.shape == (2, 2)


def test_transition_matrix_row_stochastic(fitted_var):
    P = fitted_var.get_transition_matrix()
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


# ─── Regime probabilities ────────────────────────────────────────────────────────

def test_filtered_probs_shape(fitted_var, var_data):
    fp = fitted_var.filtered_probabilities_
    assert fp.shape[0] == len(var_data) - fitted_var.order
    assert fp.shape[1] == fitted_var.n_regimes


def test_smoothed_probs_shape(fitted_var, var_data):
    sp = fitted_var.regime_probabilities_
    assert sp.shape[0] == len(var_data) - fitted_var.order
    assert sp.shape[1] == fitted_var.n_regimes


def test_probs_sum_to_one(fitted_var):
    assert np.allclose(fitted_var.filtered_probabilities_.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(fitted_var.regime_probabilities_.sum(axis=1), 1.0, atol=1e-6)


def test_predict_regime_shape(fitted_var, var_data):
    regimes = fitted_var.predict_regime()
    assert len(regimes) == len(var_data) - fitted_var.order
    assert regimes.min() >= 0
    assert regimes.max() < fitted_var.n_regimes


def test_predict_regime_causal(fitted_var):
    r_smoothed = fitted_var.predict_regime(causal=False)
    r_filtered = fitted_var.predict_regime(causal=True)
    assert len(r_smoothed) == len(r_filtered)


def test_predict_regime_proba_shape(fitted_var, var_data):
    proba = fitted_var.predict_regime_proba()
    assert proba.shape == (len(var_data) - fitted_var.order, fitted_var.n_regimes)


# ─── Forecasting ────────────────────────────────────────────────────────────────

def test_forecast_basic_shape(fitted_var):
    fc = fitted_var.forecast(steps=10)
    assert fc.shape == (10, 3)
    assert isinstance(fc, np.ndarray)


def test_forecast_no_nan(fitted_var):
    fc = fitted_var.forecast(steps=10)
    assert not np.any(np.isnan(fc))


def test_forecast_with_covariance_keys(fitted_var):
    result = fitted_var.forecast(steps=5, return_covariance=True)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "covariance" in result
    assert "regime_probabilities" in result
    assert "regime_forecasts" in result


def test_forecast_covariance_shape(fitted_var):
    result = fitted_var.forecast(steps=5, return_covariance=True)
    assert result["mean"].shape == (5, 3)
    assert result["covariance"].shape == (5, 3, 3)


def test_forecast_covariance_psd(fitted_var):
    """Forecast covariance matrices should be positive semi-definite."""
    result = fitted_var.forecast(steps=5, return_covariance=True)
    for h in range(5):
        eigvals = np.linalg.eigvalsh(result["covariance"][h])
        assert np.all(eigvals >= -1e-10)


def test_forecast_covariance_symmetric(fitted_var):
    result = fitted_var.forecast(steps=5, return_covariance=True)
    for h in range(5):
        C = result["covariance"][h]
        assert np.allclose(C, C.T)


def test_forecast_regime_probs_shape(fitted_var):
    result = fitted_var.forecast(steps=5, return_covariance=True)
    assert result["regime_probabilities"].shape == (5, fitted_var.n_regimes)
    assert np.allclose(result["regime_probabilities"].sum(axis=1), 1.0)


def test_forecast_single_regime(fitted_var):
    fc0 = fitted_var.forecast(steps=5, regime=0)
    fc1 = fitted_var.forecast(steps=5, regime=1)
    assert fc0.shape == (5, 3)
    assert fc1.shape == (5, 3)


def test_forecast_horizons(fitted_var):
    for steps in [1, 5, 10, 20]:
        fc = fitted_var.forecast(steps=steps)
        assert fc.shape == (steps, 3)


def test_forecast_regime_probabilities(fitted_var):
    probs = fitted_var.forecast_regime_probabilities(steps=10)
    assert probs.shape == (10, fitted_var.n_regimes)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ─── Parameters ─────────────────────────────────────────────────────────────────

def test_get_regime_parameters_shape(fitted_var):
    params = fitted_var.get_regime_parameters()
    assert isinstance(params, pd.DataFrame)
    assert len(params) == fitted_var.n_regimes


def test_get_regime_parameters_columns(fitted_var):
    params = fitted_var.get_regime_parameters()
    expected_cols = [
        "regime", "spectral_radius", "avg_volatility",
        "min_volatility", "max_volatility", "avg_correlation", "log_det_Sigma",
    ]
    for col in expected_cols:
        assert col in params.columns


def test_spectral_radius_reasonable(fitted_var):
    params = fitted_var.get_regime_parameters()
    # Spectral radius should be < 1 for stationary VAR
    assert np.all(params["spectral_radius"] < 2.0)


def test_get_var_matrices(fitted_var):
    for k in range(fitted_var.n_regimes):
        mats = fitted_var.get_var_matrices(k)
        assert "intercept" in mats
        assert "ar_coeffs" in mats
        assert "Sigma" in mats
        assert mats["intercept"].shape == (3,)
        assert len(mats["ar_coeffs"]) == fitted_var.order
        assert mats["ar_coeffs"][0].shape == (3, 3)
        assert mats["Sigma"].shape == (3, 3)


def test_get_var_matrices_invalid_regime(fitted_var):
    with pytest.raises(ValueError, match="Regime"):
        fitted_var.get_var_matrices(5)


def test_sigma_positive_definite(fitted_var):
    for k in range(fitted_var.n_regimes):
        Sigma = fitted_var.get_var_matrices(k)["Sigma"]
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)


def test_interpret_regimes(fitted_var):
    interp = fitted_var.interpret_regimes()
    assert len(interp) == fitted_var.n_regimes
    for r in range(fitted_var.n_regimes):
        assert isinstance(interp[r], str)


# ─── Before-fit guards ───────────────────────────────────────────────────────────

def test_before_fit_guards():
    ms = MarkovSwitchingVAR()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.predict_regime()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.predict_regime_proba()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.forecast(steps=5)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.get_regime_parameters()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.get_transition_matrix()


# ─── Persistence ────────────────────────────────────────────────────────────────

def test_save_load(fitted_var, tmp_path):
    path = str(tmp_path / "ms_var.joblib")
    fitted_var.save(path)
    loaded = MarkovSwitchingVAR.load(path)

    assert loaded.is_fitted
    assert loaded.n_regimes == fitted_var.n_regimes
    assert loaded.order == fitted_var.order
    assert loaded.n_assets_ == fitted_var.n_assets_

    orig_fc = fitted_var.forecast(steps=5)
    load_fc = loaded.forecast(steps=5)
    np.testing.assert_allclose(orig_fc, load_fc)

    orig_params = fitted_var.get_regime_parameters()
    load_params = loaded.get_regime_parameters()
    pd.testing.assert_frame_equal(
        orig_params.reset_index(drop=True),
        load_params.reset_index(drop=True),
    )


# ─── VAR(2) ────────────────────────────────────────────────────────────────────

def test_var2_fit(var_data):
    """Ensure order=2 fits without error."""
    ms = MarkovSwitchingVAR(n_regimes=2, order=2, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(var_data, num_restarts=2, maxiter=30)
    assert ms.is_fitted
    assert ms.order == 2
    fc = ms.forecast(steps=5)
    assert fc.shape == (5, 3)


# ─── 2-asset case ──────────────────────────────────────────────────────────────

def test_two_assets():
    """Minimal 2-asset test to confirm n_assets boundary."""
    rng = np.random.default_rng(99)
    Y = rng.standard_normal((200, 2)) * 0.05
    # Add some AR(1) structure
    for t in range(1, 200):
        Y[t] += 0.3 * Y[t - 1]

    ms = MarkovSwitchingVAR(n_regimes=2, order=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(Y, num_restarts=2, maxiter=30)
    assert ms.is_fitted
    assert ms.n_assets_ == 2
    fc = ms.forecast(steps=3)
    assert fc.shape == (3, 2)