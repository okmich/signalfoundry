"""Tests for MarkovSwitchingGARCH."""

import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.markov_switching.garch import MarkovSwitchingGARCH


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def garch_series():
    """Synthetic two-regime GARCH(1,1) series."""
    rng = np.random.default_rng(7)
    T = 600
    y = np.empty(T)
    y[0] = 0.0
    sigma2 = np.empty(T)
    regime = 0

    # Regime 0: low volatility,  α=0.05, β=0.90, ω=0.002
    # Regime 1: high volatility, α=0.12, β=0.82, ω=0.010
    params = [
        {"omega": 0.002, "alpha": 0.05, "beta": 0.90, "mu": 0.001},
        {"omega": 0.010, "alpha": 0.12, "beta": 0.82, "mu": -0.001},
    ]
    sigma2[0] = params[regime]["omega"] / (1 - params[regime]["alpha"] - params[regime]["beta"])

    for t in range(1, T):
        if rng.random() < 0.05:
            regime = 1 - regime
        p = params[regime]
        sigma2[t] = p["omega"] + p["alpha"] * y[t - 1] ** 2 + p["beta"] * sigma2[t - 1]
        y[t] = p["mu"] + np.sqrt(sigma2[t]) * rng.standard_normal()

    return y


@pytest.fixture
def fitted_garch(garch_series):
    ms = MarkovSwitchingGARCH(n_regimes=2, order=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(garch_series, num_restarts=3, maxiter=50)
    return ms


# ─── Constructor ────────────────────────────────────────────────────────────────

def test_constructor_defaults():
    ms = MarkovSwitchingGARCH()
    assert ms.n_regimes == 2
    assert ms.order == 1
    assert not ms.is_fitted


def test_constructor_invalid_n_regimes():
    with pytest.raises(ValueError, match="n_regimes must be at least 2"):
        MarkovSwitchingGARCH(n_regimes=1)


def test_constructor_invalid_order():
    with pytest.raises(ValueError, match="order must be at least 1"):
        MarkovSwitchingGARCH(order=0)


# ─── Fitting ────────────────────────────────────────────────────────────────────

def test_fit_returns_self(garch_series):
    ms = MarkovSwitchingGARCH(n_regimes=2, order=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ms.fit(garch_series, num_restarts=1, maxiter=20)
    assert result is ms


def test_fit_sets_is_fitted(fitted_garch):
    assert fitted_garch.is_fitted


def test_fit_short_series():
    ms = MarkovSwitchingGARCH(n_regimes=2, order=1)
    with pytest.raises(ValueError, match="Time series too short"):
        ms.fit(np.random.randn(10))


def test_fit_stores_data(fitted_garch, garch_series):
    assert fitted_garch.data_ is not None
    assert len(fitted_garch.data_) == len(garch_series)


def test_fit_aic_bic_set(fitted_garch):
    assert fitted_garch.aic is not None
    assert fitted_garch.bic is not None
    assert fitted_garch.bic >= fitted_garch.aic  # BIC penalises more


# ─── Transition matrix ───────────────────────────────────────────────────────────

def test_transition_matrix_shape(fitted_garch):
    P = fitted_garch.get_transition_matrix()
    assert P.shape == (2, 2)


def test_transition_matrix_row_stochastic(fitted_garch):
    P = fitted_garch.get_transition_matrix()
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


# ─── Regime probabilities ────────────────────────────────────────────────────────

def test_filtered_probs_shape(fitted_garch, garch_series):
    fp = fitted_garch.filtered_probabilities_
    assert fp.shape[0] == len(garch_series) - fitted_garch.order
    assert fp.shape[1] == fitted_garch.n_regimes


def test_smoothed_probs_shape(fitted_garch, garch_series):
    sp = fitted_garch.regime_probabilities_
    assert sp.shape[0] == len(garch_series) - fitted_garch.order
    assert sp.shape[1] == fitted_garch.n_regimes


def test_probs_sum_to_one(fitted_garch):
    assert np.allclose(fitted_garch.filtered_probabilities_.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(fitted_garch.regime_probabilities_.sum(axis=1), 1.0, atol=1e-6)


def test_predict_regime_shape(fitted_garch, garch_series):
    regimes = fitted_garch.predict_regime()
    assert len(regimes) == len(garch_series) - fitted_garch.order
    assert regimes.min() >= 0
    assert regimes.max() < fitted_garch.n_regimes


def test_predict_regime_causal(fitted_garch):
    r_smoothed = fitted_garch.predict_regime(causal=False)
    r_filtered = fitted_garch.predict_regime(causal=True)
    assert len(r_smoothed) == len(r_filtered)


def test_predict_regime_proba_shape(fitted_garch, garch_series):
    proba = fitted_garch.predict_regime_proba()
    assert proba.shape == (len(garch_series) - fitted_garch.order, fitted_garch.n_regimes)


# ─── Forecasting ────────────────────────────────────────────────────────────────

def test_forecast_basic_shape(fitted_garch):
    fc = fitted_garch.forecast(steps=10)
    assert len(fc) == 10
    assert isinstance(fc, np.ndarray)


def test_forecast_no_nan(fitted_garch):
    fc = fitted_garch.forecast(steps=10)
    assert not np.any(np.isnan(fc))


def test_forecast_with_variance_keys(fitted_garch):
    result = fitted_garch.forecast(steps=5, return_variance=True)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "variance" in result
    assert "regime_probabilities" in result
    assert "regime_forecasts" in result


def test_forecast_variance_positive(fitted_garch):
    result = fitted_garch.forecast(steps=5, return_variance=True)
    assert np.all(result["variance"] > 0)


def test_forecast_regime_probs_shape(fitted_garch):
    result = fitted_garch.forecast(steps=5, return_variance=True)
    assert result["regime_probabilities"].shape == (5, fitted_garch.n_regimes)
    assert np.allclose(result["regime_probabilities"].sum(axis=1), 1.0)


def test_forecast_single_regime(fitted_garch):
    fc0 = fitted_garch.forecast(steps=5, regime=0)
    fc1 = fitted_garch.forecast(steps=5, regime=1)
    assert len(fc0) == 5
    assert len(fc1) == 5


def test_forecast_horizons(fitted_garch):
    for steps in [1, 5, 10, 20]:
        fc = fitted_garch.forecast(steps=steps)
        assert len(fc) == steps


def test_forecast_regime_probabilities(fitted_garch):
    probs = fitted_garch.forecast_regime_probabilities(steps=10)
    assert probs.shape == (10, fitted_garch.n_regimes)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ─── Parameters ─────────────────────────────────────────────────────────────────

def test_get_regime_parameters_shape(fitted_garch):
    params = fitted_garch.get_regime_parameters()
    assert isinstance(params, pd.DataFrame)
    assert len(params) == fitted_garch.n_regimes


def test_get_regime_parameters_columns(fitted_garch):
    params = fitted_garch.get_regime_parameters()
    assert "regime" in params.columns
    assert "omega" in params.columns
    assert "alpha" in params.columns
    assert "beta" in params.columns
    assert "garch_persistence" in params.columns
    assert "unconditional_variance" in params.columns


def test_garch_stationarity(fitted_garch):
    """α + β < 1 for all regimes."""
    params = fitted_garch.get_regime_parameters()
    assert np.all(params["garch_persistence"] < 1.0)


def test_omega_positive(fitted_garch):
    params = fitted_garch.get_regime_parameters()
    assert np.all(params["omega"] > 0)


def test_interpret_regimes(fitted_garch):
    interp = fitted_garch.interpret_regimes()
    assert len(interp) == fitted_garch.n_regimes
    for r in range(fitted_garch.n_regimes):
        assert isinstance(interp[r], str)


# ─── Before-fit guards ───────────────────────────────────────────────────────────

def test_before_fit_guards():
    ms = MarkovSwitchingGARCH()
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

def test_save_load(fitted_garch, tmp_path):
    path = str(tmp_path / "ms_garch.joblib")
    fitted_garch.save(path)
    loaded = MarkovSwitchingGARCH.load(path)

    assert loaded.is_fitted
    assert loaded.n_regimes == fitted_garch.n_regimes
    assert loaded.order == fitted_garch.order

    orig_fc = fitted_garch.forecast(steps=5)
    load_fc = loaded.forecast(steps=5)
    np.testing.assert_allclose(orig_fc, load_fc)

    orig_params = fitted_garch.get_regime_parameters()
    load_params = loaded.get_regime_parameters()
    pd.testing.assert_frame_equal(
        orig_params.reset_index(drop=True),
        load_params.reset_index(drop=True),
    )