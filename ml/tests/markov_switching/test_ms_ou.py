"""Tests for MarkovSwitchingOU."""

import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.markov_switching.ou import MarkovSwitchingOU


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def ou_series():
    """Synthetic two-regime OU series (unit time steps, well-scaled for EM stability)."""
    rng = np.random.default_rng(42)
    T = 600
    # Use dt=1 (bar units) so values stay in a numerically comfortable range.
    # Regime 0: fast mean reversion  κ=0.3, θ=0.0, σ=0.5
    # Regime 1: slow mean reversion  κ=0.05, θ=1.0, σ=1.0
    dt = 1.0
    y = np.empty(T)
    y[0] = 0.5
    regime = 0
    for t in range(1, T):
        if rng.random() < 0.05:
            regime = 1 - regime
        if regime == 0:
            kappa, theta, sigma = 0.3, 0.0, 0.5
        else:
            kappa, theta, sigma = 0.05, 1.0, 1.0
        phi = np.exp(-kappa * dt)
        sigma_disc = np.sqrt(sigma ** 2 / (2 * kappa) * (1 - phi ** 2))
        y[t] = theta + phi * (y[t - 1] - theta) + rng.normal(0, sigma_disc)
    return y


@pytest.fixture
def fitted_ms_ou(ou_series):
    ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms_ou.fit(ou_series, num_restarts=3)
    return ms_ou


# ─── Constructor ────────────────────────────────────────────────────────────────

def test_constructor_defaults():
    ms_ou = MarkovSwitchingOU()
    assert ms_ou.n_regimes == 2
    assert ms_ou.order == 1
    assert ms_ou.dt == 1.0
    assert not ms_ou.is_fitted


def test_constructor_custom_dt():
    ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1 / 52)
    assert ms_ou.dt == pytest.approx(1 / 52)


def test_constructor_invalid_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        MarkovSwitchingOU(dt=0.0)
    with pytest.raises(ValueError, match="dt must be positive"):
        MarkovSwitchingOU(dt=-1.0)


def test_order_is_fixed():
    ms_ou = MarkovSwitchingOU()
    assert ms_ou.order == 1
    with pytest.raises(ValueError, match="order=1"):
        ms_ou.order = 2


def test_n_regimes_validation():
    with pytest.raises(ValueError, match="n_regimes must be at least 2"):
        MarkovSwitchingOU(n_regimes=1)


# ─── Fitting ────────────────────────────────────────────────────────────────────

def test_fit_basic(ou_series):
    ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ms_ou.fit(ou_series, num_restarts=3)

    assert result is ms_ou
    assert ms_ou.is_fitted
    assert ms_ou.ou_parameters_ is not None
    assert len(ms_ou.ou_parameters_) == 2


def test_fit_populates_ou_parameters(fitted_ms_ou):
    params = fitted_ms_ou.ou_parameters_
    assert isinstance(params, pd.DataFrame)
    assert set(params.columns) >= {"regime", "phi", "kappa", "theta", "sigma",
                                   "sigma_disc", "half_life", "is_mean_reverting"}


def test_fit_returns_self(ou_series):
    ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert ms_ou.fit(ou_series, num_restarts=3) is ms_ou


# ─── OU parameter recovery ───────────────────────────────────────────────────────

def test_ou_parameters_shape(fitted_ms_ou):
    params = fitted_ms_ou.get_ou_parameters()
    assert params.shape == (fitted_ms_ou.n_regimes, 8)


def test_kappa_positive_for_mean_reverting(fitted_ms_ou):
    params = fitted_ms_ou.get_ou_parameters()
    mr = params[params["is_mean_reverting"]]
    assert (mr["kappa"] > 0).all()


def test_half_life_consistent_with_kappa(fitted_ms_ou):
    params = fitted_ms_ou.get_ou_parameters()
    mr = params[params["is_mean_reverting"]]
    for _, row in mr.iterrows():
        expected_hl = np.log(2) / row["kappa"]
        assert row["half_life"] == pytest.approx(expected_hl, rel=1e-6)


def test_phi_kappa_roundtrip(fitted_ms_ou):
    """κ = −ln(φ)/Δt  must be consistent with the stored φ."""
    params = fitted_ms_ou.get_ou_parameters()
    dt = fitted_ms_ou.dt
    mr = params[params["is_mean_reverting"]]
    for _, row in mr.iterrows():
        kappa_from_phi = -np.log(row["phi"]) / dt
        assert kappa_from_phi == pytest.approx(row["kappa"], rel=1e-6)


def test_sigma_continuous_vs_discrete(fitted_ms_ou):
    """σ²_disc = σ²_cont/(2κ) · (1−φ²)."""
    params = fitted_ms_ou.get_ou_parameters()
    dt = fitted_ms_ou.dt
    mr = params[params["is_mean_reverting"]]
    for _, row in mr.iterrows():
        phi = row["phi"]
        kappa = row["kappa"]
        sigma_disc_expected = row["sigma"] * np.sqrt((1 - phi ** 2) / (2 * kappa))
        assert row["sigma_disc"] == pytest.approx(sigma_disc_expected, rel=1e-5)


def test_is_mean_reverting_method(fitted_ms_ou):
    for r in range(fitted_ms_ou.n_regimes):
        result = fitted_ms_ou.is_mean_reverting(r)
        assert isinstance(result, bool)


def test_is_mean_reverting_invalid_regime(fitted_ms_ou):
    with pytest.raises(ValueError):
        fitted_ms_ou.is_mean_reverting(99)


# ─── Inherited API still works ───────────────────────────────────────────────────

def test_predict_regime(fitted_ms_ou, ou_series):
    regimes = fitted_ms_ou.predict_regime()
    assert regimes.min() >= 0
    assert regimes.max() < fitted_ms_ou.n_regimes


def test_forecast_returns_array(fitted_ms_ou):
    fc = fitted_ms_ou.forecast(steps=10)
    assert len(fc) == 10
    assert not np.any(np.isnan(fc))


def test_forecast_with_variance(fitted_ms_ou):
    result = fitted_ms_ou.forecast(steps=10, return_variance=True)
    assert "mean" in result and "variance" in result
    assert np.all(result["variance"] > 0)


def test_forecast_regime_probabilities(fitted_ms_ou):
    probs = fitted_ms_ou.forecast_regime_probabilities(steps=5)
    assert probs.shape == (5, fitted_ms_ou.n_regimes)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_get_transition_matrix(fitted_ms_ou):
    P = fitted_ms_ou.get_transition_matrix()
    assert P.shape == (2, 2)
    assert np.allclose(P.sum(axis=1), 1.0)


# ─── interpret_regimes ───────────────────────────────────────────────────────────

def test_interpret_regimes_returns_dict(fitted_ms_ou):
    interp = fitted_ms_ou.interpret_regimes()
    assert isinstance(interp, dict)
    assert len(interp) == fitted_ms_ou.n_regimes
    for r in range(fitted_ms_ou.n_regimes):
        assert isinstance(interp[r], str)
        assert len(interp[r]) > 0


# ─── Before-fit guards ───────────────────────────────────────────────────────────

def test_get_ou_parameters_before_fit():
    ms_ou = MarkovSwitchingOU()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms_ou.get_ou_parameters()


def test_forecast_before_fit():
    ms_ou = MarkovSwitchingOU()
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms_ou.forecast(steps=5)


# ─── Persistence ────────────────────────────────────────────────────────────────

def test_save_load(fitted_ms_ou, tmp_path):
    path = str(tmp_path / "ms_ou.joblib")
    fitted_ms_ou.save(path)
    loaded = MarkovSwitchingOU.load(path)

    assert loaded.is_fitted
    assert loaded.dt == fitted_ms_ou.dt
    assert loaded.n_regimes == fitted_ms_ou.n_regimes

    orig_params = fitted_ms_ou.get_ou_parameters()
    load_params = loaded.get_ou_parameters()
    pd.testing.assert_frame_equal(orig_params.reset_index(drop=True),
                                  load_params.reset_index(drop=True))

    orig_fc = fitted_ms_ou.forecast(steps=5)
    load_fc = loaded.forecast(steps=5)
    np.testing.assert_allclose(orig_fc, load_fc)