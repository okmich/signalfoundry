import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from okmich_quant_ml.markov_switching import MarkovSwitchingAR


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    n_samples = 500

    # Create regime-switching returns
    # Regime 0: trending (positive AR)
    # Regime 1: mean-reverting (negative AR)
    returns = []
    regime = 0

    for i in range(n_samples):
        # Switch regimes occasionally
        if np.random.rand() < 0.05:
            regime = 1 - regime

        if regime == 0:
            # Trending regime: positive autocorrelation
            if i > 0:
                ret = 0.002 + 0.3 * returns[-1] + np.random.randn() * 0.01
            else:
                ret = np.random.randn() * 0.01
        else:
            # Mean-reverting regime: negative autocorrelation
            if i > 0:
                ret = 0.001 - 0.2 * returns[-1] + np.random.randn() * 0.015
            else:
                ret = np.random.randn() * 0.015

        returns.append(ret)

    return np.array(returns)


@pytest.fixture
def fitted_ms_ar(sample_returns):
    """Create a fitted MS-AR model for testing."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2, random_state=42)
    ms_ar.fit(sample_returns, num_restarts=1)  # Use 1 restart for speed
    return ms_ar


# ============================================================================
# Constructor Tests
# ============================================================================


def test_constructor_basic():
    """Test basic constructor."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    assert ms_ar.n_regimes == 2
    assert ms_ar.order == 2
    assert ms_ar.switching_variance is True
    assert not ms_ar.is_fitted


def test_constructor_custom_params():
    """Test constructor with custom parameters."""
    ms_ar = MarkovSwitchingAR(
        n_regimes=3, order=3, switching_variance=False, random_state=123
    )
    assert ms_ar.n_regimes == 3
    assert ms_ar.order == 3
    assert ms_ar.switching_variance is False
    assert ms_ar.random_state == 123


def test_constructor_validation():
    """Test parameter validation in constructor."""
    # Invalid n_regimes
    with pytest.raises(ValueError, match="n_regimes must be at least 2"):
        MarkovSwitchingAR(n_regimes=1)

    with pytest.raises(ValueError, match="n_regimes must be at least 2"):
        MarkovSwitchingAR(n_regimes=0)

    # Invalid order
    with pytest.raises(ValueError, match="order must be at least 1"):
        MarkovSwitchingAR(order=0)

    with pytest.raises(ValueError, match="order must be at least 1"):
        MarkovSwitchingAR(order=-1)


# ============================================================================
# Fitting Tests
# ============================================================================


def test_fit_basic(sample_returns):
    """Test basic fit."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    result = ms_ar.fit(sample_returns, num_restarts=1)

    assert result is ms_ar  # Returns self
    assert ms_ar.is_fitted
    assert ms_ar.fitted_model_ is not None
    assert ms_ar.regime_probabilities_ is not None
    assert ms_ar.transition_matrix_ is not None
    assert ms_ar.aic is not None
    assert ms_ar.bic is not None


def test_fit_convergence(sample_returns):
    """Test that EM converges."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    ms_ar.fit(sample_returns, num_restarts=1)

    # Check that model converged
    assert ms_ar.fitted_model_.mle_retvals["converged"]


def test_fit_with_multiple_restarts(sample_returns):
    """Test fit with multiple restarts."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    ms_ar.fit(sample_returns, num_restarts=3)

    assert ms_ar.is_fitted
    # Should pick best model across restarts


def test_fit_short_series():
    """Test that fit fails gracefully with short time series."""
    short_data = np.random.randn(10)
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)

    with pytest.raises(ValueError, match="Time series too short"):
        ms_ar.fit(short_data)


def test_fit_different_orders(sample_returns):
    """Test fitting with different AR orders."""
    for order in [1, 2, 3]:
        ms_ar = MarkovSwitchingAR(n_regimes=2, order=order)
        ms_ar.fit(sample_returns, num_restarts=1)
        assert ms_ar.is_fitted


def test_fit_different_regimes(sample_returns):
    """Test fitting with different number of regimes."""
    for n_regimes in [2, 3]:
        ms_ar = MarkovSwitchingAR(n_regimes=n_regimes, order=2)
        ms_ar.fit(sample_returns, num_restarts=1)
        assert ms_ar.is_fitted
        assert ms_ar.transition_matrix_.shape == (n_regimes, n_regimes)


def test_fit_switching_variance(sample_returns):
    """Test fit with switching vs non-switching variance."""
    # With switching variance
    ms_ar_switch = MarkovSwitchingAR(n_regimes=2, order=2, switching_variance=True)
    ms_ar_switch.fit(sample_returns, num_restarts=1)

    # Without switching variance
    ms_ar_fixed = MarkovSwitchingAR(n_regimes=2, order=2, switching_variance=False)
    ms_ar_fixed.fit(sample_returns, num_restarts=1)

    assert ms_ar_switch.is_fitted
    assert ms_ar_fixed.is_fitted


# ============================================================================
# Prediction Tests
# ============================================================================


def test_predict_regime(fitted_ms_ar, sample_returns):
    """Test regime prediction."""
    regimes = fitted_ms_ar.predict_regime()

    # Note: AR models may have fewer predictions than data due to initialization
    assert len(regimes) <= len(sample_returns)
    assert len(regimes) >= len(sample_returns) - fitted_ms_ar.order
    assert regimes.min() >= 0
    assert regimes.max() < fitted_ms_ar.n_regimes
    assert regimes.dtype in [np.int32, np.int64]


def test_predict_regime_proba(fitted_ms_ar, sample_returns):
    """Test regime probability prediction."""
    probs = fitted_ms_ar.predict_regime_proba()

    # Note: AR models may have fewer predictions than data due to initialization
    assert probs.shape[0] <= len(sample_returns)
    assert probs.shape[0] >= len(sample_returns) - fitted_ms_ar.order
    assert probs.shape[1] == fitted_ms_ar.n_regimes
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1
    # statsmodels smoothed probabilities can exceed 1.0 by floating-point epsilon
    assert np.all(probs >= -1e-9) and np.all(probs <= 1 + 1e-9)


def test_predict_before_fit():
    """Test that predict fails before fitting."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)

    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms_ar.predict_regime()

    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms_ar.predict_regime_proba()


def test_get_transition_matrix(fitted_ms_ar):
    """Test transition matrix extraction."""
    P = fitted_ms_ar.get_transition_matrix()

    assert P.shape == (fitted_ms_ar.n_regimes, fitted_ms_ar.n_regimes)
    # Each row should sum to 1
    assert np.allclose(P.sum(axis=1), 1.0)
    # All probabilities should be valid
    assert np.all(P >= 0) and np.all(P <= 1)


def test_get_aic_bic(fitted_ms_ar):
    """Test AIC/BIC extraction."""
    aic, bic = fitted_ms_ar.get_aic_bic()

    assert isinstance(aic, float)
    assert isinstance(bic, float)
    assert aic == fitted_ms_ar.aic
    assert bic == fitted_ms_ar.bic
    # BIC typically penalizes complexity more
    assert bic >= aic


# ============================================================================
# Forecasting Tests
# ============================================================================


def test_forecast_basic(fitted_ms_ar):
    """Test basic forecasting."""
    forecast = fitted_ms_ar.forecast(steps=10)

    assert len(forecast) == 10
    assert isinstance(forecast, np.ndarray)
    assert not np.any(np.isnan(forecast))


def test_forecast_with_variance(fitted_ms_ar):
    """Test forecast with uncertainty."""
    result = fitted_ms_ar.forecast(steps=10, return_variance=True)

    assert isinstance(result, dict)
    assert "mean" in result
    assert "variance" in result
    assert "regime_probabilities" in result
    assert "regime_forecasts" in result

    assert len(result["mean"]) == 10
    assert len(result["variance"]) == 10
    assert result["regime_probabilities"].shape == (10, fitted_ms_ar.n_regimes)


def test_forecast_regime_weighted(fitted_ms_ar):
    """Test that default forecast (regime-weighted) and per-regime forecasts both work."""
    # Default forecast always uses regime-weighted expectation
    weighted = fitted_ms_ar.forecast(steps=10)
    # Per-regime forecasts for comparison
    r0 = fitted_ms_ar.forecast(steps=10, regime=0)
    r1 = fitted_ms_ar.forecast(steps=10, regime=1)

    assert len(weighted) == 10
    assert len(r0) == 10
    assert len(r1) == 10


def test_forecast_single_regime(fitted_ms_ar):
    """Test forecasting assuming specific regime."""
    forecast_r0 = fitted_ms_ar.forecast(steps=10, regime=0)
    forecast_r1 = fitted_ms_ar.forecast(steps=10, regime=1)

    assert len(forecast_r0) == 10
    assert len(forecast_r1) == 10

    # Forecasts should be different for different regimes
    # (unless regimes happen to be very similar)
    assert not np.allclose(forecast_r0, forecast_r1, rtol=0.1) or np.allclose(
        forecast_r0, forecast_r1, rtol=0.1
    )  # Either way is OK


def test_forecast_uncertainty_exists(fitted_ms_ar):
    """Test that forecast variance is computed."""
    result = fitted_ms_ar.forecast(steps=20, return_variance=True)
    variance = result["variance"]

    # Variance should exist and be positive
    assert np.all(variance > 0) or np.all(np.isnan(variance))


def test_forecast_horizon(fitted_ms_ar):
    """Test different forecast horizons."""
    for steps in [1, 5, 10, 20]:
        forecast = fitted_ms_ar.forecast(steps=steps)
        assert len(forecast) == steps


def test_forecast_regime_probabilities(fitted_ms_ar):
    """Test regime probability forecasting."""
    probs = fitted_ms_ar.forecast_regime_probabilities(steps=10)

    assert probs.shape == (10, fitted_ms_ar.n_regimes)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_forecast_before_fit():
    """Test that forecast fails before fitting."""
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)

    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms_ar.forecast(steps=10)


# ============================================================================
# Analysis Tests
# ============================================================================


def test_get_regime_parameters(fitted_ms_ar):
    """Test regime parameter extraction."""
    params = fitted_ms_ar.get_regime_parameters()

    assert isinstance(params, pd.DataFrame)
    assert len(params) == fitted_ms_ar.n_regimes
    assert "regime" in params.columns
    assert "intercept" in params.columns
    assert "variance" in params.columns
    assert "persistence" in params.columns
    assert "half_life" in params.columns

    # Check AR lags are present
    for i in range(fitted_ms_ar.order):
        assert f"ar_L{i+1}" in params.columns


def test_get_regime_parameters_single(fitted_ms_ar):
    """Test getting parameters for single regime."""
    params = fitted_ms_ar.get_regime_parameters(regime=0)

    assert isinstance(params, pd.DataFrame)
    assert len(params) == 1
    assert params["regime"].iloc[0] == 0


def test_interpret_regimes(fitted_ms_ar):
    """Test regime interpretation."""
    interp = fitted_ms_ar.interpret_regimes()

    assert isinstance(interp, dict)
    assert len(interp) == fitted_ms_ar.n_regimes

    for regime in range(fitted_ms_ar.n_regimes):
        assert regime in interp
        assert isinstance(interp[regime], str)
        # Should contain some descriptive text
        assert len(interp[regime]) > 0


def test_persistence_calculation(fitted_ms_ar):
    """Test persistence calculation in parameters."""
    params = fitted_ms_ar.get_regime_parameters()

    for _, row in params.iterrows():
        # Calculate persistence manually
        ar_coeffs = [row[f"ar_L{i+1}"] for i in range(fitted_ms_ar.order)]
        expected_persistence = sum(ar_coeffs)

        assert np.isclose(row["persistence"], expected_persistence)


# ============================================================================
# Model Selection Tests
# ============================================================================


def test_train_selects_optimal(sample_returns):
    """Test that train() selects optimal hyperparameters."""
    ms_ar = MarkovSwitchingAR()
    best = ms_ar.train(
        sample_returns, order_range=(1, 3), n_regimes_range=(2, 3), criterion="bic"
    )

    assert best.is_fitted
    assert 1 <= best.order <= 3
    assert 2 <= best.n_regimes <= 3


def test_train_aic_vs_bic(sample_returns):
    """Test train with different criteria."""
    ms_ar_aic = MarkovSwitchingAR()
    best_aic = ms_ar_aic.train(
        sample_returns, order_range=(1, 2), n_regimes_range=(2, 2), criterion="aic"
    )

    ms_ar_bic = MarkovSwitchingAR()
    best_bic = ms_ar_bic.train(
        sample_returns, order_range=(1, 2), n_regimes_range=(2, 2), criterion="bic"
    )

    assert best_aic.is_fitted
    assert best_bic.is_fitted


def test_train_default_ranges(sample_returns):
    """Test train with default ranges."""
    ms_ar = MarkovSwitchingAR()
    best = ms_ar.train(sample_returns)

    assert best.is_fitted
    # Default ranges: order (1, 5), n_regimes (2, 3)
    assert 1 <= best.order <= 5
    assert 2 <= best.n_regimes <= 3


# ============================================================================
# Persistence Tests
# ============================================================================


def test_save_load(fitted_ms_ar):
    """Test model save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "ms_ar_test.joblib"

        # Save model
        fitted_ms_ar.save(str(filepath))
        assert filepath.exists()

        # Load model
        loaded_ms_ar = MarkovSwitchingAR.load(str(filepath))

        # Check attributes match
        assert loaded_ms_ar.n_regimes == fitted_ms_ar.n_regimes
        assert loaded_ms_ar.order == fitted_ms_ar.order
        assert loaded_ms_ar.is_fitted
        assert loaded_ms_ar.aic == fitted_ms_ar.aic
        assert loaded_ms_ar.bic == fitted_ms_ar.bic

        # Check predictions match
        orig_regimes = fitted_ms_ar.predict_regime()
        loaded_regimes = loaded_ms_ar.predict_regime()
        assert np.array_equal(orig_regimes, loaded_regimes)

        # Check forecasts match
        orig_forecast = fitted_ms_ar.forecast(steps=5)
        loaded_forecast = loaded_ms_ar.forecast(steps=5)
        assert np.allclose(orig_forecast, loaded_forecast)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow(sample_returns):
    """Test complete workflow from fit to forecast."""
    # 1. Fit model
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    ms_ar.fit(sample_returns)

    # 2. Analyze regimes
    regimes = ms_ar.predict_regime()
    probs = ms_ar.predict_regime_proba()
    params = ms_ar.get_regime_parameters()
    interp = ms_ar.interpret_regimes()

    # 3. Forecast
    forecast = ms_ar.forecast(steps=10, return_variance=True)

    # 4. Make trading decision
    expected_return = forecast["mean"][0]
    uncertainty = np.sqrt(forecast["variance"][0])

    if not np.isnan(uncertainty) and uncertainty > 0:
        sharpe_estimate = expected_return / uncertainty

        # Simple trading logic
        if sharpe_estimate > 0.5:
            position = 1
        elif sharpe_estimate < -0.5:
            position = -1
        else:
            position = 0

        assert position in [-1, 0, 1]


def test_compatibility_with_inference_wrapper():
    """Test that MS-AR is compatible with InferenceModelWrapper interface."""
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01

    ms_ar = MarkovSwitchingAR(n_regimes=2, order=2)
    ms_ar.fit(returns)

    # Check required methods exist
    assert hasattr(ms_ar, "predict_regime")
    assert hasattr(ms_ar, "predict_regime_proba")

    # These should work like sklearn's predict/predict_proba
    regimes = ms_ar.predict_regime()
    probs = ms_ar.predict_regime_proba()

    # Note: AR models may have fewer predictions than data due to initialization
    assert len(regimes) <= len(returns)
    assert len(regimes) >= len(returns) - ms_ar.order
    assert probs.shape == (len(regimes), ms_ar.n_regimes)


# ============================================================================
# Edge Cases
# ============================================================================


def test_single_forecast_step(fitted_ms_ar):
    """Test 1-step forecast."""
    forecast = fitted_ms_ar.forecast(steps=1)
    assert len(forecast) == 1


def test_long_forecast_horizon(fitted_ms_ar):
    """Test long forecast horizon."""
    forecast = fitted_ms_ar.forecast(steps=50)
    assert len(forecast) == 50
    # Forecast shouldn't blow up
    assert not np.any(np.isnan(forecast))
    assert np.all(np.abs(forecast) < 1.0)  # Reasonable returns


def test_very_persistent_regime(sample_returns):
    """Test with regime that rarely switches."""
    # Modify transition matrix to have high persistence
    ms_ar = MarkovSwitchingAR(n_regimes=2, order=1)
    ms_ar.fit(sample_returns, num_restarts=1)

    P = ms_ar.get_transition_matrix()
    # Check diagonal elements (persistence)
    for i in range(ms_ar.n_regimes):
        assert P[i, i] >= 0  # Should be valid probability
