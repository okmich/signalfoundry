"""Tests for Online Learning L1 — streaming filter (update / current_regime_proba / current_regime)."""

import warnings

import numpy as np
import pytest

from okmich_quant_ml.markov_switching import MarkovSwitchingAR
from okmich_quant_ml.markov_switching.ou import MarkovSwitchingOU


# ─── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    n = 500
    y = np.empty(n)
    y[0] = 0.0
    regime = 0
    for t in range(1, n):
        if rng.random() < 0.05:
            regime = 1 - regime
        if regime == 0:
            y[t] = 0.002 + 0.3 * y[t - 1] + rng.normal(0, 0.01)
        else:
            y[t] = 0.001 - 0.2 * y[t - 1] + rng.normal(0, 0.015)
    return y


@pytest.fixture
def fitted(returns):
    ms = MarkovSwitchingAR(n_regimes=2, order=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(returns, num_restarts=3)
    return ms


# ─── Streaming state initialisation ─────────────────────────────────────────────

def test_streaming_state_set_after_fit(fitted):
    assert hasattr(fitted, "_last_forward_alpha_")
    assert hasattr(fitted, "_history_buffer_")
    assert hasattr(fitted, "_n_updates_")
    assert fitted._n_updates_ == 0


def test_last_forward_alpha_matches_filtered_tail(fitted):
    np.testing.assert_allclose(
        fitted._last_forward_alpha_,
        fitted.filtered_probabilities_[-1],
        rtol=1e-6,
    )


def test_history_buffer_matches_data_tail(fitted):
    np.testing.assert_allclose(
        fitted._history_buffer_,
        fitted.data_[-fitted.order:],
    )


def test_cached_params_shape(fitted):
    assert fitted._cached_ar_coeffs_.shape == (fitted.n_regimes, fitted.order)
    assert fitted._cached_intercepts_.shape == (fitted.n_regimes,)
    assert fitted._cached_sigma2s_.shape == (fitted.n_regimes,)


# ─── current_regime_proba / current_regime before any update ─────────────────────

def test_current_regime_proba_before_update(fitted):
    proba = fitted.current_regime_proba()
    assert proba.shape == (fitted.n_regimes,)
    assert np.isclose(proba.sum(), 1.0)
    np.testing.assert_allclose(proba, fitted.filtered_probabilities_[-1], rtol=1e-6)


def test_current_regime_before_update(fitted):
    r = fitted.current_regime()
    assert 0 <= r < fitted.n_regimes


# ─── single-step update ──────────────────────────────────────────────────────────

def test_update_returns_self(fitted):
    result = fitted.update(0.001)
    assert result is fitted


def test_update_increments_n_updates(fitted):
    fitted.update(0.001)
    assert fitted._n_updates_ == 1
    fitted.update(0.002)
    assert fitted._n_updates_ == 2


def test_update_changes_belief_state(fitted):
    proba_before = fitted.current_regime_proba().copy()
    fitted.update(0.05)  # large positive return — should shift belief
    proba_after = fitted.current_regime_proba()
    assert not np.allclose(proba_before, proba_after)


def test_update_belief_sums_to_one(fitted):
    fitted.update(0.001)
    assert np.isclose(fitted.current_regime_proba().sum(), 1.0)


def test_update_belief_non_negative(fitted):
    fitted.update(-0.03)
    assert np.all(fitted.current_regime_proba() >= 0)


def test_current_regime_after_update(fitted):
    fitted.update(0.001)
    r = fitted.current_regime()
    assert 0 <= r < fitted.n_regimes


# ─── batch update ────────────────────────────────────────────────────────────────

def test_batch_update_equivalent_to_sequential(fitted):
    import copy
    # Start both from identical state — avoids label-switching across independent fits
    ms1 = copy.deepcopy(fitted)
    ms2 = copy.deepcopy(fitted)
    new_obs = np.array([0.001, -0.002, 0.003, -0.001, 0.005])

    # Batch
    ms1.update(new_obs)

    # Sequential
    for y in new_obs:
        ms2.update(y)

    np.testing.assert_allclose(ms1.current_regime_proba(), ms2.current_regime_proba(), rtol=1e-10)
    assert ms1._n_updates_ == ms2._n_updates_ == len(new_obs)


def test_batch_update_increments_n_updates(fitted):
    obs = np.array([0.001, -0.002, 0.003])
    fitted.update(obs)
    assert fitted._n_updates_ == 3


# ─── history buffer rolls correctly ──────────────────────────────────────────────

def test_history_buffer_rolls(fitted):
    y_new = 0.042
    fitted.update(y_new)
    assert fitted._history_buffer_[-1] == pytest.approx(y_new)


def test_history_buffer_length_unchanged(fitted):
    original_len = len(fitted._history_buffer_)
    fitted.update(np.array([0.001, 0.002, 0.003]))
    assert len(fitted._history_buffer_) == original_len


# ─── forecast uses updated state ─────────────────────────────────────────────────

def test_forecast_uses_updated_belief(returns):
    ms = MarkovSwitchingAR(n_regimes=2, order=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(returns, num_restarts=3)

    fc_before = ms.forecast(steps=5)
    ms.update(np.array([0.05, 0.04, 0.03]))  # strong positive shock
    fc_after = ms.forecast(steps=5)

    # Forecast should differ after a state update
    assert not np.allclose(fc_before, fc_after)


def test_forecast_regime_probs_uses_updated_alpha(returns):
    ms = MarkovSwitchingAR(n_regimes=2, order=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(returns, num_restarts=3)

    probs_before = ms.forecast_regime_probabilities(steps=3, causal=True)
    ms.update(0.05)
    probs_after = ms.forecast_regime_probabilities(steps=3, causal=True)

    assert not np.allclose(probs_before, probs_after)


# ─── before-fit guards ───────────────────────────────────────────────────────────

def test_update_before_fit():
    ms = MarkovSwitchingAR(n_regimes=2, order=2)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.update(0.001)


def test_current_regime_proba_before_fit():
    ms = MarkovSwitchingAR(n_regimes=2, order=2)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.current_regime_proba()


def test_current_regime_before_fit():
    ms = MarkovSwitchingAR(n_regimes=2, order=2)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.current_regime()


# ─── MS-OU inherits streaming ────────────────────────────────────────────────────

def test_ms_ou_inherits_update():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, 500).cumsum() * 0.01 + rng.normal(0, 0.5, 500)
    ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms_ou.fit(y, num_restarts=3)

    proba_before = ms_ou.current_regime_proba().copy()
    ms_ou.update(np.array([0.5, 1.0, -0.5]))
    proba_after = ms_ou.current_regime_proba()

    assert ms_ou._n_updates_ == 3
    assert not np.allclose(proba_before, proba_after)
    assert np.isclose(proba_after.sum(), 1.0)