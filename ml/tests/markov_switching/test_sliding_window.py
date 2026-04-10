"""Tests for Online Learning L2 — sliding-window refit."""

import warnings
from collections import deque

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
    y[0] = 0.5
    regime = 0
    for t in range(1, n):
        if rng.random() < 0.05:
            regime = 1 - regime
        if regime == 0:
            y[t] = 0.002 + 0.3 * y[t - 1] + rng.normal(0, 0.5)
        else:
            y[t] = 0.001 - 0.2 * y[t - 1] + rng.normal(0, 0.8)
    return y


@pytest.fixture
def fitted(returns):
    ms = MarkovSwitchingAR(n_regimes=2, order=2, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms.fit(returns, num_restarts=3)
    return ms


# ─── Buffer initialisation ───────────────────────────────────────────────────────

def test_buffer_created_on_first_call(fitted):
    assert not hasattr(fitted, "_window_buffer_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.1, window=100)
    assert hasattr(fitted, "_window_buffer_")
    assert isinstance(fitted._window_buffer_, deque)


def test_buffer_maxlen(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.1, window=150)
    assert fitted._window_buffer_.maxlen == 150


def test_buffer_seeded_from_training_data(fitted):
    window = 100
    # Save tail before any call that might overwrite data_
    training_tail = fitted.data_[-window:].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # refit_every=5 so the first call doesn't refit and overwrite data_
        fitted.refit_window(0.1, window=window, refit_every=5)
    buf = list(fitted._window_buffer_)
    assert buf[-1] == pytest.approx(0.1)
    np.testing.assert_allclose(buf[:-1], training_tail[-(window - 1):])


def test_buffer_reinitialised_on_window_change(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.1, window=100)
        fitted.refit_window(0.1, window=200)
    assert fitted._window_buffer_.maxlen == 200


def test_n_refits_initialised_to_zero(fitted):
    assert not hasattr(fitted, "_n_refits_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.1, window=100, refit_every=5)
    assert hasattr(fitted, "_n_refits_")


# ─── refit_every=1 ───────────────────────────────────────────────────────────────

def test_refit_every_1_increments_n_refits(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            fitted.refit_window(0.1, window=100, refit_every=1, num_restarts=1)
    assert fitted._n_refits_ == 3


def test_refit_every_1_resets_bars_since_refit(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.1, window=100, refit_every=1, num_restarts=1)
    assert fitted._bars_since_refit_ == 0


def test_returns_self(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fitted.refit_window(0.1, window=100, refit_every=1, num_restarts=1)
    assert result is fitted


# ─── refit_every=N ───────────────────────────────────────────────────────────────

def test_refit_every_n_delays_refit(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # First 4 calls should NOT refit (refit_every=5)
        for _ in range(4):
            fitted.refit_window(0.1, window=100, refit_every=5, num_restarts=1)
    assert fitted._n_refits_ == 0
    assert fitted._bars_since_refit_ == 4


def test_refit_every_n_triggers_on_nth(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(5):
            fitted.refit_window(0.1, window=100, refit_every=5, num_restarts=1)
    assert fitted._n_refits_ == 1
    assert fitted._bars_since_refit_ == 0


def test_batch_refit_every_n(fitted):
    """Passing an array should trigger the correct number of refits."""
    obs = np.ones(10) * 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(obs, window=100, refit_every=5, num_restarts=1)
    assert fitted._n_refits_ == 2  # refits at bar 5 and bar 10


# ─── Belief state updates ────────────────────────────────────────────────────────

def test_belief_state_valid_after_refit(fitted):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.5, window=100, refit_every=1, num_restarts=1)
    proba = fitted.current_regime_proba()
    assert proba.shape == (fitted.n_regimes,)
    assert np.isclose(proba.sum(), 1.0)
    assert np.all(proba >= 0)


def test_belief_state_changes_after_refit(fitted):
    proba_before = fitted.current_regime_proba().copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.5, window=100, refit_every=1, num_restarts=1)
    proba_after = fitted.current_regime_proba()
    # Params or belief state (or both) should change after refit on new data
    assert not np.allclose(proba_before, proba_after)


def test_l1_update_used_between_refits(fitted):
    """Between refits, belief state should still advance via L1 forward steps."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.refit_window(0.1, window=100, refit_every=5, num_restarts=1)  # no refit yet
    assert fitted._n_updates_ >= 1
    proba = fitted.current_regime_proba()
    assert np.isclose(proba.sum(), 1.0)


# ─── Refit failure fallback ──────────────────────────────────────────────────────

def test_fallback_to_l1_on_refit_failure(fitted):
    """If refit fails (e.g. too-short window), should fall back to L1 update."""
    proba_before = fitted.current_regime_proba().copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # window=5 is too short for order=2 (needs order+10=12) — refit will fail
        fitted.refit_window(0.1, window=5, refit_every=1, num_restarts=1)
    proba_after = fitted.current_regime_proba()
    # L1 update should still have changed belief state
    assert not np.allclose(proba_before, proba_after)
    assert fitted._n_refits_ == 0  # no successful refits


# ─── Before-fit guard ────────────────────────────────────────────────────────────

def test_refit_window_before_fit():
    ms = MarkovSwitchingAR(n_regimes=2, order=2)
    with pytest.raises(RuntimeError, match="Model has not been fitted"):
        ms.refit_window(0.1, window=100)


# ─── MS-OU inherits sliding window ───────────────────────────────────────────────

def test_ms_ou_inherits_refit_window(returns):
    ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1.0, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms_ou.fit(returns, num_restarts=3)
        ms_ou.refit_window(np.ones(5) * 0.1, window=100, refit_every=5, num_restarts=1)
    assert ms_ou._n_refits_ == 1
    assert np.isclose(ms_ou.current_regime_proba().sum(), 1.0)