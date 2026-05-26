"""Tests for per-bar predictive log-likelihood on BasePomegranateHMM.

Verifies:
- Shape matches input ``T``.
- Sum equals the joint log-likelihood ``log P(o_{1:T})`` derived
  independently from the model's own ``log_probability`` (when available)
  or from a manual forward-pass total.
- Rejects unfitted model, empty input, NaN/Inf input.
- Both PomegranateHMM and PomegranateMixtureHMM inherit the method.
- Output is finite for valid input.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

from okmich_quant_ml.hmm import DistType, InferenceMode, PomegranateHMM, PomegranateMixtureHMM


def _make_regime_data(seed, n=120, d=2, n_regimes=2):
    """Same data structure as test_fixed_lag_smoothing for cross-test consistency."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n, d))
    regime_size = n // n_regimes
    for r in range(n_regimes):
        start = r * regime_size
        end = start + regime_size if r < n_regimes - 1 else n
        mean = rng.uniform(-3, 3, size=d)
        data[start:end] = rng.normal(mean, 0.5, size=(end - start, d))
    return data


@pytest.fixture
def sample_data():
    return _make_regime_data(seed=42)


@pytest.fixture
def fitted_hmm(sample_data):
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, inference_mode=InferenceMode.FILTERING,
                           random_state=42)
    model.fit(sample_data)
    return model


@pytest.fixture
def fitted_mixture_hmm(sample_data):
    model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=2, n_components=2,
                                  inference_mode=InferenceMode.FILTERING, random_state=42)
    model.fit(sample_data)
    return model


def _joint_loglik_from_alpha(model, X):
    """Independent reference: total joint log-likelihood via the forward pass terminal alpha."""
    X_pp = model._preprocess_input(np.asarray(X))
    log_pi, log_A, log_B = model._extract_hmm_parameters(X_pp)
    T, K = log_B.shape
    log_alpha = np.empty((T, K), dtype=np.float64)
    log_alpha[0] = log_pi + log_B[0]
    for t in range(1, T):
        log_alpha[t] = np.logaddexp.reduce(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0) + log_B[t]
    return float(logsumexp(log_alpha[-1]))


# --- shape + finiteness -------------------------------------------------------

def test_compute_per_bar_loglik_returns_shape_T(fitted_hmm, sample_data):
    out = fitted_hmm.compute_per_bar_predictive_loglik(sample_data)

    assert out.shape == (sample_data.shape[0],)
    assert out.dtype == np.float64


def test_compute_per_bar_loglik_is_finite(fitted_hmm, sample_data):
    out = fitted_hmm.compute_per_bar_predictive_loglik(sample_data)

    assert np.all(np.isfinite(out))


# --- joint-loglik invariant ---------------------------------------------------

def test_compute_per_bar_loglik_sum_equals_joint_loglik(fitted_hmm, sample_data):
    out = fitted_hmm.compute_per_bar_predictive_loglik(sample_data)
    expected_joint = _joint_loglik_from_alpha(fitted_hmm, sample_data)

    assert float(out.sum()) == pytest.approx(expected_joint, rel=1e-9, abs=1e-9)


def test_compute_per_bar_loglik_first_element_equals_marginal_at_t0(fitted_hmm, sample_data):
    """loglik[0] == logsumexp(log_alpha[0]) == log P(o_0)."""
    out = fitted_hmm.compute_per_bar_predictive_loglik(sample_data)

    X_pp = fitted_hmm._preprocess_input(np.asarray(sample_data))
    log_pi, _, log_B = fitted_hmm._extract_hmm_parameters(X_pp)
    expected_loglik_0 = float(logsumexp(log_pi + log_B[0]))

    assert float(out[0]) == pytest.approx(expected_loglik_0, rel=1e-12, abs=1e-12)


# --- inheritance --------------------------------------------------------------

def test_mixture_hmm_inherits_compute_per_bar_loglik(fitted_mixture_hmm, sample_data):
    out = fitted_mixture_hmm.compute_per_bar_predictive_loglik(sample_data)

    assert out.shape == (sample_data.shape[0],)
    assert np.all(np.isfinite(out))


def test_mixture_hmm_sum_equals_joint_loglik(fitted_mixture_hmm, sample_data):
    out = fitted_mixture_hmm.compute_per_bar_predictive_loglik(sample_data)
    expected_joint = _joint_loglik_from_alpha(fitted_mixture_hmm, sample_data)

    assert float(out.sum()) == pytest.approx(expected_joint, rel=1e-9, abs=1e-9)


# --- input validation ---------------------------------------------------------

def test_compute_per_bar_loglik_rejects_unfitted_model():
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    X = _make_regime_data(seed=0, n=50)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.compute_per_bar_predictive_loglik(X)


def test_compute_per_bar_loglik_rejects_empty_input(fitted_hmm):
    empty = np.zeros((0, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="must not be empty"):
        fitted_hmm.compute_per_bar_predictive_loglik(empty)


def test_compute_per_bar_loglik_rejects_nan_input(fitted_hmm, sample_data):
    bad = sample_data.copy()
    bad[5, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or Inf"):
        fitted_hmm.compute_per_bar_predictive_loglik(bad)


def test_compute_per_bar_loglik_rejects_inf_input(fitted_hmm, sample_data):
    bad = sample_data.copy()
    bad[3, 1] = np.inf

    with pytest.raises(ValueError, match="NaN or Inf"):
        fitted_hmm.compute_per_bar_predictive_loglik(bad)


# --- input reshape ------------------------------------------------------------

def test_compute_per_bar_loglik_accepts_1d_input():
    """1-D input should be reshaped to (T, 1) by _preprocess_input."""
    rng = np.random.default_rng(0)
    X_1d = rng.normal(size=80)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=0)
    model.fit(X_1d.reshape(-1, 1))

    out = model.compute_per_bar_predictive_loglik(X_1d)

    assert out.shape == (80,)
    assert np.all(np.isfinite(out))
