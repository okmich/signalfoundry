"""
Tests for fixed-lag smoothing on BasePomegranateHMM (streaming/open-end semantics).

The fixed-lag smoother uses open-end semantics: backward frontier is always
initialized to uniform (log_beta = 0). This matches live trading where there
is no known sequence terminal point. It is intentionally NOT equivalent to
pomegranate's full smoothing at large lags.

Verifies:
- lag=0 matches FILTERING mode (multiple seeds)
- Backward isolation: posterior at t only uses observations up to t+lag
- Frozen output: posterior at t is identical whether computed from X[0:t+L+1] or longer
- Monotonic information gain: more lag -> posterior moves away from filtering
- Open-end semantics: no terminal conditioning (large lag != pomegranate smoothing)
- Both PomegranateHMM and PomegranateMixtureHMM
- Sweep API matches individual calls
- Edge cases: negative lag, HSMM, unfitted model, boolean lag, small sequence

Run with: pytest test_fixed_lag_smoothing.py -v
"""

import numpy as np
import pytest

from okmich_quant_ml.hmm import PomegranateHMM, PomegranateMixtureHMM, DistType, InferenceMode


# ============================================================================
# Fixtures
# ============================================================================

def _make_regime_data(seed, n=100, d=2, n_regimes=2):
    """Data with clear regime changes, parameterised by seed."""
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
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, inference_mode=InferenceMode.FILTERING)
    model.fit(sample_data)
    return model


@pytest.fixture
def fitted_mixture_hmm(sample_data):
    model = PomegranateMixtureHMM(distribution_type=DistType.NORMAL, n_states=2, n_components=2, inference_mode=InferenceMode.FILTERING)
    model.fit(sample_data)
    return model


# ============================================================================
# Test: lag=0 matches FILTERING (multiple seeds)
# ============================================================================

@pytest.mark.parametrize("seed", [0, 7, 13, 42, 99])
def test_lag_zero_matches_filtering_multi_seed(seed):
    """Fixed-lag with lag=0 must match FILTERING across diverse random data."""
    X = _make_regime_data(seed, n=80, d=3)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=seed)
    model.fit(X)

    fixed_lag_proba = model.predict_proba_fixed_lag(X, lag=0)
    model.inference_mode = InferenceMode.FILTERING
    filtering_proba = model.predict_proba(X)

    np.testing.assert_array_almost_equal(fixed_lag_proba, filtering_proba, decimal=6)


def test_lag_zero_matches_filtering_mixture(fitted_mixture_hmm, sample_data):
    fixed_lag_proba = fitted_mixture_hmm.predict_proba_fixed_lag(sample_data, lag=0)
    fitted_mixture_hmm.inference_mode = InferenceMode.FILTERING
    filtering_proba = fitted_mixture_hmm.predict_proba(sample_data)
    np.testing.assert_array_almost_equal(fixed_lag_proba, filtering_proba, decimal=6)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_lag_zero_matches_filtering_3state(seed):
    X = _make_regime_data(seed, n=90, d=2, n_regimes=3)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=3, random_state=seed)
    model.fit(X)

    fixed_lag_proba = model.predict_proba_fixed_lag(X, lag=0)
    model.inference_mode = InferenceMode.FILTERING
    filtering_proba = model.predict_proba(X)

    np.testing.assert_array_almost_equal(fixed_lag_proba, filtering_proba, decimal=6)


# ============================================================================
# Test: Backward isolation (causality)
# ============================================================================

def test_backward_isolation(fitted_hmm, sample_data):
    """Posteriors within lag boundary must not change when future data is appended."""
    lag = 3
    proba_original = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=lag)

    rng = np.random.default_rng(99)
    extra = rng.normal(10.0, 5.0, size=(20, 2))
    extended = np.vstack([sample_data, extra])
    proba_extended = fitted_hmm.predict_proba_fixed_lag(extended, lag=lag)

    T = sample_data.shape[0]
    safe_end = T - lag
    np.testing.assert_array_almost_equal(
        proba_original[:safe_end], proba_extended[:safe_end], decimal=10,
        err_msg="Posteriors within lag boundary must not change when future data is appended",
    )


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_backward_isolation_multi_seed(seed):
    """Backward isolation across multiple seeds to catch edge cases."""
    X = _make_regime_data(seed, n=80, d=2)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=seed)
    model.fit(X)

    lag = 5
    proba_original = model.predict_proba_fixed_lag(X, lag=lag)

    rng = np.random.default_rng(seed + 100)
    extra = rng.normal(0.0, 3.0, size=(30, 2))
    extended = np.vstack([X, extra])
    proba_extended = model.predict_proba_fixed_lag(extended, lag=lag)

    safe_end = len(X) - lag
    np.testing.assert_array_almost_equal(
        proba_original[:safe_end], proba_extended[:safe_end], decimal=10,
    )


# ============================================================================
# Test: Frozen output — posterior at t is fixed once t+L observations arrive
# ============================================================================

def test_frozen_output(fitted_hmm, sample_data):
    """Posterior at t computed from X[0:t+L+1] must equal posterior at t from full X.

    This is the streaming contract: once bar t+L arrives, the belief about t is final.
    """
    lag = 4
    T = sample_data.shape[0]
    full_proba = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=lag)

    # For several timesteps, compute posterior from truncated sequence
    for t in [0, 5, 10, 20, 40]:
        if t + lag >= T:
            continue
        truncated = sample_data[: t + lag + 1]
        trunc_proba = fitted_hmm.predict_proba_fixed_lag(truncated, lag=lag)
        np.testing.assert_array_almost_equal(
            full_proba[t], trunc_proba[t], decimal=10,
            err_msg=f"Posterior at t={t} should be frozen once t+lag={t+lag} is observed",
        )


# ============================================================================
# Test: Open-end semantics — large lag is NOT pomegranate smoothing
# ============================================================================

def test_open_end_no_terminal_conditioning():
    """On a very short sequence, open-end and pomegranate smoothing must diverge.

    With T=3 and lag=T, the backward initialization is only 2 steps from the
    first timestep, so the end-probability initialization is not washed out.
    """
    # Train on a longer sequence, then test on a very short one
    X_train = _make_regime_data(seed=42, n=200, d=2)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=42)
    model.fit(X_train)

    # Check that end probs are non-uniform
    ends = model._model.ends.detach().cpu().numpy()
    end_probs = np.exp(ends)
    if np.allclose(end_probs, end_probs.mean(), atol=1e-3):
        pytest.skip("End probs happen to be uniform; test not applicable")

    # Very short test sequence — backward init is not washed out
    X_short = X_train[50:53]  # T=3
    fixed_lag_proba = model.predict_proba_fixed_lag(X_short, lag=10)

    model.inference_mode = InferenceMode.SMOOTHING
    smoothing_proba = model.predict_proba(X_short)

    max_diff = np.max(np.abs(fixed_lag_proba - smoothing_proba))
    assert max_diff > 1e-6, (
        f"On T=3 with non-uniform end probs {end_probs}, open-end and smoothing "
        f"should differ but max_diff={max_diff}"
    )


# ============================================================================
# Test: Nonzero lag differs from filtering
# ============================================================================

def test_nonzero_lag_differs_from_filtering(fitted_hmm, sample_data):
    """Nonzero lag should generally produce posteriors that differ from filtering (lag=0).

    Note: strict monotonic increase in distance is NOT guaranteed — the backward
    pass is nonlinear and aggregate distance can decrease at intermediate lags.
    With high-certainty regimes, the backward pass may change nothing at all.
    We test a weak structural property: large lag should not produce less change
    than small lag by a large margin.
    """
    proba_0 = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=0)

    # Compute distances at several lags
    distances = {}
    for lag in [1, 5, 10]:
        proba_l = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=lag)
        distances[lag] = np.mean(np.abs(proba_l - proba_0))

    # If the model is extremely confident (saturated posteriors), the backward
    # pass may not change anything — skip in that case
    max_confidence = np.max(proba_0, axis=1).mean()
    if max_confidence > 0.9999:
        pytest.skip(f"Model posteriors are near-saturated (mean max confidence={max_confidence:.6f})")

    # At least some lag should produce a measurable difference
    assert any(d > 1e-8 for d in distances.values()), (
        f"No lag produced any difference from filtering: {distances}"
    )

    # Weak ordering: large lag should not be drastically less than small lag
    assert distances[10] > distances[1] - 0.01, (
        f"lag=10 distance ({distances[10]:.6f}) should not be much less than lag=1 ({distances[1]:.6f})"
    )


# ============================================================================
# Test: Diagnostics output
# ============================================================================

def test_diagnostics_shapes_and_invariants(fitted_hmm, sample_data):
    lag = 3
    diag = fitted_hmm.fixed_lag_diagnostics(sample_data, lag=lag)

    T = sample_data.shape[0]
    K = fitted_hmm.n_states

    assert diag["posteriors"].shape == (T, K)
    assert diag["map_labels"].shape == (T,)
    assert diag["max_posterior"].shape == (T,)
    assert diag["entropy"].shape == (T,)
    assert diag["posterior_delta"].shape == (T, K)

    row_sums = diag["posteriors"].sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(T), decimal=10)

    assert np.all(diag["max_posterior"] >= 1.0 / K - 1e-10)
    assert np.all(diag["max_posterior"] <= 1.0 + 1e-10)
    assert np.all(diag["entropy"] >= -1e-10)
    assert np.all(diag["map_labels"] >= 0)
    assert np.all(diag["map_labels"] < K)
    np.testing.assert_array_almost_equal(diag["posterior_delta"][0], np.zeros(K), decimal=10)


# ============================================================================
# Test: predict_fixed_lag matches argmax
# ============================================================================

def test_predict_fixed_lag_matches_argmax(fitted_hmm, sample_data):
    lag = 5
    labels = fitted_hmm.predict_fixed_lag(sample_data, lag=lag)
    proba = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=lag)
    np.testing.assert_array_equal(labels, np.argmax(proba, axis=1))


# ============================================================================
# Test: Sweep API
# ============================================================================

def test_sweep_matches_individual_calls(fitted_hmm, sample_data):
    """Sweep must produce identical results to individual calls."""
    lags_to_test = [0, 1, 3, 5, sample_data.shape[0]]
    sweep = fitted_hmm.predict_proba_fixed_lag_sweep(sample_data, lags_to_test)

    for lag in lags_to_test:
        individual = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=lag)
        np.testing.assert_array_almost_equal(sweep[lag], individual, decimal=10,
                                             err_msg=f"Sweep and individual differ at lag={lag}")


# ============================================================================
# Test: Fast-path performance (lag=0 should not run custom FB)
# ============================================================================

def test_lag_zero_fast_path(sample_data):
    """lag=0 delegates to filtering, not custom forward-backward."""
    import time
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    model.fit(sample_data)

    model.inference_mode = InferenceMode.FILTERING
    t0 = time.perf_counter()
    for _ in range(10):
        model.predict_proba(sample_data)
    filtering_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10):
        model.predict_proba_fixed_lag(sample_data, lag=0)
    lag0_time = time.perf_counter() - t0

    assert lag0_time < filtering_time * 3 + 0.1, (
        f"lag=0 took {lag0_time:.3f}s vs filtering {filtering_time:.3f}s — fast path not working"
    )


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_negative_lag_raises(fitted_hmm, sample_data):
    with pytest.raises(ValueError, match="lag must be >= 0"):
        fitted_hmm.predict_proba_fixed_lag(sample_data, lag=-1)


def test_unfitted_model_raises():
    """Unfitted model must raise RuntimeError, not AttributeError, for all lag values."""
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_proba_fixed_lag(X, lag=0)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_proba_fixed_lag(X, lag=3)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_proba_fixed_lag_sweep(X, lags=[0, 1, 5])


def test_boolean_lag_raises(fitted_hmm, sample_data):
    with pytest.raises(ValueError, match="integer.*bool"):
        fitted_hmm.predict_proba_fixed_lag(sample_data, lag=True)
    with pytest.raises(ValueError, match="integer.*bool"):
        fitted_hmm.predict_proba_fixed_lag_sweep(sample_data, lags=[False])


def test_hsmm_raises_not_implemented(sample_data):
    from okmich_quant_ml.hmm.duration import PoissonDuration
    dur = PoissonDuration(n_states=2, max_duration=50)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, duration_model=dur)
    model.fit(sample_data)
    with pytest.raises(NotImplementedError, match="HSMM"):
        model.predict_proba_fixed_lag(sample_data, lag=3)




def test_small_sequence():
    """Small T: lag larger than T-1 runs open-end backward over entire sequence."""
    X = _make_regime_data(seed=42, n=5, d=2)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    model.fit(_make_regime_data(seed=0, n=50, d=2))

    proba_0 = model.predict_proba_fixed_lag(X, lag=0)
    proba_large = model.predict_proba_fixed_lag(X, lag=100)
    assert proba_0.shape == (5, 2)
    assert proba_large.shape == (5, 2)
    # Row sums must be 1
    np.testing.assert_array_almost_equal(proba_large.sum(axis=1), np.ones(5), decimal=10)


# ============================================================================
# Test: Truncated-buffer oracle (Fix 1 — proper cross-validation)
# ============================================================================

@pytest.mark.parametrize("seed", [0, 42, 99])
def test_truncated_buffer_oracle(seed):
    """Cross-validate: predict_proba_fixed_lag on truncated buffers must match
    the full-sequence call at each timestep.

    This is the streaming contract oracle — it verifies that the posterior at t
    is determined solely by observations [0..t+lag] and not by anything beyond.
    Replaces the role HMMOnlineSmoother was incorrectly serving (terminal-conditioned
    smoothing is a different quantity).
    """
    X = _make_regime_data(seed, n=60, d=2)
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=seed)
    model.fit(X)

    lag = 4
    full_proba = model.predict_proba_fixed_lag(X, lag=lag)

    # Verify every timestep where the full lag window is available
    for t in range(0, len(X) - lag):
        truncated = X[: t + lag + 1]
        trunc_proba = model.predict_proba_fixed_lag(truncated, lag=lag)
        np.testing.assert_array_almost_equal(
            full_proba[t], trunc_proba[t], decimal=10,
            err_msg=f"Oracle mismatch at t={t}, seed={seed}",
        )


# ============================================================================
# Test: Input validation (Fix 4)
# ============================================================================

def test_non_integer_lag_raises(fitted_hmm, sample_data):
    with pytest.raises(ValueError, match="integer"):
        fitted_hmm.predict_proba_fixed_lag(sample_data, lag=2.5)


def test_float_lag_raises_in_sweep(fitted_hmm, sample_data):
    with pytest.raises(ValueError, match="integer"):
        fitted_hmm.predict_proba_fixed_lag_sweep(sample_data, lags=[0, 1.5])


def test_empty_array_raises(fitted_hmm):
    with pytest.raises(ValueError, match="empty"):
        fitted_hmm.predict_proba_fixed_lag(np.array([]).reshape(0, 2), lag=1)


def test_empty_array_raises_in_sweep(fitted_hmm):
    with pytest.raises(ValueError, match="empty"):
        fitted_hmm.predict_proba_fixed_lag_sweep(np.array([]).reshape(0, 2), lags=[1])


def test_nan_input_raises(fitted_hmm, sample_data):
    X_bad = sample_data.copy()
    X_bad[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        fitted_hmm.predict_proba_fixed_lag(X_bad, lag=3)


def test_inf_input_raises(fitted_hmm, sample_data):
    X_bad = sample_data.copy()
    X_bad[5, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        fitted_hmm.predict_proba_fixed_lag_sweep(X_bad, lags=[0, 2])


def test_nan_input_raises_on_fit():
    """NaN in training data should raise ValueError before EM, not opaque errors."""
    X = np.random.default_rng(0).normal(size=(100, 2))
    X[50, 0] = np.nan
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.fit(X)


def test_inf_input_raises_on_fit():
    """Inf in training data should raise ValueError before EM, not opaque errors."""
    X = np.random.default_rng(0).normal(size=(100, 2))
    X[25, 1] = np.inf
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, random_state=0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        model.fit(X)


# ============================================================================
# Test: Sweep dedup (Fix 6)
# ============================================================================

def test_sweep_dedup_handles_duplicate_lags(fitted_hmm, sample_data):
    """Duplicate lags should not cause errors and should produce correct results."""
    sweep = fitted_hmm.predict_proba_fixed_lag_sweep(sample_data, lags=[0, 3, 3, 5, 5, 5])
    individual_3 = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=3)
    individual_5 = fitted_hmm.predict_proba_fixed_lag(sample_data, lag=5)
    np.testing.assert_array_almost_equal(sweep[3], individual_3, decimal=10)
    np.testing.assert_array_almost_equal(sweep[5], individual_5, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
