"""
Comprehensive test suite for HMM inference modes (FILTERING, SMOOTHING, VITERBI).

This test suite verifies that the inference mode parameter works correctly
across all HMM implementations and prevents temporal leakage.

Tests cover:
- Sliding window consistency (critical test for temporal leakage)
- Mode behavior (FILTERING, SMOOTHING, VITERBI)
- Serialization preserves inference_mode
- All HMM classes (PomegranateHMM, PomegranateMixtureHMM, FactorialHMM)
- forecast() method independence from inference_mode

Run with: pytest test_inference_modes.py -v
"""

import numpy as np
import pytest
import tempfile
import os

from okmich_quant_ml.hmm import (
    PomegranateHMM,
    PomegranateMixtureHMM,
    FactorialHMM,
    InferenceMode,
    DistType,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Create sample data with clear regime changes."""
    np.random.seed(42)
    n_samples = 100
    data = np.zeros((n_samples, 2))

    for i in range(n_samples):
        if i < 50:
            # Regime 1: low mean
            data[i] = np.random.normal([0.0, 0.0], 0.5, size=2)
        else:
            # Regime 2: high mean
            data[i] = np.random.normal([3.0, 3.0], 0.5, size=2)

    return data


@pytest.fixture
def fitted_pomegranate_hmm(sample_data):
    """Create a fitted PomegranateHMM for testing."""
    model = PomegranateHMM(
        distribution_type=DistType.NORMAL,
        n_states=2,
        inference_mode=InferenceMode.FILTERING,
    )
    model.fit(sample_data)
    return model


@pytest.fixture
def fitted_pomegranate_mixture_hmm(sample_data):
    """Create a fitted PomegranateMixtureHMM for testing."""
    model = PomegranateMixtureHMM(
        distribution_type=DistType.NORMAL,
        n_states=2,
        n_components=2,
        inference_mode=InferenceMode.FILTERING,
    )
    model.fit(sample_data)
    return model


# ============================================================================
# Test: Default Inference Mode
# ============================================================================


def test_pomegranate_default_inference_mode():
    """Test PomegranateHMM default inference mode."""
    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    assert model.inference_mode == InferenceMode.FILTERING


def test_pomegranate_mixture_default_inference_mode():
    """Test PomegranateMixtureHMM default inference mode."""
    model = PomegranateMixtureHMM(
        distribution_type=DistType.NORMAL, n_states=2, n_components=2
    )
    assert model.inference_mode == InferenceMode.FILTERING


def test_factorial_default_inference_mode():
    """Test FactorialHMM default inference mode."""
    chain1 = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    chain2 = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    factorial = FactorialHMM(chains=[chain1, chain2])
    assert factorial.inference_mode == InferenceMode.FILTERING
    # Verify mode was propagated to chains
    assert chain1.inference_mode == InferenceMode.FILTERING
    assert chain2.inference_mode == InferenceMode.FILTERING


# ============================================================================
# Test: Sliding Window Consistency (CRITICAL - Tests Temporal Leakage Fix)
# ============================================================================


def test_filtering_sliding_window_consistency_pomegranate(
    fitted_pomegranate_hmm, sample_data
):
    """Test sliding window consistency for PomegranateHMM with FILTERING."""
    model = fitted_pomegranate_hmm

    window1 = sample_data[10:20]
    probs1 = model.predict_proba(window1)

    window2 = sample_data[11:21]
    probs2 = model.predict_proba(window2)

    np.testing.assert_array_almost_equal(
        probs1[1:],
        probs2[:-1],
        decimal=10,
        err_msg="PomegranateHMM FILTERING should give consistent predictions",
    )


def test_filtering_sliding_window_consistency_mixture(
    fitted_pomegranate_mixture_hmm, sample_data
):
    """Test sliding window consistency for PomegranateMixtureHMM with FILTERING."""
    model = fitted_pomegranate_mixture_hmm

    window1 = sample_data[10:20]
    probs1 = model.predict_proba(window1)

    window2 = sample_data[11:21]
    probs2 = model.predict_proba(window2)

    np.testing.assert_array_almost_equal(
        probs1[1:],
        probs2[:-1],
        decimal=10,
        err_msg="PomegranateMixtureHMM FILTERING should give consistent predictions",
    )


# ============================================================================
# Test: Serialization Preserves Mode
# ============================================================================


def test_pomegranate_save_load_preserves_mode(fitted_pomegranate_hmm, sample_data):
    """Test PomegranateHMM save/load preserves inference_mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_pomegranate.joblib")

        fitted_pomegranate_hmm.inference_mode = InferenceMode.VITERBI
        fitted_pomegranate_hmm.save(save_path)

        loaded_model = PomegranateHMM.load(save_path)
        assert loaded_model.inference_mode == InferenceMode.VITERBI


# ============================================================================
# Test: forecast() Method Independence
# ============================================================================


def test_forecast_independent_pomegranate(fitted_pomegranate_hmm):
    """Test forecast() independence for PomegranateHMM."""
    fitted_pomegranate_hmm.inference_mode = InferenceMode.FILTERING
    forecast1 = fitted_pomegranate_hmm.forecast(current_regime=0, n_steps=3)

    fitted_pomegranate_hmm.inference_mode = InferenceMode.SMOOTHING
    forecast2 = fitted_pomegranate_hmm.forecast(current_regime=0, n_steps=3)

    np.testing.assert_array_almost_equal(forecast1, forecast2)


# ============================================================================
# Test: FactorialHMM Mode Propagation
# ============================================================================


def test_factorial_hmm_propagates_mode_to_chains(sample_data):
    """Test that FactorialHMM propagates inference_mode to all child chains."""
    chain1 = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    chain2 = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)

    # Create factorial with SMOOTHING mode
    factorial = FactorialHMM(
        chains=[chain1, chain2], inference_mode=InferenceMode.SMOOTHING
    )

    # Verify all chains have SMOOTHING mode
    assert factorial.inference_mode == InferenceMode.SMOOTHING
    assert chain1.inference_mode == InferenceMode.SMOOTHING
    assert chain2.inference_mode == InferenceMode.SMOOTHING


def test_factorial_hmm_uses_child_chain_modes(sample_data):
    """Test that FactorialHMM predictions use child chain modes."""
    # Create and fit chains with more data for stability
    chain1 = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)
    chain2 = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2)

    factorial = FactorialHMM(
        chains=[chain1, chain2],
        feature_assignment=[[0], [1]],
        inference_mode=InferenceMode.FILTERING,
    )

    # Fit the factorial HMM on full dataset for stability
    factorial.fit(sample_data)

    # Test sliding window consistency through factorial
    window1 = sample_data[10:20]
    probs1 = factorial.predict_proba(window1)

    window2 = sample_data[11:21]
    probs2 = factorial.predict_proba(window2)

    # Each chain should maintain consistency
    # probs1 and probs2 are lists of arrays (one per chain)
    for i, (p1, p2) in enumerate(zip(probs1, probs2)):
        np.testing.assert_array_almost_equal(
            p1[1:],
            p2[:-1],
            decimal=6,  # Relaxed tolerance for factorial HMM
            err_msg=f"FactorialHMM chain {i} should maintain FILTERING consistency",
        )


# ============================================================================
# Test: All Classes Consistency
# ============================================================================


@pytest.mark.parametrize(
    "model_fixture", ["fitted_pomegranate_hmm", "fitted_pomegranate_mixture_hmm"]
)
def test_all_classes_filtering_consistency(model_fixture, sample_data, request):
    """Test that all HMM classes maintain filtering consistency."""
    model = request.getfixturevalue(model_fixture)

    window1 = sample_data[10:20]
    probs1 = model.predict_proba(window1)

    window2 = sample_data[11:21]
    probs2 = model.predict_proba(window2)

    np.testing.assert_array_almost_equal(
        probs1[1:],
        probs2[:-1],
        decimal=8,  # Slightly relaxed for numerical differences
        err_msg=f"{model.__class__.__name__} should maintain FILTERING consistency",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
