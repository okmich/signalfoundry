"""The fit recovers a known input-driven hazard; the nested LR test separates a real input from an irrelevant one."""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, input_likelihood_ratio_test

from ._synthetic import FIT_KW, TRUE_B, TRUE_STDS, TRUE_W


def _state_map(model) -> np.ndarray:
    """perm[k] = the fitted state that plays true state k (states differ by variance only)."""
    fitted_var = np.array([float(np.ravel(p["covs"])[0]) for p in model.parameters])
    return np.argsort(fitted_var)[np.argsort(np.argsort(TRUE_STDS ** 2))]


def test_recovers_emissions_baseline_and_input_weights(fitted_clock_model):
    model = fitted_clock_model
    perm = _state_map(model)
    fitted_std = np.sqrt([float(np.ravel(p["covs"])[0]) for p in model.parameters])
    np.testing.assert_allclose(fitted_std[perm], TRUE_STDS, rtol=0.03)
    b, W = model.baseline_logits(raw=True), model.input_coefficients(raw=True)
    for i, j in [(0, 1), (1, 0)]:
        assert b[perm[i], perm[j]] == pytest.approx(TRUE_B[i, j], abs=0.25)
        assert W[perm[i], perm[j], 0] == pytest.approx(TRUE_W[i, j, 0], abs=0.25)


def test_input_beats_the_homogeneous_model(clock_data):
    X, U, _ = clock_data
    result = input_likelihood_ratio_test(X, U, distribution_type=DistType.NORMAL, n_states=2, **FIT_KW)
    assert result.df == 2
    assert result.full_log_likelihood >= result.null_log_likelihood
    assert result.p_value < 1e-20


def test_irrelevant_input_is_not_flagged(clock_data):
    X, _, _ = clock_data
    noise = np.random.default_rng(9).normal(size=(len(X), 1))
    result = input_likelihood_ratio_test(X, noise, distribution_type=DistType.NORMAL, n_states=2, **FIT_KW)
    assert result.full_log_likelihood >= result.null_log_likelihood
    assert result.p_value > 0.01


def test_filtered_states_track_the_truth(fitted_clock_model, clock_data):
    X, U, states = clock_data
    perm = _state_map(fitted_clock_model)
    decoded = np.argsort(perm)[fitted_clock_model.predict(X, U=U)]  # fitted label -> true label
    assert np.mean(decoded == states) > 0.85
