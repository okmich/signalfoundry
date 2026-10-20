"""The inputs a cross-axis study actually uses: persistent features known at bar close, several at once, 3 states and
several emission features, and another HMM's filtered posterior.
"""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, InputTiming, PomegranateIOHMM, input_likelihood_ratio_test

from ._synthetic import FIT_KW, K3_B, K3_MEANS, K3_STD, K3_W, Interaction, match_states


def test_recovers_three_states_two_inputs_at_bar_close(fitted_three_state_model):
    model = fitted_three_state_model
    perm = match_states(np.array([p["means"] for p in model.parameters]), K3_MEANS)
    np.testing.assert_allclose(np.array([p["means"] for p in model.parameters])[perm], K3_MEANS, atol=0.05)
    np.testing.assert_allclose(np.sqrt(np.array([p["covs"] for p in model.parameters]))[perm], K3_STD, rtol=0.03)
    b, W = model.baseline_logits(raw=True), model.input_coefficients(raw=True)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert b[perm[i], perm[j]] == pytest.approx(K3_B[i, j], abs=0.25)
                assert W[perm[i], perm[j], 0] == pytest.approx(K3_W[i, j, 0], abs=0.25)  # the driving input
                assert abs(W[perm[i], perm[j], 1]) < 0.25  # the irrelevant one


def test_the_bar_close_shift_is_part_of_the_fit(three_state_data, fitted_three_state_model):
    """Declaring the same raw inputs KNOWN_AHEAD uses each bar's own value one transition too early: a worse fit."""
    X, U, _, lengths = three_state_data
    early = PomegranateIOHMM(DistType.NORMAL, 3, input_timing=InputTiming.KNOWN_AHEAD,
                             **FIT_KW).fit(X, lengths, U=U, init_model=fitted_three_state_model)
    right = fitted_three_state_model.selection_scores(X, lengths, U=U, with_entropy=False).log_likelihood
    wrong = early.selection_scores(X, lengths, U=U, with_entropy=False).log_likelihood
    assert right - wrong > 10.0


def test_lr_test_with_several_inputs(three_state_data):
    X, U, _, lengths = three_state_data
    both = input_likelihood_ratio_test(X, U, lengths, distribution_type=DistType.NORMAL, n_states=3,
                                       input_timing=InputTiming.AT_BAR_CLOSE, **FIT_KW)
    assert both.df == 3 * 2 * 2
    assert both.p_value < 1e-20
    irrelevant = input_likelihood_ratio_test(X, U[:, 1:], lengths, distribution_type=DistType.NORMAL, n_states=3,
                                             input_timing=InputTiming.AT_BAR_CLOSE, **FIT_KW)
    assert irrelevant.df == 6
    assert irrelevant.p_value > 0.01


def test_another_hmms_posterior_as_input(worlds):
    """A volatility HMM's filtered posterior drives a path-structure IOHMM: in the timing world it raises the hazard of
    every move, which is the true interaction."""
    _, result = worlds[Interaction.TIMING]
    assert result.p_value < 1e-10
    W = result.full_model.input_coefficients()
    assert np.all(W[~np.eye(3, dtype=bool)] > 0)
