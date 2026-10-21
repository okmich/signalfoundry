"""The interaction readout (``input_response``) recovers the true interaction.

- Against a true IOHMM: the dwell curves match the truth's.
- In the two-axis worlds, with a volatility HMM's posterior as the input, the readout tells the three apart:
  timing (every regime ends sooner in high volatility), level (the input points at one regime: moves into it rise,
  moves out of it fall), independent (no material response).
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax

from ._synthetic import K3_B, K3_MEANS, K3_W, Interaction, TwoAxisWorld, match_states


def _dwell_by_true_state(model, values, true_means) -> np.ndarray:
    """(len(values), K) expected dwell in bars, columns in TRUE state order."""
    response = model.input_response(0, values)
    dwell = response.pivot(index="value", columns="state", values="expected_dwell").to_numpy()
    perm = match_states(np.array([p["means"] for p in model.parameters]), true_means)
    return dwell[:, perm]


def test_dwell_curves_match_a_true_iohmm(fitted_three_state_model, three_state_data):
    model = fitted_three_state_model
    U = three_state_data[1]
    values = np.array([-1.5, -0.5, 0.5, 1.5])
    fitted = _dwell_by_true_state(model, values, K3_MEANS)
    other = U[:, 1].mean()  # the response holds input 1 at its training mean
    true_stay = np.array([[softmax(K3_B[i] + K3_W[i] @ np.array([v, other]))[i] for i in range(3)] for v in values])
    np.testing.assert_allclose(fitted, 1.0 / (1.0 - true_stay), rtol=0.2)


def test_readout_shows_timing(worlds):
    _, result = worlds[Interaction.TIMING]
    world = TwoAxisWorld()
    dwell = _dwell_by_true_state(result.full_model, [0.0, 1.0], world.path_means)
    leave = (world.n_path_states - 1) * np.exp(world.path_logits()[0, 1] + np.array([0.0, world.timing_log_odds]))
    true_dwell = (1 + leave) / leave  # calm, high: the same for every regime
    assert np.all(dwell[1] < dwell[0] / 2)  # every regime ends sooner in high volatility
    for row, truth in zip(dwell, true_dwell):
        assert np.all((row > truth / 1.5) & (row < truth * 1.5))


def test_readout_shows_level(worlds):
    _, result = worlds[Interaction.LEVEL]
    world = TwoAxisWorld()
    dwell = _dwell_by_true_state(result.full_model, [0.0, 1.0], world.path_means)
    loud = int(np.argmax(world.level_sigma))  # the path regime that carries high volatility
    quiet = [k for k in range(world.n_path_states) if k != loud]
    assert dwell[1, loud] > 2 * dwell[0, loud]  # high volatility keeps you in the loud regime ...
    assert np.all(dwell[1, quiet] < dwell[0, quiet] / 2)  # ... and pushes you out of the quiet ones


def test_readout_flat_when_independent(worlds):
    _, result = worlds[Interaction.INDEPENDENT]
    dwell = _dwell_by_true_state(result.full_model, [0.0, 1.0], TwoAxisWorld().path_means)
    ratio = dwell[1] / dwell[0]
    assert np.all((ratio > 2 / 3) & (ratio < 1.5))
    assert result.p_value > 0.01  # and the LR test does not flag the input
