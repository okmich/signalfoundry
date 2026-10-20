"""Causality and timing invariants. Every failure here is silent in aggregate metrics (a leak looks like skill).

Model inputs: column 0 is a clock (KNOWN_AHEAD), column 1 a trailing volatility proxy computed from bar t's own
output (AT_BAR_CLOSE), so it may only drive the transition into t + 1.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm import DistType, InferenceMode, InputTiming, PomegranateIOHMM

from ._synthetic import FIT_KW, clock_sequence

N_BARS = 4_000
PROBE_BARS = (0, 1, 250, 1999, 3998)


def _inputs(X: np.ndarray, U_clock: np.ndarray) -> np.ndarray:
    vol = pd.Series(np.abs(X[:, 0])).rolling(8, min_periods=1).mean().to_numpy()  # uses bar t: known at its close
    return np.column_stack([U_clock[:, 0], vol])


@pytest.fixture(scope="module")
def setup():
    X, U_clock, _ = clock_sequence(N_BARS, seed=11)
    U = _inputs(X, U_clock)
    model = PomegranateIOHMM(DistType.NORMAL, 2, input_timing=[InputTiming.KNOWN_AHEAD, InputTiming.AT_BAR_CLOSE],
                             **FIT_KW).fit(X, U=U)
    return model, X, U


def test_filtered_posterior_is_prefix_consistent(setup):
    model, X, U = setup
    full = model.predict_proba(X, U=U)
    for t in PROBE_BARS:
        np.testing.assert_allclose(model.predict_proba(X[:t + 1], U=U[:t + 1]), full[:t + 1], atol=1e-12)


def test_causal_viterbi_is_prefix_consistent(setup):
    model, X, U = setup
    model = copy.deepcopy(model)
    model.inference_mode = InferenceMode.CAUSAL_VITERBI
    full = model.predict(X, U=U)
    for t in PROBE_BARS:
        np.testing.assert_array_equal(model.predict(X[:t + 1], U=U[:t + 1]), full[:t + 1])


@pytest.mark.parametrize("t", [100, 2500])
def test_future_bars_do_not_move_the_past(setup, t):
    model, X, U = setup
    rng = np.random.default_rng(t)
    X2, U2 = X.copy(), U.copy()
    X2[t + 1:] = rng.normal(size=X2[t + 1:].shape)
    U2[t + 1:] = rng.normal(size=U2[t + 1:].shape)
    np.testing.assert_array_equal(model.predict_proba(X2, U=U2)[:t + 1], model.predict_proba(X, U=U)[:t + 1])
    lag = 5  # a fixed-lag posterior at s sees up to s + lag, so bars <= t - lag must not move
    np.testing.assert_array_equal(model.predict_proba_fixed_lag(X2, lag, U=U2)[:t - lag + 1],
                                  model.predict_proba_fixed_lag(X, lag, U=U)[:t - lag + 1])


def test_at_bar_close_input_drives_only_the_next_transition(setup):
    model, X, U = setup
    t = 1500
    U2 = U.copy()
    U2[t, 1] += 5 * U[:, 1].std()
    base, moved = model.predict_proba(X, U=U), model.predict_proba(X, U=U2)
    np.testing.assert_array_equal(moved[:t + 1], base[:t + 1])  # bar t's own posterior cannot see its close value
    assert np.abs(moved[t + 1] - base[t + 1]).max() > 1e-6  # ...but the transition into t + 1 does
    U3 = U.copy()
    U3[-1, 1] += 5 * U[:, 1].std()  # the last bar's close value governs a transition that has not happened
    np.testing.assert_array_equal(model.predict_proba(X, U=U3), base)


def test_known_ahead_input_drives_its_own_bar(setup):
    model, X, U = setup
    t = 1500
    U2 = U.copy()
    U2[t, 0] += 3.0
    base, moved = model.predict_proba(X, U=U), model.predict_proba(X, U=U2)
    np.testing.assert_array_equal(moved[:t], base[:t])
    assert np.abs(moved[t] - base[t]).max() > 1e-6


def test_shift_stays_inside_each_sequence(setup):
    model, X, U = setup
    lengths = [1500, N_BARS - 1500]
    A = model.transition_matrices(U, lengths)
    assert np.isnan(A[0]).all() and np.isnan(A[1500]).all()  # first bar of each sequence has no transition
    U2 = U.copy()
    U2[1499, 1] += 5 * U[:, 1].std()  # last close value of sequence 1
    A2 = model.transition_matrices(U2, lengths)
    np.testing.assert_array_equal(A2[1500:], A[1500:])  # must not reach sequence 2
