"""Session fixtures over the synthetic ground truth in ``_synthetic``."""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, InputTiming, PomegranateIOHMM, input_likelihood_ratio_test

from ._synthetic import (FIT_KW, WORLD_FIT_KW, Interaction, clock_sequence, homogeneous_sequence,
                         three_state_sequences, two_axis_data)


@pytest.fixture(scope="session")
def clock_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return clock_sequence(30_000, seed=3)


@pytest.fixture(scope="session")
def fitted_clock_model(clock_data) -> PomegranateIOHMM:
    X, U, _ = clock_data
    return PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).fit(X, U=U)


@pytest.fixture(scope="session")
def homogeneous_data() -> np.ndarray:
    return homogeneous_sequence(5_000, seed=1)


@pytest.fixture(scope="session")
def three_state_data():
    return three_state_sequences(seed=5)


@pytest.fixture(scope="session")
def fitted_three_state_model(three_state_data) -> PomegranateIOHMM:
    X, U, _, lengths = three_state_data
    return PomegranateIOHMM(DistType.NORMAL, 3, input_timing=InputTiming.AT_BAR_CLOSE, n_restarts=3,
                            **FIT_KW).fit(X, lengths, U=U)


@pytest.fixture(scope="session")
def worlds() -> dict:
    """
    Per interaction: the sample and the nested likelihood-ratio test of a 3-state path IOHMM whose transitions are
    driven by a volatility HMM's posterior, fitted on the first half (one sequence per day).
    """
    out = {}
    for interaction in Interaction:
        sample, vp, train, day = two_axis_data(interaction)
        lengths = np.bincount(day[train]).tolist()
        result = input_likelihood_ratio_test(sample.X[train], vp[train].reshape(-1, 1), lengths,
                                             distribution_type=DistType.NORMAL, n_states=3,
                                             input_timing=InputTiming.AT_BAR_CLOSE, **WORLD_FIT_KW)
        out[interaction] = (sample, result)
    return out
