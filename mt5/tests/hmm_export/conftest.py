import numpy as np
import pytest

from okmich_quant_mt5.hmm_export import FeatureSpec, Mql5Feature, StateOrder, build_params

SPEC = FeatureSpec(names=(Mql5Feature.MACD_ATR, Mql5Feature.ATR_CLOSE), macd_fast=18, macd_slow=40, macd_signal=11,
                   atr_period=14)

#: Persistent chain - the regime-switching case the indicators are built for.
TRANSITIONS = np.array([[0.94, 0.04, 0.02], [0.05, 0.90, 0.05], [0.03, 0.07, 0.90]])
STARTS = np.array([0.5, 0.3, 0.2])
MEANS = np.array([[-1.30, 0.00090], [0.05, 0.00045], [1.25, 0.00120]])
COVARIANCES = np.array([
    [[0.90, 1.0e-5], [1.0e-5, 4.0e-8]],
    [[0.35, 2.0e-6], [2.0e-6, 1.0e-8]],
    [[1.10, 1.5e-5], [1.5e-5, 6.0e-8]],
])


@pytest.fixture
def spec():
    return SPEC


@pytest.fixture
def params():
    """A well-conditioned 3-state / 2-feature Gaussian model.

    ``StateOrder.NONE`` keeps the hand-written state indices intact so tests can
    reason about which state is which.
    """
    return build_params(model_id="test_model", spec=SPEC, starts=STARTS, edges=TRANSITIONS, means=MEANS,
                        covariances=COVARIANCES, state_order=StateOrder.NONE)


@pytest.fixture
def observations():
    """400 draws from the model's own emissions, walked through the chain."""
    rng = np.random.default_rng(2024)
    n = 400
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(3, p=TRANSITIONS[states[t - 1]])

    return np.stack([rng.multivariate_normal(MEANS[s], COVARIANCES[s]) for s in states])


@pytest.fixture
def bars():
    """Synthetic OHLC long enough to clear the 49-bar feature warm-up."""
    rng = np.random.default_rng(99)
    close = 1.1000 + np.cumsum(rng.normal(0.0, 0.0008, 600))
    half_range = np.abs(rng.normal(0.0, 0.0004, 600)) + 0.0002
    return close + half_range, close - half_range, close
