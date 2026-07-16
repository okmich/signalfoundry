"""Shared fixtures for hierarchical-HMM tests.

Fitting an HHMM runs pomegranate EM and (on first import) numba JIT compilation, so the fitted
models and synthetic streams are built once at session scope and reused across the suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import (
    ALPHABET_SIZE,
    AssetGroup,
    HierarchicalHMM,
    ZigzagDirection,
    ZigzagObservationPipeline,
    get_asset_group_config,
)
from okmich_quant_ml.hmm.hierarchical.config import symbols_for_direction
from okmich_quant_ml.hmm.util import DistType


def pytest_configure(config):
    """Register the ``slow`` marker locally so multi-fit EM tests can be deselected with -m 'not slow'."""
    config.addinivalue_line(
        "markers", "slow: fits one or more EM models inline; deselect with -m 'not slow' for fast unit runs."
    )

# Ground-truth 4-state masked HHMM: block 0 = Run (persistent, strong emissions),
# block 1 = Reversal (less persistent, weak emissions).
TRUE_EDGES = np.array([
    [0.00, 0.85, 0.00, 0.15],   # RunP+  -> RunP- (hold) / RevP- (switch)
    [0.85, 0.00, 0.15, 0.00],   # RunP-  -> RunP+ / RevP+
    [0.00, 0.40, 0.00, 0.60],   # RevP+  -> RunP- / RevP-
    [0.40, 0.00, 0.60, 0.00],   # RevP-  -> RunP+ / RevP+
])


def _emission(is_up: bool, strong: bool) -> np.ndarray:
    p = np.full(ALPHABET_SIZE, 1e-6)
    sub = list(symbols_for_direction(ZigzagDirection.UP if is_up else ZigzagDirection.DOWN))
    for s in (sub[6:9] if strong else sub[0:3]):
        p[s] = 1.0
    return p / p.sum()


TRUE_EMISSIONS = np.stack([
    _emission(True, True),    # 0 RunP+ strong
    _emission(False, True),   # 1 RunP- strong
    _emission(True, False),   # 2 RevP+ weak
    _emission(False, False),  # 3 RevP- weak
])


def _generate_stream(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = 0
    for t in range(1, n):
        states[t] = rng.choice(4, p=TRUE_EDGES[states[t - 1]])
    symbols = np.array([rng.choice(ALPHABET_SIZE, p=TRUE_EMISSIONS[s]) for s in states], dtype=np.int64)
    block = states // 2
    magnitudes = np.where(block == 0, rng.normal(0.004, 0.0008, n), rng.normal(0.001, 0.0003, n)).clip(1e-5)
    return {"states": states, "symbols": symbols, "magnitudes": magnitudes, "block": block}


@pytest.fixture(scope="session")
def synthetic_stream() -> dict:
    """Symbol stream + magnitudes generated from a known HHMM."""
    return _generate_stream(1200, seed=11)


@pytest.fixture(scope="session")
def fitted_hhmm(synthetic_stream) -> HierarchicalHMM:
    """A categorical HHMM fitted once on the synthetic stream (symmetry-breaking init)."""
    model = HierarchicalHMM(DistType.CATEGORICAL, n_init=1, tol=1e-3, random_state=3, max_iter=40)
    model.fit(synthetic_stream["symbols"], magnitudes=synthetic_stream["magnitudes"])
    return model


@pytest.fixture(scope="session")
def price_series() -> pd.Series:
    """Synthetic 1-min intraday close series with regime-switching drift/vol (~3 days)."""
    rng = np.random.default_rng(7)
    n = 3 * 24 * 60
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    ret = np.zeros(n)
    state = 0
    for i in range(n):
        if rng.random() < 0.01:
            state ^= 1
        ret[i] = rng.normal(0.0002, 0.0008) if state == 0 else rng.normal(-0.00005, 0.0018)
    return pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="close")


@pytest.fixture(scope="session")
def fx_config():
    return get_asset_group_config(AssetGroup.FX_MAJORS)


@pytest.fixture(scope="session")
def fitted_pipeline(price_series, fx_config) -> ZigzagObservationPipeline:
    """A BOCPD-flow observation pipeline fitted once on the price series."""
    return ZigzagObservationPipeline(fx_config).fit(price_series)


@pytest.fixture(scope="session")
def observations(fitted_pipeline, price_series):
    return fitted_pipeline.transform(price_series)
