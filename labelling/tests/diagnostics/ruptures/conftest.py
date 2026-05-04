"""Shared fixtures for the ruptures diagnostic labelling tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def mean_shift_signal() -> np.ndarray:
    """Synthetic series with a single clean mean shift at bar 200.

    Pre-shift: N(0, 1) for 200 bars. Post-shift: N(2.5, 1) for 200 bars.
    Generous enough magnitude that PELT under L2 lands at index 200 for any
    sensible penalty.
    """
    rng = np.random.default_rng(seed=20260504)
    pre = rng.standard_normal(200)
    post = rng.standard_normal(200) + 2.5
    return np.concatenate([pre, post])


@pytest.fixture
def two_shift_signal() -> np.ndarray:
    """Two-shift series for multi-segment tests."""
    rng = np.random.default_rng(seed=20260505)
    a = rng.standard_normal(150)
    b = rng.standard_normal(150) + 2.0
    c = rng.standard_normal(150) - 2.0
    return np.concatenate([a, b, c])
