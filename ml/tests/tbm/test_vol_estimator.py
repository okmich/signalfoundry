"""Tests for ml.tbm.vol_estimator — incremental EWMA."""

import math

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.tbm.vol_estimator import EWMAVolatilityEstimator


def test_warm_up_returns_none():
    est = EWMAVolatilityEstimator(span=10, warm_up_bars=5)
    assert est.update(100.0) is None
    for i in range(4):
        assert est.update(100.0 + i) is None
    out = est.update(110.0)
    assert out is not None and out > 0


def test_invalid_span():
    with pytest.raises(ValueError):
        EWMAVolatilityEstimator(span=1)


def test_reset_clears_state():
    est = EWMAVolatilityEstimator(span=5, warm_up_bars=2)
    for p in [100.0, 101.0, 102.0, 101.5]:
        est.update(p)
    assert est.current_vol is not None
    est.reset()
    assert est.current_vol is None
    assert est.update(100.0) is None


def test_matches_batch_ewm_within_tolerance():
    rng = np.random.default_rng(42)
    n = 1000
    log_rets = rng.normal(0, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))

    span = 50
    est = EWMAVolatilityEstimator(span=span, warm_up_bars=span)
    live = []
    for p in prices:
        live.append(est.update(float(p)))

    series = pd.Series(prices)
    log_ret_series = np.log(series / series.shift(1))
    batch = log_ret_series.ewm(span=span, adjust=False).std()

    for i in range(span + 50, n):
        assert live[i] is not None
        assert math.isclose(live[i], batch.iloc[i], rel_tol=0.05, abs_tol=1e-5)
