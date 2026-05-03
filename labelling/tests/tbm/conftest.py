"""Shared fixtures for tbm tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def gbm_close():
    rng = np.random.default_rng(42)
    n = 1000
    log_rets = rng.normal(0, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    index = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.Series(prices, index=index, name="close")


@pytest.fixture
def gbm_ohlc():
    rng = np.random.default_rng(7)
    n = 500
    log_rets = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, size=n))
    index = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)
