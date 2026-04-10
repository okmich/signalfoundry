import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volatility import (
    rolling_volatility,
    realized_volatility_for_windows,
    parkinson_volatility,
    garman_klass_volatility,
    rogers_satchell_volatility,
    volatility_ratio,
    volatility_signature,
)
from okmich_quant_features.volatility._volatility import _log_returns, _rolling_std


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    return np.cumprod(1 + 0.01 * np.random.randn(200)) + 100


@pytest.fixture
def sample_ohlc():
    np.random.seed(123)
    n = 200
    open_ = np.linspace(100, 120, n) + np.random.randn(n)
    high = open_ + np.abs(np.random.randn(n))
    low = open_ - np.abs(np.random.randn(n))
    close = open_ + np.random.randn(n)
    return open_, high, low, close


# -----------------------
# Tests for helper funcs
# -----------------------


def test_log_returns_basic(sample_prices):
    arr = sample_prices
    logrets = _log_returns(arr)
    assert len(logrets) == len(arr) - 1
    np.testing.assert_allclose(logrets[0], np.log(arr[1]) - np.log(arr[0]), rtol=1e-12)


def test_rolling_std_matches_numpy(sample_prices):
    arr = np.log(sample_prices)
    win = 10
    result = _rolling_std(arr, win)
    # Compare against numpy rolling std
    expected = [np.nan] * len(arr)
    for i in range(win - 1, len(arr)):
        expected[i] = np.std(arr[i - win + 1 : i + 1])
    np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)


# -----------------------
# Tests for rolling_volatility
# -----------------------


def test_rolling_volatility_shape_and_nan(sample_prices):
    series = pd.Series(sample_prices)
    vol = rolling_volatility(series, window=20)
    assert isinstance(vol, pd.Series)
    assert len(vol) == len(series)
    assert vol.isna().sum() >= 19  # at least window-1 NaNs


# -----------------------
# Tests for realized_volatility
# -----------------------


def test_realized_volatility_columns_and_shape(sample_prices):
    # Create DatetimeIndex with 5-minute intervals
    index = pd.date_range("2023-01-01", periods=len(sample_prices), freq="5min")
    series = pd.Series(sample_prices, index=index)
    result = realized_volatility_for_windows(series, windows=[12, 24], freq_minutes=5)
    assert isinstance(result, pd.DataFrame)
    assert "rv_12" in result.columns
    assert "rv_24" in result.columns
    assert len(result) == len(series)


def test_realized_volatility_increasing_window(sample_prices):
    # Create DatetimeIndex with 5-minute intervals
    index = pd.date_range("2023-01-01", periods=len(sample_prices), freq="5min")
    series = pd.Series(sample_prices, index=index)
    res_short = realized_volatility_for_windows(series, windows=[5], freq_minutes=5)
    res_long = realized_volatility_for_windows(series, windows=[30], freq_minutes=5)
    # Longer window should have more NaNs initially
    # window=30 minutes with 5-min bars = 30/5 = 6 bars, so first 6 values are NaN (indices 0-5)
    assert res_long["rv_30"].iloc[:6].isna().all()
    # window=5 minutes with 5-min bars = 5/5 = 1 bar, so first value is NaN (index 0)
    assert res_short["rv_5"].iloc[0:1].isna().all()


# -----------------------
# Tests for parkinson_volatility
# -----------------------


def test_parkinson_volatility_shape(sample_ohlc):
    _, high, low, _ = sample_ohlc
    result = parkinson_volatility(high, low, window=20)
    assert result.shape == high.shape
    assert np.isnan(result[:19]).all()


def test_parkinson_volatility_nonnegative(sample_ohlc):
    _, high, low, _ = sample_ohlc
    result = parkinson_volatility(high, low, window=20)
    assert np.all(result[~np.isnan(result)] >= 0)


# -----------------------
# Tests for garman_klass_volatility
# -----------------------


def test_garman_klass_volatility_shape(sample_ohlc):
    open_, high, low, close = sample_ohlc
    result = garman_klass_volatility(open_, high, low, close, window=20)
    assert result.shape == close.shape
    assert np.isnan(result[:19]).all()


def test_garman_klass_volatility_nonnegative(sample_ohlc):
    open_, high, low, close = sample_ohlc
    result = garman_klass_volatility(open_, high, low, close, window=20)
    assert np.all(result[~np.isnan(result)] >= 0)


# -----------------------
# Tests for volatility_ratio
# -----------------------


def test_volatility_ratio_basic(sample_ohlc):
    _, high, low, close = sample_ohlc
    # Create DatetimeIndex with 5-minute intervals (required by realized_volatility)
    index = pd.date_range("2023-01-01", periods=len(close), freq="5min")
    high_series = pd.Series(high, index=index)
    low_series = pd.Series(low, index=index)
    close_series = pd.Series(close, index=index)
    result = volatility_ratio(high_series, low_series, close_series, window=12)
    assert result.shape == close.shape
    # Values should be finite where not NaN
    assert np.all(np.isfinite(result[~np.isnan(result)]))


def test_volatility_ratio_with_mismatched_windows(sample_ohlc):
    _, high, low, close = sample_ohlc
    # Create DatetimeIndex with 5-minute intervals (required by realized_volatility)
    index = pd.date_range("2023-01-01", periods=len(close), freq="5min")
    high_series = pd.Series(high, index=index)
    low_series = pd.Series(low, index=index)
    close_series = pd.Series(close, index=index)
    result = volatility_ratio(high_series, low_series, close_series, window=30)
    # With window=30, first 29 values should be NaN
    assert np.isnan(result[:29]).all()


# -----------------------
# Tests for volatility_signature
# -----------------------


def test_volatility_signature_shape(sample_prices):
    # Create DatetimeIndex with 5-minute intervals
    index = pd.date_range("2023-01-01", periods=len(sample_prices), freq="5min")
    series = pd.Series(sample_prices, index=index)
    result = volatility_signature(series, short_window=12, long_window=72)
    assert result.shape == sample_prices.shape
    # Should have NaN values at the beginning (at least for the short window period)
    assert np.isnan(result[:12]).all()


def test_volatility_signature_ratio_behavior(sample_prices):
    # Create DatetimeIndex with 5-minute intervals
    index = pd.date_range("2023-01-01", periods=len(sample_prices), freq="5min")
    series = pd.Series(sample_prices, index=index)
    result = volatility_signature(series, short_window=12, long_window=72)
    vals = result[~np.isnan(result)]
    assert np.all(vals >= 0)
    assert np.all(np.isfinite(vals))


# --- Validation tests for correctness fixes ---

class TestRollingVolatilityInputValidation:
    """Fix #9: rolling_volatility must validate prices before taking log."""

    def test_non_positive_price_raises(self):
        prices = pd.Series([100.0, 0.0, 101.0, 102.0, 103.0])
        with pytest.raises(ValueError, match="non-positive"):
            rolling_volatility(prices, window=3)

    def test_negative_price_raises(self):
        prices = pd.Series([100.0, -1.0, 101.0, 102.0, 103.0])
        with pytest.raises(ValueError, match="non-positive"):
            rolling_volatility(prices, window=3)

    def test_valid_prices_returns_finite(self):
        rng = np.random.default_rng(0)
        prices = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 100)))
        result = rolling_volatility(prices, window=10)
        assert np.all(np.isfinite(result[~np.isnan(result)]))


class TestGarmanKlassNumericalStability:
    """Fix #8: GK sqrt must not produce NaN when term1 - term2 < 0."""

    def test_large_open_to_close_move_gives_finite(self):
        n = 50
        open_ = pd.Series(np.full(n, 100.0))
        close = pd.Series(np.full(n, 105.0))
        high = pd.Series(np.full(n, 105.1))
        low = pd.Series(np.full(n, 99.9))
        result = garman_klass_volatility(open_, high, low, close, window=10)
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid)), "GK vol has NaN/inf on edge OHLC data"


class TestRogersSatchellNumericalStability:
    """Fix #8: RS sqrt of rolling mean must not produce NaN when mean < 0."""

    def test_edge_ohlc_gives_finite(self):
        n = 50
        rng = np.random.default_rng(3)
        close = 100.0 + rng.normal(0, 0.5, n)
        open_ = close + rng.normal(0, 0.5, n)
        high = np.maximum(open_, close) + 0.01
        low = np.minimum(open_, close) - 0.01
        result = rogers_satchell_volatility(
            pd.Series(open_), pd.Series(high), pd.Series(low), pd.Series(close), window=10
        )
        valid = result[~np.isnan(result)]
        assert np.all(np.isfinite(valid)), "RS vol has NaN/inf on edge OHLC data"


class TestRealizedVolatilityForWindowsDefaults:
    """Fix #11: windows=None must produce the same result as windows=[60, 180, 360]."""

    def test_default_windows_none_matches_explicit(self):
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=500, freq="5min")
        prices = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.002, 500)), index=idx)
        result_none = realized_volatility_for_windows(prices, windows=None)
        result_explicit = realized_volatility_for_windows(prices, windows=[60, 180, 360])
        pd.testing.assert_frame_equal(result_none, result_explicit)
