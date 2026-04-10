"""
Tests for microstructure.price_structure module.

Functions:
    intrabar_efficiency_ratio, realized_skewness, realized_kurtosis,
    distance_to_extremes, range_compression_ratio
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    intrabar_efficiency_ratio,
    realized_skewness,
    realized_kurtosis,
    distance_to_extremes,
    range_compression_ratio,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlc(n=80, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.005, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return (
        pd.Series(open_, index=idx, name="open"),
        pd.Series(high, index=idx, name="high"),
        pd.Series(low, index=idx, name="low"),
        pd.Series(close, index=idx, name="close"),
    )


def _make_ohlc_np(n=80, seed=42):
    open_, high, low, close = _make_ohlc(n, seed)
    return open_.values, high.values, low.values, close.values


# ─── TestIntrabarsEfficiencyRatio ─────────────────────────────────────────────

class TestIntrabarsEfficiencyRatio:

    def test_returns_series_for_series_input(self):
        open_, high, low, close = _make_ohlc()
        result = intrabar_efficiency_ratio(open_, high, low, close, window=10)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        open_, high, low, close = _make_ohlc_np()
        result = intrabar_efficiency_ratio(open_, high, low, close, window=10)
        assert isinstance(result, np.ndarray)

    def test_early_values_are_nan(self):
        open_, high, low, close = _make_ohlc(n=50)
        window = 10
        result = intrabar_efficiency_ratio(open_, high, low, close, window=window)
        assert np.all(np.isnan(result.values[:window - 1]))
        assert np.isfinite(result.values[window - 1])

    def test_values_in_zero_one(self):
        """IER must be in [0, 1] since |close-open| ≤ high-low for valid bars."""
        open_, high, low, close = _make_ohlc()
        result = intrabar_efficiency_ratio(open_, high, low, close)
        valid = result.dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_full_trend_bars_gives_one(self):
        """Bars where close == high and open == low → IER = 1."""
        n = 30
        low = np.linspace(100, 110, n)
        high = low + 5.0
        open_ = low.copy()
        close = high.copy()
        result = intrabar_efficiency_ratio(open_, high, low, close, window=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_doji_bars_gives_zero(self):
        """Bars where close == open → net body = 0 → IER = 0."""
        n = 30
        close = np.linspace(100, 110, n)
        open_ = close.copy()
        high = close + 2.0
        low = close - 2.0
        result = intrabar_efficiency_ratio(open_, high, low, close, window=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_name_contains_window(self):
        open_, high, low, close = _make_ohlc()
        result = intrabar_efficiency_ratio(open_, high, low, close, window=15)
        assert result.name == "ier_15"


# ─── TestRealizedSkewness ─────────────────────────────────────────────────────

class TestRealizedSkewness:

    def test_returns_series_for_series_input(self):
        _, _, _, close = _make_ohlc()
        result = realized_skewness(close, window=20)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        _, _, _, close = _make_ohlc_np()
        result = realized_skewness(close, window=20)
        assert isinstance(result, np.ndarray)

    def test_early_values_are_nan(self):
        _, _, _, close = _make_ohlc(n=60)
        window = 20
        result = realized_skewness(close, window=window)
        # pandas rolling skew needs at least 3 obs and returns NaN for first window-1 positions
        assert np.all(np.isnan(result.values[:window - 1]))

    def test_positive_skew_for_right_tailed_returns(self):
        """Returns with a large positive outlier should give positive skewness."""
        n = 40
        # Inject one large positive return at position 10
        returns = np.full(n, 0.0)
        returns[10] = 0.2  # big positive spike
        close = 100.0 * np.cumprod(1 + returns)
        close_s = pd.Series(close)
        result = realized_skewness(close_s, window=30)
        valid = result.dropna()
        # Window covering the spike should show positive skewness
        assert valid.iloc[0] > 0

    def test_negative_skew_for_left_tailed_returns(self):
        """Returns with a large negative outlier should give negative skewness."""
        n = 40
        returns = np.full(n, 0.001)
        returns[10] = -0.2  # big negative spike
        close = 100.0 * np.cumprod(1 + returns)
        close_s = pd.Series(close)
        result = realized_skewness(close_s, window=30)
        valid = result.dropna()
        assert valid.iloc[0] < 0

    def test_name_contains_window(self):
        _, _, _, close = _make_ohlc()
        result = realized_skewness(close, window=25)
        assert result.name == "skew_25"


# ─── TestRealizedKurtosis ─────────────────────────────────────────────────────

class TestRealizedKurtosis:

    def test_returns_series_for_series_input(self):
        _, _, _, close = _make_ohlc()
        result = realized_kurtosis(close, window=20)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        _, _, _, close = _make_ohlc_np()
        result = realized_kurtosis(close, window=20)
        assert isinstance(result, np.ndarray)

    def test_early_values_are_nan(self):
        _, _, _, close = _make_ohlc(n=60)
        window = 20
        result = realized_kurtosis(close, window=window)
        assert np.all(np.isnan(result.values[:window - 1]))

    def test_fat_tails_give_high_kurtosis(self):
        """Returns with extreme outliers should give excess kurtosis > 0."""
        n = 60
        returns = np.full(n, 0.001)
        returns[10] = 0.15   # large positive
        returns[20] = -0.15  # large negative
        close = 100.0 * np.cumprod(1 + returns)
        result = realized_kurtosis(close, window=40)
        valid = result[~np.isnan(result)]
        # At the peak (window covering both spikes), kurtosis > 0
        assert valid.max() > 0

    def test_near_normal_returns_low_kurtosis(self):
        """Many observations of a near-normal distribution → excess kurtosis near 0."""
        rng = np.random.default_rng(99)
        n = 1000
        returns = rng.normal(0, 0.01, n)
        close = pd.Series(100.0 * np.cumprod(1 + returns))
        result = realized_kurtosis(close, window=200)
        valid = result.dropna()
        # For 200-bar windows of normal returns, excess kurtosis should be near 0
        assert abs(valid.mean()) < 2.0  # within 2 units of 0

    def test_name_contains_window(self):
        _, _, _, close = _make_ohlc()
        result = realized_kurtosis(close, window=30)
        assert result.name == "kurt_30"


# ─── TestDistanceToExtremes ───────────────────────────────────────────────────

class TestDistanceToExtremes:

    def test_returns_series_for_series_input(self):
        _, high, low, close = _make_ohlc()
        result = distance_to_extremes(high, low, close, window=10)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        _, high, low, close = _make_ohlc_np()
        result = distance_to_extremes(high, low, close, window=10)
        assert isinstance(result, np.ndarray)

    def test_early_values_are_nan(self):
        _, high, low, close = _make_ohlc(n=50)
        window = 10
        result = distance_to_extremes(high, low, close, window=window)
        assert np.all(np.isnan(result.values[:window - 1]))
        assert np.isfinite(result.values[window - 1])

    def test_close_at_rolling_high_gives_positive(self):
        """When close is consistently at rolling high, DTE should be positive."""
        n = 40
        # Monotonically rising close at H=close+0.1
        close = np.linspace(100, 110, n)
        high = close + 0.1
        low = close - 2.0  # low is far below
        result = distance_to_extremes(high, low, close, window=10)
        valid = result[~np.isnan(result)]
        assert (valid > 0).all()

    def test_close_at_rolling_low_gives_negative(self):
        """When close is consistently at rolling low, DTE should be negative."""
        n = 40
        close = np.linspace(110, 100, n)  # declining
        low = close - 0.1
        high = close + 2.0  # high is far above
        result = distance_to_extremes(high, low, close, window=10)
        valid = result[~np.isnan(result)]
        assert (valid < 0).all()

    def test_close_at_midpoint_gives_near_zero(self):
        """When close is exactly at the rolling mid-range, DTE should be near 0."""
        n = 40
        # Close always in exact center of a fixed range
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)  # = mid = (105+95)/2
        result = distance_to_extremes(high, low, close, window=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-6)

    def test_name_contains_window(self):
        _, high, low, close = _make_ohlc()
        result = distance_to_extremes(high, low, close, window=12)
        assert result.name == "dte_12"


# ─── TestRangeCompressionRatio ────────────────────────────────────────────────

class TestRangeCompressionRatio:

    def test_returns_series_for_series_input(self):
        _, high, low, _ = _make_ohlc()
        result = range_compression_ratio(high, low)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        _, high, low, _ = _make_ohlc_np()
        result = range_compression_ratio(high, low, short_span=5, long_span=20)
        assert isinstance(result, np.ndarray)

    def test_constant_range_gives_one(self):
        """Uniform bars → EMA_short == EMA_long → RCR = 1."""
        n = 60
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        result = range_compression_ratio(high, low, short_span=5, long_span=20)
        # After EMA warm-up, all values should converge to 1
        valid = result[30:]  # skip first 30 to allow EMA convergence
        np.testing.assert_allclose(valid, 1.0, atol=1e-6)

    def test_compression_below_one(self):
        """Narrowing range: short-term EMA < long-term EMA → RCR < 1."""
        n = 80
        # Start wide, finish narrow
        widths = np.linspace(10.0, 1.0, n)
        high = pd.Series(100.0 + widths / 2)
        low = pd.Series(100.0 - widths / 2)
        result = range_compression_ratio(high, low, short_span=5, long_span=20)
        # At the end, short-term EMA of narrow bars < long-term EMA of wide bars
        assert result.iloc[-1] < 1.0

    def test_expansion_above_one(self):
        """Expanding range: short-term EMA > long-term EMA → RCR > 1."""
        n = 80
        # Start narrow, finish wide
        widths = np.linspace(1.0, 10.0, n)
        high = pd.Series(100.0 + widths / 2)
        low = pd.Series(100.0 - widths / 2)
        result = range_compression_ratio(high, low, short_span=5, long_span=20)
        # At the end, short-term EMA of wide bars > long-term EMA of narrow bars
        assert result.iloc[-1] > 1.0

    def test_non_negative_values(self):
        _, high, low, _ = _make_ohlc()
        result = range_compression_ratio(high, low)
        assert (result > 0).all()

    def test_name_contains_spans(self):
        _, high, low, _ = _make_ohlc()
        result = range_compression_ratio(high, low, short_span=3, long_span=12)
        assert result.name == "rcr_3_12"