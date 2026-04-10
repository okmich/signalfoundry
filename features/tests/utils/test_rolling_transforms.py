"""
Tests for rolling_transforms module.

Covers:
- Both numpy array and pandas Series inputs
- Output type and index/name preservation
- Leading NaN behaviour
- Mathematical correctness against numpy/pandas references
- NaN propagation
- Input validation (window bounds, scale > 0)
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.utils.rolling_transforms import (
    rolling_zscore,
    rolling_percentile_rank,
    rolling_volatility_scale,
    rolling_slope,
    rolling_persistence,
    tanh_compress,
    sigmoid_compress,
)

# =============================================================================
# Fixtures
# =============================================================================

WINDOW = 5
N = 20


@pytest.fixture
def linear_series():
    """Strictly increasing series: 1, 2, ..., N."""
    return pd.Series(np.arange(1.0, N + 1), name="price")


@pytest.fixture
def linear_array():
    return np.arange(1.0, N + 1)


@pytest.fixture
def constant_series():
    return pd.Series(np.ones(N), name="const")


@pytest.fixture
def random_series():
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(N), name="signal")


@pytest.fixture
def random_array():
    rng = np.random.default_rng(42)
    return rng.standard_normal(N)


@pytest.fixture
def series_with_nan(random_series):
    s = random_series.copy()
    s.iloc[7] = np.nan
    return s


@pytest.fixture
def indexed_series():
    """Series with a non-default datetime index."""
    idx = pd.date_range("2024-01-01", periods=N, freq="D")
    rng = np.random.default_rng(0)
    return pd.Series(rng.standard_normal(N), index=idx, name="ret")


# =============================================================================
# Helpers
# =============================================================================

def _leading_nans(result, window):
    """Assert first (window-1) values are NaN and the rest are finite."""
    if isinstance(result, pd.Series):
        vals = result.values
    else:
        vals = result
    assert np.all(np.isnan(vals[: window - 1])), "Leading values should be NaN"
    assert np.all(np.isfinite(vals[window - 1 :])), "Remaining values should be finite"


# =============================================================================
# rolling_zscore
# =============================================================================

class TestRollingZscore:

    def test_returns_series_for_series_input(self, random_series):
        result = rolling_zscore(random_series, WINDOW)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = rolling_zscore(random_array, WINDOW)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = rolling_zscore(indexed_series, WINDOW)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_zscore"

    def test_integer_series_name(self):
        s = pd.Series(np.arange(10.0), name=42)
        result = rolling_zscore(s, 3)
        assert result.name == "42_zscore"

    def test_leading_nans(self, random_series):
        result = rolling_zscore(random_series, WINDOW)
        _leading_nans(result, WINDOW)

    def test_length_preserved(self, random_series):
        result = rolling_zscore(random_series, WINDOW)
        assert len(result) == len(random_series)

    def test_constant_series_returns_zero(self, constant_series):
        result = rolling_zscore(constant_series, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 0.0)

    def test_window_1_returns_zero(self, random_series):
        """window=1: std=0 for every single-element window → all zeros."""
        result = rolling_zscore(random_series, 1)
        np.testing.assert_array_equal(result.values, 0.0)

    def test_matches_pandas_reference(self, random_series):
        result = rolling_zscore(random_series, WINDOW)
        mu = random_series.rolling(WINDOW).mean()
        sigma = random_series.rolling(WINDOW).std(ddof=0)
        expected = ((random_series - mu) / sigma).fillna(0.0)
        # Only compare valid (non-NaN) portion
        valid = slice(WINDOW - 1, None)
        np.testing.assert_allclose(result.values[valid], expected.values[valid], rtol=1e-10)

    def test_nan_propagation(self, series_with_nan):
        result = rolling_zscore(series_with_nan, WINDOW)
        # Windows containing index 7 (NaN) should output NaN
        nan_indices = range(7, 7 + WINDOW)
        for i in nan_indices:
            if i < len(result):
                assert np.isnan(result.iloc[i]), f"Expected NaN at index {i}"

    def test_window_exceeds_length_raises(self, random_series):
        with pytest.raises(ValueError, match="exceeds series length"):
            rolling_zscore(random_series, len(random_series) + 1)

    def test_window_zero_raises(self, random_series):
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_zscore(random_series, 0)

    def test_window_negative_raises(self, random_series):
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_zscore(random_series, -3)

    def test_float_window_coerced(self, random_series):
        result = rolling_zscore(random_series, 5.0)
        assert len(result) == len(random_series)


# =============================================================================
# rolling_percentile_rank
# =============================================================================

class TestRollingPercentileRank:

    def test_returns_series_for_series_input(self, random_series):
        result = rolling_percentile_rank(random_series, WINDOW)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = rolling_percentile_rank(random_array, WINDOW)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = rolling_percentile_rank(indexed_series, WINDOW)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_pct_rank"

    def test_leading_nans(self, random_series):
        result = rolling_percentile_rank(random_series, WINDOW)
        _leading_nans(result, WINDOW)

    def test_values_in_range(self, random_series):
        result = rolling_percentile_rank(random_series, WINDOW)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid < 1.0).all()

    def test_constant_series_returns_zero(self, constant_series):
        result = rolling_percentile_rank(constant_series, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 0.0)

    def test_strictly_increasing_max_rank(self, linear_series):
        """For a strictly increasing series the last bar in each window is
        always the maximum, so rank = (window-1)/window."""
        result = rolling_percentile_rank(linear_series, WINDOW)
        expected_rank = (WINDOW - 1) / WINDOW
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_allclose(valid.values, expected_rank, rtol=1e-12)

    def test_strictly_decreasing_min_rank(self):
        """For a strictly decreasing series the last bar is always the minimum → rank 0."""
        s = pd.Series(np.arange(N, 0.0, -1.0), name="dec")
        result = rolling_percentile_rank(s, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 0.0)

    def test_nan_at_current_bar_propagates(self):
        """NaN at x[i] itself must produce NaN output."""
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        result = rolling_percentile_rank(s, 3)
        assert np.isnan(result.iloc[2])

    def test_nan_in_window_propagates(self, series_with_nan):
        result = rolling_percentile_rank(series_with_nan, WINDOW)
        for i in range(7, 7 + WINDOW):
            if i < len(result):
                assert np.isnan(result.iloc[i])

    def test_window_exceeds_length_raises(self, random_series):
        with pytest.raises(ValueError, match="exceeds series length"):
            rolling_percentile_rank(random_series, N + 1)


# =============================================================================
# rolling_volatility_scale
# =============================================================================

class TestRollingVolatilityScale:

    def test_returns_series_for_series_input(self, random_series):
        result = rolling_volatility_scale(random_series, WINDOW)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = rolling_volatility_scale(random_array, WINDOW)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = rolling_volatility_scale(indexed_series, WINDOW)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_volscale"

    def test_leading_nans(self, random_series):
        result = rolling_volatility_scale(random_series, WINDOW)
        _leading_nans(result, WINDOW)

    def test_constant_series_returns_zero(self, constant_series):
        result = rolling_volatility_scale(constant_series, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 0.0)

    def test_matches_reference(self, random_series):
        result = rolling_volatility_scale(random_series, WINDOW)
        sigma = random_series.rolling(WINDOW).std(ddof=0)
        expected = random_series / sigma
        valid = slice(WINDOW - 1, None)
        np.testing.assert_allclose(result.values[valid], expected.values[valid], rtol=1e-10)

    def test_nan_propagation(self, series_with_nan):
        result = rolling_volatility_scale(series_with_nan, WINDOW)
        for i in range(7, 7 + WINDOW):
            if i < len(result):
                assert np.isnan(result.iloc[i])

    def test_window_exceeds_length_raises(self, random_series):
        with pytest.raises(ValueError, match="exceeds series length"):
            rolling_volatility_scale(random_series, N + 1)

    def test_float_window_coerced(self, random_series):
        result = rolling_volatility_scale(random_series, 5.0)
        assert len(result) == len(random_series)


# =============================================================================
# rolling_slope
# =============================================================================

class TestRollingSlope:

    def test_returns_series_for_series_input(self, random_series):
        result = rolling_slope(random_series, WINDOW)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = rolling_slope(random_array, WINDOW)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = rolling_slope(indexed_series, WINDOW)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_slope"

    def test_leading_nans(self, random_series):
        result = rolling_slope(random_series, WINDOW)
        _leading_nans(result, WINDOW)

    def test_constant_series_slope_zero(self, constant_series):
        result = rolling_slope(constant_series, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-12)

    def test_linear_series_slope_equals_step(self, linear_series):
        """x = [1,2,...,N] has slope exactly 1.0 per bar."""
        result = rolling_slope(linear_series, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_allclose(valid.values, 1.0, rtol=1e-10)

    def test_matches_numpy_polyfit(self, random_series):
        """Verify against numpy.polyfit for each window."""
        result = rolling_slope(random_series, WINDOW)
        t = np.arange(WINDOW, dtype=float)
        for i in range(WINDOW - 1, N):
            w = random_series.values[i - WINDOW + 1: i + 1]
            expected_slope = np.polyfit(t, w, 1)[0]
            np.testing.assert_allclose(result.iloc[i], expected_slope, rtol=1e-8,
                                       err_msg=f"Mismatch at index {i}")

    def test_negative_slope_for_decreasing_series(self):
        s = pd.Series(np.arange(N, 0.0, -1.0), name="dec")
        result = rolling_slope(s, WINDOW)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_allclose(valid.values, -1.0, rtol=1e-10)

    def test_nan_propagation(self, series_with_nan):
        result = rolling_slope(series_with_nan, WINDOW)
        for i in range(7, 7 + WINDOW):
            if i < len(result):
                assert np.isnan(result.iloc[i])

    def test_window_1_raises(self, random_series):
        with pytest.raises(ValueError, match="window must be >= 2"):
            rolling_slope(random_series, 1)

    def test_window_exceeds_length_raises(self, random_series):
        with pytest.raises(ValueError, match="exceeds series length"):
            rolling_slope(random_series, N + 1)


# =============================================================================
# rolling_persistence
# =============================================================================

class TestRollingPersistence:

    def test_returns_series_for_series_input(self, random_series):
        result = rolling_persistence(random_series, WINDOW)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = rolling_persistence(random_array, WINDOW)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = rolling_persistence(indexed_series, WINDOW)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_persistence"

    def test_leading_nans(self, random_series):
        result = rolling_persistence(random_series, WINDOW)
        _leading_nans(result, WINDOW)

    def test_values_in_range(self, random_series):
        result = rolling_persistence(random_series, WINDOW)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_all_above_threshold_returns_one(self):
        s = pd.Series(np.ones(N) * 5.0, name="high")
        result = rolling_persistence(s, WINDOW, threshold=0.0, above=True)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 1.0)

    def test_none_above_threshold_returns_zero(self):
        s = pd.Series(-np.ones(N), name="low")
        result = rolling_persistence(s, WINDOW, threshold=0.0, above=True)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 0.0)

    def test_above_false(self):
        s = pd.Series(-np.ones(N), name="low")
        result = rolling_persistence(s, WINDOW, threshold=0.0, above=False)
        valid = result.iloc[WINDOW - 1 :]
        np.testing.assert_array_equal(valid.values, 1.0)

    def test_known_fraction(self):
        """First 3 of 5-window bars above 0, last 2 below → fraction=3/5."""
        # Build a series where each window of 5 contains exactly 3 positives.
        vals = np.array([1.0, 1.0, 1.0, -1.0, -1.0] * 4, dtype=float)
        s = pd.Series(vals)
        result = rolling_persistence(s, 5, threshold=0.0, above=True)
        valid = result.dropna()
        # All valid values should be either 0.4, 0.6, or 0.2 etc. depending on window.
        # Just verify they are within range and sum of positives / 5 for each window.
        t = np.arange(len(vals))
        for i in range(4, len(vals)):
            w = vals[i - 4: i + 1]
            expected = np.sum(w > 0.0) / 5.0
            np.testing.assert_allclose(result.iloc[i], expected, rtol=1e-12)

    def test_nan_propagation(self, series_with_nan):
        result = rolling_persistence(series_with_nan, WINDOW)
        for i in range(7, 7 + WINDOW):
            if i < len(result):
                assert np.isnan(result.iloc[i])

    def test_window_exceeds_length_raises(self, random_series):
        with pytest.raises(ValueError, match="exceeds series length"):
            rolling_persistence(random_series, N + 1)


# =============================================================================
# tanh_compress
# =============================================================================

class TestTanhCompress:

    def test_returns_series_for_series_input(self, random_series):
        result = tanh_compress(random_series)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = tanh_compress(random_array)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = tanh_compress(indexed_series)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_tanh"

    def test_values_strictly_between_minus1_and_1(self, random_series):
        # Multiply by 5 keeps |x| < ~10, well below float64 tanh saturation (~19)
        result = tanh_compress(random_series * 5)
        assert (result > -1.0).all()
        assert (result < 1.0).all()

    def test_zero_maps_to_zero(self):
        s = pd.Series([0.0])
        result = tanh_compress(s)
        np.testing.assert_allclose(result.values, 0.0, atol=1e-15)

    def test_antisymmetry(self, random_series):
        pos = tanh_compress(random_series)
        neg = tanh_compress(-random_series)
        np.testing.assert_allclose(pos.values, -neg.values, rtol=1e-12)

    def test_large_positive_near_one(self):
        result = tanh_compress(pd.Series([1e6]))
        assert result.iloc[0] > 0.9999

    def test_large_negative_near_minus_one(self):
        result = tanh_compress(pd.Series([-1e6]))
        assert result.iloc[0] < -0.9999

    def test_custom_scale(self, random_series):
        scale = 2.0
        result = tanh_compress(random_series, scale=scale)
        expected = np.tanh(random_series.values / scale)
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)

    def test_nan_input_propagates(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = tanh_compress(s)
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[2])

    def test_scale_zero_raises(self, random_series):
        with pytest.raises(ValueError, match="scale must be > 0"):
            tanh_compress(random_series, scale=0.0)

    def test_scale_negative_raises(self, random_series):
        with pytest.raises(ValueError, match="scale must be > 0"):
            tanh_compress(random_series, scale=-1.0)


# =============================================================================
# sigmoid_compress
# =============================================================================

class TestSigmoidCompress:

    def test_returns_series_for_series_input(self, random_series):
        result = sigmoid_compress(random_series)
        assert isinstance(result, pd.Series)

    def test_returns_array_for_array_input(self, random_array):
        result = sigmoid_compress(random_array)
        assert isinstance(result, np.ndarray)

    def test_preserves_index_and_name(self, indexed_series):
        result = sigmoid_compress(indexed_series)
        pd.testing.assert_index_equal(result.index, indexed_series.index)
        assert result.name == indexed_series.name + "_sigmoid"

    def test_values_strictly_between_0_and_1(self, random_series):
        # Multiply by 5 keeps |x| < ~10, well below float64 sigmoid saturation (~36)
        result = sigmoid_compress(random_series * 5)
        assert (result > 0.0).all()
        assert (result < 1.0).all()

    def test_zero_maps_to_half(self):
        s = pd.Series([0.0])
        result = sigmoid_compress(s)
        np.testing.assert_allclose(result.values, 0.5, atol=1e-15)

    def test_complement_symmetry(self, random_series):
        """sigmoid(x) + sigmoid(-x) == 1 for all x."""
        pos = sigmoid_compress(random_series)
        neg = sigmoid_compress(-random_series)
        np.testing.assert_allclose((pos + neg).values, 1.0, rtol=1e-12)

    def test_large_positive_near_one(self):
        result = sigmoid_compress(pd.Series([1e6]))
        assert result.iloc[0] > 0.9999

    def test_large_negative_near_zero(self):
        result = sigmoid_compress(pd.Series([-1e6]))
        assert result.iloc[0] < 1e-4

    def test_custom_scale(self, random_series):
        scale = 3.0
        result = sigmoid_compress(random_series, scale=scale)
        expected = 1.0 / (1.0 + np.exp(-random_series.values / scale))
        np.testing.assert_allclose(result.values, expected, rtol=1e-12)

    def test_nan_input_propagates(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = sigmoid_compress(s)
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[0])
        assert not np.isnan(result.iloc[2])

    def test_scale_zero_raises(self, random_series):
        with pytest.raises(ValueError, match="scale must be > 0"):
            sigmoid_compress(random_series, scale=0.0)

    def test_scale_negative_raises(self, random_series):
        with pytest.raises(ValueError, match="scale must be > 0"):
            sigmoid_compress(random_series, scale=-0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])