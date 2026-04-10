from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.fractional_diff import (
    get_weights,
    _auto_determine_window_size,
    fractional_differentiate_series,
    get_optimal_fractional_differentiation_order,
    FractionalDifferentiator,
)


class TestGetWeights:
    """Test suite for get_weights function"""

    def test_get_weights_basic(self):
        """Test basic weight calculation"""
        weights = get_weights(0.5, 5)
        assert isinstance(weights, np.ndarray)
        assert len(weights) == 5
        assert weights[0] == 1.0  # First weight should always be 1

    def test_get_weights_d_zero(self):
        """Test weights for d=0 (should be [1, 0, 0, ...])"""
        weights = get_weights(0.0, 5)
        expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_get_weights_d_one(self):
        """Test weights for d=1 (should be [1, -1, 0, 0, ...])"""
        weights = get_weights(1.0, 5)
        expected = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(weights, expected)

    def test_get_weights_negative_d(self):
        """Test weights for negative d"""
        weights = get_weights(-0.5, 4)
        # Verify the binomial expansion formula holds
        assert weights[0] == 1.0
        assert weights[1] == 0.5  # -(-0.5)/1 = 0.5
        assert weights[2] == 0.375  # -0.5 * (0.5 - 1)/2 = 0.375

    def test_get_weights_large_window(self):
        """Test weights with large window size"""
        weights = get_weights(0.7, 100)
        assert len(weights) == 100
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))


class TestAutoDetermineWindowSize:
    """Test suite for _auto_determine_window_size function"""

    def test_auto_determine_window_size_convergence(self):
        """Test window size determination for convergence"""
        window_size = _auto_determine_window_size(0.5, 1.e-4)
        assert isinstance(window_size, int)
        assert window_size > 0
        assert window_size <= 1000

    def test_auto_determine_window_size_d_zero(self):
        """Test window size for d=0 (should converge quickly)"""
        window_size = _auto_determine_window_size(0.0, 1e-5)
        assert window_size <= 150  # Should converge very quickly

    def test_auto_determine_window_size_d_one(self):
        """Test window size for d=1 (should converge quickly)"""
        window_size = _auto_determine_window_size(1.0, 1e-5)
        assert window_size <= 150

    def test_auto_determine_window_size_custom_threshold(self):
        """Test with custom threshold"""
        window_size_tight = _auto_determine_window_size(0.5, 1e-7)
        window_size_loose = _auto_determine_window_size(0.5, 1e-3)
        assert window_size_tight >= window_size_loose


class TestFractionalDifferentiateSeries:
    """Test suite for fractional_differentiate_series function"""

    @pytest.fixture
    def sample_series(self):
        """Create sample time series data"""
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(100)) + 100)

    def test_fractional_differentiate_series_basic(self, sample_series):
        """Test basic fractional differentiation"""
        result, window_size = fractional_differentiate_series(sample_series, d=0.5)
        assert isinstance(result, np.ndarray)
        assert isinstance(window_size, int)
        assert len(result) == len(sample_series)
        assert window_size > 0

    def test_fractional_differentiate_series_with_window(self, sample_series):
        """Test with explicit window size"""
        result, window_size = fractional_differentiate_series(
            sample_series, d=0.5, window_size=50
        )
        assert window_size == 50
        # Indices 0..window_size-2 are NaN; index window_size-1 is the first valid value.
        assert np.all(np.isnan(result[:window_size - 1]))
        assert not np.isnan(result[window_size - 1])

    def test_first_valid_index_is_window_minus_one(self, sample_series):
        """Off-by-one fix: result[window_size-1] must be valid, result[window_size-2] NaN."""
        result, window_size = fractional_differentiate_series(
            sample_series, d=0.5, window_size=10
        )
        assert np.isnan(result[window_size - 2])
        assert not np.isnan(result[window_size - 1])

    def test_fractional_differentiate_series_d_zero(self, sample_series):
        """Test differentiation with d=0 (should return original series)"""
        result, window_size = fractional_differentiate_series(sample_series, d=0.0)
        np.testing.assert_array_almost_equal(
            result[window_size:], sample_series.values[window_size:]
        )

    def test_fractional_differentiate_series_d_one(self, sample_series):
        """Test differentiation with d=1 (should return first differences)"""
        result, window_size = fractional_differentiate_series(
            sample_series, d=1.0, window_size=2
        )
        expected_diff = np.diff(sample_series.values)
        computed_values = result[window_size:]
        expected_values = expected_diff[1:]

        np.testing.assert_array_almost_equal(computed_values, expected_values)

    def test_fractional_differentiate_series_numpy_array(self):
        """Test with numpy array input"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result, window_size = fractional_differentiate_series(
            data, d=0.5, window_size=3
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_fractional_differentiate_series_edge_cases(self):
        """Test edge cases"""
        # Very short series
        short_series = pd.Series([1, 2, 3])
        result, window_size = fractional_differentiate_series(
            short_series, d=0.5, window_size=2
        )
        assert len(result) == 3

        # Constant series
        constant_series = pd.Series([5, 5, 5, 5, 5])
        result, window_size = fractional_differentiate_series(constant_series, d=0.5)
        # For constant series, fractional diff should be close to zero (except initial values)
        assert np.allclose(result[window_size:], 0, atol=1e-10)


class TestGetOptimalFractionalDifferentiationOrder:

    @pytest.fixture
    def sample_non_stationary_series(self):
        """Create a non-stationary time series (random walk)"""
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(100)))

    @patch("okmich_quant_features.fractional_diff.adfuller")
    def test_get_optimal_d_stationary_case(
        self, mock_adfuller, sample_non_stationary_series
    ):
        """Test optimal d finding with mock ADF results"""
        # Mock ADF to return stationary results for all d
        mock_adfuller.return_value = (
            MagicMock(),
            0.001,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        optimal_d, window_size, diff_series, adf_results = (
            get_optimal_fractional_differentiation_order(
                sample_non_stationary_series, max_d=1.0, min_d=0.0, step=0.1
            )
        )

        assert isinstance(optimal_d, float)
        assert isinstance(window_size, int)
        assert isinstance(diff_series, np.ndarray)
        assert isinstance(adf_results, dict)
        assert optimal_d == 1.0

    @patch("okmich_quant_features.fractional_diff.adfuller")
    def test_get_optimal_d_non_stationary_case(
        self, mock_adfuller, sample_non_stationary_series
    ):
        """Test optimal d finding when no d makes it stationary"""
        # Mock ADF to return non-stationary results for all d
        mock_adfuller.return_value = (
            MagicMock(),
            0.5,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        optimal_d, window_size, diff_series, adf_results = (
            get_optimal_fractional_differentiation_order(
                sample_non_stationary_series, max_d=1.0, min_d=0.0, step=0.1
            )
        )

        assert optimal_d == 1.0  # Should fallback to max_d

    def test_get_optimal_d_parameter_ranges(self, sample_non_stationary_series):
        """Test with different parameter ranges"""
        with patch("okmich_quant_features.fractional_diff.adfuller") as mock_adfuller:
            mock_adfuller.return_value = (
                MagicMock(),
                0.001,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )

            # Test that the function runs without error and returns reasonable values
            optimal_d, window_size, diff_series, adf_results = (
                get_optimal_fractional_differentiation_order(
                    sample_non_stationary_series, max_d=0.8, min_d=0.2, step=0.05
                )
            )

            # Basic sanity checks
            assert isinstance(optimal_d, float)
            assert isinstance(window_size, int)
            assert isinstance(diff_series, np.ndarray)
            assert isinstance(adf_results, dict)
            assert len(diff_series) == len(sample_non_stationary_series)

    def test_get_optimal_d_adf_threshold(self, sample_non_stationary_series):
        """Test different ADF thresholds - accept any valid result"""
        with patch("okmich_quant_features.fractional_diff.adfuller") as mock_adfuller:
            mock_adfuller.return_value = (
                MagicMock(),
                0.001,
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )
            optimal_d, window_size, diff_series, adf_results = (
                get_optimal_fractional_differentiation_order(
                    sample_non_stationary_series,
                    max_d=1.0,
                    min_d=0.0,
                    step=0.1,
                    adf_threshold=0.05,
                )
            )
            assert isinstance(optimal_d, float)
            assert isinstance(window_size, int)
            assert isinstance(diff_series, np.ndarray)
            assert isinstance(adf_results, dict)

            assert 0.0 <= optimal_d <= 1.0


class TestFractionalDifferentiator:
    """Test suite for FractionalDifferentiator class"""

    @pytest.fixture
    def differentiator(self):
        """Create a FractionalDifferentiator instance"""
        return FractionalDifferentiator(window_size=50)

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(100)) + 100)

    def test_initialization(self):
        """Test class initialization"""
        diff = FractionalDifferentiator(window_size=100)
        assert diff.window_size == 100
        assert diff._weights_cache == {}

    def test_differentiate_basic(self, differentiator, sample_data):
        """Test basic differentiation"""
        result = differentiator.differentiate(sample_data, d=0.5)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data)
        # window_size=50: indices 0..48 are NaN, index 49 is first valid.
        assert np.all(np.isnan(result[:49]))
        assert not np.isnan(result[49])

    def test_differentiate_custom_window(self, differentiator, sample_data):
        """Test with custom window size"""
        result = differentiator.differentiate(sample_data, d=0.5, window_size=30)
        # window_size=30: indices 0..28 are NaN, index 29 is first valid.
        assert np.all(np.isnan(result[:29]))
        assert not np.isnan(result[29])

    def test_differentiate_cache_usage(self, differentiator, sample_data):
        """Test that weights cache is used"""
        # First call should populate cache
        result1 = differentiator.differentiate(sample_data, d=0.5)
        cache_size_first = len(differentiator._weights_cache)

        # Second call should use cache
        result2 = differentiator.differentiate(sample_data, d=0.5)
        cache_size_second = len(differentiator._weights_cache)

        assert cache_size_first == cache_size_second == 1
        np.testing.assert_array_equal(result1, result2)

    def test_differentiate_different_parameters(self, differentiator, sample_data):
        """Test with different d values"""
        result1 = differentiator.differentiate(sample_data, d=0.3)
        result2 = differentiator.differentiate(sample_data, d=0.7)

        # Results should be different for different d values
        assert not np.array_equal(result1[50:], result2[50:], equal_nan=True)
        assert len(differentiator._weights_cache) == 2

    def test_clear_cache(self, differentiator, sample_data):
        """Test cache clearing"""
        # Populate cache
        differentiator.differentiate(sample_data, d=0.5)
        differentiator.differentiate(sample_data, d=0.7)
        assert len(differentiator._weights_cache) == 2

        # Clear cache
        differentiator.clear_cache()
        assert len(differentiator._weights_cache) == 0

    def test_differentiate_edge_cases(self, differentiator):
        """Test edge cases"""
        # Short series
        short_data = pd.Series([1, 2, 3])
        result = differentiator.differentiate(short_data, d=0.5, window_size=2)
        assert len(result) == 3

        # Constant series
        constant_data = pd.Series([5, 5, 5, 5, 5])
        result = differentiator.differentiate(constant_data, d=0.5)
        # Should be close to zero after window_size
        assert np.allclose(result[50:], 0, atol=1e-10)


class TestIntegration:
    """Integration tests for the complete functionality"""

    @pytest.fixture
    def sample_random_walk(self):
        """Create a random walk series"""
        np.random.seed(42)
        return pd.Series(np.cumsum(np.random.randn(200)))

    def test_integration_workflow(self, sample_random_walk):
        """Test complete workflow from raw series to optimal differentiation"""
        # Test that we can find optimal d and apply it
        optimal_d, window_size, diff_series, adf_results = (
            get_optimal_fractional_differentiation_order(
                sample_random_walk, max_d=1.0, min_d=0.0, step=0.1
            )
        )

        assert isinstance(optimal_d, float)
        assert 0.0 <= optimal_d <= 1.0
        assert isinstance(window_size, int)
        assert window_size > 0
        assert len(diff_series) == len(sample_random_walk)
        assert isinstance(adf_results, dict)
        assert len(adf_results) > 0

    def test_class_integration(self, sample_random_walk):
        """Test class-based integration"""
        differentiator = FractionalDifferentiator(window_size=100)

        # Test multiple differentiations
        result_05 = differentiator.differentiate(sample_random_walk, d=0.5)
        result_07 = differentiator.differentiate(sample_random_walk, d=0.7)

        assert len(result_05) == len(sample_random_walk)
        assert len(result_07) == len(sample_random_walk)
        assert not np.array_equal(result_05[100:], result_07[100:], equal_nan=True)


# Additional edge case tests
def test_extreme_values():
    """Test with extreme input values"""
    # Very large values
    large_series = pd.Series([1e10, 2e10, 3e10, 4e10, 5e10])
    result, _ = fractional_differentiate_series(large_series, d=0.5, window_size=3)
    assert not np.any(np.isinf(result))

    # Very small values
    small_series = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    result, _ = fractional_differentiate_series(small_series, d=0.5, window_size=3)
    assert not np.any(np.isnan(result[3:]))


def test_nan_handling():
    """Test behavior with NaN values in input"""
    series_with_nan = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
    result, window_size = fractional_differentiate_series(
        series_with_nan, d=0.5, window_size=3
    )
    assert len(result) == len(series_with_nan)
    assert np.all(np.isnan(result[:window_size - 1]))
    assert isinstance(result, np.ndarray)
    assert len(result) == len(series_with_nan)


# ─── Validation tests for correctness fixes ───────────────────────────────────

def test_optimal_window_consistency():
    """
    Fix #3: get_optimal_fractional_differentiation_order must return the window
    that corresponds to optimal_d, not the window from the last scanned d.
    Verify by re-running fractional_differentiate_series(optimal_d) independently
    and comparing the output element-wise.
    """
    rng = np.random.default_rng(7)
    series = pd.Series(np.cumsum(rng.standard_normal(300)))

    optimal_d, optimal_window, opt_series, _ = get_optimal_fractional_differentiation_order(
        series, min_d=0.0, max_d=1.0, step=0.1
    )

    # Recompute independently for the returned optimal_d
    ref_series, ref_window = fractional_differentiate_series(series, d=optimal_d)

    assert optimal_window == ref_window, (
        f"Returned window {optimal_window} does not match independently computed "
        f"window {ref_window} for optimal_d={optimal_d:.2f}"
    )
    np.testing.assert_array_equal(
        opt_series, ref_series,
        err_msg="Optimal series differs from independent recomputation for the same d.",
    )


def test_off_by_one_window_size_boundary():
    """
    Fix #2: index window_size-1 must be the FIRST valid output; index window_size-2
    must still be NaN. Both assertions together pin the exact boundary.
    """
    series = pd.Series(np.arange(1.0, 201.0))
    for ws in [5, 10, 20, 50]:
        result, _ = fractional_differentiate_series(series, d=0.5, window_size=ws)
        assert np.isnan(result[ws - 2]), f"result[{ws-2}] should be NaN for window_size={ws}"
        assert not np.isnan(result[ws - 1]), f"result[{ws-1}] should be valid for window_size={ws}"
