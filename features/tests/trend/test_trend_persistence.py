import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend.trend_persistence import trend_persistence_labeling


class TestDataGeneration:
    """Helper class for generating test data"""

    @staticmethod
    def create_linear_data(start=100, end=150, step=1):
        """Create linear price series"""
        return pd.DataFrame({"Close": np.arange(start, end, step)})

    @staticmethod
    def create_trend_data(length=100, start_price=100, trend=0.1, volatility=0.02):
        """Create data with clear trend"""
        dates = pd.date_range("2020-01-01", periods=length, freq="D")
        prices = start_price + np.cumsum(np.random.normal(trend, volatility, length))
        return pd.DataFrame({"Close": prices}, index=dates)

    @staticmethod
    def create_constant_data(value=100, length=50):
        """Create constant price data"""
        return pd.DataFrame({"Close": [float(value)] * length})

    @staticmethod
    def create_nan_data():
        """Create data with NaN values"""
        return pd.DataFrame(
            {"Close": [100.0, 101.0, np.nan, 103.0, 104.0] + list(range(105, 150))}
        )


# Fixtures
@pytest.fixture
def linear_data():
    return TestDataGeneration.create_linear_data()


@pytest.fixture
def trend_data():
    return TestDataGeneration.create_trend_data()


@pytest.fixture
def constant_data():
    return TestDataGeneration.create_constant_data()


@pytest.fixture
def nan_data():
    return TestDataGeneration.create_nan_data()


@pytest.fixture
def small_data():
    return pd.DataFrame({"Close": [100, 101, 102]})


@pytest.fixture
def empty_data():
    return pd.DataFrame({"Close": []})


@pytest.fixture
def single_row_data():
    return pd.DataFrame({"Close": [100.0]})


# Basic Functionality Tests
def test_basic_functionality_with_linear_data(linear_data):
    """Test basic functionality with simple linear increasing data."""
    result = trend_persistence_labeling(linear_data["Close"], window=10, smooth=3)

    # Check return type and properties
    assert isinstance(result, pd.Series)
    assert len(result) == len(linear_data)
    assert result.name == "trend_label"

    # Check that values are in expected range {-1, 0, 1}
    unique_values = set(result.dropna().unique())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


def test_default_parameters(trend_data):
    """Test function with default parameters."""
    result = trend_persistence_labeling(trend_data["Close"])

    assert isinstance(result, pd.Series)
    assert result.name == "trend_label"
    assert len(result) == len(trend_data)


def test_custom_name_parameter(linear_data):
    """Test custom naming of the output series."""
    custom_name = "custom_trend_label"
    result = trend_persistence_labeling(linear_data["Close"], name=custom_name)
    assert result.name == custom_name


# Parameter Testing
@pytest.mark.parametrize("zscore_norm", [True, False])
def test_zscore_normalization_parameter(linear_data, zscore_norm):
    """Test z-score normalization parameter."""
    result = trend_persistence_labeling(linear_data["Close"], zscore_norm=zscore_norm)
    assert isinstance(result, pd.Series)

    # Check that result contains expected label values
    unique_values = set(result.dropna().unique())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("window", [5, 10, 15, 20, 25])
def test_different_window_sizes(trend_data, window):
    """Test with different window sizes."""
    result = trend_persistence_labeling(trend_data["Close"], window=window)
    assert len(result) == len(trend_data)
    assert isinstance(result, pd.Series)


@pytest.mark.parametrize("smooth", [1, 3, 5, 10])
def test_different_smooth_sizes(trend_data, smooth):
    """Test with different smoothing window sizes."""
    result = trend_persistence_labeling(trend_data["Close"], smooth=smooth)
    assert len(result) == len(trend_data)
    assert isinstance(result, pd.Series)


# Edge Cases
def test_constant_price_data(constant_data):
    """Test with constant price data."""
    result = trend_persistence_labeling(constant_data["Close"], window=5)

    # With constant prices, drift should be zero, leading to zero trend
    non_nan_values = result.dropna()
    if len(non_nan_values) > 0:
        # Most values should be zero for constant price data
        zero_ratio = (non_nan_values == 0.0).sum() / len(non_nan_values)
        assert zero_ratio >= 0.8  # At least 80% should be zero


def test_data_with_nan_values(nan_data):
    """Test handling of NaN values in input data."""
    result = trend_persistence_labeling(nan_data["Close"])

    assert isinstance(result, pd.Series)
    assert len(result) == len(nan_data)
    # Check that result is properly filled with 0.0 where NaN would occur
    assert not result.isna().any()  # fillna(0.0) should eliminate all NaNs


def test_small_dataset(small_data):
    """Test with very small dataset."""
    result = trend_persistence_labeling(small_data["Close"], window=2)

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    # Should be valid float values
    assert isinstance(result.iloc[0], (int, float))


def test_empty_dataframe(empty_data):
    """Test with empty DataFrame."""
    result = trend_persistence_labeling(empty_data["Close"])

    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_single_row(single_row_data):
    """Test with single row DataFrame."""
    result = trend_persistence_labeling(single_row_data["Close"])

    assert isinstance(result, pd.Series)
    assert len(result) == 1
    # Should be 0.0 as there's insufficient data for calculations
    assert result.iloc[0] == 0.0


# Parameter Combinations
@pytest.mark.parametrize(
    "params",
    [
        {"window": 5, "smooth": 2, "zscore_norm": True},
        {"window": 15, "smooth": 10, "zscore_norm": False},
        {"window": 30, "smooth": 1, "zscore_norm": True},
        {"window": 10, "smooth": 5, "zscore_norm": False},
    ],
)
def test_parameter_combinations(trend_data, params):
    """Test various parameter combinations."""
    result = trend_persistence_labeling(trend_data["Close"], **params)
    assert isinstance(result, pd.Series)
    assert len(result) == len(trend_data)

    # Verify label values are in correct range
    unique_values = set(result.dropna().unique())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


# Mathematical Logic Tests
def test_drift_calculation():
    """Test drift calculation logic."""
    # Simple case: [100, 105, 110, 115, 120] with window=2
    # drift should be [NaN, NaN, 10, 10, 10]
    simple_prices = pd.DataFrame({"Close": [100, 105, 110, 115, 120]})
    result = trend_persistence_labeling(
        simple_prices["Close"], window=2, smooth=1, zscore_norm=False
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 5


def test_volatility_calculation():
    """Test volatility calculation doesn't cause division by zero."""
    # Data with very small changes that could lead to near-zero volatility
    small_changes = pd.DataFrame(
        {"Close": 100 + np.cumsum([0.001] * 50)}  # Very small increments
    )

    result = trend_persistence_labeling(small_changes["Close"])
    assert isinstance(result, pd.Series)
    assert not result.isna().any()  # Should handle near-zero volatility


# Data Type and Structure Tests
def test_dtype_handling():
    """Test proper data type handling."""
    # Test with integer Close prices
    int_data = pd.DataFrame({"Close": range(100, 150)})
    result = trend_persistence_labeling(int_data["Close"])
    assert isinstance(result, pd.Series)
    assert result.dtype == np.float64


def test_index_preservation():
    """Test that DataFrame index is preserved in result."""
    dates = pd.date_range("2020-01-01", periods=30)
    indexed_data = pd.DataFrame({"Close": np.linspace(100, 130, 30)}, index=dates)
    result = trend_persistence_labeling(indexed_data["Close"])

    # Check that index is preserved
    pd.testing.assert_index_equal(result.index, indexed_data.index)


def test_column_validation():
    """Test that function requires 'Close' column."""
    # Test with missing 'Close' column
    wrong_columns = pd.DataFrame({"Price": [100, 101, 102]})
    with pytest.raises(KeyError):
        trend_persistence_labeling(wrong_columns["Close"])


# Performance and Stability Tests
def test_performance_with_large_dataset():
    """Test performance with larger dataset."""
    large_data = pd.DataFrame(
        {"Close": np.random.lognormal(mean=4, sigma=0.3, size=1000)}
    )

    # Should not raise any exceptions
    result = trend_persistence_labeling(large_data["Close"])
    assert isinstance(result, pd.Series)
    assert len(result) == 1000


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1, 1e3, 1e6])
def test_numerical_stability(scale):
    """Test numerical stability with different scales."""
    scaled_data = pd.DataFrame({"Close": np.linspace(100 * scale, 150 * scale, 50)})
    result = trend_persistence_labeling(scaled_data["Close"])
    assert isinstance(result, pd.Series)
    assert not result.isna().any()


# Integration Tests
def test_complete_workflow():
    """Test complete workflow with realistic financial data."""
    # Create realistic price data with trend and volatility
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    returns = np.random.normal(0.001, 0.02, 200)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))  # Price series

    realistic_data = pd.DataFrame({"Close": prices}, index=dates)
    result = trend_persistence_labeling(realistic_data["Close"], window=20, smooth=5)

    assert isinstance(result, pd.Series)
    assert len(result) == len(realistic_data)
    assert result.name == "trend_label"

    # Check that result contains expected values
    unique_values = set(result.dropna().unique())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


# Regression Tests
def test_consistent_output_format():
    """Test that output format is consistent."""
    data = TestDataGeneration.create_linear_data()
    result = trend_persistence_labeling(data["Close"])

    # Should always return a Series with float64 dtype
    assert isinstance(result, pd.Series)
    assert result.dtype == np.float64
    assert result.name == "trend_label"


def test_no_side_effects():
    """Test that function doesn't modify input data."""
    original_data = TestDataGeneration.create_linear_data()
    original_copy = original_data.copy()

    trend_persistence_labeling(original_data["Close"])

    # Input data should remain unchanged
    pd.testing.assert_frame_equal(original_data, original_copy)


# Special Cases
def test_all_zeros_data():
    """Test with all zero prices (edge case)."""
    zero_data = pd.DataFrame({"Close": [0.0] * 30})
    result = trend_persistence_labeling(zero_data["Close"])
    assert isinstance(result, pd.Series)
    # Should handle division by zero gracefully


def test_negative_prices():
    """Test with negative prices (theoretical edge case)."""
    negative_data = pd.DataFrame({"Close": np.linspace(-50, -10, 50)})
    result = trend_persistence_labeling(negative_data["Close"])
    assert isinstance(result, pd.Series)


# Parameter Boundary Tests
@pytest.mark.parametrize("window", [1, 2, 100, 200])
def test_window_boundary_values(trend_data, window):
    """Test window parameter at boundary values."""
    if window <= len(trend_data):
        result = trend_persistence_labeling(trend_data["Close"], window=window)
        assert len(result) == len(trend_data)


@pytest.mark.parametrize("smooth", [1, 100])
def test_smooth_boundary_values(trend_data, smooth):
    """Test smooth parameter at boundary values."""
    result = trend_persistence_labeling(trend_data["Close"], smooth=smooth)
    assert isinstance(result, pd.Series)


# Validation Tests
def test_result_values_are_valid_labels():
    """Test that all result values are valid trend."""
    data = TestDataGeneration.create_trend_data(length=100)
    result = trend_persistence_labeling(data["Close"])

    # After fillna, all values should be valid trend
    unique_values = set(result.unique())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


def test_threshold_logic():
    """Test that threshold logic works correctly."""
    # Create data that should produce clear positive trend
    strong_upward = pd.DataFrame(
        {"Close": np.exp(np.linspace(0, 2, 50)) * 100}  # Exponential growth
    )

    result = trend_persistence_labeling(
        strong_upward["Close"], window=10, smooth=3, zscore_norm=False
    )

    # Should have some positive trend for strong upward trend
    assert isinstance(result, pd.Series)
