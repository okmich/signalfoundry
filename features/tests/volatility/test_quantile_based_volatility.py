import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volatility import quantile_based_volatility_labeling


class TestQuantileBasedVolatilityClusterLabeling:

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing"""
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        # Create synthetic price data with some volatility clustering
        prices = [100]
        for i in range(1, 100):
            # Introduce volatility clustering: higher volatility periods
            if 30 <= i <= 50 or 70 <= i <= 80:
                change = np.random.normal(0, 0.03)  # High volatility
            else:
                change = np.random.normal(0, 0.01)  # Low volatility
            prices.append(prices[-1] * np.exp(change))

        return pd.DataFrame({"Close": prices}, index=dates)

    def test_basic_functionality(self, sample_data):
        """Test basic functionality with default parameters"""
        result = quantile_based_volatility_labeling(sample_data.Close)

        # Check return type and shape
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)
        assert result.name == "vol_label"

        # Check that all values are in {-1, 0, 1}
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_custom_window_size(self):
        """Test with custom window size"""
        # Create simple test data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        close_prices = [100 + i * 0.1 for i in range(50)]  # Linear trend
        df = pd.DataFrame({"Close": close_prices}, index=dates)

        window = 10
        result = quantile_based_volatility_labeling(df.Close, window=window)

        # Check return type and shape
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_custom_quantiles(self):
        """Test with custom quantile parameters"""
        # Create data with known volatility characteristics
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")

        # Create prices with alternating high/low volatility
        prices = [100]
        for i in range(1, 60):
            if i % 20 < 10:  # Low volatility period
                change = np.random.normal(0, 0.005)
            else:  # High volatility period
                change = np.random.normal(0, 0.03)
            prices.append(prices[-1] * np.exp(change))

        df = pd.DataFrame({"Close": prices}, index=dates)

        # Test with extreme quantiles
        result = quantile_based_volatility_labeling(
            df.Close, window=15, upper_q=0.9, lower_q=0.1
        )

        assert isinstance(result, pd.Series)
        # Should still produce valid trend
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_custom_series_name(self):
        """Test custom series name parameter"""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"Close": [100 + i * 0.01 for i in range(30)]}, index=dates)

        custom_name = "my_custom_label"
        result = quantile_based_volatility_labeling(df.Close, name=custom_name)

        assert result.name == custom_name

    def test_constant_price_data(self):
        """Test with constant price data (zero volatility)"""
        dates = pd.date_range("2023-01-01", periods=40, freq="D")
        constant_prices = [100] * 40
        df = pd.DataFrame({"Close": constant_prices}, index=dates)

        result = quantile_based_volatility_labeling(df.Close, window=10)

        # Check return type and shape
        assert isinstance(result, pd.Series)
        assert len(result) == 40

        # With constant prices, returns are 0, volatility is 0
        # But due to the rolling quantile calculation, some edge effects may occur
        # The important thing is that it doesn't crash and produces valid trend
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_monotonic_increasing_prices(self):
        """Test with monotonic increasing prices (low but consistent volatility)"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        # Small consistent positive returns
        prices = [100]
        for i in range(1, 50):
            prices.append(prices[-1] * 1.001)  # 0.1% daily return

        df = pd.DataFrame({"Close": prices}, index=dates)
        result = quantile_based_volatility_labeling(df.Close, window=10)

        assert isinstance(result, pd.Series)
        # With constant returns, volatility is constant, but quantiles will vary due to rolling window
        # Just verify it produces valid output
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_edge_case_small_dataset(self):
        """Test with very small dataset"""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"Close": [100, 101, 102, 101, 100]}, index=dates)

        result = quantile_based_volatility_labeling(df.Close, window=3)

        assert isinstance(result, pd.Series)
        assert len(result) == 5
        # Verify it produces valid trend
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_nan_handling_in_input(self):
        """Test handling of NaN values in input Close prices"""
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        close_prices = [100, 101, np.nan, 102, 103] + [
            100 + i * 0.1 for i in range(5, 25)
        ]
        df = pd.DataFrame({"Close": close_prices}, index=dates)

        result = quantile_based_volatility_labeling(df.Close, window=5)

        assert isinstance(result, pd.Series)
        assert len(result) == 25
        # Function should handle NaN gracefully
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_reproducibility(self):
        """Test that function produces reproducible results"""
        np.random.seed(456)
        dates = pd.date_range("2023-01-01", periods=40, freq="D")
        prices = [100]
        for i in range(1, 40):
            change = np.random.normal(0, 0.02)
            prices.append(prices[-1] * np.exp(change))

        df = pd.DataFrame({"Close": prices}, index=dates)

        result1 = quantile_based_volatility_labeling(df.Close, window=10)
        result2 = quantile_based_volatility_labeling(df.Close, window=10)

        # Results should be identical for same input
        pd.testing.assert_series_equal(result1, result2)

    def test_window_larger_than_data(self):
        """Test edge case where window is larger than dataset"""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"Close": [100 + i * 0.1 for i in range(10)]}, index=dates)

        result = quantile_based_volatility_labeling(df.Close, window=20)

        assert isinstance(result, pd.Series)
        assert len(result) == 10
        # Verify it produces valid trend
        unique_values = set(result.unique())
        assert unique_values.issubset({-1.0, 0.0, 1.0})

    def test_extreme_quantile_values(self):
        """Test with extreme quantile values"""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"Close": [100 + i * 0.01 for i in range(30)]}, index=dates)

        # Test boundary quantile values
        result1 = quantile_based_volatility_labeling(df.Close, upper_q=1.0, lower_q=0.0)
        result2 = quantile_based_volatility_labeling(
            df.Close, upper_q=0.5, lower_q=0.05
        )

        assert isinstance(result1, pd.Series)
        assert isinstance(result2, pd.Series)
        # Both should produce valid trend
        unique_values1 = set(result1.unique())
        unique_values2 = set(result2.unique())
        assert unique_values1.issubset({-1.0, 0.0, 1.0})
        assert unique_values2.issubset({-1.0, 0.0, 1.0})

    def test_index_preservation(self):
        """Test that the original index is preserved in the result"""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame({"Close": [100 + i * 0.01 for i in range(20)]}, index=dates)

        result = quantile_based_volatility_labeling(df.Close)

        # Index should be preserved
        pd.testing.assert_index_equal(result.index, df.index)

    def test_mathematical_correctness(self):
        """Test the mathematical correctness of the implementation"""
        # Create a simple test case where we can manually verify the calculations
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101]
        df = pd.DataFrame({"Close": prices}, index=dates)

        result = quantile_based_volatility_labeling(
            df.Close, window=5, upper_q=0.7, lower_q=0.3
        )

        # Manual calculation — fix #10: NaN is preserved, not filled with 0.0
        ret = np.log(df["Close"]).diff()
        vol = ret.rolling(5).std()
        uq = vol.rolling(5).quantile(0.7)
        lq = vol.rolling(5).quantile(0.3)

        # Check that the function produces the same trend as manual calculation
        expected_labels = pd.Series(0.0, index=df.index, name="vol_label")
        expected_labels[vol >= uq] = 1.0
        expected_labels[vol <= lq] = -1.0
        expected_labels = expected_labels.fillna(0.0)

        pd.testing.assert_series_equal(result, expected_labels)


# Parametrized test for different window sizes
@pytest.mark.parametrize("window_size", [5, 10, 15, 20])
def test_different_window_sizes(window_size):
    """Test function with different window sizes"""
    np.random.seed(789)
    dates = pd.date_range("2023-01-01", periods=50, freq="D")
    prices = [100]
    for i in range(1, 50):
        change = np.random.normal(0, 0.01 + (i % 10) * 0.001)  # Varying volatility
        prices.append(prices[-1] * np.exp(change))

    df = pd.DataFrame({"Close": prices}, index=dates)
    result = quantile_based_volatility_labeling(df.Close, window=window_size)

    assert isinstance(result, pd.Series)
    assert len(result) == 50
    unique_values = set(result.unique())
    assert unique_values.issubset({-1.0, 0.0, 1.0})


# --- Validation tests for correctness fixes ---

from okmich_quant_features.volatility import optimize_quantile_based_volatility_labels


class TestQuantileVolatilityNaNPreservation:
    """Fix #10: fillna(0.0) removed — early-window vol must not be biased by an artificial zero return."""

    def test_early_vol_unaffected_by_zero_fill(self):
        """
        With a large constant-volatility return series, the rolling std at position
        window-1 should reflect only real returns (positions 1..window-1), not a
        zero-filled position 0. Verify by comparing against manual NaN-clean computation.
        """
        rng = np.random.default_rng(1)
        n = 50
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=dates)

        result = quantile_based_volatility_labeling(prices, window=10)

        # Compute what the labels SHOULD be without zero-fill bias
        ret_clean = np.log(prices / prices.shift())   # NaN at 0, not 0.0
        vol_clean = ret_clean.rolling(10).std()
        uq = vol_clean.rolling(10).quantile(0.75)
        lq = vol_clean.rolling(10).quantile(0.25)
        expected = pd.Series(0.0, index=prices.index, name="vol_label")
        expected[vol_clean >= uq] = 1.0
        expected[vol_clean <= lq] = -1.0
        expected = expected.fillna(0.0)

        pd.testing.assert_series_equal(result, expected)


class TestOptimizeQuantileVerbose:
    """Fix #10: print calls gated behind verbose=False."""

    def test_verbose_false_no_stdout(self, capsys):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        prices = pd.Series(100.0 * np.cumprod(1 + np.random.default_rng(2).normal(0, 0.01, 60)),
                           index=dates)
        optimize_quantile_based_volatility_labels(
            prices,
            window_range=[10, 15],
            upper_q_range=[0.7],
            lower_q_range=[0.3],
            verbose=False,
        )
        captured = capsys.readouterr()
        assert captured.out == "", "verbose=False must not print anything"

    def test_verbose_true_prints(self, capsys):
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        prices = pd.Series(100.0 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.01, 60)),
                           index=dates)
        optimize_quantile_based_volatility_labels(
            prices,
            window_range=[10, 15],
            upper_q_range=[0.7],
            lower_q_range=[0.3],
            verbose=True,
        )
        captured = capsys.readouterr()
        assert "Optimization complete" in captured.out


class TestOptimizeQuantileMutableDefaults:
    """Fix #11: calling without window_range/upper_q_range/lower_q_range must use defaults."""

    def test_default_args_produce_result(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.Series(100.0 * np.cumprod(1 + np.random.default_rng(4).normal(0, 0.01, 100)),
                           index=dates)
        params, labels = optimize_quantile_based_volatility_labels(prices)
        assert params is not None
        assert isinstance(labels, pd.Series)

    def test_repeated_calls_give_same_result(self):
        """Mutable default isolation: second call must not be affected by first."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        prices = pd.Series(100.0 * np.cumprod(1 + np.random.default_rng(5).normal(0, 0.01, 100)),
                           index=dates)
        params1, _ = optimize_quantile_based_volatility_labels(prices)
        params2, _ = optimize_quantile_based_volatility_labels(prices)
        assert params1["window"] == params2["window"]
        assert params1["upper_q"] == params2["upper_q"]
