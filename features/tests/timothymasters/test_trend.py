import numpy as np
import pytest
from okmich_quant_features.timothymasters.trend import (
    linear_trend,
    quadratic_trend,
    cubic_trend,
    linear_deviation,
    quadratic_deviation,
    cubic_deviation,
    adx,
    aroon_up,
    aroon_down,
    aroon_diff,
)


class TestLinearTrend:
    """Test Linear Trend (#12)."""

    def test_linear_data_high_trend(self):
        """Linear increasing data should show high positive trend."""
        close = np.arange(1, 201, dtype=np.float64)  # Perfect linear
        high = close + 1.0
        low = close - 1.0
        result = linear_trend(high, low, close, period=50)

        valid = result[~np.isnan(result)]
        # Should be strongly positive (approaching 50)
        assert np.mean(valid) > 40, f"Expected high trend for linear data, got {np.mean(valid)}"

    def test_constant_data_zero_trend(self):
        """Constant data should show near-zero trend."""
        close = np.ones(200) * 100.0
        high = close + 1.0
        low = close - 1.0
        result = linear_trend(high, low, close, period=50)

        valid = result[~np.isnan(result)]
        # Should be near 0
        assert np.abs(np.mean(valid)) < 5, f"Expected ~0 trend for constant, got {np.mean(valid)}"

    def test_output_range(self):
        """Trend should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = linear_trend(high, low, close, period=50)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min trend {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max trend {valid.max()} above 50"


class TestQuadraticTrend:
    """Test Quadratic Trend (#13)."""

    def test_output_range(self):
        """Quadratic trend should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = quadratic_trend(high, low, close, period=50)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestCubicTrend:
    """Test Cubic Trend (#14)."""

    def test_output_range(self):
        """Cubic trend should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = cubic_trend(high, low, close, period=50)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestTrendDeviation:
    """Test Trend Deviation indicators (#15-17)."""

    def test_linear_data_low_deviation(self):
        """Perfect linear data should have low deviation from trend."""
        close = np.arange(1, 201, dtype=np.float64)
        result = linear_deviation(close, period=50)

        valid = result[~np.isnan(result)]
        # Deviation should be very low for perfect linear
        assert np.mean(valid) < 10, f"Expected low deviation for linear, got {np.mean(valid)}"

    def test_noisy_data_high_deviation(self):
        """Noisy data should have high deviation spread from trend."""
        np.random.seed(42)
        close = np.random.randn(200) * 10 + np.arange(200) * 0.1 + 100
        result = linear_deviation(close, period=50)

        valid = result[~np.isnan(result)]
        # Deviation mean should be near 0, but variance/spread should be high
        # Check that std deviation is > 5 (indicates high variability)
        assert np.std(valid) > 10, f"Expected high deviation spread, got std={np.std(valid)}"


class TestQuadraticDeviation:
    """Test Quadratic Deviation (#16)."""

    def test_output_range(self):
        """Quadratic deviation should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = quadratic_deviation(close, period=50)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestCubicDeviation:
    """Test Cubic Deviation (#17)."""

    def test_output_range(self):
        """Cubic deviation should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = cubic_deviation(close, period=50)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestADX:
    """Test ADX (#18)."""

    def test_trending_market_high_adx(self):
        """Strong trend should produce high ADX."""
        # Strong uptrend
        close = np.arange(1, 201, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0

        result = adx(high, low, close, period=14)

        valid = result[~np.isnan(result)]
        # ADX should be high for strong trend
        assert np.mean(valid) > 30, f"Expected high ADX for trend, got {np.mean(valid)}"

    def test_ranging_market_low_adx(self):
        """Ranging market should produce low ADX."""
        # Oscillating pattern
        close = 100.0 + 5.0 * np.sin(np.arange(200) * 0.5)
        high = close + 1.0
        low = close - 1.0

        result = adx(high, low, close, period=14)

        valid = result[~np.isnan(result)]
        # ADX should be lower for ranging
        assert np.mean(valid) < 40, f"Expected lower ADX for ranging, got {np.mean(valid)}"

    def test_output_range(self):
        """ADX should be in range [0, ~50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))

        result = adx(high, low, close, period=14)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0), f"Min ADX {valid.min()} below 0"
        assert np.all(valid <= 50), f"Max ADX {valid.max()} above 50"


class TestAroon:
    """Test Aroon indicators (#19-21)."""

    def test_uptrend_high_aroon_up(self):
        """Uptrend should have high Aroon Up."""
        close = np.arange(1, 201, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        result = aroon_up(high, low, period=25)

        valid = result[~np.isnan(result)]
        # Should be high (approaching 50)
        assert np.mean(valid) > 30, f"Expected high Aroon Up for uptrend, got {np.mean(valid)}"

    def test_downtrend_high_aroon_down(self):
        """Downtrend should have high Aroon Down."""
        close = np.arange(200, 0, -1, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0
        result = aroon_down(high, low, period=25)

        valid = result[~np.isnan(result)]
        # Should be high (approaching 50)
        assert np.mean(valid) > 30, f"Expected high Aroon Down for downtrend, got {np.mean(valid)}"

    def test_output_range(self):
        """Aroon should be in range [0, 100] (or [-100, 100] for diff)."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))

        # aroon_up and aroon_down are in [0, 100]
        for func in [aroon_up, aroon_down]:
            result = func(high, low, period=25)
            valid = result[~np.isnan(result)]
            assert np.all(valid >= 0), f"{func.__name__} min {valid.min()} below 0"
            assert np.all(valid <= 100), f"{func.__name__} max {valid.max()} above 100"

        # aroon_diff is in [-100, 100]
        result = aroon_diff(high, low, period=25)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100), f"aroon_diff min {valid.min()} below -100"
        assert np.all(valid <= 100), f"aroon_diff max {valid.max()} above 100"
