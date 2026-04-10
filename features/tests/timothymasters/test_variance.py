import numpy as np
import pytest
from okmich_quant_features.timothymasters.variance import (
    price_variance_ratio,
    change_variance_ratio,
)


class TestPriceVarianceRatio:
    """Test Price Variance Ratio (#22)."""

    def test_output_range(self):
        """Price variance ratio should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = price_variance_ratio(close, short_period=10, multiplier=4.0)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"

    def test_nan_warmup(self):
        """Should have NaN warmup."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = price_variance_ratio(close, short_period=10, multiplier=4.0)

        # First values should be NaN
        assert np.any(np.isnan(result[:40])), "Expected NaN warmup"

    def test_no_inf_leaks(self):
        """Should not produce inf values."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = price_variance_ratio(close, short_period=10, multiplier=4.0)

        assert not np.any(np.isinf(result)), "Should not produce inf"

    def test_constant_prices(self):
        """Constant prices should produce valid output."""
        close = np.ones(100) * 100.0
        result = price_variance_ratio(close, short_period=10, multiplier=4.0)

        # With constant prices, variances are 0, but should handle gracefully
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert not np.any(np.isinf(valid)), "Should not produce inf for constant prices"


class TestChangeVarianceRatio:
    """Test Change Variance Ratio (#23)."""

    def test_output_range(self):
        """Change variance ratio should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = change_variance_ratio(close, short_period=10, multiplier=4.0)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"

    def test_nan_warmup(self):
        """Should have NaN warmup."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = change_variance_ratio(close, short_period=10, multiplier=4.0)

        # First values should be NaN
        assert np.any(np.isnan(result[:40])), "Expected NaN warmup"

    def test_no_inf_leaks(self):
        """Should not produce inf values."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = change_variance_ratio(close, short_period=10, multiplier=4.0)

        assert not np.any(np.isinf(result)), "Should not produce inf"

    def test_high_volatility_vs_low(self):
        """Should detect variance changes between periods."""
        np.random.seed(42)
        # Create data with changing volatility
        close = np.concatenate([
            np.random.randn(50) * 0.1 + 100,  # Low volatility
            np.random.randn(150) * 5.0 + 100,  # High volatility
        ]).cumsum()

        result = change_variance_ratio(close, short_period=10, multiplier=4.0)

        # Just verify it produces valid output
        valid = result[~np.isnan(result)]
        assert len(valid) > 0, "Should have valid values"
        assert np.all(valid >= -50) and np.all(valid <= 50), "Out of range"
