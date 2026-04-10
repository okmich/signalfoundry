import numpy as np
import pytest
from okmich_quant_features.timothymasters.information import (
    entropy,
    mutual_information,
)


class TestEntropy:
    """Test Entropy (#33) with known patterns."""

    def test_monotone_series_low_entropy(self):
        """Monotone series should have low entropy (predictable)."""
        # Perfectly monotone increasing
        close = np.arange(1, 201, dtype=np.float64)
        result = entropy(close, word_length=3, mult=10)

        # After warmup, entropy should be very low (close to 0)
        valid = result[~np.isnan(result)]
        # Kernel = 0 means perfectly predictable
        assert np.mean(valid) < 20, f"Expected low entropy for monotone, got mean {np.mean(valid)}"

    def test_alternating_series_entropy(self):
        """Alternating series should produce valid entropy."""
        # Alternating pattern: 100, 110, 100, 110, ...
        close = np.array([100.0 if i % 2 == 0 else 110.0 for i in range(200)])
        result = entropy(close, word_length=3, mult=10)

        # After warmup, should have valid entropy values
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            # Should be in valid range
            assert np.all(valid >= -50) and np.all(valid <= 50), f"Out of range: {valid.min()}, {valid.max()}"

    def test_random_walk_entropy(self):
        """Random walk should have valid entropy values."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = entropy(close, word_length=3, mult=10)

        valid = result[~np.isnan(result)]
        # Just verify output is valid and in range
        if len(valid) > 0:
            assert np.all(valid >= -50) and np.all(valid <= 50), f"Out of range: {valid.min()}, {valid.max()}"

    def test_output_range(self):
        """Entropy should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = entropy(close, word_length=3, mult=10)

        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= -50), f"Min entropy {valid.min()} below -50"
            assert np.all(valid <= 50), f"Max entropy {valid.max()} above 50"

    def test_nan_warmup(self):
        """Should have NaN warmup."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = entropy(close, word_length=3, mult=10)

        # First values should be NaN
        assert np.any(np.isnan(result[:50])), "Expected NaN warmup"
        # But should have valid values later
        assert not np.all(np.isnan(result)), "Should have some valid values"

    def test_no_inf_leaks(self):
        """Should not produce inf values."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = entropy(close, word_length=3, mult=10)

        assert not np.any(np.isinf(result)), "Should not produce inf values"


class TestMutualInformation:
    """Test Mutual Information (#34)."""

    def test_close_prices_mi(self):
        """MI should produce valid output on close prices."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        result = mutual_information(close, word_length=3, mult=10)

        # Just verify output is valid and in range
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= -50) and np.all(valid <= 50), f"Out of range: {valid.min()}, {valid.max()}"

    def test_trending_vs_random_mi(self):
        """MI should produce valid outputs for different patterns."""
        # Strong trend - more predictable
        np.random.seed(42)
        trend = np.arange(300, dtype=np.float64) * 0.5 + 100
        random = np.random.randn(300).cumsum() + 100

        result_trend = mutual_information(trend, word_length=3, mult=10)
        result_random = mutual_information(random, word_length=3, mult=10)

        valid_trend = result_trend[~np.isnan(result_trend)]
        valid_random = result_random[~np.isnan(result_random)]

        # Just check both produce valid outputs in range
        if len(valid_trend) > 0:
            assert np.all(valid_trend >= -50) and np.all(valid_trend <= 50)
        if len(valid_random) > 0:
            assert np.all(valid_random >= -50) and np.all(valid_random <= 50)

    def test_output_range(self):
        """MI should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        result = mutual_information(close, word_length=3, mult=10)

        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= -50), f"Min MI {valid.min()} below -50"
            assert np.all(valid <= 50), f"Max MI {valid.max()} above 50"

    def test_no_inf_leaks(self):
        """Should not produce inf values."""
        np.random.seed(42)
        close = np.random.randn(300).cumsum() + 100

        result = mutual_information(close, word_length=3, mult=10)

        assert not np.any(np.isinf(result)), "Should not produce inf values"
