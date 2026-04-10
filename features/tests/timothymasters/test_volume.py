import numpy as np
import pytest
from okmich_quant_features.timothymasters.volume import (
    intraday_intensity,
    money_flow,
    price_volume_fit,
    vwma_ratio,
    normalized_obv,
    delta_obv,
    normalized_pvi,
    normalized_nvi,
    volume_momentum,
)


class TestIntradayIntensity:
    """Test Intraday Intensity (#24)."""

    def test_output_range(self):
        """Intraday intensity should be in reasonable range."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        volume = np.abs(np.random.randn(200)) * 1000 + 1000

        result = intraday_intensity(high, low, close, volume, period=14)

        valid = result[~np.isnan(result)]
        # Should not have inf
        assert not np.any(np.isinf(valid)), "Should not produce inf"

    def test_nan_warmup(self):
        """Should have NaN warmup."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        high = close + 1
        low = close - 1
        volume = np.ones(100) * 1000

        result = intraday_intensity(high, low, close, volume, period=14)

        assert np.all(np.isnan(result[:13])), "Expected NaN warmup"


class TestMoneyFlow:
    """Test Money Flow (#25)."""

    def test_output_range(self):
        """Money flow should be in range [-100, 100]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        volume = np.abs(np.random.randn(200)) * 1000 + 1000

        result = money_flow(high, low, close, volume, period=14)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100), f"Min {valid.min()} below -100"
        assert np.all(valid <= 100), f"Max {valid.max()} above 100"


class TestNormalizedOBV:
    """Test Normalized OBV (#28)."""

    def test_rising_prices_positive_obv(self):
        """Rising prices should produce positive OBV."""
        close = np.arange(1, 101, dtype=np.float64)  # Monotone rising
        volume = np.ones(100) * 1000

        result = normalized_obv(close, volume, period=20)

        valid = result[~np.isnan(result)]
        # Should be predominantly positive
        assert np.mean(valid) > 0, f"Expected positive OBV for rising prices, got {np.mean(valid)}"

    def test_falling_prices_negative_obv(self):
        """Falling prices should produce negative OBV."""
        close = np.arange(100, 0, -1, dtype=np.float64)  # Monotone falling
        volume = np.ones(100) * 1000

        result = normalized_obv(close, volume, period=20)

        valid = result[~np.isnan(result)]
        # Should be predominantly negative
        assert np.mean(valid) < 0, f"Expected negative OBV for falling prices, got {np.mean(valid)}"

    def test_output_range(self):
        """Normalized OBV should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        volume = np.abs(np.random.randn(200)) * 1000 + 1000

        result = normalized_obv(close, volume, period=20)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestVolumeMomentum:
    """Test Volume Momentum (#32)."""

    def test_output_range(self):
        """Volume momentum should be in range [-50, 50]."""
        np.random.seed(42)
        volume = np.abs(np.random.randn(200)) * 1000 + 1000

        result = volume_momentum(volume, short_period=10, multiplier=4.0)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"

    def test_increasing_volume_positive_momentum(self):
        """Steadily increasing volume should produce positive momentum."""
        volume = np.arange(1, 101, dtype=np.float64) * 100  # Increasing

        result = volume_momentum(volume, short_period=10, multiplier=4.0)

        valid = result[~np.isnan(result)]
        # Should be predominantly positive
        assert np.mean(valid) > 0, f"Expected positive momentum for increasing volume, got {np.mean(valid)}"


class TestPriceVolumeFit:
    """Test Price Volume Fit (#26)."""

    def test_output_range(self):
        """Price volume fit should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        volume = np.abs(np.random.randn(200)) * 1000 + 1000
        result = price_volume_fit(close, volume, period=50)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestVWMARatio:
    """Test VWMA Ratio (#27)."""

    def test_output_range(self):
        """VWMA ratio should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        volume = np.abs(np.random.randn(200)) * 1000 + 1000
        result = vwma_ratio(close, volume, period=20)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestDeltaOBV:
    """Test Delta OBV (#29)."""

    def test_output_range(self):
        """Delta OBV should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        volume = np.abs(np.random.randn(200)) * 1000 + 1000
        result = delta_obv(close, volume, period=20, delta_period=5)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestNormalizedPVI:
    """Test Normalized PVI (#30)."""

    def test_output_range(self):
        """Normalized PVI should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        volume = np.abs(np.random.randn(200)) * 1000 + 1000
        result = normalized_pvi(close, volume, period=100)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestNormalizedNVI:
    """Test Normalized NVI (#31)."""

    def test_output_range(self):
        """Normalized NVI should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        volume = np.abs(np.random.randn(200)) * 1000 + 1000
        result = normalized_nvi(close, volume, period=100)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes
# ---------------------------------------------------------------------------

class TestIntradayIntensityCorrectnessFixes:
    """Regression tests for the intraday intensity EMA fix (Fix #2 & #5)."""

    def _make_ohlcv(self, n: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        close = np.maximum(100.0 + rng.standard_normal(n).cumsum(), 1.0)
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        volume = rng.uniform(500, 2000, n)
        return high, low, close, volume

    # -- Fix #2: incremental EMA correctness ---------------------------------

    def test_smooth_no_inf(self):
        """EMA-smoothed path should produce finite values."""
        high, low, close, volume = self._make_ohlcv(300)
        result = intraday_intensity(high, low, close, volume, period=14, smooth_period=10)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid)), "EMA smoothed path produced inf"

    def test_smooth_equals_no_smooth_when_period_1(self):
        """smooth_period=1 is effectively disabled — should match smooth_period=0."""
        high, low, close, volume = self._make_ohlcv(200)
        r0 = intraday_intensity(high, low, close, volume, period=14, smooth_period=0)
        r1 = intraday_intensity(high, low, close, volume, period=14, smooth_period=1)
        mask = ~np.isnan(r0) & ~np.isnan(r1)
        np.testing.assert_array_equal(r0[mask], r1[mask])

    def test_smooth_changes_output(self):
        """Enabling EMA smoothing should change the output."""
        high, low, close, volume = self._make_ohlcv(200)
        r0 = intraday_intensity(high, low, close, volume, period=14, smooth_period=0)
        rs = intraday_intensity(high, low, close, volume, period=14, smooth_period=5)
        mask = ~np.isnan(r0) & ~np.isnan(rs)
        assert not np.allclose(r0[mask], rs[mask]), "Smoothing should change values"

    def test_ema_path_deterministic(self):
        """Same call twice returns identical results (no stale global state)."""
        high, low, close, volume = self._make_ohlcv(200)
        r1 = intraday_intensity(high, low, close, volume, period=14, smooth_period=10)
        r2 = intraday_intensity(high, low, close, volume, period=14, smooth_period=10)
        np.testing.assert_array_equal(r1, r2)

    # -- Fix #5: parameter guards --------------------------------------------

    def test_period_zero_raises(self):
        high, low, close, volume = self._make_ohlcv(50)
        with pytest.raises(ValueError, match="period must be >= 1"):
            intraday_intensity(high, low, close, volume, period=0)

    def test_smooth_period_negative_raises(self):
        high, low, close, volume = self._make_ohlcv(50)
        with pytest.raises(ValueError, match="smooth_period must be >= 0"):
            intraday_intensity(high, low, close, volume, period=14, smooth_period=-1)

    def test_money_flow_period_zero_raises(self):
        high, low, close, volume = self._make_ohlcv(50)
        with pytest.raises(ValueError, match="period must be >= 1"):
            money_flow(high, low, close, volume, period=0)
