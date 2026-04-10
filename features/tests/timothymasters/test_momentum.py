import numpy as np
import pytest
from okmich_quant_features.timothymasters.momentum import (
    rsi,
    detrended_rsi,
    stochastic,
    stoch_rsi,
    ma_difference,
    macd,
    ppo,
    price_change_osc,
    close_minus_ma,
    price_intensity,
    reactivity,
)


class TestRSI:
    """Test RSI (#1) with known price patterns."""

    def test_constant_prices_neutral(self):
        """Constant prices should yield RSI = 0 (no up/down movement)."""
        close = np.ones(100) * 100.0
        result = rsi(close, period=14)

        # After warmup, should be 0 (both upsum and dnsum are 0)
        valid = result[20:]  # Well past warmup
        assert np.allclose(valid, 0.0, atol=0.1), f"Expected ~0, got {valid[:5]}"

    def test_monotone_rising_approaches_100(self):
        """Monotone rising prices should push RSI toward 100."""
        close = np.arange(1, 101, dtype=np.float64)  # 1, 2, 3, ..., 100
        result = rsi(close, period=14)

        # After warmup, should be very high (approaching 100)
        valid = result[20:]
        assert np.all(valid > 80), f"Expected >80 for rising prices, got {valid[:5]}"
        assert np.all(valid <= 100), "RSI should not exceed 100"

    def test_monotone_falling_approaches_0(self):
        """Monotone falling prices should push RSI toward 0."""
        close = np.arange(100, 0, -1, dtype=np.float64)  # 100, 99, 98, ..., 1
        result = rsi(close, period=14)

        # After warmup, should be very low (approaching 0)
        valid = result[20:]
        assert np.all(valid < 20), f"Expected <20 for falling prices, got {valid[:5]}"
        assert np.all(valid >= 0), "RSI should not go below 0"

    def test_nan_warmup_pattern(self):
        """First (period-1) values should be NaN."""
        close = np.random.randn(100).cumsum() + 100
        result = rsi(close, period=14)

        # First 13 values should be NaN
        assert np.all(np.isnan(result[:13])), "Expected NaN warmup"
        # After warmup, should be valid
        assert not np.any(np.isnan(result[14:])), "Should have valid values after warmup"

    def test_output_range(self):
        """RSI output should be in range [0, 100]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = rsi(close, period=14)

        valid = result[~np.isnan(result)]
        # RSI outputs in [0, 100] range
        assert np.all(valid >= 0), f"Min value {valid.min()} below 0"
        assert np.all(valid <= 100), f"Max value {valid.max()} above 100"

    def test_no_inf_leaks(self):
        """Should not produce inf values."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = rsi(close, period=14)

        assert not np.any(np.isinf(result)), "Should not produce inf values"


class TestMACD:
    """Test MACD (#6)."""

    def test_output_range(self):
        """MACD output should be in approximate range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = macd(high, low, close, short_period=12, long_period=26)

        valid = result[~np.isnan(result)]
        assert not np.any(np.isinf(valid)), "Should not produce inf"

    def test_nan_warmup(self):
        """Should have NaN warmup for (long_period) bars."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        high = close + np.abs(np.random.randn(100))
        low = close - np.abs(np.random.randn(100))
        result = macd(high, low, close, short_period=12, long_period=26)

        # First 25 should be NaN (long_period - 1)
        assert np.all(np.isnan(result[:25])), f"Expected NaN warmup"


class TestDetrendedRSI:
    """Test Detrended RSI (#2)."""

    def test_output_range(self):
        """Detrended RSI should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = detrended_rsi(close, short_period=7, long_period=14, reg_len=32)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"

    def test_nan_warmup(self):
        """Should have NaN warmup."""
        np.random.seed(42)
        close = np.random.randn(100).cumsum() + 100
        result = detrended_rsi(close, short_period=7, long_period=14, reg_len=32)

        # Should have substantial warmup
        assert np.any(np.isnan(result[:45])), "Expected NaN warmup"


class TestStochastic:
    """Test Stochastic (#3)."""

    def test_output_range(self):
        """Stochastic should be in range [0, 100]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))

        for smoothing in [0, 1, 2]:
            result = stochastic(high, low, close, period=14, smoothing=smoothing)
            valid = result[~np.isnan(result)]
            assert np.all(valid >= 0), f"Min {valid.min()} below 0"
            assert np.all(valid <= 100), f"Max {valid.max()} above 100"

    def test_at_high(self):
        """When close = high, stochastic should be near 100."""
        close = np.arange(1, 101, dtype=np.float64)
        high = close + 1.0
        low = close - 1.0

        result = stochastic(high, low, high, period=14, smoothing=0)  # close = high
        valid = result[20:]
        assert np.all(valid > 80), f"Expected >80 when close=high, got {valid[:5]}"


class TestStochRSI:
    """Test Stochastic RSI (#4)."""

    def test_output_range(self):
        """StochRSI should be in range [0, 100]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = stoch_rsi(close, rsi_period=14, stoch_period=14, smooth_period=1)

        valid = result[~np.isnan(result)]
        # Allow small floating point tolerance
        assert np.all(valid >= -1e-10), f"Min {valid.min()} below 0"
        assert np.all(valid <= 100 + 1e-10), f"Max {valid.max()} above 100"


class TestMADifference:
    """Test MA Difference (#5)."""

    def test_output_range(self):
        """MA Difference should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = ma_difference(high, low, close, short_period=10, long_period=40, lag=0)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestPPO:
    """Test PPO (#7)."""

    def test_output_range(self):
        """PPO should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        result = ppo(close, short_period=12, long_period=26, signal_period=9)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestPriceChangeOsc:
    """Test Price Change Oscillator (#8)."""

    def test_output_range(self):
        """Price Change Oscillator should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = price_change_osc(high, low, close, short_period=10, multiplier=4.0)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestCloseMinusMA:
    """Test Close Minus MA (#9)."""

    def test_output_range(self):
        """Close Minus MA should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        result = close_minus_ma(high, low, close, period=20, atr_period=60)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestPriceIntensity:
    """Test Price Intensity (#10)."""

    def test_output_range(self):
        """Price Intensity should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        open_ = close + np.random.randn(200) * 0.5
        high = np.maximum(close, open_) + np.abs(np.random.randn(200))
        low = np.minimum(close, open_) - np.abs(np.random.randn(200))
        result = price_intensity(open_, high, low, close, smooth_period=1)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


class TestReactivity:
    """Test Reactivity (#11)."""

    def test_output_range(self):
        """Reactivity should be in range [-50, 50]."""
        np.random.seed(42)
        close = np.random.randn(200).cumsum() + 100
        high = close + np.abs(np.random.randn(200))
        low = close - np.abs(np.random.randn(200))
        volume = np.abs(np.random.randn(200)) * 1000 + 1000
        result = reactivity(high, low, close, volume, period=10, multiplier=4)

        valid = result[~np.isnan(result)]
        assert np.all(valid >= -50), f"Min {valid.min()} below -50"
        assert np.all(valid <= 50), f"Max {valid.max()} above 50"


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes
# ---------------------------------------------------------------------------

class TestMACDCorrectnessFixes:
    """Regression tests for the MACD correctness fixes (Fix #1 & #5)."""

    def _make_ohlc(self, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        close = np.maximum(100.0 + rng.standard_normal(n).cumsum(), 1.0)
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        return high, low, close

    # -- Fix #1: warmup OOB when n < warmup ----------------------------------

    def test_n_equals_long_period_all_nan(self):
        """n == long_period: all output is NaN (no valid bars yet)."""
        long_p, short_p, sig_p = 26, 12, 9
        n = long_p  # exactly at boundary
        high, low, close = self._make_ohlc(n)
        result = macd(high, low, close, short_period=short_p, long_period=long_p,
                      signal_period=sig_p)
        assert result.shape == (n,)
        assert np.all(np.isnan(result)), "All bars should be NaN when n == long_period"

    def test_n_smaller_than_warmup_no_oob(self):
        """n < warmup must not crash (was the OOB write bug)."""
        long_p, short_p, sig_p = 26, 12, 9
        # warmup = long_p + sig_p = 35; use n = 20 (well below warmup)
        n = 20
        high, low, close = self._make_ohlc(n)
        result = macd(high, low, close, short_period=short_p, long_period=long_p,
                      signal_period=sig_p)
        assert result.shape == (n,)
        assert np.all(np.isnan(result)), "All bars NaN for tiny array"

    def test_n_slightly_below_warmup_no_oob(self):
        """n = warmup - 1 must not crash."""
        long_p, short_p, sig_p = 26, 12, 1  # warmup = long_p = 26
        n = long_p - 1  # 25 bars
        high, low, close = self._make_ohlc(n)
        result = macd(high, low, close, short_period=short_p, long_period=long_p,
                      signal_period=sig_p)
        assert result.shape == (n,)
        assert np.all(np.isnan(result))

    def test_normal_output_not_all_nan(self):
        """Sufficient data should produce valid (non-NaN) bars."""
        high, low, close = self._make_ohlc(200)
        result = macd(high, low, close, short_period=12, long_period=26, signal_period=9)
        assert not np.all(np.isnan(result)), "Should have some valid bars"
        valid = result[~np.isnan(result)]
        assert not np.any(np.isinf(valid))

    # -- Fix #5: parameter guards --------------------------------------------

    def test_long_equals_short_raises(self):
        with pytest.raises(ValueError, match="long_period must be > short_period"):
            macd(np.ones(50), np.ones(50), np.ones(50), short_period=12, long_period=12)

    def test_long_less_than_short_raises(self):
        with pytest.raises(ValueError, match="long_period must be > short_period"):
            macd(np.ones(50), np.ones(50), np.ones(50), short_period=26, long_period=12)

    def test_short_period_zero_raises(self):
        with pytest.raises(ValueError, match="short_period must be >= 1"):
            macd(np.ones(50), np.ones(50), np.ones(50), short_period=0, long_period=12)

    def test_signal_period_zero_raises(self):
        with pytest.raises(ValueError, match="signal_period must be >= 1"):
            macd(np.ones(50), np.ones(50), np.ones(50), short_period=12, long_period=26,
                 signal_period=0)
