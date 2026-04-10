"""
Unit tests for microstructure primitives.

Tests cover edge cases, NaN handling, range validation, and behavioral properties
of the four core primitive functions: beta_clv, buy_sell_volume, typical_price,
and normalized_spread.
"""

import numpy as np
import pytest
from okmich_quant_features.microstructure import (
    beta_clv,
    buy_sell_volume,
    typical_price,
    normalized_spread,
)


class TestBetaCLV:
    """Test Close Location Value (β) computation."""

    def test_close_at_high_gives_one(self):
        """Close at high → β = 1.0 (maximum bullish)."""
        high = np.array([100.0, 105.0, 110.0])
        low = np.array([95.0, 100.0, 105.0])
        close = high.copy()  # Close at high

        beta = beta_clv(high, low, close)

        assert np.allclose(beta, 1.0), f"Expected 1.0, got {beta}"

    def test_close_at_low_gives_zero(self):
        """Close at low → β = 0.0 (maximum bearish)."""
        high = np.array([100.0, 105.0, 110.0])
        low = np.array([95.0, 100.0, 105.0])
        close = low.copy()  # Close at low

        beta = beta_clv(high, low, close)

        assert np.allclose(beta, 0.0), f"Expected 0.0, got {beta}"

    def test_close_at_midpoint_gives_half(self):
        """Close at midpoint → β = 0.5 (neutral)."""
        high = np.array([100.0, 110.0, 120.0])
        low = np.array([90.0, 100.0, 110.0])
        close = (high + low) / 2  # Midpoint

        beta = beta_clv(high, low, close)

        assert np.allclose(beta, 0.5, atol=1e-10), f"Expected 0.5, got {beta}"

    def test_doji_bar_gives_half(self):
        """Doji bar (high == low) → β = 0.5 (neutral by convention)."""
        high = np.array([100.0, 100.0, 100.0])
        low = np.array([100.0, 100.0, 100.0])
        close = np.array([100.0, 100.0, 100.0])

        beta = beta_clv(high, low, close)

        assert np.allclose(beta, 0.5), f"Expected 0.5 for doji, got {beta}"

    def test_close_above_high_clamps_to_one(self):
        """Close > high (data error) → clamp to 1.0."""
        high = np.array([100.0])
        low = np.array([95.0])
        close = np.array([105.0])  # Above high

        beta = beta_clv(high, low, close)

        assert np.allclose(beta, 1.0), f"Expected clamped to 1.0, got {beta}"

    def test_close_below_low_clamps_to_zero(self):
        """Close < low (data error) → clamp to 0.0."""
        high = np.array([100.0])
        low = np.array([95.0])
        close = np.array([90.0])  # Below low

        beta = beta_clv(high, low, close)

        assert np.allclose(beta, 0.0), f"Expected clamped to 0.0, got {beta}"

    def test_output_range(self):
        """Beta should always be in [0, 1]."""
        np.random.seed(42)
        n = 1000
        high = np.random.uniform(100, 110, n)
        low = np.random.uniform(90, 100, n)
        close = np.random.uniform(low, high)  # Close between low and high

        beta = beta_clv(high, low, close)

        assert np.all(beta >= 0.0), f"Min beta {beta.min()} below 0"
        assert np.all(beta <= 1.0), f"Max beta {beta.max()} above 1"

    def test_no_nan_for_valid_input(self):
        """Valid input should not produce NaN."""
        high = np.array([105.0, 110.0, 115.0])
        low = np.array([100.0, 105.0, 110.0])
        close = np.array([103.0, 107.0, 113.0])

        beta = beta_clv(high, low, close)

        assert not np.any(np.isnan(beta)), "Should not produce NaN for valid input"

    def test_no_inf(self):
        """Should not produce inf values."""
        np.random.seed(42)
        high = np.random.uniform(100, 110, 100)
        low = np.random.uniform(90, 100, 100)
        close = np.random.uniform(low, high)

        beta = beta_clv(high, low, close)

        assert not np.any(np.isinf(beta)), "Should not produce inf values"

    def test_bullish_bar_high_beta(self):
        """Bullish bar (close near high) → β > 0.7."""
        high = np.array([110.0])
        low = np.array([100.0])
        close = np.array([109.0])  # Very close to high

        beta = beta_clv(high, low, close)

        assert beta[0] > 0.7, f"Expected β > 0.7 for bullish bar, got {beta[0]}"

    def test_bearish_bar_low_beta(self):
        """Bearish bar (close near low) → β < 0.3."""
        high = np.array([110.0])
        low = np.array([100.0])
        close = np.array([101.0])  # Very close to low

        beta = beta_clv(high, low, close)

        assert beta[0] < 0.3, f"Expected β < 0.3 for bearish bar, got {beta[0]}"


class TestBuySellVolume:
    """Test buy/sell volume split using beta."""

    def test_close_at_high_all_buy_volume(self):
        """Close at high → all volume is buy volume."""
        high = np.array([100.0, 105.0])
        low = np.array([95.0, 100.0])
        close = high.copy()
        volume = np.array([1000.0, 2000.0])

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        assert np.allclose(buy_vol, volume), "Expected all volume as buy"
        assert np.allclose(sell_vol, 0.0), "Expected zero sell volume"

    def test_close_at_low_all_sell_volume(self):
        """Close at low → all volume is sell volume."""
        high = np.array([100.0, 105.0])
        low = np.array([95.0, 100.0])
        close = low.copy()
        volume = np.array([1000.0, 2000.0])

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        assert np.allclose(buy_vol, 0.0), "Expected zero buy volume"
        assert np.allclose(sell_vol, volume), "Expected all volume as sell"

    def test_close_at_midpoint_equal_split(self):
        """Close at midpoint → 50/50 buy/sell split."""
        high = np.array([100.0, 110.0])
        low = np.array([90.0, 100.0])
        close = (high + low) / 2
        volume = np.array([1000.0, 2000.0])

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        expected = volume / 2
        assert np.allclose(buy_vol, expected, atol=1e-10), f"Expected {expected}, got {buy_vol}"
        assert np.allclose(sell_vol, expected, atol=1e-10), f"Expected {expected}, got {sell_vol}"

    def test_buy_plus_sell_equals_total(self):
        """Buy volume + sell volume should equal total volume."""
        np.random.seed(42)
        n = 100
        high = np.random.uniform(100, 110, n)
        low = np.random.uniform(90, 100, n)
        close = np.random.uniform(low, high)
        volume = np.random.uniform(1000, 5000, n)

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        total = buy_vol + sell_vol
        assert np.allclose(total, volume, atol=1e-6), "Buy + sell should equal total volume"

    def test_doji_bar_50_50_split(self):
        """Doji bar → 50/50 split."""
        high = np.array([100.0])
        low = np.array([100.0])
        close = np.array([100.0])
        volume = np.array([1000.0])

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        assert np.allclose(buy_vol, 500.0), "Expected 500 buy volume"
        assert np.allclose(sell_vol, 500.0), "Expected 500 sell volume"

    def test_no_negative_volumes(self):
        """Buy and sell volumes should never be negative."""
        np.random.seed(42)
        n = 100
        high = np.random.uniform(100, 110, n)
        low = np.random.uniform(90, 100, n)
        close = np.random.uniform(low, high)
        volume = np.random.uniform(1000, 5000, n)

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        assert np.all(buy_vol >= 0.0), "Buy volume should be non-negative"
        assert np.all(sell_vol >= 0.0), "Sell volume should be non-negative"

    def test_nan_volume_propagates(self):
        """NaN in volume should produce NaN in buy/sell."""
        high = np.array([100.0, 105.0])
        low = np.array([95.0, 100.0])
        close = np.array([98.0, 103.0])
        volume = np.array([1000.0, np.nan])

        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)

        assert not np.isnan(buy_vol[0]), "First bar should be valid"
        assert np.isnan(buy_vol[1]), "Second bar should be NaN"
        assert np.isnan(sell_vol[1]), "Second bar should be NaN"


class TestTypicalPrice:
    """Test typical price calculation."""

    def test_typical_price_formula(self):
        """Typical price = (H + L + C) / 3."""
        high = np.array([105.0, 110.0, 115.0])
        low = np.array([100.0, 105.0, 110.0])
        close = np.array([103.0, 107.0, 113.0])

        tp = typical_price(high, low, close)
        expected = (high + low + close) / 3

        assert np.allclose(tp, expected, atol=1e-10), f"Expected {expected}, got {tp}"

    def test_typical_price_equals_close_for_doji(self):
        """For doji bars (H=L=C), typical price = close."""
        high = np.array([100.0, 100.0])
        low = np.array([100.0, 100.0])
        close = np.array([100.0, 100.0])

        tp = typical_price(high, low, close)

        assert np.allclose(tp, close), "Typical price should equal close for doji"

    def test_typical_price_between_high_and_low(self):
        """Typical price should be between high and low."""
        np.random.seed(42)
        n = 100
        high = np.random.uniform(100, 110, n)
        low = np.random.uniform(90, 100, n)
        close = np.random.uniform(low, high)

        tp = typical_price(high, low, close)

        assert np.all(tp >= low), "Typical price should be >= low"
        assert np.all(tp <= high), "Typical price should be <= high"

    def test_no_nan_for_valid_input(self):
        """Valid input should not produce NaN."""
        high = np.array([105.0, 110.0])
        low = np.array([100.0, 105.0])
        close = np.array([103.0, 107.0])

        tp = typical_price(high, low, close)

        assert not np.any(np.isnan(tp)), "Should not produce NaN for valid input"

    def test_nan_propagates(self):
        """NaN in input should produce NaN in output."""
        high = np.array([105.0, np.nan])
        low = np.array([100.0, 100.0])
        close = np.array([103.0, 105.0])

        tp = typical_price(high, low, close)

        assert not np.isnan(tp[0]), "First value should be valid"
        assert np.isnan(tp[1]), "Second value should be NaN"


class TestNormalizedSpread:
    """Test normalized spread calculation."""

    def test_normalized_spread_formula(self):
        """Normalized spread = spread / mid_price."""
        spread = np.array([0.05, 0.10, 0.15])
        mid_price = np.array([100.0, 150.0, 200.0])

        norm = normalized_spread(spread, mid_price)
        expected = spread / mid_price

        assert np.allclose(norm, expected, atol=1e-10), f"Expected {expected}, got {norm}"

    def test_zero_spread_gives_zero(self):
        """Zero spread → normalized spread = 0."""
        spread = np.array([0.0, 0.0])
        mid_price = np.array([100.0, 150.0])

        norm = normalized_spread(spread, mid_price)

        assert np.allclose(norm, 0.0), "Zero spread should give zero normalized spread"

    def test_zero_mid_price_gives_nan(self):
        """Zero mid-price → NaN (division by zero guard)."""
        spread = np.array([0.05])
        mid_price = np.array([0.0])

        norm = normalized_spread(spread, mid_price)

        assert np.isnan(norm[0]), "Zero mid-price should produce NaN"

    def test_output_range(self):
        """Normalized spread should be in [0, 1] for reasonable spreads."""
        # Reasonable spreads: 0.01% to 1% of mid-price
        spread = np.array([0.01, 0.10, 1.0])
        mid_price = np.array([100.0, 100.0, 100.0])

        norm = normalized_spread(spread, mid_price)

        assert np.all(norm >= 0.0), "Normalized spread should be non-negative"
        assert np.all(norm <= 1.0), "Normalized spread should be <= 1"

    def test_negative_spread_clamps_to_zero(self):
        """Negative spread (data error) → clamp to 0."""
        spread = np.array([-0.05])
        mid_price = np.array([100.0])

        norm = normalized_spread(spread, mid_price)

        assert norm[0] == 0.0, "Negative spread should be clamped to 0"

    def test_excessive_spread_clamps_to_one(self):
        """Spread > mid_price (data error) → clamp to 1."""
        spread = np.array([150.0])  # Spread > mid_price
        mid_price = np.array([100.0])

        norm = normalized_spread(spread, mid_price)

        assert norm[0] == 1.0, "Excessive spread should be clamped to 1"

    def test_typical_fx_spread(self):
        """Typical FX spread: 1-5 bps → normalized spread 0.0001-0.0005."""
        # EUR/USD with 2 pip spread at 1.0850
        spread = np.array([0.0002])  # 2 pips
        mid_price = np.array([1.0850])

        norm = normalized_spread(spread, mid_price)
        expected = 0.0002 / 1.0850  # ~0.000184 = 1.84 bps

        assert np.allclose(norm, expected, atol=1e-7), f"Expected ~0.000184, got {norm[0]}"
        assert 0.00015 < norm[0] < 0.00020, "Should be in typical FX range"

    def test_typical_stock_spread(self):
        """Typical stock spread: 5-20 bps → normalized spread 0.0005-0.002."""
        # Stock at $150 with $0.10 spread
        spread = np.array([0.10])
        mid_price = np.array([150.0])

        norm = normalized_spread(spread, mid_price)
        expected = 0.10 / 150.0  # ~0.000667 = 6.67 bps

        assert np.allclose(norm, expected, atol=1e-7), f"Expected ~0.000667, got {norm[0]}"
        assert 0.0005 < norm[0] < 0.002, "Should be in typical stock range"

    def test_no_inf(self):
        """Should not produce inf values."""
        spread = np.array([0.05, 0.10, 0.15])
        mid_price = np.array([100.0, 150.0, 200.0])

        norm = normalized_spread(spread, mid_price)

        assert not np.any(np.isinf(norm)), "Should not produce inf values"

    def test_nan_propagates(self):
        """NaN in input should produce NaN in output."""
        spread = np.array([0.05, np.nan])
        mid_price = np.array([100.0, 150.0])

        norm = normalized_spread(spread, mid_price)

        assert not np.isnan(norm[0]), "First value should be valid"
        assert np.isnan(norm[1]), "Second value should be NaN"


class TestPrimitivesIntegration:
    """Integration tests combining multiple primitives."""

    def test_vwap_workflow(self):
        """Typical workflow: beta → buy/sell → typical price for VWAP."""
        high = np.array([105.0, 110.0, 115.0])
        low = np.array([100.0, 105.0, 110.0])
        close = np.array([103.0, 107.0, 113.0])
        volume = np.array([1000.0, 1500.0, 2000.0])

        # Compute primitives
        buy_vol, sell_vol = buy_sell_volume(high, low, close, volume)
        tp = typical_price(high, low, close)

        # Check buy + sell = volume
        assert np.allclose(buy_vol + sell_vol, volume), "Buy + sell should equal total"

        # Typical price for VWAP
        vwap = np.sum(tp * volume) / np.sum(volume)
        assert 100.0 < vwap < 115.0, "VWAP should be in price range"

    def test_liquidity_workflow(self):
        """Liquidity workflow: spread → normalized spread → z-score."""
        spread = np.array([0.05, 0.10, 0.08, 0.06, 0.12])
        mid_price = np.array([100.0, 102.0, 101.0, 99.0, 103.0])

        # Normalize spread
        norm = normalized_spread(spread, mid_price)

        # Compute z-score
        mean = np.mean(norm)
        std = np.std(norm)
        z_score = (norm - mean) / std

        # Check z-score properties
        assert np.abs(np.mean(z_score)) < 0.1, "Z-score mean should be ~0"
        assert np.abs(np.std(z_score) - 1.0) < 0.1, "Z-score std should be ~1"


class TestPandasSeriesSupport:
    """Test that all primitives work with pandas Series and preserve index."""

    def test_beta_clv_with_series(self):
        """Beta CLV should work with pandas Series and preserve index."""
        import pandas as pd

        index = pd.date_range('2024-01-01', periods=3)
        high_s = pd.Series([105.0, 110.0, 115.0], index=index)
        low_s = pd.Series([100.0, 105.0, 110.0], index=index)
        close_s = pd.Series([103.0, 107.0, 113.0], index=index)

        result = beta_clv(high_s, low_s, close_s)

        # Should return Series
        assert isinstance(result, pd.Series), "Should return pandas Series"

        # Should preserve index
        assert result.index.equals(index), "Should preserve index"

        # Should have same values as numpy version
        high_arr = high_s.values
        low_arr = low_s.values
        close_arr = close_s.values
        expected = beta_clv(high_arr, low_arr, close_arr)
        assert np.allclose(result.values, expected), "Values should match numpy version"

    def test_buy_sell_volume_with_series(self):
        """Buy/sell volume should work with pandas Series and preserve index."""
        import pandas as pd

        index = pd.date_range('2024-01-01', periods=3)
        high_s = pd.Series([105.0, 110.0, 115.0], index=index)
        low_s = pd.Series([100.0, 105.0, 110.0], index=index)
        close_s = pd.Series([103.0, 107.0, 113.0], index=index)
        volume_s = pd.Series([1000.0, 1500.0, 2000.0], index=index)

        buy_s, sell_s = buy_sell_volume(high_s, low_s, close_s, volume_s)

        # Should return Series
        assert isinstance(buy_s, pd.Series), "Buy volume should be Series"
        assert isinstance(sell_s, pd.Series), "Sell volume should be Series"

        # Should preserve index
        assert buy_s.index.equals(index), "Buy volume should preserve index"
        assert sell_s.index.equals(index), "Sell volume should preserve index"

        # Should have correct names
        assert buy_s.name == 'buy_volume', "Buy volume should have name"
        assert sell_s.name == 'sell_volume', "Sell volume should have name"

        # Should sum to total volume
        assert np.allclose((buy_s + sell_s).values, volume_s.values), "Should sum to total"

    def test_typical_price_with_series(self):
        """Typical price should work with pandas Series and preserve index."""
        import pandas as pd

        index = pd.date_range('2024-01-01', periods=3)
        high_s = pd.Series([105.0, 110.0, 115.0], index=index)
        low_s = pd.Series([100.0, 105.0, 110.0], index=index)
        close_s = pd.Series([103.0, 107.0, 113.0], index=index)

        result = typical_price(high_s, low_s, close_s)

        # Should return Series
        assert isinstance(result, pd.Series), "Should return pandas Series"

        # Should preserve index
        assert result.index.equals(index), "Should preserve index"

        # Should have correct name
        assert result.name == 'typical_price', "Should have name"

        # Should have same values as numpy version
        expected = (high_s.values + low_s.values + close_s.values) / 3
        assert np.allclose(result.values, expected), "Values should match formula"

    def test_normalized_spread_with_series(self):
        """Normalized spread should work with pandas Series and preserve index."""
        import pandas as pd

        index = pd.date_range('2024-01-01', periods=3)
        spread_s = pd.Series([0.05, 0.10, 0.08], index=index)
        mid_s = pd.Series([100.0, 150.0, 125.0], index=index)

        result = normalized_spread(spread_s, mid_s)

        # Should return Series
        assert isinstance(result, pd.Series), "Should return pandas Series"

        # Should preserve index
        assert result.index.equals(index), "Should preserve index"

        # Should have correct name
        assert result.name == 'normalized_spread', "Should have name"

        # Should have same values as numpy version
        expected = spread_s.values / mid_s.values
        assert np.allclose(result.values, expected), "Values should match formula"

    def test_mixed_input_types(self):
        """Functions should handle mixed input types (Series + array)."""
        import pandas as pd

        # Create Series for some inputs, arrays for others
        index = pd.date_range('2024-01-01', periods=3)
        high_s = pd.Series([105.0, 110.0, 115.0], index=index)
        low_arr = np.array([100.0, 105.0, 110.0])
        close_s = pd.Series([103.0, 107.0, 113.0], index=index)

        # Should work and return Series (based on first argument)
        result = beta_clv(high_s, low_arr, close_s)

        assert isinstance(result, pd.Series), "Should return Series (first arg is Series)"
        assert result.index.equals(index), "Should preserve index from Series"

    def test_dataframe_workflow(self):
        """Test typical DataFrame workflow."""
        import pandas as pd

        # Create sample DataFrame
        df = pd.DataFrame({
            'high': [105.0, 110.0, 115.0],
            'low': [100.0, 105.0, 110.0],
            'close': [103.0, 107.0, 113.0],
            'volume': [1000.0, 1500.0, 2000.0]
        }, index=pd.date_range('2024-01-01', periods=3))

        # Compute features
        df['beta'] = beta_clv(df['high'], df['low'], df['close'])
        df['buy_vol'], df['sell_vol'] = buy_sell_volume(
            df['high'], df['low'], df['close'], df['volume']
        )
        df['typical_price'] = typical_price(df['high'], df['low'], df['close'])

        # All should preserve index
        assert df['beta'].index.equals(df.index), "Beta should preserve index"
        assert df['buy_vol'].index.equals(df.index), "Buy volume should preserve index"
        assert df['typical_price'].index.equals(df.index), "Typical price should preserve index"

        # Check values make sense
        assert np.all(df['beta'] >= 0.0) and np.all(df['beta'] <= 1.0), "Beta in [0,1]"
        assert np.allclose((df['buy_vol'] + df['sell_vol']).values, df['volume'].values), "Volumes sum to total"
