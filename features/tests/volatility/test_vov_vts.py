"""
Tests for volatility_of_volatility, vov_normalized, volatility_term_structure.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volatility import (
    volatility_of_volatility,
    vov_normalized,
    volatility_term_structure,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlc(n=100, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return (
        pd.Series(high, index=idx, name="high"),
        pd.Series(low, index=idx, name="low"),
        pd.Series(close, index=idx, name="close"),
    )


def _make_ohlc_np(n=100, seed=42):
    high, low, close = _make_ohlc(n, seed)
    return high.values, low.values, close.values


# ─── TestVolatilityOfVolatility ───────────────────────────────────────────────

class TestVolatilityOfVolatility:

    def test_returns_series_for_series_input(self):
        high, low, _ = _make_ohlc()
        result = volatility_of_volatility(high, low, window=10, vov_window=10)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        high, low, _ = _make_ohlc_np()
        result = volatility_of_volatility(high, low, window=10, vov_window=10)
        assert isinstance(result, np.ndarray)

    def test_early_values_are_nan(self):
        high, low, _ = _make_ohlc(n=50)
        window, vov_window = 10, 10
        result = volatility_of_volatility(high, low, window=window, vov_window=vov_window)
        # Need window bars for Parkinson, then vov_window bars of Parkinson → warmup = window + vov_window - 2
        warmup = window + vov_window - 2
        assert np.all(np.isnan(result.values[:warmup]))
        assert np.all(np.isfinite(result.values[warmup:]))

    def test_non_negative_values(self):
        """VoV is a std dev — must be non-negative."""
        high, low, _ = _make_ohlc()
        result = volatility_of_volatility(high, low)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_higher_in_volatile_regime(self):
        """VoV should be higher when underlying vol is changing rapidly."""
        n = 100
        # Calm regime: constant range bars
        high_calm = np.full(n, 101.0)
        low_calm = np.full(n, 99.0)
        vov_calm = volatility_of_volatility(high_calm, low_calm, window=5, vov_window=10)

        # Volatile regime: alternating wide/narrow bars
        high_vol = np.array([101.0 if i % 2 == 0 else 110.0 for i in range(n)], dtype=float)
        low_vol = np.array([99.0 if i % 2 == 0 else 90.0 for i in range(n)], dtype=float)
        vov_volatile = volatility_of_volatility(high_vol, low_vol, window=5, vov_window=10)

        # Last valid values: volatile should have higher VoV
        assert vov_volatile[-1] > vov_calm[-1]

    def test_name_matches_windows(self):
        high, low, _ = _make_ohlc()
        result = volatility_of_volatility(high, low, window=15, vov_window=8)
        assert result.name == "vov_15_8"


# ─── TestVovNormalized ────────────────────────────────────────────────────────

class TestVovNormalized:

    def test_returns_series_for_series_input(self):
        high, low, _ = _make_ohlc()
        result = vov_normalized(high, low)
        assert isinstance(result, pd.Series)

    def test_early_values_are_nan(self):
        high, low, _ = _make_ohlc(n=60)
        result = vov_normalized(high, low, window=10, vov_window=10)
        warmup = 10 + 10 - 2
        assert np.all(np.isnan(result.values[:warmup]))

    def test_constant_vol_gives_zero(self):
        """When Parkinson vol is constant, VoV = 0 → normalized VoV = 0."""
        n = 60
        # Identical bars → identical Parkinson vol every period
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        result = vov_normalized(high, low, window=5, vov_window=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_normalized_proportional_to_raw(self):
        """vov_norm = vov / mean_vol: same sign, positive correlation with raw VoV."""
        high, low, _ = _make_ohlc()
        raw = volatility_of_volatility(high, low, window=10, vov_window=10)
        norm = vov_normalized(high, low, window=10, vov_window=10)
        valid_raw = raw.dropna()
        valid_norm = norm.dropna()
        # Both must be non-negative
        assert (valid_norm >= 0).all()
        # Correlation between raw and normalized must be positive (same signal direction)
        assert np.corrcoef(valid_raw.values, valid_norm.values)[0, 1] > 0.99

    def test_name_matches_windows(self):
        high, low, _ = _make_ohlc()
        result = vov_normalized(high, low, window=12, vov_window=6)
        assert result.name == "vov_norm_12_6"


# ─── TestVolatilityTermStructure ──────────────────────────────────────────────

class TestVolatilityTermStructure:

    def test_returns_series_for_series_input(self):
        high, low, _ = _make_ohlc()
        result = volatility_term_structure(high, low)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        high, low, _ = _make_ohlc_np()
        result = volatility_term_structure(high, low, short_window=5, long_window=20)
        assert isinstance(result, np.ndarray)

    def test_early_values_are_nan(self):
        high, low, _ = _make_ohlc(n=50)
        result = volatility_term_structure(high, low, short_window=5, long_window=20)
        # long_window - 1 = 19 NaN values
        assert np.all(np.isnan(result.values[:19]))
        assert np.all(np.isfinite(result.values[19:]))

    def test_equal_windows_gives_one(self):
        """When short_window == long_window, ratio must be 1.0."""
        high, low, _ = _make_ohlc()
        result = volatility_term_structure(high, low, short_window=10, long_window=10)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, rtol=1e-10)

    def test_contango_in_calm_market(self):
        """
        In a calm market with uniform H-L range, short_vol == long_vol → VTS ≈ 1.
        """
        n = 100
        # Uniform H-L range throughout → both windows see same vol
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        result = volatility_term_structure(high, low, short_window=5, long_window=20)
        # Uniform bars → Parkinson vol identical in all windows → ratio = 1
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 1.0, rtol=0.01)

    def test_backwardation_spike(self):
        """A sudden range expansion makes short-term > long-term → VTS > 1."""
        n = 80
        # Start with narrow range
        high = np.full(n, 101.0, dtype=float)
        low = np.full(n, 99.0, dtype=float)
        # Last 10 bars: much wider range
        high[-10:] = 110.0
        low[-10:] = 90.0
        result = volatility_term_structure(high, low, short_window=5, long_window=20)
        # At the end, short-term vol captures the spike; long-term is smoothed
        assert result[-1] > 1.0

    def test_non_negative_values(self):
        high, low, _ = _make_ohlc()
        result = volatility_term_structure(high, low)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_name_matches_windows(self):
        high, low, _ = _make_ohlc()
        result = volatility_term_structure(high, low, short_window=3, long_window=15)
        assert result.name == "vts_3_15"