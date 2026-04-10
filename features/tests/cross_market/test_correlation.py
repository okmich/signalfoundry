"""Tests for cross_market.correlation — indicators #1 and #2."""

import math

import numpy as np
import pytest

from okmich_quant_features.timothymasters.cross_market import correlation, delta_correlation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_pair():
    rng = np.random.default_rng(42)
    n = 200
    base = 100.0 + rng.standard_normal(n).cumsum()
    close1 = base + rng.standard_normal(n) * 0.5
    close2 = base + rng.standard_normal(n) * 0.5
    return close1, close2


# ---------------------------------------------------------------------------
# correlation
# ---------------------------------------------------------------------------

class TestCorrelation:

    def test_output_shape(self, random_pair):
        c1, c2 = random_pair
        out = correlation(c1, c2, period=20)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_pair):
        c1, c2 = random_pair
        out = correlation(c1, c2, period=20)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_pair):
        c1, c2 = random_pair
        period = 20
        out = correlation(c1, c2, period=period)
        assert np.all(np.isnan(out[:period - 1]))
        assert not np.isnan(out[period - 1])

    def test_identical_series_gives_plus50(self):
        """Perfectly correlated series → rho = 1 → output = 50."""
        rng = np.random.default_rng(7)
        c = 100.0 + rng.standard_normal(100).cumsum()
        out = correlation(c, c, period=20)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 50.0, atol=1e-10)

    def test_negatively_correlated_series(self):
        """Perfectly anti-correlated (one is reversed) → rho = −1 → output = −50."""
        n = 100
        c1 = np.arange(1.0, n + 1)    # strictly increasing
        c2 = np.arange(n, 0.0, -1.0)  # strictly decreasing
        out = correlation(c1, c2, period=20)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, -50.0, atol=1e-10)

    def test_output_range(self, random_pair):
        c1, c2 = random_pair
        out = correlation(c1, c2, period=20)
        valid = out[~np.isnan(out)]
        assert np.all(valid >= -50.0 - 1e-9)
        assert np.all(valid <= 50.0 + 1e-9)

    def test_no_inf_values(self, random_pair):
        c1, c2 = random_pair
        out = correlation(c1, c2, period=20)
        assert not np.any(np.isinf(out))

    def test_symmetry(self, random_pair):
        """correlation(c1, c2) should equal correlation(c2, c1) (Spearman is symmetric)."""
        c1, c2 = random_pair
        out12 = correlation(c1, c2, period=30)
        out21 = correlation(c2, c1, period=30)
        np.testing.assert_allclose(out12, out21, atol=1e-10, equal_nan=True)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            correlation(np.ones(10), np.ones(11), period=5)

    def test_default_period_63(self):
        rng = np.random.default_rng(1)
        c = 100.0 + rng.standard_normal(200).cumsum()
        out = correlation(c, c)
        assert np.isnan(out[61])
        assert not np.isnan(out[62])


# ---------------------------------------------------------------------------
# delta_correlation
# ---------------------------------------------------------------------------

class TestDeltaCorrelation:

    def test_output_shape(self, random_pair):
        c1, c2 = random_pair
        out = delta_correlation(c1, c2, period=20, delta_period=10)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_pair):
        c1, c2 = random_pair
        out = delta_correlation(c1, c2, period=20, delta_period=10)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_pair):
        c1, c2 = random_pair
        period, delta = 20, 10
        out = delta_correlation(c1, c2, period=period, delta_period=delta)
        front_bad = period - 1 + delta
        assert np.all(np.isnan(out[:front_bad]))
        assert not np.isnan(out[front_bad])

    def test_constant_correlation_gives_zero_delta(self):
        """When correlation is constant, delta should be zero."""
        # Identical series → corr = 50 always → delta = 0
        rng = np.random.default_rng(3)
        c = 100.0 + rng.standard_normal(200).cumsum()
        out = delta_correlation(c, c, period=20, delta_period=10)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_no_inf_values(self, random_pair):
        c1, c2 = random_pair
        out = delta_correlation(c1, c2, period=20, delta_period=10)
        assert not np.any(np.isinf(out))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            delta_correlation(np.ones(10), np.ones(11))
