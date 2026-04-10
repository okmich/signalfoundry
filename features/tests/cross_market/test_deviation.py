"""Tests for cross_market.deviation — indicator #3."""

import math

import numpy as np
import pytest

from okmich_quant_features.timothymasters.cross_market import deviation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_pair():
    rng = np.random.default_rng(42)
    n = 200
    base = 100.0 + rng.standard_normal(n).cumsum()
    close1 = base * (1.0 + rng.standard_normal(n) * 0.02)
    close2 = base * (1.0 + rng.standard_normal(n) * 0.02)
    return close1, close2


# ---------------------------------------------------------------------------
# deviation
# ---------------------------------------------------------------------------

class TestDeviation:

    def test_output_shape(self, random_pair):
        c1, c2 = random_pair
        out = deviation(c1, c2, period=20)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_pair):
        c1, c2 = random_pair
        out = deviation(c1, c2, period=20)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_pair):
        c1, c2 = random_pair
        period = 20
        out = deviation(c1, c2, period=period)
        assert np.all(np.isnan(out[:period - 1]))
        assert not np.isnan(out[period - 1])

    def test_output_range_approx(self, random_pair):
        c1, c2 = random_pair
        out = deviation(c1, c2, period=20)
        valid = out[~np.isnan(out)]
        # CDF compression keeps output in (-50, 50)
        assert np.all(valid > -50.0)
        assert np.all(valid < 50.0)

    def test_no_inf_values(self, random_pair):
        c1, c2 = random_pair
        out = deviation(c1, c2, period=20)
        assert not np.any(np.isinf(out))

    def test_smoothing_changes_values(self, random_pair):
        """smooth_period > 1 must change values vs no smoothing."""
        c1, c2 = random_pair
        out_raw = deviation(c1, c2, period=20, smooth_period=0)
        out_smooth = deviation(c1, c2, period=20, smooth_period=5)
        # First valid bar should be identical (smoothing starts from second)
        assert out_raw[19] == out_smooth[19]
        # Subsequent bars should differ
        assert not np.allclose(out_raw[20:], out_smooth[20:], equal_nan=True)

    def test_smoothing_first_bar_unchanged(self, random_pair):
        """C++ leaves the first valid bar unsmoothed."""
        c1, c2 = random_pair
        out_raw = deviation(c1, c2, period=20, smooth_period=0)
        out_smooth = deviation(c1, c2, period=20, smooth_period=3)
        assert out_raw[19] == out_smooth[19]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            deviation(np.ones(10), np.ones(11))

    def test_large_positive_deviation(self):
        """When market 1 is far above the predicted relationship, output > 0."""
        rng = np.random.default_rng(9)
        n = 100
        c2 = np.abs(100.0 + rng.standard_normal(n).cumsum()) + 1.0
        # close1 follows close2 closely for 80 bars, then spikes up
        c1 = c2.copy()
        c1[80:] *= 1.5   # sudden large deviation upward
        out = deviation(c1, c2, period=20)
        # The bar right after the spike (bar 80+) should be positive
        assert out[80] > 0.0

    def test_large_negative_deviation(self):
        """When market 1 is far below the predicted relationship, output < 0."""
        rng = np.random.default_rng(11)
        n = 100
        c2 = np.abs(100.0 + rng.standard_normal(n).cumsum()) + 1.0
        c1 = c2.copy()
        c1[80:] *= 0.5   # sudden large drop
        out = deviation(c1, c2, period=20)
        assert out[80] < 0.0
