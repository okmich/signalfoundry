"""Tests for cross_market.trend_diff — indicators #6 and #7."""

import numpy as np
import pytest

from okmich_quant_features.timothymasters.cross_market import trend_diff, cmma_diff


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_ohlc_pair():
    """Two random OHLC series sharing a common base."""
    rng = np.random.default_rng(42)
    n = 200
    base = 100.0 + rng.standard_normal(n).cumsum()

    close1 = base + rng.standard_normal(n) * 0.5
    close2 = base + rng.standard_normal(n) * 0.5

    def make_ohlc(close):
        noise = np.abs(rng.standard_normal(n)) * 0.3
        high = close + noise
        low = close - noise
        return high, low, close

    h1, l1, c1 = make_ohlc(close1)
    h2, l2, c2 = make_ohlc(close2)
    return h1, l1, c1, h2, l2, c2


# ---------------------------------------------------------------------------
# trend_diff
# ---------------------------------------------------------------------------

class TestTrendDiff:

    def test_output_shape(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = trend_diff(h1, l1, c1, h2, l2, c2, period=20)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = trend_diff(h1, l1, c1, h2, l2, c2, period=20)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_ohlc_pair):
        """Warmup should propagate from the underlying linear_trend calls."""
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        period = 20
        atr_period = 60
        out = trend_diff(h1, l1, c1, h2, l2, c2, period=period, atr_period=atr_period)
        # ATR with length=atr_period is first valid at index atr_period (needs bar 1..atr_period)
        # Legendre window needs period-1 bars.  First valid = max(period-1, atr_period).
        first_valid = max(period - 1, atr_period)
        assert np.all(np.isnan(out[:first_valid]))
        assert not np.isnan(out[first_valid])

    def test_identical_markets_gives_zero(self, random_ohlc_pair):
        """If both markets are identical, trend_diff should be 0."""
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = trend_diff(h1, l1, c1, h1, l1, c1, period=20, atr_period=60)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_output_range_approx(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = trend_diff(h1, l1, c1, h2, l2, c2, period=20)
        valid = out[~np.isnan(out)]
        assert np.all(valid > -200.0)
        assert np.all(valid < 200.0)

    def test_no_inf_values(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = trend_diff(h1, l1, c1, h2, l2, c2, period=20)
        assert not np.any(np.isinf(out))

    def test_antisymmetry(self, random_ohlc_pair):
        """Swapping market1 and market2 should negate the output."""
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out12 = trend_diff(h1, l1, c1, h2, l2, c2, period=20)
        out21 = trend_diff(h2, l2, c2, h1, l1, c1, period=20)
        np.testing.assert_allclose(out12, -out21, atol=1e-10, equal_nan=True)

    def test_trending_market_positive(self):
        """Market 1 trending up strongly, market 2 flat → positive trend_diff."""
        n = 200
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(n) * 0.1

        # Market 1: strong uptrend
        c1 = np.linspace(100, 150, n) + noise
        h1 = c1 + 0.5
        l1 = c1 - 0.5

        # Market 2: flat
        c2 = np.full(n, 100.0) + rng.standard_normal(n) * 0.1
        h2 = c2 + 0.5
        l2 = c2 - 0.5

        out = trend_diff(h1, l1, c1, h2, l2, c2, period=20, atr_period=60)
        valid = out[~np.isnan(out)]
        assert np.mean(valid) > 0.0


# ---------------------------------------------------------------------------
# cmma_diff
# ---------------------------------------------------------------------------

class TestCmmaDiff:

    def test_output_shape(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = cmma_diff(h1, l1, c1, h2, l2, c2, period=20)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = cmma_diff(h1, l1, c1, h2, l2, c2, period=20)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        period = 20
        atr_period = 60
        out = cmma_diff(h1, l1, c1, h2, l2, c2, period=period, atr_period=atr_period)
        first_valid = max(period - 1, atr_period)
        assert np.all(np.isnan(out[:first_valid]))
        assert not np.isnan(out[first_valid])

    def test_identical_markets_gives_zero(self, random_ohlc_pair):
        """If both markets are identical, cmma_diff should be 0."""
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = cmma_diff(h1, l1, c1, h1, l1, c1, period=20, atr_period=60)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_output_range_approx(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = cmma_diff(h1, l1, c1, h2, l2, c2, period=20)
        valid = out[~np.isnan(out)]
        assert np.all(valid > -200.0)
        assert np.all(valid < 200.0)

    def test_no_inf_values(self, random_ohlc_pair):
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = cmma_diff(h1, l1, c1, h2, l2, c2, period=20)
        assert not np.any(np.isinf(out))

    def test_antisymmetry(self, random_ohlc_pair):
        """Swapping markets should negate the output."""
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out12 = cmma_diff(h1, l1, c1, h2, l2, c2, period=20)
        out21 = cmma_diff(h2, l2, c2, h1, l1, c1, period=20)
        np.testing.assert_allclose(out12, -out21, atol=1e-10, equal_nan=True)

    def test_market1_above_ma(self):
        """Market 1 well above its MA, market 2 at MA → positive cmma_diff."""
        n = 200
        rng = np.random.default_rng(7)

        # Market 2: flat around 100
        c2 = np.full(n, 100.0) + rng.standard_normal(n) * 0.05
        h2 = c2 + 0.1
        l2 = c2 - 0.1

        # Market 1: flat for first 100 bars, then jumps up
        c1 = np.full(n, 100.0) + rng.standard_normal(n) * 0.05
        c1[100:] += 5.0   # well above MA
        h1 = c1 + 0.1
        l1 = c1 - 0.1

        out = cmma_diff(h1, l1, c1, h2, l2, c2, period=20, atr_period=60)
        # Well into the spike region, market 1 should be above its MA
        assert out[150] > 0.0

    def test_default_period(self, random_ohlc_pair):
        """Calling with defaults should not raise and should return correct shape."""
        h1, l1, c1, h2, l2, c2 = random_ohlc_pair
        out = cmma_diff(h1, l1, c1, h2, l2, c2)
        assert out.shape == (200,)


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes
# ---------------------------------------------------------------------------

class TestTrendDiffLengthValidation:
    """Tests for Fix #6: explicit length checks in trend_diff and cmma_diff."""

    def _make_ohlc(self, n: int = 100, seed: int = 0):
        rng = np.random.default_rng(seed)
        close = np.maximum(100.0 + rng.standard_normal(n).cumsum(), 1.0)
        high = close + rng.uniform(0.1, 1.0, n)
        low = close - rng.uniform(0.1, 1.0, n)
        return high, low, close

    # trend_diff length checks
    def test_trend_diff_market_length_mismatch_raises(self):
        h1, l1, c1 = self._make_ohlc(100)
        h2, l2, c2 = self._make_ohlc(90)
        with pytest.raises(ValueError, match="same length"):
            trend_diff(h1, l1, c1, h2, l2, c2)

    def test_trend_diff_intramarket1_mismatch_raises(self):
        h1, l1, c1 = self._make_ohlc(100)
        h2, l2, c2 = self._make_ohlc(100)
        with pytest.raises(ValueError, match="same length"):
            trend_diff(h1[:90], l1, c1, h2, l2, c2)

    def test_trend_diff_intramarket2_mismatch_raises(self):
        h1, l1, c1 = self._make_ohlc(100)
        h2, l2, c2 = self._make_ohlc(100)
        with pytest.raises(ValueError, match="same length"):
            trend_diff(h1, l1, c1, h2[:90], l2, c2)

    # cmma_diff length checks
    def test_cmma_diff_market_length_mismatch_raises(self):
        h1, l1, c1 = self._make_ohlc(100)
        h2, l2, c2 = self._make_ohlc(90)
        with pytest.raises(ValueError, match="same length"):
            cmma_diff(h1, l1, c1, h2, l2, c2)

    def test_cmma_diff_intramarket1_mismatch_raises(self):
        h1, l1, c1 = self._make_ohlc(100)
        h2, l2, c2 = self._make_ohlc(100)
        with pytest.raises(ValueError, match="same length"):
            cmma_diff(h1, l1[:90], c1, h2, l2, c2)

    def test_cmma_diff_intramarket2_mismatch_raises(self):
        h1, l1, c1 = self._make_ohlc(100)
        h2, l2, c2 = self._make_ohlc(100)
        with pytest.raises(ValueError, match="same length"):
            cmma_diff(h1, l1, c1, h2, l2[:90], c2)
