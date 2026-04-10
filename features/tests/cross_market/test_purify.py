"""Tests for cross_market.purify — indicators #4–5 (PURIFY, LOG_PURIFY)."""

import numpy as np
import pytest

from okmich_quant_features.timothymasters.cross_market import purify, log_purify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_pair():
    """Two correlated random-walk price series (200 bars)."""
    rng = np.random.default_rng(42)
    n = 200
    base = 100.0 + rng.standard_normal(n).cumsum()
    close1 = np.abs(base * (1.0 + rng.standard_normal(n) * 0.02)) + 1.0
    close2 = np.abs(base * (1.0 + rng.standard_normal(n) * 0.02)) + 1.0
    return close1, close2


@pytest.fixture
def long_pair():
    """Longer series (500 bars) for more robust statistical tests."""
    rng = np.random.default_rng(99)
    n = 500
    base = 100.0 + rng.standard_normal(n).cumsum() * 0.5
    close1 = np.abs(base * (1.0 + rng.standard_normal(n) * 0.01)) + 1.0
    close2 = np.abs(base * (1.0 + rng.standard_normal(n) * 0.01)) + 1.0
    return close1, close2


# ---------------------------------------------------------------------------
# PURIFY tests
# ---------------------------------------------------------------------------

class TestPurify:

    def test_output_shape(self, random_pair):
        c1, c2 = random_pair
        out = purify(c1, c2)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_pair):
        c1, c2 = random_pair
        out = purify(c1, c2)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_pair):
        c1, c2 = random_pair
        lookback, trend, accel, vol = 60, 20, 20, 20
        front_bad = lookback + max(trend, accel, vol) - 1  # 79
        out = purify(c1, c2, lookback=lookback, trend_length=trend,
                     accel_length=accel, vol_length=vol)
        assert np.all(np.isnan(out[:front_bad]))
        # At least some valid bars after warmup
        assert not np.all(np.isnan(out[front_bad:]))

    def test_output_range(self, long_pair):
        c1, c2 = long_pair
        out = purify(c1, c2)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert np.all(valid > -50.0)
        assert np.all(valid < 50.0)

    def test_no_inf_in_valid_bars(self, random_pair):
        c1, c2 = random_pair
        out = purify(c1, c2)
        valid = out[~np.isnan(out)]
        assert not np.any(np.isinf(valid))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            purify(np.ones(10), np.ones(11))

    def test_default_params_work(self, random_pair):
        """Default parameters should produce valid output without error."""
        c1, c2 = random_pair
        out = purify(c1, c2)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0

    def test_short_lookback(self):
        """Small lookback with small predictor lengths should work."""
        rng = np.random.default_rng(7)
        n = 50
        c1 = np.abs(100.0 + rng.standard_normal(n).cumsum()) + 1.0
        c2 = np.abs(100.0 + rng.standard_normal(n).cumsum()) + 1.0
        out = purify(c1, c2, lookback=5, trend_length=3, accel_length=3, vol_length=3)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid))

    def test_asymmetric_swap(self, random_pair):
        """Swapping close1/close2 should give different results."""
        c1, c2 = random_pair
        out_12 = purify(c1, c2)
        out_21 = purify(c2, c1)
        valid_mask = ~np.isnan(out_12) & ~np.isnan(out_21)
        assert not np.allclose(out_12[valid_mask], out_21[valid_mask])


# ---------------------------------------------------------------------------
# LOG_PURIFY tests
# ---------------------------------------------------------------------------

class TestLogPurify:

    def test_output_shape(self, random_pair):
        c1, c2 = random_pair
        out = log_purify(c1, c2)
        assert out.shape == (200,)

    def test_dtype_float64(self, random_pair):
        c1, c2 = random_pair
        out = log_purify(c1, c2)
        assert out.dtype == np.float64

    def test_warmup_nans(self, random_pair):
        c1, c2 = random_pair
        lookback, trend, accel, vol = 60, 20, 20, 20
        front_bad = lookback + max(trend, accel, vol) - 1
        out = log_purify(c1, c2, lookback=lookback, trend_length=trend,
                         accel_length=accel, vol_length=vol)
        assert np.all(np.isnan(out[:front_bad]))
        assert not np.all(np.isnan(out[front_bad:]))

    def test_output_range(self, long_pair):
        c1, c2 = long_pair
        out = log_purify(c1, c2)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert np.all(valid > -50.0)
        assert np.all(valid < 50.0)

    def test_no_inf_in_valid_bars(self, random_pair):
        c1, c2 = random_pair
        out = log_purify(c1, c2)
        valid = out[~np.isnan(out)]
        assert not np.any(np.isinf(valid))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            log_purify(np.ones(10), np.ones(11))

    def test_default_params_work(self, random_pair):
        c1, c2 = random_pair
        out = log_purify(c1, c2)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# PURIFY vs LOG_PURIFY comparison tests
# ---------------------------------------------------------------------------

class TestPurifyVsLogPurify:

    def test_different_results(self, random_pair):
        """PURIFY and LOG_PURIFY should give different results on the same data."""
        c1, c2 = random_pair
        out_raw = purify(c1, c2)
        out_log = log_purify(c1, c2)
        valid_mask = ~np.isnan(out_raw) & ~np.isnan(out_log)
        assert not np.allclose(out_raw[valid_mask], out_log[valid_mask])


# ---------------------------------------------------------------------------
# Disabled predictors tests
# ---------------------------------------------------------------------------

class TestDisabledPredictors:

    def test_no_trend(self, random_pair):
        """trend_length=0 disables trend predictor — should still work."""
        c1, c2 = random_pair
        out = purify(c1, c2, lookback=30, trend_length=0, accel_length=10, vol_length=10)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid))

    def test_no_accel(self, random_pair):
        """accel_length=0 disables acceleration predictor."""
        c1, c2 = random_pair
        out = purify(c1, c2, lookback=30, trend_length=10, accel_length=0, vol_length=10)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0

    def test_no_vol(self, random_pair):
        """vol_length=0 disables volatility predictor."""
        c1, c2 = random_pair
        out = purify(c1, c2, lookback=30, trend_length=10, accel_length=10, vol_length=0)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0

    def test_all_disabled(self, random_pair):
        """All predictors disabled → constant-only regression → deviation-like."""
        c1, c2 = random_pair
        out = purify(c1, c2, lookback=30, trend_length=0, accel_length=0, vol_length=0)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        # Should still be in valid range
        assert np.all(valid > -50.0)
        assert np.all(valid < 50.0)

    def test_disabled_changes_result(self, random_pair):
        """Disabling a predictor should change the output."""
        c1, c2 = random_pair
        out_full = purify(c1, c2, lookback=30, trend_length=10, accel_length=10, vol_length=10)
        out_no_trend = purify(c1, c2, lookback=30, trend_length=0, accel_length=10, vol_length=10)
        valid_mask = ~np.isnan(out_full) & ~np.isnan(out_no_trend)
        assert not np.allclose(out_full[valid_mask], out_no_trend[valid_mask])


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes
# ---------------------------------------------------------------------------

class TestPurifyLegendreBasisFix:
    """
    Regression tests for Fix #3: Legendre weights now generated per sub-window.

    Before the fix, weights were always generated for 'lookback' but applied
    to shorter sub-windows (trend_length / accel_length), causing a basis
    mismatch.  After the fix, sub-window weights match the actual window size.
    """

    def _make_pair(self, n: int = 300, seed: int = 7):
        rng = np.random.default_rng(seed)
        base = 100.0 + rng.standard_normal(n).cumsum() * 0.5
        c1 = np.abs(base * (1.0 + rng.standard_normal(n) * 0.01)) + 1.0
        c2 = np.abs(base * (1.0 + rng.standard_normal(n) * 0.01)) + 1.0
        return c1, c2

    def test_nondefault_trend_length_produces_finite_output(self):
        """trend_length != lookback should still produce finite, in-range output."""
        c1, c2 = self._make_pair()
        out = purify(c1, c2, lookback=60, trend_length=10, accel_length=10, vol_length=10)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid))
        assert np.all(valid > -50.0) and np.all(valid < 50.0)

    def test_trend_length_equal_lookback_vs_different(self):
        """When trend_length == lookback, old and new code agree.
        When trend_length != lookback, old code gave wrong geometry; new code
        produces different (correct) values.
        """
        c1, c2 = self._make_pair(n=400)

        # Case where trend_length == lookback (fix is a no-op for this case)
        out_same = purify(c1, c2, lookback=20, trend_length=20, accel_length=20)
        valid_same = out_same[~np.isnan(out_same)]
        assert len(valid_same) > 0

        # Case where trend_length != lookback (fix changes values vs old bug)
        out_diff = purify(c1, c2, lookback=60, trend_length=20, accel_length=20)
        valid_diff = out_diff[~np.isnan(out_diff)]
        assert len(valid_diff) > 0

        # Different params → different outputs (trivially true)
        assert not np.allclose(out_same[-100:], out_diff[-100:])

    def test_log_purify_nondefault_lengths_finite(self):
        """Same basis fix applies to log_purify."""
        c1, c2 = self._make_pair()
        out = log_purify(c1, c2, lookback=60, trend_length=15, accel_length=15)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid))

    def test_trend_only_nondefault_length(self):
        """Only trend predictor active with sub-window smaller than lookback."""
        c1, c2 = self._make_pair()
        out = purify(c1, c2, lookback=40, trend_length=10, accel_length=0, vol_length=0)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid))

    def test_accel_only_nondefault_length(self):
        """Only acceleration predictor active with sub-window smaller than lookback."""
        c1, c2 = self._make_pair()
        out = purify(c1, c2, lookback=40, trend_length=0, accel_length=10, vol_length=0)
        valid = out[~np.isnan(out)]
        assert len(valid) > 0
        assert not np.any(np.isinf(valid))
