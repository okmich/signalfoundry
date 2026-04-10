"""
Tests for path_structure.auto_corr.

Covers:
  - Return type mirrors input type (Series in → Series out, ndarray in → ndarray out)
  - Correct NaN warmup length (window + lag - 1)
  - Output bounded in [-1, 1]
  - Trending / mean-reverting series produce correct sign
  - Constant series returns 0 (no variance)
  - Arbitrary lag > 1
  - Input validation errors
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.path_structure import auto_corr


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_series(n=200, seed=42):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    return pd.Series(values, index=idx)


def _make_array(n=200, seed=42):
    return _make_series(n, seed).to_numpy()


def _trending_returns(n=200, seed=0):
    """Persistent (trending) returns: each value = 0.8 * prev + noise."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.8 * r[i - 1] + rng.standard_normal() * 0.1
    return r


def _mean_reverting_returns(n=200, seed=0):
    """Mean-reverting returns: each value = -0.8 * prev + noise."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = -0.8 * r[i - 1] + rng.standard_normal() * 0.1
    return r


# ─── Return type ─────────────────────────────────────────────────────────────

class TestReturnType:

    def test_series_in_series_out(self):
        result = auto_corr(_make_series(), window=20)
        assert isinstance(result, pd.Series)

    def test_ndarray_in_ndarray_out(self):
        result = auto_corr(_make_array(), window=20)
        assert isinstance(result, np.ndarray)

    def test_series_index_preserved(self):
        s = _make_series()
        result = auto_corr(s, window=20)
        pd.testing.assert_index_equal(result.index, s.index)


# ─── NaN warmup ──────────────────────────────────────────────────────────────

class TestNanWarmup:

    @pytest.mark.parametrize("window,lag", [(10, 1), (20, 1), (15, 3), (10, 5)])
    def test_warmup_length(self, window, lag):
        arr = _make_array(n=200)
        result = auto_corr(arr, window=window, lag=lag)
        expected_nan = window + lag - 1
        assert np.all(np.isnan(result[:expected_nan])), (
            f"Expected first {expected_nan} values to be NaN"
        )
        assert np.any(~np.isnan(result[expected_nan:])), (
            "Expected at least some valid values after warmup"
        )

    def test_first_valid_index_series(self):
        window, lag = 20, 1
        s = _make_series(n=100)
        result = auto_corr(s, window=window, lag=lag)
        first_valid = result.first_valid_index()
        expected_pos = window + lag - 1
        assert result.index.get_loc(first_valid) == expected_pos


# ─── Output range ─────────────────────────────────────────────────────────────

class TestOutputRange:

    def test_values_bounded_minus1_to_1(self):
        arr = _make_array(n=300)
        result = auto_corr(arr, window=30)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0 - 1e-9)
        assert np.all(valid <= 1.0 + 1e-9)

    def test_values_bounded_with_higher_lag(self):
        arr = _make_array(n=300)
        result = auto_corr(arr, window=30, lag=5)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0 - 1e-9)
        assert np.all(valid <= 1.0 + 1e-9)


# ─── Directional correctness ──────────────────────────────────────────────────

class TestDirectionalCorrectness:

    def test_trending_series_positive_acf(self):
        """AR(1) with positive coefficient → median ACF(1) should be positive."""
        arr = _trending_returns(n=500)
        result = auto_corr(arr, window=50, lag=1)
        valid = result[~np.isnan(result)]
        assert np.median(valid) > 0.0, (
            f"Trending series should have positive median ACF(1), got {np.median(valid):.4f}"
        )

    def test_mean_reverting_series_negative_acf(self):
        """AR(1) with negative coefficient → median ACF(1) should be negative."""
        arr = _mean_reverting_returns(n=500)
        result = auto_corr(arr, window=50, lag=1)
        valid = result[~np.isnan(result)]
        assert np.median(valid) < 0.0, (
            f"Mean-reverting series should have negative median ACF(1), got {np.median(valid):.4f}"
        )


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_constant_series_returns_zero(self):
        """Constant series has no variance → autocorrelation = 0."""
        arr = np.ones(100)
        result = auto_corr(arr, window=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid == 0.0)

    def test_lag_equals_window_minus_1(self):
        """Maximum valid lag = window - 1."""
        arr = _make_array(n=200)
        result = auto_corr(arr, window=10, lag=9)
        assert isinstance(result, np.ndarray)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_lag_2_produces_different_result_than_lag_1(self):
        arr = _make_array(n=300)
        r1 = auto_corr(arr, window=30, lag=1)
        r2 = auto_corr(arr, window=30, lag=2)
        # compare only indices valid in both (lag=2 has one extra NaN at the start)
        both_valid = ~(np.isnan(r1) | np.isnan(r2))
        assert both_valid.sum() > 0
        assert not np.allclose(r1[both_valid], r2[both_valid])

    def test_series_and_array_give_same_values(self):
        """Series and ndarray paths must produce numerically identical output."""
        s = _make_series(n=200)
        arr = s.to_numpy()
        r_series = auto_corr(s, window=20, lag=1).to_numpy()
        r_array = auto_corr(arr, window=20, lag=1)
        np.testing.assert_array_equal(r_series, r_array)


# ─── Agrees with pandas rolling corr (reference) ─────────────────────────────

class TestAgainstPandasReference:

    @pytest.mark.parametrize("window,lag", [(10, 1), (20, 2), (15, 3)])
    def test_matches_pandas_rolling_corr(self, window, lag):
        """Numba kernel must match pandas rolling().corr() within float64 tolerance."""
        s = _make_series(n=300)
        expected = s.rolling(window).corr(s.shift(lag)).to_numpy()
        result = auto_corr(s, window=window, lag=lag).to_numpy()
        # compare only where both are non-NaN
        mask = ~(np.isnan(expected) | np.isnan(result))
        assert mask.sum() > 0
        np.testing.assert_allclose(result[mask], expected[mask], rtol=1e-10, atol=1e-12)


# ─── Input validation ─────────────────────────────────────────────────────────

class TestInputValidation:

    def test_window_less_than_2_raises(self):
        with pytest.raises(ValueError, match="window"):
            auto_corr(_make_array(), window=1)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window"):
            auto_corr(_make_array(), window=0)

    def test_lag_zero_raises(self):
        with pytest.raises(ValueError, match="lag"):
            auto_corr(_make_array(), window=10, lag=0)

    def test_lag_negative_raises(self):
        with pytest.raises(ValueError, match="lag"):
            auto_corr(_make_array(), window=10, lag=-1)

    def test_lag_ge_window_raises(self):
        with pytest.raises(ValueError, match="lag"):
            auto_corr(_make_array(), window=10, lag=10)