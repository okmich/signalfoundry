"""
Tests for microstructure.depth module.

Functions:
    bar_absorption_ratio, wick_imbalance, range_volume_depth,
    absorption_weighted_depth_score
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import bar_absorption_ratio, wick_imbalance, range_volume_depth, \
    absorption_weighted_depth_score



# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n=60, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.006, n))
    vol = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return (
        pd.Series(open_, index=idx, name="open"),
        pd.Series(high, index=idx, name="high"),
        pd.Series(low, index=idx, name="low"),
        pd.Series(close, index=idx, name="close"),
        pd.Series(vol, index=idx, name="volume"),
    )


def _make_ohlcv_np(n=60, seed=42):
    open_, high, low, close, vol = _make_ohlcv(n, seed)
    return open_.values, high.values, low.values, close.values, vol.values


# ─── TestBarAbsorptionRatio ───────────────────────────────────────────────────

class TestBarAbsorptionRatio:

    def test_returns_series_for_series_input(self):
        open_, _, _, close, vol = _make_ohlcv()
        result = bar_absorption_ratio(open_, close, vol)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        open_, _, _, close, vol = _make_ohlcv_np()
        result = bar_absorption_ratio(open_, close, vol)
        assert isinstance(result, np.ndarray)

    def test_doji_bars_clipped_to_finite(self):
        """When close == open, |C-O| = 0 → AR should be clipped to a finite value, not NaN."""
        n = 10
        open_ = pd.Series(np.full(n, 100.0))
        close = pd.Series(np.full(n, 100.0))  # doji
        vol = pd.Series(np.full(n, 1000.0))
        result = bar_absorption_ratio(open_, close, vol)
        assert np.all(np.isfinite(result.values))
        assert np.all(~np.isnan(result.values))

    def test_high_volume_small_body_gives_high_ar(self):
        """High volume with tiny body → very high AR."""
        open_ = pd.Series([100.0])
        close = pd.Series([100.01])  # 0.01 body
        vol = pd.Series([10000.0])
        result = bar_absorption_ratio(open_, close, vol)
        assert result.iloc[0] == pytest.approx(10000.0 / 0.01)

    def test_low_volume_large_body_gives_low_ar(self):
        """Low volume with large body → low AR."""
        open_ = pd.Series([100.0])
        close = pd.Series([110.0])  # 10.0 body
        vol = pd.Series([100.0])
        result = bar_absorption_ratio(open_, close, vol)
        assert result.iloc[0] == pytest.approx(100.0 / 10.0)

    def test_non_negative_values(self):
        open_, _, _, close, vol = _make_ohlcv()
        result = bar_absorption_ratio(open_, close, vol)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_name(self):
        open_, _, _, close, vol = _make_ohlcv()
        result = bar_absorption_ratio(open_, close, vol)
        assert result.name == "bar_absorption_ratio"


# ─── TestWickImbalance ────────────────────────────────────────────────────────

class TestWickImbalance:

    def test_returns_series_for_series_input(self):
        open_, high, low, close, _ = _make_ohlcv()
        result = wick_imbalance(high, low, open_, close)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        open_, high, low, close, _ = _make_ohlcv_np()
        result = wick_imbalance(high, low, open_, close)
        assert isinstance(result, np.ndarray)

    def test_values_in_minus_one_to_one(self):
        """WI is bounded in [-1, +1]."""
        open_, high, low, close, _ = _make_ohlcv()
        result = wick_imbalance(high, low, open_, close)
        assert (result >= -1.0).all() and (result <= 1.0).all()

    def test_long_lower_wick_gives_positive_wi(self):
        """
        Bar with large lower wick and no upper wick:
        open = close = high → upper wick = 0, lower wick = body
        """
        # Bullish hammer: open at high, close at high, low far below
        high = pd.Series([105.0])
        open_ = pd.Series([105.0])
        close = pd.Series([105.0])
        low = pd.Series([95.0])
        result = wick_imbalance(high, low, open_, close)
        # upper_wick = 105 - 105 = 0, lower_wick = 105 - 95 = 10
        # WI = (10 - 0) / 10 = 1.0
        assert result.iloc[0] == pytest.approx(1.0)

    def test_long_upper_wick_gives_negative_wi(self):
        """
        Bar with large upper wick and no lower wick → WI = -1.
        """
        high = pd.Series([115.0])
        open_ = pd.Series([105.0])
        close = pd.Series([105.0])
        low = pd.Series([105.0])
        result = wick_imbalance(high, low, open_, close)
        # upper_wick = 115 - 105 = 10, lower_wick = 105 - 105 = 0
        # WI = (0 - 10) / 10 = -1.0
        assert result.iloc[0] == pytest.approx(-1.0)

    def test_equal_wicks_gives_zero(self):
        """Symmetric bar (equal upper and lower wicks) → WI = 0."""
        high = pd.Series([110.0])
        low = pd.Series([90.0])
        open_ = pd.Series([100.0])
        close = pd.Series([100.0])
        result = wick_imbalance(high, low, open_, close)
        # upper_wick = 10, lower_wick = 10 → (10-10)/20 = 0
        assert result.iloc[0] == pytest.approx(0.0)

    def test_doji_bar_gives_zero(self):
        """Doji bar (H=L=O=C) → WI = 0 (no range)."""
        high = pd.Series([100.0])
        low = pd.Series([100.0])
        open_ = pd.Series([100.0])
        close = pd.Series([100.0])
        result = wick_imbalance(high, low, open_, close)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_name(self):
        open_, high, low, close, _ = _make_ohlcv()
        result = wick_imbalance(high, low, open_, close)
        assert result.name == "wick_imbalance"


# ─── TestRangeVolumeDepth ─────────────────────────────────────────────────────

class TestRangeVolumeDepth:

    def test_returns_dataframe(self):
        _, high, low, _, vol = _make_ohlcv()
        result = range_volume_depth(high, low, vol)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        _, high, low, _, vol = _make_ohlcv()
        result = range_volume_depth(high, low, vol)
        assert set(result.columns) == {"rvd", "rvd_z"}

    def test_works_with_ndarray_input(self):
        _, high, low, _, vol = _make_ohlcv_np()
        result = range_volume_depth(high, low, vol)
        assert isinstance(result, pd.DataFrame)
        assert "rvd" in result.columns

    def test_doji_bars_give_nan_rvd(self):
        """When H == L, RVD should be NaN."""
        n = 10
        high = pd.Series(np.full(n, 100.0))
        low = pd.Series(np.full(n, 100.0))
        vol = pd.Series(np.full(n, 1000.0))
        result = range_volume_depth(high, low, vol)
        assert np.all(np.isnan(result["rvd"].values))

    def test_zscore_early_values_nan(self):
        """First (window-1) z-scores should be NaN (insufficient history)."""
        _, high, low, _, vol = _make_ohlcv(n=50)
        window = 10
        result = range_volume_depth(high, low, vol, window=window)
        assert np.all(np.isnan(result["rvd_z"].values[:window - 1]))

    def test_high_volume_narrow_range_gives_high_rvd(self):
        """Narrow range + high volume → RVD >> typical."""
        _, high, low, _, vol = _make_ohlcv(n=40)
        # Inject a narrow-range high-volume bar at position 25
        high_arr = high.values.copy()
        low_arr = low.values.copy()
        vol_arr = vol.values.copy()
        high_arr[25] = 100.01
        low_arr[25] = 99.99   # very narrow range
        vol_arr[25] = 100000.0  # very high volume
        result = range_volume_depth(
            pd.Series(high_arr, index=high.index),
            pd.Series(low_arr, index=low.index),
            pd.Series(vol_arr, index=vol.index),
            window=10,
        )
        # RVD z-score at bar 25 should be large positive
        assert result["rvd_z"].iloc[25] > 2.0

    def test_rvd_non_negative(self):
        """RVD must be non-negative (ratio of positives)."""
        _, high, low, _, vol = _make_ohlcv()
        result = range_volume_depth(high, low, vol)
        valid = result["rvd"].dropna()
        assert (valid > 0).all()


# ─── TestAbsorptionWeightedDepthScore ─────────────────────────────────────────

class TestAbsorptionWeightedDepthScore:

    def test_returns_series_for_series_input(self):
        open_, _, _, close, vol = _make_ohlcv()
        result = absorption_weighted_depth_score(open_, close, vol)
        assert isinstance(result, pd.Series)

    def test_returns_ndarray_for_ndarray_input(self):
        open_, _, _, close, vol = _make_ohlcv_np()
        result = absorption_weighted_depth_score(open_, close, vol)
        assert isinstance(result, np.ndarray)

    def test_doji_bars_give_zero(self):
        """When close == open, sign=0 and AR is finite → DS = 0."""
        n = 30
        open_ = pd.Series(np.full(n, 100.0))
        close = pd.Series(np.full(n, 100.0))
        vol = pd.Series(np.ones(n) * 1000.0)
        result = absorption_weighted_depth_score(open_, close, vol)
        assert np.all(np.isfinite(result.values))
        assert np.allclose(result.values, 0.0)

    def test_bullish_bar_above_average_absorption_gives_positive(self):
        """Bullish bar (close > open) with AR > EMA → DS > 0."""
        n = 30
        # Uniform small-body bullish bars with volume 100
        open_ = np.full(n, 100.0)
        close = np.full(n, 101.0)
        vol = np.full(n, 100.0)
        # Spike last bar with high volume → AR >> EMA → DS > 1
        vol[-1] = 10000.0
        result = absorption_weighted_depth_score(
            pd.Series(open_), pd.Series(close), pd.Series(vol), window=10
        )
        assert result.iloc[-1] > 1.0

    def test_bearish_bar_above_average_absorption_gives_negative(self):
        """Bearish bar (close < open) with AR > EMA → DS < 0."""
        n = 30
        open_ = np.full(n, 101.0)
        close = np.full(n, 100.0)  # bearish
        vol = np.full(n, 100.0)
        # Spike last bar with high volume → AR >> EMA → |DS| > 1, negative
        vol[-1] = 10000.0
        result = absorption_weighted_depth_score(
            pd.Series(open_), pd.Series(close), pd.Series(vol), window=10
        )
        assert result.iloc[-1] < -1.0

    def test_average_bar_gives_near_one(self):
        """Uniform bars → AR ≈ EMA → DS ≈ ±1."""
        n = 50
        open_ = pd.Series(np.full(n, 100.0))
        close = pd.Series(np.full(n, 101.0))  # all bullish
        vol = pd.Series(np.full(n, 1000.0))
        result = absorption_weighted_depth_score(open_, close, vol, window=10)
        # After EMA warm-up, AR/EMA → 1.0, sign=+1 → DS → +1
        valid = result.iloc[15:]  # skip warm-up
        np.testing.assert_allclose(valid.values, 1.0, rtol=0.01)

    def test_name(self):
        open_, _, _, close, vol = _make_ohlcv()
        result = absorption_weighted_depth_score(open_, close, vol)
        assert result.name == "absorption_depth_score"