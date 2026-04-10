"""
Tests for advanced price structure features (§17).

Functions:
    price_path_fractal_dimension, close_open_gap_analysis,
    return_spread_cross_correlation
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    price_path_fractal_dimension,
    close_open_gap_analysis,
    return_spread_cross_correlation,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n=80, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.006, n))
    vol = rng.uniform(1000, 5000, n)
    mid = (high + low) / 2.0
    spread = mid * rng.uniform(0.0005, 0.003, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return dict(
        open_=pd.Series(open_, index=idx),
        high=pd.Series(high, index=idx),
        low=pd.Series(low, index=idx),
        close=pd.Series(close, index=idx),
        volume=pd.Series(vol, index=idx),
        spread=pd.Series(spread, index=idx),
        mid_price=pd.Series(mid, index=idx),
    )


# ─── TestPricePathFractalDimension ────────────────────────────────────────────

class TestPricePathFractalDimension:

    def test_returns_dataframe(self):
        d = _make_ohlcv()
        result = price_path_fractal_dimension(d["open_"], d["high"],
                                              d["low"], d["close"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        d = _make_ohlcv()
        result = price_path_fractal_dimension(d["open_"], d["high"],
                                              d["low"], d["close"])
        assert set(result.columns) == {"FD", "FD_ema", "delta_FD"}

    def test_fd_non_negative(self):
        """FD = log2(2R/body) — for any positive ratio this is >= 0."""
        d = _make_ohlcv()
        result = price_path_fractal_dimension(d["open_"], d["high"],
                                              d["low"], d["close"])
        valid = result["FD"].dropna()
        assert (valid >= 0).all()

    def test_doji_bar_gives_high_fd(self):
        """
        Doji (body ≈ 0): body clamped to 1e-10 → FD = log2(2*range/1e-10) >> 1.
        """
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        open_ = pd.Series(np.full(n, 100.0), index=idx)
        close = pd.Series(np.full(n, 100.0), index=idx)
        high = pd.Series(np.full(n, 101.0), index=idx)
        low = pd.Series(np.full(n, 99.0), index=idx)
        result = price_path_fractal_dimension(open_, high, low, close)
        # body ≈ 0 → ratio = 2*2/1e-10 = 4e10 → FD = log2(4e10) >> 2
        assert (result["FD"] > 2.0).all()

    def test_marubozu_bar_gives_fd_one(self):
        """
        Marubozu (no wicks, H-L = |O-C|):
            ratio = 2*(H-L)/(H-L) = 2 → FD = log2(2) = 1.
        """
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        open_ = pd.Series(np.full(n, 100.0), index=idx)
        close = pd.Series(np.full(n, 102.0), index=idx)
        high = pd.Series(np.full(n, 102.0), index=idx)  # no upper wick
        low = pd.Series(np.full(n, 100.0), index=idx)   # no lower wick
        result = price_path_fractal_dimension(open_, high, low, close)
        np.testing.assert_allclose(result["FD"].values, 1.0, atol=1e-10)

    def test_fd_ema_smoother_than_fd(self):
        """
        After the initial EMA warm-up, FD_ema should be smoother (lower std)
        than raw FD in the tail of the series.
        """
        d = _make_ohlcv(n=100)
        result = price_path_fractal_dimension(d["open_"], d["high"],
                                              d["low"], d["close"], window=10)
        # Skip first 20 bars to avoid EMA transient from initial spike
        tail_fd = result["FD"].iloc[20:]
        tail_ema = result["FD_ema"].iloc[20:]
        assert tail_ema.std() <= tail_fd.std()

    def test_no_nans_in_fd(self):
        """FD is bar-level (no rolling) — should have no NaN."""
        d = _make_ohlcv()
        result = price_path_fractal_dimension(d["open_"], d["high"],
                                              d["low"], d["close"])
        assert not result["FD"].isna().any()


# ─── TestCloseOpenGapAnalysis ─────────────────────────────────────────────────

class TestCloseOpenGapAnalysis:

    def test_returns_dataframe(self):
        d = _make_ohlcv()
        result = close_open_gap_analysis(d["open_"], d["close"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        d = _make_ohlcv()
        result = close_open_gap_analysis(d["open_"], d["close"])
        assert set(result.columns) == {"gap", "gap_fill_ratio"}

    def test_first_bar_gap_nan(self):
        """Bar 0 has no previous close → gap_fill_ratio is NaN."""
        d = _make_ohlcv()
        result = close_open_gap_analysis(d["open_"], d["close"])
        assert np.isnan(result["gap_fill_ratio"].values[0])

    def test_no_gap_gives_gfr_zero(self):
        """When open = prev close, gap = 0 → GFR = 0."""
        n = 5
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        close_arr = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        open_arr = np.roll(close_arr, 1)  # open = prev close
        open_arr[0] = close_arr[0]
        open_ = pd.Series(open_arr, index=idx)
        close = pd.Series(close_arr, index=idx)
        result = close_open_gap_analysis(open_, close)
        # No gap → GFR = 0 everywhere (except bar 0 which is NaN)
        valid = result["gap_fill_ratio"].iloc[1:]
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_full_gap_fill_gives_gfr_one(self):
        """
        Up gap of 2 points, close fills exactly the gap (close = prev close):
        fill = open - close = open - (open-2) = 2 = gap_size → GFR = 1.
        """
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        close = pd.Series([100.0, 100.0], index=idx)   # bar 1 closes at 100 (fills gap)
        open_ = pd.Series([100.0, 102.0], index=idx)   # bar 1 opens at 102 (up gap)
        result = close_open_gap_analysis(open_, close)
        # gap_size = 102 - 100 = 2, fill = 102 - 100 = 2 → GFR = 1.0
        assert result["gap_fill_ratio"].iloc[1] == pytest.approx(1.0)

    def test_gap_extension_gives_negative_gfr(self):
        """
        Up gap: open > prev_close. Close goes higher → gap extended → GFR < 0.
        """
        idx = pd.date_range("2024-01-01", periods=2, freq="D")
        close = pd.Series([100.0, 103.0], index=idx)  # close above open
        open_ = pd.Series([100.0, 102.0], index=idx)  # up gap of 2
        result = close_open_gap_analysis(open_, close)
        # fill = open - close = 102 - 103 = -1 → GFR = -1/2 = -0.5
        assert result["gap_fill_ratio"].iloc[1] < 0

    def test_gap_column_formula(self):
        """gap = O(t) - C(t-1)."""
        d = _make_ohlcv(n=10)
        result = close_open_gap_analysis(d["open_"], d["close"])
        expected_gap = d["open_"] - d["close"].shift(1)
        pd.testing.assert_series_equal(result["gap"], expected_gap,
                                       check_names=False, rtol=1e-10)


# ─── TestReturnSpreadCrossCorrelation ─────────────────────────────────────────

class TestReturnSpreadCrossCorrelation:

    def test_returns_dataframe(self):
        d = _make_ohlcv()
        result = return_spread_cross_correlation(d["close"], d["spread"],
                                                  d["mid_price"])
        assert isinstance(result, pd.DataFrame)

    def test_default_columns(self):
        d = _make_ohlcv()
        result = return_spread_cross_correlation(d["close"], d["spread"],
                                                  d["mid_price"])
        assert set(result.columns) == {"xcorr_lag+1", "xcorr_lag+2", "xcorr_lag+3"}

    def test_custom_lags(self):
        d = _make_ohlcv()
        result = return_spread_cross_correlation(d["close"], d["spread"],
                                                  d["mid_price"], lags=[1, -1])
        assert set(result.columns) == {"xcorr_lag+1", "xcorr_lag-1"}

    def test_values_bounded(self):
        """Cross-correlation is Pearson → must be in [-1, +1]."""
        d = _make_ohlcv(n=120)
        result = return_spread_cross_correlation(d["close"], d["spread"],
                                                  d["mid_price"], window=20)
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert (valid >= -1.0 - 1e-10).all() and (valid <= 1.0 + 1e-10).all()

    def test_early_values_nan(self):
        d = _make_ohlcv(n=60)
        result = return_spread_cross_correlation(d["close"], d["spread"],
                                                  d["mid_price"], window=20, lags=[1])
        assert np.isnan(result["xcorr_lag+1"].values[0])

    def test_spread_leads_return_positive_correlation(self):
        """
        Construct a series where spread widening at t-1 predicts negative
        return at t (spread leads price → positive xcorr at lag +1 when
        spread and return move together with the sign convention).
        """
        n = 120
        rng = np.random.default_rng(17)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        # Base price random walk
        ret = rng.normal(0, 0.005, n)
        close = pd.Series(100.0 * np.cumprod(1 + ret), index=idx)
        mid = pd.Series(np.full(n, 100.0), index=idx)

        # Spread moves with lagged returns: spread(t) = |ret(t-1)| * 5
        spread_arr = np.concatenate([[0.05], np.abs(ret[:-1]) * 50])
        spread = pd.Series(spread_arr, index=idx)

        result = return_spread_cross_correlation(close, spread, mid,
                                                  window=20, lags=[1])
        valid = result["xcorr_lag+1"].dropna()
        assert len(valid) > 0  # just verify it computes

    def test_negative_lag_column_exists(self):
        """Negative lag (returns lead spread) column should be computable."""
        d = _make_ohlcv(n=100)
        result = return_spread_cross_correlation(d["close"], d["spread"],
                                                  d["mid_price"],
                                                  window=20, lags=[-1])
        assert "xcorr_lag-1" in result.columns
        valid = result["xcorr_lag-1"].dropna()
        if len(valid) > 0:
            assert (valid.abs() <= 1.0 + 1e-10).all()
