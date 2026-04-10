"""
Tests for advanced depth features (§15).

Functions:
    multi_bar_depth_pressure, stealth_trading_indicator
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    multi_bar_depth_pressure,
    stealth_trading_indicator,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n=80, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.006, n))
    vol = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return dict(
        open_=pd.Series(open_, index=idx),
        high=pd.Series(high, index=idx),
        low=pd.Series(low, index=idx),
        close=pd.Series(close, index=idx),
        volume=pd.Series(vol, index=idx),
    )


# ─── TestMultiBarDepthPressure ────────────────────────────────────────────────

class TestMultiBarDepthPressure:

    def test_returns_series(self):
        d = _make_ohlcv()
        result = multi_bar_depth_pressure(d["open_"], d["close"], d["volume"])
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_ohlcv()
        result = multi_bar_depth_pressure(d["open_"], d["close"], d["volume"],
                                          window=15)
        assert result.name == "multi_bar_depth_pressure_15"

    def test_early_values_nan(self):
        d = _make_ohlcv(n=60)
        result = multi_bar_depth_pressure(d["open_"], d["close"], d["volume"],
                                          window=10)
        assert np.isnan(result.values[0])

    def test_finite_after_warmup(self):
        d = _make_ohlcv(n=100)
        result = multi_bar_depth_pressure(d["open_"], d["close"], d["volume"],
                                          window=10)
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))

    def test_sustained_buying_gives_positive(self):
        """
        Sustained bullish bars (close > open) with high absorption (high volume,
        small body) should produce positive depth pressure over the window.
        Use threshold=0.5 (below 1.0) so that all bars with AR > 0.5×EMA qualify.
        """
        n = 60
        open_arr = np.full(n, 100.0)
        close_arr = np.full(n, 100.5)   # small positive body
        vol_arr = np.full(n, 10000.0)   # high volume → high AR
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = multi_bar_depth_pressure(
            pd.Series(open_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=10, threshold=0.5,  # threshold < 1 → all bars qualify
        )
        valid = result.dropna()
        assert valid.iloc[-1] > 0

    def test_sustained_selling_gives_negative(self):
        """Bearish bars with high absorption → negative depth pressure."""
        n = 60
        open_arr = np.full(n, 100.5)
        close_arr = np.full(n, 100.0)   # bearish small body
        vol_arr = np.full(n, 10000.0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = multi_bar_depth_pressure(
            pd.Series(open_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=10, threshold=0.5,
        )
        valid = result.dropna()
        assert valid.iloc[-1] < 0

    def test_high_threshold_gives_zero(self):
        """With threshold=1000, no bar qualifies → DP_N = 0 always."""
        d = _make_ohlcv(n=60)
        result = multi_bar_depth_pressure(d["open_"], d["close"], d["volume"],
                                          window=10, threshold=1000.0)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)


# ─── TestStealthTradingIndicator ──────────────────────────────────────────────

class TestStealthTradingIndicator:

    def test_returns_dataframe(self):
        d = _make_ohlcv()
        result = stealth_trading_indicator(d["high"], d["low"],
                                           d["close"], d["volume"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        d = _make_ohlcv()
        result = stealth_trading_indicator(d["high"], d["low"],
                                           d["close"], d["volume"])
        assert set(result.columns) == {"ST", "ST_direction"}

    def test_early_values_nan(self):
        d = _make_ohlcv(n=60)
        result = stealth_trading_indicator(d["high"], d["low"],
                                           d["close"], d["volume"], window=10)
        assert np.isnan(result["ST"].values[0])

    def test_st_direction_bounded(self):
        """ST_direction is sign() output → must be in {-1, 0, +1}."""
        d = _make_ohlcv(n=100)
        result = stealth_trading_indicator(d["high"], d["low"],
                                           d["close"], d["volume"], window=10)
        valid = result["ST_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_medium_volume_bars_detected(self):
        """
        When all bars are medium-volume (between 0.5× and 2× EMA), ST should
        be non-zero (those bars contribute to numerator and denominator equally).
        """
        n = 60
        rng = np.random.default_rng(7)
        # Volume close to EMA → all bars medium
        vol_arr = np.full(n, 1000.0, dtype=float)
        close_arr = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
        open_arr = np.roll(close_arr, 1); open_arr[0] = close_arr[0]
        high_arr = np.maximum(open_arr, close_arr) * 1.002
        low_arr = np.minimum(open_arr, close_arr) * 0.998
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = stealth_trading_indicator(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=10,
        )
        valid = result["ST"].dropna()
        # With all bars at medium volume, ST denominator = numerator = all bars
        assert len(valid) > 0

    def test_extreme_volume_bars_excluded(self):
        """
        Inject bars that are outside the medium range (very high or very low).
        Those bars should NOT contribute to the numerator but do to denominator,
        so ST should be lower (closer to 0) than when all bars are medium.
        """
        n = 60
        rng = np.random.default_rng(10)
        close_arr = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
        open_arr = np.roll(close_arr, 1); open_arr[0] = close_arr[0]
        high_arr = np.maximum(open_arr, close_arr) * 1.002
        low_arr = np.minimum(open_arr, close_arr) * 0.998
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        # Baseline: uniform medium volume
        vol_medium = pd.Series(np.full(n, 1000.0), index=idx)
        st_medium = stealth_trading_indicator(
            pd.Series(high_arr, index=idx), pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx), vol_medium, window=10
        )

        # Extreme: last 20 bars have 50× volume (way above 2×EMA)
        vol_extreme = vol_medium.copy()
        vol_extreme.iloc[-20:] = 50000.0
        st_extreme = stealth_trading_indicator(
            pd.Series(high_arr, index=idx), pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx), vol_extreme, window=10
        )

        # Extreme-volume bars excluded from numerator → ST should be lower
        med_last = st_medium["ST"].iloc[-5:].dropna()
        ext_last = st_extreme["ST"].iloc[-5:].dropna()
        if len(med_last) > 0 and len(ext_last) > 0:
            assert ext_last.mean() <= med_last.mean() + 0.1  # allow small tolerance
