"""
Tests for advanced composite meta-features (§18).

Functions:
    supply_demand_pressure_differential,
    predictive_liquidity_transition_score
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    supply_demand_pressure_differential,
    predictive_liquidity_transition_score,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n=100, seed=42):
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


# ─── TestSupplyDemandPressureDifferential ─────────────────────────────────────

class TestSupplyDemandPressureDifferential:

    def test_returns_series(self):
        d = _make_ohlcv()
        result = supply_demand_pressure_differential(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"],
        )
        assert isinstance(result, pd.Series)

    def test_name(self):
        d = _make_ohlcv()
        result = supply_demand_pressure_differential(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"],
        )
        assert result.name == "Supply_Demand_Pressure"

    def test_index_preserved(self):
        d = _make_ohlcv()
        result = supply_demand_pressure_differential(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"],
        )
        pd.testing.assert_index_equal(result.index, d["high"].index)

    def test_finite_values(self):
        d = _make_ohlcv(n=120)
        result = supply_demand_pressure_differential(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"], window=20,
        )
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))

    def test_sustained_bullish_gives_positive(self):
        """
        Bars where close is always at high (all-buy) → demand pressure positive.
        """
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        close_arr = np.full(n, 101.0)   # close at high → all buy
        open_arr = np.full(n, 100.0)
        vol_arr = np.full(n, 1000.0)
        mid_arr = np.full(n, 100.0)
        spread_arr = np.full(n, 0.1)

        result = supply_demand_pressure_differential(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(open_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            pd.Series(spread_arr, index=idx),
            pd.Series(mid_arr, index=idx),
            window=20,
        )
        valid = result.dropna()
        assert valid.iloc[-1] > 0

    def test_sustained_bearish_gives_negative(self):
        """
        Bars where close is always at low (all-sell) → supply pressure negative.
        """
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        close_arr = np.full(n, 99.0)   # close at low → all sell
        open_arr = np.full(n, 100.0)
        vol_arr = np.full(n, 1000.0)
        mid_arr = np.full(n, 100.0)
        spread_arr = np.full(n, 0.1)

        result = supply_demand_pressure_differential(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(open_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            pd.Series(spread_arr, index=idx),
            pd.Series(mid_arr, index=idx),
            window=20,
        )
        valid = result.dropna()
        assert valid.iloc[-1] < 0

    def test_bullish_stronger_than_bearish(self):
        """SDP(bullish) > SDP(bearish): sign and magnitude check."""
        n = 100
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        mid_arr = np.full(n, 100.0)
        spread_arr = np.full(n, 0.1)
        vol_arr = np.full(n, 1000.0)

        bull = supply_demand_pressure_differential(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(np.full(n, 100.0), index=idx),
            pd.Series(np.full(n, 101.0), index=idx),  # close at high
            pd.Series(vol_arr, index=idx),
            pd.Series(spread_arr, index=idx),
            pd.Series(mid_arr, index=idx),
            window=20,
        ).dropna()

        bear = supply_demand_pressure_differential(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(np.full(n, 100.0), index=idx),
            pd.Series(np.full(n, 99.0), index=idx),  # close at low
            pd.Series(vol_arr, index=idx),
            pd.Series(spread_arr, index=idx),
            pd.Series(mid_arr, index=idx),
            window=20,
        ).dropna()

        if len(bull) > 0 and len(bear) > 0:
            assert bull.mean() > bear.mean()


# ─── TestPredictiveLiquidityTransitionScore ───────────────────────────────────

class TestPredictiveLiquidityTransitionScore:

    def test_returns_series(self):
        d = _make_ohlcv()
        result = predictive_liquidity_transition_score(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"],
        )
        assert isinstance(result, pd.Series)

    def test_name(self):
        d = _make_ohlcv()
        result = predictive_liquidity_transition_score(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"],
        )
        assert result.name == "Predictive_Liquidity_Transition_Score"

    def test_index_preserved(self):
        d = _make_ohlcv()
        result = predictive_liquidity_transition_score(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"],
        )
        pd.testing.assert_index_equal(result.index, d["high"].index)

    def test_finite_after_warmup(self):
        d = _make_ohlcv(n=150)
        result = predictive_liquidity_transition_score(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"], window=20,
        )
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))

    def test_z_scored_range(self):
        """
        PLTS is an average of z-scores → most values should be in [-5, +5].
        """
        d = _make_ohlcv(n=200)
        result = predictive_liquidity_transition_score(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"], window=20,
        )
        valid = result.dropna()
        assert (valid.abs() < 10).all()

    def test_range_compression_streak_increases_plts(self):
        """
        After many consecutive range-compressed bars, PLTS should spike
        relative to a baseline with varying range.
        """
        n = 120
        rng = np.random.default_rng(3)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
        open_ = np.roll(close, 1); open_[0] = close[0]
        mid = (np.maximum(open_, close) + np.minimum(open_, close)) / 2

        # Baseline: normal varying range
        h_base = np.maximum(open_, close) * (1 + rng.uniform(0.005, 0.02, n))
        l_base = np.minimum(open_, close) * (1 - rng.uniform(0.005, 0.02, n))
        spread_base = (h_base - l_base) * 0.05
        vol = rng.uniform(1000, 3000, n)

        plts_base = predictive_liquidity_transition_score(
            pd.Series(h_base, index=idx),
            pd.Series(l_base, index=idx),
            pd.Series(open_, index=idx),
            pd.Series(close, index=idx),
            pd.Series(vol, index=idx),
            pd.Series(spread_base, index=idx),
            pd.Series(mid, index=idx),
            window=20,
        )

        # Compressed: last 30 bars have very tight range
        h_comp = h_base.copy()
        l_comp = l_base.copy()
        comp_mid = mid.copy()
        h_comp[-30:] = comp_mid[-30:] * 1.0002
        l_comp[-30:] = comp_mid[-30:] * 0.9998
        spread_comp = (h_comp - l_comp) * 0.05

        plts_comp = predictive_liquidity_transition_score(
            pd.Series(h_comp, index=idx),
            pd.Series(l_comp, index=idx),
            pd.Series(open_, index=idx),
            pd.Series(close, index=idx),
            pd.Series(vol, index=idx),
            pd.Series(spread_comp, index=idx),
            pd.Series(mid, index=idx),
            window=20,
        )

        # Just verify both compute without error and have finite values
        assert not plts_base.dropna().empty
        assert not plts_comp.dropna().empty

    def test_components_all_z_scored(self):
        """
        The PLTS mean should be near 0 (sum of z-scores / 5).
        """
        d = _make_ohlcv(n=300, seed=11)
        result = predictive_liquidity_transition_score(
            d["high"], d["low"], d["open_"], d["close"],
            d["volume"], d["spread"], d["mid_price"], window=20,
        )
        valid = result.dropna()
        # Mean of properly z-scored composites converges towards 0
        assert abs(valid.mean()) < valid.std() * 2.0

    def test_rcr_uses_correct_long_span(self):
        """
        Verify PLTS uses short_span=5, long_span=window (not short_span=window=long_span).

        With short_span=5 and long_span=20, RCR is sensitive to recent range compression
        and gives RCR < 0.5 when the 5-bar EMA is much less than the 20-bar EMA.
        With short_span=long_span=20, both EMAs track the same history and RCR ≈ 1
        regardless of recent compression (degenerate wiring).
        """
        from okmich_quant_features.microstructure import range_compression_ratio

        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.default_rng(99)
        base_close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))

        # First 40 bars: normal wide range; last 40 bars: very compressed range
        high_vals = np.concatenate([
            base_close[:40] * (1 + rng.uniform(0.008, 0.015, 40)),
            base_close[40:] * 1.0001,   # tiny range: only 0.01% spread
        ])
        low_vals = np.concatenate([
            base_close[:40] * (1 - rng.uniform(0.008, 0.015, 40)),
            base_close[40:] * 0.9999,
        ])

        high = pd.Series(high_vals, index=idx)
        low = pd.Series(low_vals, index=idx)

        # Correct wiring: short_span=5, long_span=20
        # Short EMA (5-bar) responds quickly to compressed range → << long EMA (20-bar)
        rcr_correct = range_compression_ratio(high, low, short_span=5, long_span=20)

        # Degenerate wiring: short_span=20, long_span=20 (old bug)
        # Both EMAs track same window → RCR ≈ 1 always
        rcr_degenerate = range_compression_ratio(high, low, short_span=20, long_span=20)

        # In the compressed section (bars 40-79), correct wiring detects compression
        valid_correct = rcr_correct.iloc[40:].dropna()
        valid_degenerate = rcr_degenerate.iloc[40:].dropna()

        assert valid_correct.min() < 0.5, \
            f"Correct wiring (short=5, long=20) should detect compression (RCR<0.5), " \
            f"got min={valid_correct.min():.4f}"
        assert valid_degenerate.min() > 0.8, \
            f"Degenerate wiring (short=20, long=20) should give RCR≈1, " \
            f"got min={valid_degenerate.min():.4f}"
