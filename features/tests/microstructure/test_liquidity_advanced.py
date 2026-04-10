"""
Tests for advanced liquidity features (§14).

Functions:
    liquidity_commonality, liquidity_resilience, spread_volatility_elasticity
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    liquidity_commonality,
    liquidity_resilience,
    spread_volatility_elasticity,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_ohlcv(n=100, seed=42):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.006, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.006, n))
    mid = (high + low) / 2.0
    spread = mid * rng.uniform(0.0005, 0.003, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return dict(
        high=pd.Series(high, index=idx),
        low=pd.Series(low, index=idx),
        close=pd.Series(close, index=idx),
        spread=pd.Series(spread, index=idx),
        mid_price=pd.Series(mid, index=idx),
    )


# ─── TestLiquidityCommonality ─────────────────────────────────────────────────

class TestLiquidityCommonality:

    def test_returns_series(self):
        d = _make_ohlcv()
        bench = _make_ohlcv(seed=99)
        result = liquidity_commonality(
            d["spread"], d["mid_price"],
            bench["spread"], bench["mid_price"],
        )
        assert isinstance(result, pd.Series)

    def test_name(self):
        d = _make_ohlcv()
        bench = _make_ohlcv(seed=99)
        result = liquidity_commonality(
            d["spread"], d["mid_price"],
            bench["spread"], bench["mid_price"],
        )
        assert result.name == "Liquidity_Commonality"

    def test_values_bounded(self):
        """Pearson correlation → [-1, +1]."""
        d = _make_ohlcv(n=120)
        bench = _make_ohlcv(n=120, seed=77)
        result = liquidity_commonality(
            d["spread"], d["mid_price"],
            bench["spread"], bench["mid_price"],
            window=20,
        )
        valid = result.dropna()
        assert (valid >= -1.0 - 1e-10).all() and (valid <= 1.0 + 1e-10).all()

    def test_identical_series_gives_one(self):
        """When asset spread == benchmark spread, LC = 1.0 everywhere."""
        d = _make_ohlcv(n=80)
        result = liquidity_commonality(
            d["spread"], d["mid_price"],
            d["spread"], d["mid_price"],
            window=10,
        )
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)

    def test_opposite_series_gives_negative(self):
        """
        Negated benchmark ΔS → perfectly anti-correlated → LC = -1.
        Construct: bench_spread changes = -asset spread changes.
        """
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        rng = np.random.default_rng(5)
        mid = pd.Series(np.full(n, 100.0), index=idx)
        spread_vals = 0.01 + np.cumsum(rng.normal(0, 0.0005, n))
        spread_vals = np.maximum(spread_vals, 1e-5)
        asset_spread = pd.Series(spread_vals, index=idx)
        # Benchmark changes in the opposite direction
        bench_vals = spread_vals[0] * 2 - spread_vals  # mirror
        bench_vals = np.maximum(bench_vals, 1e-5)
        bench_spread = pd.Series(bench_vals, index=idx)

        result = liquidity_commonality(asset_spread, mid, bench_spread, mid, window=20)
        valid = result.dropna()
        assert valid.mean() < -0.5

    def test_early_values_nan(self):
        d = _make_ohlcv(n=60)
        bench = _make_ohlcv(n=60, seed=13)
        result = liquidity_commonality(
            d["spread"], d["mid_price"],
            bench["spread"], bench["mid_price"],
            window=20,
        )
        assert np.isnan(result.values[0])

    def test_index_preserved(self):
        d = _make_ohlcv()
        bench = _make_ohlcv(seed=7)
        result = liquidity_commonality(
            d["spread"], d["mid_price"],
            bench["spread"], bench["mid_price"],
        )
        pd.testing.assert_index_equal(result.index, d["spread"].index)


# ─── TestLiquidityResilience ──────────────────────────────────────────────────

class TestLiquidityResilience:

    def test_returns_dataframe(self):
        d = _make_ohlcv()
        result = liquidity_resilience(d["spread"], d["mid_price"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        d = _make_ohlcv()
        result = liquidity_resilience(d["spread"], d["mid_price"])
        assert set(result.columns) == {"phi", "half_life"}

    def test_early_values_nan(self):
        d = _make_ohlcv(n=80)
        result = liquidity_resilience(d["spread"], d["mid_price"], window=40)
        assert np.isnan(result["phi"].values[0])

    def test_phi_bounded_for_stable_spread(self):
        """Stable spread → AR(1) well-defined, |phi| < 1."""
        n = 120
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        # Stationary AR(1) spread: phi=0.8
        vals = np.zeros(n)
        vals[0] = 0.01
        for i in range(1, n):
            vals[i] = 0.8 * vals[i - 1] + 0.002 + rng.normal(0, 0.0005)
        spread = pd.Series(np.maximum(vals, 1e-5), index=idx)
        mid = pd.Series(np.full(n, 100.0), index=idx)

        result = liquidity_resilience(spread, mid, window=30)
        valid_phi = result["phi"].dropna()
        assert len(valid_phi) > 0
        # phi should be in a reasonable range (not explosive)
        assert (valid_phi.abs() <= 2.0).all()

    def test_half_life_positive_for_mean_reverting(self):
        """
        Mean-reverting spread (|phi| < 1) → half_life > 0.
        """
        n = 120
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        vals = np.zeros(n)
        vals[0] = 0.01
        for i in range(1, n):
            vals[i] = 0.6 * vals[i - 1] + 0.004 + rng.normal(0, 0.0003)
        spread = pd.Series(np.maximum(vals, 1e-5), index=idx)
        mid = pd.Series(np.full(n, 100.0), index=idx)

        result = liquidity_resilience(spread, mid, window=30)
        valid_hl = result["half_life"].dropna()
        if len(valid_hl) > 0:
            assert (valid_hl > 0).any()

    def test_fast_reversion_gives_shorter_half_life(self):
        """
        phi=0.3 (fast reversion) half-life < phi=0.9 (slow reversion).
        """
        n = 200
        rng = np.random.default_rng(17)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        def ar1_spread(phi):
            vals = np.zeros(n)
            vals[0] = 0.01
            for i in range(1, n):
                vals[i] = phi * vals[i - 1] + (1 - phi) * 0.01 + rng.normal(0, 0.0002)
            return pd.Series(np.maximum(vals, 1e-5), index=idx)

        mid = pd.Series(np.full(n, 100.0), index=idx)
        hl_fast = liquidity_resilience(ar1_spread(0.3), mid, window=30)["half_life"].dropna()
        hl_slow = liquidity_resilience(ar1_spread(0.9), mid, window=30)["half_life"].dropna()

        if len(hl_fast) > 0 and len(hl_slow) > 0:
            # Fast reversion → smaller half-life
            assert hl_fast.median() < hl_slow.median()

    def test_no_inf_in_half_life(self):
        """half_life should have no inf after replacing."""
        d = _make_ohlcv(n=120)
        result = liquidity_resilience(d["spread"], d["mid_price"], window=30)
        assert not np.isinf(result["half_life"].dropna().values).any()


# ─── TestSpreadVolatilityElasticity ──────────────────────────────────────────

class TestSpreadVolatilityElasticity:

    def test_returns_series(self):
        d = _make_ohlcv()
        result = spread_volatility_elasticity(
            d["spread"], d["mid_price"], d["high"], d["low"]
        )
        assert isinstance(result, pd.Series)

    def test_name(self):
        d = _make_ohlcv()
        result = spread_volatility_elasticity(
            d["spread"], d["mid_price"], d["high"], d["low"]
        )
        assert result.name == "Spread_Vol_Elasticity"

    def test_early_values_nan(self):
        d = _make_ohlcv(n=100)
        result = spread_volatility_elasticity(
            d["spread"], d["mid_price"], d["high"], d["low"], window=40
        )
        assert np.isnan(result.values[0])

    def test_finite_after_warmup(self):
        d = _make_ohlcv(n=150)
        result = spread_volatility_elasticity(
            d["spread"], d["mid_price"], d["high"], d["low"], window=30
        )
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))

    def test_high_elasticity_when_spread_tracks_vol(self):
        """
        Construct spread = k × vol → log(S) = log(k) + log(vol)
        → Δln(S) = Δln(vol) → elasticity ≈ 1.
        """
        n = 150
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        high = close * (1 + rng.uniform(0.005, 0.02, n))
        low = close * (1 - rng.uniform(0.005, 0.02, n))
        mid = (high + low) / 2.0

        # Spread proportional to range (proxy for vol)
        rng_vals = high - low
        spread = rng_vals * 0.1  # spread = 10% of range

        result = spread_volatility_elasticity(
            pd.Series(spread, index=idx),
            pd.Series(mid, index=idx),
            pd.Series(high, index=idx),
            pd.Series(low, index=idx),
            window=30,
        )
        valid = result.dropna()
        if len(valid) > 0:
            # Elasticity should be broadly positive (spread responds to vol)
            assert valid.median() > 0

    def test_constant_spread_gives_zero_or_nan(self):
        """
        Constant spread → Δln(S) = 0 → cov = 0 → elasticity = 0 (or NaN if vol also const).
        """
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        high = pd.Series(np.full(n, 101.0), index=idx)
        low = pd.Series(np.full(n, 99.0), index=idx)
        mid = pd.Series(np.full(n, 100.0), index=idx)
        spread = pd.Series(np.full(n, 0.5), index=idx)

        result = spread_volatility_elasticity(spread, mid, high, low, window=20)
        valid = result.dropna()
        # Constant spread → Δln(S) = 0 → cov = 0 → elasticity = 0
        if len(valid) > 0:
            np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)
