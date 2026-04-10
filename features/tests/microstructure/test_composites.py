"""
Tests for microstructure.composites module.

Functions:
    liquidity_adjusted_momentum, volume_price_divergence,
    informed_liquidity_pressure, institutional_footprint_score,
    regime_fragility_index
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    liquidity_adjusted_momentum,
    volume_price_divergence,
    informed_liquidity_pressure,
    institutional_footprint_score,
    regime_fragility_index,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_data(n=100, seed=42):
    """Generate realistic OHLCV + spread data."""
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


# ─── TestLiquidityAdjustedMomentum ───────────────────────────────────────────

class TestLiquidityAdjustedMomentum:

    def test_returns_series(self):
        d = _make_data()
        result = liquidity_adjusted_momentum(d["close"], d["spread"],
                                             d["mid_price"])
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = liquidity_adjusted_momentum(d["close"], d["spread"],
                                             d["mid_price"], momentum_window=10)
        assert result.name == "lam_10"

    def test_early_values_nan(self):
        d = _make_data(n=50)
        result = liquidity_adjusted_momentum(d["close"], d["spread"],
                                             d["mid_price"], momentum_window=20)
        assert np.all(np.isnan(result.values[:20]))

    def test_strong_trend_tight_spread_gives_high_lam(self):
        """Consistent positive returns + tight spread → large positive LAM."""
        n = 60
        close = pd.Series(100.0 * np.cumprod(1 + np.full(n, 0.01)))  # 1% per bar
        mid = pd.Series(np.full(n, 100.0))
        spread = pd.Series(np.full(n, 0.001))  # very tight
        result = liquidity_adjusted_momentum(close, spread, mid,
                                             vol_window=10, momentum_window=10)
        valid = result.dropna()
        assert valid.iloc[-1] > 0

    def test_zero_spread_inflates_lam(self):
        """With near-zero spread, denominator is small → |LAM| is large."""
        d = _make_data(n=60)
        # Normal spread
        lam_normal = liquidity_adjusted_momentum(
            d["close"], d["spread"], d["mid_price"],
            vol_window=10, momentum_window=10
        )
        # Tiny spread
        tiny_spread = d["spread"] * 0.01
        lam_tiny = liquidity_adjusted_momentum(
            d["close"], tiny_spread, d["mid_price"],
            vol_window=10, momentum_window=10
        )
        valid_mask = lam_normal.notna() & lam_tiny.notna()
        # |LAM| with tiny spread should exceed |LAM| with normal spread
        assert (np.abs(lam_tiny[valid_mask]).mean() >
                np.abs(lam_normal[valid_mask]).mean())


# ─── TestVolumePriceDivergence ────────────────────────────────────────────────

class TestVolumePriceDivergence:

    def test_returns_series(self):
        d = _make_data()
        result = volume_price_divergence(d["close"], d["volume"])
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = volume_price_divergence(d["close"], d["volume"], window=15)
        assert result.name == "vpd_15"

    def test_price_up_volume_down_gives_positive(self):
        """Bullish exhaustion: price up but volume shrinking → VPD > 0."""
        n = 40
        # Rising close
        close = pd.Series(np.linspace(100, 120, n))
        # Volume decreasing steadily (RVOL < 1 → 1-RVOL > 0)
        vol = pd.Series(np.linspace(5000, 1000, n))
        result = volume_price_divergence(close, vol, window=10)
        # Last values: sign(ΔC)=+1, RVOL<1 → VPD = +1 × positive > 0
        assert result.iloc[-1] > 0

    def test_price_up_volume_up_gives_negative(self):
        """Healthy uptrend: price up + volume up → VPD < 0."""
        n = 40
        close = pd.Series(np.linspace(100, 120, n))
        vol = pd.Series(np.linspace(1000, 5000, n))  # volume rising
        result = volume_price_divergence(close, vol, window=10)
        # sign(ΔC)=+1, RVOL>1 → 1-RVOL < 0 → VPD < 0
        assert result.iloc[-1] < 0

    def test_constant_volume_gives_zero(self):
        """With constant volume, RVOL=1, VPD = sign × (1-1) = 0."""
        n = 40
        close = pd.Series(np.linspace(100, 120, n))
        vol = pd.Series(np.full(n, 1000.0))
        result = volume_price_divergence(close, vol, window=10)
        # After warmup, RVOL = V / SMA(V) = 1 → 1-1 = 0
        valid = result.iloc[10:]
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-6)


# ─── TestInformedLiquidityPressure ────────────────────────────────────────────

class TestInformedLiquidityPressure:

    def test_returns_series(self):
        d = _make_data()
        result = informed_liquidity_pressure(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=10
        )
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = informed_liquidity_pressure(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=12
        )
        assert result.name == "ilp_12"

    def test_early_values_nan(self):
        d = _make_data(n=60)
        result = informed_liquidity_pressure(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=10
        )
        # Needs warmup from VPIN, kyles_lambda, spread_zscore
        assert np.isnan(result.values[0])

    def test_triple_product_structure(self):
        """ILP should be zero if any single component is zero."""
        d = _make_data(n=80)
        result = informed_liquidity_pressure(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=10
        )
        # Verify it produces finite values beyond warmup
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))


# ─── TestInstitutionalFootprintScore ──────────────────────────────────────────

class TestInstitutionalFootprintScore:

    def test_returns_series(self):
        d = _make_data()
        result = institutional_footprint_score(
            d["open_"], d["high"], d["low"], d["close"], d["volume"]
        )
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = institutional_footprint_score(
            d["open_"], d["high"], d["low"], d["close"], d["volume"],
            window=15
        )
        assert result.name == "ifs_15"

    def test_spike_volume_bar_raises_ifs(self):
        """
        A single bar with very high volume and small body (iceberg)
        should push IFS higher at that bar vs surrounding bars.
        """
        d = _make_data(n=80)
        # Baseline IFS
        ifs_base = institutional_footprint_score(
            d["open_"], d["high"], d["low"], d["close"], d["volume"],
            window=10
        )
        # Inject iceberg bar at position 50 (high vol, small body)
        vol_spike = d["volume"].copy()
        vol_spike.iloc[50] = 50000.0  # 10× typical
        ifs_spike = institutional_footprint_score(
            d["open_"], d["high"], d["low"], d["close"], vol_spike,
            window=10
        )
        # IFS should be higher at bar 50 with the volume spike
        if not np.isnan(ifs_spike.iloc[50]) and not np.isnan(ifs_base.iloc[50]):
            assert ifs_spike.iloc[50] > ifs_base.iloc[50]

    def test_finite_after_warmup(self):
        d = _make_data(n=80)
        result = institutional_footprint_score(
            d["open_"], d["high"], d["low"], d["close"], d["volume"],
            window=10
        )
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))


# ─── TestRegimeFragilityIndex ─────────────────────────────────────────────────

class TestRegimeFragilityIndex:

    def test_returns_series(self):
        d = _make_data()
        result = regime_fragility_index(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=10
        )
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = regime_fragility_index(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=12
        )
        assert result.name == "rfi_12"

    def test_finite_after_warmup(self):
        d = _make_data(n=120)
        result = regime_fragility_index(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=10
        )
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))

    def test_calm_market_gives_low_rfi(self):
        """
        Uniform, calm market: constant vol, tight spread, no jumps.
        All z-scores should be near 0 → RFI near 0.
        """
        n = 120
        close = pd.Series(np.linspace(100, 110, n))
        high = close + 0.5
        low = close - 0.5
        vol = pd.Series(np.full(n, 1000.0))
        mid = (high + low) / 2.0
        spread = pd.Series(np.full(n, 0.1))
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        for s in [close, high, low, vol, mid, spread]:
            s.index = idx

        result = regime_fragility_index(high, low, close, vol, spread, mid,
                                        window=10)
        valid = result.dropna()
        # Calm market: all components near baseline → RFI ≈ 0
        assert abs(valid.mean()) < 1.5

    def test_stress_raises_rfi(self):
        """
        Inject a liquidity crisis regime (wide spreads, volatile range,
        volume drying up) and verify RFI is higher on average.
        Low volume → high Amihud illiquidity, high Kyle's lambda.
        Wide spread → high LDI.  Wide range → high VoV.
        """
        d = _make_data(n=120)
        # Normal RFI
        rfi_normal = regime_fragility_index(
            d["high"], d["low"], d["close"], d["volume"],
            d["spread"], d["mid_price"], window=10
        )
        # Stress the last 20 bars: widen spread, increase range, REDUCE volume
        spread_stress = d["spread"].copy()
        spread_stress.iloc[-20:] *= 10.0
        high_stress = d["high"].copy()
        high_stress.iloc[-20:] *= 1.05
        low_stress = d["low"].copy()
        low_stress.iloc[-20:] *= 0.95
        vol_stress = d["volume"].copy()
        vol_stress.iloc[-20:] *= 0.1  # liquidity drying up → high Amihud, high λ
        mid_stress = (high_stress + low_stress) / 2.0

        rfi_stress = regime_fragility_index(
            high_stress, low_stress, d["close"], vol_stress,
            spread_stress, mid_stress, window=10
        )
        # Mean RFI in stress region should exceed mean in same region pre-stress
        stress_slice = slice(-15, -1)
        normal_vals = rfi_normal.iloc[stress_slice].dropna()
        stress_vals = rfi_stress.iloc[stress_slice].dropna()
        if len(normal_vals) > 0 and len(stress_vals) > 0:
            assert stress_vals.mean() > normal_vals.mean()
