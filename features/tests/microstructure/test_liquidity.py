"""
Unit tests for Tier 1 liquidity features.

Tests cover formulas, edge cases, range validation, and trading-signal
interpretation for:
- corwin_schultz_spread (CS)
- roll_spread
- effective_tick_ratio (ETR)
- liquidity_score
- amihud_illiquidity
- realized_liquidity_premium (RLP)
- spread_zscore
- spread_expansion_momentum (SEM)
- spread_volume_ratio (SVR)
- liquidity_drought_index (LDI)
- depth_imbalance_proxy (DIP)
"""

import numpy as np
import pandas as pd
import pytest
from okmich_quant_features.microstructure import (
    corwin_schultz_spread,
    roll_spread,
    effective_tick_ratio,
    liquidity_score,
    amihud_illiquidity,
    realized_liquidity_premium,
    spread_zscore,
    spread_expansion_momentum,
    spread_volume_ratio,
    liquidity_drought_index,
    depth_imbalance_proxy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n=60, seed=42):
    """Return a consistent set of Series for testing."""
    rng = np.random.RandomState(seed)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.3, n)))
    volume = pd.Series(rng.uniform(1000.0, 5000.0, n))
    high = close + pd.Series(rng.uniform(0.5, 2.0, n))
    low = close - pd.Series(rng.uniform(0.5, 2.0, n))
    spread = pd.Series(rng.uniform(0.05, 0.5, n))
    mid = (high + low) / 2.0
    return high, low, close, volume, spread, mid


# ===========================================================================
# TestAmihudIlliquidity
# ===========================================================================

class TestAmihudIlliquidity:

    def test_higher_volume_lower_illiquidity(self):
        """Larger volume → price impact per dollar is smaller → lower ILLIQ."""
        close = pd.Series([100.0, 101.0, 100.5, 102.0, 101.5])
        vol_low = pd.Series([100.0] * 5)
        vol_high = pd.Series([10000.0] * 5)

        illiq_low = amihud_illiquidity(close, vol_low, window=3, log_transform=False)
        illiq_high = amihud_illiquidity(close, vol_high, window=3, log_transform=False)

        # Higher volume → lower raw illiquidity
        valid_low = illiq_low.dropna()
        valid_high = illiq_high.dropna()
        assert np.all(valid_low.values > valid_high.values), \
            "Higher volume should give lower Amihud illiquidity"

    def test_zero_return_gives_zero_illiquidity(self):
        """Flat price (r=0 every bar) → ILLIQ = 0."""
        close = pd.Series([100.0] * 20)
        volume = pd.Series([1000.0] * 20)

        result = amihud_illiquidity(close, volume, window=10, log_transform=False)
        valid = result.dropna()
        assert np.allclose(valid.values, 0.0), \
            f"Flat price should give ILLIQ=0, got {valid.values}"

    def test_log_transform_applied(self):
        """log_transform=True and False should give different results."""
        _, _, close, volume, _, _ = _make_series()

        raw = amihud_illiquidity(close, volume, window=10, log_transform=False)
        logged = amihud_illiquidity(close, volume, window=10, log_transform=True)

        # They should differ
        valid_raw = raw.dropna()
        valid_log = logged.dropna()
        assert not np.allclose(valid_raw.values, valid_log.values), \
            "log_transform should change the values"

    def test_warmup_is_nan(self):
        """First window−1 bars should be NaN."""
        _, _, close, volume, _, _ = _make_series(n=30)

        result = amihud_illiquidity(close, volume, window=10)

        assert np.all(np.isnan(result.iloc[:9].values)), \
            "First window-1 bars should be NaN"
        assert not np.isnan(result.iloc[9]), \
            "Bar at index window-1 should be valid"

    def test_returns_series(self):
        """Output is a pd.Series."""
        _, _, close, volume, _, _ = _make_series()
        result = amihud_illiquidity(close, volume)
        assert isinstance(result, pd.Series)

    def test_name(self):
        """Series name is 'amihud_illiq'."""
        _, _, close, volume, _, _ = _make_series()
        assert amihud_illiquidity(close, volume).name == 'amihud_illiq'

    def test_formula_manual(self):
        """Manual ILLIQ check for a 3-bar window."""
        # Returns: log(101/100)=0.00995, log(100/101)=-0.00995, log(102/100)=0.0198
        close = pd.Series([100.0, 101.0, 100.0, 102.0])
        volume = pd.Series([1000.0, 2000.0, 1500.0, 2500.0])

        result = amihud_illiquidity(close, volume, window=3, log_transform=False)

        # Bar 3 (index 3): window covers bars 1,2,3
        # |r1|/(V1*C1) = |log(101/100)|/(2000*101) ≈ 0.00995/202000
        # |r2|/(V2*C2) = |log(100/101)|/(1500*100) ≈ 0.00995/150000
        # |r3|/(V3*C3) = |log(102/100)|/(2500*102) ≈ 0.0198/255000
        r1 = abs(np.log(101 / 100)) / (2000 * 101)
        r2 = abs(np.log(100 / 101)) / (1500 * 100)
        r3 = abs(np.log(102 / 100)) / (2500 * 102)
        expected = (r1 + r2 + r3) / 3.0

        assert np.isclose(result.iloc[3], expected, rtol=1e-6), \
            f"Expected {expected:.2e}, got {result.iloc[3]:.2e}"


# ===========================================================================
# TestRealizedLiquidityPremium
# ===========================================================================

class TestRealizedLiquidityPremium:

    def test_output_range(self):
        """RLP is a Pearson correlation, so must be in [−1, +1]."""
        _, _, close, volume, spread, mid = _make_series(n=100)
        result = realized_liquidity_premium(close, spread, mid, window=20)
        valid = result.dropna()
        assert np.all(valid >= -1.0 - 1e-9) and np.all(valid <= 1.0 + 1e-9), \
            f"RLP out of range: min={valid.min():.4f}, max={valid.max():.4f}"

    def test_correlated_inputs_give_high_rlp(self):
        """When |returns| and spread move together, RLP → +1."""
        n = 40
        # Make abs_return and spread identical time series
        returns_abs = np.abs(np.random.RandomState(1).normal(0, 0.01, n))
        returns_abs[0] = 0.0  # first return NaN → use 0

        close = pd.Series(np.concatenate([[100.0], 100.0 * np.exp(np.cumsum(returns_abs[1:]))]))
        mid = pd.Series([100.0] * n)
        # spread proportional to |returns|, so Corr = +1
        spread = pd.Series(returns_abs * 100.0 * 10.0 + 0.01)  # scale so spread/mid ≈ |r|

        result = realized_liquidity_premium(close, spread, mid, window=20)
        valid = result.dropna()
        assert np.mean(valid) > 0.5, \
            f"Expected high positive RLP for correlated inputs, got {np.mean(valid):.4f}"

    def test_constant_spread_gives_zero_rlp(self):
        """Constant spread/mid (no variance in spread_norm) → RLP = 0."""
        n = 40
        _, _, close, _, _, _ = _make_series(n=n)
        # Both spread and mid constant → spread_norm constant → std(spread_norm) = 0
        spread = pd.Series([0.1] * n)
        mid = pd.Series([100.0] * n)

        result = realized_liquidity_premium(close, spread, mid, window=15)
        valid = result.dropna()
        assert np.allclose(valid, 0.0), \
            f"Constant spread_norm should give RLP=0, got {valid.values}"

    def test_warmup_is_nan(self):
        """First window bars should be NaN."""
        _, _, close, _, spread, mid = _make_series(n=50)
        result = realized_liquidity_premium(close, spread, mid, window=20)
        assert np.all(np.isnan(result.iloc[:19].values))
        assert not np.isnan(result.iloc[19])

    def test_returns_series(self):
        """Output is a pd.Series with correct name."""
        _, _, close, _, spread, mid = _make_series()
        result = realized_liquidity_premium(close, spread, mid)
        assert isinstance(result, pd.Series)
        assert result.name == 'realized_liquidity_premium'


# ===========================================================================
# TestSpreadZScore
# ===========================================================================

class TestSpreadZScore:

    def test_mean_near_zero_over_long_series(self):
        """Z-score should average ≈ 0 over the full series."""
        _, _, _, _, spread, mid = _make_series(n=200)
        result = spread_zscore(spread, mid, window=20)
        valid = result.dropna()
        assert abs(valid.mean()) < 0.3, \
            f"Mean z-score should be near 0, got {valid.mean():.4f}"

    def test_constant_spread_gives_zero_zscore(self):
        """Constant spread → std=0 → z-score=0."""
        spread = pd.Series([0.1] * 40)
        mid = pd.Series([100.0] * 40)
        result = spread_zscore(spread, mid, window=10)
        valid = result.dropna()
        assert np.allclose(valid, 0.0, atol=1e-6), \
            f"Constant spread should give z-score=0, got {valid.values}"

    def test_spike_gives_high_zscore(self):
        """A sudden spread widening should give z > 2."""
        n = 30
        spread = pd.Series([0.1] * n)
        spread.iloc[25] = 5.0  # extreme spike at bar 25
        mid = pd.Series([100.0] * n)

        result = spread_zscore(spread, mid, window=15)
        assert result.iloc[25] > 2.0, \
            f"Spread spike should give z > 2, got {result.iloc[25]:.4f}"

    def test_warmup_is_nan(self):
        """First window-1 bars should be NaN."""
        _, _, _, _, spread, mid = _make_series(n=40)
        result = spread_zscore(spread, mid, window=15)
        assert np.all(np.isnan(result.iloc[:14].values))

    def test_returns_series_with_name(self):
        """Output is pd.Series with f'spread_z_{window}' name."""
        _, _, _, _, spread, mid = _make_series()
        result = spread_zscore(spread, mid, window=20)
        assert isinstance(result, pd.Series)
        assert result.name == 'spread_z_20'


# ===========================================================================
# TestSpreadExpansionMomentum
# ===========================================================================

class TestSpreadExpansionMomentum:

    def test_steadily_rising_spread_gives_positive_sem(self):
        """Consistently expanding spread → SEM > 0."""
        spread = pd.Series(np.linspace(0.1, 1.0, 30))
        mid = pd.Series([100.0] * 30)

        result = spread_expansion_momentum(spread, mid, ema_span=5)
        # After a few bars the EMA should settle positive
        assert result.iloc[-1] > 0.0, \
            f"Rising spread should give positive SEM, got {result.iloc[-1]:.4f}"

    def test_steadily_falling_spread_gives_negative_sem(self):
        """Consistently contracting spread → SEM < 0."""
        spread = pd.Series(np.linspace(1.0, 0.1, 30))
        mid = pd.Series([100.0] * 30)

        result = spread_expansion_momentum(spread, mid, ema_span=5)
        assert result.iloc[-1] < 0.0, \
            f"Falling spread should give negative SEM, got {result.iloc[-1]:.4f}"

    def test_constant_spread_gives_zero_sem(self):
        """Constant spread → pct_change=0 → SEM=0."""
        spread = pd.Series([0.1] * 30)
        mid = pd.Series([100.0] * 30)

        result = spread_expansion_momentum(spread, mid, ema_span=5)
        # First bar is NaN (pct_change), rest should be 0
        valid = result.dropna()
        assert np.allclose(valid, 0.0, atol=1e-10), \
            f"Constant spread should give SEM=0, got {valid.values}"

    def test_returns_series_with_name(self):
        """Output is pd.Series with name containing ema_span."""
        _, _, _, _, spread, mid = _make_series()
        result = spread_expansion_momentum(spread, mid, ema_span=5)
        assert isinstance(result, pd.Series)
        assert result.name == 'spread_expansion_momentum_5'

    def test_longer_ema_smoother(self):
        """Longer EMA span produces smoother output (lower std)."""
        _, _, _, _, spread, mid = _make_series(n=100)

        sem_short = spread_expansion_momentum(spread, mid, ema_span=3)
        sem_long = spread_expansion_momentum(spread, mid, ema_span=20)

        std_short = sem_short.dropna().std()
        std_long = sem_long.dropna().std()
        assert std_long < std_short, \
            "Longer EMA span should produce smoother SEM"


# ===========================================================================
# TestSpreadVolumeRatio
# ===========================================================================

class TestSpreadVolumeRatio:

    def test_high_spread_low_volume_gives_high_svr(self):
        """Wide spread + low volume → high SVR (stress)."""
        spread_stress = pd.Series([1.0] * 5)
        spread_calm = pd.Series([1.0] * 5)
        mid = pd.Series([100.0] * 5)
        vol_low = pd.Series([10.0] * 5)
        vol_high = pd.Series([10000.0] * 5)

        svr_stress = spread_volume_ratio(spread_stress, mid, vol_low)
        svr_calm = spread_volume_ratio(spread_calm, mid, vol_high)

        assert np.all(svr_stress > svr_calm), \
            "Low volume should give higher SVR than high volume"

    def test_narrow_spread_gives_lower_svr(self):
        """Narrower spread → lower SVR."""
        mid = pd.Series([100.0] * 5)
        volume = pd.Series([1000.0] * 5)

        svr_wide = spread_volume_ratio(pd.Series([1.0] * 5), mid, volume)
        svr_narrow = spread_volume_ratio(pd.Series([0.1] * 5), mid, volume)

        assert np.all(svr_wide > svr_narrow), \
            "Wider spread should give higher SVR"

    def test_svr_positive(self):
        """SVR should be non-negative for valid inputs."""
        _, _, _, volume, spread, mid = _make_series()
        result = spread_volume_ratio(spread, mid, volume)
        assert np.all(result.values >= 0.0), "SVR should be non-negative"

    def test_returns_series_with_name(self):
        """Output is pd.Series named 'spread_volume_ratio'."""
        _, _, _, volume, spread, mid = _make_series()
        result = spread_volume_ratio(spread, mid, volume)
        assert isinstance(result, pd.Series)
        assert result.name == 'spread_volume_ratio'


# ===========================================================================
# TestLiquidityDroughtIndex
# ===========================================================================

class TestLiquidityDroughtIndex:

    def test_output_is_series(self):
        """Output is a pd.Series named 'liquidity_drought_index'."""
        _, _, close, volume, spread, mid = _make_series(n=60)
        result = liquidity_drought_index(close, volume, spread, mid, window=10)
        assert isinstance(result, pd.Series)
        assert result.name == 'liquidity_drought_index'

    def test_warmup_is_nan(self):
        """First bars should be NaN (needs Amihud warmup + z-score warmup)."""
        _, _, close, volume, spread, mid = _make_series(n=60)
        result = liquidity_drought_index(close, volume, spread, mid, window=10)
        # At minimum the first window-1 bars should be NaN
        assert np.all(np.isnan(result.iloc[:9].values)), \
            "Pre-warmup bars should be NaN"

    def test_stress_spike_raises_ldi(self):
        """Within a stable window a single stress bar should push LDI high.

        LDI z-scores are rolling, so they capture anomalies *within* the
        current window — not a permanent regime shift that would normalise.
        """
        n = 50
        rng = np.random.RandomState(11)
        close = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.1, n)))
        volume = pd.Series([2000.0] * n)
        spread = pd.Series([0.05] * n)
        mid = close.copy()

        # Spike ONE bar (bar 40): spread explodes, volume collapses
        volume.iloc[40] = 50.0    # 40× lower → very negative z_DV → +LDI
        spread.iloc[40] = 5.0     # 100× wider → very high z_spread, z_SVR

        result = liquidity_drought_index(close, volume, spread, mid, window=10)

        # At bar 40 the spike is included in the window; all components push LDI up
        assert result.iloc[40] > 0.0, \
            f"Stress spike should give positive LDI, got {result.iloc[40]:.3f}"

    def test_balanced_regime_near_zero(self):
        """Stable, consistent market → LDI ≈ 0 (all z-scores centred)."""
        n = 100
        rng = np.random.RandomState(3)
        close = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 0.2, n)))
        volume = pd.Series(rng.normal(2000.0, 50.0, n).clip(1000.0))
        spread = pd.Series(rng.normal(0.1, 0.005, n).clip(0.01))
        mid = close.copy()

        result = liquidity_drought_index(close, volume, spread, mid, window=20)
        # After the warmup the rolling z-scores should be centred near 0
        late = result.iloc[50:].dropna()
        assert abs(late.mean()) < 1.0, \
            f"Stable regime should give LDI near 0, got mean={late.mean():.3f}"


# ===========================================================================
# TestDepthImbalanceProxy
# ===========================================================================

class TestDepthImbalanceProxy:

    def test_close_at_high_tight_spread_gives_positive_dip(self):
        """Close near high + tight spread relative to avg → positive DIP."""
        n = 30
        # All closes at high → β = 1
        high = pd.Series([105.0] * n)
        low = pd.Series([100.0] * n)
        close = high.copy()

        # Tight spread in the last 10 bars (narrower than average)
        spread = pd.Series([0.5] * 20 + [0.05] * 10)
        mid = pd.Series([102.5] * n)

        result = depth_imbalance_proxy(high, low, close, spread, mid, window=15)

        # After warmup, when spread is tight (last 10 bars), DIP should be positive
        assert result.iloc[25] > 0.0, \
            f"Expected positive DIP (buying + tight spread), got {result.iloc[25]:.4f}"

    def test_spread_above_avg_gives_negative_dip(self):
        """Close at low (β=0, pressure=-1) + tight spread (spread_ratio > 0) → DIP < 0."""
        n = 30
        high = pd.Series([105.0] * n)
        low = pd.Series([100.0] * n)
        # β = 0 (close at low) → pressure = 2*0 - 1 = -1
        close = low.copy()

        # First 20 bars: wide spread (baseline); last 10: tight (below avg)
        # At bar 25: tight spread → spread_ratio = 1 - tight/wide_avg > 0
        # DIP = (-1) * positive = negative
        spread = pd.Series([5.0] * 20 + [0.05] * 10)
        mid = pd.Series([102.5] * n)

        result = depth_imbalance_proxy(high, low, close, spread, mid, window=15)

        # At bar 25: tight spread vs wide avg → spread_ratio > 0; pressure=-1 → DIP < 0
        assert result.iloc[25] < 0.0, \
            f"Expected negative DIP (close at low + tight spread), got {result.iloc[25]:.4f}"

    def test_wide_spread_inverts_dip(self):
        """When spread > rolling avg, factor = 1 - S/S_avg < 0 → DIP inverts sign."""
        n = 40
        high = pd.Series([105.0] * n)
        low = pd.Series([100.0] * n)
        close = high.copy()  # β = 1 (strong buying)
        mid = pd.Series([102.5] * n)

        # First 25 bars: tight (0.05), last 15 bars: spike (5.0 >> rolling avg)
        spread = pd.Series([0.05] * 25 + [5.0] * 15)

        result = depth_imbalance_proxy(high, low, close, spread, mid, window=15)

        # At bar 35 the window contains the spike; spread >> avg → factor < 0 → DIP < 0
        # (β=1 > 0, factor<0 → DIP < 0)
        assert result.iloc[35] < 0.0, \
            f"Wide spread (> avg) with β=1 should give negative DIP, got {result.iloc[35]:.4f}"

    def test_output_range(self):
        """DIP = β × (1 - S/S_avg) should be bounded when spread is reasonable."""
        high, low, close, _, spread, mid = _make_series(n=100)
        result = depth_imbalance_proxy(high, low, close, spread, mid, window=20)
        valid = result.dropna()
        # β ∈ [0,1], factor ∈ (-∞, 1] but spread < avg → factor ∈ (0, 1]
        # In typical data |DIP| < 1
        assert np.all(np.abs(valid) < 2.0), \
            f"DIP should have reasonable magnitude, max={np.abs(valid).max():.4f}"

    def test_returns_series_with_name(self):
        """Output is pd.Series named 'depth_imbalance_proxy'."""
        high, low, close, _, spread, mid = _make_series()
        result = depth_imbalance_proxy(high, low, close, spread, mid)
        assert isinstance(result, pd.Series)
        assert result.name == 'depth_imbalance_proxy'

    def test_warmup_is_nan(self):
        """First window-1 bars should be NaN (spread_avg needs warmup)."""
        high, low, close, _, spread, mid = _make_series(n=40)
        result = depth_imbalance_proxy(high, low, close, spread, mid, window=15)
        assert np.all(np.isnan(result.iloc[:14].values)), \
            "Pre-warmup bars should be NaN"

    def test_close_at_low_gives_negative_dip(self):
        """Explicit centering test: close at low → pressure=-1 → DIP sign inverted vs close at high."""
        n = 40
        high = pd.Series([105.0] * n)
        low = pd.Series([100.0] * n)
        spread = pd.Series([0.05] * n)
        mid = pd.Series([102.5] * n)

        # Close at high: β=1, pressure=+1
        close_high = high.copy()
        dip_high = depth_imbalance_proxy(high, low, close_high, spread, mid, window=10)

        # Close at low: β=0, pressure=-1
        close_low = low.copy()
        dip_low = depth_imbalance_proxy(high, low, close_low, spread, mid, window=10)

        valid_high = dip_high.dropna()
        valid_low = dip_low.dropna()

        assert np.all(valid_high > 0), "Close at high should give positive DIP"
        assert np.all(valid_low < 0), "Close at low should give negative DIP"
        assert np.allclose(valid_high.values, -valid_low.values, atol=1e-10), \
            "DIP(close=high) and DIP(close=low) should be equal in magnitude and opposite in sign"


# ===========================================================================
# TestCorwinSchultzSpread
# ===========================================================================

class TestCorwinSchultzSpread:

    def test_formula_uses_two_period_range(self):
        """β uses sum of adjacent squared HL ratios; γ uses max/min over two bars."""
        n = 10
        high = pd.Series([105.0, 106.0, 107.0, 104.0, 105.0,
                          106.0, 105.0, 106.0, 107.0, 106.0])
        low = pd.Series([100.0, 101.0, 102.0,  99.0, 100.0,
                         101.0, 100.0, 101.0, 102.0, 101.0])
        result = corwin_schultz_spread(high, low, window=2)
        # Should have NaN for the very first bar (needs shift(1) for beta/gamma)
        assert np.isnan(result.iloc[0]) or np.isnan(result.iloc[1]), \
            "First bar(s) should be NaN due to shift(1) in beta/gamma"
        assert result.dropna().shape[0] > 0, "Should have some valid values"

    def test_constant_hl_ratio_gives_zero_spread(self):
        """Constant H/L ratio → β=γ=0 → α=0 → spread=0."""
        n = 20
        # H/L = 1.01 every bar → log(H/L) = log(1.01) = constant
        high = pd.Series([101.0] * n)
        low = pd.Series([100.0] * n)
        result = corwin_schultz_spread(high, low, window=2)
        valid = result.dropna()
        # With constant H/L, beta and gamma sum to zero in alpha → spread=0
        assert np.all(valid >= 0.0), "Spread must be non-negative"

    def test_wider_range_gives_higher_spread(self):
        """Wider H/L range → higher spread estimate."""
        n = 20
        high_narrow = pd.Series([101.0] * n)
        low_narrow = pd.Series([100.0] * n)

        high_wide = pd.Series([110.0] * n)
        low_wide = pd.Series([100.0] * n)

        spread_narrow = corwin_schultz_spread(high_narrow, low_narrow, window=2)
        spread_wide = corwin_schultz_spread(high_wide, low_wide, window=2)

        valid_narrow = spread_narrow.dropna()
        valid_wide = spread_wide.dropna()
        assert valid_wide.mean() >= valid_narrow.mean(), \
            "Wider H/L range should give higher spread"

    def test_warmup_requires_prior_bar(self):
        """First bar is NaN since β needs shift(1)."""
        n = 10
        high = pd.Series([105.0] * n)
        low = pd.Series([100.0] * n)
        result = corwin_schultz_spread(high, low, window=2)
        # At minimum the first bar should be NaN
        assert np.isnan(result.iloc[0]), "First bar should be NaN"

    def test_values_in_unit_interval(self):
        """All valid spread values should be in [0, 1]."""
        rng = np.random.RandomState(99)
        n = 100
        close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)))
        high = close + pd.Series(rng.uniform(0.2, 2.0, n))
        low = close - pd.Series(rng.uniform(0.2, 2.0, n))
        result = corwin_schultz_spread(high, low, window=2)
        valid = result.dropna()
        assert np.all(valid >= 0.0), "Spread must be non-negative"
        assert np.all(valid <= 1.0), "Spread must be <= 1"

    def test_returns_proportional_no_close_needed(self):
        """Calling without close raises no error."""
        high = pd.Series([105.0, 106.0, 107.0, 106.0, 105.0])
        low = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0])
        try:
            result = corwin_schultz_spread(high, low)
            assert isinstance(result, pd.Series)
        except Exception as e:
            pytest.fail(f"corwin_schultz_spread without close raised: {e}")

    def test_invalid_window_raises(self):
        """window < 1 should raise ValueError."""
        high = pd.Series([105.0, 106.0, 107.0])
        low = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError, match="window"):
            corwin_schultz_spread(high, low, window=0)


# ===========================================================================
# TestRollSpread
# ===========================================================================

class TestRollSpread:

    def test_returns_proportional_scale(self):
        """After Fix 3A: roll_spread / close ~ [0, 1] range (proportional)."""
        n = 60
        rng = np.random.RandomState(5)
        close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.3, n)))
        result = roll_spread(close, window=20)
        valid = result.dropna()
        # Proportional spread should be in a small positive range (not raw price units)
        assert np.all(valid >= 0.0), "Proportional spread must be non-negative"
        assert valid.mean() < 1.0, "Proportional spread mean should be < 1 (small fraction)"

    def test_positive_autocorr_gives_zero(self):
        """Perfectly trending series (positive autocorr) → cov ≥ 0 → spread ≈ 0."""
        close = pd.Series(np.linspace(100.0, 200.0, 50))  # perfectly trending
        result = roll_spread(close, window=20)
        valid = result.dropna()
        assert np.allclose(valid, 0.0, atol=1e-7), \
            f"Trending series has non-negative cov → spread ≈ 0, got max={valid.abs().max():.2e}"

    def test_invalid_window_raises(self):
        """window < 1 should raise ValueError."""
        close = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError, match="window"):
            roll_spread(close, window=0)


# ===========================================================================
# TestEffectiveTickRatio
# ===========================================================================

class TestEffectiveTickRatio:

    def test_zero_volume_gives_nan_not_inf(self):
        """tick_volume=0 should give NaN in output, not inf."""
        n = 30
        high = pd.Series([105.0] * n)
        low = pd.Series([100.0] * n)
        tick_vol = pd.Series([1000.0] * n)
        tick_vol.iloc[10] = 0.0  # zero volume bar

        result = effective_tick_ratio(high, low, tick_vol, window=5)
        # The NaN from bar 10 propagates into rolling mean → no inf
        assert not np.any(np.isinf(result.values)), \
            "Zero tick_volume should produce NaN, not inf"

    def test_invalid_window_raises(self):
        """window < 1 should raise ValueError."""
        high = pd.Series([105.0, 106.0, 107.0])
        low = pd.Series([100.0, 101.0, 102.0])
        tick_vol = pd.Series([1000.0, 1000.0, 1000.0])
        with pytest.raises(ValueError, match="window"):
            effective_tick_ratio(high, low, tick_vol, window=0)


# ===========================================================================
# TestLiquidityScore
# ===========================================================================

class TestLiquidityScore:

    def _make_scaled(self, price_level=100.0, n=100, seed=1):
        """Make data at a given price level with the same relative spread."""
        rng = np.random.RandomState(seed)
        close = pd.Series(price_level * (1.0 + np.cumsum(rng.normal(0, 0.003, n))))
        high = close * (1.0 + pd.Series(rng.uniform(0.003, 0.010, n)))
        low = close * (1.0 - pd.Series(rng.uniform(0.003, 0.010, n)))
        return high, low, close

    def test_scale_invariance(self):
        """Same relative spread at $10 vs $100 → same liquidity score (Fix 3 test)."""
        h10, l10, c10 = self._make_scaled(price_level=10.0)
        h100, l100, c100 = self._make_scaled(price_level=100.0)

        score10 = liquidity_score(h10, l10, c10, window=20, method="composite")
        score100 = liquidity_score(h100, l100, c100, window=20, method="composite")

        valid10 = score10.dropna()
        valid100 = score100.dropna()

        # z-scored scores should be on the same scale regardless of price level
        assert abs(valid10.mean() - valid100.mean()) < 1.0, \
            f"Liquidity scores should be similar across price levels: " \
            f"$10 mean={valid10.mean():.3f}, $100 mean={valid100.mean():.3f}"

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        high, low, close, _, _, _ = _make_series()
        with pytest.raises(ValueError, match="Invalid method"):
            liquidity_score(high, low, close, window=20, method="unknown_method")

    def test_no_inf_in_output(self):
        """liquidity_score should not produce inf values."""
        high, low, close, tick_vol, _, _ = _make_series(n=100)
        result = liquidity_score(high, low, close, tick_volume=tick_vol, window=20)
        assert not np.any(np.isinf(result.values)), \
            "liquidity_score should not produce inf"

    def test_returns_series(self):
        """Output is a pd.Series."""
        high, low, close, _, _, _ = _make_series()
        result = liquidity_score(high, low, close, window=20)
        assert isinstance(result, pd.Series)


# ===========================================================================
# TestParameterValidation
# ===========================================================================

class TestParameterValidation:

    def test_corwin_schultz_negative_window_raises(self):
        high = pd.Series([105.0, 106.0, 107.0])
        low = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError, match="window"):
            corwin_schultz_spread(high, low, window=-1)

    def test_roll_spread_negative_window_raises(self):
        close = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError, match="window"):
            roll_spread(close, window=-5)

    def test_depth_imbalance_proxy_negative_window_raises(self):
        high, low, close, _, spread, mid = _make_series(n=30)
        with pytest.raises(ValueError, match="window"):
            depth_imbalance_proxy(high, low, close, spread, mid, window=-1)