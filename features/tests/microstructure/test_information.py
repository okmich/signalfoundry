"""
Tests for microstructure.information module.

Functions:
    pin_proxy, adverse_selection_component, smart_money_confidence_index
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import pin_proxy, adverse_selection_component, smart_money_confidence_index


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_data(n=80, seed=42):
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
        high=pd.Series(high, index=idx),
        low=pd.Series(low, index=idx),
        close=pd.Series(close, index=idx),
        volume=pd.Series(vol, index=idx),
        spread=pd.Series(spread, index=idx),
        mid_price=pd.Series(mid, index=idx),
    )


# ─── TestPINProxy ─────────────────────────────────────────────────────────────

class TestPINProxy:

    def test_returns_series(self):
        d = _make_data()
        result = pin_proxy(**d, window=10)
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = pin_proxy(**d, window=15)
        assert result.name == "pin_proxy_15"

    def test_early_values_are_nan(self):
        """First (window-1) values should be NaN (insufficient VPIN history)."""
        d = _make_data(n=50)
        window = 10
        result = pin_proxy(**d, window=window)
        assert np.all(np.isnan(result.values[:window - 1]))

    def test_values_non_negative(self):
        """PIN proxy should be non-negative (VPIN is in [0,1], denominator > 1)."""
        d = _make_data()
        result = pin_proxy(**d, window=10)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_wide_spread_deflates_pin(self):
        """
        When spread is much wider than average, S_rel >> 1, so
        PIN = VPIN / (1 + S_rel) should be smaller than VPIN.
        """
        d = _make_data(n=60)
        # First compute with normal spread
        pin_normal = pin_proxy(**d, window=10)
        # Now widen spread for last 10 bars
        d2 = dict(d)
        spread_wide = d["spread"].copy()
        spread_wide.iloc[-10:] *= 10.0
        d2["spread"] = spread_wide
        pin_wide = pin_proxy(**d2, window=10)
        # Last valid bar: wide-spread PIN should be smaller
        assert pin_wide.iloc[-1] < pin_normal.iloc[-1]

    def test_zero_spread_maximizes_pin(self):
        """
        When spread=0, S_norm=0, S_rel=0, denom=(1+0)=1 → PIN = VPIN.
        With positive spread, denom > 1 → PIN < VPIN.
        """
        d = _make_data(n=60)
        from okmich_quant_features.microstructure import vpin
        vpin_vals = vpin(d["high"], d["low"], d["close"], d["volume"], window=10)

        # Zero spread
        d_zero = dict(d)
        d_zero["spread"] = pd.Series(np.zeros(60), index=d["spread"].index)
        pin_zero = pin_proxy(**d_zero, window=10)

        # With zero spread: PIN ≈ VPIN (within numerical tolerance from the 1e-10 guards)
        valid_mask = ~np.isnan(pin_zero.values) & ~np.isnan(vpin_vals.values)
        np.testing.assert_allclose(
            pin_zero.values[valid_mask], vpin_vals.values[valid_mask], rtol=1e-4
        )


# ─── TestAdverseSelectionComponent ────────────────────────────────────────────

class TestAdverseSelectionComponent:

    def test_returns_series(self):
        d = _make_data()
        result = adverse_selection_component(d["high"], d["low"], d["close"],
                                             d["volume"], window=10)
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = adverse_selection_component(d["high"], d["low"], d["close"],
                                             d["volume"], window=15)
        assert result.name == "adverse_selection_15"

    def test_early_values_are_nan(self):
        d = _make_data(n=50)
        window = 10
        result = adverse_selection_component(d["high"], d["low"], d["close"],
                                             d["volume"], window=window)
        # First bar has NaN return; second window-1 bars may have NaN
        assert np.all(np.isnan(result.values[:window - 1]))

    def test_values_bounded(self):
        """AS is a Pearson correlation → must be in [-1, +1]."""
        d = _make_data()
        result = adverse_selection_component(d["high"], d["low"], d["close"],
                                             d["volume"], window=10)
        valid = result.dropna()
        assert (valid >= -1.0 - 1e-10).all() and (valid <= 1.0 + 1e-10).all()

    def test_perfectly_correlated_flow_gives_high_as(self):
        """
        Construct data where positive returns always coincide with
        positive volume delta (close at high) → AS should be strongly positive.
        """
        n = 50
        # Alternating up/down bars with volume delta matching direction
        close = np.zeros(n)
        close[0] = 100.0
        high = np.zeros(n)
        low = np.zeros(n)
        vol = np.full(n, 1000.0)

        for i in range(1, n):
            if i % 2 == 1:
                # Up bar: close at high → β close to 1 → V_buy >> V_sell → δV > 0
                close[i] = close[i - 1] * 1.02
                high[i] = close[i] + 0.01
                low[i] = close[i] - 2.0
            else:
                # Down bar: close at low → β close to 0 → V_buy << V_sell → δV < 0
                close[i] = close[i - 1] * 0.98
                high[i] = close[i] + 2.0
                low[i] = close[i] - 0.01

        result = adverse_selection_component(
            pd.Series(high), pd.Series(low), pd.Series(close),
            pd.Series(vol), window=20
        )
        valid = result.dropna()
        # With perfectly correlated flow and returns, AS should be strongly positive
        assert valid.mean() > 0.3

    def test_random_data_has_structural_positive_as(self):
        """
        CLV-based buy/sell volume mechanically correlates with bar direction:
        close near high → β ≈ 1 → δV > 0 AND positive return. So AS has a
        positive structural baseline even for random data. Verify it's positive
        but bounded (not at +1).
        """
        d = _make_data(n=200, seed=99)
        result = adverse_selection_component(d["high"], d["low"], d["close"],
                                             d["volume"], window=20)
        valid = result.dropna()
        # Structural positive correlation from CLV
        assert valid.mean() > 0.3
        # But not perfect correlation
        assert valid.mean() < 0.95

    def test_constant_returns_give_zero(self):
        """If returns are all identical, r_var → 0 → AS = 0."""
        n = 40
        # Constant close → r = 0 everywhere
        close = pd.Series(np.full(n, 100.0))
        high = pd.Series(np.full(n, 101.0))
        low = pd.Series(np.full(n, 99.0))
        vol = pd.Series(np.full(n, 1000.0))
        result = adverse_selection_component(high, low, close, vol, window=10)
        # All returns are 0 → r_var < ε → AS = 0
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)


# ─── TestSmartMoneyConfidenceIndex ────────────────────────────────────────────

class TestSmartMoneyConfidenceIndex:

    def test_returns_series(self):
        d = _make_data()
        result = smart_money_confidence_index(**d, window=10)
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = smart_money_confidence_index(**d, window=12)
        assert result.name == "smci_12"

    def test_early_values_are_nan(self):
        d = _make_data(n=50)
        window = 10
        result = smart_money_confidence_index(**d, window=window)
        assert np.all(np.isnan(result.values[:window - 1]))

    def test_values_bounded(self):
        """SMCI is a weighted VIR average → must be in [-1, +1]."""
        d = _make_data()
        result = smart_money_confidence_index(**d, window=10)
        valid = result.dropna()
        assert (valid >= -1.0 - 1e-10).all() and (valid <= 1.0 + 1e-10).all()

    def test_no_smart_bars_gives_zero(self):
        """If no bars qualify as smart (low RVOL), SMCI should be 0."""
        d = _make_data(n=60)
        # Set very high threshold so no bar qualifies
        result = smart_money_confidence_index(**d, window=10, rvol_threshold=100.0)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)

    def test_bullish_smart_bars_give_positive(self):
        """
        Smart bars (high RVOL + tight spread) that are bullish (close at high)
        should produce positive SMCI.
        """
        n = 60
        rng = np.random.default_rng(77)
        # Baseline: moderate volume, typical spread
        close = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        vol = np.full(n, 1000.0, dtype=float)
        mid = np.full(n, 100.0)
        spread = np.full(n, 0.2)  # 20bps

        # Make last 5 bars: very high volume + tight spread + close at high
        vol[-5:] = 5000.0       # 5× average → RVOL = 5 > 2
        spread[-5:] = 0.05      # Much tighter than 0.2 avg → qualifies
        close[-5:] = 101.0      # Close at high → β ≈ 1 → VIR ≈ +1
        high[-5:] = 101.0
        low[-5:] = 99.0

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = smart_money_confidence_index(
            high=pd.Series(high, index=idx),
            low=pd.Series(low, index=idx),
            close=pd.Series(close, index=idx),
            volume=pd.Series(vol, index=idx),
            spread=pd.Series(spread, index=idx),
            mid_price=pd.Series(mid, index=idx),
            window=10,
        )
        # Last bar should show positive SMCI
        assert result.iloc[-1] > 0.3

    def test_bearish_smart_bars_give_negative(self):
        """Smart bars with close at low → VIR < 0 → SMCI < 0."""
        n = 60
        close = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        vol = np.full(n, 1000.0, dtype=float)
        mid = np.full(n, 100.0)
        spread = np.full(n, 0.2)

        # Smart bearish bars
        vol[-5:] = 5000.0
        spread[-5:] = 0.05
        close[-5:] = 99.0     # Close at low → β ≈ 0 → VIR ≈ -1
        high[-5:] = 101.0
        low[-5:] = 99.0

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = smart_money_confidence_index(
            high=pd.Series(high, index=idx),
            low=pd.Series(low, index=idx),
            close=pd.Series(close, index=idx),
            volume=pd.Series(vol, index=idx),
            spread=pd.Series(spread, index=idx),
            mid_price=pd.Series(mid, index=idx),
            window=10,
        )
        assert result.iloc[-1] < -0.3

    def test_wide_spread_disqualifies_bars(self):
        """
        Even with high RVOL, if spread > avg, bar should NOT qualify as smart.
        """
        n = 60
        close = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        vol = np.full(n, 1000.0, dtype=float)
        mid = np.full(n, 100.0)
        spread = np.full(n, 0.1)

        # High volume but WIDER spread → should NOT qualify
        vol[-5:] = 5000.0
        spread[-5:] = 0.5   # Much wider than avg → disqualified

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = smart_money_confidence_index(
            high=pd.Series(high, index=idx),
            low=pd.Series(low, index=idx),
            close=pd.Series(close, index=idx),
            volume=pd.Series(vol, index=idx),
            spread=pd.Series(spread, index=idx),
            mid_price=pd.Series(mid, index=idx),
            window=10,
        )
        # Wide-spread bars don't qualify; SMCI should stay near 0
        assert abs(result.iloc[-1]) < 0.1
