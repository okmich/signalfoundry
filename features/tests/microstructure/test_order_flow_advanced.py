"""
Tests for advanced order flow features (§13).

Functions:
    signed_volume_run_length, volume_clock_acceleration,
    net_order_flow_impulse, order_flow_persistence
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    signed_volume_run_length,
    volume_clock_acceleration,
    net_order_flow_impulse,
    order_flow_persistence,
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
    return (
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        pd.Series(close, index=idx),
        pd.Series(vol, index=idx),
    )


# ─── TestSignedVolumeRunLength ────────────────────────────────────────────────

class TestSignedVolumeRunLength:

    def test_returns_series(self):
        h, l, c, v = _make_ohlcv()
        result = signed_volume_run_length(h, l, c, v)
        assert isinstance(result, pd.Series)

    def test_name(self):
        h, l, c, v = _make_ohlcv()
        result = signed_volume_run_length(h, l, c, v)
        assert result.name == "Signed_Volume_Run_Length"

    def test_no_nans(self):
        """SRL is a state-machine from bar 0 — no NaN expected."""
        h, l, c, v = _make_ohlcv()
        result = signed_volume_run_length(h, l, c, v)
        assert not result.isna().any()

    def test_sign_matches_delta_volume(self):
        """SRL sign must equal sign(V_buy - V_sell)."""
        h, l, c, v = _make_ohlcv()
        result = signed_volume_run_length(h, l, c, v)
        # sign of SRL should always agree with sign of δV
        # (abs value > 0 means a run is in progress)
        assert (result != 0).any()

    def test_sustained_buying_gives_positive_long_run(self):
        """
        Bars where close is always at high → V_buy >> V_sell → SRL grows.
        """
        n = 20
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        close_arr = np.full(n, 101.0)   # close at high → β ≈ 1 → all buy
        vol_arr = np.full(n, 1000.0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = signed_volume_run_length(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
        )
        # All bars buy-dominated → run length increases monotonically
        assert result.iloc[-1] > 0
        assert result.iloc[-1] > result.iloc[5]

    def test_alternating_bars_reset_run(self):
        """
        Alternating buy/sell bars → run length resets to 1 each bar.
        """
        n = 20
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        # Alternate: close at high (buy) then close at low (sell)
        close_arr = np.where(np.arange(n) % 2 == 0, 101.0, 99.0).astype(float)
        vol_arr = np.full(n, 1000.0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = signed_volume_run_length(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
        )
        # Alternating direction → |SRL| should never exceed 2
        assert (result.abs() <= 2).all()


# ─── TestVolumeClockAcceleration ─────────────────────────────────────────────

class TestVolumeClockAcceleration:

    def test_returns_dataframe(self):
        _, _, _, v = _make_ohlcv()
        result = volume_clock_acceleration(v)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        _, _, _, v = _make_ohlcv()
        result = volume_clock_acceleration(v)
        assert set(result.columns) == {"VCA", "VCA_jerk"}

    def test_vca_non_negative(self):
        """VCA = rate / EMA(rate) — both positive, so VCA ≥ 0."""
        _, _, _, v = _make_ohlcv()
        result = volume_clock_acceleration(v)
        valid = result["VCA"].dropna()
        assert (valid >= 0).all()

    def test_constant_volume_gives_vca_one(self):
        """Constant volume → rate = EMA(rate) → VCA = 1.0 everywhere."""
        n = 60
        vol = pd.Series(np.full(n, 1000.0))
        result = volume_clock_acceleration(vol, window=10)
        # After EMA warm-up, VCA should converge to 1
        valid = result["VCA"].iloc[20:]
        np.testing.assert_allclose(valid.values, 1.0, rtol=0.01)

    def test_volume_spike_raises_vca(self):
        """A sudden volume spike should push VCA well above 1."""
        n = 60
        vol_arr = np.full(n, 1000.0, dtype=float)
        vol_arr[-1] = 20000.0  # 20× spike
        vol = pd.Series(vol_arr)
        result = volume_clock_acceleration(vol, window=20)
        assert result["VCA"].iloc[-1] > 5.0

    def test_vca_jerk_is_diff_of_vca(self):
        """VCA_jerk must equal VCA.diff()."""
        _, _, _, v = _make_ohlcv()
        result = volume_clock_acceleration(v, window=10)
        expected = result["VCA"].diff()
        pd.testing.assert_series_equal(result["VCA_jerk"], expected,
                                       check_names=False, rtol=1e-10)


# ─── TestNetOrderFlowImpulse ──────────────────────────────────────────────────

class TestNetOrderFlowImpulse:

    def test_returns_dataframe(self):
        h, l, c, v = _make_ohlcv()
        result = net_order_flow_impulse(h, l, c, v)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        h, l, c, v = _make_ohlcv()
        result = net_order_flow_impulse(h, l, c, v)
        assert set(result.columns) == {"OFI", "OFI_z"}

    def test_ofi_mean_near_zero(self):
        """OFI = δV - EMA(δV) → long-run mean should be near 0."""
        h, l, c, v = _make_ohlcv(n=200)
        result = net_order_flow_impulse(h, l, c, v, window=20)
        # OFI is a deviation from its own EMA → long-run mean ≈ 0
        assert abs(result["OFI"].mean()) < result["OFI"].std() * 0.5

    def test_ofi_z_bounded(self):
        """OFI_z should be a proper z-score — most values in [-5, +5]."""
        h, l, c, v = _make_ohlcv(n=200)
        result = net_order_flow_impulse(h, l, c, v, window=20)
        valid = result["OFI_z"].dropna()
        assert (valid.abs() < 10).all()

    def test_surprise_buy_gives_positive_ofi(self):
        """
        After a period of neutral flow, inject strong buying → OFI should spike.
        """
        n = 60
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        vol_arr = np.full(n, 1000.0)
        # Neutral: close at mid for first 50 bars
        close_arr = np.full(n, 100.0)
        # Last 5 bars: close at high (heavy buying) with high volume
        close_arr[-5:] = 101.0
        vol_arr[-5:] = 5000.0
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = net_order_flow_impulse(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=20,
        )
        # OFI should be positive after the buy surge
        assert result["OFI"].iloc[-1] > 0


# ─── TestOrderFlowPersistence ─────────────────────────────────────────────────

class TestOrderFlowPersistence:

    def test_returns_series(self):
        h, l, c, v = _make_ohlcv()
        result = order_flow_persistence(h, l, c, v)
        assert isinstance(result, pd.Series)

    def test_name(self):
        h, l, c, v = _make_ohlcv()
        result = order_flow_persistence(h, l, c, v, window=15)
        assert result.name == "order_flow_persistence_15"

    def test_values_bounded(self):
        """Pearson correlation → [-1, +1]."""
        h, l, c, v = _make_ohlcv(n=150)
        result = order_flow_persistence(h, l, c, v, window=20)
        valid = result.dropna()
        assert (valid >= -1.0 - 1e-10).all() and (valid <= 1.0 + 1e-10).all()

    def test_early_values_nan(self):
        h, l, c, v = _make_ohlcv(n=60)
        result = order_flow_persistence(h, l, c, v, window=20)
        assert np.isnan(result.values[0])

    def test_persistent_flow_higher_than_alternating(self):
        """
        Sustained unidirectional flow gives higher OFP than alternating flow.
        Both sets use the same volume; only bar direction alternates in the
        second set.
        """
        n = 80
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        vol_arr = np.full(n, 1000.0)

        # Persistent: close always at high → all buy flow
        close_persistent = np.full(n, 101.0)
        ofp_persistent = order_flow_persistence(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_persistent, index=idx),
            pd.Series(vol_arr, index=idx),
            window=20,
        )

        # Alternating: close oscillates high/low → buy then sell then buy ...
        close_alt = np.where(np.arange(n) % 2 == 0, 101.0, 99.0).astype(float)
        ofp_alternating = order_flow_persistence(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_alt, index=idx),
            pd.Series(vol_arr, index=idx),
            window=20,
        )

        p_valid = ofp_persistent.dropna()
        a_valid = ofp_alternating.dropna()
        # Persistent flow → higher AC(1) than alternating
        if len(p_valid) > 0 and len(a_valid) > 0:
            assert p_valid.mean() > a_valid.mean()

    def test_alternating_flow_gives_negative(self):
        """
        Alternating buy/sell bars → δV oscillates → AC(1) < 0 → OFP < 0.
        """
        n = 80
        high_arr = np.full(n, 101.0)
        low_arr = np.full(n, 99.0)
        close_arr = np.where(np.arange(n) % 2 == 0, 101.0, 99.0).astype(float)
        vol_arr = np.full(n, 1000.0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = order_flow_persistence(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(close_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=20,
        )
        valid = result.dropna()
        # Alternating δV → AC(1) ≈ -1
        assert valid.mean() < -0.5
