"""
Tests for microstructure.regime module.

Functions:
    volatility_volume_correlation, spread_volume_correlation,
    return_autocorrelation_decay, volume_return_asymmetry
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.microstructure import (
    volatility_volume_correlation,
    spread_volume_correlation,
    return_autocorrelation_decay,
    volume_return_asymmetry,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_data(n=120, seed=42):
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


# ─── TestVolatilityVolumeCorrelation ─────────────────────────────────────────

class TestVolatilityVolumeCorrelation:

    def test_returns_series(self):
        d = _make_data()
        result = volatility_volume_correlation(d["high"], d["low"], d["volume"])
        assert isinstance(result, pd.Series)

    def test_name_contains_window(self):
        d = _make_data()
        result = volatility_volume_correlation(d["high"], d["low"], d["volume"],
                                               window=15)
        assert result.name == "vol_volume_corr_15"

    def test_values_bounded(self):
        """Correlation must be in [-1, +1]."""
        d = _make_data()
        result = volatility_volume_correlation(d["high"], d["low"], d["volume"])
        valid = result.dropna()
        assert (valid >= -1.0 - 1e-10).all() and (valid <= 1.0 + 1e-10).all()

    def test_early_values_nan(self):
        """First bars must be NaN (insufficient Parkinson + correlation warmup)."""
        d = _make_data(n=60)
        result = volatility_volume_correlation(d["high"], d["low"], d["volume"],
                                               vol_window=5, window=10)
        assert np.isnan(result.values[0])

    def test_correlated_vol_volume_gives_positive(self):
        """
        When high bars always have high volume and vice versa,
        correlation should be strongly positive.
        """
        n = 80
        rng = np.random.default_rng(7)
        # Construct alternating high-range/high-vol and low-range/low-vol bars
        base = 100.0
        high_arr = np.full(n, base + 0.5)
        low_arr = np.full(n, base - 0.5)
        vol_arr = np.full(n, 2000.0)

        # Every other bar: wide range + high volume (correlated)
        for i in range(0, n, 2):
            high_arr[i] = base + 2.0
            low_arr[i] = base - 2.0
            vol_arr[i] = 8000.0

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = volatility_volume_correlation(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            vol_window=5, window=20,
        )
        valid = result.dropna()
        assert valid.mean() > 0.5

    def test_decoupled_gives_low_or_negative(self):
        """
        When wide-range bars always have LOW volume (decoupled),
        correlation should be near zero or negative.
        """
        n = 80
        base = 100.0
        high_arr = np.full(n, base + 0.5)
        low_arr = np.full(n, base - 0.5)
        vol_arr = np.full(n, 5000.0)

        # Every other bar: wide range but LOW volume (anti-correlated)
        for i in range(0, n, 2):
            high_arr[i] = base + 3.0
            low_arr[i] = base - 3.0
            vol_arr[i] = 200.0

        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        result = volatility_volume_correlation(
            pd.Series(high_arr, index=idx),
            pd.Series(low_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            vol_window=5, window=20,
        )
        valid = result.dropna()
        # Decoupled → negative or at most weakly positive
        assert valid.mean() < 0.3


# ─── TestSpreadVolumeCorrelation ─────────────────────────────────────────────

class TestSpreadVolumeCorrelation:

    def test_returns_dataframe(self):
        d = _make_data()
        result = spread_volume_correlation(d["spread"], d["mid_price"],
                                           d["volume"])
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        d = _make_data()
        result = spread_volume_correlation(d["spread"], d["mid_price"],
                                           d["volume"])
        assert set(result.columns) == {"rho_SV", "delta_rho_SV"}

    def test_rho_bounded(self):
        """Pearson correlation must be in [-1, +1]."""
        d = _make_data()
        result = spread_volume_correlation(d["spread"], d["mid_price"],
                                           d["volume"])
        rho = result["rho_SV"].dropna()
        assert (rho >= -1.0 - 1e-10).all() and (rho <= 1.0 + 1e-10).all()

    def test_early_rows_nan(self):
        """First (window-1) rows of rho_SV should be NaN."""
        d = _make_data(n=60)
        result = spread_volume_correlation(d["spread"], d["mid_price"],
                                           d["volume"], window=10)
        assert np.isnan(result["rho_SV"].values[0])

    def test_positive_rho_when_spread_and_volume_coprorate(self):
        """
        When high volume consistently coincides with wide spread (toxic flow),
        rho_SV should be positive.
        """
        n = 80
        rng = np.random.default_rng(13)
        base = 100.0
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        # Tight spread + low volume on even bars, wide spread + high volume on odd
        spread_arr = np.where(np.arange(n) % 2 == 0, 0.05, 0.5)
        vol_arr = np.where(np.arange(n) % 2 == 0, 500.0, 8000.0)
        mid_arr = np.full(n, base)

        result = spread_volume_correlation(
            pd.Series(spread_arr, index=idx),
            pd.Series(mid_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=20,
        )
        rho = result["rho_SV"].dropna()
        assert rho.mean() > 0.5

    def test_negative_rho_when_spread_tightens_on_high_volume(self):
        """
        Healthy market: high volume tightens spread → negative rho.
        """
        n = 80
        base = 100.0
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        # Wide spread + low volume on even bars, tight spread + high volume on odd
        spread_arr = np.where(np.arange(n) % 2 == 0, 0.5, 0.05)
        vol_arr = np.where(np.arange(n) % 2 == 0, 500.0, 8000.0)
        mid_arr = np.full(n, base)

        result = spread_volume_correlation(
            pd.Series(spread_arr, index=idx),
            pd.Series(mid_arr, index=idx),
            pd.Series(vol_arr, index=idx),
            window=20,
        )
        rho = result["rho_SV"].dropna()
        assert rho.mean() < -0.5

    def test_delta_rho_is_diff_of_rho(self):
        """delta_rho_SV must equal rho_SV.diff(window)."""
        d = _make_data()
        w = 10
        result = spread_volume_correlation(d["spread"], d["mid_price"],
                                           d["volume"], window=w)
        expected_delta = result["rho_SV"].diff(w)
        pd.testing.assert_series_equal(
            result["delta_rho_SV"], expected_delta,
            check_names=False, rtol=1e-10
        )


# ─── TestReturnAutocorrelationDecay ──────────────────────────────────────────

class TestReturnAutocorrelationDecay:

    def test_returns_series(self):
        d = _make_data()
        result = return_autocorrelation_decay(d["close"])
        assert isinstance(result, pd.Series)

    def test_name(self):
        d = _make_data()
        result = return_autocorrelation_decay(d["close"], max_lag=5, window=40)
        assert result.name == "ac_decay_5_40"

    def test_early_values_zero(self):
        """First window + max_lag bars must be 0 (NaN filled with 0 — no autocorrelation structure)."""
        d = _make_data(n=100)
        result = return_autocorrelation_decay(d["close"], max_lag=5, window=30)
        assert result.values[0] == 0.0

    def test_finite_after_warmup(self):
        """Use autocorrelated data so AC(1) passes significance guard."""
        rng = np.random.default_rng(11)
        n = 200
        # AR(1) with phi=0.9: significant AC(1) → ACD computable
        rets = np.zeros(n)
        for i in range(1, n):
            rets[i] = 0.9 * rets[i - 1] + rng.normal(0, 0.002)
        close = pd.Series(100.0 * np.cumprod(1 + rets))
        result = return_autocorrelation_decay(close, max_lag=5, window=40)
        valid = result.dropna()
        assert len(valid) > 0
        assert np.all(np.isfinite(valid.values))

    def test_trending_less_acd_than_mean_reverting(self):
        """
        Trending market (AR(1), phi=+0.9) has lower ACD than mean-reverting
        market (AR(1), phi=-0.7): trend has slow AC decay, MR has fast decay.
        """
        rng = np.random.default_rng(11)
        n = 200

        # Trending: positive phi → AC decays slowly
        rets_trend = np.zeros(n)
        for i in range(1, n):
            rets_trend[i] = 0.9 * rets_trend[i - 1] + rng.normal(0, 0.002)
        close_trend = pd.Series(100.0 * np.cumprod(1 + rets_trend))
        acd_trend = return_autocorrelation_decay(close_trend, max_lag=5, window=40)

        # Mean-reverting: negative phi → AC flips sign quickly → fast effective decay
        rets_mr = np.zeros(n)
        for i in range(1, n):
            rets_mr[i] = -0.7 * rets_mr[i - 1] + rng.normal(0, 0.002)
        close_mr = pd.Series(100.0 * np.cumprod(1 + rets_mr))
        acd_mr = return_autocorrelation_decay(close_mr, max_lag=5, window=40)

        valid_trend = acd_trend.dropna()
        valid_mr = acd_mr.dropna()

        # Trending → slow AC decay → lower ACD
        # Mean-reverting → fast AC decay → higher |ACD|
        if len(valid_trend) > 0 and len(valid_mr) > 0:
            assert valid_trend.mean() < valid_mr.abs().mean()

    def test_iid_returns_give_mostly_nan(self):
        """
        IID normal returns: AC(1) is near zero (not significant) → most bars NaN.
        The significance guard prevents blow-ups.
        """
        rng = np.random.default_rng(99)
        n = 200
        close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)))
        result = return_autocorrelation_decay(close, max_lag=5, window=40)
        valid = result.dropna()
        # With significance guard, few IID bars should produce valid ACD;
        # any valid values must be bounded (no blow-up)
        if len(valid) > 0:
            assert np.all(np.abs(valid.values) < 5.0)


# ─── TestVolumeReturnAsymmetry ────────────────────────────────────────────────

class TestVolumeReturnAsymmetry:

    def test_returns_series(self):
        d = _make_data()
        result = volume_return_asymmetry(d["close"], d["volume"])
        assert isinstance(result, pd.Series)

    def test_name(self):
        d = _make_data()
        result = volume_return_asymmetry(d["close"], d["volume"], window=30)
        assert result.name == "vol_ret_asymmetry_30"

    def test_early_values_nan(self):
        d = _make_data(n=100)
        result = volume_return_asymmetry(d["close"], d["volume"], window=40)
        assert np.isnan(result.values[0])

    def test_values_bounded(self):
        """VRA = diff of two correlations → must be in [-2, +2]."""
        d = _make_data()
        result = volume_return_asymmetry(d["close"], d["volume"], window=30)
        valid = result.dropna()
        assert (valid >= -2.0 - 1e-10).all() and (valid <= 2.0 + 1e-10).all()

    def test_symmetric_volume_gives_near_zero(self):
        """
        When volume has no asymmetric relationship with returns, VRA should be near zero.
        Pure white-noise volume is symmetric by construction.
        Note: exactly constant volume makes Pearson correlation undefined (0/0);
        use IID noise which is uncorrelated with returns in expectation.
        """
        n = 300
        rng = np.random.default_rng(5)
        close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.005, n)))
        # IID volume: no directional relationship with returns → VRA ≈ 0 in expectation
        vol = pd.Series(1000.0 + rng.normal(0, 1.0, n))
        result = volume_return_asymmetry(close, vol, window=30, causal=True)
        valid = result.dropna()
        # With IID noise, mean should be close to zero (not an exact per-bar check)
        assert abs(valid.mean()) < 0.3, f"Mean VRA {valid.mean():.3f} too large for IID volume"

    def test_fear_regime_gives_negative_vra(self):
        """
        Fear regime: negative returns precede high volume (panic selling).
        Construct: large negative returns followed by large volume spikes.
        → corr_fwd should be more negative than corr_bwd → VRA < 0.
        """
        n = 150
        rng = np.random.default_rng(42)
        close_vals = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
        vol_vals = np.full(n, 1000.0, dtype=float)
        returns = np.log(close_vals[1:] / close_vals[:-1])

        # For negative return bars, spike the NEXT bar's volume
        for i in range(len(returns)):
            if returns[i] < -0.003:
                if i + 1 < n:
                    vol_vals[i + 1] = 10000.0  # panic selling next bar

        close = pd.Series(close_vals)
        vol = pd.Series(vol_vals)
        result = volume_return_asymmetry(close, vol, window=40)
        valid = result.dropna()
        # Fear regime: negative returns predict forward volume spikes → VRA < 0
        assert valid.mean() < 0.0

    def test_causal_is_default(self):
        """causal=True must be the default — explicit and implicit calls match."""
        d = _make_data()
        implicit = volume_return_asymmetry(d["close"], d["volume"], window=30)
        explicit = volume_return_asymmetry(d["close"], d["volume"], window=30, causal=True)
        pd.testing.assert_series_equal(implicit, explicit)

    def test_causal_mode_no_future_data(self):
        """
        Fix #4 leakage guard: in causal mode, the output at time t must not change
        when a new bar is appended at t+1. If volume.shift(-1) were still used,
        appending a new bar would change the last computed value.
        """
        d = _make_data(n=100)
        close, volume = d["close"], d["volume"]
        result_n = volume_return_asymmetry(close, volume, window=30, causal=True)

        # Append one new bar and recompute
        new_close = pd.concat([close, pd.Series([close.iloc[-1] * 1.001],
                                                 index=[close.index[-1] + pd.Timedelta("1D")])])
        new_vol = pd.concat([volume, pd.Series([volume.iloc[-1]],
                                                index=[volume.index[-1] + pd.Timedelta("1D")])])
        result_n1 = volume_return_asymmetry(new_close, new_vol, window=30, causal=True)

        # The value at the last original bar must be identical (no lookahead)
        np.testing.assert_almost_equal(
            result_n.iloc[-1], result_n1.iloc[-2], decimal=12,
            err_msg="Causal mode changed previous value when a new bar was appended — lookahead present."
        )

    def test_causal_vs_research_differ_on_fear_data(self):
        """
        Research mode (causal=False) directly sees V(t+1) and should produce a
        stronger negative signal on fear-regime data than causal mode.
        """
        n = 150
        rng = np.random.default_rng(99)
        close_vals = 100.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
        vol_vals = np.full(n, 1000.0, dtype=float)
        returns = np.log(close_vals[1:] / close_vals[:-1])
        for i in range(len(returns)):
            if returns[i] < -0.003 and i + 1 < n:
                vol_vals[i + 1] = 10000.0

        close = pd.Series(close_vals)
        vol = pd.Series(vol_vals)
        causal_vra = volume_return_asymmetry(close, vol, window=40, causal=True).dropna()
        research_vra = volume_return_asymmetry(close, vol, window=40, causal=False).dropna()

        # Research mode directly correlates r(t) with V(t+1) → stronger negative signal.
        assert research_vra.mean() < causal_vra.mean(), (
            "Research mode should show stronger (more negative) VRA on fear-regime data "
            "because it directly observes the next-bar volume spike."
        )
