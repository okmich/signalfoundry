"""
Tests for timothymasters/multi_market/risk.py

Covers indicators #11–15: mahal, abs_ratio, abs_shift, coherence, delta_coherence.
"""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_features.timothymasters.multi_market.risk import (
    mahal,
    abs_ratio,
    abs_shift,
    coherence,
    delta_coherence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_BARS = 300
N_MARKETS = 5
LOOKBACK = 60


def _make_close(seed: int, n: int = N_BARS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = 100.0 + np.cumsum(rng.standard_normal(n))
    return np.maximum(c, 1.0)


@pytest.fixture
def random_closes():
    return [_make_close(seed=i * 3 + 1) for i in range(N_MARKETS)]


# ---------------------------------------------------------------------------
# Generic property tests — all 5 indicators
# ---------------------------------------------------------------------------

ALL_FUNCTIONS = [
    ("mahal",           lambda c: mahal(c, lookback=LOOKBACK)),
    ("abs_ratio",       lambda c: abs_ratio(c, lookback=LOOKBACK)),
    ("abs_shift",       lambda c: abs_shift(c, lookback=LOOKBACK, long_lookback=30, short_lookback=5)),
    ("coherence",       lambda c: coherence(c, lookback=LOOKBACK)),
    ("delta_coherence", lambda c: delta_coherence(c, lookback=LOOKBACK, delta_length=10)),
]


class TestShape:
    @pytest.mark.parametrize("name,fn", ALL_FUNCTIONS)
    def test_output_length(self, random_closes, name, fn):
        result = fn(random_closes)
        assert len(result) == N_BARS, f"{name}: expected {N_BARS}, got {len(result)}"

    @pytest.mark.parametrize("name,fn", ALL_FUNCTIONS)
    def test_output_dtype_float64(self, random_closes, name, fn):
        result = fn(random_closes)
        assert result.dtype == np.float64, f"{name}: expected float64"

    @pytest.mark.parametrize("name,fn", ALL_FUNCTIONS)
    def test_has_valid_values(self, random_closes, name, fn):
        result = fn(random_closes)
        assert not np.all(np.isnan(result)), f"{name}: all NaN"

    @pytest.mark.parametrize("name,fn", ALL_FUNCTIONS)
    def test_no_inf_in_output(self, random_closes, name, fn):
        result = fn(random_closes)
        valid = result[~np.isnan(result)]
        assert not np.any(np.isinf(valid)), f"{name}: got inf"


class TestWarmup:
    def test_mahal_warmup(self, random_closes):
        result = mahal(random_closes, lookback=LOOKBACK)
        assert np.all(np.isnan(result[:LOOKBACK]))
        assert not np.all(np.isnan(result[LOOKBACK:]))

    def test_abs_ratio_warmup(self, random_closes):
        result = abs_ratio(random_closes, lookback=LOOKBACK)
        assert np.all(np.isnan(result[:LOOKBACK - 1]))
        assert not np.all(np.isnan(result[LOOKBACK - 1:]))

    def test_coherence_warmup(self, random_closes):
        result = coherence(random_closes, lookback=LOOKBACK)
        assert np.all(np.isnan(result[:LOOKBACK - 1]))
        assert not np.all(np.isnan(result[LOOKBACK - 1:]))

    def test_delta_coherence_warmup(self, random_closes):
        delta = 10
        result = delta_coherence(random_closes, lookback=LOOKBACK, delta_length=delta)
        expected_warmup = LOOKBACK - 1 + delta
        assert np.all(np.isnan(result[:expected_warmup]))
        assert not np.all(np.isnan(result[expected_warmup:]))

    def test_abs_shift_warmup(self, random_closes):
        long_lb = 30
        short_lb = 5
        result = abs_shift(random_closes, lookback=LOOKBACK, long_lookback=long_lb, short_lookback=short_lb)
        expected_warmup = (LOOKBACK - 1) + (long_lb - 1)
        assert np.all(np.isnan(result[:expected_warmup]))


class TestTooFewMarkets:
    @pytest.mark.parametrize("name,fn", ALL_FUNCTIONS)
    def test_raises_on_single_market(self, name, fn):
        closes = [_make_close(seed=0)]
        with pytest.raises(ValueError, match="At least 2 markets"):
            fn(closes)


class TestLengthMismatch:
    @pytest.mark.parametrize("name,fn", ALL_FUNCTIONS)
    def test_raises_on_unequal_lengths(self, random_closes, name, fn):
        bad_closes = [random_closes[0][:200]] + random_closes[1:]
        with pytest.raises(ValueError):
            fn(bad_closes)


# ---------------------------------------------------------------------------
# Deterministic tests — abs_ratio
# ---------------------------------------------------------------------------

class TestAbsRatio:
    def test_output_in_0_100(self, random_closes):
        result = abs_ratio(random_closes, lookback=LOOKBACK)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1e-6) and np.all(valid <= 100 + 1e-6), \
            f"abs_ratio out of [0, 100]: min={valid.min():.4f}, max={valid.max():.4f}"

    def test_single_market_fraction_gives_near_100(self):
        """With fraction≈1/N (one eigenvalue out of N), on perfectly correlated markets
        the top eigenvalue captures all variance → ratio near 100."""
        n = N_BARS
        base = _make_close(seed=42, n=n)
        closes = [base * (1.0 + 1e-8 * i) for i in range(5)]
        result = abs_ratio(closes, lookback=LOOKBACK, fraction=0.2)
        valid = result[~np.isnan(result)]
        # On highly correlated markets, the top eigenvalue dominates → > 50%
        assert np.mean(valid) > 50.0, f"Expected >50 for correlated markets, got {np.mean(valid):.2f}"


# ---------------------------------------------------------------------------
# Deterministic tests — coherence
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_output_range(self, random_closes):
        result = coherence(random_closes, lookback=LOOKBACK)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -100 - 1e-6) and np.all(valid <= 100 + 1e-6), \
            f"coherence out of [-100, 100]: min={valid.min():.4f}, max={valid.max():.4f}"

    def test_high_correlation_markets_high_coherence(self):
        """Perfectly correlated markets → coherence near 100."""
        n = N_BARS
        base = _make_close(seed=99, n=n)
        closes = [base * (1.0 + 1e-7 * i) for i in range(5)]
        result = coherence(closes, lookback=LOOKBACK)
        valid = result[~np.isnan(result)]
        assert np.mean(valid) > 50.0, f"Expected high coherence for correlated markets, got {np.mean(valid):.2f}"

    def test_independent_markets_lower_coherence(self, random_closes):
        """Random independent markets should have lower average coherence than perfectly correlated ones."""
        result_indep = coherence(random_closes, lookback=LOOKBACK)
        base = _make_close(seed=100)
        closes_corr = [base * (1.0 + 1e-7 * i) for i in range(N_MARKETS)]
        result_corr = coherence(closes_corr, lookback=LOOKBACK)
        mean_indep = np.nanmean(result_indep)
        mean_corr = np.nanmean(result_corr)
        assert mean_corr > mean_indep, \
            f"Correlated markets should have higher coherence: {mean_corr:.2f} vs {mean_indep:.2f}"


# ---------------------------------------------------------------------------
# Deterministic tests — delta_coherence
# ---------------------------------------------------------------------------

class TestDeltaCoherence:
    def test_constant_markets_give_zero_delta(self):
        """Constant prices → returns are 0 → coherence is stable → delta ≈ 0."""
        n = N_BARS
        closes = [np.full(n, float(100 + i)) for i in range(N_MARKETS)]
        result = delta_coherence(closes, lookback=LOOKBACK, delta_length=10)
        valid = result[~np.isnan(result)]
        # Constant prices give zero log returns; cov is all zeros; coherence = 0
        # delta of constant series = 0
        np.testing.assert_allclose(valid, 0.0, atol=1e-6)

    def test_output_length_matches_input(self, random_closes):
        result = delta_coherence(random_closes, lookback=LOOKBACK, delta_length=10)
        assert len(result) == N_BARS


# ---------------------------------------------------------------------------
# Deterministic tests — mahal
# ---------------------------------------------------------------------------

class TestMahal:
    def test_mahal_returns_log_odds_range(self, random_closes):
        """log-odds of [0.5, 0.99999] is in [0, log(99999)]."""
        result = mahal(random_closes, lookback=LOOKBACK)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0), f"mahal should be >= 0 (p clipped at 0.5)"
        upper = float(np.log(0.99999 / 0.00001)) + 1e-3
        assert np.all(valid <= upper), f"mahal exceeded expected upper bound"

    def test_mahal_smoothing_reduces_noise(self, random_closes):
        """EMA smoothing should reduce std dev of output."""
        raw = mahal(random_closes, lookback=LOOKBACK, smoothing=0)
        smoothed = mahal(random_closes, lookback=LOOKBACK, smoothing=10)
        std_raw = np.nanstd(raw)
        std_smooth = np.nanstd(smoothed)
        assert std_smooth <= std_raw + 1e-6, \
            f"Smoothing should reduce std: raw={std_raw:.4f}, smoothed={std_smooth:.4f}"

    def test_abs_shift_long_lookback_enforced(self):
        """If long_lookback < short_lookback + 1, it should be auto-corrected."""
        closes = [_make_close(seed=i) for i in range(N_MARKETS)]
        # This should not raise even though long < short + 1
        result = abs_shift(closes, lookback=LOOKBACK, long_lookback=5, short_lookback=10)
        assert len(result) == N_BARS


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes (Fix #5: parameter guards)
# ---------------------------------------------------------------------------

class TestRiskParameterGuards:
    """Tests for Fix #5: parameter validation guards on multi_market risk functions."""

    @pytest.fixture
    def closes_3(self):
        return [_make_close(seed=i) for i in range(3)]

    # mahal guards
    def test_mahal_lookback_zero_raises(self, closes_3):
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            mahal(closes_3, lookback=0)

    def test_mahal_lookback_one_raises(self, closes_3):
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            mahal(closes_3, lookback=1)

    def test_mahal_smoothing_negative_raises(self, closes_3):
        with pytest.raises(ValueError, match="smoothing must be >= 0"):
            mahal(closes_3, lookback=10, smoothing=-1)

    # abs_ratio guards
    def test_abs_ratio_lookback_zero_raises(self, closes_3):
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            abs_ratio(closes_3, lookback=0)

    def test_abs_ratio_fraction_zero_raises(self, closes_3):
        with pytest.raises(ValueError, match="fraction must be in"):
            abs_ratio(closes_3, lookback=10, fraction=0.0)

    def test_abs_ratio_fraction_negative_raises(self, closes_3):
        with pytest.raises(ValueError, match="fraction must be in"):
            abs_ratio(closes_3, lookback=10, fraction=-0.1)

    # coherence guards
    def test_coherence_lookback_one_raises(self, closes_3):
        with pytest.raises(ValueError, match="lookback must be >= 2"):
            coherence(closes_3, lookback=1)

    # delta_coherence guards
    def test_delta_coherence_delta_zero_raises(self, closes_3):
        with pytest.raises(ValueError, match="delta_length must be >= 1"):
            delta_coherence(closes_3, lookback=10, delta_length=0)

    def test_delta_coherence_delta_negative_raises(self, closes_3):
        with pytest.raises(ValueError, match="delta_length must be >= 1"):
            delta_coherence(closes_3, lookback=10, delta_length=-5)

    # adx guard (in single/trend.py)
    def test_adx_period_zero_raises(self):
        from okmich_quant_features.timothymasters.trend import adx
        close = np.ones(50)
        high = close + 0.5
        low = close - 0.5
        with pytest.raises(ValueError, match="period must be >= 1"):
            adx(high, low, close, period=0)
