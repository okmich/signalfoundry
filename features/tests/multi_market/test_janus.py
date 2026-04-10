"""
Tests for timothymasters/multi_market/janus.py

Covers the Janus class and all 25 functional wrappers.
"""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_features.timothymasters.multi_market.janus import (
    Janus,
    janus_market_index,
    janus_rs,
    janus_rs_fractile,
    janus_delta_rs_fractile,
    janus_rss,
    janus_delta_rss,
    janus_dom,
    janus_doe,
    janus_dom_index,
    janus_rm,
    janus_rm_fractile,
    janus_delta_rm_fractile,
    janus_rs_leader_equity,
    janus_rs_laggard_equity,
    janus_rs_ps,
    janus_rs_leader_advantage,
    janus_rs_laggard_advantage,
    janus_rm_leader_equity,
    janus_rm_laggard_equity,
    janus_rm_ps,
    janus_rm_leader_advantage,
    janus_rm_laggard_advantage,
    janus_oos_avg,
    janus_cma_oos,
    janus_leader_cma_oos,
)

# ---------------------------------------------------------------------------
# Constants & fixtures
# ---------------------------------------------------------------------------

N_BARS = 500
N_MARKETS = 5
LOOKBACK = 60


def _make_close(seed: int, n: int = N_BARS, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal(n) * 0.01 + drift
    c = 100.0 * np.exp(np.cumsum(returns))
    return np.maximum(c, 1.0)


@pytest.fixture
def random_closes():
    return [_make_close(seed=i * 7 + 1) for i in range(N_MARKETS)]


@pytest.fixture
def janus_obj(random_closes):
    return Janus(random_closes, lookback=LOOKBACK)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_valid_construction(self, random_closes):
        j = Janus(random_closes, lookback=LOOKBACK)
        assert j.n_bars == N_BARS
        assert j.n_markets == N_MARKETS

    def test_two_markets_minimum(self):
        closes = [_make_close(0), _make_close(1)]
        j = Janus(closes, lookback=LOOKBACK)
        assert j.n_markets == 2

    def test_raises_on_one_market(self):
        with pytest.raises(ValueError, match="At least 2"):
            Janus([_make_close(0)], lookback=LOOKBACK)

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="At least 2"):
            Janus([], lookback=LOOKBACK)

    def test_raises_on_unequal_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            Janus([_make_close(0, n=300), _make_close(1, n=400)], lookback=LOOKBACK)

    def test_raises_on_small_lookback(self):
        with pytest.raises(ValueError, match="lookback"):
            Janus([_make_close(0), _make_close(1)], lookback=1)


# ---------------------------------------------------------------------------
# Generic output property tests
# ---------------------------------------------------------------------------

# Scalar (non-per-market) outputs
SCALAR_OUTPUTS = [
    ("market_index",        lambda j: j.market_index),
    ("dom_index_equity",    lambda j: j.dom_index_equity),
    ("rss",                 lambda j: j.rss),
    ("rss_change",          lambda j: j.rss_change),
    ("oos_avg",             lambda j: j.oos_avg),
    ("rs_leader_equity",    lambda j: j.rs_leader_equity),
    ("rs_laggard_equity",   lambda j: j.rs_laggard_equity),
    ("rs_ps",               lambda j: j.rs_ps),
    ("rs_leader_advantage", lambda j: j.rs_leader_advantage),
    ("rs_laggard_advantage", lambda j: j.rs_laggard_advantage),
    ("rm_leader_equity",    lambda j: j.rm_leader_equity),
    ("rm_laggard_equity",   lambda j: j.rm_laggard_equity),
    ("rm_ps",               lambda j: j.rm_ps),
    ("rm_leader_advantage", lambda j: j.rm_leader_advantage),
    ("rm_laggard_advantage", lambda j: j.rm_laggard_advantage),
    ("cma_oos",             lambda j: j.cma_oos),
    ("leader_cma_oos",      lambda j: j.leader_cma_oos),
]

# Per-market outputs
PER_MARKET_OUTPUTS = [
    ("rs",           lambda j: j.rs(0)),
    ("rs_fractile",  lambda j: j.rs_fractile(0)),
    ("dom",          lambda j: j.dom(0)),
    ("doe",          lambda j: j.doe(0)),
    ("rm",           lambda j: j.rm(0)),
    ("rm_fractile",  lambda j: j.rm_fractile(0)),
]

ALL_OUTPUTS = SCALAR_OUTPUTS + PER_MARKET_OUTPUTS


class TestOutputShape:
    @pytest.mark.parametrize("name,fn", ALL_OUTPUTS)
    def test_output_length(self, janus_obj, name, fn):
        result = fn(janus_obj)
        assert len(result) == N_BARS, f"{name}: expected {N_BARS}, got {len(result)}"

    @pytest.mark.parametrize("name,fn", ALL_OUTPUTS)
    def test_output_dtype(self, janus_obj, name, fn):
        result = fn(janus_obj)
        assert result.dtype == np.float64, f"{name}: expected float64"

    @pytest.mark.parametrize("name,fn", ALL_OUTPUTS)
    def test_has_valid_values(self, janus_obj, name, fn):
        result = fn(janus_obj)
        assert not np.all(np.isnan(result)), f"{name}: all NaN"

    @pytest.mark.parametrize("name,fn", ALL_OUTPUTS)
    def test_no_inf(self, janus_obj, name, fn):
        result = fn(janus_obj)
        valid = result[~np.isnan(result)]
        assert not np.any(np.isinf(valid)), f"{name}: contains inf"


class TestWarmup:
    def test_rs_warmup(self, janus_obj):
        result = janus_obj.rs(0)
        assert np.all(np.isnan(result[:LOOKBACK]))

    def test_rs_fractile_warmup(self, janus_obj):
        result = janus_obj.rs_fractile(0)
        assert np.all(np.isnan(result[:LOOKBACK]))

    def test_rm_warmup(self, janus_obj):
        result = janus_obj.rm(0)
        # RM needs DOM to be valid first, which needs RS + RSS change
        # so warmup is longer than RS
        warmup_count = np.sum(np.isnan(result))
        assert warmup_count >= LOOKBACK

    def test_market_index_starts_at_zero(self, janus_obj):
        mi = janus_obj.market_index
        assert mi[0] == 0.0


# ---------------------------------------------------------------------------
# RS-specific tests
# ---------------------------------------------------------------------------


class TestRS:
    def test_rs_range(self, janus_obj):
        for m in range(N_MARKETS):
            rs = janus_obj.rs(m)
            valid = rs[~np.isnan(rs)]
            assert np.all(valid >= -200.0), f"market {m}: RS < -200"
            assert np.all(valid <= 200.0), f"market {m}: RS > 200"

    def test_rs_fractile_range(self, janus_obj):
        for m in range(N_MARKETS):
            frac = janus_obj.rs_fractile(m)
            valid = frac[~np.isnan(frac)]
            assert np.all(valid >= 0.0), f"market {m}: fractile < 0"
            assert np.all(valid <= 1.0), f"market {m}: fractile > 1"

    def test_identical_markets_rs_near_zero(self):
        c = _make_close(42, n=300)
        closes = [c.copy() for _ in range(5)]
        j = Janus(closes, lookback=LOOKBACK)
        rs = j.rs(0)
        valid = rs[~np.isnan(rs)]
        assert np.allclose(valid, 0.0, atol=1e-6), "Identical markets should give RS≈0"

    def test_identical_markets_rss_zero(self):
        c = _make_close(42, n=300)
        closes = [c.copy() for _ in range(5)]
        j = Janus(closes, lookback=LOOKBACK)
        rss = j.rss
        valid = rss[~np.isnan(rss)]
        assert np.allclose(valid, 0.0, atol=1e-6), "Identical markets should give RSS=0"

    def test_strong_trend_has_positive_rs(self):
        """A market that consistently outperforms should have positive RS."""
        base = [_make_close(i, n=400) for i in range(5)]
        # Make market 0 strongly trending up
        base[0] = _make_close(0, n=400, drift=0.005)
        j = Janus(base, lookback=LOOKBACK)
        rs = j.rs(0)
        valid = rs[~np.isnan(rs)]
        # At least the last portion should be positive on average
        tail = valid[-100:]
        assert np.mean(tail) > 0, "Strongly trending market should have positive RS"

    def test_strong_trend_has_high_fractile(self):
        base = [_make_close(i, n=400) for i in range(5)]
        base[0] = _make_close(0, n=400, drift=0.005)
        j = Janus(base, lookback=LOOKBACK)
        frac = j.rs_fractile(0)
        valid = frac[~np.isnan(frac)]
        tail = valid[-100:]
        assert np.mean(tail) > 0.5, "Strongly trending market should have high fractile"


class TestRSS:
    def test_rss_non_negative(self, janus_obj):
        rss = janus_obj.rss
        valid = rss[~np.isnan(rss)]
        assert np.all(valid >= -1e-10), "RSS should be non-negative (spread width)"

    def test_rss_change_is_signed(self, janus_obj):
        rss_chg = janus_obj.rss_change
        valid = rss_chg[~np.isnan(rss_chg)]
        if len(valid) > 10:
            has_pos = np.any(valid > 0)
            has_neg = np.any(valid < 0)
            assert has_pos or has_neg, "rss_change should have non-zero values"


# ---------------------------------------------------------------------------
# DOM / DOE tests
# ---------------------------------------------------------------------------


class TestDOMDOE:
    def test_dom_per_market_finite(self, janus_obj):
        for m in range(N_MARKETS):
            dom = janus_obj.dom(m)
            valid = dom[~np.isnan(dom)]
            assert len(valid) > 0, f"market {m}: DOM all NaN"
            assert np.all(np.isfinite(valid)), f"market {m}: DOM has inf"

    def test_doe_per_market_finite(self, janus_obj):
        for m in range(N_MARKETS):
            doe = janus_obj.doe(m)
            valid = doe[~np.isnan(doe)]
            assert len(valid) > 0, f"market {m}: DOE all NaN"
            assert np.all(np.isfinite(valid)), f"market {m}: DOE has inf"

    def test_dom_index_returns_index_version(self, janus_obj):
        dom_idx = janus_obj.dom(None)
        dom_equity = janus_obj.dom_index_equity
        np.testing.assert_array_equal(dom_idx, dom_equity)

    def test_doe_index_returns_index_version(self, janus_obj):
        doe_idx = janus_obj.doe(None)
        valid = doe_idx[~np.isnan(doe_idx)]
        assert len(valid) > 0


# ---------------------------------------------------------------------------
# RM tests
# ---------------------------------------------------------------------------


class TestRM:
    def test_rm_range(self, janus_obj):
        for m in range(N_MARKETS):
            rm = janus_obj.rm(m)
            valid = rm[~np.isnan(rm)]
            if len(valid) > 0:
                assert np.all(valid >= -300.0), f"market {m}: RM < -300"
                assert np.all(valid <= 300.0), f"market {m}: RM > 300"

    def test_rm_fractile_range(self, janus_obj):
        for m in range(N_MARKETS):
            frac = janus_obj.rm_fractile(m)
            valid = frac[~np.isnan(frac)]
            if len(valid) > 0:
                assert np.all(valid >= 0.0), f"market {m}: RM fractile < 0"
                assert np.all(valid <= 1.0), f"market {m}: RM fractile > 1"


# ---------------------------------------------------------------------------
# Performance spread tests
# ---------------------------------------------------------------------------


class TestPerformanceSpread:
    def test_rs_ps_equals_leader_minus_laggard(self, janus_obj):
        ps = janus_obj.rs_ps
        leader = janus_obj.rs_leader_equity
        laggard = janus_obj.rs_laggard_equity
        valid = ~np.isnan(ps)
        if np.any(valid):
            np.testing.assert_allclose(
                ps[valid],
                leader[valid] - laggard[valid],
                atol=1e-12,
            )

    def test_rm_ps_equals_leader_minus_laggard(self, janus_obj):
        ps = janus_obj.rm_ps
        leader = janus_obj.rm_leader_equity
        laggard = janus_obj.rm_laggard_equity
        valid = ~np.isnan(ps)
        if np.any(valid):
            np.testing.assert_allclose(
                ps[valid],
                leader[valid] - laggard[valid],
                atol=1e-12,
            )

    def test_rs_leader_advantage_equals_leader_minus_avg(self, janus_obj):
        adv = janus_obj.rs_leader_advantage
        leader = janus_obj.rs_leader_equity
        avg = janus_obj.oos_avg
        valid = ~np.isnan(adv)
        if np.any(valid):
            np.testing.assert_allclose(
                adv[valid],
                leader[valid] - avg[valid],
                atol=1e-12,
            )

    def test_rs_laggard_advantage_equals_laggard_minus_avg(self, janus_obj):
        adv = janus_obj.rs_laggard_advantage
        laggard = janus_obj.rs_laggard_equity
        avg = janus_obj.oos_avg
        valid = ~np.isnan(adv)
        if np.any(valid):
            np.testing.assert_allclose(
                adv[valid],
                laggard[valid] - avg[valid],
                atol=1e-12,
            )

    def test_rs_leader_beats_laggard_on_average(self):
        """With diverse markets, leader should generally outperform laggard."""
        closes = [_make_close(i, n=400, drift=0.001 * (i - 2)) for i in range(5)]
        j = Janus(closes, lookback=LOOKBACK)
        leader = j.rs_leader_equity
        laggard = j.rs_laggard_equity
        valid_l = leader[~np.isnan(leader)]
        valid_g = laggard[~np.isnan(laggard)]
        if len(valid_l) > 0 and len(valid_g) > 0:
            assert valid_l[-1] >= valid_g[-1], "Leader should beat laggard overall"


# ---------------------------------------------------------------------------
# CMA tests
# ---------------------------------------------------------------------------


class TestCMA:
    def test_cma_oos_finite(self, janus_obj):
        cma = janus_obj.cma_oos
        valid = cma[~np.isnan(cma)]
        assert len(valid) > 0, "cma_oos all NaN"
        assert np.all(np.isfinite(valid)), "cma_oos has inf"

    def test_leader_cma_oos_finite(self, janus_obj):
        cma = janus_obj.leader_cma_oos
        valid = cma[~np.isnan(cma)]
        assert len(valid) > 0, "leader_cma_oos all NaN"
        assert np.all(np.isfinite(valid)), "leader_cma_oos has inf"


# ---------------------------------------------------------------------------
# Functional wrapper tests
# ---------------------------------------------------------------------------

WRAPPER_FUNCTIONS = [
    ("janus_market_index",        lambda c: janus_market_index(c, lookback=LOOKBACK)),
    ("janus_rs",                  lambda c: janus_rs(c, market=0, lookback=LOOKBACK)),
    ("janus_rs_fractile",         lambda c: janus_rs_fractile(c, market=0, lookback=LOOKBACK)),
    ("janus_delta_rs_fractile",   lambda c: janus_delta_rs_fractile(c, market=0, lookback=LOOKBACK, delta_length=10)),
    ("janus_rss",                 lambda c: janus_rss(c, lookback=LOOKBACK)),
    ("janus_delta_rss",           lambda c: janus_delta_rss(c, lookback=LOOKBACK)),
    ("janus_dom",                 lambda c: janus_dom(c, market=0, lookback=LOOKBACK)),
    ("janus_doe",                 lambda c: janus_doe(c, market=0, lookback=LOOKBACK)),
    ("janus_dom_index",           lambda c: janus_dom_index(c, lookback=LOOKBACK)),
    ("janus_rm",                  lambda c: janus_rm(c, market=0, lookback=LOOKBACK)),
    ("janus_rm_fractile",         lambda c: janus_rm_fractile(c, market=0, lookback=LOOKBACK)),
    ("janus_delta_rm_fractile",   lambda c: janus_delta_rm_fractile(c, market=0, lookback=LOOKBACK, delta_length=10)),
    ("janus_rs_leader_equity",    lambda c: janus_rs_leader_equity(c, lookback=LOOKBACK)),
    ("janus_rs_laggard_equity",   lambda c: janus_rs_laggard_equity(c, lookback=LOOKBACK)),
    ("janus_rs_ps",               lambda c: janus_rs_ps(c, lookback=LOOKBACK)),
    ("janus_rs_leader_advantage", lambda c: janus_rs_leader_advantage(c, lookback=LOOKBACK)),
    ("janus_rs_laggard_advantage", lambda c: janus_rs_laggard_advantage(c, lookback=LOOKBACK)),
    ("janus_rm_leader_equity",    lambda c: janus_rm_leader_equity(c, lookback=LOOKBACK)),
    ("janus_rm_laggard_equity",   lambda c: janus_rm_laggard_equity(c, lookback=LOOKBACK)),
    ("janus_rm_ps",               lambda c: janus_rm_ps(c, lookback=LOOKBACK)),
    ("janus_rm_leader_advantage", lambda c: janus_rm_leader_advantage(c, lookback=LOOKBACK)),
    ("janus_rm_laggard_advantage", lambda c: janus_rm_laggard_advantage(c, lookback=LOOKBACK)),
    ("janus_oos_avg",             lambda c: janus_oos_avg(c, lookback=LOOKBACK)),
    ("janus_cma_oos",             lambda c: janus_cma_oos(c, lookback=LOOKBACK)),
    ("janus_leader_cma_oos",      lambda c: janus_leader_cma_oos(c, lookback=LOOKBACK)),
]


class TestWrappers:
    @pytest.mark.parametrize("name,fn", WRAPPER_FUNCTIONS)
    def test_wrapper_output_length(self, random_closes, name, fn):
        result = fn(random_closes)
        assert len(result) == N_BARS, f"{name}: expected {N_BARS}, got {len(result)}"

    @pytest.mark.parametrize("name,fn", WRAPPER_FUNCTIONS)
    def test_wrapper_output_dtype(self, random_closes, name, fn):
        result = fn(random_closes)
        assert result.dtype == np.float64, f"{name}: expected float64"

    @pytest.mark.parametrize("name,fn", WRAPPER_FUNCTIONS)
    def test_wrapper_has_valid_values(self, random_closes, name, fn):
        result = fn(random_closes)
        assert not np.all(np.isnan(result)), f"{name}: all NaN"

    def test_wrapper_rs_matches_class(self, random_closes):
        j = Janus(random_closes, lookback=LOOKBACK)
        wrapper_result = janus_rs(random_closes, market=0, lookback=LOOKBACK)
        np.testing.assert_array_equal(j.rs(0), wrapper_result)

    def test_wrapper_market_index_matches_class(self, random_closes):
        j = Janus(random_closes, lookback=LOOKBACK)
        wrapper_result = janus_market_index(random_closes, lookback=LOOKBACK)
        np.testing.assert_array_equal(j.market_index, wrapper_result)

    def test_wrapper_rss_matches_class(self, random_closes):
        j = Janus(random_closes, lookback=LOOKBACK)
        wrapper_result = janus_rss(random_closes, lookback=LOOKBACK)
        np.testing.assert_array_equal(j.rss, wrapper_result)


# ---------------------------------------------------------------------------
# Delta / smoothing tests
# ---------------------------------------------------------------------------


class TestDeltaAndSmoothing:
    def test_delta_rs_fractile_warmup(self, random_closes):
        delta_len = 15
        result = janus_delta_rs_fractile(
            random_closes, market=0, lookback=LOOKBACK, delta_length=delta_len,
        )
        # First delta_length bars should be NaN
        assert np.all(np.isnan(result[:delta_len]))

    def test_delta_rs_fractile_range(self, random_closes):
        result = janus_delta_rs_fractile(
            random_closes, market=0, lookback=LOOKBACK, delta_length=10,
        )
        valid = result[~np.isnan(result)]
        assert np.all(valid >= -1.0) and np.all(valid <= 1.0)

    def test_delta_rm_fractile_range(self, random_closes):
        result = janus_delta_rm_fractile(
            random_closes, market=0, lookback=LOOKBACK, delta_length=10,
        )
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= -1.0) and np.all(valid <= 1.0)

    def test_rss_smoothing_reduces_variance(self, random_closes):
        raw = janus_rss(random_closes, lookback=LOOKBACK, smoothing=0)
        smoothed = janus_rss(random_closes, lookback=LOOKBACK, smoothing=20)
        raw_valid = raw[~np.isnan(raw)]
        smooth_valid = smoothed[~np.isnan(smoothed)]
        if len(raw_valid) > 10 and len(smooth_valid) > 10:
            assert np.std(smooth_valid) <= np.std(raw_valid) + 1e-10

    def test_delta_rss_smoothing(self, random_closes):
        raw = janus_delta_rss(random_closes, lookback=LOOKBACK, smoothing=0)
        smoothed = janus_delta_rss(random_closes, lookback=LOOKBACK, smoothing=20)
        assert len(raw) == len(smoothed) == N_BARS


# ---------------------------------------------------------------------------
# Deterministic / edge-case tests
# ---------------------------------------------------------------------------


class TestDeterministic:
    def test_all_identical_markets(self):
        """All markets identical → RS=0, RSS=0 for all."""
        c = _make_close(99, n=250)
        closes = [c.copy() for _ in range(4)]
        j = Janus(closes, lookback=LOOKBACK)

        for m in range(4):
            rs = j.rs(m)
            valid = rs[~np.isnan(rs)]
            assert np.allclose(valid, 0.0, atol=1e-6)

        rss = j.rss
        valid_rss = rss[~np.isnan(rss)]
        assert np.allclose(valid_rss, 0.0, atol=1e-6)

    def test_market_index_is_cumsum_of_median_returns(self, random_closes):
        """market_index should be cumulative sum of median log-returns."""
        j = Janus(random_closes, lookback=LOOKBACK)
        mi = j.market_index

        # Recompute from scratch
        lr = np.array([np.log(c[1:] / c[:-1]) for c in random_closes])
        med = np.median(lr, axis=0)
        expected = np.empty(N_BARS)
        expected[0] = 0.0
        expected[1:] = np.cumsum(med)

        np.testing.assert_allclose(mi, expected, atol=1e-12)

    def test_different_lookbacks_give_different_rs(self, random_closes):
        j1 = Janus(random_closes, lookback=40)
        j2 = Janus(random_closes, lookback=100)
        rs1 = j1.rs(0)
        rs2 = j2.rs(0)
        # They should not be identical (different windows)
        # Both have valid values — compare where both are valid
        both_valid = ~np.isnan(rs1) & ~np.isnan(rs2)
        if np.sum(both_valid) > 10:
            assert not np.allclose(rs1[both_valid], rs2[both_valid])

    def test_different_markets_have_different_rs(self, janus_obj):
        rs0 = janus_obj.rs(0)
        rs1 = janus_obj.rs(1)
        both_valid = ~np.isnan(rs0) & ~np.isnan(rs1)
        if np.sum(both_valid) > 10:
            assert not np.allclose(rs0[both_valid], rs1[both_valid])

    def test_strongly_divergent_markets_have_large_rss(self):
        """Markets with very different behaviors → large RSS."""
        n = 300
        closes = []
        for i in range(5):
            drift = 0.01 * (i - 2)  # from -0.02 to +0.02
            closes.append(_make_close(i, n=n, drift=drift))
        j = Janus(closes, lookback=LOOKBACK)
        rss = j.rss
        valid = rss[~np.isnan(rss)]
        assert np.mean(valid[-50:]) > 5.0, "Divergent markets should have large RSS"

    def test_per_market_dom_doe_covers_all_markets(self, janus_obj):
        """Each market should have dom/doe values."""
        for m in range(N_MARKETS):
            dom = janus_obj.dom(m)
            doe = janus_obj.doe(m)
            assert not np.all(np.isnan(dom)), f"market {m}: DOM all NaN"
            assert not np.all(np.isnan(doe)), f"market {m}: DOE all NaN"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_janus_class_importable_from_multi_market(self):
        from okmich_quant_features.timothymasters.multi_market import Janus as J
        assert J is Janus

    def test_functional_wrappers_importable_from_multi_market(self):
        from okmich_quant_features.timothymasters.multi_market import janus_rs
        assert callable(janus_rs)

    def test_functional_wrappers_importable_from_janus_module(self):
        from okmich_quant_features.timothymasters.multi_market.janus import janus_rs
        assert callable(janus_rs)


# ---------------------------------------------------------------------------
# Regression tests for correctness fixes
# ---------------------------------------------------------------------------

class TestDeltaFractileGuard:
    """Tests for Fix #7: delta_length=0 guard in delta wrapper functions."""

    @pytest.fixture
    def closes_5(self):
        return [_make_close(seed=i * 7 + 1) for i in range(5)]

    def test_delta_rs_fractile_zero_raises(self, closes_5):
        with pytest.raises(ValueError, match="delta_length must be >= 1"):
            janus_delta_rs_fractile(closes_5, market=0, lookback=LOOKBACK, delta_length=0)

    def test_delta_rs_fractile_negative_raises(self, closes_5):
        with pytest.raises(ValueError, match="delta_length must be >= 1"):
            janus_delta_rs_fractile(closes_5, market=0, lookback=LOOKBACK, delta_length=-5)

    def test_delta_rs_fractile_one_valid(self, closes_5):
        """delta_length=1 should produce a valid result array."""
        result = janus_delta_rs_fractile(closes_5, market=0, lookback=LOOKBACK, delta_length=1)
        assert result.shape == (N_BARS,)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0

    def test_delta_rm_fractile_zero_raises(self, closes_5):
        with pytest.raises(ValueError, match="delta_length must be >= 1"):
            janus_delta_rm_fractile(closes_5, market=0, lookback=LOOKBACK, delta_length=0)

    def test_delta_rm_fractile_negative_raises(self, closes_5):
        with pytest.raises(ValueError, match="delta_length must be >= 1"):
            janus_delta_rm_fractile(closes_5, market=0, lookback=LOOKBACK, delta_length=-3)

    def test_delta_rm_fractile_one_valid(self, closes_5):
        """delta_length=1 should produce a valid result array."""
        result = janus_delta_rm_fractile(closes_5, market=0, lookback=LOOKBACK, delta_length=1)
        assert result.shape == (N_BARS,)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
