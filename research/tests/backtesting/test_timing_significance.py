"""Tests for the timing-significance & beta/timing decomposition primitives (planted-edge cases)."""
import numpy as np
import pytest

from okmich_quant_research.backtesting.timing_significance import (
    net_bar_returns,
    circular_shift_null,
    beta_timing_decomposition,
    CircularShiftNull,
    BetaTimingDecomposition,
)


# --- net_bar_returns ---------------------------------------------------------------------------------

def test_net_bar_returns_causal_execution():
    # position known at t-1 earns bar t's return; no cost
    pos = np.array([0.0, 1.0, 1.0, 1.0])
    r = np.array([0.1, 0.2, 0.3, 0.4])
    net = net_bar_returns(pos, r)
    np.testing.assert_allclose(net, [0.0, 0.0, 0.3, 0.4])   # lag = [0,0,1,1] -> gross = [0,0,0.3,0.4]


def test_net_bar_returns_charges_turnover_cost():
    pos = np.array([1.0, -1.0])           # open long (turnover 1), flip to short (turnover 2)
    net = net_bar_returns(pos, np.array([0.0, 0.0]), spread_frac=0.01)
    np.testing.assert_allclose(net, [-0.01, -0.02])


def test_net_bar_returns_missing_return_zeroes_gross_but_still_charges_cost():
    # position flips on the NaN-return bar -> gross 0 there, but the turnover cost is STILL charged (no upward bias)
    net = net_bar_returns(np.array([1.0, -1.0, -1.0]), np.array([0.1, np.nan, 0.1]), spread_frac=0.02)
    np.testing.assert_allclose(net, [-0.02, -0.04, -0.1])  # turnover=[1,2,0]; gross=[0,0,-0.1]; cost=[.02,.04,0]


def test_net_bar_returns_missing_return_zero_when_no_turnover():
    net = net_bar_returns(np.array([1.0, 1.0, 1.0]), np.array([0.5, np.nan, 0.5]))
    np.testing.assert_allclose(net, [0.0, 0.0, 0.5])       # lag=[0,1,1]; NaN bar -> 0; no turnover -> no cost


def test_net_bar_returns_shape_mismatch_raises():
    with pytest.raises(ValueError):
        net_bar_returns(np.zeros(5), np.zeros(4))


def test_rejects_invalid_parameters():
    r = np.zeros(120)
    good_pos = np.sign(np.random.default_rng(0).standard_normal(120))
    with pytest.raises(ValueError):                                    # 2-D spread would broadcast to a 2-D result
        net_bar_returns(np.zeros(120), r, spread_frac=np.zeros((120, 1)))
    with pytest.raises(ValueError):                                    # negative spread = rebate
        net_bar_returns(np.zeros(120), r, spread_frac=-0.01)
    with pytest.raises(ValueError):                                    # NaN spread would erase bars
        net_bar_returns(np.zeros(120), r, spread_frac=np.nan)
    with pytest.raises(ValueError):                                    # inf position (nan_to_num would make it huge)
        net_bar_returns(np.full(120, np.inf), r)
    with pytest.raises(ValueError):
        circular_shift_null(good_pos, r, periods_per_year=252, n_shuffle=0)
    with pytest.raises(ValueError):
        circular_shift_null(good_pos, r, periods_per_year=0)


# --- beta_timing_decomposition -----------------------------------------------------------------------

def test_beta_timing_algebra_sums_to_total():
    rng = np.random.default_rng(1)
    pos = rng.integers(-1, 2, 500).astype(float)
    r = rng.standard_normal(500)
    r[::37] = np.nan                                       # NaN-safe
    d = beta_timing_decomposition(pos, r)
    assert isinstance(d, BetaTimingDecomposition)
    np.testing.assert_allclose(d.beta + d.timing, d.total, atol=1e-12)


def test_beta_timing_constant_exposure_on_trend_is_all_beta():
    rng = np.random.default_rng(2)
    n = 1000
    r = 0.001 + 0.001 * rng.standard_normal(n)             # positive drift = trend
    pos = np.ones(n)                                       # constant long -> the return is pure beta
    d = beta_timing_decomposition(pos, r)
    assert d.beta > 0
    assert abs(d.timing_share) < 0.05                      # timing negligible vs beta
    assert d.mean_exposure == pytest.approx(0.999, abs=1e-3)


def test_beta_timing_perfect_foresight_zero_mean_is_all_timing():
    rng = np.random.default_rng(3)
    n = 1000
    r = rng.standard_normal(n)                             # symmetric, ~zero mean
    pos = np.zeros(n)
    pos[:-1] = np.sign(r[1:])                              # perfect 1-bar foresight -> gross = |r|
    d = beta_timing_decomposition(pos, r)
    assert d.total > 0
    assert abs(d.mean_exposure) < 0.1                      # ~balanced long/short -> ~zero average exposure
    assert d.timing_share > 0.9                            # essentially all timing


# --- circular_shift_null -----------------------------------------------------------------------------

def test_circular_shift_null_detects_perfect_timing():
    rng = np.random.default_rng(4)
    n = 2000
    r = rng.standard_normal(n)
    pos = np.zeros(n)
    pos[:-1] = np.sign(r[1:])                              # perfect foresight -> real >> any re-alignment
    cs = circular_shift_null(pos, r, periods_per_year=252, n_shuffle=200, min_offset=50, seed=0)
    assert isinstance(cs, CircularShiftNull)
    assert cs.real_sharpe > cs.beta_sharpe                 # real beats the beta/structure benchmark
    assert cs.percentile > 0.95
    assert cs.clears(0.95) is True


def test_circular_shift_null_no_timing_real_near_null_center():
    # position independent of returns -> the real alignment is exchangeable with the null draws, so the real
    # Sharpe sits near the null mean. Robust 4-sigma bound (seeded, deterministic) — not a min/max bracket.
    r = np.random.default_rng(5).standard_normal(2000)
    pos = np.sign(np.random.default_rng(999).standard_normal(2000))     # structure, independent of r
    cs = circular_shift_null(pos, r, periods_per_year=252, n_shuffle=300, min_offset=50, seed=0)
    assert abs(cs.real_sharpe - cs.null_sharpes.mean()) < 4.0 * cs.null_sharpes.std()
    assert not cs.clears(0.95)


def test_circular_shift_null_preserves_turnover_exactly():
    # r=0 with a constant spread: each shifted null = -rotation(turnover)·spread; a rotation preserves the cost
    # multiset, so EVERY shifted Sharpe is identical -> turnover is exactly preserved. (Non-circular scoring
    # would let the boundary turnover vary from shift to shift and break this equality.)
    pos = np.sign(np.random.default_rng(11).standard_normal(500))
    cs = circular_shift_null(pos, np.zeros(500), periods_per_year=252, n_shuffle=100, spread_frac=0.01,
                             min_offset=50, seed=0)
    finite = cs.null_sharpes[np.isfinite(cs.null_sharpes)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, finite[0])


def test_circular_shift_null_is_deterministic_under_seed():
    rng = np.random.default_rng(6)
    r = rng.standard_normal(800)
    pos = np.sign(rng.standard_normal(800))
    a = circular_shift_null(pos, r, periods_per_year=252, seed=7)
    b = circular_shift_null(pos, r, periods_per_year=252, seed=7)
    np.testing.assert_array_equal(a.null_sharpes, b.null_sharpes)
    assert a.real_sharpe == b.real_sharpe


def test_circular_shift_null_too_short_raises():
    with pytest.raises(ValueError):
        circular_shift_null(np.array([1.0, -1.0, 1.0]), np.array([0.1, 0.2, 0.3]), periods_per_year=252)
