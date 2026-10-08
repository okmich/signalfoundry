"""Regression tests for the look-ahead fixes of 2026-10-08.

Each function below once normalised or binned with a statistic of the WHOLE series (a doji cap at the 99th
percentile, ``pd.qcut`` volume bins, a per-bin mean, a global min/max, a centred smoother), so its value at bar ``t``
changed when later bars were appended. The test is exact: compute on the frame and on its first ``CUT`` rows; the
shared rows must be identical. The fixture is doji- and tie-rich on purpose — the leaks lived in those edge cases.

The controls re-create the old whole-series computation and show the same check fails on it, so a pass here means
"causal", not "the check is blind".
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.directional_change import normalise_minmax
from okmich_quant_features.filters import smooth_loess, smooth_wavelet
from okmich_quant_features.microstructure import (absorption_weighted_depth_score, bar_absorption_ratio,
                                                  core_microstructure_features, multi_bar_depth_pressure)
from okmich_quant_features.microstructure.composites import institutional_footprint_score
from okmich_quant_features.utils.causal import (count_edges_at_or_below, prior_rolling_bins,
                                                prior_rolling_group_mean, prior_rolling_quantile)
from okmich_quant_features.volume import (binned_mfi_delta, categorized_mfi_trend, core_volume_features,
                                          discretize_volume, market_facilitation_index, mfi_volume_bin_ratio,
                                          volume_bin_mfi_persistence)

N = 2400
CUT = 1600


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    """Tick-grid prices (many dojis), integer volumes (many ties) with bursts, integer spreads."""
    rng = np.random.default_rng(7)
    tick = 0.0001
    close = 1.1 + np.cumsum(np.round(rng.standard_t(4, N) * 3.0) * tick)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.round(rng.exponential(2.0, N)) * tick
    low = np.minimum(open_, close) - np.round(rng.exponential(2.0, N)) * tick
    volume = rng.integers(20, 300, N).astype(float)
    volume[rng.random(N) < 0.02] *= 8
    idx = pd.date_range("2024-01-01", periods=N, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "tick_volume": volume,
                         "spread": rng.integers(0, 12, N).astype(float)}, index=idx)


def _frame(out, n: int) -> pd.DataFrame:
    """Numeric per-bar columns of a feature output (Series, ndarray, list, DataFrame or a tuple of those)."""
    parts = out if isinstance(out, tuple) else (out,)
    cols = {}
    for i, p in enumerate(parts):
        if isinstance(p, pd.DataFrame):
            cols.update({f"{i}:{c}": p[c].to_numpy() for c in p.columns})
        elif isinstance(p, (pd.Series, np.ndarray, list)) and len(p) == n:
            cols[f"{i}"] = np.asarray(p)
    return pd.DataFrame(cols)


def changed_rows(fn, df: pd.DataFrame, cut: int = CUT) -> dict[str, int]:
    """Per output column, how many of the first ``cut`` rows differ between the full and the truncated run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = _frame(fn(df), len(df)).iloc[:cut]
        part = _frame(fn(df.iloc[:cut]), cut)
    out = {}
    for c in full.columns:
        a, b = full[c].to_numpy(), part[c].to_numpy()
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            a, b = a.astype(float), b.astype(float)
            same = np.isclose(a, b, rtol=1e-9, atol=1e-12) | (np.isnan(a) & np.isnan(b))
        else:                                                       # categorical codes: None/NaN compare equal
            same = np.array([x == y or (pd.isna(x) and pd.isna(y)) for x, y in zip(a, b)])
        out[c] = int((~same).sum())
    assert out, "feature produced no per-bar column"
    return out


def assert_causal(fn, df: pd.DataFrame, cut: int = CUT):
    changed = changed_rows(fn, df, cut)
    assert not any(changed.values()), f"rows changed when later bars were appended: {changed}"


def assert_leaks(fn, df: pd.DataFrame, cut: int = CUT):
    assert any(changed_rows(fn, df, cut).values()), "expected a look-ahead leak, found none"


# ── the fixed functions are truncation-invariant ──────────────────────────────────────────────────

FIXED = {
    "bar_absorption_ratio": lambda d: bar_absorption_ratio(d.open, d.close, d.tick_volume),
    "absorption_weighted_depth_score": lambda d: absorption_weighted_depth_score(d.open, d.close, d.tick_volume),
    "multi_bar_depth_pressure": lambda d: multi_bar_depth_pressure(d.open, d.close, d.tick_volume),
    "institutional_footprint_score": lambda d: institutional_footprint_score(d.open, d.high, d.low, d.close,
                                                                             d.tick_volume),
    "core_microstructure_features": lambda d: core_microstructure_features(d),
    "binned_mfi_delta": lambda d: binned_mfi_delta(d.high, d.low, d.close, d.tick_volume),
    "mfi_volume_bin_ratio": lambda d: mfi_volume_bin_ratio(d.high, d.low, d.close, d.tick_volume),
    "categorized_mfi_trend": lambda d: categorized_mfi_trend(d.high, d.low, d.close, d.tick_volume),
    "volume_bin_mfi_persistence": lambda d: volume_bin_mfi_persistence(d.high, d.low, d.close, d.tick_volume),
    "discretize_volume": lambda d: discretize_volume(d.tick_volume.to_numpy(), bins=4)[0],
    "market_facilitation_index": lambda d: market_facilitation_index(d.high, d.low, d.tick_volume)[:3],
    "core_volume_features": lambda d: core_volume_features(d),
    "normalise_minmax": lambda d: normalise_minmax(d.close)[0],
    "smooth_wavelet": lambda d: smooth_wavelet(d.close),
    "smooth_loess": lambda d: smooth_loess(d.close),
}


@pytest.mark.parametrize("name", sorted(FIXED))
def test_fixed_feature_is_truncation_invariant(ohlcv, name):
    assert_causal(FIXED[name], ohlcv)


@pytest.mark.parametrize("name", ["bar_absorption_ratio", "binned_mfi_delta", "discretize_volume", "smooth_wavelet"])
def test_fixed_feature_produces_values_after_warm_up(ohlcv, name):
    """Causal must not mean empty: past the warm-up the bulk of bars carry a value."""
    out = _frame(FIXED[name](ohlcv), len(ohlcv)).iloc[600:]
    assert out.notna().mean().min() > 0.5


# ── controls: the old whole-series computations fail the same check ──────────────────────────────

def _old_doji_cap(d: pd.DataFrame) -> np.ndarray:
    body = (d.close - d.open).abs().to_numpy()
    ar = np.where(body > 1e-10, d.tick_volume.to_numpy() / np.where(body > 1e-10, body, 1.0), np.inf)
    finite = ar[np.isfinite(ar)]
    return np.where(np.isfinite(ar), ar, np.percentile(finite, 99))


def _old_qcut_bins(d: pd.DataFrame) -> np.ndarray:
    return pd.qcut(d.tick_volume.rank(method="first"), 5, labels=False).to_numpy().astype(float)


@pytest.mark.parametrize("old", [_old_doji_cap, _old_qcut_bins,
                                 lambda d: normalise_minmax(d.close, min_val=d.close.min(), max_val=d.close.max())[0]])
def test_old_whole_series_statistic_leaks(ohlcv, old):
    assert_leaks(old, ohlcv)


@pytest.mark.parametrize("smoother", [smooth_wavelet, smooth_loess])
def test_centred_smoother_still_available_and_still_leaks(ohlcv, smoother):
    with pytest.warns(UserWarning):
        smoother(ohlcv.close.iloc[:500], causal=False)
    assert_leaks(lambda d: smoother(d.close, causal=False), ohlcv)


# ── frozen parameters are honoured ────────────────────────────────────────────────────────────────

def test_normalise_minmax_fixed_bounds_are_used_as_given(ohlcv):
    lo, hi = float(ohlcv.close.min()), float(ohlcv.close.max())
    out, mn, mx = normalise_minmax(ohlcv.close, min_val=lo, max_val=hi)
    np.testing.assert_allclose(out.to_numpy(), ((ohlcv.close - lo) / (hi - lo)).to_numpy())
    assert (mn, mx) == (lo, hi)


def test_normalise_minmax_default_is_a_trailing_window():
    s = pd.Series([0.0, 10.0, 5.0, 5.0, 20.0, 0.0])
    out, mn, mx = normalise_minmax(s, lookback=3)
    np.testing.assert_allclose(out.to_numpy(), [0.0, 1.0, 0.5, 0.0, 1.0, 0.0])
    assert (mn, mx) == (0.0, 20.0)


def test_market_facilitation_index_fixed_edges_are_used_as_given(ohlcv):
    vol_edges = [0.0, 100.0, 200.0, np.inf]
    mfi = (ohlcv.high - ohlcv.low) / ohlcv.tick_volume
    mfi_edges = list(mfi.quantile([0.25, 0.5, 0.75])) + [np.inf]
    _, _, _, v_out, m_out = market_facilitation_index(ohlcv.high, ohlcv.low, ohlcv.tick_volume,
                                                      fixed_bin_edges=vol_edges, fixed_mfi_bin_edges=mfi_edges)
    np.testing.assert_array_equal(v_out, vol_edges)
    np.testing.assert_array_equal(m_out, mfi_edges)
    # frozen edges are constants, so the frozen path is causal by construction
    assert_causal(lambda d: market_facilitation_index(d.high, d.low, d.tick_volume, fixed_bin_edges=vol_edges,
                                                      fixed_mfi_bin_edges=mfi_edges)[:3], ohlcv)


def test_discretize_volume_bins_against_the_prior_range():
    vols = np.r_[np.arange(1.0, 9.0), 100.0, 0.0, 4.5]                # prior range of bar 8 is [1, 8]
    bins, bin_map = discretize_volume(vols, bins=4, lookback=8)
    assert np.isnan(bins[0]) and np.isnan(bins[1])                    # warm-up: < lookback // 4 prior bars
    assert bins[8] == 3.0                                             # above the prior range: clipped to the top bin
    assert bins[9] == 0.0                                             # below it: clipped to the bottom bin
    assert set(bin_map) == {0, 1, 2, 3} and bin_map[3][1] == np.inf


# ── the causal helpers ────────────────────────────────────────────────────────────────────────────

def test_prior_rolling_quantile_excludes_the_current_bar():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    q = prior_rolling_quantile(x, 1.0, lookback=3, min_periods=2)
    np.testing.assert_array_equal(q.to_numpy(), [np.nan, np.nan, 2.0, 3.0, 4.0])   # bar 4 never sees its own 100


def test_prior_rolling_bins_match_digitize_on_prior_edges():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.integers(0, 50, 400).astype(float))
    bins = prior_rolling_bins(x, 4, lookback=100)
    t = 300
    edges = np.quantile(x.iloc[t - 100:t], [0.25, 0.5, 0.75])
    assert bins.iloc[t] == float(np.digitize(x.iloc[t], edges))
    assert bins.iloc[:24].isna().all() and bins.iloc[25:].notna().all()


def test_count_edges_is_nan_until_every_edge_is_known():
    x = pd.Series([1.0, 2.0, 3.0])
    out = count_edges_at_or_below(x, [pd.Series([np.nan, 1.5, 1.5]), pd.Series([np.nan, np.nan, 2.5])])
    assert out.isna().tolist() == [True, True, False] and out.iloc[2] == 2.0


def test_prior_rolling_group_mean_uses_same_group_history_only():
    x = pd.Series([1.0, 10.0, 3.0, 20.0, 5.0, 30.0])
    g = pd.Series([0, 1, 0, 1, 0, 1])
    out = prior_rolling_group_mean(x, g, n_groups=2, lookback=10, min_periods=2)
    np.testing.assert_array_equal(out.to_numpy(), [np.nan, np.nan, np.nan, np.nan, 2.0, 15.0])
