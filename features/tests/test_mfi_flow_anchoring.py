"""Regression tests for the frame-anchored MFI flows fixed on 2026-10-08.

``mfi_volume_features`` returned the input close as a column, and both MFI builders took ``cum_dfp``, ``cum_dmfi`` and
``cum_bsdi`` as cumulative sums from the first bar of the frame. Those are integrated levels: causal, but their value
depends on where the history starts, so a live process with a shorter history computes a different number, and a
screen ranks the level by spurious regression (measured: Boruta-confirmed in 9 of 16 screens of the semivariance
study, the close at #1 on two symbols). The flows are now sums over the trailing ``rolling_window`` bars.

The start-cut test computes on the frame and on the frame without its first ``START`` bars; after ``SETTLE`` bars of
history the two must agree. The control re-creates the old cumsum and shows the same check fails on it.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volume import mfi_features, mfi_volume_features

N = 2400
START = 300
SETTLE = 200            # > rolling_window (60): the adaptive expanding warm-up has been replaced by the rolling sum
FLOWS = ["cum_dfp", "cum_dmfi", "cum_bsdi"]
ANCHORABLE = FLOWS + [f"{c}_mean" for c in FLOWS] + [f"{c}_z" for c in FLOWS] + ["norm_cum_bsdi", "norm_cum_dfp",
                                                                                "flow_momentum_dfp", "flow_momentum_bsdi"]


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    tick = 0.0001
    close = 1.1 + np.cumsum(np.round(rng.standard_t(4, N) * 3.0) * tick)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.round(rng.exponential(2.0, N)) * tick
    low = np.minimum(open_, close) - np.round(rng.exponential(2.0, N)) * tick
    vol = rng.integers(20, 300, N).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "tick_volume": vol},
                        index=pd.date_range("2024-01-01", periods=N, freq="5min"))


def _start_cut_diff(full: pd.Series, part: pd.Series) -> float:
    a = full.to_numpy(float)[START + SETTLE:]
    b = part.to_numpy(float)[SETTLE:]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 1000 and not (np.isfinite(a) ^ np.isfinite(b)).any()
    return float(np.max(np.abs(a[both] - b[both])) / np.std(a[both]))


@pytest.mark.parametrize("builder", [mfi_features, mfi_volume_features])
def test_flows_do_not_depend_on_where_the_history_starts(ohlcv, builder):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = builder(ohlcv)
        part = builder(ohlcv.iloc[START:])
    for col in ANCHORABLE:
        assert _start_cut_diff(full[col], part[col]) < 1e-9, col


def test_old_cumsum_flow_fails_the_same_check(ohlcv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfp_full = mfi_features(ohlcv)["dfp"]
        dfp_part = mfi_features(ohlcv.iloc[START:])["dfp"]
    assert _start_cut_diff(dfp_full.cumsum(), dfp_part.cumsum()) > 0.01


def test_flow_is_the_trailing_window_sum(ohlcv):
    out = mfi_features(ohlcv, rolling_window=60)
    for flow, base in [("cum_dfp", "dfp"), ("cum_dmfi", "dmfi"), ("cum_bsdi", "bsdi")]:
        expected = out[base].fillna(0.0).rolling(60, min_periods=60).sum()
        pd.testing.assert_series_equal(out[flow].iloc[59:], expected.iloc[59:], check_names=False)


def test_zero_range_bar_contributes_nothing_and_does_not_blank_the_flow(ohlcv):
    """dmfi and bsdi are NaN on a zero-range bar (the range is NaN-guarded); the cumsum skipped it, and the trailing
    sum must too — left NaN, one such bar would blank the next rolling_window values."""
    flat = ohlcv.copy()
    i = 1000
    flat.iloc[i, flat.columns.get_indexer(["open", "high", "low", "close"])] = flat["close"].iloc[i]
    out = mfi_features(flat, rolling_window=60)
    assert np.isnan(out["dmfi"].iloc[i]) and np.isnan(out["bsdi"].iloc[i])
    assert out[["cum_dmfi", "cum_bsdi"]].iloc[i:i + 60].notna().all().all()


def test_mfi_volume_features_does_not_pass_the_input_through(ohlcv):
    out = mfi_volume_features(ohlcv)
    assert not set(out.columns) & set(ohlcv.columns)
    assert {"mfi_classic", "cum_dfp", "dominance_ratio"} <= set(out.columns)
