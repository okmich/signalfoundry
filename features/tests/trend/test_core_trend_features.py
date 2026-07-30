"""Smoke tests for core_trend_features — the concatenated core trend bundle.

Focus: the CTL feature bundle is wired in (omega-tagged columns), the frame is index-aligned, and the
omega-tagged ctl_direction column still matches the standalone labeler.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import continuous_trend_labeling, core_trend_features


@pytest.fixture
def ohlc_df():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    # deterministic up-down-up path so CTL actually triggers and flips
    close = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, n))
    return pd.DataFrame({"high": close + 1.0, "low": close - 1.0, "close": close}, index=idx)


def test_ctl_bundle_columns_present_and_omega_tagged(ohlc_df):
    out = core_trend_features(ohlc_df, continuous_omega=0.05)
    for stem in ["ctl_direction", "ctl_trend_age", "ctl_retrace_frac", "ctl_leg_return", "ctl_flip_count"]:
        assert f"{stem}_0_05" in out.columns, stem
    # old redundant raw column is gone (ctl_direction replaces it)
    assert "continuous_trend_0_05" not in out.columns


def test_index_preserved(ohlc_df):
    out = core_trend_features(ohlc_df, continuous_omega=0.05)
    pd.testing.assert_index_equal(out.index, ohlc_df.index)


def test_ctl_direction_matches_standalone_labeler(ohlc_df):
    out = core_trend_features(ohlc_df, continuous_omega=0.05)
    batch = continuous_trend_labeling(ohlc_df["close"], omega=0.05)
    np.testing.assert_array_equal(out["ctl_direction_0_05"].to_numpy(), batch.to_numpy())


def test_flip_window_threads_through(ohlc_df):
    wide = core_trend_features(ohlc_df, continuous_omega=0.05, continuous_flip_window=50)
    narrow = core_trend_features(ohlc_df, continuous_omega=0.05, continuous_flip_window=2)
    # a shorter trailing window can only count fewer-or-equal flips at the last bar
    assert narrow["ctl_flip_count_0_05"].iloc[-1] <= wide["ctl_flip_count_0_05"].iloc[-1]
