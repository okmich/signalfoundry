"""Golden-value characterization tests for trend_persistence_labeling.

The existing test_trend_persistence.py is mostly shape/type assertions. This file adds
golden-value tests that lock current behavior at specific indices — including the
warmup-as-zero quirk (the function fillna(0.0)s NaN warmup, conflating "insufficient
data" with "neutral").

Fixes that change these contracts must update these tests deliberately.
"""

import numpy as np
import pandas as pd

from okmich_quant_features.trend import trend_persistence_labeling


# ---------------------------------------------------------------------------
# Warmup behavior — NaN means "insufficient data", distinct from neutral 0
# ---------------------------------------------------------------------------

def test_warmup_region_is_nan():
    """First (window - 1) rows have no rolling vol -> score NaN -> label NaN."""
    prices = pd.Series(np.linspace(100, 200, 30))
    out = trend_persistence_labeling(prices, window=5, smooth=2, zscore_norm=False)
    assert out.iloc[:5].isna().all()
    assert not out.iloc[5:].isna().any()


def test_warmup_region_with_zscore_norm_extends():
    """zscore_norm adds another rolling pass -> warmup region is longer (6 NaN rows)."""
    prices = pd.Series(np.linspace(100, 200, 30))
    out = trend_persistence_labeling(prices, window=5, smooth=2, zscore_norm=True)
    assert out.iloc[:6].isna().all()
    assert not out.iloc[6:].isna().any()


def test_constant_prices_yield_warmup_nan_then_zeros():
    """No drift, no signal — warmup is NaN, post-warmup is 0 (neutral)."""
    prices = pd.Series([100.0] * 20)
    out = trend_persistence_labeling(prices, window=5, smooth=2, zscore_norm=False)
    assert out.iloc[:5].isna().all()
    np.testing.assert_array_equal(out.iloc[5:].values, np.zeros(15))


def test_single_row_returns_nan():
    out = trend_persistence_labeling(pd.Series([100.0]))
    assert len(out) == 1
    assert np.isnan(out.iloc[0])


# ---------------------------------------------------------------------------
# Golden label sequences
# ---------------------------------------------------------------------------

def test_monotonic_uptrend_labels_one_after_warmup():
    prices = pd.Series(np.linspace(100, 200, 30))
    out = trend_persistence_labeling(prices, window=5, smooth=2, zscore_norm=False)
    assert out.iloc[:5].isna().all()
    np.testing.assert_array_equal(out.iloc[5:].values, np.ones(25))


def test_monotonic_uptrend_with_zscore_norm():
    prices = pd.Series(np.linspace(100, 200, 30))
    out = trend_persistence_labeling(prices, window=5, smooth=2, zscore_norm=True)
    assert out.iloc[:6].isna().all()
    np.testing.assert_array_equal(out.iloc[6:].values, np.ones(24))


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------

def test_returns_series_with_input_index():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    prices = pd.Series(np.linspace(100, 120, 20), index=idx)
    out = trend_persistence_labeling(prices)
    assert isinstance(out, pd.Series)
    pd.testing.assert_index_equal(out.index, idx)


def test_name_is_trend_label():
    """The output Series is always named 'trend_label'; there is no `name=` parameter."""
    out = trend_persistence_labeling(pd.Series([100.0] * 10))
    assert out.name == "trend_label"


def test_output_dtype_is_float64():
    out = trend_persistence_labeling(pd.Series([100.0] * 10))
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Threshold (currently hardcoded at +/-0.25)
# ---------------------------------------------------------------------------

def test_strong_uptrend_with_zscore_norm_fires_positive():
    """Exponential growth -> z-scored drift well above 0.25."""
    prices = pd.Series(np.exp(np.linspace(0, 2, 50)) * 100)
    out = trend_persistence_labeling(prices, window=10, smooth=3, zscore_norm=True)
    # After warmup, should be predominantly +1
    post_warmup = out.iloc[15:]
    assert (post_warmup == 1.0).mean() > 0.5