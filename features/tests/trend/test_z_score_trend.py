"""Tests for zscore_trend_features."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import zscore_trend_features


def test_returns_dataframe_with_three_columns():
    prices = pd.Series(np.linspace(100, 110, 50))
    out = zscore_trend_features(prices, window=10, deriv_window=3)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["zscore_10", "zscore_deriv_10", "zscore_abs_10"]


def test_index_preserved():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    prices = pd.Series(np.linspace(100, 110, 50), index=idx)
    out = zscore_trend_features(prices, window=10, deriv_window=3)
    pd.testing.assert_index_equal(out.index, idx)


def test_window_below_2_raises():
    with pytest.raises(ValueError, match="window must be >= 2"):
        zscore_trend_features(pd.Series([100.0] * 20), window=1)


def test_deriv_window_below_1_raises():
    with pytest.raises(ValueError, match="deriv_window must be >= 1"):
        zscore_trend_features(pd.Series([100.0] * 20), window=10, deriv_window=0)


def test_abs_column_is_non_negative_where_defined():
    prices = pd.Series(np.linspace(100, 110, 50))
    out = zscore_trend_features(prices, window=10, deriv_window=3)
    non_nan = out["zscore_abs_10"].dropna()
    assert (non_nan >= 0).all()
