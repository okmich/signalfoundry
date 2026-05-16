"""Characterization tests for continuous_trend_labeling.

These tests lock in the CURRENT behavior of the CTL function so refactors can be made safely.
Pre-trigger zeros and bare-numpy returns are intentionally asserted — they describe how the
function behaves today, not how it should behave. Fixes that change these contracts must
update these tests deliberately.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import continuous_trend_labeling


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------

def test_empty_series_returns_empty_series():
    out = continuous_trend_labeling(pd.Series([], dtype=float), omega=0.1)
    assert isinstance(out, pd.Series)
    assert len(out) == 0
    assert out.dtype == np.float64


def test_empty_numpy_returns_empty_array():
    out = continuous_trend_labeling(np.array([], dtype=float), omega=0.1)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


def test_single_bar_returns_single_nan():
    """One bar can't trigger anything — output is NaN (pre-trigger)."""
    out = continuous_trend_labeling(pd.Series([100.0]), omega=0.1)
    assert isinstance(out, pd.Series)
    assert out.isna().all()


# ---------------------------------------------------------------------------
# No-trigger paths — whole series is NaN (pre-trigger / unknown regime)
# ---------------------------------------------------------------------------

def test_no_significant_move_returns_all_nan():
    """Price never moves more than omega% from FP — whole series is NaN."""
    prices = pd.Series([100.0, 100.5, 100.2, 100.8, 100.3, 100.6])
    out = continuous_trend_labeling(prices, omega=0.1)
    assert isinstance(out, pd.Series)
    assert out.isna().all()


# ---------------------------------------------------------------------------
# Golden label sequences
# ---------------------------------------------------------------------------

def test_first_trigger_up_then_reversal_to_down():
    """
    Prices: 100 -> 110 -> 115 -> 90 -> 85, omega=0.05.
    Bar 0: pre-trigger (NaN). Bar 1: triggers up (+1). Bar 2: still up (+1).
    Bar 3: retraces > 5% from 115 -> flips to -1. Bar 4: still down (-1).
    """
    prices = pd.Series([100.0, 110.0, 115.0, 90.0, 85.0])
    out = continuous_trend_labeling(prices, omega=0.05).values
    assert np.isnan(out[0])
    np.testing.assert_array_equal(out[1:], np.array([1.0, 1.0, -1.0, -1.0]))


def test_first_trigger_down_then_reversal_to_up():
    """Mirror of the up case."""
    prices = pd.Series([100.0, 90.0, 85.0, 110.0, 115.0])
    out = continuous_trend_labeling(prices, omega=0.05).values
    assert np.isnan(out[0])
    np.testing.assert_array_equal(out[1:], np.array([-1.0, -1.0, 1.0, 1.0]))


def test_multi_reversal_sequence():
    """Down -> up -> down -> up: locks in the exact reversal cadence; pre-trigger bars are NaN."""
    prices = pd.Series([100.0, 95.0, 90.0, 100.0, 110.0, 105.0, 92.0, 95.0, 108.0])
    out = continuous_trend_labeling(prices, omega=0.05).values
    # Bars 0, 1 are pre-trigger; trigger fires at bar 2 (downward).
    assert np.isnan(out[0])
    assert np.isnan(out[1])
    np.testing.assert_array_equal(out[2:], np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0]))


# ---------------------------------------------------------------------------
# Return-type contract — Series in -> Series out (index preserved)
# ---------------------------------------------------------------------------

def test_series_input_returns_series_with_index_preserved():
    """Index is preserved end-to-end."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 110.0, 115.0, 90.0, 85.0], index=idx)
    out = continuous_trend_labeling(prices, omega=0.05)
    assert isinstance(out, pd.Series)
    pd.testing.assert_index_equal(out.index, idx)


def test_numpy_input_returns_numpy_array():
    prices = np.array([100.0, 110.0, 115.0, 90.0, 85.0])
    out = continuous_trend_labeling(prices, omega=0.05)
    assert isinstance(out, np.ndarray)


def test_output_dtype_is_float64():
    out = continuous_trend_labeling(pd.Series([100.0, 110.0, 90.0]), omega=0.05)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# Pre-trigger semantics — NaN means "unknown regime", distinct from neutral 0
# ---------------------------------------------------------------------------

def test_pre_trigger_region_is_nan():
    """Pre-trigger bars are NaN — distinct from neutral 0 which CTL never emits anyway."""
    prices = pd.Series([100.0, 100.5, 110.0])  # triggers up only on bar 2
    out = continuous_trend_labeling(prices, omega=0.05)
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == 1.0


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def test_omega_zero_raises():
    with pytest.raises(ValueError, match="omega must be > 0"):
        continuous_trend_labeling(pd.Series([100.0, 110.0]), omega=0)


def test_omega_negative_raises():
    with pytest.raises(ValueError, match="omega must be > 0"):
        continuous_trend_labeling(pd.Series([100.0, 110.0]), omega=-0.1)