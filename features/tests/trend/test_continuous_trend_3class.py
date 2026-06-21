"""Characterization tests for the band-gated 3-class derivation chain.

Locks current behavior of BandParams, compute_band_state, emit_three_class, attach_labels,
apply_3class_labels. Several assertions encode current quirks (band_state warmup conflated
with inside-band as 0; emit_three_class returning int8 zero for both NaN warmup and inside-band).
Fixes that change these contracts must update these tests deliberately.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend.continuous_trend import (
    BandParams, compute_band_state, emit_three_class, attach_labels, apply_3class_labels,
)


# ---------------------------------------------------------------------------
# BandParams validation
# ---------------------------------------------------------------------------

def test_band_params_rejects_ma_period_below_2():
    with pytest.raises(ValueError, match="ma_period must be >= 2"):
        BandParams(ma_period=1, atr_period=5, k_atr=2.0)


def test_band_params_rejects_atr_period_below_2():
    with pytest.raises(ValueError, match="atr_period must be >= 2"):
        BandParams(ma_period=5, atr_period=1, k_atr=2.0)


def test_band_params_rejects_zero_k_atr():
    with pytest.raises(ValueError, match="k_atr must be > 0"):
        BandParams(ma_period=5, atr_period=5, k_atr=0)


def test_band_params_rejects_negative_k_atr():
    with pytest.raises(ValueError, match="k_atr must be > 0"):
        BandParams(ma_period=5, atr_period=5, k_atr=-1.0)


def test_band_params_rejects_nan_k_atr():
    with pytest.raises(ValueError, match="k_atr must be > 0"):
        BandParams(ma_period=5, atr_period=5, k_atr=float("nan"))


def test_band_params_rejects_non_integer_ma_period():
    with pytest.raises(ValueError, match="ma_period must be an integer"):
        BandParams(ma_period=5.5, atr_period=5, k_atr=1.0)


def test_band_params_rejects_non_integer_atr_period():
    with pytest.raises(ValueError, match="atr_period must be an integer"):
        BandParams(ma_period=5, atr_period=5.5, k_atr=1.0)


def test_band_params_accepts_numpy_integer_periods():
    """math.ceil-style scaling and JSON configs can hand back numpy ints — those must be accepted."""
    band = BandParams(ma_period=np.int64(20), atr_period=np.int32(14), k_atr=1.0)
    assert band.ma_period == 20


def test_band_params_is_frozen():
    band = BandParams(ma_period=5, atr_period=5, k_atr=1.0)
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
        band.k_atr = 2.0


# ---------------------------------------------------------------------------
# compute_band_state
# ---------------------------------------------------------------------------

def test_band_state_classifies_above_below_inside():
    close = pd.Series([100, 105, 110, 95, 90, 100], dtype=float)
    upper = pd.Series([np.nan, 108, 108, 108, 108, 108], dtype=float)
    lower = pd.Series([np.nan, 92, 92, 92, 92, 92], dtype=float)
    state = compute_band_state(close, upper, lower)
    # Bar 0: warmup (NaN bands) -> 0
    # Bar 1: 105 inside (92..108) -> 0
    # Bar 2: 110 > 108 -> +1
    # Bar 3: 95 inside -> 0
    # Bar 4: 90 < 92 -> -1
    # Bar 5: 100 inside -> 0
    np.testing.assert_array_equal(state, np.array([0, 0, 1, 0, -1, 0], dtype=np.int8))


def test_band_state_returns_int8():
    close = pd.Series([100.0], dtype=float)
    upper = pd.Series([np.nan], dtype=float)
    lower = pd.Series([np.nan], dtype=float)
    state = compute_band_state(close, upper, lower)
    assert state.dtype == np.int8


def test_band_state_warmup_is_zero_not_nan():
    """Lock current contract: warmup bars (NaN bands) get 0, conflated with inside-band."""
    close = pd.Series([100, 100], dtype=float)
    upper = pd.Series([np.nan, 105], dtype=float)
    lower = pd.Series([np.nan, 95], dtype=float)
    state = compute_band_state(close, upper, lower)
    assert state[0] == 0  # warmup
    assert state[1] == 0  # inside band - same value, different meaning


def test_band_state_length_mismatch_raises():
    close = pd.Series([1.0] * 6)
    upper = pd.Series([1.0] * 5)
    lower = pd.Series([1.0] * 6)
    with pytest.raises(ValueError, match="length mismatch"):
        compute_band_state(close, upper, lower)


def test_band_state_is_positional_not_index_aligned():
    """Reordered indexes must NOT realign: classification follows position, not label."""
    close = pd.Series([110.0, 90.0], index=[1, 0])   # deliberately reversed index
    upper = pd.Series([108.0, 108.0], index=[0, 1])
    lower = pd.Series([92.0, 92.0], index=[0, 1])
    # Positional: pos0 close=110 > upper=108 -> +1 ; pos1 close=90 < lower=92 -> -1
    np.testing.assert_array_equal(compute_band_state(close, upper, lower), np.array([1, -1], dtype=np.int8))


def test_band_state_accepts_ndarray_inputs():
    close = np.array([110.0, 90.0, 100.0])
    upper = np.array([108.0, 108.0, 108.0])
    lower = np.array([92.0, 92.0, 92.0])
    np.testing.assert_array_equal(compute_band_state(close, upper, lower), np.array([1, -1, 0], dtype=np.int8))


# ---------------------------------------------------------------------------
# emit_three_class
# ---------------------------------------------------------------------------

def test_three_class_zeros_out_inside_band_bars():
    """Where band_state == 0, the CTL label is zeroed regardless of its value."""
    ctl = np.array([1, 1, -1, -1, 0, 1], dtype=np.int8)
    bs = np.array([0, 1, -1, 0, 1, -1], dtype=np.int8)
    out = emit_three_class(ctl, bs)
    # bs==0 -> 0; bs!=0 -> ctl
    np.testing.assert_array_equal(out, np.array([0, 1, -1, 0, 0, 1], dtype=np.int8))


def test_three_class_returns_int8():
    out = emit_three_class(np.array([1, -1], dtype=np.int8), np.array([1, 1], dtype=np.int8))
    assert out.dtype == np.int8


def test_three_class_length_mismatch_raises():
    ctl = np.array([1, 1, -1, -1, 0, 1], dtype=np.int8)
    bs = np.array([0, 1, -1, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="length mismatch"):
        emit_three_class(ctl, bs)


def test_three_class_handles_nan_ctl_directly():
    """Direct callers passing raw CTL output (with NaN warmup) get safe int8 output."""
    ctl = np.array([np.nan, np.nan, 1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    bs = np.array([1, 0, 1, -1, -1, 0], dtype=np.int8)
    out = emit_three_class(ctl, bs)
    # NaN CTL is treated as no-signal; +1 ctl survives even when band_state=-1 (gate-only).
    np.testing.assert_array_equal(out, np.array([0, 0, 1, 1, -1, 0], dtype=np.int8))
    assert out.dtype == np.int8


# ---------------------------------------------------------------------------
# attach_labels — full chain
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlc_df():
    return pd.DataFrame({
        "high":  [101.0, 111.0, 116.0, 91.0, 86.0, 91.0, 106.0, 111.0, 116.0, 91.0],
        "low":   [99.0,  109.0, 114.0, 89.0, 84.0, 89.0, 104.0, 109.0, 114.0, 89.0],
        "close": [100.0, 110.0, 115.0, 90.0, 85.0, 90.0, 105.0, 110.0, 115.0, 90.0],
    }, index=pd.RangeIndex(10))


def test_attach_labels_adds_expected_columns(ohlc_df):
    band = BandParams(ma_period=3, atr_period=3, k_atr=1.0)
    out = attach_labels(ohlc_df, omega=0.05, band=band)
    expected_cols = {"high", "low", "close", "upper", "ma", "lower",
                     "band_state", "ctl_label", "ctl_label_3class"}
    assert expected_cols.issubset(set(out.columns))


def test_attach_labels_does_not_mutate_input(ohlc_df):
    original = ohlc_df.copy()
    band = BandParams(ma_period=3, atr_period=3, k_atr=1.0)
    attach_labels(ohlc_df, omega=0.05, band=band)
    pd.testing.assert_frame_equal(ohlc_df, original)


def test_attach_labels_golden_output(ohlc_df):
    """Lock the exact output of the full chain on a deterministic fixture."""
    band = BandParams(ma_period=3, atr_period=3, k_atr=1.0)
    out = attach_labels(ohlc_df, omega=0.05, band=band)

    expected_ctl = [0, 1, 1, -1, -1, 1, 1, 1, 1, -1]
    expected_band_state = [0, 0, 0, -1, -1, 0, 0, 0, 0, -1]
    expected_three_class = [0, 0, 0, -1, -1, 0, 0, 0, 0, -1]
    np.testing.assert_array_equal(out["ctl_label"].values, expected_ctl)
    np.testing.assert_array_equal(out["band_state"].values, expected_band_state)
    np.testing.assert_array_equal(out["ctl_label_3class"].values, expected_three_class)


def test_attach_labels_dtype_contract(ohlc_df):
    band = BandParams(ma_period=3, atr_period=3, k_atr=1.0)
    out = attach_labels(ohlc_df, omega=0.05, band=band)
    assert out["ctl_label"].dtype == np.int8
    assert out["band_state"].dtype == np.int8
    assert out["ctl_label_3class"].dtype == np.int8


def test_attach_labels_custom_column_names(ohlc_df):
    band = BandParams(ma_period=3, atr_period=3, k_atr=1.0)
    out = attach_labels(ohlc_df, omega=0.05, band=band,
                        binary_col="my_bin", ternary_col="my_tern")
    assert "my_bin" in out.columns
    assert "my_tern" in out.columns
    assert "ctl_label" not in out.columns
    assert "ctl_label_3class" not in out.columns


# ---------------------------------------------------------------------------
# apply_3class_labels — TF inference and scaling
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlc_df_5min():
    df = pd.DataFrame({
        "high":  [101.0, 111.0, 116.0, 91.0, 86.0, 91.0, 106.0, 111.0, 116.0, 91.0],
        "low":   [99.0,  109.0, 114.0, 89.0, 84.0, 89.0, 104.0, 109.0, 114.0, 89.0],
        "close": [100.0, 110.0, 115.0, 90.0, 85.0, 90.0, 105.0, 110.0, 115.0, 90.0],
    }, index=pd.date_range("2024-01-01", periods=10, freq="5min"))
    return df


def test_apply_3class_with_explicit_tf(ohlc_df_5min):
    """tf=5, persisted=15 -> scale=3; band periods 12*3=36 exceed df length, all warmup."""
    band = BandParams(ma_period=12, atr_period=6, k_atr=1.0)
    out = apply_3class_labels(ohlc_df_5min, omega=0.05, band=band,
                              persisted_tf_minutes=15, tf_minutes=5)
    # CTL is computed independently of band scaling
    np.testing.assert_array_equal(out["ctl_label"].values, [0, 1, 1, -1, -1, 1, 1, 1, 1, -1])
    # Band is all warmup -> all 0
    np.testing.assert_array_equal(out["band_state"].values, np.zeros(10, dtype=np.int8))
    np.testing.assert_array_equal(out["ctl_label_3class"].values, np.zeros(10, dtype=np.int8))


def test_apply_3class_infers_tf_from_datetimeindex(ohlc_df_5min):
    band = BandParams(ma_period=4, atr_period=4, k_atr=1.0)
    out = apply_3class_labels(ohlc_df_5min, omega=0.05, band=band)
    assert "ctl_label_3class" in out.columns


def test_apply_3class_rejects_non_datetime_index():
    df = pd.DataFrame({
        "high":  [101.0] * 5, "low": [99.0] * 5, "close": [100.0] * 5,
    }, index=pd.RangeIndex(5))
    with pytest.raises(ValueError, match="Could not infer tf_minutes"):
        apply_3class_labels(df, omega=0.05, band=BandParams(ma_period=4, atr_period=4, k_atr=1.0))


def test_apply_3class_rejects_non_positive_persisted_tf(ohlc_df_5min):
    band = BandParams(ma_period=4, atr_period=4, k_atr=1.0)
    with pytest.raises(ValueError, match="persisted_tf_minutes must be positive"):
        apply_3class_labels(ohlc_df_5min, omega=0.05, band=band, persisted_tf_minutes=-1)


def test_apply_3class_rejects_non_positive_explicit_tf(ohlc_df_5min):
    band = BandParams(ma_period=4, atr_period=4, k_atr=1.0)
    with pytest.raises(ValueError, match="tf_minutes must be positive"):
        apply_3class_labels(ohlc_df_5min, omega=0.05, band=band,
                            persisted_tf_minutes=15, tf_minutes=0)


def test_apply_3class_band_scaling_respects_minimum_period(ohlc_df_5min):
    """Even if scaled period rounds to <2, attach_labels still gets ma_period >= 2."""
    # tiny persisted band, tf finer -> scaled would go below 2; max(2, ...) clamps it
    band = BandParams(ma_period=2, atr_period=2, k_atr=1.0)
    # persisted_tf=1, tf=5 -> scale=0.2 -> 2*0.2 = 0.4 -> rounds to 0 -> clamped to 2
    out = apply_3class_labels(ohlc_df_5min, omega=0.05, band=band,
                              persisted_tf_minutes=1, tf_minutes=5)
    assert "ctl_label_3class" in out.columns