"""Characterization tests for ctl_trend_features — the causal feature bundle projected from the CTL FSM.

Locks the per-bar output (columns, dtypes, warmup NaN contract, golden values) so the projection can be
refactored safely. ctl_direction is asserted bar-for-bar identical to continuous_trend_labeling; the derived
columns are checked against hand-traced golden sequences.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import (CTLFeatures, CTLState, continuous_trend_labeling,
                                          ctl_trend_features)


EXPECTED_COLS = ["ctl_direction", "ctl_trend_age", "ctl_retrace_frac", "ctl_leg_return", "ctl_flip_count"]


# ---------------------------------------------------------------------------
# Shape / dtype / index contract
# ---------------------------------------------------------------------------

def test_returns_dataframe_with_expected_columns():
    out = ctl_trend_features(pd.Series([100.0, 110.0, 115.0, 90.0, 85.0]), omega=0.05)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == EXPECTED_COLS


def test_all_columns_float64():
    out = ctl_trend_features(pd.Series([100.0, 110.0, 115.0, 90.0, 85.0]), omega=0.05)
    for col in EXPECTED_COLS:
        assert out[col].dtype == np.float64, col


def test_series_index_preserved():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    out = ctl_trend_features(pd.Series([100.0, 110.0, 115.0, 90.0, 85.0], index=idx), omega=0.05)
    pd.testing.assert_index_equal(out.index, idx)


def test_ndarray_input_gets_rangeindex():
    out = ctl_trend_features(np.array([100.0, 110.0, 115.0, 90.0, 85.0]), omega=0.05)
    assert isinstance(out.index, pd.RangeIndex)
    assert len(out) == 5


def test_empty_input_returns_empty_frame():
    out = ctl_trend_features(pd.Series([], dtype=float), omega=0.05)
    assert list(out.columns) == EXPECTED_COLS
    assert len(out) == 0


# ---------------------------------------------------------------------------
# ctl_direction == continuous_trend_labeling (single source of truth for sign)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prices", [
    [100.0, 110.0, 115.0, 90.0, 85.0],
    [100.0, 95.0, 90.0, 100.0, 110.0, 105.0, 92.0, 95.0, 108.0],
    [100.0, 100.5, 100.2, 100.8, 100.3, 100.6],  # never triggers -> all NaN
])
def test_direction_matches_batch_labeler(prices):
    s = pd.Series(prices)
    out = ctl_trend_features(s, omega=0.05)
    batch = continuous_trend_labeling(s, omega=0.05)
    # assert_array_equal treats same-position NaNs as equal
    np.testing.assert_array_equal(out["ctl_direction"].to_numpy(), batch.to_numpy())


# ---------------------------------------------------------------------------
# Warmup contract
# ---------------------------------------------------------------------------

def test_pre_trigger_is_nan_except_flip_count():
    # triggers up only at bar 2
    out = ctl_trend_features(pd.Series([100.0, 100.5, 110.0]), omega=0.05)
    warm = out.iloc[:2]
    for col in ["ctl_direction", "ctl_trend_age", "ctl_retrace_frac", "ctl_leg_return"]:
        assert warm[col].isna().all(), col
    # flip_count is a trailing count -> meaningful 0 during warmup, never NaN
    assert (out["ctl_flip_count"].iloc[:2] == 0).all()
    assert out["ctl_flip_count"].notna().all()


def test_never_triggers_all_nan_direction_zero_flips():
    out = ctl_trend_features(pd.Series([100.0, 100.5, 100.2, 100.8, 100.3]), omega=0.05)
    assert out["ctl_direction"].isna().all()
    assert out["ctl_trend_age"].isna().all()
    assert (out["ctl_flip_count"] == 0).all()


# ---------------------------------------------------------------------------
# Golden values on a hand-traced sequence
# ---------------------------------------------------------------------------

def test_golden_bundle_up_then_reversal():
    """Prices 100->110->115->90->85, omega=0.05 (hand-traced against the FSM)."""
    out = ctl_trend_features(pd.Series([100.0, 110.0, 115.0, 90.0, 85.0]), omega=0.05)

    np.testing.assert_array_equal(out["ctl_direction"].to_numpy(), [np.nan, 1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(out["ctl_trend_age"].to_numpy(), [np.nan, 0.0, 1.0, 0.0, 1.0])
    # price hugs each leg's running extreme every bar here -> retrace is 0 on all confirmed bars
    np.testing.assert_array_equal(out["ctl_retrace_frac"].to_numpy(), [np.nan, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(out["ctl_leg_return"].to_numpy(),
                               [np.nan, 0.10, 0.15, -25.0 / 115.0, -30.0 / 115.0], rtol=1e-12)
    # one reversal (up -> down) at bar 3; the initial 0 -> +1 trigger is not a flip
    np.testing.assert_array_equal(out["ctl_flip_count"].to_numpy(), [0.0, 0.0, 0.0, 1.0, 1.0])


def test_partial_pullback_gives_fractional_retrace():
    """A pullback that stays under omega does not flip; ctl_retrace_frac lands in (0, 1)."""
    # bar1: 115 triggers up (>100*1.10); bar2: 110 is a 5/115 pullback from the high, < omega=0.10
    out = ctl_trend_features(pd.Series([100.0, 115.0, 110.0]), omega=0.10)
    assert out["ctl_direction"].iloc[2] == 1.0            # no flip
    expected = (115.0 - 110.0) / 115.0 / 0.10             # ~0.4348
    assert 0.0 < out["ctl_retrace_frac"].iloc[2] < 1.0
    np.testing.assert_allclose(out["ctl_retrace_frac"].iloc[2], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# flip_count semantics
# ---------------------------------------------------------------------------

def test_flip_count_counts_reversals_not_initial_trigger():
    # direction: [nan, nan, -1, 1, 1, 1, -1, -1, 1] -> reversals at bars 3, 6, 8 (trigger at bar 2 excluded)
    out = ctl_trend_features(
        pd.Series([100.0, 95.0, 90.0, 100.0, 110.0, 105.0, 92.0, 95.0, 108.0]), omega=0.05)
    fc = out["ctl_flip_count"].to_numpy()
    assert (fc[:3] == 0).all()          # trigger is not a flip
    assert fc[-1] == 3.0                # total reversals over a window wider than the series
    assert (np.diff(fc) >= 0).all()     # cumulative within one window -> non-decreasing


def test_flip_count_respects_window():
    """A short trailing window drops old flips out of the count."""
    prices = pd.Series([100.0, 95.0, 90.0, 100.0, 110.0, 105.0, 92.0, 95.0, 108.0])
    wide = ctl_trend_features(prices, omega=0.05, flip_window=20)["ctl_flip_count"]
    narrow = ctl_trend_features(prices, omega=0.05, flip_window=2)["ctl_flip_count"]
    assert wide.iloc[-1] == 3.0
    assert narrow.iloc[-1] <= wide.iloc[-1]  # window shorter than the flip spacing -> fewer counted


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_flip_window_below_one_raises():
    with pytest.raises(ValueError, match="flip_window must be >= 1"):
        ctl_trend_features(pd.Series([100.0, 110.0]), omega=0.05, flip_window=0)


def test_omega_non_positive_raises():
    with pytest.raises(ValueError, match="omega must be > 0"):
        ctl_trend_features(pd.Series([100.0, 110.0]), omega=0.0)


def test_non_finite_price_raises():
    with pytest.raises(ValueError, match="NaN or infinite"):
        ctl_trend_features(pd.Series([100.0, np.nan, 110.0]), omega=0.05)


# ---------------------------------------------------------------------------
# Streaming twin — CTLState.step_features must equal the batch row for the same bar
# ---------------------------------------------------------------------------

def test_ctl_features_fields_match_batch_columns():
    """The NamedTuple fields must line up with the DataFrame columns, or live/backtest vectors won't align."""
    cols = list(ctl_trend_features(pd.Series([100.0, 110.0, 90.0]), omega=0.05).columns)
    assert list(CTLFeatures._fields) == cols


@pytest.mark.parametrize("flip_window", [1, 2, 3, 20])
@pytest.mark.parametrize("prices", [
    [100.0, 110.0, 115.0, 90.0, 85.0],
    [100.0, 95.0, 90.0, 100.0, 110.0, 105.0, 92.0, 95.0, 108.0],
    [100.0, 100.5, 100.2, 100.8, 100.3],                          # never triggers
    [100.0, 103.0, 101.0, 106.0, 104.0, 99.0, 97.0, 101.0, 108.0, 96.0, 110.0],
])
def test_step_features_matches_batch(prices, flip_window):
    """Per-bar streaming replay is identical to the batch DataFrame, column-for-column, bar-for-bar."""
    s = pd.Series(prices)
    batch = ctl_trend_features(s, omega=0.05, flip_window=flip_window)

    state = CTLState(omega=0.05, flip_window=flip_window)
    rows = [state.step_features(float(p)) for p in prices]
    streamed = pd.DataFrame(rows, index=s.index)  # pandas names columns from the NamedTuple fields

    assert list(streamed.columns) == list(batch.columns)
    pd.testing.assert_frame_equal(streamed, batch, check_dtype=False)


def test_step_features_holds_on_non_finite_tick():
    """A bad tick returns the last bundle unchanged and does not advance the machine (matches ctl_step)."""
    state = CTLState(omega=0.05, flip_window=5)
    for p in [100.0, 110.0, 115.0]:
        good = state.step_features(p)

    held = state.step_features(float("nan"))
    assert held == good  # last vector repeated verbatim

    # the bad tick was skipped, so the next real bar continues exactly where the last good bar left off
    twin = CTLState(omega=0.05, flip_window=5)
    for p in [100.0, 110.0, 115.0]:
        twin.step_features(p)
    assert state.step_features(112.0) == twin.step_features(112.0)


def test_step_features_warmup_returns_nan_bundle_then_confirms():
    state = CTLState(omega=0.05, flip_window=5)
    warm = state.step_features(100.0)  # first bar: pre-trigger
    assert np.isnan(warm.ctl_direction) and warm.ctl_flip_count == 0.0
    conf = state.step_features(110.0)  # triggers up
    assert conf.ctl_direction == 1.0 and conf.ctl_trend_age == 0.0


def test_ctlstate_rejects_flip_window_below_one():
    with pytest.raises(ValueError, match="flip_window must be >= 1"):
        CTLState(omega=0.05, flip_window=0)


def test_ctlstate_step_features_bookkeeping_excluded_from_repr():
    """Feature bookkeeping fields must not leak into repr/eq — keeps the FSM state clean to inspect."""
    state = CTLState(omega=0.05)
    r = repr(state)
    assert "_flip_indices" not in r and "_leg_start_idx" not in r
