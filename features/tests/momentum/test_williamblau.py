"""Tests for okmich_quant_features.momentum._williamblau.

Coverage:
  _validate_periods                  — positive-int gate used by every public function
  _validate_aligned                  — index-equality gate for multi-Series functions
  true_strength_index                — Series/ndarray parity (P0 regression), as_percent,
                                       signal=0 skip, fillna ffill, period validation
  slope_divergence_tsi               — 6-tuple shape, method gate, slope_period validation
  stochastic_momentum_index          — happy path, as_percent toggle, alignment, periods
  directional_trend_index            — signal=None branch, signal=int branch, alignment
  directional_efficiency_index       — happy path, as_percent toggle, alignment
  tick_volume_indicator              — bounded output, doji tie-break, all-up/all-down
                                       limits, as_percent parity, alignment

The Series/ndarray parity test in TestTrueStrengthIndex is the regression test
for the P0 bug where the numpy branch returned ``tsi - signal`` (subtracting
the int parameter) instead of ``tsi - sig`` (the signal line).
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.momentum._williamblau import (
    _validate_aligned,
    _validate_periods,
    directional_efficiency_index,
    directional_trend_index,
    slope_divergence_tsi,
    stochastic_momentum_index,
    tick_volume_indicator,
    true_strength_index,
)


# --------------------------------------------------------------------------- #
# Shared fixtures                                                             #
# --------------------------------------------------------------------------- #

def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, open_) + rng.uniform(0, 0.5, n)
    low = np.minimum(close, open_) - rng.uniform(0, 0.5, n)
    volume = rng.lognormal(5, 1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                        index=idx)


# --------------------------------------------------------------------------- #
# Private validation helpers                                                  #
# --------------------------------------------------------------------------- #

class TestValidatePeriods:
    def test_accepts_positive_ints(self):
        _validate_periods(p=1, q=42, r=1_000_000)  # should not raise

    def test_accepts_numpy_integer(self):
        _validate_periods(p=np.int32(5))

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match=r"p must be a positive integer"):
            _validate_periods(p=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match=r"q must be a positive integer"):
            _validate_periods(q=-1)

    def test_rejects_float(self):
        with pytest.raises(ValueError, match=r"r must be a positive integer"):
            _validate_periods(r=3.5)

    def test_rejects_none(self):
        with pytest.raises(ValueError, match=r"s must be a positive integer"):
            _validate_periods(s=None)

    def test_reports_offending_arg_name(self):
        with pytest.raises(ValueError, match=r"smoothing_period"):
            _validate_periods(r=1, s=1, smoothing_period=0)


class TestValidateAligned:
    def test_identical_indices_pass(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        a = pd.Series(range(10), index=idx)
        b = pd.Series(range(10), index=idx)
        _validate_aligned(a=a, b=b)

    def test_single_input_is_noop(self):
        _validate_aligned(only=pd.Series([1, 2, 3]))

    def test_zero_inputs_is_noop(self):
        _validate_aligned()

    def test_mismatched_length_raises(self):
        a = pd.Series(range(10))
        b = pd.Series(range(11))
        with pytest.raises(ValueError, match=r"index mismatch"):
            _validate_aligned(a=a, b=b)

    def test_same_length_different_dates_raises(self):
        a = pd.Series(range(5), index=pd.date_range("2024-01-01", periods=5))
        b = pd.Series(range(5), index=pd.date_range("2024-02-01", periods=5))
        with pytest.raises(ValueError, match=r"index mismatch"):
            _validate_aligned(a=a, b=b)

    def test_error_message_names_first_mismatching_pair(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        a = pd.Series(range(5), index=idx)
        b = pd.Series(range(5), index=idx)
        bad = pd.Series(range(5), index=pd.date_range("2024-02-01", periods=5))
        with pytest.raises(ValueError, match=r"'a' and 'bad'"):
            _validate_aligned(a=a, b=b, bad=bad)


# --------------------------------------------------------------------------- #
# True Strength Index                                                          #
# --------------------------------------------------------------------------- #

class TestTrueStrengthIndex:
    def test_returns_three_series_for_series_input(self):
        df = _make_ohlcv(300)
        tsi, sig, diff = true_strength_index(df["close"])
        assert isinstance(tsi, pd.Series)
        assert isinstance(sig, pd.Series)
        assert isinstance(diff, pd.Series)
        assert len(tsi) == len(df) == len(sig) == len(diff)

    def test_returns_three_ndarrays_for_array_input(self):
        df = _make_ohlcv(300)
        arr = df["close"].to_numpy()
        tsi, sig, diff = true_strength_index(arr)
        assert isinstance(tsi, np.ndarray)
        assert isinstance(sig, np.ndarray)
        assert isinstance(diff, np.ndarray)
        assert tsi.shape == sig.shape == diff.shape == arr.shape

    def test_series_ndarray_parity_p0_regression(self):
        """Regression: numpy branch used to return ``tsi - signal_int`` instead of
        ``tsi - sig`` (the signal line). All three outputs must match across
        Series and ndarray entry points."""
        df = _make_ohlcv(300)
        s_tsi, s_sig, s_diff = true_strength_index(df["close"], as_percent=True)
        a_tsi, a_sig, a_diff = true_strength_index(df["close"].to_numpy(), as_percent=True)
        np.testing.assert_allclose(s_tsi.to_numpy(), a_tsi, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(s_sig.to_numpy(), a_sig, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(s_diff.to_numpy(), a_diff, rtol=1e-12, equal_nan=True)

    def test_as_percent_toggle_is_pure_100x(self):
        df = _make_ohlcv(300)
        ratio, _, _ = true_strength_index(df["close"], as_percent=False)
        pct, _, _ = true_strength_index(df["close"], as_percent=True)
        np.testing.assert_allclose(pct.dropna().to_numpy(),
                                    100.0 * ratio.dropna().to_numpy(),
                                    rtol=1e-12)

    def test_signal_zero_returns_none_signal_and_diff(self):
        """signal=0 is the backward-compatible sentinel for 'skip signal line'."""
        df = _make_ohlcv(100)
        tsi, sig, diff = true_strength_index(df["close"].to_numpy(), signal=0)
        assert sig is None
        assert diff is None
        assert isinstance(tsi, np.ndarray)

    def test_fillna_ffill_no_deprecation_warning(self):
        """Regression: previously used pandas-deprecated fillna(method='ffill')."""
        df = _make_ohlcv(100)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            true_strength_index(df["close"].to_numpy(), fillna=True)

    def test_rejects_zero_r(self):
        with pytest.raises(ValueError, match=r"r must be a positive integer"):
            true_strength_index(np.arange(100, dtype=float), r=0)

    def test_rejects_negative_s(self):
        with pytest.raises(ValueError, match=r"s must be a positive integer"):
            true_strength_index(np.arange(100, dtype=float), s=-1)

    def test_rejects_non_1d(self):
        with pytest.raises(ValueError, match="1D"):
            true_strength_index(np.zeros((10, 2)))


# --------------------------------------------------------------------------- #
# Slope Divergence TSI                                                         #
# --------------------------------------------------------------------------- #

class TestSlopeDivergenceTsi:
    def test_returns_six_tuple_of_series(self):
        df = _make_ohlcv(200)
        result = slope_divergence_tsi(df["close"])
        assert len(result) == 6
        for x in result:
            assert isinstance(x, pd.Series)

    def test_method_diff(self):
        df = _make_ohlcv(200)
        _, _, _, slope, _, _ = slope_divergence_tsi(df["close"], method="diff")
        assert slope.notna().sum() > 0

    def test_method_ols(self):
        df = _make_ohlcv(200)
        _, _, _, slope, _, _ = slope_divergence_tsi(df["close"], method="ols", slope_period=5)
        assert slope.notna().sum() > 0

    def test_invalid_method_raises(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match=r"method must be"):
            slope_divergence_tsi(df["close"], method="invalid")

    def test_rejects_zero_slope_period(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match=r"slope_period must be a positive integer"):
            slope_divergence_tsi(df["close"], slope_period=0)

    def test_divergence_flags_are_0_or_1(self):
        df = _make_ohlcv(200)
        _, _, _, _, bull, bear = slope_divergence_tsi(df["close"])
        assert set(bull.unique()).issubset({0, 1})
        assert set(bear.unique()).issubset({0, 1})


# --------------------------------------------------------------------------- #
# Stochastic Momentum Index                                                    #
# --------------------------------------------------------------------------- #

class TestStochasticMomentumIndex:
    def test_returns_three_series(self):
        df = _make_ohlcv(200)
        smi, sig, diff = stochastic_momentum_index(df["high"], df["low"], df["close"])
        assert all(isinstance(x, pd.Series) for x in (smi, sig, diff))
        assert len(smi) == len(df)

    def test_as_percent_toggle_is_pure_100x(self):
        df = _make_ohlcv(200)
        ratio, _, _ = stochastic_momentum_index(df["high"], df["low"], df["close"], as_percent=False)
        pct, _, _ = stochastic_momentum_index(df["high"], df["low"], df["close"], as_percent=True)
        np.testing.assert_allclose(pct.dropna().to_numpy(),
                                    100.0 * ratio.dropna().to_numpy(),
                                    rtol=1e-12)

    def test_index_mismatch_raises(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match=r"index mismatch"):
            stochastic_momentum_index(df["high"].iloc[:-1], df["low"], df["close"])

    def test_zero_k_period_raises(self):
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match=r"k_period must be a positive integer"):
            stochastic_momentum_index(df["high"], df["low"], df["close"], k_period=0)


# --------------------------------------------------------------------------- #
# Directional Trend Index                                                      #
# --------------------------------------------------------------------------- #

class TestDirectionalTrendIndex:
    def test_signal_none_returns_none_for_signal_and_diff(self):
        df = _make_ohlcv(200)
        dti, sig, diff = directional_trend_index(df["high"], df["low"], signal=None)
        assert isinstance(dti, pd.Series)
        assert sig is None
        assert diff is None

    def test_signal_int_returns_series_for_signal_and_diff(self):
        df = _make_ohlcv(200)
        dti, sig, diff = directional_trend_index(df["high"], df["low"], signal=9)
        assert isinstance(sig, pd.Series)
        assert isinstance(diff, pd.Series)

    def test_index_mismatch_raises(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match=r"index mismatch"):
            directional_trend_index(df["high"].iloc[:-1], df["low"])

    def test_zero_r_raises(self):
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match=r"r must be a positive integer"):
            directional_trend_index(df["high"], df["low"], r=0)

    def test_zero_signal_raises_when_provided(self):
        """signal=None is the skip sentinel; signal=0 should be rejected since
        it would be ambiguous with the int-skip convention used in TSI."""
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match=r"signal must be a positive integer"):
            directional_trend_index(df["high"], df["low"], signal=0)


# --------------------------------------------------------------------------- #
# Directional Efficiency Index                                                 #
# --------------------------------------------------------------------------- #

class TestDirectionalEfficiencyIndex:
    def test_returns_three_series(self):
        df = _make_ohlcv(200)
        dei, sig, diff = directional_efficiency_index(df["high"], df["low"], df["close"])
        assert all(isinstance(x, pd.Series) for x in (dei, sig, diff))

    def test_as_percent_toggle_is_pure_100x(self):
        df = _make_ohlcv(200)
        ratio, _, _ = directional_efficiency_index(df["high"], df["low"], df["close"], as_percent=False)
        pct, _, _ = directional_efficiency_index(df["high"], df["low"], df["close"], as_percent=True)
        np.testing.assert_allclose(pct.dropna().to_numpy(),
                                    100.0 * ratio.dropna().to_numpy(),
                                    rtol=1e-12)

    def test_index_mismatch_raises(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match=r"index mismatch"):
            directional_efficiency_index(df["high"], df["low"], df["close"].iloc[:-1])

    def test_zero_signal_raises(self):
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match=r"signal must be a positive integer"):
            directional_efficiency_index(df["high"], df["low"], df["close"], signal=0)


# --------------------------------------------------------------------------- #
# Tick Volume Indicator                                                        #
# --------------------------------------------------------------------------- #

class TestTickVolumeIndicator:
    def test_returns_three_series(self):
        df = _make_ohlcv(500)
        tvi, sig, diff = tick_volume_indicator(df["open"], df["close"], df["volume"])
        assert all(isinstance(x, pd.Series) for x in (tvi, sig, diff))
        assert len(tvi) == len(df)

    def test_bounded_in_unit_range_by_default(self):
        df = _make_ohlcv(500)
        tvi, _, _ = tick_volume_indicator(df["open"], df["close"], df["volume"])
        finite = tvi.dropna()
        assert finite.min() >= -1.0 - 1e-12
        assert finite.max() <= 1.0 + 1e-12

    def test_as_percent_toggle_is_pure_100x(self):
        df = _make_ohlcv(500)
        ratio, _, _ = tick_volume_indicator(df["open"], df["close"], df["volume"], as_percent=False)
        pct, _, _ = tick_volume_indicator(df["open"], df["close"], df["volume"], as_percent=True)
        np.testing.assert_allclose(pct.to_numpy(), 100.0 * ratio.to_numpy(), rtol=1e-12)

    def test_all_up_bars_converge_to_plus_one(self):
        """All green bars (close > open) → all volume in up stream → TVI = +1."""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        open_ = pd.Series(np.full(n, 100.0), index=idx)
        close = pd.Series(np.full(n, 101.0), index=idx)
        volume = pd.Series(np.full(n, 1000.0), index=idx)
        tvi, _, _ = tick_volume_indicator(open_, close, volume)
        assert tvi.iloc[-50:].mean() == pytest.approx(1.0, abs=1e-9)

    def test_all_doji_bars_converge_to_minus_one(self):
        """Doji tie-break: close == open is bucketed DOWN. All-doji → TVI = -1."""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        open_ = pd.Series(np.full(n, 100.0), index=idx)
        close = pd.Series(np.full(n, 100.0), index=idx)
        volume = pd.Series(np.full(n, 1000.0), index=idx)
        tvi, _, _ = tick_volume_indicator(open_, close, volume)
        assert tvi.iloc[-50:].mean() == pytest.approx(-1.0, abs=1e-9)

    def test_all_down_bars_converge_to_minus_one(self):
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="5min")
        open_ = pd.Series(np.full(n, 101.0), index=idx)
        close = pd.Series(np.full(n, 100.0), index=idx)
        volume = pd.Series(np.full(n, 1000.0), index=idx)
        tvi, _, _ = tick_volume_indicator(open_, close, volume)
        assert tvi.iloc[-50:].mean() == pytest.approx(-1.0, abs=1e-9)

    def test_index_mismatch_raises(self):
        df = _make_ohlcv(100)
        with pytest.raises(ValueError, match=r"index mismatch"):
            tick_volume_indicator(df["open"].iloc[:-1], df["close"], df["volume"])

    def test_zero_period_raises(self):
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match=r"period must be a positive integer"):
            tick_volume_indicator(df["open"], df["close"], df["volume"], period=0)

    def test_zero_signal_raises(self):
        df = _make_ohlcv(50)
        with pytest.raises(ValueError, match=r"signal must be a positive integer"):
            tick_volume_indicator(df["open"], df["close"], df["volume"], signal=0)
