"""Tests for okmich_quant_features.momentum.david_varadi.

Coverage:
  _resolve_column      — column lookup, case folding, ambiguity, missing
  _pct_rank_1d         — sliding percent rank helper used by dvo
  _percent_rank_hlc    — HLC pool percent rank helper used by aggregate_m
  dvo                  — full pipeline including detrend branch
  dv2                  — thin wrapper over dvo
  aggregate_m_components — all four output columns, EMA fast/slow paths
  aggregate_m          — backward-compat wrapper
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.momentum.david_varadi import (
    _percent_rank_hlc,
    _pct_rank_1d,
    _resolve_column,
    aggregate_m,
    aggregate_m_components,
    dv2,
    dvo,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close}, index=idx)


def _reference_pct_rank_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Original loop-based implementation used as ground-truth reference."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        current = arr[i]
        prior = arr[i - window + 1 : i]
        valid = prior[~np.isnan(prior)]
        if len(valid) == 0 or np.isnan(current):
            continue
        result[i] = float(np.sum(valid < current) / len(valid) * 100.0)
    return result


def _reference_percent_rank_hlc(close_s, high_s, low_s, period: int) -> np.ndarray:
    """Original loop-based implementation used as ground-truth reference."""
    close_arr = close_s.to_numpy(dtype=float)
    high_arr = high_s.to_numpy(dtype=float)
    low_arr = low_s.to_numpy(dtype=float)
    n = len(close_arr)
    ranks = np.full(n, np.nan)
    hlc = np.column_stack([high_arr, low_arr, close_arr])
    for i in range(period - 1, n):
        current = close_arr[i]
        if np.isnan(current):
            continue
        window = hlc[i - period + 1 : i + 1]
        pool = window.ravel()[:-1]
        valid = pool[~np.isnan(pool)]
        if len(valid) == 0:
            continue
        ranks[i] = 100.0 * np.sum(valid < current) / len(valid)
    return ranks


def _reference_ema(raw_m: np.ndarray, alpha: float, seed_idx: int) -> np.ndarray:
    """Original gap-tolerant EMA loop as ground-truth reference."""
    beta = 1.0 - alpha
    n = len(raw_m)
    result = np.full(n, np.nan)
    last_valid = raw_m[seed_idx]
    result[seed_idx] = last_valid
    for i in range(seed_idx + 1, n):
        if np.isnan(raw_m[i]):
            continue
        last_valid = beta * last_valid + alpha * raw_m[i]
        result[i] = last_valid
    return result


@pytest.fixture
def ohlcv():
    return _make_ohlcv()


@pytest.fixture
def small_ohlcv():
    return _make_ohlcv(n=50)


# ===========================================================================
# _resolve_column
# ===========================================================================

class TestResolveColumn:
    def test_exact_match(self):
        df = pd.DataFrame({"close": [1.0], "High": [2.0]})
        assert _resolve_column(df, "close") == "close"
        assert _resolve_column(df, "High") == "High"

    def test_case_insensitive_match(self):
        df = pd.DataFrame({"Close": [1.0]})
        assert _resolve_column(df, "close") == "Close"
        assert _resolve_column(df, "CLOSE") == "Close"

    def test_exact_match_wins_over_case_fold(self):
        df = pd.DataFrame({"close": [1.0]})
        assert _resolve_column(df, "close") == "close"

    def test_missing_column_raises(self):
        df = pd.DataFrame({"open": [1.0]})
        with pytest.raises(ValueError, match="must contain"):
            _resolve_column(df, "close")

    def test_ambiguous_columns_raises(self):
        # "CLOSE" doesn't exactly match either "close" or "Close", so the
        # case-fold path finds both and must raise.
        df = pd.DataFrame([[1.0, 2.0]], columns=["close", "Close"])
        with pytest.raises(ValueError, match="Ambiguous"):
            _resolve_column(df, "CLOSE")

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="must contain"):
            _resolve_column(df, "close")


# ===========================================================================
# _pct_rank_1d
# ===========================================================================

class TestPctRank1d:
    def test_hand_computed_values(self):
        # arr=[10,20,15,25,5], window=3
        # t=2: prior=[10,20], 1 < 15 => rank = 1/2*100 = 50.0
        # t=3: prior=[20,15], 2 < 25 => rank = 2/2*100 = 100.0
        # t=4: prior=[15,25], 0 < 5  => rank = 0/2*100 = 0.0
        arr = np.array([10.0, 20.0, 15.0, 25.0, 5.0])
        result = _pct_rank_1d(arr, window=3)
        assert np.isnan(result[0]) and np.isnan(result[1])
        assert result[2] == pytest.approx(50.0)
        assert result[3] == pytest.approx(100.0)
        assert result[4] == pytest.approx(0.0)

    def test_matches_reference_loop_clean(self):
        arr = RNG.standard_normal(500)
        for window in (2, 10, 50):
            result = _pct_rank_1d(arr, window)
            expected = _reference_pct_rank_1d(arr, window)
            np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)

    def test_matches_reference_loop_with_nan(self):
        arr = RNG.standard_normal(200).astype(float)
        arr[[10, 50, 100]] = np.nan
        result = _pct_rank_1d(arr, window=20)
        expected = _reference_pct_rank_1d(arr, window=20)
        np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)

    def test_warmup_is_nan(self):
        arr = np.arange(10, dtype=float)
        result = _pct_rank_1d(arr, window=5)
        assert np.all(np.isnan(result[:4]))
        assert not np.isnan(result[4])

    def test_n_less_than_window_all_nan(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _pct_rank_1d(arr, window=10)
        assert np.all(np.isnan(result))
        assert len(result) == 3

    def test_output_range(self):
        arr = RNG.standard_normal(300)
        result = _pct_rank_1d(arr, window=30)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_constant_array_gives_zero(self):
        arr = np.full(50, 5.0)
        result = _pct_rank_1d(arr, window=10)
        valid = result[~np.isnan(result)]
        # All prior values equal current: 0 are strictly less-than
        np.testing.assert_array_equal(valid, 0.0)

    def test_strictly_increasing_gives_100(self):
        arr = np.arange(50, dtype=float)
        result = _pct_rank_1d(arr, window=5)
        valid = result[~np.isnan(result)]
        np.testing.assert_array_equal(valid, 100.0)

    def test_window_2_is_binary(self):
        arr = np.array([1.0, 3.0, 2.0, 4.0])
        result = _pct_rank_1d(arr, window=2)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(100.0)  # 3 > 1
        assert result[2] == pytest.approx(0.0)    # 2 > 3 is False → rank 0
        assert result[3] == pytest.approx(100.0)  # 4 > 2

    def test_index_length_preserved(self):
        arr = np.arange(100, dtype=float)
        assert len(_pct_rank_1d(arr, window=10)) == 100


# ===========================================================================
# _percent_rank_hlc
# ===========================================================================

class TestPercentRankHlc:
    def _series(self, arr):
        return pd.Series(arr, dtype=float)

    def test_hand_computed_values(self):
        # period=2
        # bar0: H=3, L=1, C=2  → warm-up NaN
        # bar1: H=5, L=3, C=4  pool=[H0=3,L0=1,C0=2,H1=5,L1=3]
        #       count < 4: 3,1,2,3 → 4 out of 5 → 80.0
        # bar2: H=4, L=2, C=3  pool=[H1=5,L1=3,C1=4,H2=4,L2=2]
        #       count < 3: only 2 → 1 out of 5 → 20.0
        h = self._series([3.0, 5.0, 4.0])
        l = self._series([1.0, 3.0, 2.0])
        c = self._series([2.0, 4.0, 3.0])
        result = _percent_rank_hlc(c, h, l, period=2)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(80.0)
        assert result[2] == pytest.approx(20.0)

    def test_matches_reference_loop_clean(self, ohlcv):
        for period in (2, 5, 20, 78):
            result = _percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], period)
            expected = _reference_percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], period)
            np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)

    def test_matches_reference_loop_with_nan_in_high(self):
        df = _make_ohlcv(n=100)
        df.loc[df.index[10], "high"] = np.nan
        df.loc[df.index[50], "high"] = np.nan
        result = _percent_rank_hlc(df["close"], df["high"], df["low"], period=5)
        expected = _reference_percent_rank_hlc(df["close"], df["high"], df["low"], period=5)
        np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)

    def test_matches_reference_loop_with_nan_in_close(self):
        df = _make_ohlcv(n=100)
        df.loc[df.index[30], "close"] = np.nan
        result = _percent_rank_hlc(df["close"], df["high"], df["low"], period=5)
        expected = _reference_percent_rank_hlc(df["close"], df["high"], df["low"], period=5)
        np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)

    def test_warmup_is_nan(self, ohlcv):
        period = 10
        result = _percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], period)
        assert np.all(np.isnan(result[: period - 1]))
        assert not np.isnan(result[period - 1])

    def test_n_less_than_period_all_nan(self):
        h = pd.Series([1.0, 2.0])
        l = pd.Series([0.5, 1.5])
        c = pd.Series([0.8, 1.8])
        result = _percent_rank_hlc(c, h, l, period=5)
        assert np.all(np.isnan(result))

    def test_output_range(self, ohlcv):
        result = _percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], period=20)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_period_1_close_vs_hl_of_same_bar(self):
        # period=1: pool = [H[0], L[0]] (close[0] excluded as current)
        # pool_size = 3*1 - 1 = 2
        # If close is between L and H: rank = 1/2 * 100 = 50
        h = pd.Series([10.0, 10.0])
        l = pd.Series([0.0, 0.0])
        c = pd.Series([5.0, 5.0])
        result = _percent_rank_hlc(c, h, l, period=1)
        # bar0: pool=[H=10, L=0], current=5 → 1 < 5 (L=0) → rank = 1/2 * 100 = 50
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(50.0)

    def test_invalid_period_raises(self):
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="positive integer"):
            _percent_rank_hlc(s, s, s, period=0)
        with pytest.raises(ValueError, match="positive integer"):
            _percent_rank_hlc(s, s, s, period=-1)

    def test_returns_ndarray(self, ohlcv):
        result = _percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], period=5)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_length_matches_input(self, ohlcv):
        result = _percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], period=10)
        assert len(result) == len(ohlcv)


# ===========================================================================
# dvo
# ===========================================================================

class TestDvo:
    def test_returns_series_with_correct_name(self, ohlcv):
        result = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=2, pct_lookback=30, detrend=False)
        assert isinstance(result, pd.Series)
        assert result.name == "DVO_2_30"

    def test_index_preserved(self, ohlcv):
        result = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=2, pct_lookback=30, detrend=False)
        pd.testing.assert_index_equal(result.index, ohlcv.index)

    def test_output_range(self, ohlcv):
        result = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=2, pct_lookback=30, detrend=False)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_warmup_is_nan(self, ohlcv):
        n_avg, pct_lookback = 2, 20
        result = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=n_avg, pct_lookback=pct_lookback, detrend=False)
        # _pct_rank_1d produces its first output at index pct_lookback-1, using the
        # NaN-aware path when avg_ratio[0..n_avg-2] are NaN. Warmup = pct_lookback-1.
        warmup = pct_lookback - 1
        assert result.iloc[:warmup].isna().all()
        assert not pd.isna(result.iloc[warmup])

    def test_matches_reference_via_pct_rank_1d(self, ohlcv):
        # Reconstruct the avg_ratio pipeline and verify _pct_rank_1d gives identical output.
        n_avg, pct_lookback = 2, 30
        hl2 = (ohlcv["high"] + ohlcv["low"]) / 2.0
        ratio = ohlcv["close"] / hl2.where(hl2 > 0)
        avg_ratio = ratio.rolling(window=n_avg, min_periods=n_avg).mean()
        arr = avg_ratio.to_numpy(dtype=float)
        expected_ranks = _reference_pct_rank_1d(arr, pct_lookback)
        result = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=n_avg, pct_lookback=pct_lookback, detrend=False)
        np.testing.assert_allclose(result.to_numpy(), expected_ranks, rtol=1e-12, equal_nan=True)

    def test_detrend_changes_output(self, ohlcv):
        without = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], pct_lookback=30, detrend=False)
        with_ = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], pct_lookback=30, detrend=True, n_dt=30)
        # Detrending changes values; they should not be identical
        valid = ~(without.isna() | with_.isna())
        assert not np.allclose(without[valid].to_numpy(), with_[valid].to_numpy())

    @pytest.mark.parametrize("n_avg,pct_lookback", [(1, 10), (2, 20), (5, 50)])
    def test_parametric_name(self, ohlcv, n_avg, pct_lookback):
        result = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=n_avg, pct_lookback=pct_lookback, detrend=False)
        assert result.name == f"DVO_{n_avg}_{pct_lookback}"

    def test_invalid_n_avg_raises(self, ohlcv):
        with pytest.raises(ValueError, match="n_avg"):
            dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=0)

    def test_invalid_pct_lookback_raises(self, ohlcv):
        with pytest.raises(ValueError, match="pct_lookback"):
            dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], pct_lookback=1)

    def test_invalid_n_dt_raises_when_detrend(self, ohlcv):
        with pytest.raises(ValueError, match="n_dt"):
            dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], detrend=True, n_dt=0)

    def test_no_input_mutation(self, ohlcv):
        original = ohlcv.copy()
        dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        pd.testing.assert_frame_equal(ohlcv, original)


# ===========================================================================
# dv2
# ===========================================================================

class TestDv2:
    def test_matches_dvo_no_detrend_n_avg_2(self, ohlcv):
        expected = dvo(ohlcv["high"], ohlcv["low"], ohlcv["close"], n_avg=2, pct_lookback=50, detrend=False)
        result = dv2(ohlcv["high"], ohlcv["low"], ohlcv["close"], pct_lookback=50)
        pd.testing.assert_series_equal(result, expected)

    def test_returns_series(self, ohlcv):
        result = dv2(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        assert isinstance(result, pd.Series)


# ===========================================================================
# aggregate_m_components
# ===========================================================================

class TestAggregateMComponents:
    def test_returns_dataframe_with_correct_columns(self, ohlcv):
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"slow_rank", "fast_rank", "raw_m", "agg_m"}

    def test_index_preserved(self, ohlcv):
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        pd.testing.assert_index_equal(result.index, ohlcv.index)

    def test_length_matches_input(self, ohlcv):
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        assert len(result) == len(ohlcv)

    def test_slow_rank_matches_reference(self, ohlcv):
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        expected = _reference_percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], 50)
        np.testing.assert_allclose(result["slow_rank"].to_numpy(), expected, rtol=1e-12, equal_nan=True)

    def test_fast_rank_matches_reference(self, ohlcv):
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        expected = _reference_percent_rank_hlc(ohlcv["close"], ohlcv["high"], ohlcv["low"], 5)
        np.testing.assert_allclose(result["fast_rank"].to_numpy(), expected, rtol=1e-12, equal_nan=True)

    def test_raw_m_is_weighted_blend(self, ohlcv):
        trend_weight = 60
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5, trend_weight=trend_weight)
        expected_raw = (result["slow_rank"] * trend_weight + result["fast_rank"] * (100 - trend_weight)) / 100.0
        np.testing.assert_allclose(result["raw_m"].to_numpy(), expected_raw.to_numpy(), rtol=1e-12, equal_nan=True)

    def test_agg_m_matches_reference_ema(self, ohlcv):
        alpha = 0.70
        current_bar_weight = int(alpha * 100)
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5, current_bar_weight=current_bar_weight)
        raw_m = result["raw_m"].to_numpy()
        seed_idx = 50 - 1  # max(slow_period, fast_period) - 1
        while seed_idx < len(raw_m) and np.isnan(raw_m[seed_idx]):
            seed_idx += 1
        expected_agg = _reference_ema(raw_m, alpha=alpha, seed_idx=seed_idx)
        np.testing.assert_allclose(result["agg_m"].to_numpy(), expected_agg, rtol=1e-10, equal_nan=True)

    def test_warmup_nans_correct(self, ohlcv):
        slow, fast = 50, 5
        result = aggregate_m_components(ohlcv, slow_period=slow, fast_period=fast)
        assert result["slow_rank"].iloc[: slow - 1].isna().all()
        assert not pd.isna(result["slow_rank"].iloc[slow - 1])
        assert result["fast_rank"].iloc[: fast - 1].isna().all()
        assert not pd.isna(result["fast_rank"].iloc[fast - 1])

    def test_output_range_valid(self, ohlcv):
        result = aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        for col in ("slow_rank", "fast_rank", "raw_m", "agg_m"):
            valid = result[col].dropna()
            assert (valid >= 0.0).all(), f"{col} has values below 0"
            assert (valid <= 100.0).all(), f"{col} has values above 100"

    def test_case_insensitive_column_resolution(self, ohlcv):
        df_upper = ohlcv.rename(columns={"high": "High", "low": "Low", "close": "Close"})
        result_lower = aggregate_m_components(ohlcv, slow_period=30, fast_period=3)
        result_upper = aggregate_m_components(df_upper, slow_period=30, fast_period=3, high_column="high", low_column="low", close_column="close")
        np.testing.assert_allclose(result_lower["agg_m"].to_numpy(), result_upper["agg_m"].to_numpy(), rtol=1e-12, equal_nan=True)

    def test_missing_column_raises(self, ohlcv):
        df_bad = ohlcv.drop(columns=["high"])
        with pytest.raises(ValueError, match="must contain"):
            aggregate_m_components(df_bad, slow_period=30, fast_period=3)

    def test_invalid_slow_period_raises(self, ohlcv):
        with pytest.raises(ValueError, match="slow_period"):
            aggregate_m_components(ohlcv, slow_period=0)

    def test_invalid_fast_period_raises(self, ohlcv):
        with pytest.raises(ValueError, match="fast_period"):
            aggregate_m_components(ohlcv, fast_period=-1)

    def test_invalid_current_bar_weight_raises(self, ohlcv):
        with pytest.raises(ValueError, match="current_bar_weight"):
            aggregate_m_components(ohlcv, current_bar_weight=101)

    def test_invalid_trend_weight_raises(self, ohlcv):
        with pytest.raises(ValueError, match="trend_weight"):
            aggregate_m_components(ohlcv, trend_weight=-1)

    def test_n_less_than_slow_period_all_nan_agg(self, small_ohlcv):
        result = aggregate_m_components(small_ohlcv, slow_period=200, fast_period=3)
        assert result["agg_m"].isna().all()

    def test_gap_tolerant_ema_slow_path(self, ohlcv):
        # Inject a NaN gap mid-series into close to force raw_m to have NaN after warmup.
        df_gap = ohlcv.copy()
        gap_idx = 100
        df_gap.loc[df_gap.index[gap_idx], "close"] = np.nan
        result = aggregate_m_components(df_gap, slow_period=50, fast_period=5)
        # agg_m should be NaN at the gap and recover afterwards
        assert pd.isna(result["agg_m"].iloc[gap_idx])
        assert not pd.isna(result["agg_m"].iloc[gap_idx + 1])

    def test_no_input_mutation(self, ohlcv):
        original = ohlcv.copy()
        aggregate_m_components(ohlcv, slow_period=50, fast_period=5)
        pd.testing.assert_frame_equal(ohlcv, original)

    @pytest.mark.parametrize("trend_weight,current_bar_weight", [(0, 60), (100, 60), (50, 0), (50, 100)])
    def test_boundary_weights(self, ohlcv, trend_weight, current_bar_weight):
        result = aggregate_m_components(ohlcv, slow_period=30, fast_period=3, trend_weight=trend_weight, current_bar_weight=current_bar_weight)
        assert isinstance(result, pd.DataFrame)
        assert not result["agg_m"].dropna().empty


# ===========================================================================
# aggregate_m  (wrapper)
# ===========================================================================

class TestAggregateM:
    def test_returns_series(self, ohlcv):
        result = aggregate_m(ohlcv, slow_period=50, fast_period=5)
        assert isinstance(result, pd.Series)

    def test_identical_to_agg_m_column(self, ohlcv):
        components = aggregate_m_components(ohlcv, slow_period=50, fast_period=5, current_bar_weight=60, trend_weight=50)
        scalar = aggregate_m(ohlcv, slow_period=50, fast_period=5, current_bar_weight=60, trend_weight=50)
        np.testing.assert_allclose(scalar.to_numpy(), components["agg_m"].to_numpy(), rtol=1e-14, equal_nan=True)

    def test_index_preserved(self, ohlcv):
        result = aggregate_m(ohlcv, slow_period=50, fast_period=5)
        pd.testing.assert_index_equal(result.index, ohlcv.index)

    def test_output_range(self, ohlcv):
        result = aggregate_m(ohlcv, slow_period=50, fast_period=5)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_default_parameters_do_not_raise(self, ohlcv):
        # slow_period=252 > 300 bars, so output will be mostly NaN — that is fine.
        result = aggregate_m(ohlcv)
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv)

    def test_no_input_mutation(self, ohlcv):
        original = ohlcv.copy()
        aggregate_m(ohlcv, slow_period=50, fast_period=5)
        pd.testing.assert_frame_equal(ohlcv, original)

    @pytest.mark.parametrize("slow,fast", [(20, 3), (50, 5), (78, 3)])
    def test_parametric(self, ohlcv, slow, fast):
        result = aggregate_m(ohlcv, slow_period=slow, fast_period=fast)
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv)
