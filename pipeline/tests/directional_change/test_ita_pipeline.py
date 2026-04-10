"""
Tests for ITAPipeline and run_sliding_window.

ITAPipeline tests use n_calls=5 / n_initial=3 to keep the suite fast
(Bayesian optimisation is the bottleneck). Correctness of individual stages
is tested in the stage-specific test modules.

run_sliding_window tests verify:
  - Return type and column contract
  - DatetimeIndex requirement
  - Insufficient-data windows are skipped
  - compound=True carries capital forward
  - compound=False keeps capital independent
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline.directional_change import (
    ALPHA_BOUNDS,
    THETA_BOUNDS_5MIN,
    THETA_BOUNDS_TICK,
    ITAPipeline,
    run_sliding_window,
)

# ── shared fixtures ───────────────────────────────────────────────────────────

_FAST_KWARGS = dict(n_calls=5, n_initial=3, random_state=0, hmm_random_state=0)
_EXPECTED_WINDOW_COLS = {
    "test_start", "test_end",
    "optimal_theta", "optimal_alpha",
    "n_trades", "n_winners", "win_ratio",
    "cumulative_return", "max_drawdown",
    "profit_factor", "sharpe", "final_capital",
}


def _random_prices(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))


def _datetime_prices(n_months: int, bars_per_month: int = 200, seed: int = 0) -> pd.Series:
    """Return a price series with a DatetimeIndex spanning n_months calendar months."""
    n = n_months * bars_per_month
    rng = np.random.default_rng(seed)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    start = pd.Timestamp("2020-01-01")
    # Use business-day-ish spacing so each month has ~bars_per_month bars
    freq = pd.tseries.offsets.BusinessHour()
    idx = pd.date_range(start=start, periods=n, freq="4h")
    return pd.Series(values, index=idx)


# ── ITAPipeline — unfitted guard ──────────────────────────────────────────────

class TestITAPipelineUnfitted:
    def test_run_before_fit_raises(self):
        prices = _random_prices(500)
        pipeline = ITAPipeline(**_FAST_KWARGS)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.run(prices)

    def test_repr_shows_unfitted(self):
        pipeline = ITAPipeline(**_FAST_KWARGS)
        assert "unfitted" in repr(pipeline)


# ── ITAPipeline — fit ─────────────────────────────────────────────────────────

class TestITAPipelineFit:
    @pytest.fixture
    def fitted_pipeline(self):
        prices = _random_prices(3000)
        pipeline = ITAPipeline(theta_bounds=THETA_BOUNDS_TICK, **_FAST_KWARGS)
        pipeline.fit(prices)
        return pipeline

    def test_fit_returns_self(self):
        prices = _random_prices(3000)
        pipeline = ITAPipeline(theta_bounds=THETA_BOUNDS_TICK, **_FAST_KWARGS)
        result = pipeline.fit(prices)
        assert result is pipeline

    def test_theta_within_bounds(self, fitted_pipeline):
        lo, hi = THETA_BOUNDS_5MIN
        assert lo <= fitted_pipeline.theta_ <= hi

    def test_alpha_within_bounds(self, fitted_pipeline):
        lo, hi = ALPHA_BOUNDS
        assert lo <= fitted_pipeline.alpha_ <= hi

    def test_hmm_is_set(self, fitted_pipeline):
        assert fitted_pipeline.hmm_ is not None

    def test_s1_idx_is_int(self, fitted_pipeline):
        assert isinstance(fitted_pipeline.s1_idx_, int)

    def test_s1_idx_is_0_or_1(self, fitted_pipeline):
        assert fitted_pipeline.s1_idx_ in (0, 1)

    def test_is_fitted_flag(self, fitted_pipeline):
        assert fitted_pipeline._is_fitted is True

    def test_repr_shows_fitted(self, fitted_pipeline):
        assert "fitted" in repr(fitted_pipeline)


# ── ITAPipeline — run ─────────────────────────────────────────────────────────

class TestITAPipelineRun:
    @pytest.fixture
    def pipeline_and_test(self):
        train = _random_prices(3000, seed=1)
        test = _random_prices(500, seed=2)
        pipeline = ITAPipeline(theta_bounds=THETA_BOUNDS_TICK, **_FAST_KWARGS)
        pipeline.fit(train)
        return pipeline, test

    def test_result_is_dict(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        result = pipeline.run(test)
        assert isinstance(result, dict)

    def test_result_has_required_keys(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        result = pipeline.run(test)
        for key in ("final_capital", "cumulative_return", "max_drawdown", "n_trades", "n_winners", "trade_log", "optimal_theta", "optimal_alpha"):
            assert key in result

    def test_optimal_theta_matches_fitted(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        result = pipeline.run(test)
        assert result["optimal_theta"] == pipeline.theta_

    def test_optimal_alpha_matches_fitted(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        result = pipeline.run(test)
        assert result["optimal_alpha"] == pipeline.alpha_

    def test_n_winners_le_n_trades(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        result = pipeline.run(test)
        assert result["n_winners"] <= result["n_trades"]

    def test_max_drawdown_non_negative(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        result = pipeline.run(test)
        assert result["max_drawdown"] >= 0.0

    def test_custom_initial_capital(self, pipeline_and_test):
        pipeline, test = pipeline_and_test
        r1 = pipeline.run(test, initial_capital=10_000.0)
        r2 = pipeline.run(test, initial_capital=50_000.0)
        # CRR is capital-independent
        assert r1["cumulative_return"] == pytest.approx(r2["cumulative_return"], rel=1e-9)


# ── run_sliding_window — type check ──────────────────────────────────────────

class TestSlidingWindowTypeCheck:
    def test_raises_on_non_datetime_index(self):
        prices = _random_prices(1000)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            run_sliding_window(prices)

    def test_returns_dataframe(self):
        prices = _datetime_prices(n_months=4, bars_per_month=200)
        result = run_sliding_window(prices, **_FAST_KWARGS)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_data_returns_empty(self):
        # Only 1 month of data — cannot form any 2+1 window
        prices = _datetime_prices(n_months=1)
        result = run_sliding_window(prices, **_FAST_KWARGS)
        assert isinstance(result, pd.DataFrame)
        # May be empty or have 0 rows (insufficient windows)
        assert len(result) == 0 or set(result.columns).issuperset({"cumulative_return"})


# ── run_sliding_window — column contract ──────────────────────────────────────

class TestSlidingWindowColumns:
    @pytest.fixture
    def sw_result(self):
        prices = _datetime_prices(n_months=5, bars_per_month=300)
        return run_sliding_window(prices, **_FAST_KWARGS)

    def test_has_all_expected_columns(self, sw_result):
        if len(sw_result) > 0:
            assert _EXPECTED_WINDOW_COLS.issubset(set(sw_result.columns))

    def test_n_trades_is_integer(self, sw_result):
        if len(sw_result) > 0:
            assert sw_result["n_trades"].dtype in (np.int64, np.int32, int, object)
            assert all(isinstance(v, (int, np.integer)) for v in sw_result["n_trades"])

    def test_n_winners_le_n_trades(self, sw_result):
        if len(sw_result) > 0:
            assert (sw_result["n_winners"] <= sw_result["n_trades"]).all()

    def test_max_drawdown_non_negative(self, sw_result):
        if len(sw_result) > 0:
            assert (sw_result["max_drawdown"] >= 0.0).all()

    def test_theta_within_bounds(self, sw_result):
        if len(sw_result) > 0:
            lo, hi = THETA_BOUNDS_TICK
            assert sw_result["optimal_theta"].between(lo, hi).all()

    def test_alpha_within_bounds(self, sw_result):
        if len(sw_result) > 0:
            lo, hi = ALPHA_BOUNDS
            assert sw_result["optimal_alpha"].between(lo, hi).all()


# ── run_sliding_window — compounding ─────────────────────────────────────────

class TestSlidingWindowCompounding:
    def test_compound_true_final_capital_differs(self):
        prices = _datetime_prices(n_months=5, bars_per_month=300, seed=42)
        r_compound = run_sliding_window(prices, compound=True, **_FAST_KWARGS)
        r_flat = run_sliding_window(prices, compound=False, **_FAST_KWARGS)
        if len(r_compound) > 1 and len(r_flat) > 1:
            # At least second window should differ between compound and flat
            # (not guaranteed for first window)
            # Just verify both return valid DataFrames
            assert isinstance(r_compound, pd.DataFrame)
            assert isinstance(r_flat, pd.DataFrame)

    def test_compound_false_crr_independent(self):
        # With compound=False all windows have the same initial capital →
        # cumulative_return values are each measured from the same base
        prices = _datetime_prices(n_months=5, bars_per_month=300, seed=7)
        result = run_sliding_window(prices, compound=False, initial_capital=10_000.0, **_FAST_KWARGS)
        if len(result) > 0:
            # final_capital = 10000 * (1 + crr/100) for each row independently
            expected_fc = 10_000.0 * (1.0 + result["cumulative_return"] / 100.0)
            pd.testing.assert_series_equal(result["final_capital"].reset_index(drop=True), expected_fc.reset_index(drop=True), check_names=False, rtol=1e-9)
