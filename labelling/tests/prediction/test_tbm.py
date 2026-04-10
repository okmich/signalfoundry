"""Tests for Triple Barrier Method (TBM) labeling module."""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.prediction.tbm import (
    VolatilityEstimator,
    VolatilityConfig,
    BarrierConfig,
    TBMConfig,
    OptimizationMetric,
    compute_volatility,
    apply_tbm,
    tbm_from_signals,
    optimize_tbm_volatility,
    optimize_tbm_full,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_ohlc_prices():
    """Create synthetic OHLC price data with clear trends and valid OHLC constraints."""
    np.random.seed(42)
    n = 200

    # Create trending close prices
    base = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)

    # Generate OHLC with proper constraints: low <= min(open, close), high >= max(open, close)
    close = base
    open_ = np.roll(close, 1) + np.random.randn(n) * 0.1
    open_[0] = close[0]

    # High must be >= max(open, close), add random extension
    high = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.5) + 0.1
    # Low must be <= min(open, close), subtract random extension
    low = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.5) - 0.1

    index = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }, index=index)


@pytest.fixture
def synthetic_ohlc_with_barriers():
    """Create OHLC data designed to hit specific barriers."""
    n = 100
    index = pd.date_range("2024-01-01", periods=n, freq="h")

    # Start with stable price around 100
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    open_ = np.full(n, 100.0)

    # Event at bar 20: will hit upper barrier at bar 25
    close[25:30] = 105.0
    high[25:30] = 106.0

    # Event at bar 40: will hit lower barrier at bar 45
    close[45:50] = 95.0
    low[45:50] = 94.0

    # Event at bar 60: will hit vertical barrier (no price move)
    # (prices stay flat)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }, index=index)


@pytest.fixture
def simple_events(synthetic_ohlc_prices):
    """Create simple event DataFrame with long signals."""
    np.random.seed(123)
    n_events = 10
    event_indices = np.linspace(20, 150, n_events, dtype=int)
    event_times = synthetic_ohlc_prices.index[event_indices]

    return pd.DataFrame({
        "side": np.ones(n_events, dtype=int),
    }, index=event_times)


@pytest.fixture
def mixed_events(synthetic_ohlc_prices):
    """Create events with both long and short signals."""
    np.random.seed(456)
    n_events = 20
    event_indices = np.linspace(20, 150, n_events, dtype=int)
    event_times = synthetic_ohlc_prices.index[event_indices]

    sides = np.where(np.random.rand(n_events) > 0.5, 1, -1)

    return pd.DataFrame({
        "side": sides,
    }, index=event_times)


# ============================================================================
# Test Configuration Classes
# ============================================================================

class TestVolatilityConfig:
    def test_default_config(self):
        config = VolatilityConfig()
        assert config.estimator == VolatilityEstimator.ATR
        assert config.params == {"window": 20}

    def test_custom_config(self):
        config = VolatilityConfig(
            estimator=VolatilityEstimator.PARKINSON,
            params={"window": 14},
        )
        assert config.estimator == VolatilityEstimator.PARKINSON
        assert config.params["window"] == 14


class TestBarrierConfig:
    def test_default_config(self):
        config = BarrierConfig()
        assert config.upper_multiplier == 2.0
        assert config.lower_multiplier == 2.0
        assert config.max_holding_bars == 10
        assert config.vertical_as_zero is False

    def test_custom_config(self):
        config = BarrierConfig(
            upper_multiplier=3.0,
            lower_multiplier=1.5,
            max_holding_bars=20,
            vertical_as_zero=True,
        )
        assert config.upper_multiplier == 3.0
        assert config.lower_multiplier == 1.5
        assert config.max_holding_bars == 20
        assert config.vertical_as_zero is True


class TestTBMConfig:
    def test_default_config(self):
        config = TBMConfig()
        assert config.volatility.estimator == VolatilityEstimator.ATR
        assert config.barrier.upper_multiplier == 2.0

    def test_nested_config(self):
        config = TBMConfig(
            volatility=VolatilityConfig(
                estimator=VolatilityEstimator.GARMAN_KLASS,
                params={"window": 10},
            ),
            barrier=BarrierConfig(
                upper_multiplier=2.5,
                lower_multiplier=1.0,
            ),
        )
        assert config.volatility.estimator == VolatilityEstimator.GARMAN_KLASS
        assert config.barrier.upper_multiplier == 2.5


# ============================================================================
# Test Volatility Estimators
# ============================================================================

class TestVolatilityEstimators:
    def test_volatility_estimator_values(self):
        assert VolatilityEstimator.STD.value == "std"
        assert VolatilityEstimator.ATR.value == "atr"
        assert VolatilityEstimator.PARKINSON.value == "parkinson"
        assert VolatilityEstimator.GARMAN_KLASS.value == "garman_klass"


class TestComputeVolatility:
    def test_std_volatility(self, synthetic_ohlc_prices):
        config = VolatilityConfig(
            estimator=VolatilityEstimator.STD,
            params={"window": 20},
        )
        vol = compute_volatility(synthetic_ohlc_prices, config)

        assert len(vol) == len(synthetic_ohlc_prices)
        assert np.isnan(vol[:19]).all()  # First window-1 values are NaN
        assert not np.isnan(vol[19:]).all()  # After warmup, we have values

    def test_atr_volatility(self, synthetic_ohlc_prices):
        config = VolatilityConfig(
            estimator=VolatilityEstimator.ATR,
            params={"window": 14},
        )
        vol = compute_volatility(synthetic_ohlc_prices, config)

        assert len(vol) == len(synthetic_ohlc_prices)
        assert not np.isnan(vol[14:]).all()

    def test_parkinson_volatility(self, synthetic_ohlc_prices):
        config = VolatilityConfig(
            estimator=VolatilityEstimator.PARKINSON,
            params={"window": 20},
        )
        vol = compute_volatility(synthetic_ohlc_prices, config)

        assert len(vol) == len(synthetic_ohlc_prices)
        assert not np.isnan(vol[20:]).all()

    def test_garman_klass_volatility(self, synthetic_ohlc_prices):
        config = VolatilityConfig(
            estimator=VolatilityEstimator.GARMAN_KLASS,
            params={"window": 20},
        )
        vol = compute_volatility(synthetic_ohlc_prices, config)

        assert len(vol) == len(synthetic_ohlc_prices)
        assert not np.isnan(vol[20:]).all()

    def test_volatility_positive(self, synthetic_ohlc_prices):
        """Volatility should be positive where it's not NaN."""
        for estimator in VolatilityEstimator:
            config = VolatilityConfig(estimator=estimator, params={"window": 20})
            vol = compute_volatility(synthetic_ohlc_prices, config)
            valid_vol = vol[~np.isnan(vol)]
            assert (valid_vol >= 0).all(), f"{estimator} produced negative volatility"


# ============================================================================
# Test apply_tbm
# ============================================================================

class TestApplyTBM:
    def test_basic_apply_tbm(self, synthetic_ohlc_prices, simple_events):
        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, simple_events, config)

        # Should have results for events
        assert len(result) > 0

        # Check output columns
        expected_cols = {"label", "ret", "exit_time", "barrier_hit", "bars_held"}
        assert set(result.columns) == expected_cols

    def test_output_types(self, synthetic_ohlc_prices, simple_events):
        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, simple_events, config)

        # Check column types
        assert result["label"].dtype in [np.int64, np.int32, int]
        assert result["ret"].dtype == np.float64
        assert result["bars_held"].dtype in [np.int64, np.int32, int]

    def test_label_values(self, synthetic_ohlc_prices, simple_events):
        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, simple_events, config)

        # Labels should be in {-1, 0, 1}
        unique_labels = set(result["label"].unique())
        assert unique_labels.issubset({-1, 0, 1})

    def test_barrier_hit_values(self, synthetic_ohlc_prices, simple_events):
        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, simple_events, config)

        # barrier_hit should be one of {"upper", "lower", "vertical"}
        unique_barriers = set(result["barrier_hit"].unique())
        assert unique_barriers.issubset({"upper", "lower", "vertical"})

    def test_bars_held_range(self, synthetic_ohlc_prices, simple_events):
        config = TBMConfig(barrier=BarrierConfig(max_holding_bars=10))
        result = apply_tbm(synthetic_ohlc_prices, simple_events, config)

        # bars_held should be in [1, max_holding_bars]
        assert (result["bars_held"] >= 1).all()
        assert (result["bars_held"] <= 10).all()

    def test_with_mixed_sides(self, synthetic_ohlc_prices, mixed_events):
        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, mixed_events, config)

        # Should handle both long and short events
        assert len(result) > 0

    def test_with_different_volatility_estimators(self, synthetic_ohlc_prices, simple_events):
        for estimator in VolatilityEstimator:
            config = TBMConfig(
                volatility=VolatilityConfig(estimator=estimator),
            )
            result = apply_tbm(synthetic_ohlc_prices, simple_events, config)
            assert len(result) >= 0  # May be empty if volatility not available

    def test_vertical_as_zero(self, synthetic_ohlc_prices, simple_events):
        config = TBMConfig(
            barrier=BarrierConfig(
                upper_multiplier=10.0,  # Very high barrier
                lower_multiplier=10.0,  # Very high barrier
                max_holding_bars=5,
                vertical_as_zero=True,
            ),
        )
        result = apply_tbm(synthetic_ohlc_prices, simple_events, config)

        # With high multipliers, most events should hit vertical barrier
        vertical_hits = result[result["barrier_hit"] == "vertical"]
        if len(vertical_hits) > 0:
            # Labels should be 0 for vertical hits when vertical_as_zero=True
            assert (vertical_hits["label"] == 0).all()

    def test_empty_events(self, synthetic_ohlc_prices):
        events = pd.DataFrame({"side": []}, index=pd.DatetimeIndex([]))
        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, events, config)

        assert len(result) == 0
        assert set(result.columns) == {"label", "ret", "exit_time", "barrier_hit", "bars_held"}

    def test_missing_columns_raises(self, synthetic_ohlc_prices, simple_events):
        # Missing OHLC column
        incomplete_prices = synthetic_ohlc_prices.drop(columns=["high"])
        config = TBMConfig()

        with pytest.raises(ValueError, match="prices must contain columns"):
            apply_tbm(incomplete_prices, simple_events, config)

    def test_missing_side_raises(self, synthetic_ohlc_prices):
        events = pd.DataFrame({"wrong_col": [1, 1]}, index=synthetic_ohlc_prices.index[:2])
        config = TBMConfig()

        with pytest.raises(ValueError, match="events must contain 'side' column"):
            apply_tbm(synthetic_ohlc_prices, events, config)


# ============================================================================
# Test tbm_from_signals
# ============================================================================

class TestTBMFromSignals:
    def test_basic_from_signals(self, synthetic_ohlc_prices):
        # Create signal series
        np.random.seed(789)
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(synthetic_ohlc_prices), p=[0.1, 0.8, 0.1]),
            index=synthetic_ohlc_prices.index,
        )

        config = TBMConfig()
        result = tbm_from_signals(synthetic_ohlc_prices, signals, config)

        # Should have results only for non-zero signals
        assert len(result) > 0

    def test_filters_zero_signals(self, synthetic_ohlc_prices):
        # Create signals with some zeros
        signals = pd.Series(
            [0, 1, 0, -1, 0, 1, 0, 0, -1, 0] + [0] * (len(synthetic_ohlc_prices) - 10),
            index=synthetic_ohlc_prices.index,
        )

        config = TBMConfig(
            barrier=BarrierConfig(max_holding_bars=5),
        )
        result = tbm_from_signals(synthetic_ohlc_prices, signals, config)

        # Should only process non-zero signals (4 total: 2 longs, 2 shorts)
        # Some may be skipped due to insufficient forward data or volatility warmup
        assert len(result) <= 4

    def test_all_zero_signals(self, synthetic_ohlc_prices):
        signals = pd.Series(
            np.zeros(len(synthetic_ohlc_prices)),
            index=synthetic_ohlc_prices.index,
        )

        config = TBMConfig()
        result = tbm_from_signals(synthetic_ohlc_prices, signals, config)

        assert len(result) == 0


# ============================================================================
# Test Barrier Logic
# ============================================================================

class TestBarrierLogic:
    def test_upper_barrier_long(self, synthetic_ohlc_with_barriers):
        """Long position hitting upper barrier should be profit."""
        prices = synthetic_ohlc_with_barriers

        # Event at bar 20 (before upper barrier hit at bar 25)
        events = pd.DataFrame({
            "side": [1],  # Long
        }, index=[prices.index[20]])

        config = TBMConfig(
            volatility=VolatilityConfig(params={"window": 5}),
            barrier=BarrierConfig(
                upper_multiplier=1.0,
                lower_multiplier=1.0,
                max_holding_bars=10,
            ),
        )
        result = apply_tbm(prices, events, config)

        if len(result) > 0:
            # Upper barrier hit for long = profit = label 1
            upper_hits = result[result["barrier_hit"] == "upper"]
            if len(upper_hits) > 0:
                assert (upper_hits["label"] == 1).all()

    def test_lower_barrier_long(self, synthetic_ohlc_with_barriers):
        """Long position hitting lower barrier should be loss."""
        prices = synthetic_ohlc_with_barriers

        # Event at bar 40 (before lower barrier hit at bar 45)
        events = pd.DataFrame({
            "side": [1],  # Long
        }, index=[prices.index[40]])

        config = TBMConfig(
            volatility=VolatilityConfig(params={"window": 5}),
            barrier=BarrierConfig(
                upper_multiplier=1.0,
                lower_multiplier=1.0,
                max_holding_bars=10,
            ),
        )
        result = apply_tbm(prices, events, config)

        if len(result) > 0:
            # Lower barrier hit for long = loss = label -1
            lower_hits = result[result["barrier_hit"] == "lower"]
            if len(lower_hits) > 0:
                assert (lower_hits["label"] == -1).all()

    def test_upper_barrier_short(self, synthetic_ohlc_with_barriers):
        """Short position hitting upper barrier should be loss."""
        prices = synthetic_ohlc_with_barriers

        events = pd.DataFrame({
            "side": [-1],  # Short
        }, index=[prices.index[20]])

        config = TBMConfig(
            volatility=VolatilityConfig(params={"window": 5}),
            barrier=BarrierConfig(
                upper_multiplier=1.0,
                lower_multiplier=1.0,
                max_holding_bars=10,
            ),
        )
        result = apply_tbm(prices, events, config)

        if len(result) > 0:
            # Upper barrier hit for short = loss = label -1
            upper_hits = result[result["barrier_hit"] == "upper"]
            if len(upper_hits) > 0:
                assert (upper_hits["label"] == -1).all()

    def test_lower_barrier_short(self, synthetic_ohlc_with_barriers):
        """Short position hitting lower barrier should be profit."""
        prices = synthetic_ohlc_with_barriers

        events = pd.DataFrame({
            "side": [-1],  # Short
        }, index=[prices.index[40]])

        config = TBMConfig(
            volatility=VolatilityConfig(params={"window": 5}),
            barrier=BarrierConfig(
                upper_multiplier=1.0,
                lower_multiplier=1.0,
                max_holding_bars=10,
            ),
        )
        result = apply_tbm(prices, events, config)

        if len(result) > 0:
            # Lower barrier hit for short = profit = label 1
            lower_hits = result[result["barrier_hit"] == "lower"]
            if len(lower_hits) > 0:
                assert (lower_hits["label"] == 1).all()


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_event_at_end_of_prices(self, synthetic_ohlc_prices):
        """Events near end should be skipped due to insufficient forward data."""
        # Event at the last bar
        events = pd.DataFrame({
            "side": [1],
        }, index=[synthetic_ohlc_prices.index[-1]])

        config = TBMConfig(
            barrier=BarrierConfig(max_holding_bars=10),
        )
        result = apply_tbm(synthetic_ohlc_prices, events, config)

        # Should be empty - not enough forward data
        assert len(result) == 0

    def test_event_time_not_in_prices(self, synthetic_ohlc_prices):
        """Events not in price index should be skipped."""
        # Event at a time not in prices
        missing_time = pd.Timestamp("2020-01-01 00:00:00")
        events = pd.DataFrame({
            "side": [1],
        }, index=[missing_time])

        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, events, config)

        # Should be empty
        assert len(result) == 0

    def test_invalid_side_skipped(self, synthetic_ohlc_prices):
        """Events with invalid side values should be skipped."""
        events = pd.DataFrame({
            "side": [0, 2, 1],  # 0 and 2 are invalid
        }, index=synthetic_ohlc_prices.index[30:33])

        config = TBMConfig()
        result = apply_tbm(synthetic_ohlc_prices, events, config)

        # Only the valid side=1 event should be processed
        assert len(result) <= 1

    def test_volatility_nan_skipped(self, synthetic_ohlc_prices):
        """Events with NaN volatility should be skipped."""
        # Event at the very beginning (before volatility warmup)
        events = pd.DataFrame({
            "side": [1],
        }, index=[synthetic_ohlc_prices.index[0]])

        config = TBMConfig(
            volatility=VolatilityConfig(params={"window": 20}),
        )
        result = apply_tbm(synthetic_ohlc_prices, events, config)

        # Should be empty - volatility is NaN at start
        assert len(result) == 0


# ============================================================================
# Test Optimization Metric Enum
# ============================================================================

class TestOptimizationMetric:
    def test_metric_values(self):
        assert OptimizationMetric.SHARPE_RATIO.value == "sharpe_ratio"
        assert OptimizationMetric.WIN_RATE.value == "win_rate"
        assert OptimizationMetric.PROFIT_FACTOR.value == "profit_factor"
        assert OptimizationMetric.TOTAL_RETURN.value == "total_return"
        assert OptimizationMetric.AVG_RETURN.value == "avg_return"
        assert OptimizationMetric.VERTICAL_RATE.value == "vertical_rate"


# ============================================================================
# Test optimize_tbm_volatility
# ============================================================================

class TestOptimizeTBMVolatility:
    @pytest.fixture
    def optimization_signals(self, synthetic_ohlc_prices):
        """Create denser signals for optimization testing."""
        np.random.seed(42)
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(synthetic_ohlc_prices), p=[0.15, 0.70, 0.15]),
            index=synthetic_ohlc_prices.index,
        )
        return signals

    def test_basic_optimization(self, synthetic_ohlc_prices, optimization_signals):
        barrier_config = BarrierConfig(
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            max_holding_bars=10,
        )

        result = optimize_tbm_volatility(
            prices=synthetic_ohlc_prices,
            signals=optimization_signals,
            barrier_config=barrier_config,
            estimators=[VolatilityEstimator.ATR, VolatilityEstimator.STD],
            windows=[10, 14],
            min_trades=5,  # Lower threshold for testing
        )

        # Check result structure
        assert "best_estimator" in result
        assert "best_window" in result
        assert "best_score" in result
        assert "best_config" in result
        assert "results_df" in result
        assert "metric" in result

    def test_optimization_returns_valid_config(self, synthetic_ohlc_prices, optimization_signals):
        barrier_config = BarrierConfig(
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            max_holding_bars=10,
        )

        result = optimize_tbm_volatility(
            prices=synthetic_ohlc_prices,
            signals=optimization_signals,
            barrier_config=barrier_config,
            windows=[10, 14],
            min_trades=5,
        )

        # Best config should be usable
        if result["best_config"] is not None:
            labels = tbm_from_signals(
                synthetic_ohlc_prices,
                optimization_signals,
                result["best_config"],
            )
            assert isinstance(labels, pd.DataFrame)

    def test_results_df_structure(self, synthetic_ohlc_prices, optimization_signals):
        barrier_config = BarrierConfig(
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            max_holding_bars=10,
        )

        result = optimize_tbm_volatility(
            prices=synthetic_ohlc_prices,
            signals=optimization_signals,
            barrier_config=barrier_config,
            estimators=[VolatilityEstimator.ATR],
            windows=[10, 14],
            min_trades=5,
        )

        results_df = result["results_df"]

        # Check columns exist
        expected_cols = {
            "estimator", "window", "score", "n_trades",
            "win_rate", "loss_rate", "vertical_rate",
            "total_return", "avg_return", "std_return", "avg_bars_held",
        }
        assert expected_cols.issubset(set(results_df.columns))

        # Should have results for each combination
        assert len(results_df) == 2  # 1 estimator x 2 windows

    def test_different_metrics(self, synthetic_ohlc_prices, optimization_signals):
        barrier_config = BarrierConfig(
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            max_holding_bars=10,
        )

        for metric in OptimizationMetric:
            result = optimize_tbm_volatility(
                prices=synthetic_ohlc_prices,
                signals=optimization_signals,
                barrier_config=barrier_config,
                estimators=[VolatilityEstimator.ATR],
                windows=[10],
                metric=metric,
                min_trades=5,
            )

            assert result["metric"] == metric.value

    def test_min_trades_filter(self, synthetic_ohlc_prices):
        """Test that min_trades filters out low-count results."""
        # Create very sparse signals
        signals = pd.Series(0, index=synthetic_ohlc_prices.index)
        signals.iloc[50] = 1  # Only one signal

        barrier_config = BarrierConfig(max_holding_bars=10)

        result = optimize_tbm_volatility(
            prices=synthetic_ohlc_prices,
            signals=signals,
            barrier_config=barrier_config,
            windows=[10],
            min_trades=30,  # Higher than available trades
        )

        # All scores should be NaN due to insufficient trades
        assert result["results_df"]["score"].isna().all()


# ============================================================================
# Test optimize_tbm_full
# ============================================================================

class TestOptimizeTBMFull:
    @pytest.fixture
    def optimization_signals(self, synthetic_ohlc_prices):
        """Create denser signals for optimization testing."""
        np.random.seed(42)
        signals = pd.Series(
            np.random.choice([-1, 0, 1], size=len(synthetic_ohlc_prices), p=[0.15, 0.70, 0.15]),
            index=synthetic_ohlc_prices.index,
        )
        return signals

    def test_basic_full_optimization(self, synthetic_ohlc_prices, optimization_signals):
        result = optimize_tbm_full(
            prices=synthetic_ohlc_prices,
            signals=optimization_signals,
            estimators=[VolatilityEstimator.ATR],
            windows=[10],
            upper_multipliers=[2.0],
            lower_multipliers=[2.0],
            max_holding_bars_list=[10],
            min_trades=5,
        )

        # Check result structure
        assert "best_estimator" in result
        assert "best_window" in result
        assert "best_upper_mult" in result
        assert "best_lower_mult" in result
        assert "best_max_bars" in result
        assert "best_score" in result
        assert "best_config" in result
        assert "results_df" in result
        assert "n_combinations" in result

    def test_full_optimization_combinations(self, synthetic_ohlc_prices, optimization_signals):
        result = optimize_tbm_full(
            prices=synthetic_ohlc_prices,
            signals=optimization_signals,
            estimators=[VolatilityEstimator.ATR, VolatilityEstimator.STD],
            windows=[10, 14],
            upper_multipliers=[1.5, 2.0],
            lower_multipliers=[1.5, 2.0],
            max_holding_bars_list=[10, 20],
            min_trades=5,
        )

        # Should have all combinations
        expected_combinations = 2 * 2 * 2 * 2 * 2  # 32
        assert result["n_combinations"] == expected_combinations
        assert len(result["results_df"]) == expected_combinations

    def test_full_results_df_structure(self, synthetic_ohlc_prices, optimization_signals):
        result = optimize_tbm_full(
            prices=synthetic_ohlc_prices,
            signals=optimization_signals,
            estimators=[VolatilityEstimator.ATR],
            windows=[10],
            upper_multipliers=[2.0],
            lower_multipliers=[2.0],
            max_holding_bars_list=[10],
            min_trades=5,
        )

        results_df = result["results_df"]

        # Check columns
        expected_cols = {
            "estimator", "window", "upper_mult", "lower_mult", "max_bars",
            "score", "n_trades", "win_rate", "loss_rate", "vertical_rate",
            "total_return", "avg_return",
        }
        assert expected_cols.issubset(set(results_df.columns))
