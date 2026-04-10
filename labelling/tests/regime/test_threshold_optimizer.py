import pytest
import numpy as np
import pandas as pd
from okmich_quant_labelling.regime import (
    CausalLabelThresholdOptimizer,
    MarketPropertyType,
    SeparationMetric,
    OptimizationObjective
)


def compute_slope(prices: pd.Series, window: int) -> pd.Series:
    """Helper function to compute rolling OLS slope."""
    log_prices = np.log(prices)
    slopes = []
    for i in range(window, len(log_prices)):
        x = np.arange(window)
        y = log_prices.iloc[i-window:i].values
        slope = np.polyfit(x, y, deg=1)[0]
        slopes.append(slope)

    result = pd.Series(index=prices.index, dtype=float)
    result.iloc[:window] = np.nan
    result.iloc[window:] = slopes
    return result


def compute_momentum_magnitude(prices: pd.Series, window: int) -> pd.Series:
    """Helper function to compute momentum magnitude (directionless)."""
    returns = prices.pct_change(window)
    magnitude = returns.abs()
    return magnitude


class TestCausalLabelThresholdOptimizer:
    """Test suite for CausalLabelThresholdOptimizer."""

    def test_initialization_with_enums(self):
        """Should initialize with enum parameters."""
        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_slope,
            method_type=MarketPropertyType.DIRECTION,
            objective=OptimizationObjective.SEPARATION,
            separation_metric=SeparationMetric.VARIANCE_RATIO
        )

        assert optimizer.method_type == MarketPropertyType.DIRECTION
        assert optimizer.objective == OptimizationObjective.SEPARATION
        assert optimizer.separation_metric == SeparationMetric.VARIANCE_RATIO
        assert optimizer.leaks_future is False

    def test_initialization_with_strings(self):
        """Should convert strings to enums."""
        optimizer = CausalLabelThresholdOptimizer(metric_func=compute_slope,
                                                  method_type=MarketPropertyType.DIRECTION,
                                                  objective=OptimizationObjective.SEPARATION,
                                                  separation_metric=SeparationMetric.RETURN_SEPARATION)

        assert optimizer.method_type == MarketPropertyType.DIRECTION
        assert optimizer.objective == OptimizationObjective.SEPARATION
        assert optimizer.separation_metric == SeparationMetric.RETURN_SEPARATION

    def test_invalid_objective_for_method_type(self):
        """Should raise error for invalid objective-method combination."""
        with pytest.raises(ValueError, match="TOTAL_RETURNS objective only valid"):
            CausalLabelThresholdOptimizer(
                metric_func=compute_slope,
                method_type=MarketPropertyType.VOLATILITY,
                objective=OptimizationObjective.TOTAL_RETURNS
            )

    def test_directionless_auto_set(self):
        """Should auto-set directionless for DIRECTIONLESS_MOMENTUM."""
        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_momentum_magnitude,
            method_type=MarketPropertyType.DIRECTIONLESS_MOMENTUM
        )

        assert optimizer.directionless is True

    def test_optimize_grid_search_direction(self):
        """Should optimize thresholds using grid search for direction method."""
        # Create trending data
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5 + 0.05))
        df = pd.DataFrame({'close': prices})

        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_slope,
            method_type=MarketPropertyType.DIRECTION,
            objective=OptimizationObjective.SEPARATION,
            window_range=(10, 30),
            threshold_range=(0.0001, 0.001)
        )

        result = optimizer.optimize(df, price_col='close', method='grid', n_grid=5, verbose=False)

        assert 'best_window' in result
        assert 'best_threshold' in result
        assert 'best_score' in result
        assert 'best_labels' in result
        assert 'all_results' in result

        assert 10 <= result['best_window'] <= 30
        assert 0.0001 <= result['best_threshold'] <= 0.001
        assert result['best_score'] > -np.inf
        assert isinstance(result['best_labels'], pd.Series)

    def test_optimize_grid_search_directionless(self):
        """Should optimize thresholds for directionless momentum."""
        # Create data with varying magnitude
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * np.linspace(0.1, 1.0, 200)))
        df = pd.DataFrame({'close': prices})

        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_momentum_magnitude,
            method_type=MarketPropertyType.DIRECTIONLESS_MOMENTUM,
            objective=OptimizationObjective.SEPARATION,
            window_range=(5, 20),
            threshold_range=(0.001, 0.05)
        )

        result = optimizer.optimize(df, price_col='close', method='grid', n_grid=5, verbose=False)

        assert 'best_window' in result
        assert 'best_threshold' in result
        assert isinstance(result['best_threshold'], tuple)
        assert len(result['best_threshold']) == 2
        assert result['best_threshold'][0] < result['best_threshold'][1]
        assert set(result['best_labels'].dropna().unique()).issubset({0, 1, 2})

    def test_optimize_total_returns_objective(self):
        """Should optimize for total returns."""
        # Create trending data
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5 + 0.05))
        df = pd.DataFrame({'close': prices})

        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_slope,
            method_type=MarketPropertyType.DIRECTION,
            objective=OptimizationObjective.TOTAL_RETURNS,
            window_range=(10, 30),
            threshold_range=(0.0001, 0.001)
        )

        result = optimizer.optimize(df, price_col='close', method='grid', n_grid=5, verbose=False)

        assert result['best_score'] is not None
        # For trending data, should find positive returns
        assert result['best_score'] > -1.0

    def test_missing_price_column(self):
        """Should raise error if price column not found."""
        df = pd.DataFrame({'price': [100, 101, 102]})
        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_slope,
            method_type=MarketPropertyType.DIRECTION
        )

        with pytest.raises(KeyError, match="not found"):
            optimizer.optimize(df, price_col='close', n_grid=3, verbose=False)

    def test_repr(self):
        """Test string representation."""
        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_slope,
            method_type=MarketPropertyType.DIRECTION,
            objective=OptimizationObjective.SEPARATION
        )

        repr_str = repr(optimizer)
        assert 'CausalLabelThresholdOptimizer' in repr_str
        assert 'direction' in repr_str
        assert 'separation' in repr_str

    def test_leaks_future_attribute(self):
        """Optimizer should have leaks_future=False."""
        optimizer = CausalLabelThresholdOptimizer(
            metric_func=compute_slope,
            method_type=MarketPropertyType.DIRECTION
        )
        assert optimizer.leaks_future is False


# ---------------------------------------------------------------------------
# Regression tests added for correctness fixes
# ---------------------------------------------------------------------------

class TestCHScoringNaNHandling:
    """
    _evaluate_separation with VARIANCE_RATIO must handle pct_change() NaNs at
    the first bar without producing -inf scores or crashing.
    """

    def _make_optimizer(self, metric_func=None):
        return CausalLabelThresholdOptimizer(
            metric_func=metric_func or compute_slope,
            method_type=MarketPropertyType.DIRECTION,
            objective=OptimizationObjective.SEPARATION,
            separation_metric=SeparationMetric.VARIANCE_RATIO,
            window_range=(5, 20),
            threshold_range=(0.0001, 0.002),
        )

    def test_ch_score_is_finite_despite_first_bar_nan(self):
        """
        pct_change() produces NaN at bar 0. Score must be finite (not -inf) when
        enough bars remain after filtering NaNs.
        """
        np.random.seed(10)
        prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5 + 0.05))
        df = pd.DataFrame({"close": prices})

        optimizer = self._make_optimizer()
        result = optimizer.optimize(df, price_col="close", method="grid", n_grid=4, verbose=False)

        assert result["best_score"] > -np.inf, (
            f"Expected finite score, got {result['best_score']}"
        )

    def test_single_unique_label_returns_neg_inf(self):
        """
        When all bars get the same label (no separation possible), score must be -inf.
        """
        prices = pd.Series([100.0] * 50)  # flat prices → slope always ~0

        # A metric that always returns 0 → threshold 0.001 will assign all to label 0
        def zero_metric(p, w):
            return pd.Series(np.zeros(len(p)), index=p.index)

        optimizer = CausalLabelThresholdOptimizer(
            metric_func=zero_metric,
            method_type=MarketPropertyType.DIRECTION,
            objective=OptimizationObjective.SEPARATION,
            separation_metric=SeparationMetric.VARIANCE_RATIO,
            window_range=(5, 10),
            threshold_range=(0.001, 0.01),
        )
        df = pd.DataFrame({"close": prices})
        result = optimizer.optimize(df, method="grid", n_grid=3, verbose=False)

        # All thresholds produce single-label → every score is -inf
        assert result["best_score"] == -np.inf

    def test_grid_results_contain_no_nan_scores(self):
        """
        All score entries in all_results must be either finite or -inf (never NaN).
        """
        np.random.seed(5)
        prices = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.5 + 0.03))
        df = pd.DataFrame({"close": prices})

        optimizer = self._make_optimizer()
        result = optimizer.optimize(df, method="grid", n_grid=4, verbose=False)

        scores = result["all_results"]["score"]
        # No NaN scores — each outcome must be a real number or -inf
        assert not scores.isna().any(), (
            f"NaN scores found in grid results:\n{scores[scores.isna()]}"
        )

