import numpy as np
import pandas as pd
from enum import StrEnum
from typing import Callable
from scipy.optimize import differential_evolution
from sklearn.metrics import calinski_harabasz_score


class MarketPropertyType(StrEnum):
    """Types of regime detection methods."""

    DIRECTION = 'direction'                      # Up/Neutral/Down (-1/0/+1)
    MOMENTUM = 'momentum'                        # Decelerating/Stable/Accelerating (-1/0/+1)
    DIRECTIONLESS_MOMENTUM = 'directionless_momentum'  # Small/Medium/High (0/1/2)
    VOLATILITY = 'volatility'                    # Low/Normal/High (-1/0/+1 or 0/1/2)
    PATH_STRUCTURE = 'path_structure'            # Smooth/Normal/Choppy (-1/0/+1)
    LIQUIDITY = 'liquidity'                      # Thin/Normal/Liquid (-1/0/+1)


class SeparationMetric(StrEnum):
    """Metrics for measuring regime separation quality."""

    VARIANCE_RATIO = 'variance_ratio'            # Calinski-Harabasz score
    MEAN_SEPARATION = 'mean_separation'          # Normalized mean difference (Cohen's d)
    RETURN_SEPARATION = 'return_separation'      # For direction/momentum
    VOLATILITY_SEPARATION = 'volatility_separation'  # For volatility regimes
    EFFICIENCY_SEPARATION = 'efficiency_separation'  # For path structure


class OptimizationObjective(StrEnum):
    """Optimization objectives for threshold search."""

    SEPARATION = 'separation'                    # Maximize regime distinctness
    TOTAL_RETURNS = 'total_returns'              # Maximize P&L (direction/momentum only)


class CausalLabelThresholdOptimizer:
    """
    Optimize thresholds for causal regime labeling.

    Searches for optimal (window, threshold) parameters that maximize an objective function (separation or returns).

    Parameters
    ----------
    metric_func : Callable
        Function that computes the metric from prices and window.
        Signature: metric_func(prices: pd.Series, window: int) -> pd.Series

    method_type : MarketPropertyType | str
        Type of regime method (direction, momentum, etc.)

    objective : OptimizationObjective | str, default=SEPARATION
        What to optimize for

    separation_metric : SeparationMetric | str, default=VARIANCE_RATIO
        Which separation metric to use (when objective=SEPARATION)

    window_range : tuple[int, int], default=(10, 100)
        Search range for lookback window

    threshold_range : tuple[float, float], default=(0.0001, 0.01)
        Search range for threshold value

    directionless : bool, default=None
        If True, use magnitude (0/1/2 labels). Auto-set from method_type.

    Attributes
    ----------
    leaks_future : bool
        Always False - this optimizer uses only causal data
    """

    def __init__(self, metric_func: Callable, method_type: MarketPropertyType,
                 objective: OptimizationObjective = OptimizationObjective.SEPARATION,
                 separation_metric: SeparationMetric = SeparationMetric.VARIANCE_RATIO,
                 window_range: tuple[int, int] = (10, 100), threshold_range: tuple[float, float] = (0.0001, 0.01),
                 directionless: bool = None):

        # Validate objective compatibility
        if objective == OptimizationObjective.TOTAL_RETURNS:
            if method_type not in {MarketPropertyType.DIRECTION, MarketPropertyType.MOMENTUM}:
                raise ValueError(
                    f"TOTAL_RETURNS objective only valid for DIRECTION/MOMENTUM methods, "
                    f"got {method_type}"
                )

        # Auto-set directionless flag
        if directionless is None:
            directionless = (method_type == MarketPropertyType.DIRECTIONLESS_MOMENTUM)

        self.metric_func = metric_func
        self.method_type = method_type
        self.objective = objective
        self.separation_metric = separation_metric
        self.window_range = window_range
        self.threshold_range = threshold_range
        self.directionless = directionless
        self.leaks_future = False

    def optimize(self, df: pd.DataFrame, price_col: str = 'close', method: str = 'grid', n_grid: int = 20,
                 verbose: bool = True) -> dict:
        """
        Optimize window and threshold parameters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price data
        price_col : str, default='close'
            Column name for price
        method : str, default='grid'
            Optimization method: 'grid' or 'scipy'
        n_grid : int, default=20
            Number of grid points per dimension (for grid search)
        verbose : bool, default=True
            Print progress

        Returns
        -------
        dict
            Results dictionary with keys:
            - best_window: int
            - best_threshold: float (or tuple if directionless)
            - best_score: float
            - best_labels: pd.Series
            - all_results: pd.DataFrame (grid search results)
        """
        if price_col not in df.columns:
            raise KeyError(f"Column '{price_col}' not found in DataFrame")

        prices = df[price_col]
        if method == 'grid':
            return self._grid_search(df, prices, n_grid, verbose)
        elif method == 'scipy':
            return self._scipy_optimize(df, prices, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'grid' or 'scipy'")

    def _grid_search(self, df: pd.DataFrame, prices: pd.Series, n_grid: int, verbose: bool) -> dict:
        """Grid search over window and threshold space."""
        # Create grid
        windows = np.linspace(self.window_range[0], self.window_range[1], n_grid, dtype=int)
        if self.directionless:
            # For directionless: search for two thresholds (low, high)
            thresholds_low = np.linspace(self.threshold_range[0], self.threshold_range[1] * 0.5, n_grid // 2)
            thresholds_high = np.linspace(self.threshold_range[1] * 0.5, self.threshold_range[1], n_grid // 2)
        else:
            thresholds = np.linspace(self.threshold_range[0], self.threshold_range[1], n_grid)

        results = []
        best_score = -np.inf
        best_params = None
        best_labels = None

        total_iterations = len(windows) * (len(thresholds_low) * len(thresholds_high) if self.directionless else len(thresholds))
        iteration = 0

        for window in windows:
            # Compute metric
            metric = self.metric_func(prices, window)
            if self.directionless:
                # Search over two thresholds
                for thresh_low in thresholds_low:
                    for thresh_high in thresholds_high:
                        if thresh_high <= thresh_low:
                            continue

                        iteration += 1
                        if verbose and iteration % 10 == 0:
                            print(f"Progress: {iteration}/{total_iterations}", end='\r')

                        # Apply thresholds: 0 if < low, 2 if > high, 1 otherwise
                        labels = self._apply_threshold_directionless(metric, thresh_low, thresh_high)

                        # Evaluate
                        score = self._evaluate(df, labels, prices)

                        results.append({
                            'window': window,
                            'threshold_low': thresh_low,
                            'threshold_high': thresh_high,
                            'score': score
                        })

                        if score > best_score:
                            best_score = score
                            best_params = (window, thresh_low, thresh_high)
                            best_labels = labels
            else:
                # Search over single threshold
                for threshold in thresholds:
                    iteration += 1
                    if verbose and iteration % 10 == 0:
                        print(f"Progress: {iteration}/{total_iterations}", end='\r')

                    # Apply threshold: -1/0/+1
                    labels = self._apply_threshold(metric, threshold)

                    # Evaluate
                    score = self._evaluate(df, labels, prices)

                    results.append({'window': window, 'threshold': threshold, 'score': score})
                    if score > best_score:
                        best_score = score
                        best_params = (window, threshold)
                        best_labels = labels
        if verbose:
            print()  # New line after progress

        results_df = pd.DataFrame(results)
        if best_params is None:
            # Every combination produced -inf (e.g. all bars get the same label)
            return {
                'best_window': None,
                'best_threshold': None,
                'best_score': best_score,
                'best_labels': best_labels,
                'all_results': results_df,
            }
        if self.directionless:
            return {
                'best_window': best_params[0],
                'best_threshold': (best_params[1], best_params[2]),
                'best_score': best_score,
                'best_labels': best_labels,
                'all_results': results_df
            }
        else:
            return {
                'best_window': best_params[0],
                'best_threshold': best_params[1],
                'best_score': best_score,
                'best_labels': best_labels,
                'all_results': results_df
            }

    def _scipy_optimize(self, df: pd.DataFrame, prices: pd.Series, verbose: bool) -> dict:
        # Define objective function for scipy
        def objective(params):
            if self.directionless:
                window, thresh_low, thresh_high = int(params[0]), params[1], params[2]
                if thresh_high <= thresh_low:
                    return 1e10  # Invalid
                metric = self.metric_func(prices, window)
                labels = self._apply_threshold_directionless(metric, thresh_low, thresh_high)
            else:
                window, threshold = int(params[0]), params[1]
                metric = self.metric_func(prices, window)
                labels = self._apply_threshold(metric, threshold)

            score = self._evaluate(df, labels, prices)
            return -score  # Minimize negative score

        # Bounds
        if self.directionless:
            bounds = [
                self.window_range,
                (self.threshold_range[0], self.threshold_range[1] * 0.5),
                (self.threshold_range[1] * 0.5, self.threshold_range[1])
            ]
        else:
            bounds = [self.window_range, self.threshold_range]

        # Optimize
        result = differential_evolution(objective, bounds, seed=42, disp=verbose)

        # Extract best parameters
        if self.directionless:
            best_window = int(result.x[0])
            best_threshold = (result.x[1], result.x[2])
            metric = self.metric_func(prices, best_window)
            best_labels = self._apply_threshold_directionless(metric, result.x[1], result.x[2])
        else:
            best_window = int(result.x[0])
            best_threshold = result.x[1]
            metric = self.metric_func(prices, best_window)
            best_labels = self._apply_threshold(metric, best_threshold)

        return {
            'best_window': best_window,
            'best_threshold': best_threshold,
            'best_score': -result.fun,
            'best_labels': best_labels,
            'all_results': None
        }

    def _apply_threshold(self, metric: pd.Series, threshold: float) -> pd.Series:
        """Apply threshold to metric: -1/0/+1."""
        labels = np.where(metric > threshold, 1, np.where(metric < -threshold, -1, 0))
        result = pd.Series(labels, index=metric.index, dtype='float64')
        result[metric.isna()] = np.nan
        return result.astype('Int8')

    def _apply_threshold_directionless(self, metric: pd.Series, thresh_low: float, thresh_high: float) -> pd.Series:
        """Apply thresholds for directionless (magnitude): 0/1/2."""
        # Assume metric is already magnitude (positive)
        labels = np.where(
            metric > thresh_high, 2,
            np.where(metric > thresh_low, 1, 0)
        )
        result = pd.Series(labels, index=metric.index, dtype='float64')
        result[metric.isna()] = np.nan
        return result.astype('Int8')

    def _evaluate(self, df: pd.DataFrame, labels: pd.Series, prices: pd.Series) -> float:
        """Evaluate quality of labels using the chosen objective."""
        if self.objective == OptimizationObjective.SEPARATION:
            return self._evaluate_separation(df, labels, prices)
        elif self.objective == OptimizationObjective.TOTAL_RETURNS:
            return self._evaluate_total_returns(df, labels, prices)
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def _evaluate_separation(self, df: pd.DataFrame, labels: pd.Series, prices: pd.Series) -> float:
        """Evaluate regime separation quality."""
        # Remove NaN
        valid_mask = labels.notna()
        labels_clean = labels[valid_mask].values

        if len(np.unique(labels_clean)) < 2:
            return -np.inf  # Need at least 2 regimes

        if self.separation_metric == SeparationMetric.VARIANCE_RATIO:
            # Calinski-Harabasz score
            # Need feature matrix - use the metric values or returns
            returns_raw = prices.pct_change()[valid_mask]
            finite_mask = np.isfinite(returns_raw.values)
            returns = returns_raw.values[finite_mask].reshape(-1, 1)
            labels_ch = labels_clean[finite_mask]
            if len(returns) < len(np.unique(labels_ch)) + 1:
                return -np.inf
            try:
                score = calinski_harabasz_score(returns, labels_ch)
                return score
            except Exception:
                return -np.inf

        elif self.separation_metric == SeparationMetric.MEAN_SEPARATION:
            # Cohen's d between regimes
            returns = prices.pct_change()[valid_mask]
            unique_labels = np.unique(labels_clean)

            if len(unique_labels) < 2:
                return -np.inf

            # For simplicity, compare first and last label (extreme regimes)
            regime_low = returns[labels_clean == unique_labels[0]]
            regime_high = returns[labels_clean == unique_labels[-1]]

            if len(regime_low) < 2 or len(regime_high) < 2:
                return -np.inf

            mean_diff = abs(regime_high.mean() - regime_low.mean())
            pooled_std = np.sqrt((regime_low.std()**2 + regime_high.std()**2) / 2)

            if pooled_std == 0:
                return -np.inf

            return mean_diff / pooled_std

        elif self.separation_metric == SeparationMetric.RETURN_SEPARATION:
            # Mean return difference between regimes
            returns = prices.pct_change()[valid_mask]
            unique_labels = sorted(np.unique(labels_clean))

            if len(unique_labels) < 2:
                return -np.inf

            # Compare extreme regimes
            returns_low = returns[labels_clean == unique_labels[0]]
            returns_high = returns[labels_clean == unique_labels[-1]]

            if len(returns_low) < 2 or len(returns_high) < 2:
                return -np.inf

            mean_diff = abs(returns_high.mean() - returns_low.mean())
            overall_std = returns.std()

            if overall_std == 0:
                return -np.inf

            return mean_diff / overall_std

        else:
            raise NotImplementedError(f"Separation metric {self.separation_metric} not yet implemented")

    def _evaluate_total_returns(self, df: pd.DataFrame, labels: pd.Series, prices: pd.Series) -> float:
        """Evaluate total returns following regime signals."""
        # Forward returns
        returns = prices.pct_change().shift(-1)  # Next bar return

        # Remove NaN
        valid_mask = labels.notna() & returns.notna()
        labels_clean = labels[valid_mask]
        returns_clean = returns[valid_mask]

        # Position: +1 for uptrend regime, -1 for downtrend, 0 for ranging
        positions = labels_clean.values

        # Strategy returns
        strategy_returns = positions * returns_clean.values

        # Total return (cumulative)
        total_return = strategy_returns.sum()

        return total_return

    def __repr__(self):
        return (
            f"CausalLabelThresholdOptimizer("
            f"method_type={self.method_type}, "
            f"objective={self.objective}, "
            f"separation_metric={self.separation_metric}, "
            f"window_range={self.window_range}, "
            f"threshold_range={self.threshold_range}, "
            f"directionless={self.directionless})"
        )

