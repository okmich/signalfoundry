"""
Trend Persistence Labeling Method

This module provides drift-based trend labeling that measures price momentum normalized by volatility.
Unlike Z-score or continuous trend methods, this approach focuses on the persistence of price movement relative to recent volatility.

Core Concept:
- Calculates drift: price change over a window
- Normalizes by (price × volatility) to make it scale-invariant
- Optional Z-score normalization for standardization
- Smooths the signal to reduce noise
- Labels based on fixed thresholds (±0.25)

Key Features:
- Volatility-adjusted momentum measurement
- Simple threshold-based classification
- Optional Z-score normalization
- Smoothing to reduce whipsaws
- Grid search optimization for parameters

Use Cases:
- Identifying persistent trends vs choppy markets
- Volatility-aware trend following
- Simple momentum strategies

================================================================================
HELPER FUNCTIONS GUIDE
================================================================================

1. trend_persistence_labeling(price_series, window, smooth, zscore_norm, name)

   USE WHEN: You want to label trends based on volatility-adjusted price drift

   DOES:
   - Calculates drift = (price - price[t-window])
   - Normalizes by (price[t-window] × volatility)
   - Optionally applies Z-score normalization
   - Smooths the score with rolling mean
   - Labels: 1 (uptrend), -1 (downtrend), 0 (neutral)

   PARAMETERS:
   - window (default=20): Lookback period for drift calculation
   - smooth (default=5): Smoothing window to reduce noise
   - zscore_norm (default=True): Whether to apply Z-score normalization
   - name (default='trend_label'): Name of the returned Series

   RETURNS: pd.Series of trend labels (1, -1, 0)

   THRESHOLDS: score >= 0.25 → uptrend, score <= -0.25 → downtrend

   EXAMPLE:
   labels = trend_persistence_labeling(
       prices,
       window=30,
       smooth=5,
       zscore_norm=True
   )


2. optimize_trend_persistence(price_series, window_range, smooth_range, zscore_options)

   USE WHEN: You want to find optimal parameters for trend persistence labeling

   DOES:
   - Tests all combinations of window, smooth, and zscore_norm parameters
   - Simulates trading strategy for each combination
   - Ranks by cumulative returns
   - Returns full results DataFrame and best parameters

   PARAMETERS:
   - window_range (default=(10, 50, 5)): (start, stop, step) for window values
   - smooth_range (default=(2, 10, 2)): (start, stop, step) for smooth values
   - zscore_options (default=(True, False)): Whether to test Z-score normalization

   RETURNS:
   - results_df: DataFrame with all tested combinations and their performance
   - best_params: Dict with best window, smooth, zscore_norm, and metrics

   METRICS COMPUTED:
   - cum_return: Total cumulative return
   - mean_return: Average per-period return
   - sharpe: Sharpe ratio (mean / std)

   EXAMPLE:
   results_df, best_params = optimize_trend_persistence(
       prices,
       window_range=(10, 50, 10),  # Test windows: 10, 20, 30, 40
       smooth_range=(3, 9, 3),     # Test smooth: 3, 6
       zscore_options=(True, False) # Test both with/without Z-score
   )
   print(f"Best window: {best_params['window']}")
   print(f"Best cumulative return: {best_params['cum_return']:.2%}")

================================================================================
Quick Decision Guide:
---------------------
- Need to label persistence-based trends? → Use trend_persistence_labeling()
- Want to optimize parameters? → Use optimize_trend_persistence()
- Want volatility-adjusted momentum? → Use this method
- Need simple threshold-based signals? → This is a good choice

================================================================================
OPTIMIZATION CRITERIA EXPLAINED
================================================================================

This module uses CUMULATIVE RETURNS as the optimization criterion (simplest approach):

GOAL: Find parameters that maximize historical cumulative returns

HOW IT SCORES:
--------------
Uses a simple backtest approach:
1. Generate trend labels with candidate parameters
2. Simulate strategy: shifted_labels × returns (avoids look-ahead)
3. Calculate cumulative return: prod(1 + returns) - 1
4. Rank by cumulative return

METRICS COMPUTED:
- Cumulative Return: Total return over the period (PRIMARY METRIC)
- Mean Return: Average per-period return
- Sharpe Ratio: Risk-adjusted returns (mean / std)

OPTIMIZATION METHOD:
-------------------
Uses GRID SEARCH (exhaustive):
- Tests ALL combinations of parameters
- Simple but thorough
- Can be slow for large parameter ranges

Example parameter space:
- window: 10, 20, 30, 40 (4 values)
- smooth: 3, 6, 9 (3 values)
- zscore_norm: True, False (2 values)
- Total combinations: 4 × 3 × 2 = 24 tests

BEST FOR:
- Maximizing historical returns (simple objective)
- Small parameter spaces (grid search is feasible)
- When you want to see all results ranked

COMPARISON WITH OTHER METHODS:
------------------------------

Trend Persistence (this module):
  ✅ Simple cumulative returns criterion
  ✅ Fast for small parameter spaces
  ✅ Easy to interpret
  ✅ Returns full results DataFrame
  ❌ Only optimizes for returns (no risk consideration in ranking)
  ❌ Grid search (slower for large spaces)

Z-Score (z_score_trend.py):
  ✅ Multi-metric composite score (Sharpe, win rate, persistence)
  ✅ Bayesian optimization (faster)
  ❌ More complex optimization
  ❌ Only returns best parameters

Continuous (continous_trend.py):
  ✅ Optimizes prediction accuracy
  ✅ Uses cross-validation
  ❌ Doesn't optimize trading performance directly
  ❌ Grid search (slower)

USAGE PATTERN:
--------------
Like other optimization functions, use in RESEARCH phase only:

# RESEARCH PHASE (run once or periodically)
results_df, best_params = optimize_trend_persistence(historical_data)

# Inspect results
print(results_df.head(10))  # See top 10 parameter combinations

# Extract best parameters
WINDOW = int(best_params['window'])       # e.g., 30
SMOOTH = int(best_params['smooth'])       # e.g., 5
ZSCORE = bool(best_params['zscore_norm']) # e.g., True

# TRADING PHASE (use fixed parameters)
def generate_signal(prices):
    labels = trend_persistence_labeling(
        prices,
        window=WINDOW,  # Fixed from optimization
        smooth=SMOOTH,  # Fixed from optimization
        zscore_norm=ZSCORE  # Fixed from optimization
    )
    return labels.iloc[-1]  # Current label

# Use in live trading
signal = generate_signal(live_prices)

CAVEAT:
-------
Optimizing purely on cumulative returns can lead to overfitting. Consider:
- Using out-of-sample testing
- Re-optimizing periodically (e.g., quarterly)
- Looking at Sharpe ratio, not just returns
- Checking consistency across different market conditions

"""

import itertools

import numpy as np
import pandas as pd

from ._trend_evaluation import evaluate_trend_performance


def trend_persistence_labeling(price_series: pd.Series, window: int = 20, smooth: int = 5, zscore_norm: bool = True,
                               name: str = "trend_label") -> pd.Series:
    px = price_series.astype(float)
    px_shifted = px.shift(window)

    drift = px - px_shifted
    returns = px.pct_change(fill_method=None)
    vol = returns.rolling(window, min_periods=window).std()

    denominator = px_shifted * (vol + 1e-12)
    score = drift / denominator

    # Vectorized z-score normalization
    if zscore_norm:
        mu = score.rolling(window, min_periods=1).mean()
        sd = score.rolling(window, min_periods=1).std()
        score = (score - mu) / (sd + 1e-12)

    # Apply smoothing with optimized rolling mean
    score = score.rolling(smooth, min_periods=1).mean()

    labels = pd.Series(
        np.select([score >= 0.25, score <= -0.25], [1.0, -1.0], default=0.0),
        index=price_series.index,
        name=name,
    )

    return labels.fillna(0.0)


def optimize_trend_persistence(price_series: pd.Series, window_range=(10, 50, 5),
                               smooth_range=(2, 10, 2), zscore_options=(True, False)):
    """
    Optimize trend persistence parameters using grid search.

    Uses the unified evaluation framework to compute metrics efficiently.
    Optimizes for cumulative returns while also computing Sharpe ratio.

    Parameters:
    -----------
    price_series : pd.Series
        Price series to optimize on
    window_range : tuple
        (start, stop, step) for window parameter
    smooth_range : tuple
        (start, stop, step) for smooth parameter
    zscore_options : tuple
        Values to test for zscore_norm (typically (True, False))

    Returns:
    --------
    results_df : pd.DataFrame
        All tested combinations sorted by cumulative return
    best_params : dict
        Best parameter combination with metrics
    """
    results = []
    for w, s, z in itertools.product(range(*window_range), range(*smooth_range), zscore_options):
        labels = trend_persistence_labeling(price_series, window=w, smooth=s, zscore_norm=z)

        # Use unified evaluation framework - only compute needed metrics for speed
        metrics = evaluate_trend_performance(price_series, labels, metrics=["cumulative_returns", "sharpe_ratio"])
        results.append(
            {
                "window": w,
                "smooth": s,
                "zscore_norm": z,
                "cum_return": metrics["cumulative_returns"],
                "mean_return": metrics["cumulative_returns"]
                / len(price_series),  # Approximate
                "sharpe": metrics["sharpe_ratio"],
            }
        )

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["cum_return"].idxmax()]
    best_params = best_row.to_dict()

    return results_df.sort_values("cum_return", ascending=False), best_params
