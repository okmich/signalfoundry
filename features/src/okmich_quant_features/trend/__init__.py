"""
================================================================================
TREND LABELING METHODS - COMPREHENSIVE GUIDE
================================================================================

This package provides multiple approaches for trend identification and labeling.
Each method has different characteristics, optimization strategies, and use cases.

This guide focuses on the THREE MAIN TREND LABELING METHODS with optimization:
1. Continuous Trend Labeling (Price Action & Moving Average approaches)
2. Z-Score Trend Labeling (Statistical deviation)
3. Trend Persistence Labeling (Volatility-adjusted drift)

================================================================================
QUICK COMPARISON TABLE
================================================================================

Method                    | Approach              | Role               | Primary Metric           | Speed
--------------------------|----------------------|--------------------|--------------------------|-------
continuous_trend_labeling | Price extremes       | Directional label  | Prediction Accuracy      | Slow
continuous_ma_trend_lab.  | MA deviation         | Directional label  | Distribution Target      | Fast
trend_persistence_lab.    | Volatility-adj drift | Directional label  | Cumulative Returns       | Medium
zscore_trend_features     | Statistical z-score  | Reversal feature   | Continuous z-score       | Fast

================================================================================
DETAILED METHOD COMPARISON
================================================================================

1. CONTINUOUS TREND LABELING (continous_trend.py)
--------------------------------------------------
Two variants available:

A) continuous_trend_labeling() - PRICE ACTION APPROACH
   └─ Tracks price extremes and reversals using state machine
   └─ Parameters: omega (threshold)
   └─ Labels: 1 (uptrend), -1 (downtrend), 0 (neutral)
   └─ NO LOOKAHEAD BIAS: Sequential processing only

   Optimization: find_optimal_parameters()
   - Method: Grid search with Logistic Regression + Time Series CV
   - Metric: Classification accuracy (predict next trend from past patterns)
   - Parameters: omega (threshold), lambda (lookback window)
   - Use case: When you want to predict future trend direction

B) continuous_ma_trend_labeling() - MOVING AVERAGE APPROACH
   └─ Measures deviation from moving average
   └─ Parameters: omega, trend_window, smooth_window
   └─ Labels: 1 (uptrend), -1 (downtrend), 0 (neutral), NaN (insufficient data)
   └─ NO LOOKAHEAD BIAS: Uses backward-looking rolling windows

   Optimization: determine_optimal_ma_omega()
   - Method: Tests percentiles to achieve target label distribution
   - Metric: Proximity to target neutral percentage (e.g., 40%)
   - Parameters: omega only (trend_window, smooth_window fixed)
   - Use case: When you want balanced label distribution

BEST FOR:
- Price action signals (variant A)
- Smoother, less noisy labels (variant B)
- Predicting trend changes (variant A optimization)
- Balanced trading opportunities (variant B optimization)


2. Z-SCORE TREND EXHAUSTION (z_score_trend.py) — REVERSAL FEATURE, NOT DIRECTIONAL LABELLER
---------------------------------------------------------------------------------------------
zscore_trend_features()  [PREFERRED]
   └─ Returns continuous z-score features for ML models
   └─ Parameters: window, deriv_window
   └─ Outputs: zscore (raw), zscore_deriv (rate of change), zscore_abs (magnitude)
   └─ NO LOOKAHEAD BIAS: Uses backward-looking rolling statistics

zscore_trend_labeling()  [BACKWARD COMPATIBILITY]
   └─ Discretizes z-score into {-1, 0, 1} labels
   └─ NOT recommended for directional ensembles — use CTL, CTL-MA, Trend Persistence instead
   └─ Empirically: only 22% of symbols profitable as a directional label (Phase 2 baseline)

ROLE IN FRAMEWORK:
- Measures how statistically extreme the current move is (trend exhaustion)
- Orthogonal to all directional labelers (pairwise kappa ~0)
- Use as a reversal/exhaustion feature or conviction modifier, not as a directional voter

BEST FOR:
- Trend exhaustion detection (is the current move overextended?)
- Reversal signal (high |z-score| = potential mean reversion)
- Continuous feature input to ML models
- Conviction modifier for directional ensemble output


3. TREND PERSISTENCE LABELING (trend_persistence.py)
-----------------------------------------------------
trend_persistence_labeling()
   └─ Volatility-adjusted price drift
   └─ Formula: (price - price[t-window]) / (price[t-window] × volatility)
   └─ Parameters: window, smooth, zscore_norm
   └─ Labels: 1 (uptrend), -1 (downtrend), 0 (neutral)
   └─ Optional z-score normalization
   └─ NO LOOKAHEAD BIAS: Uses past data only

   Optimization: optimize_trend_persistence()
   - Method: Grid search (exhaustive)
   - Metric: Cumulative returns (simplest approach)
   - Also computes: Mean return, Sharpe ratio
   - Parameters: window, smooth, zscore_norm
   - Returns: Full DataFrame of ALL results ranked by performance

BEST FOR:
- Volatility-aware momentum strategies
- Maximizing historical returns
- When you want to see ALL parameter combinations ranked
- Simple, interpretable optimization criterion

================================================================================
OPTIMIZATION CRITERIA SUMMARY
================================================================================

Method                     | Criterion              | Formula                          | Best For
---------------------------|------------------------|----------------------------------|------------------
continuous (price action)  | Prediction Accuracy    | accuracy_score(y_true, y_pred)  | Forecasting trends
continuous (MA)            | Label Distribution     | |actual% - target%|             | Balanced signals
trend_persistence          | Cumulative Returns     | Π(1 + returns) - 1              | Maximizing returns
zscore (feature)           | N/A (continuous)       | (ret - μ) / σ                   | Reversal detection

================================================================================
DECISION TREE: WHICH METHOD TO USE?
================================================================================

Start here: What's your PRIMARY goal?

├─ PREDICT future trend direction
│  └─> Use continuous_trend_labeling() with find_optimal_parameters()
│      Optimizes: Prediction accuracy via ML

├─ GET BALANCED trading opportunities (not too many/few signals)
│  └─> Use continuous_ma_trend_labeling() with determine_optimal_ma_omega()
│      Optimizes: Label distribution

├─ MAXIMIZE total returns (simplest goal)
│  └─> Use trend_persistence_labeling() with optimize_trend_persistence()
│      Optimizes: Cumulative returns
│      Bonus: Returns full results DataFrame

├─ DETECT trend exhaustion / potential reversals
│  └─> Use zscore_trend_features()
│      Returns continuous z-score, its derivative, and absolute magnitude
│      Use as feature input to ML models, NOT as a directional label

├─ NEED FAST optimization
│  └─> Distribution-based: continuous_ma_trend_labeling() (very fast)

├─ MULTIPLE ASSETS to optimize
│  └─> Use optimize_zscore_params_multiasset()
│      Parallel processing, asset-specific parameters

================================================================================
USAGE PATTERN: RESEARCH vs TRADING
================================================================================

ALL optimization functions should be used in RESEARCH PHASE ONLY:

# RESEARCH PHASE (run once or periodically - e.g., quarterly)
# ============================================================

# Option 1: Continuous (price action)
from okmich_quant_features.trend import continuous_trend_labeling, find_optimal_parameters
result = find_optimal_parameters(historical_prices)
OMEGA = result['best_omega']
LAMBDA = result['best_lambda']

# Option 2: Continuous (MA)
from okmich_quant_features.trend import continuous_ma_trend_labeling, determine_optimal_ma_omega
OMEGA = determine_optimal_ma_omega(historical_prices, target_neutral_pct=40)

# Option 3: Z-Score
from okmich_quant_features.trend import zscore_trend_labeling, optimize_zscore_trend_labeling_params
result = optimize_zscore_trend_labeling_params(
    historical_prices,
    weights={'sharpe': 2, 'win_rate': 1, 'persistence': 1, 'num_trades': 0.5}
)
WINDOW = result['window']
Z_THRESHOLD = result['z_threshold']

# Option 4: Trend Persistence
from okmich_quant_features.trend import trend_persistence_labeling, optimize_trend_persistence
results_df, best = optimize_trend_persistence(historical_prices)
WINDOW = int(best['window'])
SMOOTH = int(best['smooth'])


# TRADING PHASE (use FIXED parameters)
# =====================================

def get_trend_signal(prices):
    # Use ONE of these with FIXED parameters from research phase

    # Option 1: Continuous (price action)
    labels = continuous_trend_labeling(prices, omega=OMEGA)

    # Option 2: Continuous (MA)
    labels = continuous_ma_trend_labeling(prices, omega=OMEGA)

    # Option 3: Z-Score
    labels, zscore = zscore_trend_labeling(prices, window=WINDOW, z_threshold=Z_THRESHOLD)

    # Option 4: Trend Persistence
    labels = trend_persistence_labeling(prices, window=WINDOW, smooth=SMOOTH)

    return labels.iloc[-1]  # Current signal

# Use in live trading
current_signal = get_trend_signal(live_prices)

================================================================================
COMPARISON: PROS & CONS
================================================================================

CONTINUOUS TREND (Price Action):
  ✅ No lookahead bias (sequential)
  ✅ Sensitive to sharp reversals
  ✅ Optimizes prediction accuracy
  ✅ Works on price data directly
  ❌ Slower optimization (grid search)
  ❌ Only considers accuracy, not risk

CONTINUOUS TREND (Moving Average):
  ✅ Smoother, less noisy
  ✅ Fast optimization
  ✅ Ensures balanced signals
  ✅ Simple to interpret
  ❌ Lags more than price action
  ❌ Doesn't optimize trading performance

Z-SCORE (reversal feature, not directional labeller):
  ✅ Detects trend exhaustion and overextension
  ✅ Continuous feature (no information loss from discretization)
  ✅ Orthogonal to directional labellers (independent signal)
  ✅ Useful as conviction modifier for directional ensemble
  ❌ Not viable as standalone directional label (22% profitable)
  ❌ Discrete labels spend ~97% of time in neutral at optimal thresholds

TREND PERSISTENCE:
  ✅ Volatility-adjusted
  ✅ Simple criterion (returns)
  ✅ Returns full results DataFrame
  ✅ Easy to interpret
  ❌ Grid search (can be slow)
  ❌ Risk not primary consideration

================================================================================
OVERFITTING WARNING
================================================================================

⚠️  ALL optimization methods can overfit to historical data!

Best practices:
- Optimize on training data, validate on out-of-sample data
- Re-optimize periodically (quarterly/semi-annually), not continuously
- Don't optimize after every losing period (discipline!)
- Consider multiple market regimes in your training data
- Use the simplest method that meets your needs
- Monitor live performance vs backtested expectations

================================================================================
ADDITIONAL METHODS IN THIS PACKAGE
================================================================================

Other trend-related functions (not covered in detail above):
- fibonacci_range: Fibonacci-based support/resistance
- heiken_ashi: Heikin Ashi candles and momentum
- precision_trend: Precision trend indicator
- squeeze: Squeeze momentum indicator (TTM Squeeze)
- trading_the_trend: Trading the Trend indicator
- ttm_trend: TTM Trend indicator

These methods are available but don't have the same comprehensive optimization frameworks as the three main methods documented above.

================================================================================
"""

import pandas as pd
import numpy as np

from .continous_trend import continuous_trend_labeling, continuous_ma_trend_labeling, find_optimal_continuous_trend_parameters,\
    determine_optimal_ma_omega, analyze_optimal_ma_omega, analyze_instrument_ma_omega, compare_ma_omega_values

from .fibonacci_range import fibonacci_range
from .heiken_ashi import heiken_ashi, heiken_ashi_momentum, heiken_ashi_momentum_advanced

from .misc import bollinger_band, cci
from .precision_trend import precision_trend
from .trading_the_trend import trading_the_trend
from .trend_persistence import trend_persistence_labeling, optimize_trend_persistence
from ._trend_evaluation import evaluate_trend_performance, compare_trend_methods
from .ttm_trend import ttm_trend
from .z_score_trend import zscore_trend_labeling, zscore_trend_features, optimize_zscore_trend_labeling_params, \
    optimize_zscore_params_multiasset, optimization_objective_for_params, compute_metrics, simulate_strategy_returns


def core_trend_features(df: pd.DataFrame,
    # Bollinger Bands parameters
    bb_window: int = 24, bb_deviation_up: float = 2.0, bb_deviation_down: float = 2.0,
    # CCI parameters
    cci_window: int = 24,
    # Heiken Ashi parameters
    ha_momentum_lookback: int = 3, ha_momentum_adv_lookback: int = 3, ha_momentum_adv_smoothing: int = 2,
    # Precision Trend parameters
    precision_period: int = 14, precision_sensitivity: float = 3.0,
    # TTM Trend parameters
    ttm_period: int = 10,
    # Continuous Trend Labeling parameters
    continuous_omega: float = 0.15,
    # Continuous MA Trend Labeling parameters
    continuous_ma_omega: float = 0.02, continuous_ma_trend_window: int = 20, continuous_ma_smooth_window: int = 5,
    # Trend Persistence parameters
    persistence_window: int = 20, persistence_smooth: int = 5, persistence_zscore_norm: bool = True,
    # Z-Score Trend Labeling parameters
    zscore_window: int = 30, zscore_z_threshold: float = 1.5,
    # Column names
    open_col: str = "open", high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.DataFrame:
    #
    result = pd.DataFrame(index=df.index)

    # Extract OHLC columns
    open_price = df[open_col]
    high_price = df[high_col]
    low_price = df[low_col]
    close_price = df[close_col]

    # ==================== Bollinger Bands ====================
    bb_upper, bb_middle, bb_lower, bb_percent_b, bb_width = bollinger_band(close_price, window=bb_window,
                                                                           deviation_up=bb_deviation_up,
                                                                           deviation_down=bb_deviation_down)

    safe_close = close_price.replace(0, np.nan)
    result[f"bb_upper_{bb_window}"] = bb_upper / safe_close
    result[f"bb_middle_{bb_window}"] = bb_middle / safe_close
    result[f"bb_lower_{bb_window}"] = bb_lower / safe_close
    result[f"bb_percent_b_{bb_window}"] = bb_percent_b
    result[f"bb_width_{bb_window}"] = bb_width

    result[f"cci_{cci_window}"] = cci(high_price, low_price, close_price, window=cci_window)
    result["fibonacci_range"] = fibonacci_range(open_price, high_price, low_price, close_price)
    ha_open, ha_high, ha_low, ha_close, ha_flag = heiken_ashi(open_price, high_price, low_price, close_price)
    result["ha_flag"] = ha_flag
    result["ha_body_size"] = np.abs(ha_close - ha_open)

    # Heiken Ashi Momentum
    result[f"ha_momentum_{ha_momentum_lookback}"] = heiken_ashi_momentum(open_price, high_price, low_price, close_price,
                                                                         lookback_period=ha_momentum_lookback)

    result[f"ha_momentum_adv_{ha_momentum_adv_lookback}_{ha_momentum_adv_smoothing}"] = \
        heiken_ashi_momentum_advanced(open_price, high_price, low_price, close_price,
                                      lookback_period=ha_momentum_adv_lookback, smoothing_period=ha_momentum_adv_smoothing)

    result[f"precision_trend_{precision_period}_{int(precision_sensitivity)}"] = (
        precision_trend(open_price, high_price, low_price, close_price, ptr_period=precision_period,
                        ptr_sensitivity=precision_sensitivity))

    result[f"ttm_trend_{ttm_period}"] = ttm_trend(open_price, high_price, low_price, close_price, inp_period=ttm_period)

    result[f"continuous_trend_{continuous_omega}".replace(".", "_")] = \
        continuous_trend_labeling(close_price, omega=continuous_omega)

    result[f"continuous_ma_trend_{continuous_ma_trend_window}_{continuous_ma_smooth_window}"] = \
        continuous_ma_trend_labeling(close_price, omega=continuous_ma_omega, trend_window=continuous_ma_trend_window,
                                     smooth_window=continuous_ma_smooth_window)

    result[f"trend_persistence_{persistence_window}_{persistence_smooth}"] = (
        trend_persistence_labeling(close_price, window=persistence_window, smooth=persistence_smooth,
                                   zscore_norm=persistence_zscore_norm))

    zscore_labels, zscore_values = zscore_trend_labeling(close_price, window=zscore_window,
                                                         z_threshold=zscore_z_threshold)
    result[f"zscore_trend_{zscore_window}"] = zscore_labels
    result[f"zscore_value_{zscore_window}"] = zscore_values
    return result


__all__ = [
    # Core feature generation function
    "core_trend_features",
    # Main trend labeling functions
    "continuous_trend_labeling",
    "continuous_ma_trend_labeling",
    "zscore_trend_labeling",
    "zscore_trend_features",
    "trend_persistence_labeling",
    # Optimization functions
    "find_optimal_continuous_trend_parameters",
    "determine_optimal_ma_omega",
    "optimize_zscore_trend_labeling_params",
    "optimize_trend_persistence",
    # Unified evaluation framework
    "evaluate_trend_performance",
    "compare_trend_methods",
    # Analysis helpers
    "analyze_optimal_ma_omega",
    "analyze_instrument_ma_omega",
    "compare_ma_omega_values",
    "optimize_zscore_params_multiasset",
    "compute_metrics",
    "simulate_strategy_returns",
    # Other trend indicators
    "bollinger_band",
    "cci",
    "fibonacci_range",
    "heiken_ashi",
    "heiken_ashi_momentum",
    "heiken_ashi_momentum_advanced",
    "precision_trend",
    "trading_the_trend",
    "ttm_trend",
]
