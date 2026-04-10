"""
Continuous Trend Labeling Methods

This module provides two complementary approaches for trend labeling without lookahead bias:

1. continuous_trend_labeling: Price action-based state machine approach
   - Tracks price extremes and reversals sequentially
   - More sensitive to sharp trend changes
   - Uses only omega threshold parameter

2. continuous_ma_trend_labeling: Moving average deviation approach
   - Based on smoothed deviation from moving average
   - Smoother, less sensitive to short-term noise
   - Uses omega, trend_window, and smooth_window parameters

Both methods are suitable for backtesting and real-time trading applications.

Reference: https://www.mdpi.com/1099-4300/22/10/1162

================================================================================
HELPER FUNCTIONS GUIDE
================================================================================

For continuous_trend_labeling (Price Action Approach):
-------------------------------------------------------

1. find_optimal_parameters(prices, candidate_omegas, candidate_lambdas, n_splits, random_state)
   USE WHEN: You need to find the best omega and lookback window (lambda) parameters for trend prediction using machine learning.
   DOES: Performs grid search with time series cross-validation to test different combinations of omega (trend threshold) and
         lambda (lookback window size). Uses logistic regression to predict future trend direction.
   RETURNS: Dict with best parameters, scores, and detailed CV results
   EXAMPLE: results = find_optimal_parameters(price_data,
                                              candidate_omegas=[0.10, 0.15, 0.20],
                                              candidate_lambdas=[5, 10, 15])


For continuous_ma_trend_labeling (Moving Average Approach):
------------------------------------------------------------

2. determine_optimal_ma_omega(price_series, trend_window, smooth_window, target_neutral_pct)
   USE WHEN: You want to find an omega value that produces a specific percentage of neutral labels (e.g., 40% neutral, 30% up, 30% down).
   DOES: Tests different omega values to find one that achieves your target neutral percentage.
            Useful for balancing trend vs neutral periods.
   RETURNS: Single optimal omega value (float)
   EXAMPLE: omega = determine_optimal_ma_omega(prices, target_neutral_pct=40)


3. analyze_optimal_ma_omega(price_series, trend_window, smooth_window, volatility_window, percentiles)
   USE WHEN: You need statistical analysis of deviation and volatility distributions to understand the data characteristics before choosing omega.
   DOES: Calculates percentiles of absolute smoothed deviations and volatility.
         Helps you understand the range of values in your data.
   RETURNS: Dict with deviation percentiles, volatility percentiles, and stats
   EXAMPLE: stats = analyze_optimal_ma_omega(prices)
            print(f"75th percentile deviation: {stats['deviation_percentiles'][75]}")


4. analyze_instrument_ma_omega(price_series, instrument_name, trend_window, smooth_window, volatility_window, target_neutral_pct)
   USE WHEN: You want a complete analysis package for a specific instrument including optimal omega, label distribution,
         and volatility statistics.
   DOES: Combines omega optimization, label generation, and statistical analysis into one comprehensive report. Great for instrument profiling.
   RETURNS: Dict with instrument name, optimal omega, label distribution, and volatility stats
   EXAMPLE: report = analyze_instrument_ma_omega(eurusd_prices, "EURUSD")
            print(f"Optimal omega: {report['optimal_omega']}")
            print(f"Uptrend: {report['label_distribution']['uptrend_pct']}%")


5. compare_ma_omega_values(price_series, omega_values, trend_window, smooth_window)
   USE WHEN: You want to see how different omega values affect label distribution to make an informed choice about which omega to use.
   DOES: Tests multiple omega values and returns a comparison table showing how each affects the percentage of uptrend, downtrend, and neutral labels.
   RETURNS: DataFrame with omega values and their resulting label distributions
   EXAMPLE: comparison = compare_ma_omega_values(prices, omega_values=[0.01, 0.02, 0.03])
            print(comparison)  # See how each omega splits up/down/neutral

================================================================================

Quick Decision Guide:
---------------------
- Need ML-based optimization? → Use find_optimal_parameters()
- Want balanced label distribution? → Use determine_optimal_ma_omega()
- Exploring data characteristics? → Use analyze_optimal_ma_omega()
- Need full instrument report? → Use analyze_instrument_ma_omega()
- Comparing multiple omegas? → Use compare_ma_omega_values()

================================================================================
OPTIMIZATION CRITERIA EXPLAINED
================================================================================

The optimization functions use DIFFERENT criteria to score parameters:

1. find_optimal_parameters() - PREDICTIVE ACCURACY CRITERION
   ---------------------------------------------------------
   GOAL: Find parameters that best predict future trend direction

   HOW IT SCORES:
   - Uses Logistic Regression to predict next trend label from past price windows
   - Scores by classification accuracy (% of correct predictions)
   - Uses time series cross-validation to prevent overfitting
   - Higher accuracy = better parameters

   FORMULA: score = accuracy_score(y_true, y_pred)
            where y_pred comes from ML model trained on past price patterns

   BEST FOR: Trading strategies where you want to predict trend changes

   EXAMPLE: omega=0.15, lambda=10 → 65% accuracy
            omega=0.20, lambda=15 → 68% accuracy ← BEST (highest accuracy)


2. determine_optimal_ma_omega() - LABEL DISTRIBUTION CRITERION
   ------------------------------------------------------------
   GOAL: Find omega that produces balanced label distribution

   HOW IT SCORES:
   - Calculates percentage of neutral, uptrend, and downtrend labels
   - Scores by distance from target neutral percentage (default 40%)
   - Lower distance = better omega

   FORMULA: score = abs(actual_neutral_pct - target_neutral_pct)
            Best omega minimizes this distance

   BEST FOR: Ensuring you don't get all trending (overtrading) or
            all neutral (undertrading) labels

   EXAMPLE: omega=0.01 → 10% neutral, 45% up, 45% down
            omega=0.02 → 38% neutral, 31% up, 31% down ← BEST (closest to 40%)
            omega=0.05 → 70% neutral, 15% up, 15% down

KEY INSIGHT: Function 1 optimizes for PREDICTION QUALITY
            Function 2 optimizes for LABEL BALANCE
            Use them for different purposes!

"""

from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ._trend_evaluation import evaluate_trend_performance


def continuous_trend_labeling(prices: Union[pd.Series | np.ndarray], omega=0.15):
    """
    Implements the Continuous Trend Labeling (CTL) method without look-ahead bias.
    Processes data sequentially, only using past information for utils.

    Parameters:
    -----------
    prices : pandas.Series or numpy.ndarray
        Input time series data (prices)
    omega : float, optional (default=0.15)
        Threshold parameter for trend detection (15% as in the paper)

    Returns:
    --------
    numpy.ndarray
        Array of trend: 1 for upward trends, -1 for downward trends, 0 for neutral
    """
    # Convert pandas Series to numpy array
    if isinstance(prices, pd.Series):
        x = prices.values
    else:
        x = np.asarray(prices)

    n = len(x)
    labels = np.zeros(n)  # Initialize trend as neutral (0)

    # Handle empty input
    if n == 0:
        return labels

    # Initialize variables (Algorithm 1 from paper)
    FP = x[0]  # First price
    xH = x[0]  # Current highest price
    HT = 0  # Time of highest price
    xL = x[0]  # Current lowest price
    LT = 0  # Time of lowest price
    Cid = 0  # Current direction (0=neutral, 1=up, -1=down)
    FP_N = 0  # Index of first significant move

    # First pass: Find initial trend direction (sequential)
    for i in range(n):
        if x[i] > FP + x[0] * omega:  # Upward threshold
            xH = x[i]
            HT = i
            FP_N = i
            Cid = 1
            break
        elif x[i] < FP - x[0] * omega:  # Downward threshold
            xL = x[i]
            LT = i
            FP_N = i
            Cid = -1
            break

    # If no significant trend found, return neutral trend
    if Cid == 0:
        return labels

    # Initialize the first segment (neutral until first significant move)
    labels[:FP_N] = 0

    # Second pass: Track trends and reversals (sequential, no look-ahead)
    for i in range(FP_N, n):
        if Cid == 1:  # Current upward trend
            if x[i] > xH:  # Update highest price
                xH = x[i]
                HT = i

            # Label current point as upward trend
            labels[i] = 1

            # Check for downward reversal (using only past information)
            if x[i] < xH - xH * omega and LT <= HT:
                # Switch to downward trend
                xL = x[i]
                LT = i
                Cid = -1
                # Label the reversal point as downward trend
                labels[i] = -1

        elif Cid == -1:  # Current downward trend
            if x[i] < xL:  # Update lowest price
                xL = x[i]
                LT = i

            # Label current point as downward trend
            labels[i] = -1

            # Check for upward reversal (using only past information)
            if x[i] > xL + xL * omega and HT <= LT:
                # Switch to upward trend
                xH = x[i]
                HT = i
                Cid = 1
                # Label the reversal point as upward trend
                labels[i] = 1

    return labels


def continuous_ma_trend_labeling(price_series: pd.Series, omega, trend_window=20, smooth_window=5):
    """
    Implements continuous trend labeling using moving average deviation method.

    This method calculates the deviation of price from its moving average, smooths the deviation,
    and labels trends based on whether the smoothed deviation exceeds the omega threshold.
    No lookahead bias - uses only backward-looking rolling windows.

    Args:
        price_series (pd.Series): Series with price data and datetime index
        omega (float): Fixed threshold parameter for trend detection
                      (e.g., 0.02 = 2% deviation triggers trend label)
        trend_window (int): Window size for moving average calculation (default: 20)
        smooth_window (int): Window size for smoothing deviation rate (default: 5)

    Returns:
        pd.Series: Trend labels with same index as input
                  - 1: Uptrend (smoothed_deviation > omega)
                  - -1: Downtrend (smoothed_deviation < -omega)
                  - 0: Neutral (|smoothed_deviation| <= omega)
                  - NaN: First (trend_window + smooth_window - 1) periods with insufficient data
    """
    if omega is None:
        raise ValueError("Omega must be provided for fixed trend")

    ma = price_series.rolling(window=trend_window, min_periods=1).mean()
    deviation_rate = (price_series - ma) / ma
    smoothed_deviation = deviation_rate.rolling(
        window=smooth_window, min_periods=1
    ).mean()

    labels = pd.Series(0, index=price_series.index)

    # Apply threshold rules to identify trends
    labels[smoothed_deviation > omega] = 1  # Uptrend
    labels[smoothed_deviation < -omega] = -1  # Downtrend

    min_periods = trend_window + smooth_window - 1
    labels.iloc[:min_periods] = np.nan

    return labels


##############################################################################################################
############################################# HELPER FUNCTIONS ###############################################
##############################################################################################################


def _evaluate_single_param_combination(omega, lambda_val, prices, n_splits):
    """
    Helper function to evaluate a single (omega, lambda) parameter combination.
    Used for parallel execution in grid search.

    Parameters:
    -----------
    omega : float
        Trend threshold parameter
    lambda_val : int
        Lookback window size
    prices : pd.Series
        Price series data
    n_splits : int
        Number of CV splits

    Returns:
    --------
    tuple : ((omega, lambda_val), score)
    """
    # Skip if lambda is too large for data
    if lambda_val >= len(prices):
        return (omega, lambda_val), 0.5

    # Generate trend labels with current omega
    labels = continuous_trend_labeling(prices, omega)

    # Evaluate prediction accuracy using logistic regression with time-series CV.
    # lambda_val is the lookback window for feature construction (past returns).
    metrics = evaluate_trend_performance(
        prices,
        labels,
        metrics=["prediction_accuracy"],
        lookback_window=lambda_val,
        cv_splits=n_splits,
    )

    score = metrics["prediction_accuracy"]

    # Handle NaN (e.g., from single-class labels)
    if np.isnan(score):
        score = 0.5

    return (omega, lambda_val), score


def find_optimal_continuous_trend_parameters(prices: Union[pd.Series | np.ndarray], candidate_omegas=None,
                                             candidate_lambdas=None, n_splits=5, random_state=42, n_jobs=-1):
    """
    Finds optimal omega and lambda parameters using grid search with time series cross-validation.

    Now uses the unified evaluation framework for consistent prediction accuracy calculation.
    This function tests different combinations of omega (trend threshold) and lambda (lookback window)
    parameters to find the optimal values for trend prediction using logistic regression.

    Supports parallel execution for significant speedup on multi-core systems.

    Parameters:
    -----------
    prices : pandas.Series or numpy.ndarray
        Input time series data (prices)
    candidate_omegas : list or array, optional
        Candidate omega (threshold) values to test (default: [0.05, 0.10, 0.15, 0.20, 0.25])
    candidate_lambdas : list, optional
        Candidate lambda (lookback window size) values to test (default: [5, 7, 9, 11, 13, 15, 17, 21])
        Lambda determines how many past price points are used as features for prediction
    n_splits : int, optional
        Number of time series cross-validation splits (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    n_jobs : int, optional
        Number of parallel jobs to run (default: -1, uses all CPU cores)
        Set to 1 for sequential execution

    Returns:
    --------
    dict
        Dictionary containing:
        - 'best_omega': Optimal omega value
        - 'best_lambda': Optimal lambda value
        - 'best_score': Best cross-validation accuracy score
        - 'all_scores': Dict mapping (omega, lambda) tuples to mean scores
    """
    # Convert to Series if needed
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    # Set default candidate parameters
    if candidate_omegas is None:
        candidate_omegas = np.arange(0.05, 0.3, 0.05)
    if candidate_lambdas is None:
        candidate_lambdas = [5, 7, 9, 11, 13, 15, 17, 21]

    # Generate all parameter combinations
    param_combinations = [
        (omega, lambda_val)
        for omega in candidate_omegas
        for lambda_val in candidate_lambdas
    ]

    # Parallel grid search
    print(
        f"Testing {len(param_combinations)} parameter combinations using {n_jobs if n_jobs > 0 else 'all'} CPU cores..."
    )

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_evaluate_single_param_combination)(omega, lambda_val, prices, n_splits)
        for omega, lambda_val in tqdm(param_combinations, desc="Grid search progress")
    )

    # Convert results to dictionary
    param_scores = dict(results)

    # Find best parameters
    best_params = max(param_scores, key=param_scores.get)
    best_omega, best_lambda = best_params
    best_score = param_scores[best_params]

    return {
        "best_omega": best_omega,
        "best_lambda": best_lambda,
        "best_score": best_score,
        "all_scores": param_scores,
    }


def determine_optimal_ma_omega(price_series: pd.Series, trend_window=20, smooth_window=5, target_neutral_pct=40):
    """
    Determine the optimal fixed omega value for an instrument based on historical data.

    Args:
        price_series (pd.Series): Series with price data and datetime index
        trend_window (int): Window size for trend calculation (default: 20)
        smooth_window (int): Window size for smoothing (default: 5)
        target_neutral_pct (float): Target percentage of neutral trend (default: 40)

    Returns:
        float: Optimal fixed omega value
    """
    ma = price_series.rolling(window=trend_window, min_periods=1).mean()
    deviation_rate = (price_series - ma) / ma
    smoothed_deviation = deviation_rate.rolling(window=smooth_window, min_periods=1).mean()

    # Get the distribution of absolute smoothed deviations
    abs_smoothed_deviation = smoothed_deviation.abs().dropna()

    # Find the omega that gives us approximately target_neutral_pct neutral trend
    # We'll test a range of percentiles and find the one closest to our target
    percentiles = np.arange(50, 95, 1)
    best_omega = None
    best_diff = float("inf")

    for p in percentiles:
        test_omega = abs_smoothed_deviation.quantile(p / 100)

        # Calculate trend with this omega
        test_labels = pd.Series(0, index=price_series.index)
        test_labels[smoothed_deviation > test_omega] = 1
        test_labels[smoothed_deviation < -test_omega] = -1

        # Calculate neutral percentage
        label_counts = test_labels.value_counts(dropna=True)
        total_labels = label_counts.sum()
        neutral_pct = label_counts.get(0, 0) / total_labels * 100

        # Check how close we are to target
        diff = abs(neutral_pct - target_neutral_pct)

        if diff < best_diff:
            best_diff = diff
            best_omega = test_omega

    return best_omega


def analyze_optimal_ma_omega(price_series: pd.Series, trend_window=20, smooth_window=5, volatility_window=20,
                             percentiles=None):
    """
    Analyze the distribution of smoothed deviations to help determine an optimal omega.

    Args:
        price_series (pd.Series): Series with price data and datetime index
        trend_window (int): Window size for trend calculation (default: 20)
        smooth_window (int): Window size for smoothing (default: 5)
        volatility_window (int): Window for volatility calculation (default: 20)
        percentiles (list): Percentiles to analyze (default: [50, 60, 70, 75, 80, 85, 90, 95])

    Returns:
        dict: Analysis results including recommended omega values
    """
    if percentiles is None:
        percentiles = [50, 60, 70, 75, 80, 85, 90, 95]

    # Calculate moving average (trend indicator)
    ma = price_series.rolling(window=trend_window, min_periods=1).mean()

    # Calculate mean deviation rate
    deviation_rate = (price_series - ma) / ma

    # Smooth the deviation rate
    smoothed_deviation = deviation_rate.rolling(
        window=smooth_window, min_periods=1
    ).mean()

    # Calculate volatility
    returns = price_series.pct_change().dropna()
    volatility = returns.rolling(window=volatility_window, min_periods=1).std()

    # Get percentiles of absolute smoothed deviation
    abs_smoothed_deviation = smoothed_deviation.abs().dropna()
    deviation_percentiles = {
        p: abs_smoothed_deviation.quantile(p / 100) for p in percentiles
    }

    # Get percentiles of volatility
    volatility_percentiles = {p: volatility.quantile(p / 100) for p in percentiles}

    return {
        "deviation_percentiles": deviation_percentiles,
        "volatility_percentiles": volatility_percentiles,
        "mean_abs_deviation": abs_smoothed_deviation.mean(),
        "median_abs_deviation": abs_smoothed_deviation.median(),
        "std_abs_deviation": abs_smoothed_deviation.std(),
    }


def analyze_instrument_ma_omega(price_series: pd.Series, instrument_name, trend_window=20, smooth_window=5,
                                volatility_window=20, target_neutral_pct=40):
    """
    Complete analysis to determine the optimal omega for an instrument.

    Args:
        price_series (pd.Series): Series with price data and datetime index
        instrument_name (str): Name of the instrument for reporting
        trend_window (int): Window size for trend calculation (default: 20)
        smooth_window (int): Window size for smoothing (default: 5)
        volatility_window (int): Window for volatility calculation (default: 20)
        target_neutral_pct (float): Target percentage of neutral trend (default: 40)

    Returns:
        dict: Analysis results including optimal omega
    """
    # Determine optimal omega
    optimal_omega = determine_optimal_ma_omega(
        price_series,
        trend_window=trend_window,
        smooth_window=smooth_window,
        target_neutral_pct=target_neutral_pct,
    )

    # Generate trend with optimal omega
    labels = continuous_ma_trend_labeling(
        price_series,
        omega=optimal_omega,
        trend_window=trend_window,
        smooth_window=smooth_window,
    )

    # Calculate label distribution
    label_counts = labels.value_counts(dropna=True)
    total_labels = label_counts.sum()
    label_percentages = (label_counts / total_labels * 100).round(2)

    # Calculate volatility statistics
    returns = price_series.pct_change().dropna()
    volatility = returns.rolling(window=volatility_window, min_periods=1).std()

    return {
        "instrument_name": instrument_name,
        "optimal_omega": optimal_omega,
        "label_distribution": {
            "uptrend_count": label_counts.get(1, 0),
            "downtrend_count": label_counts.get(-1, 0),
            "neutral_count": label_counts.get(0, 0),
            "uptrend_pct": label_percentages.get(1, 0),
            "downtrend_pct": label_percentages.get(-1, 0),
            "neutral_pct": label_percentages.get(0, 0),
        },
        "volatility_stats": {
            "mean": volatility.mean(),
            "median": volatility.median(),
            "std": volatility.std(),
            "min": volatility.min(),
            "max": volatility.max(),
        },
    }


def compare_ma_omega_values(price_series: pd.Series, omega_values=None, trend_window=20, smooth_window=5):
    """
    Compare the impact of different omega values on label distribution.

    Args:
        price_series (pd.Series): Series with price data and datetime index
        omega_values (list): Omega values to test (default: [0.005, 0.01, 0.02, 0.03, 0.05, 0.1])
        trend_window (int): Window size for trend calculation (default: 20)
        smooth_window (int): Window size for smoothing (default: 5)

    Returns:
        pd.DataFrame: Comparison of label distributions for different omega values
    """
    if omega_values is None:
        omega_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

    results = []

    for omega in omega_values:
        labels = continuous_ma_trend_labeling(
            price_series,
            omega=omega,
            trend_window=trend_window,
            smooth_window=smooth_window,
        )

        # Count trend (excluding NaN)
        label_counts = labels.value_counts(dropna=True)
        total_labels = label_counts.sum()

        # Calculate percentages
        label_percentages = (label_counts / total_labels * 100).round(2)

        # Store results
        result = {
            "omega": omega,
            "uptrend_count": label_counts.get(1, 0),
            "downtrend_count": label_counts.get(-1, 0),
            "neutral_count": label_counts.get(0, 0),
            "uptrend_pct": label_percentages.get(1, 0),
            "downtrend_pct": label_percentages.get(-1, 0),
            "neutral_pct": label_percentages.get(0, 0),
        }

        results.append(result)
    return pd.DataFrame(results)
