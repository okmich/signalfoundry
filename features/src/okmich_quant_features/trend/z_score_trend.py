"""
Z-Score Trend Exhaustion / Reversal Feature

This module computes a z-score of log returns relative to their rolling statistics.
The z-score value itself is the primary output — it measures how statistically extreme the current price move is relative to recent history.

Role in the directional labeling framework:
- This is NOT a directional trend labeller. It does not answer "which way is the trend?"
- It is a trend exhaustion / reversal signal that answers "is the current move overextended?"
- Use the continuous z-score value as a feature, not the discretized {-1, 0, 1} labels
- The discrete labels are retained for backward compatibility but should not be used in directional ensembles alongside CTL, CTL-MA, and Trend Persistence

Empirical findings (Phase 2 baseline comparison, FXPIG 5-min, 27 symbols):
- As a directional labeller: only 22% of symbols profitable, mean PF=0.95
- Z-thresholds converge to 2.5+ because lower thresholds exceed MAX_REGIME_COUNT and higher thresholds leave ~97% of bars in neutral
- Structurally a mean-reversion signal: high z-score = "price far from mean" = reversal setup
- Orthogonal to all directional labelers (pairwise kappa ~0)

Recommended usage:
- zscore_trend_features(): returns continuous z-score + rolling derivative as features
- zscore_trend_labeling(): retained for backward compatibility, returns discrete labels + z-score

"""

from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ndarray, dtype, float64
from skopt import gp_minimize
from skopt.space import Integer, Real

from ._trend_evaluation import evaluate_trend_performance


def zscore_trend_labeling(price_series: pd.Series, window=30, z_threshold=1.5) -> \
        tuple[Any, ndarray[tuple[int, ...], dtype[float64]] | float | Any]:
    """
    Z-Score Trend Labeling for market regimes using symmetric thresholds.

    Parameters:
    -----------
    price_series : pd.Series
        Price series data, preferably with datetime index.
    window : int
        Rolling window size for Z-score computation.
    z_threshold : float
        Z-score threshold for strong momentum regime.
        Uses symmetric thresholds: +z_threshold for uptrend, -z_threshold for downtrend.

    Returns:
    --------
    regime_labels : pd.Series
        Regime labels: 1 for strong uptrend, -1 for strong downtrend, 0 for neutral/mean-reversion zone
    zscore : pd.Series
        The computed z-score values

    Notes:
    ------
    - Uses symmetric thresholds for balanced regime classification
    - Returns are log returns: ln(price_t / price_t-1)
    - Z-score = (returns - rolling_mean) / rolling_std
    """
    returns = np.log(price_series / price_series.shift())
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    zscore = (returns - rolling_mean) / (rolling_std + 1e-8)

    # Vectorized regime classification with symmetric thresholds
    regime_labels = pd.Series(np.where(zscore > z_threshold, 1, np.where(zscore < -z_threshold, -1, 0)), index=price_series.index)
    return regime_labels, zscore


def zscore_trend_features(price_series: pd.Series, window: int = 30, deriv_window: int = 5) -> pd.DataFrame:
    """
    Compute continuous z-score features for trend exhaustion / reversal detection.

    Unlike zscore_trend_labeling which discretizes into {-1, 0, 1}, this returns the
    raw z-score and its rolling derivative as continuous features suitable for ML models.

    Parameters
    ----------
    price_series : pd.Series
        Price series data.
    window : int
        Rolling window for z-score computation.
    deriv_window : int
        Rolling window for computing the z-score's rate of change.

    Returns
    -------
    pd.DataFrame with columns:
        - zscore: raw z-score value (how extreme the current move is)
        - zscore_deriv: rolling mean of z-score changes (acceleration/deceleration of extremity)
        - zscore_abs: absolute z-score (magnitude of extremity, direction-agnostic)
    """
    returns = np.log(price_series / price_series.shift())
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    zscore = (returns - rolling_mean) / (rolling_std + 1e-8)

    result = pd.DataFrame(index=price_series.index)
    result[f"zscore_{window}"] = zscore
    result[f"zscore_deriv_{window}"] = zscore.diff().rolling(window=deriv_window).mean()
    result[f"zscore_abs_{window}"] = zscore.abs()
    return result


##############################################################################################################
############################################# HELPER FUNCTIONS ###############################################
##############################################################################################################


def simulate_strategy_returns(df, regime_col="label", return_col="returns"):
    """
    Legacy function for backward compatibility.
    Simulates strategy returns by shifting labels to avoid lookahead bias.

    NOTE: Consider using evaluate_trend_performance() for more comprehensive metrics.
    """
    df["strategy_returns"] = df[regime_col].shift(1) * df[return_col]
    return df["strategy_returns"]


def compute_metrics(strategy_returns, regimes, weights=None, min_trades=10):
    """
    Compute composite score from multiple metrics.

    This function is now a wrapper around the unified evaluation framework.
    For backward compatibility, it maintains the same interface and composite score calculation.

    Parameters:
    -----------
    strategy_returns : pd.Series
        Strategy returns (not used directly, but kept for compatibility)
    regimes : pd.Series
        Regime labels
    weights : dict
        Weights for each metric in composite score
    min_trades : int
        Minimum number of trades required

    Returns:
    --------
    composite_score : float
        Weighted combination of metrics
    metrics : dict
        Individual metric values
    """
    if weights is None:
        weights = {"sharpe": 1, "win_rate": 1, "persistence": 1, "num_trades": 1}

    # Use unified evaluation framework - note: we need price data for this
    # For now, maintain the original calculation for exact backward compatibility
    # TODO: In future, could refactor callers to pass prices and use evaluate_trend_performance

    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)
    win_rate = (strategy_returns > 0).mean()
    persistence = np.mean(regimes.diff().fillna(0) == 0)
    num_trades = (np.sign(regimes).diff().fillna(0) != 0).sum()

    # Penalize if too few trades
    trade_penalty = 0 if num_trades >= min_trades else -1

    metrics = {
        "sharpe": sharpe,
        "win_rate": win_rate,
        "persistence": persistence,
        "num_trades": num_trades,
    }
    composite_score = (
        sharpe * weights["sharpe"]
        + win_rate * weights["win_rate"]
        + persistence * weights["persistence"]
        + (np.log1p(num_trades) * weights["num_trades"])  # scaled weight
        + trade_penalty
    )
    return composite_score, metrics


def optimization_objective_for_params(params, price_series, weights, min_trades=10):
    """
    Objective function for Bayesian optimization.

    Now uses unified evaluation framework for consistent metric calculation.

    Parameters:
    -----------
    params : tuple
        (window, z_threshold) parameter values to test
    price_series : pd.Series
        Price data
    weights : dict
        Weights for composite score calculation
    min_trades : int
        Minimum number of trades required

    Returns:
    --------
    float : Negative composite score (for minimization)
    """
    window, z_threshold = params
    labels, _ = zscore_trend_labeling(
        price_series, window=window, z_threshold=z_threshold
    )

    # Use unified evaluation framework - compute only needed metrics for speed
    metrics = evaluate_trend_performance(
        price_series,
        labels,
        metrics=["sharpe_ratio", "win_rate", "persistence", "num_trades"],
    )

    # Calculate composite score using z-score specific formula
    sharpe = metrics["sharpe_ratio"]
    win_rate = metrics["win_rate"]
    persistence = metrics["persistence"]
    num_trades = metrics["num_trades"]

    # Penalize if too few trades
    trade_penalty = 0 if num_trades >= min_trades else -1

    composite_score = (
        sharpe * weights["sharpe"]
        + win_rate * weights["win_rate"]
        + persistence * weights["persistence"]
        + (np.log1p(num_trades) * weights["num_trades"])  # scaled weight
        + trade_penalty
    )

    return -composite_score  # Negative for minimization


def optimize_zscore_trend_labeling_params(
    price_series, window_range=(5, 120), z_range=(0.5, 4.0), weights=None, min_trades=10
):
    """
    Optimize z-score trend labeling parameters using Bayesian optimization.

    Uses the unified evaluation framework for consistent metric calculation.
    Optimizes a composite score combining Sharpe ratio, win rate, persistence, and trade frequency.

    Parameters:
    -----------
    price_series : pd.Series
        Price series to optimize on
    window_range : tuple
        (min, max) range for window parameter
    z_range : tuple
        (min, max) range for z_threshold parameter
    weights : dict
        Weights for composite score: {'sharpe', 'win_rate', 'persistence', 'num_trades'}
    min_trades : int
        Minimum number of trades required (default 10)

    Returns:
    --------
    dict : Best parameters and score with keys 'window', 'z_threshold', 'best_score'
    """
    if weights is None:
        weights = {"sharpe": 1, "win_rate": 1, "persistence": 1, "num_trades": 1}

    space = [
        Integer(window_range[0], window_range[1], name="window"),
        Real(z_range[0], z_range[1], name="z_threshold"),
    ]

    res = gp_minimize(
        lambda params: optimization_objective_for_params(
            params, price_series, weights, min_trades
        ),
        dimensions=space,
        n_calls=50,
        n_initial_points=10,
        random_state=42,
        n_jobs=-1,
    )
    best_params = {"window": res.x[0], "z_threshold": res.x[1], "best_score": -res.fun}
    return best_params


def optimize_zscore_params_multiasset(
    tickers, data_source_fun, weights=None, n_jobs=-1
):
    """
    Perform parameter optimization across multiple assets.

    :param tickers:
    :param data_source_fun: a callable that takes tickers as input and returns a pandas ohlcv DataFrame
    :param weights:
    :param n_jobs:
    :return:
        pd.DataFrame - a dataframe containing z-score scores for each asset.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(
            lambda t: {
                "ticker": t,
                **optimize_zscore_trend_labeling_params(
                    data_source_fun(t), weights=weights
                ),
            }
        )(ticker)
        for ticker in tickers
    )
    return pd.DataFrame(results)
