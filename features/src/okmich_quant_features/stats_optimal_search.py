import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from .path_structure import variance_ratio, auto_corr


def optimal_autocorrelation_param_search(series, window_range, lag_range, objective_metric="absolute_autocorr",
                                         **objective_kwargs):
    """
    Optimize window and lag for rolling autocorrelation using a specified objective metric.

    Parameters:
    - series: pandas Series, the input time series (e.g., returns)
    - window_range: list, range of window sizes to test
    - lag_range: list, range of lags to test
    - objective_metric: str - objective metric to optimize - "absolute_autocorr", "trading_profit", "stat_significance"

    Returns:
    - dict, best parameters and corresponding score
    """

    def stat_significance_metric(autocorr, series, lag, window, p_threshold=0.05):
        """
        Objective: Maximize proportion of significant autocorrelations (Ljung-Box test).
        Uses the same rolling window size as the autocorrelation computation.
        """

        def ljung_box_pvalue(x):
            if len(x) < lag + 1 or np.std(x) == 0:
                return np.nan
            return acorr_ljungbox(x, lags=[lag], return_df=False)[1][0]

        p_values = series.rolling(window).apply(ljung_box_pvalue, raw=True)
        significant = (p_values < p_threshold).mean()
        return significant

    def absolute_autocorr_metric(autocorr):
        """
        Objective: Maximize the mean absolute autocorrelation.
        """
        return np.abs(autocorr).mean()

    def trading_profit_metric(autocorr, series, threshold=0.1, cost=0.001):
        """
        Objective: Maximize trading profit based on autocorrelation signals.
        - Buy when autocorr > threshold, sell when autocorr < -threshold.
        - Assumes series is returns; includes transaction cost.
        """
        signals = np.where(
            autocorr > threshold, 1, np.where(autocorr < -threshold, -1, 0)
        )
        signals = pd.Series(signals, index=autocorr.index)
        # Shift signals to avoid look-ahead bias (trade on next period's return)
        trades = signals.shift(1).fillna(0)
        # Calculate returns (multiply signals by next period's returns)
        returns = series * trades
        # Subtract transaction costs for position changes
        position_changes = np.abs(signals.diff().fillna(0))
        costs = cost * position_changes
        total_profit = (returns - costs).sum()
        return total_profit

    # Map string metrics to objective functions
    objective_functions = {
        "absolute_autocorr": absolute_autocorr_metric,
        "trading_profit": trading_profit_metric,
        "stat_significance": stat_significance_metric,
    }

    if objective_metric not in objective_functions:
        raise ValueError(
            f"Unknown objective_metric '{objective_metric}'. "
            f"Allowed values: {sorted(objective_functions)}"
        )
    objective_func = objective_functions[objective_metric]

    best_score = -np.inf
    best_params = None

    for window in window_range:
        for lag in lag_range:
            # Compute rolling autocorrelation
            autocorr = auto_corr(series, window=window, lag=lag)

            # Compute objective score
            if objective_metric == "stat_significance":
                score = objective_func(
                    autocorr, series=series, lag=lag, window=window, **objective_kwargs
                )
            elif objective_metric == "trading_profit":
                score = objective_func(autocorr, series=series, **objective_kwargs)
            else:
                score = objective_func(autocorr, **objective_kwargs)

            if score > best_score and not np.isnan(score):
                best_score = score
                best_params = {"window": window, "lag": lag}

    return best_params, best_score


def optimal_variance_ratio_param_search(series, window_range, q_range, objective_metric="max_deviation", **objective_kwargs):
    """
    Optimize window and q for rolling variance ratio using a specified objective metric.

    Parameters:
    - series: pandas Series, the input time series (e.g., returns)
    - window_range: list, range of window sizes to test
    - q_range: list, range of aggregation periods to test
    - objective_metric: str - objective metric to optimize ("max_deviation", "trading_profit" or "stat_significance")
    - objective_kwargs: dict, additional arguments for objective functions (e.g., thresholds for trading)

    Returns:
    - dict, best parameters and corresponding score
    """

    # Objective functions
    def max_deviation_metric(vr, series=None):
        """
        Objective: Maximize the mean absolute deviation of VR from 1.
        """
        return np.abs(vr - 1).mean()

    def trading_profit_metric(vr, series, buy_threshold=0.8, sell_threshold=1.2, cost=0.001):
        """
        Objective: Maximize trading profit based on VR signals.
        - Buy when VR < buy_threshold (mean reversion), sell when VR > sell_threshold (momentum).
        - Assumes series is returns; includes transaction cost.
        """
        signals = np.where(vr < buy_threshold, 1, np.where(vr > sell_threshold, -1, 0))
        signals = pd.Series(signals, index=vr.index)
        trades = signals.shift(1).fillna(0)
        # Calculate returns
        returns = series * trades
        position_changes = np.abs(signals.diff().fillna(0))
        costs = cost * position_changes
        total_profit = (returns - costs).sum()
        return total_profit

    def stat_significance_metric(vr, series, q, p_threshold=0.05):
        """
        Objective: Maximize proportion of significant VR deviations from 1.
        Uses a simplified test based on VR confidence intervals.
        """
        se = np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * len(vr)))
        z_scores = np.abs(vr - 1) / se
        significant = (z_scores > 1.96).mean()
        return significant

    # Map string metrics to objective functions
    objective_functions = {
        "max_deviation": max_deviation_metric,
        "trading_profit": trading_profit_metric,
        "stat_significance": stat_significance_metric,
    }

    if objective_metric not in objective_functions:
        raise ValueError(
            f"Unknown objective_metric '{objective_metric}'. "
            f"Allowed values: {sorted(objective_functions)}"
        )
    objective_func = objective_functions[objective_metric]
    best_score = -np.inf
    best_params = None

    for window in window_range:
        for q in q_range:
            # Compute rolling variance ratio
            vr = variance_ratio(series, window=window, q=q)

            # Compute objective score
            if objective_metric == "trading_profit":
                score = objective_func(vr, series=series, **objective_kwargs)
            elif objective_metric == "stat_significance":
                score = objective_func(vr, series=series, q=q, **objective_kwargs)
            else:
                score = objective_func(vr, **objective_kwargs)

            if score > best_score and not np.isnan(score):
                best_score = score
                best_params = {"window": window, "q": q}

    return best_params, best_score
