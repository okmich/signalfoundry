import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Optional, Dict, List, Union


def _compute_returns(prices: Union[pd.Series, pd.DataFrame], return_col: str = "close") -> pd.Series:
    if isinstance(prices, pd.DataFrame):
        price_series = prices[return_col]
    else:
        price_series = prices

    return np.log(price_series / price_series.shift(1))


def _simulate_strategy_returns(trend_labels: pd.Series, returns: pd.Series) -> pd.Series:
    return trend_labels.shift(1) * returns


def _calculate_prediction_accuracy(prices: pd.Series, trend_labels: pd.Series, lookback_window: int = 20,
                                   n_splits: int = 5) -> float:
    # Create features: past returns
    returns = _compute_returns(prices)

    # Create lagged features
    features = pd.DataFrame(index=returns.index)
    for i in range(1, lookback_window + 1):
        features[f"ret_lag_{i}"] = returns.shift(i)

    # Target: next period trend (avoid lookahead)
    target = trend_labels.shift(-1)

    # Remove NaN rows
    valid_idx = features.notna().all(axis=1) & target.notna()
    X = features[valid_idx].values
    y = target[valid_idx].values

    if len(X) < n_splits * 2:
        return np.nan

    # Check if we have at least 2 classes
    if len(np.unique(y)) < 2:
        return np.nan

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle single-class cases
        if len(np.unique(y_train)) < 2:
            # Can't train on single class, use baseline
            scores.append(0.5)
            continue

        if len(np.unique(y_test)) < 2:
            # Single class in test set, use majority baseline
            y_pred = np.full_like(y_test, y_train[0])
            scores.append(accuracy_score(y_test, y_pred))
            continue

        try:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        except (ValueError, np.linalg.LinAlgError):
            # Handle convergence issues or other errors
            scores.append(0.5)

    return np.mean(scores) if scores else np.nan


def _calculate_label_distribution(trend_labels: pd.Series) -> Dict[str, float]:
    counts = trend_labels.value_counts(normalize=True) * 100
    return {
        "pct_uptrend": counts.get(1, 0.0),
        "pct_downtrend": counts.get(-1, 0.0),
        "pct_neutral": counts.get(0, 0.0),
        "balance_score": 100 - abs(counts.get(1, 0.0) - counts.get(-1, 0.0)),
    }


def _calculate_sharpe_ratio(
    returns: pd.Series, annualization_factor: float = 252
) -> float:
    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0 or np.isnan(std_return):
        return 0.0

    return (mean_return / std_return) * np.sqrt(annualization_factor)


def _calculate_cumulative_returns(returns: pd.Series) -> float:
    return (1 + returns).prod() - 1


def _calculate_win_rate(returns: pd.Series) -> float:
    return (returns > 0).mean()


def _calculate_persistence(trend_labels: pd.Series) -> float:
    changes = (trend_labels.diff() != 0).sum()
    total_periods = len(trend_labels)

    if total_periods <= 1:
        return 1.0

    return 1 - (changes / (total_periods - 1))


def _calculate_num_trades(trend_labels: pd.Series) -> int:
    return (trend_labels.diff().fillna(0) != 0).sum()


def _calculate_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def evaluate_trend_performance(prices: Union[pd.Series, pd.DataFrame], trend_labels: pd.Series,
                               metrics: Optional[List[str]] = None, return_col: str = "close",
                               annualization_factor: float = 252, lookback_window: int = 20, cv_splits: int = 5,) -> Dict[str, Union[float, int, Dict]]:
    """
    Unified evaluation framework for trend labeling methods.

    Computes standardized metrics to enable fair comparison across different trend identification approaches.

    Parameters:
    -----------
    prices : pd.Series or pd.DataFrame
        Price data (close prices or OHLC)
    trend_labels : pd.Series or np.ndarray
        Trend labels (typically 1=uptrend, -1=downtrend, 0=neutral)
        Can be numpy array or pandas Series
    metrics : list of str or None
        Specific metrics to calculate. If None, calculates ALL metrics.
        Available: ['prediction_accuracy', 'label_distribution', 'sharpe_ratio',
                   'cumulative_returns', 'win_rate', 'persistence', 'num_trades',
                   'max_drawdown']
    return_col : str
        Column name for returns calculation (if DataFrame)
    annualization_factor : float
        Factor to annualize Sharpe ratio (252=daily, 12=monthly)
    lookback_window : int
        Lookback window for prediction accuracy calculation
    cv_splits : int
        Number of CV splits for prediction accuracy

    Returns:
    --------
    dict : Dictionary containing requested metrics

    Examples:
    ---------
    # Fast - during optimization
    >>> result = evaluate_trend_performance(prices, labels,
    ...                                     metrics=['sharpe_ratio', 'win_rate'])
    >>> print(result['sharpe_ratio'])

    # Comprehensive - for comparison
    >>> result = evaluate_trend_performance(prices, labels, metrics=None)
    >>> print(result)  # All metrics included
    """
    # If metrics is None, calculate everything
    if metrics is None:
        metrics = [
            "prediction_accuracy",
            "label_distribution",
            "sharpe_ratio",
            "cumulative_returns",
            "win_rate",
            "persistence",
            "num_trades",
            "max_drawdown",
        ]

    # Handle both numpy array and pandas Series for trend_labels
    if isinstance(trend_labels, np.ndarray):
        # Convert numpy array to Series with same index as prices
        if isinstance(prices, pd.DataFrame):
            trend_labels = pd.Series(trend_labels, index=prices.index)
        elif isinstance(prices, pd.Series):
            trend_labels = pd.Series(trend_labels, index=prices.index)
        else:
            # Both are arrays, create default index
            trend_labels = pd.Series(trend_labels)

    # Align data
    aligned = pd.DataFrame({"labels": trend_labels})

    if isinstance(prices, pd.DataFrame):
        aligned["prices"] = prices[return_col]
    else:
        aligned["prices"] = prices

    # Drop NaN
    aligned = aligned.dropna()

    if len(aligned) == 0:
        return {metric: np.nan for metric in metrics}

    # Compute returns
    returns = _compute_returns(aligned["prices"])

    # Simulate strategy returns (shift labels to avoid lookahead)
    strategy_returns = _simulate_strategy_returns(aligned["labels"], returns)
    strategy_returns = strategy_returns.dropna()

    # Calculate requested metrics
    results = {}

    for metric in metrics:
        if metric == "prediction_accuracy":
            results["prediction_accuracy"] = _calculate_prediction_accuracy(
                aligned["prices"],
                aligned["labels"],
                lookback_window=lookback_window,
                n_splits=cv_splits,
            )

        elif metric == "label_distribution":
            results["label_distribution"] = _calculate_label_distribution(
                aligned["labels"]
            )

        elif metric == "sharpe_ratio":
            results["sharpe_ratio"] = _calculate_sharpe_ratio(
                strategy_returns, annualization_factor=annualization_factor
            )

        elif metric == "cumulative_returns":
            results["cumulative_returns"] = _calculate_cumulative_returns(
                strategy_returns
            )

        elif metric == "win_rate":
            results["win_rate"] = _calculate_win_rate(strategy_returns)

        elif metric == "persistence":
            results["persistence"] = _calculate_persistence(aligned["labels"])

        elif metric == "num_trades":
            results["num_trades"] = _calculate_num_trades(aligned["labels"])

        elif metric == "max_drawdown":
            results["max_drawdown"] = _calculate_max_drawdown(strategy_returns)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


def compare_trend_methods(prices: Union[pd.Series, pd.DataFrame], methods_config: Dict[str, pd.Series],
                          return_col: str = "close", **kwargs) -> pd.DataFrame:
    """
    Compare multiple trend methods with comprehensive metrics.

    Parameters:
    -----------
    prices : pd.Series or pd.DataFrame
        Price data
    methods_config : dict
        Dictionary mapping method names to their trend labels
        Example: {'continuous': labels1, 'zscore': labels2, ...}
    return_col : str
        Column name for returns (if DataFrame)
    **kwargs : additional arguments
        Passed to evaluate_trend_performance

    Returns:
    --------
    pd.DataFrame : Comparison table with all metrics for each method

    Example:
    --------
    >>> comparison = compare_trend_methods(prices, {
    ...     'continuous_price': continuous_labels,
    ...     'continuous_ma': ma_labels,
    ...     'zscore': zscore_labels,
    ...     'trend_persistence': persistence_labels
    ... })
    >>> print(comparison)
    """
    results = []

    for method_name, trend_labels in methods_config.items():
        # Calculate all metrics for this method
        metrics = evaluate_trend_performance(
            prices=prices,
            trend_labels=trend_labels,
            metrics=None,  # Calculate everything
            return_col=return_col,
            **kwargs,
        )

        # Flatten label_distribution if present
        flat_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_metrics[sub_key] = sub_value
            else:
                flat_metrics[key] = value

        flat_metrics["method"] = method_name
        results.append(flat_metrics)

    df = pd.DataFrame(results)

    # Reorder columns: method first, then metrics
    if "method" in df.columns:
        cols = ["method"] + [c for c in df.columns if c != "method"]
        df = df[cols]

    return df
