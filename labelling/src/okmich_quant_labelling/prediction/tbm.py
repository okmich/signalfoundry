from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed

from okmich_quant_features.volatility import atr, parkinson_volatility, garman_klass_volatility, rolling_volatility


class VolatilityEstimator(Enum):
    STD = "std"
    ATR = "atr"
    PARKINSON = "parkinson"
    GARMAN_KLASS = "garman_klass"


@dataclass
class VolatilityConfig:
    """Configuration for volatility estimation."""
    estimator: VolatilityEstimator = VolatilityEstimator.ATR
    params: dict = field(default_factory=lambda: {"window": 20})


@dataclass
class BarrierConfig:
    """Configuration for barrier construction."""
    upper_multiplier: float = 2.0
    lower_multiplier: float = 2.0
    max_holding_bars: int = 10
    vertical_as_zero: bool = False


@dataclass
class TBMConfig:
    """Combined configuration for Triple Barrier Method."""
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    barrier: BarrierConfig = field(default_factory=BarrierConfig)


class OptimizationMetric(Enum):
    """Metrics for TBM parameter optimization."""
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return"
    AVG_RETURN = "avg_return"
    VERTICAL_RATE = "vertical_rate"  # Lower is better


def compute_volatility(prices: pd.DataFrame, config: VolatilityConfig) -> np.ndarray:
    """
    Compute volatility series based on configuration.

    Uses optimized implementations from okmich_quant_features.volatility.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC DataFrame with columns: open, high, low, close
    config : VolatilityConfig
        Volatility estimation configuration

    Returns
    -------
    np.ndarray
        Volatility estimates in price units, aligned with price index
    """
    window = config.params.get("window", 20)

    open_ = prices["open"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    close = prices["close"].values.astype(np.float64)

    if config.estimator == VolatilityEstimator.STD:
        # rolling_volatility returns std of log returns - scale to price units
        vol = rolling_volatility(prices["close"], window=window)
        return (vol * close).values

    elif config.estimator == VolatilityEstimator.ATR:
        atr_val, _ = atr(high, low, close, period=window)
        return atr_val

    elif config.estimator == VolatilityEstimator.PARKINSON:
        # parkinson_volatility returns annualized volatility - scale to price units
        vol = parkinson_volatility(high, low, window)
        # Convert from annualized to per-bar and scale to price
        return (vol / np.sqrt(252)) * ((high + low) / 2)

    elif config.estimator == VolatilityEstimator.GARMAN_KLASS:
        # garman_klass_volatility returns annualized volatility - scale to price units
        vol = garman_klass_volatility(open_, high, low, close, window)
        # Convert from annualized to per-bar and scale to price
        return (vol / np.sqrt(252)) * close
    else:
        raise ValueError(f"Unknown volatility estimator: {config.estimator}")


@njit(cache=True)
def _process_single_event_numba(event_idx: int, side: int, entry_price: float, volatility: float, high_arr: np.ndarray,
                                low_arr: np.ndarray, close_arr: np.ndarray, upper_multiplier: float,
                                lower_multiplier: float, max_bars: int, vertical_as_zero: bool) -> tuple:
    """
    Numba-optimized inner loop for processing a single event.

    Returns tuple: (valid, label, ret, exit_bar_idx, barrier_hit_code, bars_held)
    - valid: bool, False if insufficient data
    - barrier_hit_code: 0=upper, 1=lower, 2=vertical
    """
    n_prices = len(high_arr)

    # Check if we have enough forward data
    if event_idx + max_bars >= n_prices:
        return (False, 0, 0.0, 0, 0, 0)

    # Compute barriers
    upper_barrier = entry_price + upper_multiplier * volatility
    lower_barrier = entry_price - lower_multiplier * volatility

    # Walk forward through bars
    for bars_held in range(1, max_bars + 1):
        bar_idx = event_idx + bars_held
        bar_high = high_arr[bar_idx]
        bar_low = low_arr[bar_idx]

        upper_hit = bar_high >= upper_barrier
        lower_hit = bar_low <= lower_barrier

        # Handle simultaneous touch - worst case for the side
        if upper_hit and lower_hit:
            if side == 1:
                # Long position: lower barrier hit is loss
                ret = (lower_barrier - entry_price) / entry_price
                return True, -1, ret, bar_idx, 1, bars_held  # 1 = lower
            else:
                # Short position: upper barrier hit is loss
                ret = (entry_price - upper_barrier) / entry_price
                return True, -1, ret, bar_idx, 0, bars_held  # 0 = upper

        # Upper barrier hit
        if upper_hit:
            ret = (upper_barrier - entry_price) / entry_price
            label = 1 if side == 1 else -1
            adj_ret = ret if side == 1 else -ret
            return (True, label, adj_ret, bar_idx, 0, bars_held)  # 0 = upper

        # Lower barrier hit
        if lower_hit:
            ret = (lower_barrier - entry_price) / entry_price
            label = -1 if side == 1 else 1
            adj_ret = ret if side == 1 else -ret
            return (True, label, adj_ret, bar_idx, 1, bars_held)  # 1 = lower

    # Vertical barrier hit (max holding period reached)
    final_idx = event_idx + max_bars
    final_close = close_arr[final_idx]

    # Compute return based on position direction
    raw_ret = (final_close - entry_price) / entry_price
    ret = raw_ret if side == 1 else -raw_ret

    if vertical_as_zero:
        label = 0
    else:
        if ret > 0:
            label = 1
        elif ret < 0:
            label = -1
        else:
            label = 0

    return (True, label, ret, final_idx, 2, max_bars)  # 2 = vertical


def _process_single_event(event_idx: int, side: int, entry_price: float, volatility: float, prices: pd.DataFrame,
                          config: BarrierConfig, high_arr: np.ndarray = None, low_arr: np.ndarray = None,
                          close_arr: np.ndarray = None) -> Optional[Dict]:
    """
    Process a single event and determine barrier outcome.

    Parameters
    ----------
    event_idx : int
        Index position of event in prices DataFrame
    side : int
        Trade direction: 1 for long, -1 for short
    entry_price : float
        Entry price at event time
    volatility : float
        Volatility estimate at event time
    prices : pd.DataFrame
        Full OHLC price data
    config : BarrierConfig
        Barrier configuration
    high_arr, low_arr, close_arr : np.ndarray, optional
        Pre-extracted numpy arrays for performance (avoids repeated extraction)

    Returns
    -------
    dict or None
        Result dictionary with label, ret, exit_time, barrier_hit, bars_held
        Returns None if insufficient forward data
    """
    # Extract arrays if not provided
    if high_arr is None:
        high_arr = prices["high"].values
    if low_arr is None:
        low_arr = prices["low"].values
    if close_arr is None:
        close_arr = prices["close"].values

    barrier_names = ["upper", "lower", "vertical"]

    # Call numba-optimized function
    valid, label, ret, exit_bar_idx, barrier_hit_code, bars_held = \
        _process_single_event_numba(event_idx, side, entry_price, volatility, high_arr, low_arr, close_arr,
        config.upper_multiplier, config.lower_multiplier, config.max_holding_bars, config.vertical_as_zero)

    if not valid:
        return None

    return {
        "label": label,
        "ret": ret,
        "exit_time": prices.index[exit_bar_idx],
        "barrier_hit": barrier_names[barrier_hit_code],
        "bars_held": bars_held,
    }


def apply_tbm(prices: pd.DataFrame, events: pd.DataFrame, config: TBMConfig) -> pd.DataFrame:
    """
    Apply Triple Barrier Method to label trade outcomes.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC DataFrame with datetime index and columns: open, high, low, close
    events : pd.DataFrame
        Events DataFrame with datetime index and column: side (+1 for long, -1 for short)
    config : TBMConfig
        TBM configuration

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by event timestamp with columns:
        - label: int {-1, 0, 1}
        - ret: float (actual return)
        - exit_time: datetime
        - barrier_hit: str {"upper", "lower", "vertical"}
        - bars_held: int
    """
    # Validate inputs
    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(prices.columns):
        raise ValueError(f"prices must contain columns: {required_cols}")

    if "side" not in events.columns:
        raise ValueError("events must contain 'side' column")

    # Compute volatility
    volatility = compute_volatility(prices, config.volatility)

    # Pre-extract arrays for performance (avoid repeated extraction in loop)
    high_arr = prices["high"].values.astype(np.float64)
    low_arr = prices["low"].values.astype(np.float64)
    close_arr = prices["close"].values.astype(np.float64)

    # Process each event
    results = []
    event_timestamps = []

    for event_time, event_row in events.iterrows():
        side = event_row["side"]

        # Skip invalid sides
        if side not in (1, -1):
            continue

        # Find event index in prices
        try:
            event_idx = prices.index.get_loc(event_time)
        except KeyError:
            # Event time not in prices, skip
            continue

        # Handle potential slice objects from get_loc
        if isinstance(event_idx, slice):
            event_idx = event_idx.start

        # Get entry price and volatility
        entry_price = close_arr[event_idx]
        vol = volatility[event_idx]

        # Skip if volatility is NaN
        if np.isnan(vol) or vol <= 0:
            continue

        # Process event using pre-extracted arrays
        result = _process_single_event(event_idx=event_idx, side=side, entry_price=entry_price, volatility=vol,
                                       prices=prices, config=config.barrier, high_arr=high_arr, low_arr=low_arr,
                                       close_arr=close_arr)

        if result is not None:
            results.append(result)
            event_timestamps.append(event_time)

    if not results:
        return pd.DataFrame(columns=["label", "ret", "exit_time", "barrier_hit", "bars_held"])

    return pd.DataFrame(results, index=event_timestamps)


def tbm_from_signals(prices: pd.DataFrame, signals: pd.Series, config: TBMConfig) -> pd.DataFrame:
    """
    Convenience function: filter non-zero signals to events and apply TBM.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC DataFrame with datetime index and columns: open, high, low, close
    signals : pd.Series
        Signal series with datetime index. Values: 1=long, -1=short, 0=no signal
    config : TBMConfig
        TBM configuration

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by event timestamp with columns:
        - label: int {-1, 0, 1}
        - ret: float (actual return)
        - exit_time: datetime
        - barrier_hit: str {"upper", "lower", "vertical"}
        - bars_held: int
    """
    # Filter non-zero signals
    non_zero = signals[signals != 0]

    if len(non_zero) == 0:
        return pd.DataFrame(columns=["label", "ret", "exit_time", "barrier_hit", "bars_held"])

    # Convert to events DataFrame
    events = pd.DataFrame({"side": non_zero})

    return apply_tbm(prices, events, config)


def _apply_tbm_with_cached_volatility(prices: pd.DataFrame, events: pd.DataFrame, volatility: np.ndarray,
                                      barrier_config: BarrierConfig, high_arr: np.ndarray, low_arr: np.ndarray,
                                      close_arr: np.ndarray) -> pd.DataFrame:
    """
    Apply TBM with pre-computed volatility (for optimization with caching).

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC DataFrame with datetime index
    events : pd.DataFrame
        Events DataFrame with 'side' column
    volatility : np.ndarray
        Pre-computed volatility array
    barrier_config : BarrierConfig
        Barrier configuration
    high_arr, low_arr, close_arr : np.ndarray
        Pre-extracted price arrays

    Returns
    -------
    pd.DataFrame
        TBM results
    """
    results = []
    event_timestamps = []

    for event_time, event_row in events.iterrows():
        side = event_row["side"]

        if side not in (1, -1):
            continue

        try:
            event_idx = prices.index.get_loc(event_time)
        except KeyError:
            continue

        if isinstance(event_idx, slice):
            event_idx = event_idx.start

        entry_price = close_arr[event_idx]
        vol = volatility[event_idx]

        if np.isnan(vol) or vol <= 0:
            continue

        result = _process_single_event(event_idx=event_idx, side=side, entry_price=entry_price, volatility=vol,
                                       prices=prices, config=barrier_config, high_arr=high_arr, low_arr=low_arr,
                                       close_arr=close_arr)
        if result is not None:
            results.append(result)
            event_timestamps.append(event_time)
    if not results:
        return pd.DataFrame(columns=["label", "ret", "exit_time", "barrier_hit", "bars_held"])

    return pd.DataFrame(results, index=event_timestamps)


def _compute_metric(labels_df: pd.DataFrame, metric: OptimizationMetric) -> float:
    """
    Compute optimization metric from TBM results.

    Parameters
    ----------
    labels_df : pd.DataFrame
        TBM output DataFrame
    metric : OptimizationMetric
        Metric to compute

    Returns
    -------
    float
        Metric value (NaN if cannot be computed)
    """
    if len(labels_df) == 0:
        return np.nan

    returns = labels_df["ret"]
    labels = labels_df["label"]
    barrier_hits = labels_df["barrier_hit"]

    if metric == OptimizationMetric.SHARPE_RATIO:
        if returns.std() == 0 or len(returns) < 2:
            return np.nan
        return returns.mean() / returns.std()

    elif metric == OptimizationMetric.WIN_RATE:
        return (labels == 1).sum() / len(labels)

    elif metric == OptimizationMetric.PROFIT_FACTOR:
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else np.nan
        return gross_profit / gross_loss

    elif metric == OptimizationMetric.TOTAL_RETURN:
        return returns.sum()

    elif metric == OptimizationMetric.AVG_RETURN:
        return returns.mean()

    elif metric == OptimizationMetric.VERTICAL_RATE:
        # Lower is better - return negative so maximization finds lowest
        return -(barrier_hits == "vertical").sum() / len(barrier_hits)

    else:
        raise ValueError(f"Unknown metric: {metric}")


def _evaluate_volatility_config(prices: pd.DataFrame, signals: pd.Series, estimator: VolatilityEstimator, window: int,
                                barrier_config: BarrierConfig, metric: OptimizationMetric, min_trades: int) -> Dict:
    """Helper function for parallel volatility optimization."""
    config = TBMConfig(
        volatility=VolatilityConfig(
            estimator=estimator,
            params={"window": window},
        ),
        barrier=barrier_config,
    )

    labels_df = tbm_from_signals(prices, signals, config)
    n_trades = len(labels_df)

    if n_trades < min_trades:
        score = np.nan
    else:
        score = _compute_metric(labels_df, metric)

    if n_trades > 0:
        win_rate = (labels_df["label"] == 1).sum() / n_trades
        loss_rate = (labels_df["label"] == -1).sum() / n_trades
        vertical_rate = (labels_df["barrier_hit"] == "vertical").sum() / n_trades
        total_return = labels_df["ret"].sum()
        avg_return = labels_df["ret"].mean()
        std_return = labels_df["ret"].std()
        avg_bars = labels_df["bars_held"].mean()
    else:
        win_rate = loss_rate = vertical_rate = np.nan
        total_return = avg_return = std_return = avg_bars = np.nan

    return {
        "estimator": estimator.name,
        "window": window,
        "score": score,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "vertical_rate": vertical_rate,
        "total_return": total_return,
        "avg_return": avg_return,
        "std_return": std_return,
        "avg_bars_held": avg_bars,
    }


def optimize_tbm_volatility(prices: pd.DataFrame, signals: pd.Series, barrier_config: BarrierConfig,
                            estimators: Optional[List[VolatilityEstimator]] = None, windows: Optional[List[int]] = None,
                            metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO, min_trades: int = 30,
                            n_jobs: int = -1) -> Dict:
    """
    Optimize volatility estimator and window for TBM labeling.

    Performs grid search over volatility estimators and window sizes,
    evaluating each combination using the specified metric.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC DataFrame with datetime index and columns: open, high, low, close
    signals : pd.Series
        Signal series with datetime index. Values: 1=long, -1=short, 0=no signal
    barrier_config : BarrierConfig
        Fixed barrier configuration to use during optimization
    estimators : list of VolatilityEstimator, optional
        Volatility estimators to test. Default: all estimators
    windows : list of int, optional
        Window sizes to test. Default: [7, 10, 14, 20, 30, 50]
    metric : OptimizationMetric
        Metric to optimize. Default: SHARPE_RATIO
    min_trades : int
        Minimum number of trades required for valid evaluation. Default: 30
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores. Default: -1

    Returns
    -------
    dict
        Dictionary containing:
        - 'best_estimator': VolatilityEstimator - optimal volatility type
        - 'best_window': int - optimal window size
        - 'best_score': float - best metric value
        - 'best_config': TBMConfig - ready-to-use configuration
        - 'results_df': pd.DataFrame - full results for all combinations
        - 'metric': str - metric name used

    Examples
    --------
    >>> result = optimize_tbm_volatility(
    ...     prices, signals,
    ...     barrier_config=BarrierConfig(upper_multiplier=2.0, lower_multiplier=2.0),
    ...     metric=OptimizationMetric.SHARPE_RATIO,
    ... )
    >>> print(f"Best: {result['best_estimator'].name} window={result['best_window']}")
    >>> best_labels = tbm_from_signals(prices, signals, result['best_config'])
    """
    # Set defaults
    if estimators is None:
        estimators = list(VolatilityEstimator)
    if windows is None:
        windows = [7, 10, 14, 20, 30, 50]

    # Build parameter grid
    param_grid = [
        (estimator, window)
        for estimator in estimators
        for window in windows
    ]

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_volatility_config)(
            prices, signals, estimator, window, barrier_config, metric, min_trades
        )
        for estimator, window in param_grid
    )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best (handle NaN and metric direction)
    valid_results = results_df.dropna(subset=["score"])

    if len(valid_results) == 0:
        return {
            "best_estimator": None,
            "best_window": None,
            "best_score": np.nan,
            "best_config": None,
            "results_df": results_df,
            "metric": metric.value,
        }

    # All metrics are "higher is better" (vertical_rate is negated in computation)
    best_idx = valid_results["score"].idxmax()
    best_row = results_df.loc[best_idx]

    best_estimator = VolatilityEstimator[best_row["estimator"]]
    best_window = int(best_row["window"])
    best_score = best_row["score"]

    # For vertical_rate, convert back to positive for display
    if metric == OptimizationMetric.VERTICAL_RATE:
        best_score = -best_score

    best_config = TBMConfig(
        volatility=VolatilityConfig(
            estimator=best_estimator,
            params={"window": best_window},
        ),
        barrier=barrier_config,
    )

    return {
        "best_estimator": best_estimator,
        "best_window": best_window,
        "best_score": best_score,
        "best_config": best_config,
        "results_df": results_df,
        "metric": metric.value,
    }


def _evaluate_full_config_with_cache(prices: pd.DataFrame, events: pd.DataFrame, volatility: np.ndarray,
                                     high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, estimator_name: str,
                                     window: int, upper_mult: float, lower_mult: float, max_bars: int,
                                     metric: OptimizationMetric, min_trades: int) -> Dict:
    """Helper function for parallel full optimization with cached volatility."""
    barrier_config = BarrierConfig(upper_multiplier=upper_mult, lower_multiplier=lower_mult, max_holding_bars=max_bars)

    labels_df = _apply_tbm_with_cached_volatility(
        prices, events, volatility, barrier_config, high_arr, low_arr, close_arr
    )
    n_trades = len(labels_df)

    score = np.nan if n_trades < min_trades else _compute_metric(labels_df, metric)

    if n_trades > 0:
        win_rate = (labels_df["label"] == 1).sum() / n_trades
        loss_rate = (labels_df["label"] == -1).sum() / n_trades
        vertical_rate = (labels_df["barrier_hit"] == "vertical").sum() / n_trades
        total_return = labels_df["ret"].sum()
        avg_return = labels_df["ret"].mean()
    else:
        win_rate = loss_rate = vertical_rate = np.nan
        total_return = avg_return = np.nan

    return {
        "estimator": estimator_name,
        "window": window,
        "upper_mult": upper_mult,
        "lower_mult": lower_mult,
        "max_bars": max_bars,
        "score": score,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "vertical_rate": vertical_rate,
        "total_return": total_return,
        "avg_return": avg_return,
    }


def optimize_tbm_full(prices: pd.DataFrame, signals: pd.Series, estimators: Optional[List[VolatilityEstimator]] = None,
                      windows: Optional[List[int]] = None, upper_multipliers: Optional[List[float]] = None,
                      lower_multipliers: Optional[List[float]] = None, max_holding_bars_list: Optional[List[int]] = None,
                      metric: OptimizationMetric = OptimizationMetric.SHARPE_RATIO, min_trades: int = 30,
                      n_jobs: int = -1) -> Dict:
    """
    Full optimization of all TBM parameters.

    Performs grid search over volatility estimators, window sizes, and barrier
    configuration parameters using parallel processing.

    Parameters
    ----------
    prices : pd.DataFrame
        OHLC DataFrame with datetime index and columns: open, high, low, close
    signals : pd.Series
        Signal series with datetime index. Values: 1=long, -1=short, 0=no signal
    estimators : list of VolatilityEstimator, optional
        Volatility estimators to test. Default: all estimators
    windows : list of int, optional
        Window sizes to test. Default: [10, 14, 20]
    upper_multipliers : list of float, optional
        Upper barrier multipliers to test. Default: [1.5, 2.0, 2.5]
    lower_multipliers : list of float, optional
        Lower barrier multipliers to test. Default: [1.5, 2.0, 2.5]
    max_holding_bars_list : list of int, optional
        Max holding periods to test. Default: [10, 20, 30]
    metric : OptimizationMetric
        Metric to optimize. Default: SHARPE_RATIO
    min_trades : int
        Minimum number of trades required for valid evaluation. Default: 30
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores. Default: -1

    Returns
    -------
    dict
        Dictionary containing:
        - 'best_estimator': VolatilityEstimator
        - 'best_window': int
        - 'best_upper_mult': float
        - 'best_lower_mult': float
        - 'best_max_bars': int
        - 'best_score': float
        - 'best_config': TBMConfig
        - 'results_df': pd.DataFrame
        - 'metric': str
        - 'n_combinations': int
    """
    # Set defaults (smaller grids for full optimization)
    if estimators is None:
        estimators = list(VolatilityEstimator)
    if windows is None:
        windows = [10, 14, 20]
    if upper_multipliers is None:
        upper_multipliers = [1.5, 2.0, 2.5]
    if lower_multipliers is None:
        lower_multipliers = [1.5, 2.0, 2.5]
    if max_holding_bars_list is None:
        max_holding_bars_list = [10, 20, 30]

    n_combinations = (
        len(estimators) * len(windows) * len(upper_multipliers) *
        len(lower_multipliers) * len(max_holding_bars_list)
    )

    # Pre-extract arrays once
    high_arr = prices["high"].values.astype(np.float64)
    low_arr = prices["low"].values.astype(np.float64)
    close_arr = prices["close"].values.astype(np.float64)

    # Pre-compute events once
    non_zero = signals[signals != 0]
    if len(non_zero) == 0:
        return {
            "best_estimator": None,
            "best_window": None,
            "best_upper_mult": None,
            "best_lower_mult": None,
            "best_max_bars": None,
            "best_score": np.nan,
            "best_config": None,
            "results_df": pd.DataFrame(),
            "metric": metric.value,
            "n_combinations": n_combinations,
        }
    events = pd.DataFrame({"side": non_zero})

    # Pre-compute volatility for each unique (estimator, window) pair
    volatility_cache = {}
    for estimator in estimators:
        for window in windows:
            cache_key = (estimator.name, window)
            vol_config = VolatilityConfig(estimator=estimator, params={"window": window})
            volatility_cache[cache_key] = compute_volatility(prices, vol_config)

    # Build parameter grid with cached volatility references
    param_grid = [
        (estimator.name, window, upper_mult, lower_mult, max_bars)
        for estimator in estimators
        for window in windows
        for upper_mult in upper_multipliers
        for lower_mult in lower_multipliers
        for max_bars in max_holding_bars_list
    ]

    # Parallel execution with cached volatility
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_full_config_with_cache)(
            prices, events, volatility_cache[(estimator_name, window)],
            high_arr, low_arr, close_arr,
            estimator_name, window, upper_mult, lower_mult, max_bars, metric, min_trades
        )
        for estimator_name, window, upper_mult, lower_mult, max_bars in param_grid
    )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Find best
    valid_results = results_df.dropna(subset=["score"])

    if len(valid_results) == 0:
        return {
            "best_estimator": None,
            "best_window": None,
            "best_upper_mult": None,
            "best_lower_mult": None,
            "best_max_bars": None,
            "best_score": np.nan,
            "best_config": None,
            "results_df": results_df,
            "metric": metric.value,
            "n_combinations": n_combinations,
        }

    best_idx = valid_results["score"].idxmax()
    best_row = results_df.loc[best_idx]

    best_estimator = VolatilityEstimator[best_row["estimator"]]
    best_window = int(best_row["window"])
    best_upper_mult = float(best_row["upper_mult"])
    best_lower_mult = float(best_row["lower_mult"])
    best_max_bars = int(best_row["max_bars"])
    best_score = best_row["score"]

    if metric == OptimizationMetric.VERTICAL_RATE:
        best_score = -best_score

    best_config = TBMConfig(
        volatility=VolatilityConfig(estimator=best_estimator, params={"window": best_window}),
        barrier=BarrierConfig(upper_multiplier=best_upper_mult, lower_multiplier=best_lower_mult, max_holding_bars=best_max_bars),
    )

    return {
        "best_estimator": best_estimator,
        "best_window": best_window,
        "best_upper_mult": best_upper_mult,
        "best_lower_mult": best_lower_mult,
        "best_max_bars": best_max_bars,
        "best_score": best_score,
        "best_config": best_config,
        "results_df": results_df,
        "metric": metric.value,
        "n_combinations": n_combinations,
    }
