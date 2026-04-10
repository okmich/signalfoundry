from typing import Dict, Union

import numpy as np
import pandas as pd


def calculate_transition_rate(states: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate regime transitions per period.

    Parameters
    ----------
    states : np.ndarray or pd.Series
        State sequence

    Returns
    -------
    float
        Transition rate (transitions per period)

    Examples
    --------
    >>> states = np.array([0, 0, 1, 1, 1, 0, 0])
    >>> calculate_transition_rate(states)
    0.2857...  # 2 transitions / 7 periods
    """
    states_arr = states.values if isinstance(states, pd.Series) else states

    if len(states_arr) <= 1:
        return 0.0

    transitions = np.sum(states_arr[1:] != states_arr[:-1])
    return float(transitions) / len(states_arr)


def calculate_label_stability(states: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Calculate stability metrics for regime labels.

    Parameters
    ----------
    states : np.ndarray or pd.Series
        State sequence

    Returns
    -------
    dict
        Statistics including:
        - mean_duration: Average regime duration
        - std_duration: Standard deviation of durations
        - min_duration: Minimum regime duration
        - max_duration: Maximum regime duration
        - entropy: Shannon entropy of regime distribution
        - transition_rate: Transitions per period

    Examples
    --------
    >>> states = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    >>> stats = calculate_label_stability(states)
    >>> stats['mean_duration']
    3.0
    """
    states_arr = states.values if isinstance(states, pd.Series) else states

    if len(states_arr) == 0:
        return {
            "mean_duration": 0.0,
            "std_duration": 0.0,
            "min_duration": 0,
            "max_duration": 0,
            "entropy": 0.0,
            "transition_rate": 0.0,
        }

    # Calculate durations
    durations = []
    current_state = states_arr[0]
    current_duration = 1

    for i in range(1, len(states_arr)):
        if states_arr[i] == current_state:
            current_duration += 1
        else:
            durations.append(current_duration)
            current_state = states_arr[i]
            current_duration = 1
    durations.append(current_duration)

    # Duration statistics
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    min_duration = int(np.min(durations))
    max_duration = int(np.max(durations))

    # Entropy of regime distribution
    unique_states, counts = np.unique(states_arr, return_counts=True)
    probs = counts / len(states_arr)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Transition rate and count
    transitions = np.sum(states_arr[1:] != states_arr[:-1]) if len(states_arr) > 1 else 0
    transition_rate = float(transitions) / len(states_arr) if len(states_arr) > 0 else 0.0

    return {
        "mean_duration": float(mean_duration),
        "std_duration": float(std_duration),
        "min_duration": min_duration,
        "max_duration": max_duration,
        "entropy": float(entropy),
        "transition_rate": transition_rate,
        "num_transitions": int(transitions),
    }


def calculate_regime_sharpe(
    returns: Union[np.ndarray, pd.Series],
    regimes: Union[np.ndarray, pd.Series],
    regime_positions: Dict[int, float],
    annualization_factor: float = 252.0,
) -> Dict[str, float]:
    """
    Calculate Sharpe ratio for regime-dependent strategies.

    Parameters
    ----------
    returns : np.ndarray or pd.Series
        Asset returns with datetime index
    regimes : np.ndarray or pd.Series
        Regime labels aligned with returns
    regime_positions : dict
        Mapping from regime_id to strategy position
        Examples:
        - {0: 1.0, 1: -0.5}: Long in regime 0, short 50% in regime 1
        - {0: 1.0, 1: 0.0, 2: -1.0}: Long/neutral/short strategy
    annualization_factor : float, default=252.0
        Factor to annualize returns (252 for daily, 12 for monthly)

    Returns
    -------
    dict
        Sharpe ratios:
        - overall: Overall strategy Sharpe ratio
        - by_regime: Dict of Sharpe ratios per regime

    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005])
    >>> regimes = pd.Series([0, 0, 1, 1, 1])
    >>> positions = {0: 1.0, 1: -1.0}  # Long in regime 0, short in regime 1
    >>> sharpe = calculate_regime_sharpe(returns, regimes, positions)
    >>> sharpe['overall']  # Overall Sharpe ratio
    """
    returns_arr = returns.values if isinstance(returns, pd.Series) else returns
    regimes_arr = regimes.values if isinstance(regimes, pd.Series) else regimes

    if len(returns_arr) != len(regimes_arr):
        raise ValueError("returns and regimes must have same length")

    # Calculate strategy returns
    strategy_returns = np.zeros(len(returns_arr))

    for regime, position in regime_positions.items():
        mask = regimes_arr == regime
        strategy_returns[mask] = returns_arr[mask] * position

    # Overall Sharpe ratio
    if np.std(strategy_returns) > 0:
        overall_sharpe = (
            np.mean(strategy_returns)
            / np.std(strategy_returns)
            * np.sqrt(annualization_factor)
        )
    else:
        overall_sharpe = 0.0

    # Sharpe ratio by regime
    by_regime = {}
    for regime in np.unique(regimes_arr):
        mask = regimes_arr == regime
        regime_returns = strategy_returns[mask]

        if len(regime_returns) > 1 and np.std(regime_returns) > 0:
            regime_sharpe = (
                np.mean(regime_returns)
                / np.std(regime_returns)
                * np.sqrt(annualization_factor)
            )
        else:
            regime_sharpe = 0.0

        by_regime[int(regime)] = float(regime_sharpe)

    return {
        "overall": float(overall_sharpe),
        "by_regime": by_regime,
        "mean_return_annualized": float(
            np.mean(strategy_returns) * annualization_factor
        ),
        "volatility_annualized": float(
            np.std(strategy_returns) * np.sqrt(annualization_factor)
        ),
    }
