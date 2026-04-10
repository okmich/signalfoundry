from collections import Counter
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


def calculate_simple_transition_costs(original_regimes: Union[np.ndarray, pd.Series], smoothed_regimes: Union[np.ndarray, pd.Series],
                                      cost_per_transition_bps: float = 5.0) -> Dict[str, float]:
    """
    Calculate transaction costs assuming fixed cost per regime transition.

    Parameters
    ----------
    original_regimes : np.ndarray or pd.Series
        Raw HMM regime sequence
    smoothed_regimes : np.ndarray or pd.Series
        Smoothed regime sequence
    cost_per_transition_bps : float, default=5.0
        Fixed transaction cost per regime change (basis points)

    Returns
    -------
    dict
        Dictionary containing:
        - original_transitions: Number of transitions in raw sequence
        - smoothed_transitions: Number of transitions after smoothing
        - transitions_saved: Reduction in transition count
        - original_cost_bps: Total cost from raw sequence
        - smoothed_cost_bps: Total cost from smoothed sequence
        - savings_bps: Cost savings from smoothing
        - savings_pct: Percentage cost reduction

    Examples
    --------
    >>> original = pd.Series([0, 1, 0, 1, 0, 1, 1, 1, 2, 2])
    >>> smoothed = pd.Series([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    >>> costs = calculate_simple_transition_costs(original, smoothed, cost_per_transition_bps=5.0)
    >>> costs['transitions_saved']
    3
    >>> costs['savings_bps']
    15.0
    """
    # Convert to arrays if needed
    orig = np.asarray(original_regimes)
    smooth = np.asarray(smoothed_regimes)

    # Count transitions (where regime changes)
    orig_transitions = int(np.sum(orig[1:] != orig[:-1]))
    smooth_transitions = int(np.sum(smooth[1:] != smooth[:-1]))

    # Calculate costs
    orig_cost = orig_transitions * cost_per_transition_bps
    smooth_cost = smooth_transitions * cost_per_transition_bps
    savings = orig_cost - smooth_cost
    savings_pct = (savings / orig_cost * 100) if orig_cost > 0 else 0.0

    return {
        "original_transitions": orig_transitions,
        "smoothed_transitions": smooth_transitions,
        "transitions_saved": orig_transitions - smooth_transitions,
        "original_cost_bps": orig_cost,
        "smoothed_cost_bps": smooth_cost,
        "savings_bps": savings,
        "savings_pct": savings_pct,
    }


def calculate_regime_specific_costs(original_regimes: Union[np.ndarray, pd.Series], smoothed_regimes: Union[np.ndarray, pd.Series],
                                    cost_matrix: Dict[Tuple[int, int], float]) -> Dict[str, Union[float, Dict]]:
    """
    Calculate transaction costs with regime-pair specific costs.

    Different regime transitions incur different costs. For example:
    - Regime 0 → 1 might cost 10 bps (large position change)
    - Regime 1 → 0 might cost 7 bps (smaller position change)
    - Regime 0 → 2 might cost 15 bps (emergency rebalance)

    Parameters
    ----------
    original_regimes : np.ndarray or pd.Series
        Raw HMM regime sequence
    smoothed_regimes : np.ndarray or pd.Series
        Smoothed regime sequence
    cost_matrix : dict
        Mapping (from_regime, to_regime) -> cost_bps
        Example: {(0, 1): 10, (1, 0): 7, (0, 2): 15, ...}

    Returns
    -------
    dict
        Dictionary containing:
        - original_transitions: Total transitions in raw
        - smoothed_transitions: Total transitions in smoothed
        - original_cost_bps: Total cost from raw
        - smoothed_cost_bps: Total cost from smoothed
        - savings_bps: Cost savings
        - savings_pct: Percentage cost reduction
        - original_by_transition: Dict of transition counts in raw
        - smoothed_by_transition: Dict of transition counts in smoothed
        - cost_by_transition: Cost breakdown by transition type

    Examples
    --------
    >>> cost_matrix = {
    ...     (0, 1): 10,  # Low vol -> High vol: larger rebalance
    ...     (1, 0): 7,   # High vol -> Low vol: smaller rebalance
    ...     (0, 2): 15,  # Low vol -> Crisis: emergency exit
    ...     (2, 0): 8,   # Crisis -> Low vol: gradual entry
    ...     (1, 2): 12,  # High vol -> Crisis
    ...     (2, 1): 9    # Crisis -> High vol
    ... }
    >>> costs = calculate_regime_specific_costs(original, smoothed, cost_matrix)
    """
    orig = np.asarray(original_regimes)
    smooth = np.asarray(smoothed_regimes)

    # Find all transitions
    orig_transitions = [
        (int(orig[i]), int(orig[i + 1]))
        for i in range(len(orig) - 1)
        if orig[i] != orig[i + 1]
    ]
    smooth_transitions = [
        (int(smooth[i]), int(smooth[i + 1]))
        for i in range(len(smooth) - 1)
        if smooth[i] != smooth[i + 1]
    ]

    # Count each transition type
    orig_counts = Counter(orig_transitions)
    smooth_counts = Counter(smooth_transitions)

    # Calculate costs
    orig_cost = sum(
        orig_counts[trans] * cost_matrix.get(trans, 0) for trans in orig_counts
    )
    smooth_cost = sum(
        smooth_counts[trans] * cost_matrix.get(trans, 0) for trans in smooth_counts
    )

    # Cost breakdown by transition type
    cost_breakdown = {}
    all_transitions = set(orig_counts.keys()) | set(smooth_counts.keys())
    for trans in all_transitions:
        orig_count = orig_counts.get(trans, 0)
        smooth_count = smooth_counts.get(trans, 0)
        cost = cost_matrix.get(trans, 0)
        cost_breakdown[trans] = {
            "original_count": orig_count,
            "smoothed_count": smooth_count,
            "saved_count": orig_count - smooth_count,
            "cost_per_transition_bps": cost,
            "original_cost_bps": orig_count * cost,
            "smoothed_cost_bps": smooth_count * cost,
            "savings_bps": (orig_count - smooth_count) * cost,
        }

    savings = orig_cost - smooth_cost
    savings_pct = (savings / orig_cost * 100) if orig_cost > 0 else 0.0

    return {
        "original_transitions": len(orig_transitions),
        "smoothed_transitions": len(smooth_transitions),
        "original_cost_bps": orig_cost,
        "smoothed_cost_bps": smooth_cost,
        "savings_bps": savings,
        "savings_pct": savings_pct,
        "original_by_transition": dict(orig_counts),
        "smoothed_by_transition": dict(smooth_counts),
        "cost_by_transition": cost_breakdown,
    }


def calculate_transition_costs(original_regimes: Union[np.ndarray, pd.Series], smoothed_regimes: Union[np.ndarray, pd.Series],
                               cost_model: Union[float, Dict[Tuple[int, int], float]] = 5.0) -> Dict[str, Union[float, Dict]]:
    """
    Unified interface for calculating transaction costs.

    Automatically dispatches to simple or regime-specific cost calculation based on cost_model type.

    Parameters
    ----------
    original_regimes : np.ndarray or pd.Series
        Raw HMM regime sequence
    smoothed_regimes : np.ndarray or pd.Series
        Smoothed regime sequence
    cost_model : float or dict, default=5.0
        If float: fixed cost per transition (simple model)
        If dict: regime-pair specific costs (advanced model)

    Returns
    -------
    dict
        Cost analysis results (format depends on cost_model type)

    Examples
    --------
    # Simple fixed cost
    >>> costs = calculate_transition_costs(original, smoothed, cost_model=5.0)

    # Regime-specific costs
    >>> cost_matrix = {(0,1): 10, (1,0): 7, ...}
    >>> costs = calculate_transition_costs(original, smoothed, cost_model=cost_matrix)
    """
    if isinstance(cost_model, (int, float)):
        return calculate_simple_transition_costs(
            original_regimes, smoothed_regimes, cost_model
        )
    elif isinstance(cost_model, dict):
        return calculate_regime_specific_costs(
            original_regimes, smoothed_regimes, cost_model
        )
    else:
        raise TypeError(f"cost_model must be float or dict, got {type(cost_model)}")


def build_symmetric_cost_matrix(num_regimes: int, base_cost: float = 5.0) -> Dict[Tuple[int, int], float]:
    """
    Build symmetric cost matrix where all transitions have same cost.

    Parameters
    ----------
    num_regimes : int
        Number of regimes
    base_cost : float
        Cost for any transition

    Returns
    -------
    dict
        Cost matrix with all transitions having base_cost

    Examples
    --------
    >>> matrix = build_symmetric_cost_matrix(3, base_cost=5.0)
    >>> matrix[(0, 1)]
    5.0
    >>> matrix[(2, 0)]
    5.0
    """
    cost_matrix = {}
    for i in range(num_regimes):
        for j in range(num_regimes):
            if i != j:
                cost_matrix[(i, j)] = base_cost
    return cost_matrix


def build_asymmetric_cost_matrix(num_regimes: int, entry_costs: Dict[int, float], exit_costs: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    """
    Build asymmetric cost matrix based on entry/exit costs per regime.

    Cost for transition i→j = (exit_cost[i] + entry_cost[j]) / 2

    Parameters
    ----------
    num_regimes : int
        Number of regimes
    entry_costs : dict
        Cost to enter each regime: {regime_id: cost_bps}
    exit_costs : dict
        Cost to exit each regime: {regime_id: cost_bps}

    Returns
    -------
    dict
        Asymmetric cost matrix

    Examples
    --------
    >>> # Higher cost to enter high-vol regime, lower to exit
    >>> entry_costs = {0: 5, 1: 12, 2: 8}  # 0=low_vol, 1=high_vol, 2=crisis
    >>> exit_costs = {0: 7, 1: 6, 2: 15}   # Crisis hard to exit
    >>> matrix = build_asymmetric_cost_matrix(3, entry_costs, exit_costs)
    >>> matrix[(0, 1)]  # low_vol -> high_vol
    9.5  # (exit_low_vol + enter_high_vol) / 2 = (7 + 12) / 2
    """
    cost_matrix = {}
    for i in range(num_regimes):
        for j in range(num_regimes):
            if i != j:
                cost_matrix[(i, j)] = (
                    exit_costs.get(i, 5.0) + entry_costs.get(j, 5.0)
                ) / 2
    return cost_matrix
