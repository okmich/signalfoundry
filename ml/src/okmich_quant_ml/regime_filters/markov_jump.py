from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

from .base import BasePostProcessor


# =============================================================================
# Numba-optimized core functions
# =============================================================================


@njit(cache=True)
def _gamma_logpdf(x: float, shape: float, scale: float) -> float:
    """
    Calculate log PDF of gamma distribution using Stirling's approximation.

    Parameters
    ----------
    x : float
        Value to evaluate
    shape : float
        Shape parameter (k)
    scale : float
        Scale parameter (theta)

    Returns
    -------
    float
        Log probability
    """
    if x <= 0:
        return -np.inf

    # Stirling's approximation for log(Gamma(k))
    # log(Gamma(k)) ≈ (k-0.5)*log(k) - k + 0.5*log(2π)
    if shape > 1.0:
        log_gamma_k = (shape - 0.5) * np.log(shape) - shape + 0.5 * np.log(2 * np.pi)
    else:
        # For small shape, use simple approximation
        log_gamma_k = 0.0

    return (shape - 1) * np.log(x) - x / scale - shape * np.log(scale) - log_gamma_k


@njit(cache=True)
def _exponential_logpdf(x: float, rate: float) -> float:
    """
    Calculate log PDF of exponential distribution.

    Parameters
    ----------
    x : float
        Value to evaluate
    rate : float
        Rate parameter (lambda)

    Returns
    -------
    float
        Log probability
    """
    if x <= 0:
        return -np.inf
    return np.log(rate) - rate * x


@njit(cache=True)
def _lognormal_logpdf(x: float, mu: float, sigma: float) -> float:
    """
    Calculate log PDF of lognormal distribution.

    Parameters
    ----------
    x : float
        Value to evaluate
    mu : float
        Mean of underlying normal
    sigma : float
        Std of underlying normal

    Returns
    -------
    float
        Log probability
    """
    if x <= 0:
        return -np.inf

    log_x = np.log(x)
    return (
        -np.log(x)
        - 0.5 * np.log(2 * np.pi)
        - np.log(sigma)
        - 0.5 * ((log_x - mu) / sigma) ** 2
    )


@njit(cache=True)
def _markov_jump_regularizer_core(states: np.ndarray, dwell_distributions: np.ndarray, regularization_strength: float,
                                  transition_costs: np.ndarray, acceptance_threshold: float) -> np.ndarray:
    """
    Apply Markov jump process regularization.

    Parameters
    ----------
    states : np.ndarray, shape (T,)
        Input state sequence
    dwell_distributions : np.ndarray, shape (K, 3)
        Distribution parameters for each state
        [:, 0] = dist_type (0=gamma, 1=exponential, 2=lognormal)
        [:, 1] = param1 (shape/rate/mu)
        [:, 2] = param2 (scale/unused/sigma)
    regularization_strength : float
        Strength of regularization
    transition_costs : np.ndarray, shape (K, K)
        Transition costs matrix

    Returns
    -------
    np.ndarray, shape (T,)
        Regularized state sequence
    """
    n = len(states)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    smoothed = np.zeros(n, dtype=np.int32)
    current_state = np.int32(states[0])
    dwell_time = 1
    smoothed[0] = current_state

    for i in range(1, n):
        proposed_state = np.int32(states[i])

        if proposed_state == current_state:
            # Continue in same state
            dwell_time += 1
            smoothed[i] = current_state
        else:
            # Transition proposed — evaluate likelihood of current dwell time
            dist_type = int(dwell_distributions[current_state, 0])
            param1 = dwell_distributions[current_state, 1]
            param2 = dwell_distributions[current_state, 2]

            if dist_type == 0:  # Gamma
                log_prob = _gamma_logpdf(float(dwell_time), param1, param2)
            elif dist_type == 1:  # Exponential
                log_prob = _exponential_logpdf(float(dwell_time), param1)
            else:  # Lognormal
                log_prob = _lognormal_logpdf(float(dwell_time), param1, param2)

            # Get transition cost
            trans_cost = transition_costs[current_state, proposed_state]

            # Decision: accept transition if dwell time is reasonable and cost is acceptable
            score = log_prob - regularization_strength * trans_cost

            if score > acceptance_threshold:
                current_state = proposed_state
                dwell_time = 1
                smoothed[i] = current_state
            else:
                dwell_time += 1
                smoothed[i] = current_state

    return smoothed


# =============================================================================
# MarkovJumpProcessRegularizer
# =============================================================================


class MarkovJumpProcessRegularizer(BasePostProcessor):
    """
    Penalize transitions using regime-specific dwell time models.

    This processor models realistic regime persistence by encoding
    expected dwell time distributions for each regime. It penalizes
    transitions that violate these patterns (e.g., exiting a bull
    market after only 2 weeks when bulls typically last 2+ years).

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - state_dwell_params : dict
            Regime-specific dwell time distributions
            Format: {state_id: {'distribution': str, 'shape': float, 'scale': float}}
            Distributions: 'gamma', 'exponential', 'lognormal'
        - transition_costs : dict or np.ndarray, optional
            Transition costs matrix or dict {(from, to): cost}
        - regularization_strength : float, default=1.0
            Strength of regularization (higher = stricter)
        - estimate_from_data : bool, default=False
            If True, estimate dwell distributions from input data
        - min_samples : int, default=10
            Minimum samples required to estimate distribution

    Examples
    --------
    >>> # Model bull/bear market persistence
    >>> config = {
    ...     'state_dwell_params': {
    ...         0: {'distribution': 'gamma', 'shape': 2.5, 'scale': 120},  # Bull ~300 days
    ...         1: {'distribution': 'gamma', 'shape': 1.5, 'scale': 60},   # Bear ~90 days
    ...     },
    ...     'regularization_strength': 1.0
    ... }
    >>> processor = MarkovJumpProcessRegularizer(config)

    Notes
    -----
    For trading applications:
    - Bull markets: Use gamma/lognormal with long mean (200-500 days)
    - Bear markets: Use gamma with shorter mean (60-120 days)
    - Volatility regimes: Use exponential for mean-reverting behavior
    - Crisis: Use gamma with short mean but persistence

    The processor resists premature exits from persistent regimes while
    allowing quick detection of genuine transitions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MarkovJumpProcessRegularizer."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("state_dwell_params", {})
        config.setdefault("transition_costs", None)
        config.setdefault("regularization_strength", 1.0)
        config.setdefault("estimate_from_data", False)
        config.setdefault("min_samples", 10)
        config.setdefault("acceptance_threshold", -5.0)

        super().__init__(config)

        # Store distribution info
        self._dwell_distributions: Optional[np.ndarray] = None
        self._transition_costs: Optional[np.ndarray] = None
        self._num_states: Optional[int] = None

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["state_dwell_params"], dict):
            raise ValueError("state_dwell_params must be a dict")

        if not isinstance(self.config["regularization_strength"], (int, float)):
            raise ValueError("regularization_strength must be a number")

        if self.config["regularization_strength"] < 0:
            raise ValueError("regularization_strength must be >= 0")

        if not isinstance(self.config["acceptance_threshold"], (int, float)):
            raise ValueError("acceptance_threshold must be a number")

        # Validate each distribution
        for state_id, params in self.config["state_dwell_params"].items():
            if "distribution" not in params:
                raise ValueError(f"State {state_id} missing 'distribution' key")

            dist_type = params["distribution"]
            if dist_type not in ["gamma", "exponential", "lognormal"]:
                raise ValueError(f"Invalid distribution type: {dist_type}")

            if dist_type == "gamma":
                if "shape" not in params or "scale" not in params:
                    raise ValueError(f"Gamma distribution requires 'shape' and 'scale'")
            elif dist_type == "exponential":
                if "rate" not in params:
                    raise ValueError(f"Exponential distribution requires 'rate'")
            elif dist_type == "lognormal":
                if "mu" not in params or "sigma" not in params:
                    raise ValueError(
                        f"Lognormal distribution requires 'mu' and 'sigma'"
                    )

    def _prepare_distributions(self, num_states: int) -> np.ndarray:
        """
        Prepare distribution parameters for numba.

        Returns
        -------
        np.ndarray, shape (num_states, 3)
            Array of [dist_type, param1, param2]
        """
        dwell_dists = np.zeros((num_states, 3))

        for state_id, params in self.config["state_dwell_params"].items():
            if state_id >= num_states:
                continue

            dist_type = params["distribution"]

            if dist_type == "gamma":
                dwell_dists[state_id, 0] = 0  # Gamma
                dwell_dists[state_id, 1] = params["shape"]
                dwell_dists[state_id, 2] = params["scale"]
            elif dist_type == "exponential":
                dwell_dists[state_id, 0] = 1  # Exponential
                dwell_dists[state_id, 1] = params["rate"]
                dwell_dists[state_id, 2] = 0  # Unused
            else:  # lognormal
                dwell_dists[state_id, 0] = 2  # Lognormal
                dwell_dists[state_id, 1] = params["mu"]
                dwell_dists[state_id, 2] = params["sigma"]

        return dwell_dists

    def _prepare_transition_costs(self, num_states: int) -> np.ndarray:
        """
        Prepare transition costs matrix.

        Returns
        -------
        np.ndarray, shape (num_states, num_states)
            Transition costs
        """
        if self.config["transition_costs"] is None:
            # Default: uniform costs
            costs = np.ones((num_states, num_states))
            np.fill_diagonal(costs, 0)
            return costs

        if isinstance(self.config["transition_costs"], np.ndarray):
            return self.config["transition_costs"]

        # Dict format
        costs = np.ones((num_states, num_states))
        np.fill_diagonal(costs, 0)

        for (from_state, to_state), cost in self.config["transition_costs"].items():
            if from_state < num_states and to_state < num_states:
                costs[from_state, to_state] = cost

        return costs

    def _estimate_dwell_distributions(self, states: np.ndarray) -> None:
        """
        Estimate dwell time distributions from data.

        Parameters
        ----------
        states : np.ndarray
            State sequence to learn from
        """
        # Calculate dwell times for each state
        unique_states = np.unique(states)
        dwell_times = {int(s): [] for s in unique_states}

        current_state = states[0]
        current_duration = 1

        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                dwell_times[int(current_state)].append(current_duration)
                current_state = states[i]
                current_duration = 1

        dwell_times[int(current_state)].append(current_duration)

        # Fit distributions
        for state, durations in dwell_times.items():
            if len(durations) < self.config["min_samples"]:
                continue

            durations_arr = np.array(durations, dtype=np.float64)

            # Fit gamma distribution (good general choice)
            shape, loc, scale = stats.gamma.fit(durations_arr, floc=0)

            self.config["state_dwell_params"][state] = {
                "distribution": "gamma",
                "shape": shape,
                "scale": scale,
            }

    def process(self, states: Union[np.ndarray, pd.Series], posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        """
        Process state sequence (offline batch mode).

        Parameters
        ----------
        states : np.ndarray or pd.Series
            Input state sequence
        posteriors : optional
            Not used by this processor
        returns : optional
            Not used by this processor

        Returns
        -------
        np.ndarray or pd.Series
            Regularized state sequence
        """
        # Extract array and index
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        # Estimate distributions from data if requested
        if self.config["estimate_from_data"]:
            self._estimate_dwell_distributions(states_arr)

        # Determine number of states
        num_states = int(np.max(states_arr) + 1)

        # Prepare distributions and costs
        dwell_dists = self._prepare_distributions(num_states)
        trans_costs = self._prepare_transition_costs(num_states)

        # Apply regularization
        smoothed_arr = _markov_jump_regularizer_core(states_arr, dwell_dists, self.config["regularization_strength"], trans_costs, self.config["acceptance_threshold"])

        # Return in same format as input
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        # Initialize state on first call
        if self._online_state is None:
            # Determine number of states from config
            if self.config["state_dwell_params"]:
                num_states = max(self.config["state_dwell_params"].keys()) + 1
            else:
                num_states = state + 1

            self._online_state = {
                "current_state": state,
                "dwell_time": 1,
                "num_states": num_states,
                "dwell_dists": self._prepare_distributions(num_states),
                "trans_costs": self._prepare_transition_costs(num_states),
            }
            return state

        proposed_state = state
        current_state = self._online_state["current_state"]

        if proposed_state == current_state:
            # Continue in same state
            self._online_state["dwell_time"] += 1
            return current_state
        else:
            # Transition proposed - evaluate
            dwell_time = self._online_state["dwell_time"]
            dwell_dists = self._online_state["dwell_dists"]
            trans_costs = self._online_state["trans_costs"]

            # Evaluate dwell time likelihood
            dist_type = int(dwell_dists[current_state, 0])
            param1 = dwell_dists[current_state, 1]
            param2 = dwell_dists[current_state, 2]

            if dist_type == 0:  # Gamma
                log_prob = _gamma_logpdf(float(dwell_time), param1, param2)
            elif dist_type == 1:  # Exponential
                log_prob = _exponential_logpdf(float(dwell_time), param1)
            else:  # Lognormal
                log_prob = _lognormal_logpdf(float(dwell_time), param1, param2)

            # Get transition cost
            trans_cost = trans_costs[current_state, proposed_state]

            # Decision
            score = log_prob - self.config["regularization_strength"] * trans_cost

            if score > self.config["acceptance_threshold"]:
                # Accept transition
                self._online_state["current_state"] = proposed_state
                self._online_state["dwell_time"] = 1
                return proposed_state
            else:
                # Reject transition
                self._online_state["dwell_time"] += 1
                return current_state
