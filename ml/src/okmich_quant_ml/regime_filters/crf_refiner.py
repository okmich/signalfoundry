from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from numba import njit

from .base import BasePostProcessor


# =============================================================================
# Numba-optimized core functions
# =============================================================================


@njit(cache=True)
def _viterbi_decode(unary_costs: np.ndarray, transition_costs: np.ndarray) -> np.ndarray:
    """
    Viterbi algorithm for finding MAP state sequence.

    Parameters
    ----------
    unary_costs : np.ndarray, shape (T, K)
        Unary costs for each state at each timestep
        (lower is better)
    transition_costs : np.ndarray, shape (K, K)
        Pairwise transition costs from state i to state j
        (lower is better)

    Returns
    -------
    np.ndarray, shape (T,)
        Most likely state sequence (MAP estimate)
    """
    T, K = unary_costs.shape

    # Dynamic programming tables
    dp = np.zeros((T, K))  # Best cost to reach each state
    backpointer = np.zeros((T, K), dtype=np.int32)  # Backpointers for path

    # Initialize first timestep
    dp[0, :] = unary_costs[0, :]

    # Forward pass
    for t in range(1, T):
        for curr_state in range(K):
            # Find best previous state
            min_cost = np.inf
            best_prev = 0

            for prev_state in range(K):
                cost = (
                    dp[t - 1, prev_state]
                    + transition_costs[prev_state, curr_state]
                    + unary_costs[t, curr_state]
                )

                if cost < min_cost:
                    min_cost = cost
                    best_prev = prev_state

            dp[t, curr_state] = min_cost
            backpointer[t, curr_state] = best_prev

    # Backward pass: trace back best path
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmin(dp[T - 1, :])

    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path


@njit(cache=True)
def _compute_unary_costs(posteriors: np.ndarray, observation_weight: float) -> np.ndarray:
    """
    Convert posteriors to unary costs.

    Parameters
    ----------
    posteriors : np.ndarray, shape (T, K)
        Posterior probabilities
    observation_weight : float
        Weight for observation term

    Returns
    -------
    np.ndarray, shape (T, K)
        Unary costs (negative log probabilities)
    """
    # Cost = -weight * log(probability)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    costs = -observation_weight * np.log(posteriors + eps)
    return costs


# =============================================================================
# ConditionalRandomFieldRefiner
# =============================================================================


class ConditionalRandomFieldRefiner(BasePostProcessor):
    """
    Apply learned pairwise potentials for principled smoothing.

    This processor formulates regime smoothing as a Conditional Random
    Field (CRF) inference problem, balancing:
    - Unary potentials: How well each state explains the observation (from HMM)
    - Pairwise potentials: Preference for certain transitions

    Uses Viterbi algorithm to find the Maximum A Posteriori (MAP) sequence.

    Parameters
    ----------
    config : dict
        Configuration with keys:
        - transition_weights : np.ndarray or 'learn', optional
            Transition cost matrix (K, K) or 'learn' to estimate
            Lower values = preferred transitions
        - observation_weight : float, default=1.0
            Weight for HMM posterior term (unary potentials)
        - inference_method : str, default='viterbi'
            Inference algorithm ('viterbi' only for now)
        - smooth_transitions : float, default=0.0
            Smoothness preference (higher = smoother)

    Warning
    -------
    This processor is FUNDAMENTALLY NON-CAUSAL and NOT suitable for live trading.

    The Viterbi algorithm requires seeing the entire sequence to compute the globally optimal state path
    (backward pass from final timestep). This creates look-ahead bias that inflates backtesting performance.

    Use Cases:
    - ✓ Offline analysis and research
    - ✓ Post-hoc regime labeling
    - ✗ Live trading (use HysteresisProcessor or TransitionRateLimiter instead)
    - ✗ Realistic backtesting (will overstate performance)

    Online Mode Note:
    The online mode uses a 20-step sliding window with Viterbi, which reduces but does not eliminate
    look-ahead bias (looks ahead ~19 observations).

    Examples
    --------
    >>> # Manual transition preferences
    >>> transition_weights = np.array([
    ...     [0, 5, 10],   # From state 0: prefer staying, then 1, then 2
    ...     [5, 0, 5],    # From state 1: symmetric
    ...     [10, 5, 0]    # From state 2: prefer 1 over 0
    ... ])
    >>> processor = ConditionalRandomFieldRefiner({
    ...     'transition_weights': transition_weights,
    ...     'observation_weight': 1.0
    ... })

    Notes
    -----
    For trading applications:
    - Lower transition_weights = preferred/cheap transitions
    - Higher transition_weights = penalized/expensive transitions
    - observation_weight controls trust in HMM vs transitions
    - Set smooth_transitions > 0 to penalize all transitions

    The CRF formulation provides principled smoothing by finding the globally optimal sequence, not just greedy local decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ConditionalRandomFieldRefiner."""
        if config is None:
            config = {}

        # Set defaults
        config.setdefault("transition_weights", None)
        config.setdefault("observation_weight", 1.0)
        config.setdefault("inference_method", "viterbi")
        config.setdefault("smooth_transitions", 0.0)

        # Warn about non-causal nature
        import warnings
        warnings.warn(
            "ConditionalRandomFieldRefiner uses Viterbi algorithm which requires "
            "the full sequence (look-ahead bias). This processor is NOT suitable "
            "for live trading. Use only for offline analysis and research.",
            UserWarning,
            stacklevel=2
        )

        super().__init__(config)

        self._learned_weights: Optional[np.ndarray] = None

    def validate_config(self) -> None:
        """Validate configuration parameters."""
        if not isinstance(self.config["observation_weight"], (int, float)):
            raise ValueError("observation_weight must be a number")

        if self.config["observation_weight"] <= 0:
            raise ValueError("observation_weight must be positive")

        if self.config["inference_method"] not in ["viterbi"]:
            raise ValueError("Only 'viterbi' inference is currently supported")

        if not isinstance(self.config["smooth_transitions"], (int, float)):
            raise ValueError("smooth_transitions must be a number")

        if self.config["smooth_transitions"] < 0:
            raise ValueError("smooth_transitions must be >= 0")

    def _get_transition_weights(self, num_states: int) -> np.ndarray:
        """
        Get or create transition weight matrix.

        Parameters
        ----------
        num_states : int
            Number of states

        Returns
        -------
        np.ndarray, shape (K, K)
            Transition cost matrix
        """
        if self.config["transition_weights"] is None:
            # Default: penalize transitions equally
            weights = (
                np.ones((num_states, num_states)) * self.config["smooth_transitions"]
            )
            np.fill_diagonal(weights, 0)  # No cost for staying
            return weights

        if isinstance(self.config["transition_weights"], str):
            if self.config["transition_weights"] == "learn":
                # Use learned weights if available
                if self._learned_weights is not None:
                    return self._learned_weights
                else:
                    # Fall back to default
                    weights = (
                        np.ones((num_states, num_states))
                        * self.config["smooth_transitions"]
                    )
                    np.fill_diagonal(weights, 0)
                    return weights
            else:
                raise ValueError(
                    f"Unknown transition_weights value: {self.config['transition_weights']}"
                )

        # User-provided weights
        weights = self.config["transition_weights"]
        if weights.shape != (num_states, num_states):
            raise ValueError(
                f"transition_weights shape {weights.shape} doesn't match num_states {num_states}"
            )

        return weights

    def learn_transition_weights(self, states: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Learn transition weights from labeled data.

        Parameters
        ----------
        states : np.ndarray
            Raw state predictions
        ground_truth : np.ndarray
            True state labels

        Returns
        -------
        np.ndarray, shape (K, K)
            Learned transition weights
        """
        num_states = int(max(np.max(states), np.max(ground_truth)) + 1)

        # Count transitions in ground truth
        transition_counts = np.zeros((num_states, num_states))

        for i in range(len(ground_truth) - 1):
            from_state = int(ground_truth[i])
            to_state = int(ground_truth[i + 1])
            transition_counts[from_state, to_state] += 1

        # Convert counts to costs (inverse relationship)
        # More frequent transitions get lower costs
        total_counts = np.sum(transition_counts, axis=1, keepdims=True)
        transition_probs = np.zeros_like(transition_counts)

        for i in range(num_states):
            if total_counts[i, 0] > 0:
                transition_probs[i, :] = transition_counts[i, :] / total_counts[i, 0]
            else:
                transition_probs[i, :] = 1.0 / num_states

        # Cost = -log(probability)
        eps = 1e-10
        transition_costs = -np.log(transition_probs + eps)

        # Normalize to reasonable range
        transition_costs = (transition_costs - np.min(transition_costs)) / (
            np.max(transition_costs) - np.min(transition_costs) + eps
        )

        self._learned_weights = transition_costs
        return transition_costs

    def process(self, states: Union[np.ndarray, pd.Series],
                posteriors: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                returns: Optional[Union[np.ndarray, pd.Series]] = None) -> Union[np.ndarray, pd.Series]:
        """
        Process state sequence (offline batch mode).
        Raises
        ------
        ValueError
            If posteriors are not provided
        """
        if posteriors is None:
            raise ValueError("ConditionalRandomFieldRefiner requires posteriors")

        # Extract arrays and index
        if isinstance(states, pd.Series):
            states_arr = states.values
            index = states.index
        else:
            states_arr = states
            index = None

        if isinstance(posteriors, pd.DataFrame):
            posteriors_arr = posteriors.values
        else:
            posteriors_arr = posteriors

        # Get number of states
        num_states = posteriors_arr.shape[1]

        # Compute unary costs from posteriors
        unary_costs = _compute_unary_costs(
            posteriors_arr, self.config["observation_weight"]
        )

        # Get transition costs
        transition_costs = self._get_transition_weights(num_states)

        # Run Viterbi to find MAP sequence
        smoothed_arr = _viterbi_decode(unary_costs, transition_costs)

        # Return in same format as input
        if index is not None:
            return pd.Series(smoothed_arr, index=index)
        else:
            return smoothed_arr

    def process_online(self, state: int, posterior: Optional[np.ndarray] = None, return_value: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> int:
        """
        Process single state observation (online streaming mode).

        Warning
        -------
        This online mode uses a 20-step sliding window with Viterbi
        decoding, which looks ahead ~19 observations. While more causal than
        batch Viterbi, it still has look-ahead bias unsuitable for live trading.

        Raises
        ------
        ValueError
            If posterior is not provided
        """
        if posterior is None:
            raise ValueError("ConditionalRandomFieldRefiner requires posteriors")

        # Initialize state on first call
        if self._online_state is None:
            num_states = len(posterior)
            window_size = 20  # Fixed window for online processing

            self._online_state = {
                "state_buffer": np.zeros(window_size, dtype=np.int32),
                "posterior_buffer": np.zeros((window_size, num_states)),
                "index": 0,
                "count": 0,
                "num_states": num_states,
                "window_size": window_size,
                "transition_costs": self._get_transition_weights(num_states),
            }

        # Add to buffer
        buffer_idx = self._online_state["index"] % self._online_state["window_size"]
        self._online_state["state_buffer"][buffer_idx] = state
        self._online_state["posterior_buffer"][buffer_idx, :] = posterior
        self._online_state["index"] += 1
        self._online_state["count"] = min(
            self._online_state["count"] + 1, self._online_state["window_size"]
        )

        # Run Viterbi on current window
        active_count = self._online_state["count"]
        if active_count < self._online_state["window_size"]:
            active_posteriors = self._online_state["posterior_buffer"][:active_count, :]
        else:
            # Reorder circular buffer
            idx = self._online_state["index"]
            active_posteriors = np.vstack(
                [
                    self._online_state["posterior_buffer"][
                        idx % self._online_state["window_size"] :, :
                    ],
                    self._online_state["posterior_buffer"][
                        : idx % self._online_state["window_size"], :
                    ],
                ]
            )

        # Compute unary costs
        unary_costs = _compute_unary_costs(
            active_posteriors, self.config["observation_weight"]
        )

        # Run Viterbi
        path = _viterbi_decode(unary_costs, self._online_state["transition_costs"])

        # Return the most recent state from the path
        return int(path[-1])
