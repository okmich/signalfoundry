from functools import reduce
from operator import mul
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np

from .pomegranate import PomegranateHMM
from .pomegranate_mm import PomegranateMixtureHMM
from .util import InferenceMode


class FactorialHMM:
    """
    Multiple independent HMM chains running in parallel.

    Each chain models different aspects of observations, allowing for multi-factor regime decomposition (e.g., trend + volatility + volume).

    Parameters
    ----------
    chains : List[Union[PomegranateHMM, PomegranateMixtureHMM]]
        List of HMM instances, one per chain
    feature_assignment : Optional[List[List[int]]]
        Which features belong to which chain.
        Example: [[0,1], [2,3], [4]] assigns features 0-1 to chain 0,
                 features 2-3 to chain 1, feature 4 to chain 2

    Examples
    --------
    >>> from okmich.quant.models.hmm import FactorialHMM, PomegranateHMM, DistType
    >>>
    >>> # Create factorial HMM with 2 chains
    >>> factorial_hmm = FactorialHMM(
    ...     chains=[
    ...         PomegranateHMM(DistType.NORMAL, n_states=3),  # Trend regime
    ...         PomegranateHMM(DistType.STUDENTT, n_states=2)  # Volatility regime
    ...     ]
    ... )
    >>>
    >>> # Fit with feature assignment
    >>> factorial_hmm.fit(X, feature_assignment=[[0, 1, 2], [3, 4]])
    >>>
    >>> # Predict regimes
    >>> states = factorial_hmm.predict(X)  # shape: (n_samples, 2)
    >>> trend_regime = states[:, 0]
    >>> vol_regime = states[:, 1]
    """

    def __init__(self, chains: List[Union[PomegranateHMM, PomegranateMixtureHMM]],
                 feature_assignment: Optional[List[List[int]]] = None, inference_mode: Optional["InferenceMode"] = None):
        from . import InferenceMode  # Import here to avoid circular imports
        if not chains:
            raise ValueError("Must provide at least one HMM chain")

        # Set inference_mode with default to FILTERING
        self.inference_mode = inference_mode if inference_mode is not None else InferenceMode.FILTERING

        # Propagate inference_mode to all child chains
        for chain in chains:
            chain.inference_mode = self.inference_mode

        self.chains = chains
        self.feature_assignment = feature_assignment
        self._chain_labels: Dict[int, Dict[int, str]] = (
            {}
        )  # Chain idx -> {state idx -> label}
        self._is_fitted = False

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def n_states_per_chain(self) -> List[int]:
        return [chain.n_states for chain in self.chains]

    @property
    def n_total_joint_states(self) -> int:
        """Total number of joint states (Cartesian product)."""
        return reduce(mul, self.n_states_per_chain, 1)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        X: np.ndarray,
        lengths: Optional[Sequence[int]] = None,
        feature_assignment: Optional[List[List[int]]] = None,
    ) -> "FactorialHMM":
        if feature_assignment is not None:
            self.feature_assignment = feature_assignment

        if self.feature_assignment is None:
            raise ValueError(
                "feature_assignment must be provided either in constructor or fit()"
            )

        # Validate feature assignment
        self._validate_feature_assignment(X, self.feature_assignment)

        # Fit each chain independently
        for i, (chain, feat_idx) in enumerate(
            zip(self.chains, self.feature_assignment)
        ):
            X_chain = X[:, feat_idx]
            chain.fit(X_chain, lengths=lengths)

        self._is_fitted = True
        return self

    def train(
        self,
        X: np.ndarray,
        lengths: Optional[Sequence[int]] = None,
        feature_assignment: Optional[List[List[int]]] = None,
        n_states_range: Optional[List[Sequence[int]]] = None,
        criterion: str = "aic",
    ) -> "FactorialHMM":
        """
        Train with grid search over n_states for each chain.

        Examples
        --------
        >>> factorial_hmm = FactorialHMM(
        ...     chains=[PomegranateHMM(DistType.NORMAL), PomegranateHMM(DistType.NORMAL)]
        ... )
        >>> best = factorial_hmm.train(
        ...     X,
        ...     feature_assignment=[[0, 1], [2, 3]],
        ...     n_states_range=[(2, 4), (2, 3)],
        ...     criterion='bic'
        ... )
        """
        # Validate and set feature assignment
        if feature_assignment is not None:
            self.feature_assignment = feature_assignment

        if self.feature_assignment is None:
            raise ValueError(
                "feature_assignment must be provided either in constructor or train()"
            )

        # Validate feature assignment
        self._validate_feature_assignment(X, self.feature_assignment)

        # Train each chain independently
        best_chains = []
        for i, (chain, feat_idx) in enumerate(
            zip(self.chains, self.feature_assignment)
        ):
            X_chain = X[:, feat_idx]
            if n_states_range is not None and i < len(n_states_range):
                # Use provided state range
                state_range = n_states_range[i]
                best_chain = chain.train(
                    X_chain,
                    lengths=lengths,
                    n_states_range=state_range,
                    criterion=criterion,
                )
            else:
                # Use chain's default train()
                best_chain = chain.train(X_chain, lengths=lengths, criterion=criterion)
            best_chains.append(best_chain)

        # Create new FactorialHMM with optimally trained chains
        best_model = FactorialHMM(
            chains=best_chains, feature_assignment=self.feature_assignment.copy()
        )
        best_model._is_fitted = True
        return best_model

    def fit_predict(
        self,
        X: np.ndarray,
        lengths: Optional[Sequence[int]] = None,
        feature_assignment: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """
        Fit model and return predicted states.

        Parameters
        ----------
        X : np.ndarray
            Training data
        lengths : Optional[Sequence[int]]
            Sequence lengths
        feature_assignment : Optional[List[List[int]]]
            Feature assignment

        Returns
        -------
        states : np.ndarray, shape (n_samples, n_chains)
            Predicted states for each chain
        """
        self.fit(X, lengths=lengths, feature_assignment=feature_assignment)
        return self.predict(X)

    def predict(self, X: np.ndarray, return_joint: bool = False) -> np.ndarray:
        """
        Predict states for all chains.

        Parameters
        ----------
        X : np.ndarray
            Observations to predict
        return_joint : bool, default=False
            If True, returns combined joint state index.
            If False, returns separate states per chain.

        Returns
        -------
        states : np.ndarray
            If return_joint=False: shape (n_samples, n_chains)
            If return_joint=True: shape (n_samples,) with joint state indices
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # Predict for each chain
        chain_states = []
        for chain, feat_idx in zip(self.chains, self.feature_assignment):
            X_chain = X[:, feat_idx]
            states = chain.predict(X_chain)
            chain_states.append(states)

        # Stack as columns
        chain_states_array = np.column_stack(chain_states)

        if return_joint:
            # Encode as joint states
            return np.array(
                [self.encode_joint_state(tuple(s)) for s in chain_states_array]
            )
        else:
            return chain_states_array

    def predict_proba(
        self, X: np.ndarray, return_joint: bool = False
    ) -> Union[List[np.ndarray], np.ndarray]:
        """
        Predict state probabilities for all chains.

        Parameters
        ----------
        X : np.ndarray
            Observations
        return_joint : bool, default=False
            If True, returns joint probabilities via independence assumption.
            If False, returns list of per-chain probabilities.

        Returns
        -------
        probabilities : Union[List[np.ndarray], np.ndarray]
            If return_joint=False: List of arrays, each shape (n_samples, n_states_i)
            If return_joint=True: Array of shape (n_samples, n_total_joint_states)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # Predict probabilities for each chain
        chain_probas = []
        for chain, feat_idx in zip(self.chains, self.feature_assignment):
            X_chain = X[:, feat_idx]
            proba = chain.predict_proba(X_chain)
            chain_probas.append(proba)

        if return_joint:
            # Compute joint probabilities via independence assumption
            # P(s1, s2, ..., sn) = P(s1) * P(s2) * ... * P(sn)
            return self._compute_joint_probabilities(chain_probas)
        else:
            return chain_probas

    def get_aic_bic(self, X: np.ndarray) -> Tuple[float, float]:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        total_aic = 0.0
        total_bic = 0.0

        for chain, feat_idx in zip(self.chains, self.feature_assignment):
            X_chain = X[:, feat_idx]
            aic, bic = chain.get_aic_bic(X_chain)
            total_aic += aic
            total_bic += bic

        return total_aic, total_bic

    def score(self, X: np.ndarray) -> Tuple[float, float]:
        return self.get_aic_bic(X)

    def log_likelihood(self, X: np.ndarray) -> float:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        total_ll = 0.0
        for chain, feat_idx in zip(self.chains, self.feature_assignment):
            X_chain = X[:, feat_idx]
            # Use the chain's model log_probability method directly
            if hasattr(chain._model, "log_probability"):
                # For pomegranate models
                ll = chain._model.log_probability([X_chain])
                # Handle torch tensor conversion
                if hasattr(ll, "sum"):
                    ll = ll.sum()
                if hasattr(ll, "item"):
                    ll = ll.item()
                total_ll += float(ll)
            elif hasattr(chain._model, "score"):
                # For hmmlearn models
                total_ll += chain._model.score(X_chain)
            else:
                raise AttributeError(
                    f"Chain {type(chain).__name__} has no log_probability or score method"
                )

        return total_ll

    def transition_prob(self) -> List[np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        trans_mats = []
        for chain in self.chains:
            if hasattr(chain, "transition_prob"):
                trans_mat = chain.transition_prob()
            elif hasattr(chain, "trans_mat"):
                trans_mat = chain.trans_mat()
            else:
                raise AttributeError(
                    f"Chain {type(chain).__name__} has no transition probability method"
                )
            trans_mats.append(trans_mat)

        return trans_mats

    def get_chain_parameters(self, chain_idx: int) -> Dict[str, Any]:
        """
        Get parameters for a specific chain.

        Parameters
        ----------
        chain_idx : int
            Index of chain

        Returns
        -------
        parameters : Dict[str, Any]
            Chain parameters
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if chain_idx < 0 or chain_idx >= self.n_chains:
            raise IndexError(
                f"Chain index {chain_idx} out of range [0, {self.n_chains})"
            )

        chain = self.chains[chain_idx]
        params = {}

        # Try to extract parameters
        if hasattr(chain, "parameters"):
            params["parameters"] = chain.parameters
        if hasattr(chain, "means"):
            params["means"] = chain.means
        if hasattr(chain, "covariances"):
            params["covariances"] = chain.covariances
        elif hasattr(chain, "covars"):
            params["covariances"] = chain.covars()

        return params

    # ------------------------------------------------------------------
    # Advanced Features
    # ------------------------------------------------------------------
    def encode_joint_state(self, chain_states: Sequence[int]) -> int:
        """
        Encode individual chain states as joint state index.

        Uses mixed-radix numeral system.

        Parameters
        ----------
        chain_states : Sequence[int]
            State for each chain

        Returns
        -------
        joint_state : int
            Combined state index

        Examples
        --------
        >>> # With 3 states in chain 0 and 2 states in chain 1:
        >>> factorial_hmm.encode_joint_state((2, 1))
        5
        """
        if len(chain_states) != self.n_chains:
            raise ValueError(
                f"Expected {self.n_chains} states, got {len(chain_states)}"
            )

        joint_state = 0
        multiplier = 1

        for i, state in enumerate(chain_states):
            if state < 0 or state >= self.n_states_per_chain[i]:
                raise ValueError(
                    f"State {state} out of range for chain {i} "
                    f"(valid: 0-{self.n_states_per_chain[i] - 1})"
                )
            joint_state += state * multiplier
            multiplier *= self.n_states_per_chain[i]

        return joint_state

    def decode_joint_state(self, joint_state: int) -> Tuple[int, ...]:
        """
        Decode joint state index to individual chain states.

        Parameters
        ----------
        joint_state : int
            Combined state index

        Returns
        -------
        chain_states : Tuple[int, ...]
            State for each chain

        Examples
        --------
        >>> # With 3 states in chain 0 and 2 states in chain 1:
        >>> factorial_hmm.decode_joint_state(5)
        (2, 1)
        """
        if joint_state < 0 or joint_state >= self.n_total_joint_states:
            raise ValueError(
                f"Joint state {joint_state} out of range "
                f"[0, {self.n_total_joint_states})"
            )

        chain_states = []
        remaining = joint_state

        for n_states in self.n_states_per_chain:
            state = remaining % n_states
            chain_states.append(state)
            remaining //= n_states

        return tuple(chain_states)

    def set_chain_labels(self, chain_idx: int, labels: Dict[int, str]) -> None:
        if chain_idx < 0 or chain_idx >= self.n_chains:
            raise IndexError(
                f"Chain index {chain_idx} out of range [0, {self.n_chains})"
            )

        self._chain_labels[chain_idx] = labels

    def get_chain_label(self, chain_idx: int, state: int) -> str:
        if chain_idx in self._chain_labels and state in self._chain_labels[chain_idx]:
            return self._chain_labels[chain_idx][state]
        return f"State {state}"

    def regime_summary(self, X: Optional[np.ndarray] = None) -> str:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        lines = []
        lines.append("=" * 70)
        lines.append("Factorial HMM Summary")
        lines.append("=" * 70)
        lines.append(f"Number of chains: {self.n_chains}")
        lines.append(f"States per chain: {self.n_states_per_chain}")
        lines.append(f"Total joint states: {self.n_total_joint_states}")
        lines.append("")

        # Get states if data provided
        chain_states_list = None
        if X is not None:
            chain_states_list = [
                chain.predict(X[:, feat_idx])
                for chain, feat_idx in zip(self.chains, self.feature_assignment)
            ]

        # Summary for each chain
        for i, chain in enumerate(self.chains):
            lines.append(f"Chain {i}: {type(chain).__name__}")
            lines.append(f"  Number of states: {self.n_states_per_chain[i]}")

            # State occupancy
            if chain_states_list is not None:
                states = chain_states_list[i]
                lines.append("  State Occupancy:")
                for state in range(self.n_states_per_chain[i]):
                    count = np.sum(states == state)
                    pct = 100 * count / len(states)
                    label = self.get_chain_label(i, state)
                    lines.append(f"    {label:20s}: {count:6d} samples ({pct:5.2f}%)")
                lines.append("")

            # Transition probabilities
            trans_mat = self.transition_prob()[i]
            lines.append("  Transition Probabilities:")
            for state_from in range(self.n_states_per_chain[i]):
                probs = " ".join([f"{p:6.3f}" for p in trans_mat[state_from]])
                label = self.get_chain_label(i, state_from)
                lines.append(f"    From {label:15s}: [{probs}]")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def plot_distributions(
        self,
        X: Optional[np.ndarray] = None,
        chain_idx: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> Tuple:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if chain_idx is not None:
            # Plot single chain
            if chain_idx < 0 or chain_idx >= self.n_chains:
                raise IndexError(
                    f"Chain index {chain_idx} out of range [0, {self.n_chains})"
                )

            chain = self.chains[chain_idx]
            feat_idx = self.feature_assignment[chain_idx]
            X_chain = X[:, feat_idx] if X is not None else None

            # Call chain's plot_distributions if available
            if hasattr(chain, "plot_distributions"):
                return chain.plot_distributions(X_chain, **kwargs)
            else:
                raise NotImplementedError(
                    f"Chain {type(chain).__name__} does not support plot_distributions()"
                )
        else:
            # Plot all chains
            if figsize is None:
                figsize = (14, 4 * self.n_chains)

            fig, axes = plt.subplots(self.n_chains, 1, figsize=figsize, squeeze=False)
            axes = axes.flatten()

            for i, chain in enumerate(self.chains):
                feat_idx = self.feature_assignment[i]
                X_chain = X[:, feat_idx] if X is not None else None

                if hasattr(chain, "plot_distributions"):
                    # Create subplot for this chain
                    chain_fig, _ = chain.plot_distributions(X_chain, **kwargs)

                    # Copy to our axes (this is tricky - alternative: create separate figs)
                    axes[i].set_title(f"Chain {i}: {type(chain).__name__}")
                    axes[i].text(
                        0.5,
                        0.5,
                        f"See separate figure for Chain {i}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
                    plt.close(chain_fig)  # Close individual figure
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"Chain {i}: {type(chain).__name__}\n(no plot_distributions support)",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )

            fig.suptitle("Factorial HMM - State Distributions", fontsize=14, y=0.995)
            plt.tight_layout()

            return fig, axes

    def plot_transition_matrices(
        self, figsize: Tuple[int, int] = (12, 5), cmap: str = "Blues"
    ) -> Tuple:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        fig, axes = plt.subplots(1, self.n_chains, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        trans_mats = self.transition_prob()

        for i, trans_mat in enumerate(trans_mats):
            ax = axes[i]
            n_states = self.n_states_per_chain[i]

            # Plot heatmap
            im = ax.imshow(trans_mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability", rotation=270, labelpad=15)

            # Add text annotations
            for row in range(n_states):
                for col in range(n_states):
                    text_color = "black" if trans_mat[row, col] < 0.5 else "white"
                    ax.text(
                        col,
                        row,
                        f"{trans_mat[row, col]:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                    )

            # Labels
            state_labels = [self.get_chain_label(i, s) for s in range(n_states)]
            ax.set_xticks(np.arange(n_states))
            ax.set_yticks(np.arange(n_states))
            ax.set_xticklabels(state_labels, rotation=45, ha="right")
            ax.set_yticklabels(state_labels)
            ax.set_xlabel("To State")
            ax.set_ylabel("From State")
            ax.set_title(f"Chain {i}: {type(self.chains[i]).__name__}")

        fig.suptitle("Factorial HMM - Transition Matrices", fontsize=14, y=1.02)
        plt.tight_layout()
        return fig, axes

    def plot_regime_timeline(
        self,
        X: np.ndarray,
        figsize: Tuple[int, int] = (14, 8),
        time_index: Optional[np.ndarray] = None,
    ) -> Tuple:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # Predict states for all chains
        states = self.predict(X, return_joint=False)
        n_samples = states.shape[0]

        if time_index is None:
            time_index = np.arange(n_samples)

        fig, axes = plt.subplots(
            self.n_chains, 1, figsize=figsize, sharex=True, squeeze=False
        )
        axes = axes.flatten()

        # Define colors for states (cycle through if needed)
        colors = plt.cm.Set3(np.linspace(0, 1, 12))

        for i in range(self.n_chains):
            ax = axes[i]
            chain_states = states[:, i]
            n_states = self.n_states_per_chain[i]

            # Create color map for this chain
            state_colors = [colors[s % len(colors)] for s in range(n_states)]

            # Plot as filled regions
            for state in range(n_states):
                mask = chain_states == state
                # Create segments
                segments_start = []
                segments_end = []

                in_segment = False
                for j in range(n_samples):
                    if mask[j] and not in_segment:
                        segments_start.append(j)
                        in_segment = True
                    elif not mask[j] and in_segment:
                        segments_end.append(j - 1)
                        in_segment = False

                if in_segment:
                    segments_end.append(n_samples - 1)

                # Draw segments
                label = self.get_chain_label(i, state)
                for start, end in zip(segments_start, segments_end):
                    ax.fill_between(
                        time_index[start : end + 1],
                        i - 0.4,
                        i + 0.4,
                        color=state_colors[state],
                        alpha=0.7,
                        label=label if start == segments_start[0] else None,
                    )

            # Also draw as step line for clarity
            ax.step(
                time_index,
                chain_states,
                where="post",
                color="black",
                linewidth=1,
                alpha=0.5,
            )

            ax.set_ylabel(f"Chain {i}")
            ax.set_yticks(range(n_states))
            ax.set_yticklabels([self.get_chain_label(i, s) for s in range(n_states)])
            ax.set_title(f"Chain {i}: {type(self.chains[i]).__name__}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, ncol=n_states)

        axes[-1].set_xlabel("Time")
        fig.suptitle("Factorial HMM - Regime Timeline", fontsize=14, y=0.995)
        plt.tight_layout()

        return fig, axes

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FactorialHMM":
        return joblib.load(path)

    def __repr__(self) -> str:
        chain_types = [type(chain).__name__ for chain in self.chains]
        return (
            f"FactorialHMM("
            f"n_chains={self.n_chains}, "
            f"states={self.n_states_per_chain}, "
            f"types={chain_types})"
        )

    def _validate_feature_assignment(
        self, X: np.ndarray, feature_assignment: List[List[int]]
    ) -> None:
        """Validate feature_assignment is consistent with X and chains."""
        n_features = X.shape[1]

        # Check all features are assigned
        all_features = set(sum(feature_assignment, []))
        if all_features != set(range(n_features)):
            raise ValueError(
                f"feature_assignment must cover all {n_features} features. "
                f"Missing: {set(range(n_features)) - all_features}, "
                f"Extra: {all_features - set(range(n_features))}"
            )

        # Check no overlap
        if len(all_features) != sum(len(f) for f in feature_assignment):
            raise ValueError("Features cannot be assigned to multiple chains")

        # Check number of chains matches
        if len(feature_assignment) != len(self.chains):
            raise ValueError(
                f"feature_assignment has {len(feature_assignment)} groups "
                f"but {len(self.chains)} chains"
            )

    def _compute_joint_probabilities(
        self, chain_probas: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute joint probabilities via independence assumption.

        P(s1, s2, ..., sn) = P(s1) * P(s2) * ... * P(sn)

        Parameters
        ----------
        chain_probas : List[np.ndarray]
            List of probability arrays, each shape (n_samples, n_states_i)

        Returns
        -------
        joint_probas : np.ndarray, shape (n_samples, n_total_joint_states)
            Joint probability distribution
        """
        n_samples = chain_probas[0].shape[0]
        n_joint_states = self.n_total_joint_states

        # Initialize joint probabilities
        joint_probas = np.zeros((n_samples, n_joint_states))

        # Iterate over all joint state combinations
        for joint_idx in range(n_joint_states):
            chain_states = self.decode_joint_state(joint_idx)

            # Compute product of probabilities
            prob = np.ones(n_samples)
            for chain_idx, state in enumerate(chain_states):
                prob *= chain_probas[chain_idx][:, state]

            joint_probas[:, joint_idx] = prob

        # Explicit renormalization to handle numerical precision issues
        # This ensures probabilities sum exactly to 1.0
        row_sums = joint_probas.sum(axis=1, keepdims=True)
        joint_probas = joint_probas / np.maximum(row_sums, 1e-300)

        return joint_probas
