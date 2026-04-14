from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from pomegranate.hmm import DenseHMM
from scipy.special import logsumexp

from .util import DistType, InferenceMode


class BasePomegranateHMM(ABC):
    """
    Abstract base class for Pomegranate HMM wrappers.

    Provides common functionality for HMM training, prediction, and analysis.
    Subclasses must implement model-specific methods like _build_model() and get_aic_bic().
    """

    _REMOVED_HSMM_KWARGS: frozenset = frozenset({"duration_model", "duration_type", "max_duration"})

    def __init__(self, distribution_type: DistType, n_states: int = 2, *, random_state: int = 100,
                 max_iter: int = 100, inference_mode: Optional[InferenceMode] = None, **dist_kwargs):
        """
        Parameters
        ----------
        distribution_type : DistType
        n_states : int
            Initial / fixed number of hidden states.
        max_iter : int
            Maximum number of iterations for EM algorithm (default: 100)
        inference_mode : InferenceMode, default=InferenceMode.FILTERING
            Inference algorithm to use for predictions:
            - FILTERING: Forward algorithm only (no look-ahead bias)
            - SMOOTHING: Forward-Backward algorithm (uses future info, for analysis)
            - VITERBI: Most likely state sequence (uses future info, for labeling)
        dist_kwargs
            Extra arguments forwarded to the actual pomegranate
            distribution constructors (e.g. n_components for gmm).
        """
        bad = self._REMOVED_HSMM_KWARGS & dist_kwargs.keys()
        if bad:
            raise TypeError(
                f"HSMM support has been removed. The following kwargs are no longer accepted: {sorted(bad)}. "
                "Remove them from the call."
            )
        self.distribution_type = distribution_type
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter
        self.inference_mode = inference_mode if inference_mode is not None else InferenceMode.FILTERING
        self.dist_kwargs = dist_kwargs
        self._model: Optional[DenseHMM] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # Maximum retries for EM fitting when Cholesky / covariance errors occur
    _FIT_MAX_RETRIES: int = 3

    def _cov_param_count(self, n_features: int) -> int:
        """Number of covariance parameters per state for distributions that carry one.

        Returns 0 for distributions without a covariance structure (Gamma, Exponential,
        Poisson, etc.). For Normal / StudentT / LogNormal the count depends on the
        covariance type stored in dist_kwargs:
            diag   → n_features   (one variance per feature)
            sphere → 1            (single shared variance)
            full   → n_features * (n_features + 1) / 2  (lower-triangle of Σ)
        """
        if self.distribution_type not in {DistType.NORMAL, DistType.STUDENTT, DistType.LOGNORMAL}:
            return 0
        cov_type = str(self.dist_kwargs.get("covariance_type", "full")).lower()
        if cov_type == "diag":
            return n_features
        if cov_type == "sphere":
            return 1
        return n_features * (n_features + 1) // 2  # full

    def fit(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> "BasePomegranateHMM":
        """
        Fit the HMM to data.

        Before EM, k-means is run on the full dataset to produce n_states
        cluster centroids and per-cluster covariances. These are used to
        initialize the emission distribution parameters, placing EM in a
        sensible region of parameter space from the start. This virtually
        eliminates degenerate convergence (state collapse) at scale, at the
        cost of a single cheap k-means pass (~100ms on 40K × 9 features).

        If EM fails due to non-positive-definite covariance (Cholesky error)
        or degenerate emission parameters, the fit is retried up to
        _FIT_MAX_RETRIES times with perturbed random seeds. This is common
        with mixture emissions on near-collinear or low-variance features.
        """
        X_list = self._split_sequences(X, lengths)
        X_all = X if isinstance(X, np.ndarray) else np.vstack(X_list)

        # Validate input before expensive EM iterations
        if not np.all(np.isfinite(X_all)):
            raise ValueError("X contains NaN or Inf values. Clean input data before fitting.")

        original_seed = self.random_state
        last_error = None
        for attempt in range(self._FIT_MAX_RETRIES):
            self.random_state = original_seed + attempt
            self._kmeans_stats = self._compute_kmeans_init(X_all)
            self._model = self._build_model()

            try:
                self._model.fit(X_list)
                # Success — restore original seed for reproducibility and return
                self.random_state = original_seed
                del self._kmeans_stats
                return self
            except (np.linalg.LinAlgError, AttributeError, RuntimeError) as e:
                if not self._is_covariance_error(e):
                    self.random_state = original_seed
                    del self._kmeans_stats
                    raise
                last_error = e
                self._model = None

        # All retries exhausted — restore original seed and raise
        self.random_state = original_seed
        del self._kmeans_stats
        raise RuntimeError(
            f"EM fitting failed after {self._FIT_MAX_RETRIES} attempts due to covariance/Cholesky errors. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    @staticmethod
    def _is_covariance_error(e: Exception) -> bool:
        """Check if an exception is related to covariance/Cholesky decomposition failure."""
        if isinstance(e, np.linalg.LinAlgError):
            return True
        msg = str(e).lower()
        covariance_keywords = ("cholesky", "positive definite", "positive-definite", "_inv_cov", "covariance", "singular", "linalg")
        return any(kw in msg for kw in covariance_keywords)

    def fit_predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """Fit model and return predicted states."""
        self.fit(X, lengths)
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hidden states based on inference_mode.

        - FILTERING: argmax of filtered probabilities (causal)
        - SMOOTHING: argmax of smoothed probabilities (non-causal)
        - VITERBI: most likely state sequence (Viterbi algorithm)
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._preprocess_input(X)

        if self.inference_mode == InferenceMode.VITERBI:
            predictions = self._model.predict([X]).flatten()
            # Convert torch tensor to numpy
            if hasattr(predictions, "detach"):
                return predictions.detach().cpu().numpy()
            return predictions
        elif self.inference_mode == InferenceMode.FILTERING:
            return np.argmax(self._predict_proba_filtered(X), axis=1)
        elif self.inference_mode == InferenceMode.SMOOTHING:
            proba = self._model.predict_proba([X])
            if hasattr(proba, "detach"):
                proba = proba.detach().cpu().numpy()
            if isinstance(proba, np.ndarray) and proba.ndim == 3 and proba.shape[0] == 1:
                proba = proba[0]
            return np.argmax(proba, axis=1)
        else:
            raise ValueError(f"Unknown inference_mode: {self.inference_mode}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict state probabilities based on inference_mode.

        - FILTERING: filtered probabilities using only observations up to time t (causal)
        - SMOOTHING: smoothed probabilities using all observations (non-causal)
        - VITERBI: raises ValueError (not applicable for probabilities)
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._preprocess_input(X)

        if self.inference_mode == InferenceMode.VITERBI:
            raise ValueError(
                "predict_proba() is not applicable with inference_mode=VITERBI. "
                "Use predict() instead, or set inference_mode to FILTERING or SMOOTHING."
            )

        if self.inference_mode == InferenceMode.FILTERING:
            return self._predict_proba_filtered(X)
        elif self.inference_mode == InferenceMode.SMOOTHING:
            proba = self._model.predict_proba([X])
            if hasattr(proba, "detach"):
                proba = proba.detach().cpu().numpy()
            if isinstance(proba, np.ndarray) and proba.ndim == 3 and proba.shape[0] == 1:
                proba = proba[0]
            return proba
        else:
            raise ValueError(f"Unknown inference_mode: {self.inference_mode}")

    def _predict_proba_filtered(self, X: np.ndarray) -> np.ndarray:
        """
        Compute filtered state probabilities using only forward algorithm (causal). This ensures that the probability
        estimate at time t depends only on observations up to time t, preventing temporal leakage.
        """
        # Use pomegranate's forward method to return log probabilities
        log_probs = self._model.forward([X])

        # Convert to numpy and normalize
        if hasattr(log_probs, "detach"):
            log_probs = log_probs.detach().cpu().numpy()

        # Remove batch dimension if present
        if log_probs.ndim == 3 and log_probs.shape[0] == 1:
            log_probs = log_probs[0]

        filtered_probs = np.exp(log_probs - logsumexp(log_probs, axis=1, keepdims=True))

        # Explicit renormalization to handle numerical precision issues with float32
        # This ensures probabilities sum exactly to 1.0
        row_sums = filtered_probs.sum(axis=1, keepdims=True)
        filtered_probs = filtered_probs / row_sums

        return filtered_probs

    def state_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior state probabilities: shape (n_samples, n_states).

        (Formerly misnamed ``log_likelihood``.)
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.predict_proba([X])

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Scalar log-likelihood of the sequence X under the fitted model.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._preprocess_input(X)

        ll = self._model.log_probability([X])
        if hasattr(ll, "sum"):
            ll = ll.sum()
        if hasattr(ll, "item"):
            ll = ll.item()
        return float(ll)

    def transition_prob(self) -> np.ndarray:
        """Transition matrix (n_states, n_states)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        edges = self._model.edges

        # Convert torch tensor to numpy if needed
        if hasattr(edges, "detach"):
            edges = edges.detach().cpu().numpy()

        # Pomegranate stores log probabilities, convert to probabilities
        # Check if values are in log space (negative values)
        if np.any(edges < 0):
            edges = np.exp(edges)

        # Normalize rows to ensure they sum to 1 (handle numerical precision)
        row_sums = edges.sum(axis=1, keepdims=True)
        edges = edges / np.maximum(row_sums, 1e-300)

        return edges

    def forecast(self, current_regime: int, n_steps: int = 1) -> np.ndarray:
        """
        Forecast future regime probabilities using transition matrix.

        Parameters
        ----------
        current_regime : int
            Current regime label (0 to n_states-1)
        n_steps : int, default=1
            Number of steps ahead to forecast

        Returns
        -------
        np.ndarray
            Array of regime probabilities n_steps ahead, shape (n_states,)

        Examples
        --------
        >>> model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=3)
        >>> model.fit(data)
        >>> # If currently in regime 1, what's probability distribution in 5 steps?
        >>> probs = model.forecast(current_regime=1, n_steps=5)
        >>> print(probs)  # [0.2, 0.5, 0.3] - probabilities for regimes 0, 1, 2
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if not (0 <= current_regime < self.n_states):
            raise ValueError(f"current_regime must be between 0 and {self.n_states-1}")

        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")

        # Start with one-hot encoding of current regime
        current_state = np.zeros(self.n_states)
        current_state[current_regime] = 1.0

        # Get transition matrix
        trans_mat = self.transition_prob()

        # Matrix multiply by transition matrix n_steps times
        future_probs = current_state
        for _ in range(n_steps):
            future_probs = future_probs @ trans_mat

        return future_probs

    def get_stationary_distribution(self) -> np.ndarray:
        """
        Get the stationary/equilibrium distribution of regimes.

        This represents the long-run probability of being in each regime, independent of the starting state.

        Returns
        -------
        np.ndarray
            Stationary distribution, shape (n_states,)

        Examples
        --------
        >>> model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=3)
        >>> model.fit(data)
        >>> stationary = model.get_stationary_distribution()
        >>> print(stationary)  # [0.25, 0.50, 0.25] - long-run regime probabilities
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        trans_mat = self.transition_prob()

        # Find stationary distribution by computing eigenvector
        # corresponding to eigenvalue 1 of the transpose
        eigenvalues, eigenvectors = np.linalg.eig(trans_mat.T)

        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to sum to 1
        stationary = stationary / stationary.sum()

        return stationary

    def expected_duration(self, regime: int) -> float:
        """
        Calculate expected duration (number of time steps) in a regime.

        This tells you how long the process is expected to stay in a given regime before transitioning to another regime.

        Parameters
        ----------
        regime : int
            Regime index (0 to n_states-1)

        Returns
        -------
        float
            Expected number of time steps in the regime

        Examples
        --------
        >>> model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=3)
        >>> model.fit(data)
        >>> duration = model.expected_duration(regime=1)
        >>> print(f"Expected to stay in regime 1 for {duration:.1f} periods")
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if not (0 <= regime < self.n_states):
            raise ValueError(f"regime must be between 0 and {self.n_states-1}")

        trans_mat = self.transition_prob()

        # Expected duration = 1 / (1 - self_transition_probability)
        self_transition_prob = trans_mat[regime, regime]

        if self_transition_prob >= 1.0:
            return np.inf

        return 1.0 / (1.0 - self_transition_prob)

    # ------------------------------------------------------------------
    # Fixed-lag smoothing
    # ------------------------------------------------------------------
    def _extract_hmm_parameters(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract log initial probs, log transition matrix, and log emissions.

        Returns (log_pi, log_A, log_B) where:
            log_pi : (K,)    — log initial state probabilities
            log_A  : (K, K)  — log transition matrix
            log_B  : (T, K)  — log emission probabilities per timestep
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        log_B = self._compute_log_emissions(X)

        def _to_log_numpy(param):
            arr = param
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            arr = np.asarray(arr, dtype=np.float64)
            if np.all(arr >= 0):
                arr = np.log(np.maximum(arr, 1e-300))
            return arr

        log_A = _to_log_numpy(self._model.edges)
        log_pi = _to_log_numpy(self._model.starts)

        return log_pi, log_A, log_B

    def predict_proba_fixed_lag(self, X: np.ndarray, lag: int) -> np.ndarray:
        """Fixed-lag posterior probabilities with streaming/open-end semantics. Shape (T, n_states).

        Computes P(z_t = k | o_1, ..., o_{min(t+lag, T-1)}) for each t.

        Uses open-end semantics: the backward frontier is always initialized to
        uniform (log_beta = 0), meaning "no information about what comes after
        the lag window." This matches live trading where the sequence has no
        known terminal point.

        This is intentionally NOT equivalent to pomegranate's full smoothing at
        large lags. Pomegranate smoothing uses terminal end-state probabilities,
        which assume the sequence ends — a finite-sequence assumption that does
        not hold in live streaming.

        The posterior for timestep t is frozen once t+L observations are available.
        No retroactive updates occur.

        Parameters
        ----------
        X : np.ndarray
            Observation matrix, shape (T, D).
        lag : int
            Number of future bars to condition on. lag=0 equals filtering.
        """
        if isinstance(lag, bool):
            raise ValueError(f"lag must be an integer, got bool")
        if not isinstance(lag, (int, np.integer)):
            raise ValueError(f"lag must be an integer, got {type(lag).__name__}")
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict_proba_fixed_lag().")

        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values. Clean input data before calling fixed-lag smoothing.")
        X = self._preprocess_input(X)

        # Fast path: lag=0 is pure filtering — delegate to existing optimized path
        if lag == 0:
            return self._predict_proba_filtered(X)

        return self._fixed_lag_forward_backward(X, lag)

    def _fixed_lag_forward_backward(self, X: np.ndarray, lag: int) -> np.ndarray:
        """Core fixed-lag forward-backward with open-end semantics. Returns (T, K) posteriors.

        Callers must pass preprocessed X (via _preprocess_input).
        """
        from scipy.special import logsumexp

        log_pi, log_A, log_B = self._extract_hmm_parameters(X)
        T, K = log_B.shape

        # --- Forward pass (single pass over full sequence) ---
        log_alpha = np.empty((T, K), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0) + log_B[t]

        # --- Truncated backward pass (isolated per timestep, open-end) ---
        # log_beta always starts at 0 (uniform): no terminal conditioning.
        # This is the streaming invariant — we never assume the sequence ends.
        posteriors = np.empty((T, K), dtype=np.float64)
        for t in range(T):
            log_beta = np.zeros(K, dtype=np.float64)
            end = min(t + lag, T - 1)
            for s in range(end, t, -1):
                log_beta = logsumexp(log_A + log_B[s] + log_beta, axis=1)
            log_posterior = log_alpha[t] + log_beta
            log_posterior -= logsumexp(log_posterior)
            posteriors[t] = np.exp(log_posterior)

        return posteriors

    def predict_proba_fixed_lag_sweep(self, X: np.ndarray, lags: list[int]) -> dict[int, np.ndarray]:
        """Compute fixed-lag posteriors for multiple lags, sharing forward pass and emissions.

        Shares the forward pass and emission extraction across all lags, avoiding
        redundant O(T*K^2) work. However, the per-lag backward pass (O(T*L*K^2))
        dominates runtime, so wall-clock speedup over individual calls is marginal.
        The primary benefit is guaranteed numerical consistency across lags
        (identical forward variables).

        Parameters
        ----------
        X : np.ndarray
            Observation matrix, shape (T, D).
        lags : list[int]
            Lag values to compute. Each must be >= 0. Duplicates are removed
            and lags are processed in ascending order internally. The returned
            dict is keyed by lag value, so input ordering does not affect results.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from lag to (T, K) posterior matrix.
        """
        from scipy.special import logsumexp

        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict_proba_fixed_lag_sweep().")

        X = np.asarray(X)
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values. Clean input data before calling fixed-lag smoothing.")
        X = self._preprocess_input(X)

        result: dict[int, np.ndarray] = {}

        # Separate lag=0 fast path from custom lags
        custom_lags = []
        for lag in lags:
            if isinstance(lag, bool):
                raise ValueError(f"lag must be an integer, got bool")
            if not isinstance(lag, (int, np.integer)):
                raise ValueError(f"lag must be an integer, got {type(lag).__name__}")
            if lag < 0:
                raise ValueError(f"lag must be >= 0, got {lag}")
            if lag == 0:
                result[0] = self._predict_proba_filtered(X)
            else:
                custom_lags.append(lag)

        # Deduplicate to avoid redundant backward computation
        custom_lags = sorted(set(custom_lags))

        if not custom_lags:
            return result

        # Shared extraction and forward pass for all custom lags
        log_pi, log_A, log_B = self._extract_hmm_parameters(X)
        T, K = log_B.shape

        log_alpha = np.empty((T, K), dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1, :, np.newaxis] + log_A, axis=0) + log_B[t]

        # Per-lag backward passes (isolated per timestep, open-end)
        for lag in custom_lags:
            posteriors = np.empty((T, K), dtype=np.float64)
            for t in range(T):
                log_beta = np.zeros(K, dtype=np.float64)
                end = min(t + lag, T - 1)
                for s in range(end, t, -1):
                    log_beta = logsumexp(log_A + log_B[s] + log_beta, axis=1)
                log_posterior = log_alpha[t] + log_beta
                log_posterior -= logsumexp(log_posterior)
                posteriors[t] = np.exp(log_posterior)
            result[lag] = posteriors

        return result

    def predict_fixed_lag(self, X: np.ndarray, lag: int) -> np.ndarray:
        """MAP labels from fixed-lag posteriors. Shape (T,)."""
        return np.argmax(self.predict_proba_fixed_lag(X, lag), axis=1)

    def fixed_lag_diagnostics(self, X: np.ndarray, lag: int) -> dict:
        """Full diagnostic output for fixed-lag smoothing.

        Returns dict with keys:
            posteriors      : (T, K)  — posterior probability matrix
            map_labels      : (T,)    — argmax of posteriors
            max_posterior    : (T,)    — confidence of MAP label
            entropy         : (T,)    — -sum(p * log(p)) per timestep
            posterior_delta  : (T, K)  — change in posterior from t-1 to t
        """
        posteriors = self.predict_proba_fixed_lag(X, lag)
        map_labels = np.argmax(posteriors, axis=1)
        max_posterior = np.max(posteriors, axis=1)

        # Entropy: -sum(p * log(p)), with 0*log(0) = 0
        log_p = np.log(np.maximum(posteriors, 1e-300))
        entropy = -np.sum(posteriors * log_p, axis=1)

        # Posterior delta: change from t-1 to t (first row is zero)
        posterior_delta = np.diff(posteriors, axis=0, prepend=posteriors[:1])

        return {
            "posteriors": posteriors,
            "map_labels": map_labels,
            "max_posterior": max_posterior,
            "entropy": entropy,
            "posterior_delta": posterior_delta,
        }

    def _preprocess_input(self, X: np.ndarray) -> np.ndarray:
        """Shared input preprocessing."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(x_t | state j) for all t, j. Shape (T, N)."""
        T = X.shape[0]
        N = self.n_states
        X_tensor = torch.tensor(X, dtype=torch.float64)
        log_emit = np.empty((T, N), dtype=np.float64)
        for j, dist in enumerate(self._model.distributions):
            lp = dist.log_probability(X_tensor)
            if hasattr(lp, "detach"):
                lp = lp.detach().cpu().numpy()
            log_emit[:, j] = np.asarray(lp, dtype=np.float64)
        return log_emit

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _train_covariance_candidates(self) -> list[str]:
        """
        Return covariance types to sweep during model selection.

        Only covariance-capable distributions participate. Callers can override
        defaults by passing `covariance_type` (single) or `covariance_types`
        (list-like) in `dist_kwargs`.

        Note: StudentT is restricted to "diag"/"sphere" because its
        log_probability implementation uses element-wise `/ self.covs` rather
        than the matrix path that Normal uses, so "full" covariance produces a
        shape mismatch when covs is (d, d).
        """
        if self.distribution_type not in {DistType.NORMAL, DistType.STUDENTT, DistType.LOGNORMAL}:
            return []

        candidates = self.dist_kwargs.get("covariance_types")
        if candidates is None:
            current = self.dist_kwargs.get("covariance_type")
            if current is not None:
                candidates = [str(current)]
            elif self.distribution_type == DistType.STUDENTT:
                candidates = ["diag", "sphere"]
            else:
                candidates = ["diag", "full", "sphere"]

        valid = {"diag", "full", "sphere"}
        # StudentT.log_probability uses element-wise / covs; "full" (d,d) covs
        # causes a shape mismatch. Strip it here so train() never attempts it.
        if self.distribution_type == DistType.STUDENTT:
            valid = {"diag", "sphere"}

        ordered: list[str] = []
        for c in candidates:
            c_norm = str(c).lower()
            if c_norm in valid and c_norm not in ordered:
                ordered.append(c_norm)

        return ordered

    def _iter_train_dist_kwargs(self):
        """
        Yield dist_kwargs variants used during train-time model selection.
        """
        covariance_candidates = self._train_covariance_candidates()
        if not covariance_candidates:
            yield dict(self.dist_kwargs)
            return

        for cov_type in covariance_candidates:
            inst_kwargs = dict(self.dist_kwargs)
            inst_kwargs["covariance_type"] = cov_type
            yield inst_kwargs

    def _compute_kmeans_init(self, X: np.ndarray) -> dict:
        """
        Run pomegranate k-means and return per-state centroids and covariances.

        Uses pomegranate's own KMeans (PyTorch-backed) so it stays in the same
        compute graph and benefits from any GPU acceleration available.

        Returns a dict with:
          centroids  : (n_states, D) float64 — cluster centres
          covs_full  : list of (D, D) float64 — full covariance per state
          covs_diag  : list of (D,)   float64 — diagonal variance per state
        """
        from pomegranate.kmeans import KMeans

        D = X.shape[1]
        X64 = X.astype(np.float64)
        km = KMeans(k=self.n_states, init="random", random_state=self.random_state)
        km.fit(X64)
        labels = km.predict(X64).detach().cpu().numpy()
        centroids = km.centroids.detach().cpu().numpy().astype(np.float64)

        covs_full, covs_diag = [], []
        for k in range(self.n_states):
            pts = X64[labels == k]
            if len(pts) > 1:
                cf = np.cov(pts.T).astype(np.float64)
                if cf.ndim == 0:          # single-feature edge case
                    cf = cf.reshape(1, 1)
                cf += np.eye(D, dtype=np.float64) * 1e-6  # regularise
                cd = np.maximum(pts.var(axis=0).astype(np.float64), 1e-6)
            else:
                cf = np.eye(D, dtype=np.float64)
                cd = np.ones(D, dtype=np.float64)
            covs_full.append(cf)
            covs_diag.append(cd)

        return {"centroids": centroids, "covs_full": covs_full, "covs_diag": covs_diag}

    def _split_sequences(self, X: np.ndarray, lengths: Optional[Sequence[int]]):
        """Convert concatenated X + lengths into list of arrays."""
        if lengths is None:
            return [X]

        start, seqs = 0, []
        for l in lengths:
            seqs.append(X[start : start + l])
            start += l
        return seqs

    # ------------------------------------------------------------------
    # (De)serialisation
    # ------------------------------------------------------------------
    def __getstate__(self):
        """
        Custom serialization for joblib to handle pomegranate model with PyTorch tensors.
        Serialize the internal pomegranate model to bytes using joblib.
        """
        state = self.__dict__.copy()
        if self._model is not None:
            buffer = io.BytesIO()
            joblib.dump(self._model, buffer, compress=3)
            state["_model_bytes"] = buffer.getvalue()
            state["_model"] = None

        return state

    def __setstate__(self, state):
        """Custom deserialization to restore pomegranate model from bytes."""
        # Restore the model from joblib bytes
        if "_model_bytes" in state:
            buffer = io.BytesIO(state["_model_bytes"])
            state["_model"] = joblib.load(buffer)
            del state["_model_bytes"]
        # Reject models serialized with HSMM state — they cannot run correctly
        # after HSMM inference paths were removed.
        hsmm_keys = frozenset({"duration_model", "_hsmm_log_trans", "_hsmm_log_init"}) & state.keys()
        if hsmm_keys:
            raise RuntimeError(
                "Cannot load a model serialized with HSMM state: HSMM support has been removed. "
                f"Legacy state keys found: {sorted(hsmm_keys)}. Re-train and re-save the model."
            )
        self.__dict__.update(state)

    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "BasePomegranateHMM":
        """Load model from disk."""
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Abstract methods - must be implemented by subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_model(self) -> DenseHMM:
        """Build the pomegranate HMM model. Implemented by subclasses."""
        pass

    @abstractmethod
    def get_aic_bic(self, X: np.ndarray) -> tuple[float, float]:
        """Calculate AIC and BIC scores. Implemented by subclasses."""
        pass

    @abstractmethod
    def train(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, **kwargs) -> "BasePomegranateHMM":
        """
        Train model with automatic state selection.
        Implemented by subclasses with specific parameter ranges.
        """
        pass
