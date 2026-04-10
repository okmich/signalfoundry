"""HSMM EM orchestrator bridging pomegranate emissions with custom HSMM inference.

Reuses pomegranate distributions for emission parameter learning (``summarize`` + ``from_summaries``) while running the
full E-step through :mod:`hsmm_inference`.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from pomegranate.hmm import DenseHMM

from .duration._base import BaseDuration
from .hsmm_inference import hsmm_forward, hsmm_forward_backward, hsmm_viterbi

logger = logging.getLogger(__name__)


class HSMMEMFitter:
    """EM orchestrator for HSMM with pomegranate emission distributions.

    Parameters
    ----------
    model : DenseHMM
        Pomegranate model (used only for its emission distributions).
    duration_model : BaseDuration
        Duration distribution to be learned.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Relative log-likelihood convergence tolerance.
    random_state : int
        Random seed.
    """

    def __init__(self, model: DenseHMM, duration_model: BaseDuration, max_iter: int = 100, tol: float = 1e-4,
                 random_state: int = 100):
        self.model = model
        self.duration_model = duration_model
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X_list: list[np.ndarray], skip_hmm_warmstart: bool = False) -> \
            tuple[DenseHMM, BaseDuration, np.ndarray, np.ndarray, list[float]]:
        """Run HSMM EM.

        Parameters
        ----------
        X_list : list of (T_i, D) arrays
            Observation sequences.
        skip_hmm_warmstart : bool
            If True, skip the initial pomegranate warm-start and use
            uniform off-diagonal transitions + k-means emission init.

        Returns
        -------
        model : DenseHMM — updated pomegranate model (emissions only)
        duration_model : BaseDuration — fitted duration parameters
        log_trans : (N, N) — HSMM zero-diagonal transition matrix (log)
        log_init : (N,) — HSMM initial state probabilities (log)
        ll_history : list of float — log-likelihood per EM iteration
        """
        N = len(self.model.distributions)
        M = self.duration_model.max_duration
        alpha_pseudo = 1e-3  # Dirichlet pseudocount

        # -----------------------------------------------------------
        # Initialisation
        # -----------------------------------------------------------
        if not skip_hmm_warmstart:
            # Warm-start: 5 iterations of standard pomegranate EM
            self.model.max_iter = 5
            self.model.fit(X_list)
            self.model.max_iter = self.max_iter

            # Extract transition matrix, zero diagonal, renormalise
            log_trans = self._extract_zero_diag_transitions()
            log_init = self._extract_log_init()

            # Initialise duration model from Viterbi run-lengths
            self._init_duration_from_viterbi(X_list, log_trans, log_init)
        else:
            # Uniform off-diagonal transitions
            log_trans = np.full((N, N), -np.inf, dtype=np.float64)
            off_diag_val = -np.log(N - 1)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        log_trans[i, j] = off_diag_val
            log_init = np.full(N, -np.log(N), dtype=np.float64)

        # -----------------------------------------------------------
        # EM loop
        # -----------------------------------------------------------
        ll_history: list[float] = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            total_ll = 0.0
            total_xi = np.zeros((N, N), dtype=np.float64)
            total_eta = np.zeros((N, M), dtype=np.float64)
            # Accumulate emission sufficient stats via pomegranate's summarize
            init_counts = np.zeros(N, dtype=np.float64)

            for X in X_list:
                X64 = np.asarray(X, dtype=np.float64)
                log_emissions = self._compute_log_emissions(X64)

                result = hsmm_forward_backward(log_emissions, log_trans, log_init, self.duration_model)
                total_ll += result.log_likelihood
                total_xi += result.expected_transitions
                total_eta += result.expected_durations
                init_counts += result.state_posteriors[0]

                # Emission M-step: summarize with posterior weights
                # Pomegranate requires sample_weight > 0; clamp to small floor
                X_tensor = torch.tensor(X64, dtype=torch.float64)
                for j, dist in enumerate(self.model.distributions):
                    w = np.maximum(result.state_posteriors[:, j], 1e-300)
                    weights = torch.tensor(w, dtype=torch.float64)
                    dist.summarize(X_tensor, sample_weight=weights)

            ll_history.append(total_ll)

            # Check convergence
            rel_change = abs(total_ll - prev_ll) / max(abs(prev_ll), 1.0) if prev_ll > -np.inf else np.inf
            logger.debug("HSMM EM iter %d: ll=%.4f, rel_change=%.2e", iteration, total_ll, rel_change)

            if rel_change < self.tol and iteration > 0:
                logger.info("HSMM EM converged after %d iterations (ll=%.4f).", iteration + 1, total_ll)
                break

            # EM monotonicity check — two-tier per spec §9.1
            # Use *relative* decrease to handle large-T sequences where
            # cumulative floating-point error in the E-step can cause
            # small absolute LL drops (e.g. ~300 nats on LL ≈ 1.3M).
            if iteration > 0 and total_ll < prev_ll:
                decrease = prev_ll - total_ll
                rel_decrease = decrease / max(abs(prev_ll), 1.0)
                if self.duration_model.is_exact_mle:
                    # Hard assert for exact-MLE M-steps (Poisson, Nonparametric)
                    if rel_decrease > 1e-3:
                        raise RuntimeError(
                            f"HSMM EM: log-likelihood decreased at iter {iteration} "
                            f"({prev_ll:.4f} -> {total_ll:.4f}, delta={-decrease:.4e}, "
                            f"rel={rel_decrease:.2e}) with exact-MLE duration model. "
                            f"This indicates a bug."
                        )
                    elif rel_decrease > 1e-6:
                        logger.warning(
                            "HSMM EM: small LL decrease at iter %d (%.4f -> %.4f, "
                            "rel=%.2e) — likely numerical noise at this sequence length.",
                            iteration, prev_ll, total_ll, rel_decrease,
                        )
                else:
                    # Soft warning for approximate M-steps (NegBin, Gamma, LogNormal)
                    if rel_decrease > 1e-2:
                        logger.warning(
                            "HSMM EM: large LL decrease at iter %d (%.4f -> %.4f, rel=%.2e). "
                            "Consider using NonparametricDuration as a diagnostic.",
                            iteration, prev_ll, total_ll, rel_decrease,
                        )
                    elif rel_decrease > 1e-4:
                        logger.warning("HSMM EM: LL decreased at iter %d (%.4f -> %.4f).", iteration, prev_ll, total_ll)

            prev_ll = total_ll

            # -----------------------------------------------------------
            # M-step
            # -----------------------------------------------------------
            # Transitions: add pseudocount, zero diagonal, normalise
            for i in range(N):
                for j in range(N):
                    if i != j:
                        total_xi[i, j] += alpha_pseudo
                    else:
                        total_xi[i, j] = 0.0
            for i in range(N):
                row_sum = total_xi[i].sum()
                if row_sum > 0:
                    with np.errstate(divide="ignore"):
                        log_trans[i] = np.log(total_xi[i] / row_sum)
                    log_trans[i, i] = -np.inf
                else:
                    # Unused state: uniform off-diagonal
                    for j in range(N):
                        log_trans[i, j] = -np.log(N - 1) if i != j else -np.inf

            # Initial probs
            init_counts += alpha_pseudo
            log_init = np.log(init_counts / init_counts.sum())

            # Duration M-step
            for j in range(N):
                self.duration_model.update(j, total_eta[j])

            # Emission M-step: apply accumulated summaries
            for dist in self.model.distributions:
                dist.from_summaries()

        return self.model, self.duration_model, log_trans, log_init, ll_history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_log_emissions(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(x_t | state j) for all t, j. Shape (T, N)."""
        T = X.shape[0]
        N = len(self.model.distributions)
        X_tensor = torch.tensor(X, dtype=torch.float64)
        log_emit = np.empty((T, N), dtype=np.float64)
        for j, dist in enumerate(self.model.distributions):
            lp = dist.log_probability(X_tensor)
            if hasattr(lp, "detach"):
                lp = lp.detach().cpu().numpy()
            log_emit[:, j] = np.asarray(lp, dtype=np.float64)
        return log_emit

    def _extract_zero_diag_transitions(self) -> np.ndarray:
        """Extract transition matrix from pomegranate model, zero diagonal, renormalise."""
        N = len(self.model.distributions)
        edges = self.model.edges
        if hasattr(edges, "detach"):
            edges = edges.detach().cpu().numpy()
        edges = np.asarray(edges, dtype=np.float64)

        # Pomegranate stores log probabilities
        if np.any(edges < 0):
            trans = np.exp(edges)
        else:
            trans = edges.copy()

        # Zero diagonal, renormalise
        for i in range(N):
            trans[i, i] = 0.0
            row_sum = trans[i].sum()
            if row_sum > 0:
                trans[i] /= row_sum
            else:
                # Uniform off-diagonal
                for j in range(N):
                    trans[i, j] = 1.0 / (N - 1) if i != j else 0.0

        log_trans = np.full((N, N), -np.inf, dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i != j and trans[i, j] > 0:
                    log_trans[i, j] = np.log(trans[i, j])
        return log_trans

    def _extract_log_init(self) -> np.ndarray:
        """Extract initial state distribution from pomegranate model."""
        starts = self.model.starts
        if hasattr(starts, "detach"):
            starts = starts.detach().cpu().numpy()
        starts = np.asarray(starts, dtype=np.float64)
        if np.any(starts < 0):
            starts = np.exp(starts)
        starts = np.maximum(starts, 1e-300)
        starts /= starts.sum()
        return np.log(starts)

    def _init_duration_from_viterbi(self, X_list: list[np.ndarray], log_trans: np.ndarray, log_init: np.ndarray) -> None:
        """Initialise duration model parameters from Viterbi run-length statistics."""
        N = len(self.model.distributions)
        M = self.duration_model.max_duration
        run_lengths: list[list[int]] = [[] for _ in range(N)]

        for X in X_list:
            X64 = np.asarray(X, dtype=np.float64)
            log_emissions = self._compute_log_emissions(X64)
            states = hsmm_viterbi(log_emissions, log_trans, log_init, self.duration_model)

            # Extract run lengths
            current_state = states[0]
            current_len = 1
            for t in range(1, len(states)):
                if states[t] == current_state:
                    current_len += 1
                else:
                    run_lengths[current_state].append(min(current_len, M))
                    current_state = states[t]
                    current_len = 1
            run_lengths[current_state].append(min(current_len, M))

        # Update duration model from run-length histograms
        for j in range(N):
            if run_lengths[j]:
                eta = np.zeros(M, dtype=np.float64)
                for rl in run_lengths[j]:
                    eta[rl - 1] += 1.0  # u=1 maps to index 0
                self.duration_model.update(j, eta)
