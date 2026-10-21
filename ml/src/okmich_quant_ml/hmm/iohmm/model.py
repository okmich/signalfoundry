"""
PomegranateIOHMM: an Input-Output HMM (Bengio & Frasconi, 1995) that duck-types ``okmich_quant_ml.hmm.PomegranateHMM``.

The model in one page
---------------------
A plain HMM has hidden states ``z_t``, outputs ``x_t`` (the features), a start distribution ``pi``, ONE transition
matrix ``A`` used at every bar, and per-state emission densities ``p(x_t | z_t)``. An IOHMM adds an observed input
sequence ``u_t`` that the model conditions on but never explains. Here the input moves only the transitions::

    P(z_t = j | z_{t-1} = i, u_t) = A_t[i, j] = softmax_j( b[i, j] + W[i, j] . u_t ),   b[i, i] = W[i, i] = 0

and the emissions are shared across inputs, so a state means the same thing whatever the input is. ``b`` is the
baseline log-odds of each move against staying; ``W`` is how much one standard deviation of each input adds to those
log-odds. ``W = 0`` gives back the ordinary HMM, so "did the input help?" is a nested likelihood-ratio test
(:func:`input_likelihood_ratio_test`).

Everything else is the HMM machinery with ``A`` replaced by ``A_t``: the forward filter, forward-backward smoothing,
Viterbi, fixed-lag smoothing and the path entropy cost the same ``O(T K^2)``. Fitting is EM:

- E-step: forward-backward with ``A_t`` gives the state posteriors ``gamma_t(k)`` and the expected transitions
  ``xi_t(i, j)``.
- M-step: emissions are refitted by pomegranate's own weighted update (``summarize`` / ``from_summaries``, as
  ``DenseHMM`` does), ``pi`` by the posterior at bar 0, and the transitions by one convex weighted multinomial
  logistic regression per origin state (:class:`InputDrivenTransitions`).

Timing (:class:`InputTiming`): the effective input on row ``t`` drives the transition from ``t - 1`` into ``t``
and must be known by the close of bar ``t - 1``. Clock inputs are KNOWN_AHEAD; inputs computed from bar ``t``'s
prices are AT_BAR_CLOSE and the model shifts them by one bar itself.

Duck typing
-----------
Every public method and property of ``PomegranateHMM`` exists here with the same positional parameters; inputs are
the extra keyword-only argument ``U`` (``u`` for a single input vector). A model fitted without ``U`` is an ordinary
HMM. Differences from ``PomegranateHMM``:

- No end probabilities. Pomegranate learns end-state probabilities (a finite-sequence assumption); this model is
  open-ended, like the base class's fixed-lag smoother. Likelihoods and smoothed posteriors therefore differ from
  ``PomegranateHMM`` in the last bars by the end term; filtered posteriors and Viterbi paths do not.
- ``state_posteriors`` returns a ``(T, K)`` array, not pomegranate's tensor.
- ``forecast`` drops the deprecated hard-label entry ``current_regime``. A model with inputs needs ``u_future``.
- ``transition_prob()`` without ``u`` returns the matrix at the training-mean input (a summary, not a time series).
"""

from __future__ import annotations

import copy
import os
import warnings
from typing import Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import torch
from pomegranate._utils import partition_sequences

from ..base_pomegranate import BasePomegranateHMM
from ..pomegranate import PomegranateHMM
from ..util import DistType, InferenceMode, InputTiming, SelectionCriterion, SelectionScores, as_selection_criterion
from .emissions import log_emissions, update_emissions
from .kernels import (backward_log, concurrent_calls_supported, expected_statistics, forward_log, logsumexp,
                      max_product_forward, sequences_log_likelihood, viterbi_backpointers, windowed_fixed_lag)
from .transitions import InputDrivenTransitions

_TINY = np.finfo(np.float64).tiny


class _NonFiniteFit(RuntimeError):
    """EM produced a non-finite objective: the emission parameters degenerated."""


class PomegranateIOHMM:
    """
    Input-Output HMM with input-driven transitions and pomegranate emissions. See the module docstring for the model.

    Parameters
    ----------
    distribution_type, n_states, random_state, max_iter, n_restarts, inference_mode, dist_kwargs
        As in ``PomegranateHMM``. Initialisation is the same code path: the same k-means start and the same seeds, so
        a model fitted without inputs starts where ``PomegranateHMM`` starts.
    input_timing : InputTiming or sequence of InputTiming, default KNOWN_AHEAD
        When each input column becomes known; one value for all columns or one per column.
    l2_penalty : float, default 1.0
        Gaussian-prior penalty ``0.5 * l2_penalty * ||W||^2`` on the input weights (standardised units). It keeps
        weights finite when a transition never happens in part of the input range; at typical sample sizes it is
        negligible otherwise. It does not touch ``b``, so a model without inputs is exact Baum-Welch.
    tol : float, default 0.1
        EM stops when the penalised log-likelihood improves by less than ``tol`` (pomegranate's rule and default).
    n_jobs : int or None, default 1
        Restarts run at once (threads; ``-1`` = one per CPU). Each restart is seeded as in a sequential fit and the
        winner is chosen in restart order, so the fitted model is the same for every ``n_jobs``. The compiled kernels
        release the GIL and the CPU threads are shared out between the running restarts.
    """

    _FIT_MAX_RETRIES: int = 3
    _ENTROPY_CHUNK_BARS: int = 65_536

    def __init__(self, distribution_type: DistType, n_states: int = 2, *, random_state: int = 100, max_iter: int = 100,
                 n_restarts: int = 1, inference_mode: Optional[InferenceMode] = None,
                 input_timing: InputTiming | Sequence[InputTiming] = InputTiming.KNOWN_AHEAD, l2_penalty: float = 1.0,
                 tol: float = 0.1, n_jobs: Optional[int] = 1, **dist_kwargs):
        self.distribution_type = distribution_type
        self.n_states = n_states
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_restarts = max(1, int(n_restarts))
        self.inference_mode = inference_mode if inference_mode is not None else InferenceMode.FILTERING
        self.input_timing = input_timing
        self.l2_penalty = float(l2_penalty)
        self.tol = float(tol)
        self.n_jobs = n_jobs
        self.dist_kwargs = dist_kwargs
        self._template()  # validates dist_kwargs (e.g. rejects the removed HSMM kwargs) the way PomegranateHMM does
        self._clear_fit()

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, *, U: Optional[np.ndarray] = None,
            init_model: "PomegranateIOHMM | PomegranateHMM | None" = None) -> "PomegranateIOHMM":
        """
        Fit by EM. ``U`` (T, d) holds the inputs on the same rows as ``X``; without it the model is an ordinary HMM.

        ``init_model`` starts EM from a fitted model with the same states and emission family instead of the k-means
        start (one deterministic start, so ``n_restarts`` is ignored). Its input weights carry over when it has the
        same inputs; a model without inputs starts the weights at zero. Starting a model with inputs from the fitted
        model without them guarantees the nested pair satisfies ``loglik(full) >= loglik(null)``.
        """
        X = self._preprocess_input(X)
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values. Clean input data before fitting.")
        lengths = self._check_lengths(X.shape[0], lengths)
        self._clear_fit()  # a refit that fails must not leave the previous fit live
        U_std = self._setup_inputs(U, lengths)

        n_starts = 1 if init_model is not None else self.n_restarts
        seeds = [self.random_state + 1000 * restart for restart in range(n_starts)]
        n_jobs = min(self._resolved_n_jobs(), n_starts)
        if n_jobs > 1 and not concurrent_calls_supported():
            warnings.warn("numba's 'workqueue' threading layer cannot run the kernels from several threads at once; "
                          "running the restarts one after another (same result). Install tbb to run them in parallel.",
                          RuntimeWarning, stacklevel=2)
            n_jobs = 1
        if n_jobs == 1:
            outcomes = [self._fit_single_start(X, lengths, U_std, seed, init_model) for seed in seeds]
        else:
            threads = max(1, numba.get_num_threads() // n_jobs)
            outcomes = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
                joblib.delayed(self._fit_start_on_threads)(X, lengths, U_std, seed, threads) for seed in seeds)
        best, last_error, n_nonfinite = None, None, 0
        for result, err in outcomes:  # restart order, so the winner is the one a sequential fit picks
            if result is None:
                if isinstance(err, _NonFiniteFit):
                    n_nonfinite += 1
                else:
                    last_error = err
                continue
            if best is None or result[3] > best[3]:
                best = result

        if best is None and n_nonfinite:
            raise RuntimeError(
                f"EM produced a non-finite log-likelihood in {n_nonfinite} of {n_starts} start(s): the emission "
                f"parameters degenerated (NaN or boundary values). Last covariance error, if any: "
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error
        if best is None:
            raise RuntimeError(
                f"EM fitting failed across {n_starts} start(s), each retried {self._FIT_MAX_RETRIES}x, due to "
                f"covariance/Cholesky errors. Last error: {type(last_error).__name__}: {last_error}"
            ) from last_error
        self._distributions, self._log_pi, self._transitions, self.train_objective_, self.objective_history_ = best
        return self

    def _resolved_n_jobs(self) -> int:
        if self.n_jobs is None:
            return 1
        n_jobs = int(self.n_jobs)
        if n_jobs == 0:
            raise ValueError("n_jobs must be a positive integer, -1 or None, got 0")
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs) if n_jobs < 0 else n_jobs

    def _fit_start_on_threads(self, X, lengths, U_std, start_seed: int, threads: int):
        """:meth:`_fit_single_start` on a worker thread, with this thread's share of the compiled kernels' threads."""
        previous = numba.get_num_threads()
        numba.set_num_threads(threads)  # thread-local: the caller's own setting is untouched
        try:
            return self._fit_single_start(X, lengths, U_std, start_seed, None)
        finally:
            numba.set_num_threads(previous)  # joblib reuses its worker threads

    def _fit_single_start(self, X, lengths, U_std, start_seed: int, init_model):
        """One start with covariance-error retries. Returns ``(state, None)`` or ``(None, error)``."""
        last_error = None
        for attempt in range(self._FIT_MAX_RETRIES):
            try:
                if init_model is None:
                    state = self._initial_state(X, lengths, start_seed + attempt)
                else:
                    state = self._state_from_model(init_model)
                return self._em(X, lengths, U_std, *state), None
            except _NonFiniteFit as e:
                return None, e
            except (np.linalg.LinAlgError, AttributeError, RuntimeError) as e:
                if not BasePomegranateHMM._is_covariance_error(e):
                    raise
                last_error = e
                if init_model is not None:
                    break  # a warm start is deterministic: retrying repeats the failure
        return None, last_error

    def _initial_state(self, X: np.ndarray, lengths: list[int], seed: int):
        """PomegranateHMM's own start: its k-means statistics, its distribution builders, pomegranate's initialiser."""
        template = self._template(seed)
        template._kmeans_stats = template._compute_init_stats(X)
        dense = template._build_model()
        del template._kmeans_stats
        X_parts, _, _ = partition_sequences(self._split(X, lengths), n_dists=self.n_states)
        dense._initialize(X_parts)
        log_pi = BasePomegranateHMM._to_log_numpy(dense.starts)
        transitions = InputDrivenTransitions.from_log_matrix(BasePomegranateHMM._to_log_numpy(dense.edges),
                                                             n_inputs=self._n_inputs, l2_penalty=self.l2_penalty)
        return list(dense.distributions), log_pi - logsumexp(log_pi), transitions

    def _state_from_model(self, model: "PomegranateIOHMM | PomegranateHMM"):
        if isinstance(model, PomegranateHMM):
            model = PomegranateIOHMM.from_hmm(model)
        model._check_fitted()
        if model.n_states != self.n_states or model.distribution_type != self.distribution_type:
            raise ValueError("init_model must have the same n_states and distribution_type as this model.")
        if model._n_inputs not in (0, self._n_inputs):
            raise ValueError(f"init_model has {model._n_inputs} inputs; this fit has {self._n_inputs}.")
        transitions = InputDrivenTransitions(self.n_states, n_inputs=self._n_inputs, l2_penalty=self.l2_penalty)
        transitions.b = model._transitions.b.copy()
        if model._n_inputs and self._n_inputs:
            # Re-express the weights on this fit's standardisation:
            # W' = W s_new / s_old, b' = b + W (m_new - m_old) / s_old
            W_raw = model._transitions.W / model._u_std
            transitions.W = W_raw * self._u_std
            transitions.b = transitions.b + W_raw @ (self._u_mean - model._u_mean)
            np.fill_diagonal(transitions.b, 0.0)
        return copy.deepcopy(list(model._distributions)), model._log_pi.copy(), transitions

    def _em(self, X: np.ndarray, lengths: list[int], U_std: Optional[np.ndarray], dists: list, log_pi: np.ndarray,
            transitions: InputDrivenTransitions):
        """EM with pomegranate's stopping rule: stop, without the M-step, once the objective improves by < tol."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        X_t = torch.as_tensor(X, dtype=torch.float64)
        bounds = self._bound_arrays(lengths)
        U_trans = None if U_std is None else U_std[self._transition_rows(lengths)]
        history, last = [], None
        for iteration in range(self.max_iter):
            objective, gamma, xi, starts = self._e_step(X, X_t, bounds, U_std, dists, log_pi, transitions)
            history.append(objective)
            if iteration > 0 and objective - last < self.tol:
                break
            last = objective
            update_emissions(dists, X, X_t, gamma)
            log_pi = np.log(np.maximum(starts / starts.sum(), _TINY))
            transitions.fit(xi, U_trans)
        else:
            objective = self._e_step(X, X_t, bounds, U_std, dists, log_pi, transitions, stats=False)[0]
            history.append(objective)
        return dists, log_pi, transitions, objective, np.asarray(history)

    def _e_step(self, X: np.ndarray, X_t: torch.Tensor, bounds: tuple[np.ndarray, np.ndarray],
                U_std: Optional[np.ndarray], dists: list, log_pi: np.ndarray, transitions: InputDrivenTransitions,
                stats: bool = True):
        """``(objective, gamma, xi, start_counts)``; with ``stats=False`` only the objective (the rest is None)."""
        log_B = log_emissions(dists, X, X_t)
        log_A = transitions.log_matrices(U_std)
        if stats:
            loglik, gamma, starts, xi = expected_statistics(log_pi, log_A, log_B, *bounds)
        else:
            loglik, gamma, starts, xi = sequences_log_likelihood(log_pi, log_A, log_B, *bounds), None, None, None
        objective = loglik - transitions.penalty()
        if not np.isfinite(objective):
            raise _NonFiniteFit(f"non-finite EM objective {objective}")
        return objective, gamma, xi, starts

    def fit_predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, *,
                    U: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit model and return predicted states."""
        self.fit(X, lengths, U=U)
        return self.predict(X, U=U)

    def train(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
              n_states_range: Sequence[int] = (2, 3, 4, 5),
              criterion: SelectionCriterion | str = SelectionCriterion.AIC, *,
              U: Optional[np.ndarray] = None) -> "PomegranateIOHMM":
        """Fit every (n_states, covariance type) candidate and return the one with the lowest AIC, BIC or ICL."""
        criterion = as_selection_criterion(criterion)
        with_entropy = criterion is SelectionCriterion.ICL
        best_score, best_inst = np.inf, None
        for k in n_states_range:
            for inst_kwargs in self._template()._iter_train_dist_kwargs():
                inst = self.__class__(self.distribution_type, n_states=k, random_state=self.random_state,
                                      max_iter=self.max_iter, n_restarts=self.n_restarts,
                                      inference_mode=self.inference_mode, input_timing=self.input_timing,
                                      l2_penalty=self.l2_penalty, tol=self.tol, n_jobs=self.n_jobs, **inst_kwargs)
                inst.fit(X, lengths, U=U)
                score = inst.selection_scores(X, lengths, U=U, with_entropy=with_entropy).score(criterion)
                if score < best_score:
                    best_score, best_inst = score, inst
        return best_inst

    @classmethod
    def from_hmm(cls, hmm: PomegranateHMM) -> "PomegranateIOHMM":
        """A model without inputs carrying a fitted ``PomegranateHMM``'s start, transition and emission parameters.

        Transition rows are renormalised (pomegranate's rows sum to ``1 - end_i``); end probabilities are dropped.
        """
        if hmm._model is None:
            raise RuntimeError("hmm has not been fitted yet.")
        model = cls(hmm.distribution_type, n_states=hmm.n_states, random_state=hmm.random_state,
                    max_iter=hmm.max_iter, n_restarts=hmm.n_restarts, inference_mode=hmm.inference_mode,
                    **hmm.dist_kwargs)
        model._setup_inputs(None, [])
        model._distributions = copy.deepcopy(list(hmm._model.distributions))
        log_pi = BasePomegranateHMM._to_log_numpy(hmm._model.starts)
        model._log_pi = log_pi - logsumexp(log_pi)
        model._transitions = InputDrivenTransitions.from_log_matrix(np.log(np.maximum(hmm.transition_prob(), _TINY)),
                                                                    n_inputs=0, l2_penalty=model.l2_penalty)
        return model

    # ------------------------------------------------------------------
    # State inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Hidden states by ``inference_mode`` (FILTERING / SMOOTHING argmax, VITERBI, CAUSAL_VITERBI)."""
        self._check_fitted()
        if self.inference_mode == InferenceMode.CAUSAL_VITERBI:
            return self.predict_causal_viterbi(X, U=U)
        if self.inference_mode == InferenceMode.VITERBI:
            return self.predict_viterbi(X, U=U)
        if self.inference_mode in (InferenceMode.FILTERING, InferenceMode.SMOOTHING):
            return np.argmax(self.predict_proba(X, U=U), axis=1)
        raise ValueError(f"Unknown inference_mode: {self.inference_mode}")

    def predict_proba(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """State probabilities: filtered (causal) or smoothed (non-causal, open-ended) by ``inference_mode``."""
        self._check_fitted()
        if self.inference_mode in (InferenceMode.VITERBI, InferenceMode.CAUSAL_VITERBI):
            raise ValueError(
                f"predict_proba() is not applicable with inference_mode={self.inference_mode.name}. "
                "Use predict() instead, or set inference_mode to FILTERING or SMOOTHING."
            )
        log_pi, log_A, log_B = self._inference_inputs(X, U)
        if self.inference_mode == InferenceMode.FILTERING:
            return self._normalise_rows(forward_log(log_pi, log_A, log_B))
        if self.inference_mode == InferenceMode.SMOOTHING:
            return self._smoothed(log_pi, log_A, log_B)
        raise ValueError(f"Unknown inference_mode: {self.inference_mode}")

    def state_posteriors(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Smoothed (forward-backward, open-ended) posteriors, shape ``(T, K)``, whatever the inference mode."""
        return self._smoothed(*self._inference_inputs(X, U))

    def log_likelihood(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> float:
        """Scalar log-likelihood of the sequence X (given U) under the fitted model."""
        log_pi, log_A, log_B = self._inference_inputs(X, U)
        return float(logsumexp(forward_log(log_pi, log_A, log_B)[-1]))

    def compute_per_bar_predictive_loglik(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """``log P(o_t | o_{<t}, u_{<=t})`` per bar, shape ``(T,)``; the sum is the sequence log-likelihood."""
        log_alpha = forward_log(*self._inference_inputs(X, U))
        log_cumulative = logsumexp(log_alpha, axis=1)
        loglik = np.empty(len(log_cumulative), dtype=np.float64)
        loglik[0] = log_cumulative[0]
        loglik[1:] = np.diff(log_cumulative)
        return loglik

    def predict_viterbi(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """The most likely state SEQUENCE (true Viterbi with traceback). Non-causal: offline labelling only."""
        log_delta, psi = viterbi_backpointers(*self._inference_inputs(X, U))
        self._check_feasible(log_delta, "predict_viterbi")
        path = np.empty(len(log_delta), dtype=np.int64)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(len(path) - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def causal_viterbi_scores(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Max-product forward scores ``log_delta``, shape ``(T, K)``. Path log-probabilities, not a posterior."""
        return max_product_forward(*self._inference_inputs(X, U))

    def predict_causal_viterbi(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """Causal MAP-path labels: the terminal state of the best path over bars ``0..t``, never revised."""
        log_delta = self.causal_viterbi_scores(X, U=U)
        self._check_feasible(log_delta, "predict_causal_viterbi")
        return np.argmax(log_delta, axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    # Fixed-lag smoothing
    # ------------------------------------------------------------------
    def predict_proba_fixed_lag(self, X: np.ndarray, lag: int, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """``P(z_t | o_0..o_{min(t+lag, T-1)})`` with open-end semantics, shape ``(T, K)``. ``lag=0`` is filtering."""
        self._check_lag(lag)
        return self.predict_proba_fixed_lag_sweep(X, [lag], U=U)[lag]

    def predict_proba_fixed_lag_sweep(self, X: np.ndarray, lags: list[int], window_size: Optional[int] = None, *,
                                      U: Optional[np.ndarray] = None) -> dict[int, np.ndarray]:
        """Fixed-lag posteriors for several lags (full history, or a ``window_size`` buffer per as-of bar)."""
        if not hasattr(lags, "__iter__"):
            raise ValueError(f"lags must be an iterable of integers, got {type(lags).__name__}")
        unique_lags = sorted({self._check_lag(lag) for lag in lags})
        if window_size is not None:
            if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
                raise ValueError(f"window_size must be an integer, got {type(window_size).__name__}")
            if window_size < 1:
                raise ValueError(f"window_size must be >= 1, got {window_size}")
            violating = [lag for lag in unique_lags if lag >= window_size]
            if violating:
                raise ValueError(f"In windowed mode every lag must be less than window_size. Got window_size="
                                 f"{window_size} but lag(s) {violating} >= window_size.")
        log_pi, log_A, log_B = self._inference_inputs(X, U)
        if not unique_lags:
            return {}
        if window_size is None:
            return self._fixed_lag_full(log_pi, log_A, log_B, unique_lags)
        return self._fixed_lag_windowed(log_pi, log_A, log_B, unique_lags, int(window_size))

    def predict_fixed_lag(self, X: np.ndarray, lag: int, *, U: Optional[np.ndarray] = None) -> np.ndarray:
        """MAP labels from fixed-lag posteriors. Shape (T,)."""
        return np.argmax(self.predict_proba_fixed_lag(X, lag, U=U), axis=1)

    def fixed_lag_diagnostics(self, X: np.ndarray, lag: int, *, U: Optional[np.ndarray] = None) -> dict:
        """Posteriors, MAP labels, max posterior, entropy and posterior change for fixed-lag smoothing."""
        posteriors = self.predict_proba_fixed_lag(X, lag, U=U)
        log_p = np.log(np.maximum(posteriors, 1e-300))
        return {
            "posteriors": posteriors,
            "map_labels": np.argmax(posteriors, axis=1),
            "max_posterior": np.max(posteriors, axis=1),
            "entropy": -np.sum(posteriors * log_p, axis=1),
            "posterior_delta": np.diff(posteriors, axis=0, prepend=posteriors[:1]),
        }

    def _fixed_lag_full(self, log_pi, log_A, log_B, lags: list[int]) -> dict[int, np.ndarray]:
        """Shared forward pass; backward messages grown one lag at a time (diagonal recurrence, open end)."""
        T, K = log_B.shape
        log_alpha = forward_log(log_pi, log_A, log_B)
        result = {0: self._normalise_rows(log_alpha)} if 0 in lags else {}
        custom = [lag for lag in lags if lag > 0]
        if not custom:
            return result
        lA_next = log_A[1:] if log_A.shape[0] > 1 else log_A
        lB_next = log_B[1:]
        prev = np.zeros((T, K))
        for L in range(1, min(custom[-1], T - 1) + 1):
            curr = np.zeros((T, K))
            curr[:T - 1] = logsumexp(lA_next + (lB_next + prev[1:])[:, None, :], axis=2)
            prev = curr
            if L in custom:
                result[L] = self._normalise_rows(log_alpha + curr)
        saturated = self._normalise_rows(log_alpha + prev)
        for lag in custom:
            result.setdefault(lag, saturated)  # lags beyond T-1 see every bar
        return result

    @staticmethod
    def _fixed_lag_windowed(log_pi, log_A, log_B, lags: list[int], window_size: int) -> dict[int, np.ndarray]:
        """Each as-of bar sees only its last ``window_size`` bars, restarting from ``pi`` (a live buffer emulation)."""
        return {lag: windowed_fixed_lag(log_pi, log_A, log_B, lag, window_size) for lag in lags}

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def transition_prob(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transition matrix (K, K) for one effective input vector ``u`` (raw units, length d).

        Without ``u`` it is the matrix at the training-mean input: a summary, not what the model applies at any
        particular bar. Use :meth:`transition_matrices` for the per-bar matrices of a sequence.
        """
        self._check_fitted()
        if self._n_inputs == 0:
            if u is not None:
                raise ValueError("This model was fitted without inputs; u must be None.")
            return np.exp(self._transitions.log_matrices(None)[0])
        u_std = np.zeros((1, self._n_inputs)) if u is None else self._standardise(self._input_vector(u)[None])
        return np.exp(self._transitions.log_matrices(u_std)[0])

    def transition_matrices(self, U: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """Per-bar matrices ``A_t`` (T, K, K) for raw inputs ``U``; the first bar of each sequence is NaN (unused)."""
        self._check_fitted()
        if self._n_inputs == 0:
            raise ValueError("This model was fitted without inputs; use transition_prob().")
        U = self._input_matrix(U, None)
        lengths = self._check_lengths(U.shape[0], lengths)
        A = np.exp(self._transitions.log_matrices(self._standardise(self._align_inputs(U, lengths))))
        A[np.concatenate([[0], np.cumsum(lengths)[:-1]])] = np.nan
        return A

    def forecast(self, posterior: Optional[np.ndarray] = None, n_steps: int = 1, *,
                 u_future: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Regime probabilities ``n_steps`` ahead: ``posterior @ A_{t+1} @ ... @ A_{t+n}``.

        ``u_future`` (n_steps, d) holds the effective inputs of those transitions in raw units: row ``s`` drives the
        transition into ``t + 1 + s``. Required when the model has inputs (clock inputs are known ahead; an
        AT_BAR_CLOSE input is known for the first step only). The deprecated ``current_regime`` entry is not supported.
        """
        self._check_fitted()
        if posterior is None:
            raise TypeError("forecast() requires `posterior`, a (n_states,) array.")
        if isinstance(posterior, (int, np.integer)):
            raise TypeError("Pass `posterior` as a (n_states,) array; hard-label entry is not supported.")
        belief = np.asarray(posterior, dtype=np.float64).reshape(-1)
        if belief.shape != (self.n_states,):
            raise ValueError(f"posterior must have shape ({self.n_states},), got {belief.shape}")
        if np.any(belief < 0) or not np.isclose(belief.sum(), 1.0, atol=1e-6):
            raise ValueError("posterior must be non-negative and sum to 1")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self._n_inputs == 0:
            if u_future is not None:
                raise ValueError("This model was fitted without inputs; u_future must be None.")
            A = self.transition_prob()
            for _ in range(n_steps):
                belief = belief @ A
            return belief
        if u_future is None:
            raise ValueError("This model has inputs: pass u_future, the (n_steps, d) inputs of the forecast steps.")
        u_future = self._input_matrix(u_future, n_steps)
        for row in u_future:
            belief = belief @ self.transition_prob(row)
        return belief

    def get_stationary_distribution(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Stationary distribution of ``transition_prob(u)`` (the training-mean input by default)."""
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_prob(u).T)
        stationary = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        return stationary / stationary.sum()

    def expected_duration(self, regime: int, u: Optional[np.ndarray] = None) -> float:
        """``1 / (1 - A_ii)`` at input ``u`` (the training-mean input by default): the dwell if ``u`` stayed fixed."""
        self._check_fitted()
        if not (0 <= regime < self.n_states):
            raise ValueError(f"regime must be between 0 and {self.n_states - 1}")
        stay = self.transition_prob(u)[regime, regime]
        return np.inf if stay >= 1.0 else 1.0 / (1.0 - stay)

    # ------------------------------------------------------------------
    # Inputs: introspection
    # ------------------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        """Number of input columns the model was fitted with (0 = ordinary HMM)."""
        return self._n_inputs

    @property
    def input_timing_(self) -> list[InputTiming]:
        """The resolved per-column input timing of the fit."""
        return list(self._input_timing)

    def input_coefficients(self, raw: bool = False) -> np.ndarray:
        """``W`` (K, K, d): log-odds change of each move vs staying, per input SD (or per raw unit if ``raw``)."""
        self._check_fitted()
        W = self._transitions.W.copy()
        return W / self._u_std if raw else W

    def baseline_logits(self, raw: bool = False) -> np.ndarray:
        """``b`` (K, K): log-odds of each move vs staying at the training-mean input (or at raw input 0 if ``raw``)."""
        self._check_fitted()
        b = self._transitions.b.copy()
        if raw and self._n_inputs:
            b = b - self.input_coefficients(raw=True) @ self._u_mean
            np.fill_diagonal(b, 0.0)
        return b

    def input_effects(self, input_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """One row per (from, to, input): log-odds of the move vs staying, per input SD and per raw unit."""
        self._check_fitted()
        names = list(input_names) if input_names is not None else [f"u{m}" for m in range(self._n_inputs)]
        if len(names) != self._n_inputs:
            raise ValueError(f"input_names must have {self._n_inputs} entries, got {len(names)}")
        W, W_raw = self.input_coefficients(), self.input_coefficients(raw=True)
        rows = [{"from_state": i, "to_state": j, "input": names[m], "log_odds_per_sd": W[i, j, m],
                 "log_odds_per_unit": W_raw[i, j, m]}
                for i in range(self.n_states) for j in range(self.n_states) if i != j for m in range(self._n_inputs)]
        return pd.DataFrame(rows, columns=["from_state", "to_state", "input", "log_odds_per_sd", "log_odds_per_unit"])

    def input_response(self, input_idx: int = 0, values: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        How one input moves the switching hazard: one row per (input value, state) with the stay probability and the
        expected dwell ``1 / (1 - stay)`` in bars, the other inputs held at their training means.

        ``values`` are raw input values (default: 101 points across the training range). The dwell is what the state
        would last if the input were held at that value: the interaction readout, not a forecast.
        """
        self._check_fitted()
        if not 0 <= input_idx < self._n_inputs:
            raise ValueError(f"input_idx must be in [0, {self._n_inputs - 1}]")
        if values is None:
            values = np.linspace(self._u_min[input_idx], self._u_max[input_idx], 101)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        U = np.repeat(self._u_mean[None], len(values), axis=0)
        U[:, input_idx] = values
        stay = np.exp(np.diagonal(self._transitions.log_matrices(self._standardise(U)), axis1=1, axis2=2))
        return pd.DataFrame({"value": np.repeat(values, self.n_states),
                             "state": np.tile(np.arange(self.n_states), len(values)), "stay_prob": stay.ravel(),
                             "expected_dwell": 1.0 / np.maximum(1.0 - stay.ravel(), _TINY)})

    def plot_input_response(self, input_idx: int = 0, values: Optional[np.ndarray] = None, ax=None, figsize=(8, 4)):
        """Expected dwell of each state as one input sweeps its training range, the others held at their means."""
        response = self.input_response(input_idx, values)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        for k, part in response.groupby("state"):
            ax.plot(part["value"], part["expected_dwell"], label=f"State {k}")
        ax.set_yscale("log")
        ax.set_xlabel(f"input {input_idx} (raw units)")
        ax.set_ylabel("expected dwell (bars)")
        ax.set_title("Input response of the switching hazard")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def selection_scores(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, *,
                         U: Optional[np.ndarray] = None, with_entropy: bool = True) -> SelectionScores:
        """Log-likelihood, free-parameter count (input weights included) and exact path entropy of X (given U)."""
        self._check_fitted()
        X = self._validated_x(X)
        lengths = self._check_lengths(X.shape[0], lengths)
        log_B = self._log_emissions(X)
        log_A = self._log_transitions(U, lengths)
        loglik = float(sequences_log_likelihood(self._log_pi, log_A, log_B, *self._bound_arrays(lengths)))
        entropy = 0.0
        if with_entropy:
            for a, z in self._bounds(lengths):
                lA = log_A[a:z] if log_A.shape[0] > 1 else log_A
                entropy += self._path_entropy_one(self._log_pi, lA, log_B[a:z])
        return SelectionScores(log_likelihood=loglik, n_params=self._n_free_params(X.shape[1]), n_obs=X.shape[0],
                               path_entropy=entropy if with_entropy else None)

    def path_entropy(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, *,
                     U: Optional[np.ndarray] = None) -> float:
        """Exact entropy of the posterior over state paths ``H(z_1:T | x, u)`` in nats (open-ended)."""
        scores = self.selection_scores(X, lengths, U=U, with_entropy=True)
        return float(scores.path_entropy)

    def get_aic_bic(self, X: np.ndarray, *, U: Optional[np.ndarray] = None) -> tuple[float, float]:
        scores = self.selection_scores(X, U=U, with_entropy=False)
        return scores.aic, scores.bic

    def _n_free_params(self, n_features: int) -> int:
        """Start probabilities, transition logits and input weights, and the per-state emission parameters."""
        k = self.n_states
        return (k - 1) + self._transitions.n_free_params + k * self._template()._component_param_count(n_features)

    def _path_entropy_one(self, log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> float:
        """BasePomegranateHMM._path_entropy_one with ``A_t`` per bar and an open end (``log_end = 0``)."""
        T, K = log_B.shape
        time_varying = log_A.shape[0] > 1
        log_alpha = forward_log(log_pi, log_A, log_B)
        log_beta = backward_log(log_A, log_B, np.zeros(K))
        log_z = logsumexp(log_alpha[-1])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gamma_0 = np.exp(log_alpha[0] + log_beta[0] - log_z)
            entropy = -float(np.sum(np.where(gamma_0 > 0, gamma_0 * np.log(np.where(gamma_0 > 0, gamma_0, 1.0)), 0.0)))
            for start in range(1, T, self._ENTROPY_CHUNK_BARS):
                stop = min(start + self._ENTROPY_CHUNK_BARS, T)
                lA = log_A[start:stop] if time_varying else log_A
                log_q = (lA + (log_B[start:stop] + log_beta[start:stop])[:, np.newaxis, :]
                         - log_beta[start - 1:stop - 1][:, :, np.newaxis])
                q = np.exp(log_q)
                cond_entropy = -np.sum(np.where(q > 0, q * log_q, 0.0), axis=2)
                gamma_prev = np.exp(log_alpha[start - 1:stop - 1] + log_beta[start - 1:stop - 1] - log_z)
                entropy += float(np.sum(np.where(gamma_prev > 0, gamma_prev * cond_entropy, 0.0)))
        return entropy

    # ------------------------------------------------------------------
    # Emission parameters
    # ------------------------------------------------------------------
    @property
    def means(self):
        self._check_fitted()
        return self.extract_property("mean")

    @property
    def covariances(self):
        self._check_fitted()
        return self.extract_property("cov")

    @property
    def parameters(self):
        """A list of parameter dicts per state, as ``PomegranateHMM.parameters``."""
        self._check_fitted()
        results = []
        for d in self._distributions:
            match self.distribution_type:
                case DistType.BERNOULLI:
                    results.append({"probs": d.probs.detach().cpu().numpy()})
                case DistType.EXPONENTIAL:
                    results.append({"scales": d.scales.detach().cpu().numpy()})
                case DistType.GAMMA:
                    results.append({"shapes": d.shapes.detach().cpu().numpy(), "rates": d.rates.detach().cpu().numpy()})
                case DistType.LAMDA:
                    results.append({"means": d.mu.detach().cpu().numpy(), "sigma": d.sigma.detach().cpu().numpy(),
                                    "lambda": d.lambda_.detach().cpu().numpy()})
                case DistType.POISSON:
                    results.append({"lambdas": d.lambdas.detach().cpu().numpy()})
                case DistType.LOGNORMAL | DistType.NORMAL | DistType.STUDENTT:
                    params = {"means": d.means.detach().cpu().numpy(), "covs": d.covs.detach().cpu().numpy()}
                    if self.distribution_type == DistType.STUDENTT:
                        params["dofs"] = d.dofs.detach().cpu().numpy()
                    results.append(params)
                case _:
                    results.append(d.parameters())
        return results

    def extract_property(self, prop: str):
        """Generic extractor for distribution parameters."""
        self._check_fitted()
        values = []
        for d in self._distributions:
            if hasattr(d, prop):
                values.append(getattr(d, prop))
            elif hasattr(d, "distributions"):
                values.append([getattr(sub, prop) for sub in d.distributions])
            else:
                values.append(None)
        return np.array(values, dtype=object)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def regime_summary(self, X: Optional[np.ndarray] = None, *, U: Optional[np.ndarray] = None) -> str:
        self._check_fitted()
        lines = ["=" * 70, f"IOHMM Regime Summary ({self.distribution_type.name})", "=" * 70,
                 f"Number of states: {self.n_states}", f"Number of inputs: {self._n_inputs}", ""]
        if X is not None:
            states = self.predict(X, U=U)
            lines.append("State Occupancy:")
            for i in range(self.n_states):
                count = int(np.sum(states == i))
                lines.append(f"  State {i}: {count:6d} samples ({100 * count / len(states):5.2f}%)")
            lines.append("")
        label = "at the training-mean input" if self._n_inputs else ""
        lines.append(f"Transition Probabilities {label}:".replace(" :", ":"))
        for i, row in enumerate(self.transition_prob()):
            lines.append(f"  From State {i}: [{' '.join(f'{p:6.3f}' for p in row)}]")
        lines.append("")
        if self._n_inputs:
            lines.append("Input effects (log-odds of the move vs staying, per input SD):")
            for row in self.input_effects().itertuples(index=False):
                lines.append(f"  {row.from_state} -> {row.to_state}  {row.input}: {row.log_odds_per_sd:+.4f}")
            lines.append("")
        lines.append("Distribution Parameters:")
        for state_idx, state_params in enumerate(self.parameters):
            lines.append(f"State {state_idx}:")
            for key, val in state_params.items():
                is_array = isinstance(val, np.ndarray)
                val_str = np.array2string(val, precision=4, suppress_small=True) if is_array else str(val)
                lines.append(f"  {key}: {val_str}")
            lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def plot_distributions(self, X: Optional[np.ndarray] = None, feature_idx: int = 0, x_range: Optional[tuple] = None,
                           n_points: int = 200, figsize=(12, 4), *, U: Optional[np.ndarray] = None):
        """Per-state emission densities (Normal, StudentT, LogNormal, Gamma, Exponential), over the data if given."""
        from scipy.stats import expon, gamma, lognorm, norm, t

        self._check_fitted()
        params = self.parameters
        if x_range is None:
            if X is not None:
                X = self._preprocess_input(X)
                x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
                margin = (x_max - x_min) * 0.2
                x_range = (x_min - margin, x_max + margin)
            else:
                x_range = (-5, 5)
        x = np.linspace(x_range[0], x_range[1], n_points)
        states = self.predict(X, U=U) if X is not None else None
        fig, axes = plt.subplots(1, self.n_states, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        for k in range(self.n_states):
            ax, p = axes[k], params[k]
            if states is not None and np.any(states == k):
                ax.hist(X[states == k, feature_idx], bins=30, density=True, alpha=0.3, color="gray", label="Data")
            density = None
            if "means" in p and "covs" in p:
                mean = p["means"][feature_idx]
                covs = np.atleast_1d(p["covs"])
                var = covs[feature_idx, feature_idx] if covs.ndim == 2 else covs[min(feature_idx, len(covs) - 1)]
                std = np.sqrt(var)
                if self.distribution_type == DistType.STUDENTT:
                    density = t.pdf(x, df=np.atleast_1d(p["dofs"])[0], loc=mean, scale=std)
                elif self.distribution_type == DistType.LOGNORMAL:
                    density = lognorm.pdf(x, s=std, scale=np.exp(mean))
                else:
                    density = norm.pdf(x, loc=mean, scale=std)
            elif "shapes" in p and "rates" in p:
                density = gamma.pdf(x, a=p["shapes"][feature_idx], scale=1 / p["rates"][feature_idx])
            elif "scales" in p:
                density = expon.pdf(x, scale=p["scales"][feature_idx])
            if density is not None:
                ax.plot(x, density, "b-", linewidth=2, label="Distribution")
            ax.set_title(f"State {k}")
            ax.set_xlabel("Value")
            if k == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(f"State Distributions ({self.distribution_type.name})", fontsize=12, y=1.02)
        plt.tight_layout()
        return fig, axes

    def plot_transition_matrix(self, ax=None, figsize=(8, 6), cmap="Blues", *, u: Optional[np.ndarray] = None):
        """Heatmap of ``transition_prob(u)`` (the training-mean input by default)."""
        trans_prob = self.transition_prob(u)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(trans_prob, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax).set_label("Probability", rotation=270, labelpad=20)
        for i in range(self.n_states):
            for j in range(self.n_states):
                ax.text(j, i, f"{trans_prob[i, j]:.3f}", ha="center", va="center",
                        color="black" if trans_prob[i, j] < 0.5 else "white")
        ax.set_xticks(np.arange(self.n_states))
        ax.set_yticks(np.arange(self.n_states))
        ax.set_xticklabels([f"State {i}" for i in range(self.n_states)])
        ax.set_yticklabels([f"State {i}" for i in range(self.n_states)])
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        at_mean = self._n_inputs > 0 and u is None
        ax.set_title("Transition Probability Matrix" + (" (training-mean input)" if at_mean else ""))
        return ax

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save model to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PomegranateIOHMM":
        """Load model from disk."""
        return joblib.load(path)

    def __repr__(self) -> str:
        return (f"PomegranateIOHMM(n_states={self.n_states}, distribution={str(self.distribution_type)}, "
                f"n_inputs={self._n_inputs}, backend=numba)")

    # ------------------------------------------------------------------
    # Internals: inputs
    # ------------------------------------------------------------------
    def _setup_inputs(self, U: Optional[np.ndarray], lengths: list[int]) -> Optional[np.ndarray]:
        """Record the fit's input layout and standardisation; return the standardised effective inputs (or None)."""
        if U is None:
            self._n_inputs, self._input_timing = 0, []
            self._u_mean = self._u_std = self._u_min = self._u_max = np.zeros(0)
            return None
        U = self._input_matrix(U, int(np.sum(lengths)))
        self._n_inputs = U.shape[1]
        self._input_timing = self._resolve_timing(U.shape[1])
        effective = self._align_inputs(U, lengths)
        used = effective[self._transition_rows(lengths)]
        if used.shape[0] == 0:
            raise ValueError("No transitions to learn from: every sequence has length 1.")
        mean, std = used.mean(axis=0), used.std(axis=0)
        constant = np.flatnonzero(std <= 1e-12 * np.maximum(1.0, np.abs(mean)))
        if constant.size:
            raise ValueError(f"Input column(s) {constant.tolist()} are constant over the transitions; a constant input "
                             "duplicates the baseline logits and cannot be identified.")
        self._u_mean, self._u_std = mean, std
        self._u_min, self._u_max = used.min(axis=0), used.max(axis=0)
        return (effective - mean) / std

    def _resolve_timing(self, n_inputs: int) -> list[InputTiming]:
        timing = self.input_timing
        if isinstance(timing, str):
            return [InputTiming(timing)] * n_inputs
        resolved = [InputTiming(t) for t in timing]
        if len(resolved) != n_inputs:
            raise ValueError(f"input_timing has {len(resolved)} entries but U has {n_inputs} columns.")
        return resolved

    def _align_inputs(self, U: np.ndarray, lengths: list[int]) -> np.ndarray:
        """Effective inputs: AT_BAR_CLOSE columns shifted one bar within each sequence (row 0 is a placeholder)."""
        effective = U.copy()
        shifted = [c for c, timing in enumerate(self._input_timing) if timing is InputTiming.AT_BAR_CLOSE]
        if shifted:
            for a, z in self._bounds(lengths):
                effective[a + 1:z, shifted] = U[a:z - 1, shifted]
        return effective

    def _standardise(self, U_effective: np.ndarray) -> np.ndarray:
        return (U_effective - self._u_mean) / self._u_std

    def _log_transitions(self, U: Optional[np.ndarray], lengths: list[int]) -> np.ndarray:
        if self._n_inputs == 0:
            if U is not None:
                raise ValueError("This model was fitted without inputs; U must be None.")
            return self._transitions.log_matrices(None)
        if U is None:
            raise ValueError(f"This model was fitted with {self._n_inputs} input(s); pass U.")
        U = self._input_matrix(U, int(np.sum(lengths)))
        return self._transitions.log_matrices(self._standardise(self._align_inputs(U, lengths)))

    def _input_matrix(self, U, n_rows: Optional[int]) -> np.ndarray:
        if isinstance(U, (pd.DataFrame, pd.Series)):
            U = U.to_numpy()
        U = np.asarray(U, dtype=np.float64)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
        if U.ndim != 2:
            raise ValueError(f"U must be 1-D or 2-D, got {U.ndim}-D")
        if n_rows is not None and U.shape[0] != n_rows:
            raise ValueError(f"U has {U.shape[0]} rows but X has {n_rows}.")
        if self._n_inputs and U.shape[1] != self._n_inputs:
            raise ValueError(f"U has {U.shape[1]} columns but the model was fitted with {self._n_inputs}.")
        if not np.all(np.isfinite(U)):
            raise ValueError("U contains NaN or Inf values. Clean the inputs first.")
        return U

    def _input_vector(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        if u.shape != (self._n_inputs,):
            raise ValueError(f"u must have {self._n_inputs} entries, got {u.shape[0]}")
        if not np.all(np.isfinite(u)):
            raise ValueError("u contains NaN or Inf values.")
        return u

    # ------------------------------------------------------------------
    # Internals: shared
    # ------------------------------------------------------------------
    def _clear_fit(self) -> None:
        self._distributions = None
        self._log_pi = None
        self._transitions = None
        self._n_inputs, self._input_timing = 0, []
        self._u_mean = self._u_std = self._u_min = self._u_max = np.zeros(0)

    def _check_fitted(self) -> None:
        if self._distributions is None:
            raise RuntimeError("Model has not been fitted yet.")

    def _template(self, random_state: Optional[int] = None) -> PomegranateHMM:
        """A ``PomegranateHMM`` with this model's emission settings: the source of init, builders and counts."""
        return PomegranateHMM(self.distribution_type, n_states=self.n_states,
                              random_state=self.random_state if random_state is None else random_state,
                              max_iter=self.max_iter, inference_mode=self.inference_mode, **self.dist_kwargs)

    def _inference_inputs(self, X: np.ndarray, U: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(log_pi, log_A, log_B)`` for one sequence X (given U); ``log_A`` is (1, K, K) or (T, K, K)."""
        self._check_fitted()
        X = self._validated_x(X)
        return self._log_pi, self._log_transitions(U, [X.shape[0]]), self._log_emissions(X)

    def _validated_x(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess_input(X)
        if X.size == 0:
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or Inf values. Clean input data first.")
        return X

    def _log_emissions(self, X: np.ndarray) -> np.ndarray:
        return log_emissions(self._distributions, np.ascontiguousarray(X, dtype=np.float64))

    def _smoothed(self, log_pi: np.ndarray, log_A: np.ndarray, log_B: np.ndarray) -> np.ndarray:
        log_alpha = forward_log(log_pi, log_A, log_B)
        log_beta = backward_log(log_A, log_B, np.zeros(self.n_states))
        return self._normalise_rows(log_alpha + log_beta)

    @staticmethod
    def _normalise_rows(log_p: np.ndarray) -> np.ndarray:
        p = np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    @staticmethod
    def _check_feasible(log_delta: np.ndarray, caller: str) -> None:
        dead = np.isneginf(log_delta).all(axis=1)
        if dead.any():
            raise ValueError(f"{caller}: no feasible state path exists at bar {int(np.argmax(dead))} - every candidate "
                             "path is forced through a zero-probability transition or emission.")

    @staticmethod
    def _check_lag(lag) -> int:
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)):
            raise ValueError(f"lag must be an integer, got {type(lag).__name__}")
        if lag < 0:
            raise ValueError(f"lag must be >= 0, got {lag}")
        return int(lag)

    @staticmethod
    def _preprocess_input(X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    @staticmethod
    def _check_lengths(n_rows: int, lengths: Optional[Sequence[int]]) -> list[int]:
        if lengths is None:
            return [n_rows]
        lengths = [int(n) for n in lengths]
        if any(n < 1 for n in lengths) or sum(lengths) != n_rows:
            raise ValueError(f"lengths must be positive and sum to the number of rows ({n_rows}), got {lengths}")
        return lengths

    @staticmethod
    def _bounds(lengths: list[int]) -> list[tuple[int, int]]:
        ends = np.cumsum(lengths)
        return list(zip(np.concatenate([[0], ends[:-1]]).tolist(), ends.tolist()))

    @staticmethod
    def _bound_arrays(lengths: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """``(starts, ends)`` row bounds of the sequences, as int64 arrays for the numba kernels."""
        ends = np.cumsum(np.asarray(lengths, dtype=np.int64))
        return ends - np.asarray(lengths, dtype=np.int64), ends

    @classmethod
    def _transition_rows(cls, lengths: list[int]) -> np.ndarray:
        """Mask of rows that are the destination of a transition (every row but each sequence's first)."""
        rows = np.ones(int(np.sum(lengths)), dtype=bool)
        rows[[a for a, _ in cls._bounds(lengths)]] = False
        return rows

    @staticmethod
    def _split(X: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
        return [X[a:z] for a, z in PomegranateIOHMM._bounds(lengths)]
