"""
Input-driven transition model: one multinomial logit per origin state, with "stay" as the reference category.

For origin state ``i`` and destination ``j``::

    eta_t[i, j] = b[i, j] + W[i, j] . u_t        (eta_t[i, i] = 0: b[i, i] = 0, W[i, i] = 0)
    log A_t[i, j] = eta_t[i, j] - logsumexp_k eta_t[i, k]

so ``b[i, j]`` is the baseline log-odds of moving ``i -> j`` rather than staying, and ``W[i, j, m]`` is how much one
unit of input ``m`` adds to those log-odds. ``u_t`` are the model's standardised inputs. With no inputs this is an
ordinary transition matrix written as row-wise logits.

The weights are per (origin, destination) pair, as in statsmodels' TVTP and Mogeng/IOHMM, not per destination only
as in ``ssm.InputDrivenTransitions``. Per-destination weights cannot express "this input makes switching more likely
in every direction" (for K = 2 the input's effect on leaving state 0 is forced to be the exact negative of its effect
on leaving state 1), and that symmetric hazard is the effect a clock input is expected to have.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .kernels import input_log_matrices, logit_row_objective, logsumexp

_INTERCEPT_BOUND = 30.0  # |b| cap in the M-step: exp(-30) ~ 1e-13, so a transition never seen stays finite
_TINY = np.finfo(np.float64).tiny


class InputDrivenTransitions:
    """Transition logits ``b`` (K, K) and input weights ``W`` (K, K, d), with an L2 penalty on ``W`` only."""

    def __init__(self, n_states: int, n_inputs: int = 0, l2_penalty: float = 1.0):
        if n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {n_states}")
        if n_inputs < 0:
            raise ValueError(f"n_inputs must be >= 0, got {n_inputs}")
        if l2_penalty < 0:
            raise ValueError(f"l2_penalty must be >= 0, got {l2_penalty}")
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.l2_penalty = float(l2_penalty)
        self.b = np.zeros((n_states, n_states), dtype=np.float64)
        self.W = np.zeros((n_states, n_states, n_inputs), dtype=np.float64)

    @classmethod
    def from_log_matrix(cls, log_A: np.ndarray, n_inputs: int = 0, l2_penalty: float = 1.0) -> "InputDrivenTransitions":
        """Baseline logits from a (K, K) log transition matrix (rows need not be normalised); input weights at zero."""
        log_A = np.asarray(log_A, dtype=np.float64)
        if log_A.ndim != 2 or log_A.shape[0] != log_A.shape[1]:
            raise ValueError(f"log_A must be square (K, K), got shape {log_A.shape}")
        model = cls(log_A.shape[0], n_inputs=n_inputs, l2_penalty=l2_penalty)
        log_A = log_A - logsumexp(log_A, axis=1, keepdims=True)
        model.b = np.clip(log_A - np.diag(log_A)[:, None], -_INTERCEPT_BOUND, _INTERCEPT_BOUND)
        np.fill_diagonal(model.b, 0.0)
        return model

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def log_matrices(self, U: np.ndarray | None = None) -> np.ndarray:
        """Log transition matrices: ``(1, K, K)`` with no inputs, ``(T, K, K)`` for standardised inputs ``U`` (T, d)."""
        if self.n_inputs == 0:
            if U is not None and np.asarray(U).size:
                raise ValueError("This transition model has no inputs, but U was given.")
            eta = self.b[np.newaxis]
            return eta - logsumexp(eta, axis=2, keepdims=True)
        if U is None:
            raise ValueError(f"This transition model needs {self.n_inputs} input(s), but U is None.")
        U = np.ascontiguousarray(U, dtype=np.float64)
        if U.ndim != 2 or U.shape[1] != self.n_inputs:
            raise ValueError(f"U must have shape (T, {self.n_inputs}), got {U.shape}")
        return input_log_matrices(self.b, self.W, U)

    def penalty(self) -> float:
        """The L2 penalty ``0.5 * l2_penalty * ||W||^2`` that the M-step subtracts from the log-likelihood."""
        return 0.5 * self.l2_penalty * float(np.sum(self.W ** 2))

    @property
    def n_free_params(self) -> int:
        return self.n_states * (self.n_states - 1) * (1 + self.n_inputs)

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
    def fit(self, xi: np.ndarray, U: np.ndarray | None = None) -> None:
        """
        Maximise ``sum_t sum_ij xi_t[i, j] log A_t[i, j] - penalty`` over ``b`` and ``W``.

        ``xi`` holds expected transition counts, shape ``(N, K, K)``: one row per transition (from the E-step), or a
        single summed row when there are no inputs. ``U`` (N, d) holds the standardised inputs of those transitions.
        With no inputs the maximiser is closed form (the Baum-Welch update). With inputs each origin row is a convex
        weighted multinomial logistic regression with soft targets, solved by L-BFGS from the current values.
        """
        xi = np.asarray(xi, dtype=np.float64)
        if self.n_inputs == 0:
            self._fit_closed_form(xi.reshape(-1, self.n_states, self.n_states).sum(axis=0))
            return
        U = np.asarray(U, dtype=np.float64)
        if U.shape != (xi.shape[0], self.n_inputs):
            raise ValueError(f"U must have shape ({xi.shape[0]}, {self.n_inputs}), got {U.shape}")
        X1 = np.column_stack([np.ones(len(U)), U])
        for i in range(self.n_states):
            self._fit_row(i, xi[:, i, :], X1)

    def _fit_closed_form(self, counts: np.ndarray) -> None:
        for i in range(self.n_states):
            total = counts[i].sum()
            if total <= _TINY:
                continue  # state never occupied: keep its row
            log_row = np.log(np.maximum(counts[i] / total, _TINY))
            self.b[i] = np.clip(log_row - log_row[i], -_INTERCEPT_BOUND, _INTERCEPT_BOUND)
            self.b[i, i] = 0.0

    def _fit_row(self, i: int, counts: np.ndarray, X1: np.ndarray) -> None:
        """Origin row ``i``: ``counts`` (N, K) soft destination counts, ``X1`` (N, 1 + d) = [1, u_t]."""
        K, d = self.n_states, self.n_inputs
        others = np.array([j for j in range(K) if j != i])
        n_out = counts.sum(axis=1)
        total = float(n_out.sum())
        if total <= _TINY:
            return  # state never occupied: keep its row
        lam = self.l2_penalty / total  # objective is divided by total for conditioning, so the penalty is too
        moves = np.ascontiguousarray(counts[:, others])  # eta of "stay" is 0, so only the moves enter the sums
        n_out = np.ascontiguousarray(n_out)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            coef = theta.reshape(K - 1, d + 1)
            nll, grad = logit_row_objective(coef, X1, n_out, moves, total)
            nll += 0.5 * lam * float(np.sum(coef[:, 1:] ** 2))
            grad[:, 1:] += lam * coef[:, 1:]
            return nll, grad.ravel()

        theta0 = np.column_stack([self.b[i, others], self.W[i, others, :]]).ravel()
        intercept = (-_INTERCEPT_BOUND, _INTERCEPT_BOUND)
        bounds = [intercept if m % (d + 1) == 0 else (None, None) for m in range(theta0.size)]
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        coef = res.x.reshape(K - 1, d + 1)
        self.b[i, others] = coef[:, 0]
        self.W[i, others, :] = coef[:, 1:]
