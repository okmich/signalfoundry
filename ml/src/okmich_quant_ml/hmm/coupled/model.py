"""Coupled HMM with exact inference on the joint state space.

C chains; chain c has K_c states. Its next state depends on the previous states of its ``parents`` (itself always
included), so the joint transition factorises as

    P(z_t | z_{t-1}) = prod_c A_c(s_c,t | s_p,t-1 for p in parents[c])

Every chain's parents = {itself} gives a factorial HMM. A chain in ``observed_chains`` is clamped by a
``ClampedEmission``: its own transitions are fixed uniform (a constant factor) and only its effect on other chains is
learnt, which is the input-output HMM's one-way coupling with a one-hot input. Inference runs on all J = prod K_c joint
states: exact, and cheap for a few small chains (a direction x volatility pair is 3 x 2 = 6 states).

Reference: Brand, Oliver & Pentland (1997), "Coupled hidden Markov models for complex action recognition".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import joblib
import numpy as np

from ..util import InferenceMode
from .emissions import ClampedEmission, Emission
from .kernels import forward_backward, forward_filter

MAX_JOINT_STATES = 256


@dataclass
class FitReport:
    """The winning start's EM run: final objective (log-likelihood plus the transition prior), iterations, history."""

    loglik: float
    n_iter: int
    converged: bool
    restart: int               # fresh start index, or -1 - i for warm start i
    history: list[float]


class CoupledHMM:
    """Coupled HMM: per-chain transitions conditioned on parent chains, emissions as blocks over the chains.

    Parameters
    ----------
    n_states : states per chain, e.g. (3, 2).
    emissions : emission blocks; each reads one column of X (``GaussianEmission``, ``LocationScaleEmission``,
        ``ClampedEmission``). A column may be read by several blocks; a chain may have none.
    parents : per chain, the chains whose previous state its transition is conditioned on (itself is always added).
        None = no coupling (factorial).
    observed_chains : chains clamped to observed states by a ``ClampedEmission``.
    n_init : fresh EM starts; ``fit`` adds one start per warm-start model.
    max_iter, tol : EM stops when the objective improves by less than ``tol`` per bar.
    transition_prior : Dirichlet pseudo-count added to every transition cell (MAP-EM; 0 = maximum likelihood).
    sticky_init : stay probabilities of the fresh starts' transition tables, cycled over starts.
    random_state : seed of the fresh starts' jitter.
    inference_mode : FILTERING (causal, default) or SMOOTHING (look-ahead: teacher / diagnostic use only).

    Examples
    --------
    >>> from okmich_quant_ml.hmm.coupled import CoupledHMM, GaussianEmission, LocationScaleEmission
    >>> DIR, VOL = 0, 1
    >>> model = CoupledHMM((3, 2), [LocationScaleEmission(0, location=DIR, scale=VOL), GaussianEmission(1, chain=VOL)],
    ...                    parents=[(DIR, VOL), (VOL, DIR)])
    >>> model.fit(X)                                   # X[:, 0] = returns / clock scale, X[:, 1] = log realised var
    >>> direction = model.predict_proba(X, chain=DIR)  # causal P(bear, flat, bull) after sort_states
    """

    def __init__(self, n_states: Sequence[int], emissions: Sequence[Emission],
                 parents: Optional[Sequence[Sequence[int]]] = None, *,
                 observed_chains: Sequence[int] = (), n_init: int = 4, max_iter: int = 300, tol: float = 1e-6,
                 transition_prior: float = 1e-3, sticky_init: Sequence[float] = (0.95, 0.98, 0.9, 0.99),
                 random_state: int = 0, inference_mode: InferenceMode = InferenceMode.FILTERING):
        self.n_states = tuple(int(k) for k in n_states)
        if not self.n_states or any(k < 1 for k in self.n_states):
            raise ValueError(f"n_states needs at least one chain with >= 1 state, got {n_states}")
        n_chains = len(self.n_states)
        if parents is None:
            parents = [()] * n_chains
        if len(parents) != n_chains:
            raise ValueError(f"parents needs one entry per chain ({n_chains}), got {len(parents)}")
        if any(p < 0 or p >= n_chains for ps in parents for p in ps):
            raise ValueError(f"parents must be chain indices in 0..{n_chains - 1}, got {parents}")
        self.parents = tuple(tuple(sorted(set(int(p) for p in ps) | {c})) for c, ps in enumerate(parents))
        self.n_joint = int(np.prod(self.n_states))
        if self.n_joint > MAX_JOINT_STATES:
            raise ValueError(f"{self.n_joint} joint states exceed {MAX_JOINT_STATES}; exact inference would be slow")
        if not emissions:
            raise ValueError("at least one emission block is needed")
        self.observed_chains = tuple(int(c) for c in observed_chains)
        clamped = {e.chain for e in emissions if isinstance(e, ClampedEmission)}
        if set(self.observed_chains) != clamped:
            raise ValueError("observed_chains must be exactly the chains that have a ClampedEmission")
        if n_init < 0 or max_iter < 1 or tol < 0 or transition_prior < 0:
            raise ValueError("n_init >= 0, max_iter >= 1, tol >= 0 and transition_prior >= 0 are required")
        self.emissions = list(emissions)
        self.n_init, self.max_iter, self.tol = int(n_init), int(max_iter), float(tol)
        self.transition_prior, self.sticky_init = float(transition_prior), tuple(sticky_init)
        self.random_state, self.inference_mode = random_state, InferenceMode(inference_mode)

        self.states = np.array(np.unravel_index(np.arange(self.n_joint), self.n_states)).T   # (J, C)
        self._rows = [np.ravel_multi_index(tuple(self.states[:, p] for p in ps), [self.n_states[p] for p in ps])
                      for ps in self.parents]                                                 # per chain: (J,)
        for e in self.emissions:
            e.bind(self.states, self.n_states)
        self.transitions: list[np.ndarray] | None = None    # chain c: (*[K_p for p in parents[c]], K_c)
        self.pi: np.ndarray | None = None
        self.fit_report: FitReport | None = None

    # --------------------------------------------------------------------------------------------------------------
    # Fitting
    # --------------------------------------------------------------------------------------------------------------
    def fit(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
            warm_start: "CoupledHMM | Sequence[CoupledHMM | None] | None" = None) -> "CoupledHMM":
        """EM from ``n_init`` fresh starts plus one start per ``warm_start`` model; keeps the best objective.

        A warm start may have other parent sets (e.g. a factorial fit starting a coupled one): each of its tables is
        averaged over parents this model lacks and repeated over parents it adds, so the coupled fit starts exactly at
        the factorial solution. Emission blocks are matched by kind, column and chains; unmatched blocks start fresh.
        """
        X = self._check_X(X)
        starts = self._starts(len(X), lengths)
        rng = np.random.default_rng(self.random_state)
        warm = [] if warm_start is None else [warm_start] if isinstance(warm_start, CoupledHMM) else list(warm_start)
        warm = [w for w in warm if w is not None]
        if self.n_init + len(warm) == 0:
            raise ValueError("No start: set n_init >= 1 or give a warm_start")
        best = None
        for restart in list(range(self.n_init)) + [("warm", i) for i in range(len(warm))]:
            if isinstance(restart, tuple):
                self._init_params(X, rng, 0)          # blocks the warm model lacks keep a fresh start
                self._copy_params(warm[restart[1]])
            else:
                self._init_params(X, rng, restart)
            report = self._em(X, starts, restart)
            if best is None or report.loglik > best[0].loglik:
                best = (report, self._snapshot())
        self.fit_report = best[0]
        self._restore(best[1])
        return self

    def _init_params(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        stay = self.sticky_init[restart % len(self.sticky_init)]
        self.transitions = []
        for c, ps in enumerate(self.parents):
            k = self.n_states[c]
            shape = tuple(self.n_states[p] for p in ps) + (k,)
            if c in self.observed_chains or k == 1:
                table = np.full(shape, 1.0 / k)
            else:
                own = ps.index(c)
                table = np.full(shape, (1.0 - stay) / (k - 1))
                grid = np.indices(shape)
                table[grid[own] == grid[-1]] = stay
            self.transitions.append(table)
        self.pi = np.full(self.n_joint, 1.0 / self.n_joint)
        for e in self.emissions:
            e.init(X, rng, restart)

    def _em(self, X: np.ndarray, starts: np.ndarray, restart) -> FitReport:
        history: list[float] = []
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            log_b = self._log_emissions(X)
            gamma, xi, start_sum, loglik = forward_backward(log_b, self.transition_matrix(), self.pi, starts)
            history.append(loglik + self._log_prior())
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol * len(X):
                converged = True
                break
            self._m_step(X, gamma, xi, start_sum)
        else:
            history.append(self._loglik(X, starts) + self._log_prior())
        return FitReport(history[-1], n_iter, converged, -1 - restart[1] if isinstance(restart, tuple) else restart,
                         history)

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray, start_sum: np.ndarray) -> None:
        for c, ps in enumerate(self.parents):
            if c in self.observed_chains:
                continue
            n_rows = int(np.prod([self.n_states[p] for p in ps]))
            counts = np.zeros((n_rows, self.n_states[c]))
            np.add.at(counts, (self._rows[c][:, None], self.states[None, :, c]), xi)
            counts += self.transition_prior
            total = counts.sum(axis=1, keepdims=True)
            old = self.transitions[c].reshape(n_rows, -1)
            table = np.where(total > 0, counts / np.where(total > 0, total, 1.0), old)
            self.transitions[c] = table.reshape(self.transitions[c].shape)
        self.pi = np.maximum(start_sum, 1e-12)
        self.pi /= self.pi.sum()
        for e in self.emissions:
            e.m_step(X, gamma)

    def _log_prior(self) -> float:
        """Dirichlet(1 + prior) log density of the transition tables, so EM's objective is the MAP objective."""
        if self.transition_prior <= 0:
            return 0.0
        return float(sum(self.transition_prior * np.log(np.maximum(t, 1e-300)).sum()
                         for c, t in enumerate(self.transitions) if c not in self.observed_chains))

    # --------------------------------------------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------------------------------------------
    def transition_matrix(self) -> np.ndarray:
        """(J, J) joint transition matrix P(z_t = j | z_{t-1} = i)."""
        self._check_fitted()
        trans = np.ones((self.n_joint, self.n_joint))
        for c in range(len(self.n_states)):
            flat = self.transitions[c].reshape(-1, self.n_states[c])
            trans *= flat[self._rows[c][:, None], self.states[None, :, c]]
        return trans

    def filter(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """(T, J) causal joint posteriors P(z_t | x_1..t)."""
        X = self._check_X(X)
        alpha, _ = forward_filter(self._log_emissions(X), self.transition_matrix(), self.pi,
                                  self._starts(len(X), lengths))
        return alpha

    def smooth(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """(T, J) smoothed joint posteriors P(z_t | x_1..T): look-ahead, teacher and diagnostic use only."""
        X = self._check_X(X)
        gamma, _, _, _ = forward_backward(self._log_emissions(X), self.transition_matrix(), self.pi,
                                          self._starts(len(X), lengths))
        return gamma

    def state_posteriors(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """(T, J) joint posteriors under ``inference_mode``."""
        if self.inference_mode is InferenceMode.FILTERING:
            return self.filter(X, lengths)
        if self.inference_mode is InferenceMode.SMOOTHING:
            return self.smooth(X, lengths)
        raise ValueError(f"{self.inference_mode} is not supported; use FILTERING or SMOOTHING")

    def predict_proba(self, X: np.ndarray, chain: int, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """(T, K_chain) posterior of one chain, causal unless ``inference_mode`` is SMOOTHING."""
        return self.chain_marginal(self.state_posteriors(X, lengths), chain)

    def predict(self, X: np.ndarray, chain: int, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        return self.predict_proba(X, chain, lengths).argmax(axis=1)

    def chain_marginal(self, joint: np.ndarray, chain: int) -> np.ndarray:
        """(T, K_chain) marginal of one chain from (T, J) joint posteriors."""
        if not 0 <= chain < len(self.n_states):
            raise ValueError(f"chain must be in 0..{len(self.n_states) - 1}, got {chain}")
        out = np.zeros((joint.shape[0], self.n_states[chain]))
        for k in range(self.n_states[chain]):
            out[:, k] = joint[:, self.states[:, chain] == k].sum(axis=1)
        return out

    def log_likelihood(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> float:
        X = self._check_X(X)
        return self._loglik(X, self._starts(len(X), lengths))

    def compute_per_bar_predictive_loglik(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> np.ndarray:
        """log p(x_t | x_1..t-1) per bar (causal): out-of-sample model comparison and drift monitoring."""
        X = self._check_X(X)
        _, log_pred = forward_filter(self._log_emissions(X), self.transition_matrix(), self.pi,
                                     self._starts(len(X), lengths))
        return log_pred

    def get_aic_bic(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None) -> tuple[float, float]:
        loglik = self.log_likelihood(X, lengths)
        k, n = self.n_free_params(), len(X)
        return 2 * k - 2 * loglik, k * np.log(n) - 2 * loglik

    # --------------------------------------------------------------------------------------------------------------
    # Parameters
    # --------------------------------------------------------------------------------------------------------------
    def chain_transitions(self, chain: int) -> np.ndarray:
        """Chain ``chain``'s table, axes (*parents[chain] previous states, next state): rows sum to 1."""
        self._check_fitted()
        return self.transitions[chain].copy()

    def stationary_distribution(self) -> np.ndarray:
        """(J,) stationary distribution of the joint chain (an observed chain's uniform table makes it notional)."""
        trans = self.transition_matrix()
        values, vectors = np.linalg.eig(trans.T)
        vec = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        vec = np.abs(vec)
        return vec / vec.sum()

    def sort_states(self, chain: int, keys: np.ndarray) -> "CoupledHMM":
        """Relabel ``chain``'s states in ascending order of ``keys`` (one value per state)."""
        self._check_fitted()
        perm = np.argsort(np.asarray(keys))
        if len(perm) != self.n_states[chain]:
            raise ValueError(f"keys needs {self.n_states[chain]} values, got {len(perm)}")
        for c, ps in enumerate(self.parents):
            table = self.transitions[c]
            for axis, p in enumerate(ps):
                if p == chain:
                    table = np.take(table, perm, axis=axis)
            if c == chain:
                table = np.take(table, perm, axis=-1)
            self.transitions[c] = table
        self.pi = np.take(self.pi.reshape(self.n_states), perm, axis=chain).ravel()
        for e in self.emissions:
            e.permute(chain, perm)
        return self

    def n_free_params(self) -> int:
        n = self.n_joint - 1
        for c, ps in enumerate(self.parents):
            if c not in self.observed_chains:
                n += int(np.prod([self.n_states[p] for p in ps])) * (self.n_states[c] - 1)
        return n + sum(e.n_free_params() for e in self.emissions)

    def sample(self, n_bars: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Simulate (states (T, C), X (T, n_columns)) from the current parameters; clamped columns hold the states."""
        self._check_fitted()
        rng = np.random.default_rng(seed)
        cum = np.cumsum(self.transition_matrix(), axis=1)
        joint = np.empty(n_bars, dtype=np.int64)
        joint[0] = rng.choice(self.n_joint, p=self.pi)
        u = rng.random(n_bars)
        for t in range(1, n_bars):
            joint[t] = min(int(np.searchsorted(cum[joint[t - 1]], u[t])), self.n_joint - 1)
        states = self.states[joint]
        X = np.full((n_bars, max(e.column for e in self.emissions) + 1), np.nan)
        for e in self.emissions:
            X[:, e.column] = e.sample(states, rng)
        return states, X

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "CoupledHMM":
        return joblib.load(path)

    def __repr__(self) -> str:
        coupled = [f"{c}<-{[p for p in ps if p != c]}" for c, ps in enumerate(self.parents) if len(ps) > 1]
        return (f"CoupledHMM(n_states={self.n_states}, coupling={coupled or 'none'}, "
                f"observed={list(self.observed_chains)}, fitted={self.transitions is not None})")

    # --------------------------------------------------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------------------------------------------------
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (bars x columns), got shape {X.shape}")
        needed = max(e.column for e in self.emissions) + 1
        if X.shape[1] < needed:
            raise ValueError(f"X has {X.shape[1]} columns; the emission blocks read up to column {needed - 1}")
        if len(X) == 0:
            raise ValueError("X has no rows")
        if np.isinf(X).any():
            raise ValueError("X contains infinite values (NaN is allowed: unobserved)")
        return X

    def _check_fitted(self) -> None:
        if self.transitions is None:
            raise RuntimeError("The model is not fitted; call fit() first")

    def _log_emissions(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        log_b = np.zeros((len(X), self.n_joint))
        for e in self.emissions:
            log_b += e.log_density(X)
        return log_b

    def _loglik(self, X: np.ndarray, starts: np.ndarray) -> float:
        _, log_pred = forward_filter(self._log_emissions(X), self.transition_matrix(), self.pi, starts)
        return float(log_pred.sum())

    @staticmethod
    def _starts(n_bars: int, lengths: Optional[Sequence[int]]) -> np.ndarray:
        starts = np.zeros(n_bars, dtype=np.bool_)
        if lengths is None:
            starts[0] = True
            return starts
        lengths = [int(n) for n in lengths]
        if any(n < 1 for n in lengths) or sum(lengths) != n_bars:
            raise ValueError(f"lengths must be positive and sum to the {n_bars} rows of X, got {lengths}")
        starts[np.cumsum([0] + lengths)[:-1]] = True
        return starts

    def _snapshot(self) -> tuple:
        return [t.copy() for t in self.transitions], self.pi.copy(), [e.get_params() for e in self.emissions]

    def _restore(self, snap: tuple) -> None:
        self.transitions, self.pi = [t.copy() for t in snap[0]], snap[1].copy()
        for e, params in zip(self.emissions, snap[2]):
            e.set_params(params)

    def _copy_params(self, other: "CoupledHMM") -> None:
        if other.n_states != self.n_states:
            raise ValueError("warm_start must have the same chains and states")
        self.transitions = [self._reshape_table(c, other) for c in range(len(self.n_states))]
        for c in self.observed_chains:
            self.transitions[c] = np.full_like(self.transitions[c], 1.0 / self.n_states[c])
        self.pi = other.pi.copy()
        sources = {(type(src), src.column, src.chains): src for src in other.emissions}
        for e in self.emissions:
            src = sources.get((type(e), e.column, e.chains))
            if src is not None:
                e.load_params(src)

    def _reshape_table(self, c: int, other: "CoupledHMM") -> np.ndarray:
        """Chain c's table in this model's parent layout, from ``other``'s table averaged over parents it lacks."""
        src, src_parents, own = other.transitions[c], other.parents[c], self.parents[c]
        drop = tuple(i for i, p in enumerate(src_parents) if p not in own)
        reduced = src.mean(axis=drop) if drop else src
        # parent tuples are sorted, so the kept axes are already in this model's order
        view_shape = [self.n_states[p] if p in src_parents else 1 for p in own] + [self.n_states[c]]
        shape = tuple(self.n_states[p] for p in own) + (self.n_states[c],)
        return np.broadcast_to(reduced.reshape(view_shape), shape).copy()
