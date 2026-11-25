"""General hierarchical HMM with exact inference on the flattened leaf chain and EM on the tree's own parameters.

The model
---------
A tree of states (``StateNode``). Every internal node q has an entry distribution pi_q over its children, horizontal
transitions A_q between them and exit probabilities e_q (see ``topology``). Leaves emit through one or more
``LeafEmission`` blocks. After leaf i emits, control climbs the tree: at each ancestor the active child either ends its
parent (exit) or moves horizontally to a sibling, and a horizontal move is followed by entries down to a new leaf.

Flattening (Murphy & Paskin 2001)
---------------------------------
The probability of moving from leaf i to leaf j is the sum, over the ancestors of i at which the horizontal move can
happen, of (exits below that ancestor) x (no exit there) x A x (entries down to j). The leaf chain is therefore an
ordinary HMM whose transition matrix is a function of the tree's parameters; inference on it is exact at O(T N^2).
An exit schedule (``ExitSchedule``) can forbid or force exits per depth and bar, giving one flat matrix per pattern.

Learning
--------
EM treats the level at which each transition happened as a latent variable. Given a pair (i -> j) and its pattern,
the posterior over that level is each path's share of the flat probability, so the expected counts of exits,
non-exits, horizontal moves and entries are the pairwise posteriors split by path. The M-step then updates pi, A and e
in closed form (Dirichlet / Beta pseudo-counts ``transition_prior``, i.e. MAP-EM; 0 = maximum likelihood). Structural
zeros stay zero. This is exact EM for the tree's parameters, not a masked flat HMM.

References: Fine, Singer & Tishby (1998), "The hierarchical hidden Markov model"; Murphy & Paskin (2001),
"Linear-time inference in hierarchical HMMs"; Adam et al. (2019) for exit schedules (embedded HHMM).
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import joblib
import numpy as np

from ...util import InferenceMode, SelectionScores
from .emissions import LeafEmission
from .kernels import (causal_viterbi, causal_viterbi_scores, fixed_lag, fixed_lag_windowed, forward_backward,
                      forward_filter, path_entropy, viterbi)
from .schedule import ExitMode, ExitSchedule, resolve_schedule
from .topology import StateNode, Topology

MAX_LEAVES = 512
_ROW_TOL = 1e-9


@dataclass
class FitReport:
    """The winning start's EM run.

    ``objective`` is what EM maximises and restarts are ranked on: the log-likelihood plus the log priors implied by
    the pseudo-counts (``transition_prior`` and the categorical ``pseudo_count``). ``loglik`` is the raw
    log-likelihood. ``history`` is the objective per iteration.
    """

    objective: float
    loglik: float
    n_iter: int
    converged: bool
    restart: int               # fresh start index, or -1 - i for warm start i
    history: list[float]


@dataclass(frozen=True)
class _Path:
    """One way leaf i can move on: horizontal move at node q from child c to child c2, after the listed exits."""

    q: int
    c: int
    c2: int
    weight: float
    exits: tuple[tuple[int, int], ...]   # (node, child) pairs that ended on the way up


class HHMM:
    """Hierarchical hidden Markov model of any depth.

    Parameters
    ----------
    root : the state tree (``StateNode``); its leaves are the emitting states.
    emissions : emission blocks (``CategoricalEmission``, ``GaussianEmission``, ``StudentTEmission``); a leaf's
        density is the product over blocks. Blocks may tie their parameters at a tree level.
    n_init : fresh EM starts; ``fit`` adds one start per warm-start model.
    max_iter, tol : EM stops when the objective improves by less than ``tol`` per bar.
    transition_prior : pseudo-count added to every free entry, horizontal and exit/non-exit cell (MAP-EM).
    sticky_init : probability of staying in the same child in the fresh starts' horizontal tables (where a self-move
        is allowed), cycled over starts.
    exit_init : starting exit probability of every learnable exit, cycled over starts.
    random_state : seed of the fresh starts.
    inference_mode : default recursion of ``predict``/``predict_proba``. FILTERING and CAUSAL_VITERBI are causal;
        SMOOTHING and VITERBI look ahead (teacher / diagnostic use only).
    schedule_column, schedule_depth : optional column of X holding a block id (e.g. the trading day, as a number).
        When no ``schedule`` is passed, the model builds ``ExitSchedule.blocks(X[:, schedule_column], schedule_depth)``
        itself, so a pipeline that only passes a feature matrix still runs an embedded (clock-chunked) model correctly.
        Any injective transform of the id (e.g. a scaler) keeps the block boundaries. The column is not emitted.

    Duck typing. The methods of ``BasePomegranateHMM`` (``predict_proba``, ``predict``, ``predict_proba_fixed_lag``,
    ``compute_per_bar_predictive_loglik``, ``transition_prob``, ``forecast``, ``selection_scores``, ``save`` /
    ``load`` ...) work on the leaves: ``n_states`` is the number of leaves. The ``*_level`` methods give the same
    reads for the nodes at any depth (level 1 = the top-level states).

    Examples
    --------
    Two regimes, each a two-state sub-chain over bar features:

    >>> from okmich_quant_ml.hmm.hierarchical.engine import HHMM, StateNode, GaussianEmission
    >>> regime = lambda name: StateNode(name, [StateNode.leaf(f"{name}.a"), StateNode.leaf(f"{name}.b")])
    >>> model = HHMM(StateNode("root", [regime("bull"), regime("bear")]), [GaussianEmission([0, 1])])
    >>> model.fit(X)
    >>> p_regime = model.predict_proba_level(X, level=1)     # causal P(bull), P(bear) per bar
    >>> p_leaves = model.predict_proba(X)                    # BasePomegranateHMM-compatible: one column per leaf
    """

    def __init__(self, root: StateNode, emissions: Sequence[LeafEmission], *, n_init: int = 4, max_iter: int = 300,
                 tol: float = 1e-6, transition_prior: float = 1e-3,
                 sticky_init: Sequence[float] = (0.9, 0.95, 0.8, 0.98),
                 exit_init: Sequence[float] = (0.1, 0.05, 0.2, 0.3), random_state: int = 0,
                 inference_mode: InferenceMode = InferenceMode.FILTERING, schedule_column: Optional[int] = None,
                 schedule_depth: int = 1):
        self.topology = Topology(root)
        if self.topology.n_leaves > MAX_LEAVES:
            raise ValueError(f"{self.topology.n_leaves} leaves exceed {MAX_LEAVES}; exact inference would be slow")
        if not emissions:
            raise ValueError("At least one emission block is needed")
        if n_init < 0 or max_iter < 1 or tol < 0 or transition_prior < 0:
            raise ValueError("n_init >= 0, max_iter >= 1, tol >= 0 and transition_prior >= 0 are required")
        sticky_init, exit_init = tuple(float(s) for s in sticky_init), tuple(float(e) for e in exit_init)
        if not sticky_init or any(not 0 < s < 1 for s in sticky_init):
            raise ValueError(f"sticky_init values must lie in (0, 1), got {sticky_init}")
        if not exit_init or any(not 0 < e < 1 for e in exit_init):
            raise ValueError(f"exit_init values must lie in (0, 1), got {exit_init}")
        for e in emissions:
            if not isinstance(e, LeafEmission):
                raise TypeError(f"Emission blocks must be LeafEmission instances, got {type(e).__name__}")
        claimed: dict[int, int] = {}
        for k, e in enumerate(emissions):
            for col in e.columns:
                if col in claimed:
                    raise ValueError(f"Column {col} is read by emission blocks {claimed[col]} and {k}; each column may "
                                     f"belong to one block (a product of blocks over one column double-counts it)")
                claimed[col] = k
        if schedule_column is not None:
            schedule_column = int(schedule_column)
            if schedule_column < 0 or schedule_column in claimed:
                raise ValueError(f"schedule_column {schedule_column} must be a non-negative column no emission reads")
            if not 1 <= int(schedule_depth) <= max(self.topology.max_internal_depth, 1):
                raise ValueError(f"schedule_depth must be a depth whose nodes can end, got {schedule_depth}")
        self.schedule_column, self.schedule_depth = schedule_column, int(schedule_depth)
        # The model owns its blocks: fitting one model must never change another given the same block objects.
        self.emissions = [copy.deepcopy(e) for e in emissions]
        for e in self.emissions:
            e.bind(self.topology)
        self.n_init, self.max_iter, self.tol = int(n_init), int(max_iter), float(tol)
        self.transition_prior, self.sticky_init, self.exit_init = float(transition_prior), sticky_init, exit_init
        self.random_state, self.inference_mode = random_state, InferenceMode(inference_mode)
        top = self.topology
        self._onehots = [np.eye(e.n_groups)[e.groups] for e in self.emissions]          # (N, G) per block
        self._child_index = {(int(q), int(c)): int(top.children[q][c]) for q in top.internal
                             for c in range(len(top.children[q]))}
        self.entry_: Optional[dict[int, np.ndarray]] = None
        self.horizontal_: Optional[dict[int, np.ndarray]] = None
        self.exit_: Optional[dict[int, np.ndarray]] = None
        self.learned_depths_: Optional[frozenset[int]] = None
        self.fit_report: Optional[FitReport] = None

    # ==================================================================================================================
    # Fitting
    # ==================================================================================================================
    def fit(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, schedule: Optional[ExitSchedule] = None,
            warm_start: "HHMM | Sequence[HHMM | None] | None" = None) -> "HHMM":
        """EM from ``n_init`` fresh starts plus one start per ``warm_start`` model; keeps the best objective.

        Atomic: if anything fails, the model keeps its previous parameters, emissions and fit report.
        """
        backup = self._backup()
        try:
            return self._fit(X, lengths, schedule, warm_start)
        except BaseException:
            self._restore_backup(backup)
            raise

    def _backup(self) -> tuple:
        copy_dict = lambda d: None if d is None else {q: v.copy() for q, v in d.items()}
        return (copy_dict(self.entry_), copy_dict(self.horizontal_), copy_dict(self.exit_), self.learned_depths_,
                self.fit_report, copy.deepcopy(self.emissions))

    def _restore_backup(self, backup: tuple) -> None:
        (self.entry_, self.horizontal_, self.exit_, self.learned_depths_, self.fit_report, self.emissions) = backup

    def _fit(self, X: np.ndarray, lengths: Optional[Sequence[int]], schedule: Optional[ExitSchedule],
             warm_start) -> "HHMM":
        X = self._check_X(X)
        starts = self._starts(len(X), lengths)
        patterns, pattern_index, used = self._compile(self._schedule_for(X, schedule), len(X), starts)
        rng = np.random.default_rng(self.random_state)
        warm = [] if warm_start is None else [warm_start] if isinstance(warm_start, HHMM) else list(warm_start)
        warm = [w for w in warm if w is not None]
        if self.n_init + len(warm) == 0:
            raise ValueError("No start: set n_init >= 1 or give a warm_start")
        for w in warm:
            self._check_compatible(w)
        warm_snaps = [w._snapshot() for w in warm]          # taken first: a warm start may be this very model
        self.learned_depths_ = self._learned_depths(patterns, pattern_index, starts) if used else frozenset()
        best = None
        for restart in list(range(self.n_init)) + [("warm", i) for i in range(len(warm))]:
            if isinstance(restart, tuple):
                self._init_params(X, rng, 0)                 # data-derived floors, then the warm parameters
                self._restore(warm_snaps[restart[1]])
            else:
                self._init_params(X, rng, restart)
            report = self._em(X, starts, patterns, pattern_index, used, restart)
            if best is None or report.objective > best[0].objective:
                best = (report, self._snapshot())
        self.fit_report = best[0]
        self._restore(best[1])
        return self

    def _init_params(self, X: np.ndarray, rng: np.random.Generator, restart: int) -> None:
        top = self.topology
        stay = self.sticky_init[restart % len(self.sticky_init)]
        exit_p = self.exit_init[restart % len(self.exit_init)]
        self.entry_, self.horizontal_, self.exit_ = {}, {}, {}
        for q in top.internal:
            q = int(q)
            m = len(top.children[q])
            entry_mask = top.entry_mask(q)
            entry = np.zeros(m)
            entry[entry_mask] = rng.dirichlet(np.full(int(entry_mask.sum()), 5.0))
            self.entry_[q] = entry
            trans_mask = top.transition_mask(q)
            horizontal = np.zeros((m, m))
            for c in range(m):
                allowed = np.flatnonzero(trans_mask[c])
                if allowed.size == 0:
                    continue
                row = np.zeros(m)
                if trans_mask[c, c] and allowed.size > 1:
                    row[c] = stay
                    others = allowed[allowed != c]
                    row[others] = (1.0 - stay) * rng.dirichlet(np.full(others.size, 5.0))
                else:
                    row[allowed] = rng.dirichlet(np.full(allowed.size, 5.0))
                horizontal[c] = row
            self.horizontal_[q] = horizontal
            self.exit_[q] = np.where(top.forced_exit(q), 1.0, np.where(top.learnable_exit(q), exit_p, 0.0))
        for e in self.emissions:
            e.init(X, rng, restart)

    def _em(self, X: np.ndarray, starts: np.ndarray, patterns: np.ndarray, pattern_index: np.ndarray, used: bool,
            restart) -> FitReport:
        history: list[float] = []
        converged = False
        n_iter = 0
        loglik = -np.inf
        for n_iter in range(1, self.max_iter + 1):
            log_b = self._log_emissions(X)
            trans = self._flat_transitions(patterns, validate=used)
            gamma, xi, start_sum, loglik = forward_backward(log_b, _log(trans), pattern_index, _log(self._initial()),
                                                            starts)
            history.append(loglik + self._log_prior())
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol * len(X):
                converged = True
                break
            self._m_step(X, gamma, xi, start_sum, patterns, trans)
        else:
            loglik = self._loglik(X, starts, patterns, pattern_index, used)
            history.append(loglik + self._log_prior())
        tag = -1 - restart[1] if isinstance(restart, tuple) else restart
        return FitReport(history[-1], float(loglik), n_iter, converged, tag, history)

    def _m_step(self, X: np.ndarray, gamma: np.ndarray, xi: np.ndarray, start_sum: np.ndarray, patterns: np.ndarray,
                trans: np.ndarray) -> None:
        top = self.topology
        entry_c = {int(q): np.zeros(len(top.children[q])) for q in top.internal}
        horiz_c = {int(q): np.zeros((len(top.children[q]),) * 2) for q in top.internal}
        exit_c = {int(q): np.zeros(len(top.children[q])) for q in top.internal}
        stay_c = {int(q): np.zeros(len(top.children[q])) for q in top.internal}
        self._add_entries(0, start_sum, entry_c)
        for p, row in enumerate(patterns):
            for i in range(top.n_leaves):
                xi_row = xi[p, i]
                if not xi_row.any():
                    continue
                flat_row = trans[p, i]
                for path in self._paths(i, row):
                    child = self._child_index[(path.q, path.c2)]
                    contrib = path.weight * self._entry_vec[child]
                    r = np.where(flat_row > 0, xi_row * contrib / np.where(flat_row > 0, flat_row, 1.0), 0.0)
                    total = float(r.sum())
                    if total <= 0:
                        continue
                    horiz_c[path.q][path.c, path.c2] += total
                    if path.q != 0 and row[top.depth[path.q] - 1] == ExitMode.LEARNED:
                        stay_c[path.q][path.c] += total
                    for node, c in path.exits:
                        if row[top.depth[node] - 1] == ExitMode.LEARNED:
                            exit_c[node][c] += total
                    self._add_entries(child, r, entry_c)
        prior = self.transition_prior
        for q in top.internal:
            q = int(q)
            mask = top.entry_mask(q)
            counts = np.where(mask, entry_c[q] + prior, 0.0)
            if counts.sum() > 0:
                self.entry_[q] = counts / counts.sum()
            tmask = top.transition_mask(q)
            for c in range(len(top.children[q])):
                counts = np.where(tmask[c], horiz_c[q][c] + prior, 0.0)
                if counts.sum() > 0:
                    self.horizontal_[q][c] = counts / counts.sum()
            if q != 0 and top.depth[q] in self.learned_depths_:
                learn = top.learnable_exit(q)
                num, den = exit_c[q] + prior, exit_c[q] + stay_c[q] + 2.0 * prior
                ok = learn & (den > 0)
                self.exit_[q] = np.where(ok, num / np.where(den > 0, den, 1.0), self.exit_[q])
        for e, onehot in zip(self.emissions, self._onehots):
            e.m_step(X, gamma @ onehot)

    def _add_entries(self, node: int, leaf_weight: np.ndarray, entry_c: dict[int, np.ndarray]) -> None:
        """Credit the entry events of every internal node in ``node``'s subtree with the leaf weights below them."""
        top = self.topology
        if top.nodes[node].is_leaf:
            return
        stack = [node]
        while stack:
            n = stack.pop()
            for c, child in enumerate(top.children[n]):
                entry_c[n][c] += float(leaf_weight[top.descendants[child]].sum())
                if not top.nodes[child].is_leaf:
                    stack.append(int(child))

    def _log_prior(self) -> float:
        """Log priors implied by the pseudo-counts (the MAP objective's extra terms): Dirichlet / Beta
        (1 + transition_prior) on the free tree parameters, plus each emission block's own (e.g. the categorical
        pseudo-count)."""
        total = float(sum(e.log_prior() for e in self.emissions))
        prior = self.transition_prior
        if prior <= 0:
            return total
        top = self.topology
        for q in top.internal:
            q = int(q)
            mask = top.entry_mask(q)
            if mask.sum() > 1:
                total += prior * np.log(np.maximum(self.entry_[q][mask], 1e-300)).sum()
            tmask = top.transition_mask(q)
            for c in range(len(top.children[q])):
                if tmask[c].sum() > 1:
                    total += prior * np.log(np.maximum(self.horizontal_[q][c][tmask[c]], 1e-300)).sum()
            if q != 0 and top.depth[q] in self.learned_depths_:
                e = self.exit_[q][top.learnable_exit(q)]
                total += prior * (np.log(np.maximum(e, 1e-300)) + np.log(np.maximum(1.0 - e, 1e-300))).sum()
        return float(total)

    @staticmethod
    def _learned_depths(patterns: np.ndarray, pattern_index: np.ndarray, starts: np.ndarray) -> frozenset[int]:
        used = np.unique(pattern_index[~starts]) if (~starts).any() else np.array([], dtype=np.int64)
        depths = set()
        for p in used:
            depths |= {d + 1 for d, mode in enumerate(patterns[p]) if mode == ExitMode.LEARNED}
        return frozenset(depths)

    # ==================================================================================================================
    # Flattening
    # ==================================================================================================================
    def _entry_vectors(self) -> np.ndarray:
        """(n_nodes, N): leaf distribution reached by entering each node (one-hot for a leaf)."""
        top = self.topology
        vec = np.zeros((top.n_nodes, top.n_leaves))
        for node in reversed(range(top.n_nodes)):           # pre-order reversed: children before parents
            if top.nodes[node].is_leaf:
                vec[node, top.leaf_of_node[node]] = 1.0
            else:
                vec[node] = self.entry_[node] @ vec[top.children[node]]
        return vec

    def _effective_exit(self, q: int, c: int, row: np.ndarray) -> float:
        if q == 0:
            return 0.0
        mode = row[self.topology.depth[q] - 1]
        if mode == ExitMode.FORCED:
            return 1.0
        if mode == ExitMode.FORBIDDEN:
            return 0.0
        return float(self.exit_[q][c])

    def _paths(self, leaf: int, row: np.ndarray) -> Iterator[_Path]:
        """Every way ``leaf`` can move on under exit pattern ``row``, with its probability up to the entries."""
        top = self.topology
        anc = top.ancestors[leaf]
        up = 1.0
        exits: list[tuple[int, int]] = []
        for k in range(len(anc) - 2, -1, -1):
            q, c = int(anc[k]), int(top.local_index[anc[k + 1]])
            e = self._effective_exit(q, c, row)
            stay = up * (1.0 - e)
            if stay > 0:
                horizontal = self.horizontal_[q][c]
                for c2 in np.flatnonzero(horizontal > 0):
                    yield _Path(q, c, int(c2), stay * float(horizontal[c2]), tuple(exits))
            up *= e
            if up <= 0:
                return
            exits.append((q, c))

    def _flat_transitions(self, patterns: np.ndarray, validate: bool = True) -> np.ndarray:
        """(P, N, N) leaf-to-leaf transition matrix of every exit pattern (rows checked to sum to 1 if ``validate``)."""
        self._check_fitted()
        top = self.topology
        self._entry_vec = self._entry_vectors()
        trans = np.zeros((len(patterns), top.n_leaves, top.n_leaves))
        for p, row in enumerate(patterns):
            for i in range(top.n_leaves):
                for path in self._paths(i, row):
                    trans[p, i] += path.weight * self._entry_vec[self._child_index[(path.q, path.c2)]]
            sums = trans[p].sum(axis=1)
            bad = np.flatnonzero(np.abs(sums - 1.0) > 1e-6)
            if validate and bad.size:
                leaf = top.leaf_names[int(bad[0])]
                modes = {d + 1: ExitMode(int(m)).name for d, m in enumerate(row)}
                raise ValueError(f"Under exit modes {modes} leaf {leaf!r} has no way to continue (its row sums to "
                                 f"{sums[bad[0]]:.6f}): a child that can only exit meets a FORBIDDEN exit")
        return trans

    def _initial(self) -> np.ndarray:
        return self._entry_vectors()[0]

    # ==================================================================================================================
    # Inference: the BasePomegranateHMM-compatible surface. States = the leaves (production states).
    # ==================================================================================================================
    @property
    def n_states(self) -> int:
        """Number of states the BasePomegranateHMM-compatible methods report: the leaves."""
        return self.topology.n_leaves

    def transition_prob(self) -> np.ndarray:
        """(N, N) flat leaf transition matrix with every exit LEARNED (inside schedule blocks for embedded models)."""
        self._check_fitted()
        patterns, _ = resolve_schedule(None, 1, self.topology.max_internal_depth)
        return self._flat_transitions(patterns)[0]

    def flat_transitions(self, schedule: ExitSchedule) -> tuple[np.ndarray, np.ndarray]:
        """(P, N, N) flat matrices of every exit pattern in the schedule (each validated) and the (T,) pattern index
        of each bar. Inference itself builds only the patterns that bars after a sequence start use."""
        patterns, index = resolve_schedule(schedule, schedule.n_bars, self.topology.max_internal_depth)
        return self._flat_transitions(patterns), index

    def initial_distribution(self) -> np.ndarray:
        """(N,) leaf distribution at the first bar of a sequence (the root's entries pushed down)."""
        self._check_fitted()
        return self._initial()

    def filter(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
               schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, N) causal leaf posteriors P(leaf_t | x_1..t)."""
        alpha, _ = forward_filter(*self._inference_args(X, lengths, schedule))
        return alpha

    def smooth(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
               schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, N) smoothed leaf posteriors P(leaf_t | x_1..T): look-ahead, teacher and diagnostic use only."""
        gamma, _, _, _ = forward_backward(*self._inference_args(X, lengths, schedule))
        return gamma

    def state_posteriors(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                         schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, N) leaf posteriors under ``inference_mode`` (FILTERING or SMOOTHING)."""
        if self.inference_mode is InferenceMode.FILTERING:
            return self.filter(X, lengths, schedule)
        if self.inference_mode is InferenceMode.SMOOTHING:
            return self.smooth(X, lengths, schedule)
        raise ValueError(f"{self.inference_mode} gives a path, not posteriors; use predict() or FILTERING/SMOOTHING")

    def predict_proba(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                      schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, N) leaf posteriors under ``inference_mode`` (see :meth:`predict_proba_level` for a tree level)."""
        return self.state_posteriors(X, lengths, schedule)

    def predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T,) leaf labels: argmax of the posteriors, or the decoded path under VITERBI / CAUSAL_VITERBI."""
        if self.inference_mode is InferenceMode.CAUSAL_VITERBI:
            return self.predict_causal_viterbi(X, lengths, schedule)
        if self.inference_mode is InferenceMode.VITERBI:
            return self.predict_viterbi(X, lengths, schedule)
        return self.predict_proba(X, lengths, schedule).argmax(axis=1)

    def fit_predict(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                    schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        return self.fit(X, lengths, schedule).predict(X, lengths, schedule)

    def train(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, **kwargs) -> "HHMM":
        """Same as :meth:`fit` (the tree is fixed, so there is no state-count search)."""
        return self.fit(X, lengths, **kwargs)

    def predict_viterbi(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                        schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T,) most likely leaf path given the whole sequence: hindsight only."""
        return viterbi(*self._inference_args(X, lengths, schedule))

    def predict_causal_viterbi(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                               schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T,) leaf ending the most likely path given x_1..t at every t (causal; Tayal's online decoder)."""
        return causal_viterbi(*self._inference_args(X, lengths, schedule))

    def causal_viterbi_scores(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                              schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, N) max-product forward scores log delta: path log-probabilities, not a distribution."""
        return causal_viterbi_scores(*self._inference_args(X, lengths, schedule))

    def predict_proba_fixed_lag(self, X: np.ndarray, lag: int, lengths: Optional[Sequence[int]] = None,
                                schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, N) P(leaf_t | x_1..t+lag), open-ended (no assumption that the sequence stops); lag 0 = filter."""
        return fixed_lag(*self._inference_args(X, lengths, schedule), _check_lag(lag))

    def predict_fixed_lag(self, X: np.ndarray, lag: int, lengths: Optional[Sequence[int]] = None,
                          schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T,) MAP leaf labels from the fixed-lag posteriors."""
        return self.predict_proba_fixed_lag(X, lag, lengths, schedule).argmax(axis=1)

    def predict_proba_fixed_lag_sweep(self, X: np.ndarray, lags: Sequence[int], window_size: Optional[int] = None,
                                      lengths: Optional[Sequence[int]] = None,
                                      schedule: Optional[ExitSchedule] = None) -> dict[int, np.ndarray]:
        """{lag: (T, N)} fixed-lag posteriors for several lags.

        ``window_size=None``: full history (as :meth:`predict_proba_fixed_lag`). ``window_size=W``: bounded history,
        each posterior recomputed from the entry distribution on the last W bars up to min(t + lag, end), emulating a
        finite live buffer (as ``BasePomegranateHMM``); every lag must then be below W.
        """
        if not hasattr(lags, "__iter__"):
            raise ValueError(f"lags must be an iterable of integers, got {type(lags).__name__}")
        unique = sorted({_check_lag(lag) for lag in lags})
        if window_size is not None:
            if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)) or window_size < 1:
                raise ValueError(f"window_size must be an integer >= 1, got {window_size!r}")
            too_long = [lag for lag in unique if lag >= window_size]
            if too_long:
                raise ValueError(f"In windowed mode every lag must be below window_size={window_size}; got {too_long}")
        if not unique:
            return {}
        args = self._inference_args(X, lengths, schedule)
        if window_size is None:
            return {lag: fixed_lag(*args, lag) for lag in unique}
        return {lag: fixed_lag_windowed(*args, lag, int(window_size)) for lag in unique}

    def fixed_lag_diagnostics(self, X: np.ndarray, lag: int, lengths: Optional[Sequence[int]] = None,
                              schedule: Optional[ExitSchedule] = None) -> dict:
        """posteriors (T, N), map_labels, max_posterior, entropy and posterior_delta, as BasePomegranateHMM."""
        posteriors = self.predict_proba_fixed_lag(X, lag, lengths, schedule)
        log_p = np.log(np.maximum(posteriors, 1e-300))
        return {"posteriors": posteriors, "map_labels": posteriors.argmax(axis=1),
                "max_posterior": posteriors.max(axis=1), "entropy": -np.sum(posteriors * log_p, axis=1),
                "posterior_delta": np.diff(posteriors, axis=0, prepend=posteriors[:1])}

    def log_likelihood(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                       schedule: Optional[ExitSchedule] = None) -> float:
        _, log_pred = forward_filter(*self._inference_args(X, lengths, schedule))
        return float(log_pred.sum())

    def compute_per_bar_predictive_loglik(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                                          schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """log p(x_t | x_1..t-1) per bar (causal): out-of-sample comparison and drift monitoring."""
        _, log_pred = forward_filter(*self._inference_args(X, lengths, schedule))
        return log_pred

    def path_entropy(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                     schedule: Optional[ExitSchedule] = None) -> float:
        """Exact entropy of the posterior over leaf paths, H(z_1:T | x), in nats (the entropy term of ICL)."""
        return float(path_entropy(*self._inference_args(X, lengths, schedule)))

    def selection_scores(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, *, with_entropy: bool = True,
                         schedule: Optional[ExitSchedule] = None) -> SelectionScores:
        """Log-likelihood, free-parameter count and path entropy: AIC, BIC and ICL in one object."""
        X = self._check_X(X)
        entropy = self.path_entropy(X, lengths, schedule) if with_entropy else None
        return SelectionScores(log_likelihood=self.log_likelihood(X, lengths, schedule),
                               n_params=self.n_free_params(), n_obs=len(X), path_entropy=entropy)

    def get_aic_bic(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None,
                    schedule: Optional[ExitSchedule] = None) -> tuple[float, float]:
        scores = self.selection_scores(X, lengths, with_entropy=False, schedule=schedule)
        return scores.aic, scores.bic

    def forecast(self, posterior: Optional[np.ndarray] = None, n_steps: int = 1,
                 current_regime: Optional[int] = None) -> np.ndarray:
        """(N,) leaf probabilities ``n_steps`` ahead: ``posterior @ transition_prob() ** n_steps`` (exits LEARNED).

        ``posterior`` is a leaf belief, e.g. the last row of :meth:`predict_proba`. ``current_regime`` (a hard leaf
        label) is the deprecated entry point of ``BasePomegranateHMM.forecast``.
        """
        self._check_fitted()
        n = self.n_states
        if current_regime is not None:
            if posterior is not None:
                raise TypeError("forecast() accepts either `posterior` or `current_regime`, not both.")
            warnings.warn("forecast(current_regime=...) is deprecated; pass a one-hot `posterior` instead.",
                          DeprecationWarning, stacklevel=2)
            if not 0 <= int(current_regime) < n:
                raise ValueError(f"current_regime must be in [0, {n - 1}], got {current_regime}")
            belief = np.zeros(n)
            belief[int(current_regime)] = 1.0
        elif posterior is not None:
            belief = np.asarray(posterior, dtype=np.float64).reshape(-1)
            if belief.shape != (n,):
                raise ValueError(f"posterior must have shape ({n},), got {belief.shape}")
            if np.any(belief < 0) or not np.isclose(belief.sum(), 1.0, atol=1e-6):
                raise ValueError("posterior must be non-negative and sum to 1")
        else:
            raise TypeError("forecast() requires either `posterior` or `current_regime`.")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        return belief @ np.linalg.matrix_power(self.transition_prob(), int(n_steps))

    def get_stationary_distribution(self) -> np.ndarray:
        """(N,) stationary distribution of the leaf chain (exits LEARNED)."""
        self._check_fitted()
        values, vectors = np.linalg.eig(self.transition_prob().T)
        vec = np.abs(np.real(vectors[:, np.argmin(np.abs(values - 1.0))]))
        return vec / vec.sum()

    def expected_duration(self, regime: int) -> float:
        """Expected bars in leaf ``regime`` per visit: 1 / (1 - p_stay) (see :meth:`expected_level_duration`)."""
        self._check_fitted()
        if not 0 <= int(regime) < self.n_states:
            raise ValueError(f"regime must be between 0 and {self.n_states - 1}")
        stay = self.transition_prob()[int(regime), int(regime)]
        return np.inf if stay >= 1.0 else float(1.0 / (1.0 - stay))

    # ==================================================================================================================
    # Levels: the same reads for the nodes at any depth (level 1 = the top-level states)
    # ==================================================================================================================
    def level_names(self, level: int) -> list[str]:
        """Names of the nodes reported at ``level`` (the nodes at that depth plus any shallower leaf)."""
        return [self.topology.nodes[n].name for n in self.topology.level_nodes(level)]

    def level_marginal(self, leaf_posteriors: np.ndarray, level: int) -> np.ndarray:
        """(T, n_level_nodes) from (T, N) leaf posteriors."""
        member = self.topology.level_membership(level)
        return np.asarray(leaf_posteriors) @ member.T

    def leaf_to_level(self, leaves: np.ndarray, level: int) -> np.ndarray:
        """Map a leaf-index path to the index of each leaf's node at ``level``."""
        member = self.topology.level_membership(level)
        lookup = member.argmax(axis=0)                     # leaf -> level row
        return lookup[np.asarray(leaves, dtype=np.int64)]

    def predict_proba_level(self, X: np.ndarray, level: int = 1, lengths: Optional[Sequence[int]] = None,
                            schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, n_level_nodes) posterior of the nodes at ``level`` under ``inference_mode`` (``level_names(level)``)."""
        return self.level_marginal(self.state_posteriors(X, lengths, schedule), level)

    def predict_level(self, X: np.ndarray, level: int = 1, lengths: Optional[Sequence[int]] = None,
                      schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T,) index into ``level_names(level)``: argmax of the level posterior, or the decoded path's node under
        VITERBI / CAUSAL_VITERBI."""
        if self.inference_mode in (InferenceMode.CAUSAL_VITERBI, InferenceMode.VITERBI):
            return self.leaf_to_level(self.predict(X, lengths, schedule), level)
        return self.predict_proba_level(X, level, lengths, schedule).argmax(axis=1)

    def predict_proba_fixed_lag_level(self, X: np.ndarray, lag: int, level: int = 1,
                                      lengths: Optional[Sequence[int]] = None,
                                      schedule: Optional[ExitSchedule] = None) -> np.ndarray:
        """(T, n_level_nodes) fixed-lag posterior of the nodes at ``level``."""
        return self.level_marginal(self.predict_proba_fixed_lag(X, lag, lengths, schedule), level)

    def stationary_level_distribution(self, level: int = 1) -> np.ndarray:
        """(n_level_nodes,) stationary probability of each node at ``level``."""
        return self.level_marginal(self.get_stationary_distribution()[None, :], level)[0]

    def expected_level_duration(self, node: "int | str", level: int = 1) -> float:
        """Expected bars per contiguous visit to a node at ``level`` (index into ``level_names(level)`` or a name).

        The leaf chain restricted to the node's leaves is absorbing on leaving them; a visit starts from the
        stationary flow into the node, so the result is v (I - Q)^-1 1 with Q the within-node block of
        :meth:`transition_prob` and v the entrance distribution. For a leaf this is 1 / (1 - p_stay).
        """
        self._check_fitted()
        nodes = self.topology.level_nodes(level)
        if isinstance(node, str):
            names = self.level_names(level)
            if node not in names:
                raise KeyError(f"No node named {node!r} at level {level}; nodes there: {names}")
            node = names.index(node)
        if not 0 <= int(node) < len(nodes):
            raise ValueError(f"node must be in 0..{len(nodes) - 1}")
        inside = self.topology.descendants[nodes[int(node)]]
        outside = np.setdiff1d(np.arange(self.n_states), inside)
        T = self.transition_prob()
        if outside.size == 0:
            return np.inf
        entrance = self.get_stationary_distribution()[outside] @ T[np.ix_(outside, inside)]
        if entrance.sum() <= 0:
            return np.inf
        Q = T[np.ix_(inside, inside)]
        try:
            visits = np.linalg.solve(np.eye(len(inside)) - Q, np.ones(len(inside)))
        except np.linalg.LinAlgError:
            return np.inf
        return float(entrance @ visits / entrance.sum())

    # ==================================================================================================================
    # Parameters
    # ==================================================================================================================
    def n_free_params(self) -> int:
        """Free tree parameters (structural zeros, fixed cells and exits never learned excluded) plus emissions."""
        self._check_fitted()
        top = self.topology
        learned = self.learned_depths_ if self.learned_depths_ is not None else frozenset(range(1, top.max_depth + 1))
        n = 0
        for q in top.internal:
            q = int(q)
            n += int(top.entry_mask(q).sum()) - 1
            n += int(sum(max(int(r.sum()) - 1, 0) for r in top.transition_mask(q)))
            if q != 0 and top.depth[q] in learned:
                n += int(top.learnable_exit(q).sum())
        return n + sum(e.n_free_params() for e in self.emissions)

    def node_params(self, name: str) -> dict[str, np.ndarray]:
        """Copies of an internal node's ``entry`` (m,), ``horizontal`` (m, m) and ``exit`` (m,) parameters."""
        self._check_fitted()
        q = self._internal(name)
        return {"entry": self.entry_[q].copy(), "horizontal": self.horizontal_[q].copy(), "exit": self.exit_[q].copy()}

    def set_node_params(self, name: str, entry: Optional[np.ndarray] = None, horizontal: Optional[np.ndarray] = None,
                        exit: Optional[np.ndarray] = None) -> "HHMM":
        """Set an internal node's parameters (each optional). Structural zeros and fixed exits must be respected.

        Every supplied array is validated (shape, finite, probabilities, masks) before any is assigned, so a rejected
        update leaves the model unchanged.
        """
        self._check_fitted()
        top = self.topology
        q = self._internal(name)
        m = len(top.children[q])
        new_entry = new_h = new_e = None
        if entry is not None:
            new_entry = self._check_simplex(np.asarray(entry, dtype=np.float64), (m,), top.entry_mask(q),
                                            f"{name}.entry")
        if horizontal is not None:
            h = np.asarray(horizontal, dtype=np.float64)
            if h.shape != (m, m):
                raise ValueError(f"{name}.horizontal needs shape {(m, m)}, got {h.shape}")
            if not np.all(np.isfinite(h)):
                raise ValueError(f"{name}.horizontal must be finite")
            tmask = top.transition_mask(q)
            for c in range(m):
                if tmask[c].any():
                    self._check_simplex(h[c], (m,), tmask[c], f"{name}.horizontal[{c}]")
                elif np.any(h[c] != 0):
                    raise ValueError(f"{name}.horizontal[{c}] must be all zero (no move allowed)")
            new_h = h.copy()
        if exit is not None:
            e = np.asarray(exit, dtype=np.float64)
            if e.shape != (m,) or not np.all(np.isfinite(e)) or np.any(e < 0) or np.any(e > 1):
                raise ValueError(f"{name}.exit needs shape {(m,)} with finite values in [0, 1]")
            if q == 0 and np.any(e != 0):
                raise ValueError("The root never ends; its exit must be all zero")
            forced, allowed = top.forced_exit(q), top.exit_mask(q)
            if np.any(e[forced] != 1) or np.any(e[~allowed] != 0):
                raise ValueError(f"{name}.exit must be 1 where a child can only exit and 0 where exit is masked")
            new_e = e.copy()
        if new_entry is not None:
            self.entry_[q] = new_entry
        if new_h is not None:
            self.horizontal_[q] = new_h
        if new_e is not None:
            self.exit_[q] = new_e
        return self

    def get_params(self) -> dict:
        """All learnt parameters: tree parameters by node name, emission parameters by block position."""
        self._check_fitted()
        names = self.topology.nodes
        return {"entry": {names[q].name: v.copy() for q, v in self.entry_.items()},
                "horizontal": {names[q].name: v.copy() for q, v in self.horizontal_.items()},
                "exit": {names[q].name: v.copy() for q, v in self.exit_.items()},
                "emissions": [e.get_params() for e in self.emissions]}

    def sample(self, n_bars: int, seed: int = 0, lengths: Optional[Sequence[int]] = None,
               schedule: Optional[ExitSchedule] = None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate (leaf path (T,), X (T, n_columns)) from the current parameters; unused columns are NaN.

        A model with a ``schedule_column`` needs the ``schedule`` here, and its sample carries the block id in that
        column (the count of FORCED boundaries at ``schedule_depth`` so far), so the model can score its own sample.
        """
        self._check_fitted()
        if n_bars < 1:
            raise ValueError(f"n_bars must be >= 1, got {n_bars}")
        if self.schedule_column is not None and schedule is None:
            raise ValueError(f"This model reads its exit schedule from column {self.schedule_column}; pass the "
                             f"schedule to sample (e.g. ExitSchedule.blocks(block_ids))")
        rng = np.random.default_rng(seed)
        starts = self._starts(n_bars, lengths)
        patterns, index, used = self._compile(schedule, n_bars, starts)
        cum = np.cumsum(self._flat_transitions(patterns, validate=used), axis=2)
        init = np.cumsum(self._initial())
        n = self.topology.n_leaves
        leaves = np.empty(n_bars, dtype=np.int64)
        u = rng.random(n_bars)
        for t in range(n_bars):
            row = init if (starts[t] or t == 0) else cum[index[t], leaves[t - 1]]
            leaves[t] = min(int(np.searchsorted(row, u[t], side="right")), n - 1)
        width = max(max(e.columns) for e in self.emissions) + 1
        if self.schedule_column is not None:
            width = max(width, self.schedule_column + 1)
        X = np.full((n_bars, width), np.nan)
        for e in self.emissions:
            X[:, list(e.columns)] = e.sample(e.groups[leaves], rng)
        if self.schedule_column is not None:
            forced = np.zeros(n_bars, dtype=bool)
            for depth, modes in schedule.modes.items():
                if depth <= self.schedule_depth:
                    forced |= np.asarray(modes) == int(ExitMode.FORCED)
            forced[0] = False
            X[:, self.schedule_column] = np.cumsum(forced)
        return leaves, X

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HHMM":
        return joblib.load(path)

    #: attributes added after a model may have been saved, with the value an older model behaves as
    _STATE_DEFAULTS = {"schedule_column": None, "schedule_depth": 1, "inference_mode": InferenceMode.FILTERING,
                       "learned_depths_": None, "fit_report": None}

    def __setstate__(self, state: dict) -> None:
        """Unpickle, filling attributes that a model saved by an earlier version of this class lacks."""
        for key, value in self._STATE_DEFAULTS.items():
            state.setdefault(key, value)
        self.__dict__.update(state)

    def __repr__(self) -> str:
        top = self.topology
        return (f"HHMM(root={top.root.name!r}, leaves={top.n_leaves}, depth={top.max_depth}, "
                f"emissions={len(self.emissions)}, fitted={self.entry_ is not None})")

    # ==================================================================================================================
    # Internals
    # ==================================================================================================================
    def _internal(self, name: str) -> int:
        top = self.topology
        if name not in top.index:
            raise KeyError(f"No node named {name!r}")
        q = top.index[name]
        if top.nodes[q].is_leaf:
            raise ValueError(f"{name!r} is a leaf; leaves have emission parameters, not tree parameters")
        return q

    @staticmethod
    def _check_simplex(v: np.ndarray, shape: tuple[int, ...], mask: np.ndarray, label: str) -> np.ndarray:
        if v.shape != shape:
            raise ValueError(f"{label} needs shape {shape}, got {v.shape}")
        if not np.all(np.isfinite(v)) or np.any(v < 0) or abs(v.sum() - 1.0) > 1e-8:
            raise ValueError(f"{label} must be a probability vector")
        if np.any(v[~mask] != 0):
            raise ValueError(f"{label} puts mass on a masked (structurally zero) cell")
        return v.copy()

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (bars x columns), got shape {X.shape}")
        needed = max(max(e.columns) for e in self.emissions) + 1
        if self.schedule_column is not None:
            needed = max(needed, self.schedule_column + 1)
        if X.shape[1] < needed:
            raise ValueError(f"X has {X.shape[1]} columns; the emission blocks read up to column {needed - 1}")
        if len(X) == 0:
            raise ValueError("X has no rows")
        if np.isinf(X).any():
            raise ValueError("X contains infinite values (NaN is allowed: unobserved)")
        return X

    def _check_fitted(self) -> None:
        if self.entry_ is None:
            raise RuntimeError("The model has no parameters; call fit() first")

    def _log_emissions(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        log_b = np.zeros((len(X), self.topology.n_leaves))
        for e in self.emissions:
            log_b += e.log_density(X)[:, e.groups]
        return log_b

    def _compile(self, schedule: Optional[ExitSchedule], n_bars: int,
                 starts: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Exit patterns actually used by a transition (bars after a sequence start) and each bar's index into them.

        A sequence's first bar starts from the entry distribution, so its pattern is never used and must not be able
        to reject the sequence. Returns ``used=False`` (one placeholder pattern, never read) when no bar transitions.
        """
        patterns, index = resolve_schedule(schedule, n_bars, self.topology.max_internal_depth)
        moving = ~starts
        if not moving.any():
            return patterns[:1], np.zeros(n_bars, dtype=np.int64), False
        used = np.unique(index[moving])
        remap = np.zeros(len(patterns), dtype=np.int64)
        remap[used] = np.arange(len(used))
        compiled = remap[index]
        compiled[starts] = 0
        return patterns[used], compiled, True

    def _inference_args(self, X: np.ndarray, lengths: Optional[Sequence[int]], schedule: Optional[ExitSchedule]):
        """Kernel arguments, all in log space: log_b, log_trans (P, N, N), pattern index, log_pi, starts."""
        X = self._check_X(X)
        starts = self._starts(len(X), lengths)
        patterns, index, used = self._compile(self._schedule_for(X, schedule), len(X), starts)
        if used:
            self._warn_unlearned(patterns, index, starts)
        trans = self._flat_transitions(patterns, validate=used)
        return self._log_emissions(X), _log(trans), index, _log(self._initial()), starts

    def _schedule_for(self, X: np.ndarray, schedule: Optional[ExitSchedule]) -> Optional[ExitSchedule]:
        """The explicit schedule, else the one implied by the block-id column (if the model has one)."""
        if schedule is not None or self.schedule_column is None:
            return schedule
        ids = X[:, self.schedule_column]
        if not np.all(np.isfinite(ids)):
            raise ValueError(f"schedule_column {self.schedule_column} holds non-finite block ids")
        return ExitSchedule.blocks(ids, self.schedule_depth)

    def _warn_unlearned(self, patterns: np.ndarray, index: np.ndarray, starts: np.ndarray) -> None:
        if self.learned_depths_ is None:
            return
        used = self._learned_depths(patterns, index, starts)
        unseen = sorted(d for d in used if d not in self.learned_depths_ and self._has_learnable_exit(d))
        if unseen:
            warnings.warn(f"Exits at depths {unseen} are LEARNED here but were never learned in fit (the fit's "
                          f"schedule forbade or forced them): their exit probabilities are starting values",
                          stacklevel=3)

    def _has_learnable_exit(self, depth: int) -> bool:
        top = self.topology
        return any(top.depth[q] == depth and top.learnable_exit(int(q)).any() for q in top.internal if q != 0)

    def _loglik(self, X: np.ndarray, starts: np.ndarray, patterns: np.ndarray, index: np.ndarray,
                used: bool = True) -> float:
        _, log_pred = forward_filter(self._log_emissions(X), _log(self._flat_transitions(patterns, validate=used)),
                                     index, _log(self._initial()), starts)
        return float(log_pred.sum())

    @staticmethod
    def _starts(n_bars: int, lengths: Optional[Sequence[int]]) -> np.ndarray:
        starts = np.zeros(n_bars, dtype=np.bool_)
        if lengths is None:
            starts[0] = True
            return starts
        raw = np.asarray(list(lengths), dtype=np.float64)
        if raw.ndim != 1 or not np.all(np.isfinite(raw)) or np.any(raw != np.floor(raw)):
            raise ValueError(f"lengths must be whole numbers, got {list(lengths)}")
        lengths = [int(n) for n in raw]
        if any(n < 1 for n in lengths) or sum(lengths) != n_bars:
            raise ValueError(f"lengths must be positive and sum to the {n_bars} rows of X, got {lengths}")
        starts[np.cumsum([0] + lengths)[:-1]] = True
        return starts

    def _snapshot(self) -> tuple:
        return ({q: v.copy() for q, v in self.entry_.items()}, {q: v.copy() for q, v in self.horizontal_.items()},
                {q: v.copy() for q, v in self.exit_.items()}, [e.get_params() for e in self.emissions])

    def _restore(self, snap: tuple) -> None:
        self.entry_ = {q: v.copy() for q, v in snap[0].items()}
        self.horizontal_ = {q: v.copy() for q, v in snap[1].items()}
        self.exit_ = {q: v.copy() for q, v in snap[2].items()}
        for e, params in zip(self.emissions, snap[3]):
            e.set_params(params)

    def _check_compatible(self, other: "HHMM") -> None:
        """A warm start must have this model's exact tree (names, children, masks) and emission configuration."""
        if not isinstance(other, HHMM):
            raise TypeError(f"warm_start must be an HHMM, got {type(other).__name__}")
        other._check_fitted()
        mine, theirs = self.topology, other.topology
        if [n.name for n in mine.nodes] != [n.name for n in theirs.nodes]:
            raise ValueError("warm_start must have the same state tree (same node names in the same order)")
        for a, b in zip(mine.nodes, theirs.nodes):
            same = (a.is_leaf == b.is_leaf and np.array_equal(a._entry, b._entry)
                    and np.array_equal(a._transition, b._transition) and np.array_equal(a._exit, b._exit))
            if not same:
                raise ValueError(f"warm_start node {a.name!r} has different children or masks")
        if [e.config_signature() for e in self.emissions] != [e.config_signature() for e in other.emissions]:
            raise ValueError("warm_start must have the same emission blocks (kind, columns, tying, alphabet and "
                             "support, covariance type), in the same order")


def _log(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(p)


def _check_lag(lag) -> int:
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag!r}")
    return int(lag)
