"""Test trees, random valid parameters, and independent oracles for the HHMM engine.

``semantic_next_leaf`` walks the tree recursively from the HHMM definition (Fine, Singer & Tishby 1998): after a leaf
emits, its parent either ends (exit) and hands control up, or moves horizontally and enters a child, recursively. It
does not use the engine's flattening code, so comparing the two checks the flattening.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from okmich_quant_ml.hmm.hierarchical.engine import (CategoricalEmission, ExitMode, GaussianEmission, HHMM,
                                                     StateNode)


def two_level() -> StateNode:
    """root -> A{a1, a2}, B{b1, b2}; no masks."""
    return StateNode("root", [StateNode("A", [StateNode.leaf("a1"), StateNode.leaf("a2")]),
                              StateNode("B", [StateNode.leaf("b1"), StateNode.leaf("b2")])])


def three_level() -> StateNode:
    """Unbalanced: root -> X{Y{y1, y2}, x1}, z1 (a leaf at depth 1), with masks."""
    y = StateNode("Y", [StateNode.leaf("y1"), StateNode.leaf("y2")], exit_mask=[False, True])
    x = StateNode("X", [y, StateNode.leaf("x1")], entry_mask=[True, False],
                  transition_mask=[[True, True], [True, False]])
    return StateNode("root", [x, StateNode.leaf("z1")])


def tayal_tree() -> StateNode:
    """Tayal (2009) Fig. 4.6: R1 enters and ends on its down leg, R2 on its up leg; the top level always switches."""
    r1 = StateNode("R1", [StateNode.leaf("R1.D"), StateNode.leaf("R1.U")], entry_mask=[True, False],
                   transition_mask=[[False, True], [True, False]], exit_mask=[True, False])
    r2 = StateNode("R2", [StateNode.leaf("R2.U"), StateNode.leaf("R2.D")], entry_mask=[True, False],
                   transition_mask=[[False, True], [True, False]], exit_mask=[True, False])
    return StateNode("root", [r1, r2], transition_mask=[[False, True], [True, False]])


def flat_tree(n: int) -> StateNode:
    return StateNode("root", [StateNode.leaf(f"s{k}") for k in range(n)])


def set_random_params(model: HHMM, rng: np.random.Generator, X: np.ndarray) -> HHMM:
    """Initialise the model, then draw random tree parameters that respect every mask."""
    model._init_params(X, rng, 0)
    top = model.topology
    for q in top.internal:
        q = int(q)
        name = top.nodes[q].name
        m = len(top.children[q])
        entry = np.zeros(m)
        mask = top.entry_mask(q)
        entry[mask] = rng.dirichlet(np.ones(mask.sum()))
        horizontal = np.zeros((m, m))
        tmask = top.transition_mask(q)
        for c in range(m):
            if tmask[c].any():
                horizontal[c, tmask[c]] = rng.dirichlet(np.ones(tmask[c].sum()))
        exit_ = np.where(top.forced_exit(q), 1.0, np.where(top.learnable_exit(q), rng.uniform(0.05, 0.6, m), 0.0))
        model.set_node_params(name, entry=entry, horizontal=horizontal, exit=None if q == 0 else exit_)
    return model


def gaussian_model(root: StateNode, **kw) -> HHMM:
    return HHMM(root, [GaussianEmission([0])], **kw)


def set_leaf_means(model: HHMM, means: np.ndarray, var: float = 0.3) -> None:
    n = model.topology.n_leaves
    model.emissions[0].set_params({"means": np.asarray(means, dtype=float).reshape(n, 1),
                                   "covs": np.full((n, 1), var)})


# ---------------------------------------------------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------------------------------------------------
def _mode(modes: dict[int, ExitMode], depth: int) -> ExitMode:
    """Mode of a node at ``depth``; FORCED at any shallower depth >= 1 forces it too."""
    if any(modes.get(d, ExitMode.LEARNED) is ExitMode.FORCED for d in range(1, depth + 1)):
        return ExitMode.FORCED
    return modes.get(depth, ExitMode.LEARNED)


def semantic_next_leaf(model: HHMM, leaf: int, modes: dict[int, ExitMode]) -> np.ndarray:
    """P(next leaf | current leaf) by walking the HHMM's generative semantics recursively."""
    top = model.topology
    params = {int(q): model.node_params(top.nodes[q].name) for q in top.internal}
    out = np.zeros(top.n_leaves)

    def enter(node: int, w: float) -> None:
        if top.nodes[node].is_leaf:
            out[top.leaf_of_node[node]] += w
            return
        for c, child in enumerate(top.children[node]):
            p = params[node]["entry"][c]
            if p > 0:
                enter(int(child), w * p)

    def finished(node: int, w: float) -> None:
        parent = top.parent[node]
        c = top.local_index[node]
        pr = params[parent]
        if parent == 0:
            e = 0.0
        else:
            mode = _mode(modes, top.depth[parent])
            e = 1.0 if mode is ExitMode.FORCED else 0.0 if mode is ExitMode.FORBIDDEN else pr["exit"][c]
        for c2, a in enumerate(pr["horizontal"][c]):
            if a > 0 and (1 - e) > 0:
                enter(int(top.children[parent][c2]), w * (1 - e) * a)
        if e > 0:
            finished(parent, w * e)

    finished(int(top.leaf_nodes[leaf]), 1.0)
    return out


def semantic_initial(model: HHMM) -> np.ndarray:
    top = model.topology
    out = np.zeros(top.n_leaves)

    def enter(node: int, w: float) -> None:
        if top.nodes[node].is_leaf:
            out[top.leaf_of_node[node]] += w
            return
        entry = model.node_params(top.nodes[node].name)["entry"]
        for c, child in enumerate(top.children[node]):
            if entry[c] > 0:
                enter(int(child), w * entry[c])

    enter(0, 1.0)
    return out


def brute_force(model: HHMM, X: np.ndarray, modes_per_bar: list[dict[int, ExitMode]] | None = None):
    """log p(x), filtered, smoothed posteriors and the MAP path by enumerating every leaf path."""
    n = model.topology.n_leaves
    T = len(X)
    modes_per_bar = modes_per_bar or [{}] * T
    log_b = model._log_emissions(np.asarray(X, dtype=float).reshape(T, -1))
    with np.errstate(divide="ignore"):
        log_pi = np.log(semantic_initial(model))
        log_t = [np.log(np.array([semantic_next_leaf(model, i, modes_per_bar[t]) for i in range(n)]))
                 for t in range(T)]
    paths = np.array(list(itertools.product(range(n), repeat=T)))
    score = log_pi[paths[:, 0]] + log_b[0, paths[:, 0]]
    prefix = [score.copy()]
    for t in range(1, T):
        score = score + log_t[t][paths[:, t - 1], paths[:, t]] + log_b[t, paths[:, t]]
        prefix.append(score.copy())
    total = logsumexp(score)
    smoothed = np.array([[np.exp(logsumexp(score[paths[:, t] == k]) - total) for k in range(n)] for t in range(T)])
    filtered = np.array([[np.exp(logsumexp(prefix[t][paths[:, t] == k]) - logsumexp(prefix[t])) for k in range(n)]
                         for t in range(T)])
    map_path = paths[int(np.argmax(score))]
    causal_ends = np.array([paths[int(np.argmax(prefix[t])), t] for t in range(T)])
    return total, filtered, smoothed, map_path, causal_ends


def categorical_pinned(root: StateNode, symbols_per_leaf: int = 2, **kw) -> HHMM:
    """Categorical emission where each leaf owns disjoint symbols: the leaf path is observed exactly."""
    probe = HHMM(root, [GaussianEmission([0])])
    names = probe.topology.leaf_names
    support = {name: range(k * symbols_per_leaf, (k + 1) * symbols_per_leaf) for k, name in enumerate(names)}
    return HHMM(root, [CategoricalEmission(0, len(names) * symbols_per_leaf, support=support, pseudo_count=0.0)], **kw)
