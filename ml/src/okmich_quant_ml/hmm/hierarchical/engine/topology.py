"""State tree of a hierarchical HMM (HHMM) and its compiled layout.

An HHMM (Fine, Singer & Tishby 1998) is a tree of states. Leaves are *production* states: they emit observations.
Every internal node q runs its own Markov chain over its children with three kinds of parameter:

- ``entry``      pi_q(c):       when q is activated, the probability that it starts in child c (vertical transition);
- ``horizontal`` A_q(c, c'):    given that q does not end after child c, the probability that it moves to child c';
- ``exit``       e_q(c):        the probability that q ends (returns control to its parent) after child c.

So after child c the parent either ends (probability e_q(c)) or moves horizontally ((1 - e_q(c)) A_q(c, c')). The root
never ends. ``entry``/``horizontal``/``exit`` masks put structural zeros anywhere: e.g. a regime that may only be
entered and left through one particular child (Tayal 2009), or a left-to-right chain.

Inference flattens the tree onto its leaves (Murphy & Paskin 2001): the probability of moving from leaf i to leaf j is a
sum over the levels at which the move can happen, so inference is exact at O(T N^2) for N leaves. EM still learns the
tree's parameters, not free flat cells (see ``model.HHMM``).

Depths: the root has depth 0, its children depth 1, and so on. Leaves may sit at different depths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(eq=False)
class StateNode:
    """One node of the state tree. A node without children is a production (emitting) leaf.

    Parameters
    ----------
    name : unique name of the node; parameters and level posteriors are keyed by it.
    children : the node's children (empty for a leaf).
    entry_mask : (n_children,) bool, which children the node may start in. Default: all.
    transition_mask : (n_children, n_children) bool, allowed horizontal moves child -> child. Default: all, including
        staying in the same child.
    exit_mask : (n_children,) bool, after which children the node may end. Default: all. Ignored for the root, which
        never ends. A child with no allowed horizontal move must be allowed to exit (it then always ends the node).
    """

    name: str
    children: Sequence["StateNode"] = ()
    entry_mask: Optional[Sequence[bool]] = None
    transition_mask: Optional[Sequence[Sequence[bool]]] = None
    exit_mask: Optional[Sequence[bool]] = None
    _entry: np.ndarray = field(init=False, repr=False)
    _transition: np.ndarray = field(init=False, repr=False)
    _exit: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"A StateNode needs a non-empty string name, got {self.name!r}")
        self.children = tuple(self.children)
        if any(not isinstance(c, StateNode) for c in self.children):
            raise TypeError(f"Children of {self.name!r} must be StateNode instances")
        m = len(self.children)
        if m == 0:
            if self.entry_mask is not None or self.transition_mask is not None or self.exit_mask is not None:
                raise ValueError(f"Leaf {self.name!r} has no children, so it cannot carry entry/transition/exit masks")
            self._entry = np.zeros(0, dtype=bool)
            self._transition = np.zeros((0, 0), dtype=bool)
            self._exit = np.zeros(0, dtype=bool)
            return
        self._entry = self._mask(self.entry_mask, (m,), "entry_mask")
        self._transition = self._mask(self.transition_mask, (m, m), "transition_mask")
        self._exit = self._mask(self.exit_mask, (m,), "exit_mask")
        if not self._entry.any():
            raise ValueError(f"Node {self.name!r}: entry_mask allows no child to start")

    def _mask(self, value, shape: tuple[int, ...], label: str) -> np.ndarray:
        if value is None:
            return np.ones(shape, dtype=bool)
        arr = np.asarray(value, dtype=bool)
        if arr.shape != shape:
            raise ValueError(f"Node {self.name!r}: {label} needs shape {shape}, got {arr.shape}")
        return arr.copy()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @classmethod
    def leaf(cls, name: str) -> "StateNode":
        """A production (emitting) state."""
        return cls(name)

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"StateNode({self.name!r})"
        return f"StateNode({self.name!r}, children={[c.name for c in self.children]})"


class Topology:
    """Compiled layout of a state tree: node and leaf indexing, ancestry, masks and level membership.

    Nodes are numbered in depth-first pre-order (the root is node 0); leaves are numbered 0..N-1 in the same order.
    """

    def __init__(self, root: StateNode):
        if not isinstance(root, StateNode):
            raise TypeError("root must be a StateNode")
        if root.is_leaf:
            raise ValueError("The root must have at least one child")
        self.root = root
        self.nodes: list[StateNode] = []
        self.parent: list[int] = []
        self.depth: list[int] = []
        self.local_index: list[int] = []          # position of the node among its parent's children (-1 for root)
        self._walk(root, -1, 0, -1)
        names = [n.name for n in self.nodes]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"Node names must be unique; repeated: {dupes}")
        self.index = {name: i for i, name in enumerate(names)}
        self.children: list[np.ndarray] = [np.array([self.index[c.name] for c in n.children], dtype=np.int64)
                                           for n in self.nodes]
        self.internal = np.array([i for i, n in enumerate(self.nodes) if not n.is_leaf], dtype=np.int64)
        self.leaf_nodes = np.array([i for i, n in enumerate(self.nodes) if n.is_leaf], dtype=np.int64)
        self.n_leaves = len(self.leaf_nodes)
        self.leaf_of_node = {int(node): k for k, node in enumerate(self.leaf_nodes)}
        self.ancestors: list[np.ndarray] = [self._ancestry(int(node)) for node in self.leaf_nodes]
        self.leaf_depth = np.array([len(a) - 1 for a in self.ancestors], dtype=np.int64)
        self.max_depth = int(self.leaf_depth.max())
        # depths whose nodes can end: internal nodes below the root
        internal_depths = [self.depth[i] for i in self.internal if i != 0]
        self.max_internal_depth = max(internal_depths) if internal_depths else 0
        self.descendants = [self._descendant_leaves(i) for i in range(len(self.nodes))]
        self._validate_masks()

    # ------------------------------------------------------------------------------------------------------------
    def _walk(self, node: StateNode, parent: int, depth: int, local: int) -> None:
        self.nodes.append(node)
        self.parent.append(parent)
        self.depth.append(depth)
        self.local_index.append(local)
        me = len(self.nodes) - 1
        for k, child in enumerate(node.children):
            self._walk(child, me, depth + 1, k)

    def _ancestry(self, node: int) -> np.ndarray:
        chain = [node]
        while self.parent[chain[-1]] >= 0:
            chain.append(self.parent[chain[-1]])
        return np.array(chain[::-1], dtype=np.int64)

    def _descendant_leaves(self, node: int) -> np.ndarray:
        if self.nodes[node].is_leaf:
            return np.array([self.leaf_of_node[node]], dtype=np.int64)
        return np.concatenate([self._descendant_leaves(int(c)) for c in self.children[node]])

    def _validate_masks(self) -> None:
        for q in self.internal:
            node = self.nodes[q]
            for c in range(len(node.children)):
                moves = node._transition[c].any()
                exits = bool(node._exit[c]) and q != 0
                if not moves and not exits:
                    where = "the root never ends" if q == 0 else "exit_mask forbids ending after it"
                    raise ValueError(f"Node {node.name!r}: child {node.children[c].name!r} has no allowed horizontal "
                                     f"move and {where}, so it is a dead end")

    # ------------------------------------------------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def leaf_names(self) -> list[str]:
        return [self.nodes[i].name for i in self.leaf_nodes]

    def entry_mask(self, q: int) -> np.ndarray:
        return self.nodes[q]._entry

    def transition_mask(self, q: int) -> np.ndarray:
        return self.nodes[q]._transition

    def exit_mask(self, q: int) -> np.ndarray:
        """Exit mask of internal node q; all False for the root."""
        if q == 0:
            return np.zeros(len(self.children[0]), dtype=bool)
        return self.nodes[q]._exit

    def forced_exit(self, q: int) -> np.ndarray:
        """Children of q after which q must end (exit allowed but no horizontal move): exit probability fixed at 1."""
        return self.exit_mask(q) & ~self.transition_mask(q).any(axis=1)

    def learnable_exit(self, q: int) -> np.ndarray:
        """Children of q whose exit probability is a free parameter."""
        return self.exit_mask(q) & self.transition_mask(q).any(axis=1)

    def ancestor_at(self, leaf: int, depth: int) -> int:
        """Node index of leaf's ancestor at ``depth`` (the leaf itself when the leaf is shallower)."""
        anc = self.ancestors[leaf]
        return int(anc[min(depth, len(anc) - 1)])

    def level_nodes(self, level: int) -> list[int]:
        """Nodes reported at ``level``: the nodes at that depth plus any leaf shallower than it (DFS order)."""
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        out: list[int] = []
        for leaf in range(self.n_leaves):
            node = self.ancestor_at(leaf, level)
            if node not in out:
                out.append(node)
        return out

    def level_membership(self, level: int) -> np.ndarray:
        """(n_level_nodes, N) 0/1 matrix: leaf j counts towards level node r."""
        nodes = self.level_nodes(level)
        member = np.zeros((len(nodes), self.n_leaves))
        for r, node in enumerate(nodes):
            member[r, self.descendants[node]] = 1.0
        return member

    def group_of_leaves(self, tie_level: Optional[int]) -> np.ndarray:
        """(N,) group index per leaf: leaves sharing their ancestor at ``tie_level`` share a group (None = per leaf)."""
        if tie_level is None:
            return np.arange(self.n_leaves, dtype=np.int64)
        nodes = self.level_nodes(tie_level)
        lookup = {node: g for g, node in enumerate(nodes)}
        return np.array([lookup[self.ancestor_at(j, tie_level)] for j in range(self.n_leaves)], dtype=np.int64)

    def __repr__(self) -> str:
        return (f"Topology(root={self.root.name!r}, nodes={self.n_nodes}, leaves={self.n_leaves}, "
                f"depth={self.max_depth})")
