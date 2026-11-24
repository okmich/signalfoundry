"""The legacy SF build (2026-07 ``HierarchicalHMM``, removed 2026-11-24) expressed on the general engine.

What it was: a *flat* 4-state HMM over 18 zigzag symbols whose transition mask forces up and down legs to alternate,
with the states grouped afterwards into two blocks (A, B) labelled Run / Reversal by mean leg size. Its 8 allowed edges
let the next block's first leg depend on which leg the last block ended on, which an HHMM entry distribution cannot
express; so the faithful encoding is a depth-1 tree (four leaves under the root) plus the block grouping.

There is deliberately no loader for files the old class saved: its learnt end probabilities (a sub-stochastic
chain), session gates and Run / Reversal labelling have no counterpart here, so a conversion could only approximate it
(the filters can differ materially). Refit ``legacy_model`` on the data instead.
"""
from __future__ import annotations

import numpy as np

from ..engine import CategoricalEmission, HHMM, StateNode

ALPHABET_SIZE = 18                                        # direction (2) x leg-size bucket (3) x flow bucket (3)
SUB_ALPHABET_SIZE = 9                                     # symbols per direction: down 0..8, up 9..17
LEGACY_LEAVES = ("A.up", "A.down", "B.up", "B.down")       # the legacy flat order: A P+, A P-, B P+, B P-
LEGACY_BLOCKS = {"A": (0, 1), "B": (2, 3)}


def legacy_tree() -> StateNode:
    """Depth-1 tree: four leaves, every transition flips the leg direction."""
    mask = [[(i % 2) != (j % 2) for j in range(4)] for i in range(4)]
    return StateNode("root", [StateNode.leaf(name) for name in LEGACY_LEAVES], transition_mask=mask)


def legacy_model(*, pseudo_count: float = 1e-3, **hhmm_kwargs) -> HHMM:
    """Engine model of the legacy build over the 18-symbol alphabet (up leaves emit symbols 9..17, down 0..8).

    ``pseudo_count`` keeps every allowed symbol possible, so a symbol absent from training does not make a later
    sequence impossible (0 = maximum likelihood).
    """
    up, down = range(SUB_ALPHABET_SIZE, ALPHABET_SIZE), range(0, SUB_ALPHABET_SIZE)
    support = {"A.up": up, "B.up": up, "A.down": down, "B.down": down}
    return HHMM(legacy_tree(), [CategoricalEmission(0, ALPHABET_SIZE, support=support, pseudo_count=pseudo_count)],
                **hhmm_kwargs)


def legacy_block_posterior(leaf_posteriors: np.ndarray) -> np.ndarray:
    """(T, 2) block posteriors [A, B] from the (T, 4) legacy leaf posteriors."""
    leaf_posteriors = np.asarray(leaf_posteriors)
    return np.column_stack([leaf_posteriors[:, list(LEGACY_BLOCKS["A"])].sum(axis=1),
                            leaf_posteriors[:, list(LEGACY_BLOCKS["B"])].sum(axis=1)])
