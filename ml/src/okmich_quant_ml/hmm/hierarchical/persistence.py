"""
Parameter-dict serialization for the Hierarchical HMM.

Serialises a fitted (categorical) :class:`HierarchicalHMM` to an explicit, versioned dict — not
pickle — following the plan's schema (``pi_root``, ``A_macro``, ``A_production_run/rev``,
``B_run_pos`` ...). Two design points:

- The plan schema fields are computed faithfully from the fitted flat model for interpretability and
  spec compliance. ``A_macro`` (2x2) is a lossy block-aggregation; ``A_production_*`` recover the
  {P+, P-, T} view where ``T`` is the termination (macro-switch) mass folded into the flat topology.
- A ``_reconstruction`` block carries the raw flat arrays (``starts``, ``edges``, emissions and the
  Run block index), so :func:`from_param_dict` rebuilds a ready-to-infer model without re-running
  EM. Reload reproduces the fitted posteriors to backend float tolerance (~1e-6).

Only the CATEGORICAL model is round-trippable here (the schema's ``B_*`` are multinomials). Fitted
continuous/mixture models should be persisted with the inherited joblib ``save``/``load``.
"""
from __future__ import annotations

import json
from datetime import time
from typing import Any, Optional, Sequence

import numpy as np

from ..util import DistType
from .config import (
    N_MACRO_STATES,
    SUB_ALPHABET_SIZE,
    MacroRegime,
    SessionPolicy,
    TOPOLOGY_NAME,
    ZigzagDirection,
    symbols_for_direction,
)
from .hhmm import HierarchicalHMM, _BLOCK_STATES, state_direction

HHMM_SCHEMA_VERSION: int = 1

# Production-state ordering within the {P+, P-, T} schema view.
_PROD_PLUS, _PROD_MINUS, _PROD_TERM = 0, 1, 2


def _state_in_block(block: int, direction: ZigzagDirection) -> int:
    """Flat state index of the given direction within a macro block."""
    for s in _BLOCK_STATES[block]:
        if state_direction(s) is direction:
            return s
    raise KeyError((block, direction))


def _sub_emission(emissions: np.ndarray, state: int) -> list[float]:
    """The 9-vector sub-alphabet emission for a categorical state (structural zeros dropped)."""
    direction = state_direction(state)
    sub = list(symbols_for_direction(direction))
    vec = emissions[state][sub]
    total = vec.sum()
    return (vec / total if total > 0 else vec).tolist()


def _macro_transition(edges: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """
    2x2 macro (block-to-block) transition, averaged over the two production rows of each block,
    returned in the given block ``order``.

    Pass ``order=[run_block, rev_block]`` so rows/cols are Run/Reversal — matching ``pi_root`` and the
    ``B_*`` emissions. Physical block 0 is not necessarily Run, so ordering by physical index would
    silently mislabel the transition semantics that a downstream Viterbi consumes.
    """
    A = np.zeros((N_MACRO_STATES, N_MACRO_STATES), dtype=np.float64)
    for oi, i in enumerate(order):
        rows = [edges[s] for s in _BLOCK_STATES[i]]
        for oj, j in enumerate(order):
            A[oi, oj] = float(np.mean([r[list(_BLOCK_STATES[j])].sum() for r in rows]))
    return A


def _production_transition(edges: np.ndarray, block: int) -> np.ndarray:
    """
    {P+, P-, T} 3x3 view for one macro block.

    Within-block direction flips populate the P+/P- alternation; the cross-block mass is the
    termination probability P+/P- -> T. T is treated as absorbing (control returns to root).
    """
    A = np.zeros((3, 3), dtype=np.float64)
    plus = _state_in_block(block, ZigzagDirection.UP)
    minus = _state_in_block(block, ZigzagDirection.DOWN)
    other = 1 - block
    cross = list(_BLOCK_STATES[other])
    for src, row_idx in ((plus, _PROD_PLUS), (minus, _PROD_MINUS)):
        row = edges[src]
        # within-block flip goes to the opposite direction production state
        opp = minus if src == plus else plus
        A[row_idx, _PROD_PLUS if opp == plus else _PROD_MINUS] = float(row[opp])
        A[row_idx, _PROD_TERM] = float(row[cross].sum())
    A[_PROD_TERM, _PROD_TERM] = 1.0
    return A


def _serialize_session_gate(model: HierarchicalHMM) -> Optional[dict]:
    """Capture an attached session-policy gate so reload preserves SOFT/HARD inference behaviour."""
    policy = getattr(model, "session_policy", None)
    if policy is None:
        return None
    windows = getattr(model, "low_liquidity_windows", ())
    return {
        "policy": SessionPolicy(policy).value,
        "low_liquidity_windows": [[w[0].strftime("%H:%M"), w[1].strftime("%H:%M")] for w in windows],
        "soft_downweight": float(getattr(model, "soft_downweight", 1.0)),
    }


def _restore_session_gate(model: HierarchicalHMM, gate: dict) -> None:
    """Re-attach a serialized session-policy gate to a freshly reloaded model."""
    def _parse(hhmm: str) -> time:
        hh, mm = hhmm.split(":")
        return time(int(hh), int(mm))
    windows = tuple((_parse(a), _parse(b)) for a, b in gate.get("low_liquidity_windows", []))
    model.attach_session_policy(SessionPolicy(gate["policy"]), low_liquidity_windows=windows,
                                soft_downweight=float(gate.get("soft_downweight", 1.0)))


def to_param_dict(model: HierarchicalHMM, *, asset_group: Optional[str] = None, k: Optional[float] = None,
                  flow_feature: Optional[str] = None, session_policy: Optional[str] = None,
                  fit_metadata: Optional[dict] = None) -> dict:
    """Serialise a fitted categorical HHMM to the versioned parameter dict."""
    if model.distribution_type is not DistType.CATEGORICAL:
        raise NotImplementedError(
            "to_param_dict supports the categorical HHMM only; use the joblib save()/load() for "
            "continuous or mixture emissions."
        )
    flat = model.flat_params()
    # Human-facing schema uses the renormalised snapshot; reconstruction uses the raw internal
    # arrays so reload reproduces the fitted posteriors to backend float tolerance.
    starts = np.asarray(flat["starts"], dtype=np.float64)
    edges = np.asarray(flat["edges"], dtype=np.float64)
    emissions = np.asarray(flat["emissions"], dtype=np.float64)
    raw_starts, raw_edges, raw_ends = model._raw_starts_edges()

    run_block = model.macro_block(MacroRegime.RUN)
    rev_block = model.macro_block(MacroRegime.REVERSAL)
    session_gate = _serialize_session_gate(model)
    # An attached gate is authoritative for the top-level session_policy field when not overridden.
    if session_policy is None and session_gate is not None:
        session_policy = session_gate["policy"]

    def sub(block: int, direction: ZigzagDirection) -> list[float]:
        return _sub_emission(emissions, _state_in_block(block, direction))

    macro_start = [float(starts[list(_BLOCK_STATES[run_block])].sum()),
                   float(starts[list(_BLOCK_STATES[rev_block])].sum())]

    params = {
        "pi_root": macro_start,                                     # [P(Run), P(Reversal)] at t0
        "A_macro": _macro_transition(edges, [run_block, rev_block]).tolist(),  # 2x2 Run/Reversal

        "A_production_run": _production_transition(edges, run_block).tolist(),
        "A_production_rev": _production_transition(edges, rev_block).tolist(),
        "B_run_pos": sub(run_block, ZigzagDirection.UP),
        "B_run_neg": sub(run_block, ZigzagDirection.DOWN),
        "B_rev_pos": sub(rev_block, ZigzagDirection.UP),
        "B_rev_neg": sub(rev_block, ZigzagDirection.DOWN),
    }
    return {
        "schema_version": HHMM_SCHEMA_VERSION,
        "topology": TOPOLOGY_NAME,
        "asset_group": asset_group,
        "k": k,
        "flow_feature": flow_feature,
        "session_policy": session_policy,
        "params": params,
        "fit_metadata": fit_metadata or {},
        # Raw flat arrays for high-fidelity reconstruction (not part of the interpretable schema).
        "_reconstruction": {
            "starts": raw_starts.tolist(),
            "edges": raw_edges.tolist(),
            "ends": raw_ends.tolist(),
            "emissions": emissions.tolist(),
            "run_block": int(run_block),
            "distribution_type": DistType.CATEGORICAL.name,
            "n_components": int(model.n_components),
            "session_gate": session_gate,
        },
    }


def _migrate(d: dict) -> dict:
    """Upgrade older-schema dicts to the current version. v1 is the baseline (identity)."""
    version = int(d.get("schema_version", 0))
    if version > HHMM_SCHEMA_VERSION:
        raise ValueError(
            f"param dict schema_version {version} is newer than supported {HHMM_SCHEMA_VERSION}; "
            "upgrade okmich-quant-ml."
        )
    # Future: if version < HHMM_SCHEMA_VERSION, apply stepwise migrations here.
    return d


def from_param_dict(d: dict) -> HierarchicalHMM:
    """Rebuild a ready-to-infer categorical HHMM from a parameter dict (lossless via _reconstruction)."""
    d = _migrate(dict(d))
    recon = d.get("_reconstruction")
    if recon is None:
        raise ValueError(
            "param dict is missing the '_reconstruction' block required for exact reload. It may have "
            "been produced by a schema-only exporter; re-serialise with to_param_dict."
        )
    run_block = int(recon["run_block"])
    macro_labels = {run_block: MacroRegime.RUN.value, 1 - run_block: MacroRegime.REVERSAL.value}
    model = HierarchicalHMM.from_flat_params(
        starts=recon["starts"],
        edges=recon["edges"],
        ends=recon.get("ends"),
        emissions=recon["emissions"],
        macro_labels=macro_labels,
        distribution_type=recon.get("distribution_type", DistType.CATEGORICAL.name),
        n_components=int(recon.get("n_components", 1)),
    )
    gate = recon.get("session_gate")
    if gate is not None:
        _restore_session_gate(model, gate)
    return model


def save_param_dict(d: dict, path: str) -> None:
    """Write a parameter dict to disk as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)


def load_param_dict(path: str) -> dict:
    """Read a parameter dict from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_model(model: HierarchicalHMM, path: str, **metadata: Any) -> dict:
    """Serialise ``model`` to ``path`` (JSON param dict) and return the dict."""
    d = to_param_dict(model, **metadata)
    save_param_dict(d, path)
    return d


def load_model(path: str) -> HierarchicalHMM:
    """Load a categorical HHMM from a JSON param dict at ``path``."""
    return from_param_dict(load_param_dict(path))
