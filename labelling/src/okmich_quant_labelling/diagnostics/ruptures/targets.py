"""Hindsight target derivations from ``LabeledPosteriors`` (spec §1 catalogue).

Targets are functions of the offline ``breakpoints``/``segment_ids`` only. Every target is forward-looking and carries a
lookahead horizon ``L`` that dictates the split-edge embargo; see spec §7. Callers MUST recompute targets fold-locally
rather than caching across data extensions — adding new bars shifts historical breakpoints (spec §9 risk 4).

Public objects:
    - ``is_boundary(labeled, w)`` — binary classification target. ``L = w``.
    - ``cp_distance(labeled, horizon)`` — right-censored survival pair ``(d, event)``. ``L = horizon``.
    Uncapped ``cp_distance`` is forbidden by the spec.
    - ``within_segment_position(labeled)`` — regression in [0, 1]. ``L`` is unbounded; fold-local generation is mandatory.
    - ``censor_fold_edge_segments(labeled)`` — boolean mask flagging bars to drop from training/eval when
    ``within_segment_position`` is the target and ``signal`` was a fold-local window. Drops both the leftmost and
      rightmost segments (both edges are fold artefacts, not real boundaries).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from okmich_quant_labelling.diagnostics.ruptures.artefacts import LabeledPosteriors


@dataclass(frozen=True)
class CpDistanceTarget:
    """Right-censored survival pair for ``cp_distance``.

    Attributes
    ----------
    distance : NDArray
        Per-bar ``min(d_raw, horizon)`` where ``d_raw`` is the bars-to-next interior breakpoint (or ``+inf`` if none). Shape ``(T,)``, int64.
    event : NDArray
        Per-bar ``1{d_raw <= horizon}``. Shape ``(T,)``, int8 (0/1).
    horizon : int
        The right-censoring horizon ``H``. Stored so the caller cannot lose track of it after the function returns.
    """

    distance: NDArray
    event: NDArray
    horizon: int


def _interior_breakpoints(labeled: LabeledPosteriors) -> NDArray:
    """``B = breakpoints[:-1]`` — interior breakpoints, excluding the trailing endpoint."""
    return labeled.breakpoints[:-1].copy()


def is_boundary(labeled: LabeledPosteriors, w: int) -> NDArray:
    """Binary boundary target with tolerance ``w``.

    ``is_boundary[t] = 1{∃ b ∈ B with t ∈ [b - w, b + w - 1]}`` for
    interior breakpoints ``B = breakpoints[:-1]``.

    Parameters
    ----------
    w : int
        Tolerance half-width, ``>= 1``. The lookahead horizon ``L = w``; downstream split-edge embargo width must be at least ``w``.

    Returns
    -------
    NDArray
        Shape ``(T,)``, dtype int8 (0/1).
    """
    if int(w) < 1:
        raise ValueError(f"w must be >= 1, got {w}")
    t_total = labeled.segment_ids.shape[0]
    interior = _interior_breakpoints(labeled)
    out = np.zeros(t_total, dtype=np.int8)
    for b in interior:
        lo = max(0, int(b) - int(w))
        hi = min(t_total, int(b) + int(w))
        out[lo:hi] = 1
    return out


def cp_distance(labeled: LabeledPosteriors, horizon: int) -> CpDistanceTarget:
    """Right-censored survival target for distance to next interior breakpoint.

    For each bar ``t``, ``d_raw = min{b - t : b ∈ B, b > t}`` (or ``+inf`` if no next interior breakpoint exists).
    Returns ``d = min(d_raw, horizon)`` and ``event = 1{d_raw <= horizon}``.

    The horizon is mandatory: an uncapped ``cp_distance`` is forbidden because its lookahead would be unbounded,
    defeating the split-edge embargo and leaking eval-period structure into training labels.

    Parameters
    ----------
    horizon : int
        Right-censoring horizon ``H``, ``>= 1``. The lookahead horizon ``L = horizon``.
    """
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    t_total = labeled.segment_ids.shape[0]
    interior = _interior_breakpoints(labeled)
    distance = np.full(t_total, int(horizon), dtype=np.int64)
    event = np.zeros(t_total, dtype=np.int8)
    if interior.size == 0:
        return CpDistanceTarget(distance=distance, event=event, horizon=int(horizon))

    sorted_b = np.sort(interior).astype(np.int64)
    indices = np.arange(t_total, dtype=np.int64)
    nxt = np.searchsorted(sorted_b, indices, side="right")
    has_next = nxt < sorted_b.size
    next_b = np.where(has_next, sorted_b[np.clip(nxt, 0, sorted_b.size - 1)], 0)
    d_raw = np.where(has_next, next_b - indices, int(horizon) + 1)
    capped = np.minimum(d_raw, int(horizon))
    event_mask = has_next & (d_raw <= int(horizon))
    distance[:] = capped
    event[event_mask] = 1
    return CpDistanceTarget(distance=distance, event=event, horizon=int(horizon))


def within_segment_position(labeled: LabeledPosteriors) -> NDArray:
    """Position within the current segment, in [0, 1].

    ``within_segment_position[t] = (t - segment_start) / max(segment_length - 1, 1)`` where ``segment_start``,
    ``segment_length`` come from ``segment_ids[t]``.

    Lookahead is **unbounded** — the value at bar ``t`` depends on the segment's right endpoint, which can lie arbitrarily
    far in the future. Fold-local label generation is mandatory; no finite split-edge embargo width is sufficient.
    When ``signal`` was a fold-local window, also apply ``censor_fold_edge_segments`` to drop the leftmost and rightmost segments
    before training/eval.

    Returns
    -------
    NDArray
        Shape ``(T,)``, dtype float64.
    """
    seg_ids = labeled.segment_ids
    t_total = seg_ids.shape[0]
    if t_total == 0:
        return np.empty(0, dtype=np.float64)

    boundaries = np.r_[0, np.flatnonzero(np.diff(seg_ids)) + 1, t_total]
    out = np.empty(t_total, dtype=np.float64)
    for i in range(boundaries.size - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        length = end - start
        denom = max(length - 1, 1)
        out[start:end] = (np.arange(length, dtype=np.float64)) / float(denom)
    return out


def censor_fold_edge_segments(labeled: LabeledPosteriors) -> NDArray:
    """Boolean mask of bars to **drop** under fold-local generation for ``within_segment_position``.

    Drops both edge segments because each is a fold artefact:
    - The rightmost segment is closed at ``len(signal)`` because PELT must terminate, not because a real change point exists there.
      ``segment_length`` is truncated, so the position denominator is wrong.
    - The leftmost segment starts at index ``0`` because PELT must begin somewhere, not because a real change point sits
      at the fold boundary. ``segment_start`` is artificial, so the position numerator's anchor is wrong.

    Returns
    -------
    NDArray
        Shape ``(T,)``, dtype bool. ``True`` means "drop this bar".

    Notes
    -----
    Apply only when ``signal`` was a fold-local window. When ``signal`` is the full history, neither edge is artificial
    and this mask should not be used. Callers are responsible for tracking which case they are in — the artefact
    does not record it.

    ``is_boundary`` and ``cp_distance`` are unaffected by fold-edge artefacts: they are defined against interior breakpoints
    ``B = breakpoints[:-1]``, which already excludes the trailing endpoint, and they do not depend on segment-start anchoring.
    """
    seg_ids = labeled.segment_ids
    if seg_ids.size == 0:
        return np.zeros(0, dtype=bool)
    leftmost = seg_ids[0]
    rightmost = seg_ids[-1]
    return (seg_ids == leftmost) | (seg_ids == rightmost)
