"""Pareto classifier on (axis_separation, honesty) measurements."""
from __future__ import annotations

from enum import StrEnum

import numpy as np


class ParetoStatus(StrEnum):
    """Per-subset verdict from the screener's classifier.

    The screener classifies in two phases:

    Phase A — structural quality gate, before the Pareto check:
      FRAGILE  — the HMM produced a structurally degenerate state structure (states collapsed below the minimum, or one
                 state dominates the rest beyond the configured balance ratio, or the axis evaluator found too few significant states).
                 Fix the model before judging axis content.

    Phase B — Pareto check, on Phase-A survivors:
      ASYMMETRY_CANDIDATE — Pareto-optimal on (axis_sep high, honesty low) and not a trap. A *structural* candidate whose
                 asymmetry still needs Stage-2 confirmation (walk-forward + incremental) before it is confirmed asymmetry.
                 (Previously named KEEPER; the confirmed terminal lives in the confirmer's ``ValidationVerdict.CONFIRMED``.)
      TRAP       — honesty rate exceeds the trap threshold; manual judgment required regardless of axis_separation.
      DOMINATED  — neither Pareto-optimal nor a trap. Strictly worse than at least one candidate on both axis_sep and honesty.
    """
    ASYMMETRY_CANDIDATE = "asymmetry_candidate"
    TRAP = "trap"
    FRAGILE = "fragile"
    DOMINATED = "dominated"


def classify_pareto(measurements: list[tuple[float, float]], honesty_trap_rate: float) -> list[ParetoStatus]:
    """Classify each ``(axis_separation, honesty)`` point.

    Logic:
      1. ``TRAP`` if ``honesty > honesty_trap_rate`` — supersedes Pareto.
      2. ``ASYMMETRY_CANDIDATE`` if non-trap and Pareto-optimal among non-trap points (no other non-trap point has BOTH higher axis_sep AND lower honesty).
      3. ``DOMINATED`` otherwise.

    Parameters
    ----------
    measurements : list[tuple[float, float]]
        Per-subset ``(axis_separation, honesty)``; both floats; ``honesty ∈ [0, 1]``.
    honesty_trap_rate : float
        Threshold above which a subset is classified as a confidence trap.

    Returns
    -------
    list[ParetoStatus]
        Status per input measurement, in the same order as ``measurements``.
    """
    if not measurements:
        return []

    seps = np.asarray([m[0] for m in measurements], dtype=float)
    hons = np.asarray([m[1] for m in measurements], dtype=float)
    non_trap = hons <= honesty_trap_rate

    statuses: list[ParetoStatus] = []
    for i, (sep, hon) in enumerate(zip(seps, hons)):
        if hon > honesty_trap_rate:
            statuses.append(ParetoStatus.TRAP)
            continue
        # Strict dominance check among non-trap points:
        # is there a non-trap subset with strictly higher sep AND strictly lower honesty?
        is_dominated = bool(np.any((seps > sep) & (hons < hon) & non_trap))
        statuses.append(ParetoStatus.DOMINATED if is_dominated else ParetoStatus.ASYMMETRY_CANDIDATE)
    return statuses
