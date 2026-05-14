"""Result types for HmmFeatureScreener."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..screener._result import StageReport
from ._pareto import ParetoStatus


@dataclass(frozen=True)
class AxisEvaluation:
    """Axis-specific output from a single evaluator call.

    Returned by each ``evaluate_*`` function in ``_evaluators.py`` and folded
    into the per-subset ``SubsetEvaluation`` below.
    """
    axis_separation: float
    secondary_robustness: float
    secondary_label: str
    raw_details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubsetEvaluation:
    """One row of the screener's output, per candidate feature subset."""
    features: tuple[str, ...]
    n_features: int
    axis_separation: float
    secondary_robustness: float
    secondary_label: str
    honesty: float
    state_balance_ratio: float
    pareto_status: ParetoStatus
    warnings: tuple[str, ...] = ()
    raw_details: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error: str | None = None  # populated if the fit / evaluation raised


@dataclass
class HmmScreenerResult:
    """Output of ``HmmFeatureScreener.screen()``.

    Combines per-subset evaluations, a tidy DataFrame view for ranking, and the
    stage reports describing what was pre-filtered.
    """
    evaluations: list[SubsetEvaluation]
    results_: pd.DataFrame
    stage_reports: list[StageReport] = field(default_factory=list)

    @property
    def keepers(self) -> list[SubsetEvaluation]:
        """Pareto-optimal, non-trap subsets, ordered by axis_separation descending."""
        return sorted(
            (e for e in self.evaluations if e.pareto_status == ParetoStatus.KEEPER),
            key=lambda e: e.axis_separation, reverse=True,
        )

    @property
    def traps(self) -> list[SubsetEvaluation]:
        """Subsets in the confidence-trap quadrant, ordered by axis_separation descending."""
        return sorted(
            (e for e in self.evaluations if e.pareto_status == ParetoStatus.TRAP),
            key=lambda e: e.axis_separation, reverse=True,
        )

    @property
    def fragile(self) -> list[SubsetEvaluation]:
        """Subsets flagged FRAGILE for structural degeneracy.

        These failed the Phase-A quality gate (missing states, balance ratio
        beyond ``config.max_balance_ratio``, or fewer than
        ``config.min_significant_states`` distinguished states). They never
        reached the Pareto check and should be investigated as model-structure
        failures before any decision is made about the feature subset.
        """
        return sorted(
            (e for e in self.evaluations if e.pareto_status == ParetoStatus.FRAGILE),
            key=lambda e: e.axis_separation, reverse=True,
        )

    def __repr__(self) -> str:
        n = len(self.evaluations)
        n_keep = sum(1 for e in self.evaluations if e.pareto_status == ParetoStatus.KEEPER)
        n_trap = sum(1 for e in self.evaluations if e.pareto_status == ParetoStatus.TRAP)
        n_frag = sum(1 for e in self.evaluations if e.pareto_status == ParetoStatus.FRAGILE)
        n_err = sum(1 for e in self.evaluations if e.error is not None)
        return (f"HmmScreenerResult({n} subsets, {n_keep} keepers, {n_trap} traps, "
                f"{n_frag} fragile, {n_err} errors)")
