"""
LeakageReport — output type for LeakageDiagnostics.

Carries the result of comparing two ScreenerResult runs (full vs pruned) and,
optionally, the output of the expensive interaction-SHAP probe (filled in by
``probe``; left as None by ``compare_runs``).

LeakageReport is frozen — diagnostic methods construct a new report when they
need to add fields, rather than mutating in place. This avoids subtle bugs
when a report is passed to a second consumer or cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import pandas as pd

from ._suspects import SuspectRegistry


class Severity(StrEnum):
    """
    Coarse triage label derived from rank correlations.

    The lowest reported correlation drives the verdict — a leak that hits
    SHAP but not MDA (or vice versa) is still a leak.

    CLEAN        : worst correlation >= upper threshold
    WATCH        : worst correlation in [lower, upper)
    INVESTIGATE  : worst correlation < lower threshold (or any NaN, or no metrics)
    """
    CLEAN       = "clean"
    WATCH       = "watch"
    INVESTIGATE = "investigate"


def classify(rank_correlations: dict[str, float],
             thresholds: tuple[float, float] = (0.85, 0.70)) -> Severity:
    """
    Map a dict of rank-correlation values to a Severity.

    Parameters
    ----------
    rank_correlations : dict[str, float]
        E.g. ``{"shap_full_vs_pruned": 0.91, "mda_full_vs_pruned": 0.87}``.
    thresholds : (high, low)
        ``high``: correlations >= high are clean.
        ``low``: correlations < low warrant investigation.
        Default (0.85, 0.70).
    """
    high, low = thresholds
    if not rank_correlations:
        return Severity.INVESTIGATE

    values = list(rank_correlations.values())
    # NaN propagates as INVESTIGATE — "no comparison was possible" is a red flag.
    if any(pd.isna(v) for v in values):
        return Severity.INVESTIGATE

    worst = min(values)
    if worst >= high:
        return Severity.CLEAN
    if worst < low:
        return Severity.INVESTIGATE
    return Severity.WATCH


@dataclass(frozen=True)
class LeakageReport:
    """
    Output of LeakageDiagnostics.compare_runs (and, with interactions, probe).

    Frozen — use :func:`with_interactions` (or ``dataclasses.replace``) to
    derive a new report with the interaction table populated.

    Attributes
    ----------
    suspect_registry : SuspectRegistry
        The original analyst declaration. Retained verbatim so the report
        is self-documenting — the rationale field explains why this probe
        was run.
    label : str
        Free-text label distinguishing this comparison.
    cluster_lineage : pd.DataFrame
        From ResolvedSuspectSet.cluster_lineage on the full run — which
        clusters absorbed which suspects and what survived.
    rank_correlations : dict[str, float]
        Spearman rank correlations on the surviving-features intersection.
        Keys: ``shap_full_vs_pruned``, ``mda_full_vs_pruned``.
    top_movers : pd.DataFrame
        Features with the largest absolute rank shift between the two runs.
        Columns: feature, rank_full, rank_pruned, shap_abs_shift,
        mda_abs_shift, in_suspect_cluster. Sorted by shap_abs_shift desc.
    severity : Severity
        Triage label from ``classify(rank_correlations, thresholds)``.
    notes : tuple[str, ...]
        Human-readable observations: which checks fired, which were skipped,
        what the analyst should look at next. Tuple, not list — frozen-friendly.
    suspect_interactions : pd.DataFrame or None
        Filled in by ``LeakageDiagnostics.probe``. None when only
        ``compare_runs`` has been executed.
    """
    suspect_registry: SuspectRegistry
    label: str
    cluster_lineage: pd.DataFrame
    rank_correlations: dict[str, float]
    top_movers: pd.DataFrame
    severity: Severity
    notes: tuple[str, ...] = field(default_factory=tuple)
    suspect_interactions: pd.DataFrame | None = None

    def with_interactions(self, interactions: pd.DataFrame | None,
                          extra_notes: tuple[str, ...] = ()) -> "LeakageReport":
        """Return a new report with interactions and any extra notes appended."""
        return replace(self, suspect_interactions=interactions, notes=self.notes + tuple(extra_notes))

    def with_notes(self, extra_notes: tuple[str, ...]) -> "LeakageReport":
        """Return a new report with extra notes appended."""
        return replace(self, notes=self.notes + tuple(extra_notes))

    def __repr__(self) -> str:
        corrs = ", ".join(f"{k}={v:.3f}" if not pd.isna(v) else f"{k}=NaN"
                          for k, v in self.rank_correlations.items())
        return (
            f"LeakageReport(label={self.label!r}, severity={self.severity.value}, "
            f"{corrs}, top_movers={len(self.top_movers)})"
        )
