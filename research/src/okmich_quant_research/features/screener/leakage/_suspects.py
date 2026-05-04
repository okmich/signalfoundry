"""
Suspect declaration and resolution.

A SuspectRegistry is the analyst's hypothesis about which features in the input
matrix could be carrying source-signal leakage. Resolution joins that hypothesis
against a completed ScreenerResult to answer three questions that downstream
diagnostics need:

    1. Which originally-declared suspects survived to the confirmed set?
    2. Which Stage-3 cluster representatives absorbed a suspect's signal
       (the suspect itself was dropped, but its cluster sibling carries the
       same correlated information forward)?
    3. For each cluster that contained a suspect — what survived, what was
       dropped, and what is the analyst's audit trail?

This module is data-shaping only — no statistical computation, no SHAP, no
model fits. Those belong in the diagnostics module that consumes this output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from .._result import ScreenerResult


@dataclass(frozen=True)
class SuspectRegistry:
    """
    Analyst declaration of features suspected of carrying source-signal leakage.

    Either ``prefixes`` or ``explicit_features`` may be empty, but not both.
    A feature matches if it starts with any prefix OR appears in
    ``explicit_features`` (union semantics).

    Parameters
    ----------
    prefixes : tuple[str, ...]
        Prefix-based match. E.g. ``("tm_vol_", "feat_realized_")``.
    explicit_features : tuple[str, ...]
        Exact feature names. Useful when a single suspect doesn't share a
        common prefix with its family.
    rationale : str
        Free-text explanation of why this set is suspect. Surfaced verbatim
        in the leakage report header so the audit trail is self-documenting.
    """
    prefixes: Tuple[str, ...] = ()
    explicit_features: Tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.prefixes and not self.explicit_features:
            raise ValueError("SuspectRegistry requires at least one prefix or explicit feature")

    def matches(self, feature_name: str) -> bool:
        """True if ``feature_name`` is a declared suspect."""
        if feature_name in self.explicit_features:
            return True
        return any(feature_name.startswith(p) for p in self.prefixes)


@dataclass(frozen=True)
class ResolvedSuspectSet:
    """
    Result of joining a SuspectRegistry against a ScreenerResult.

    Attributes
    ----------
    registry : SuspectRegistry
        The original declaration, retained for the report header.
    direct_suspects : list[str]
        Features in the original input that matched the registry, regardless
        of whether they survived screening. Includes features that were
        rejected at any stage.
    cluster_inherited_suspects : list[str]
        Cluster representatives that survived Stage 3 whose cluster contained
        at least one direct suspect. These are *not* themselves declared
        suspects, but they carry forward the correlated signal of one — the
        most insidious form of leakage because the suspect's name no longer
        appears in the surviving feature list.
    confirmed_suspects : list[str]
        Subset of direct_suspects that ended up in the final confirmed set.
        These are suspects the screener could not eliminate.
    cluster_lineage : pd.DataFrame
        One row per cluster that contained at least one suspect.
        Columns:
            cluster_id          : int
            members             : list[str] — all features in the cluster
            suspects_in_cluster : list[str] — direct suspects in this cluster
            survivor            : str — feature picked as cluster representative
            survivor_is_suspect : bool — True if survivor itself matched the registry
        Sorted by cluster_id ascending.
    """
    registry: SuspectRegistry
    direct_suspects: List[str] = field(default_factory=list)
    cluster_inherited_suspects: List[str] = field(default_factory=list)
    confirmed_suspects: List[str] = field(default_factory=list)
    cluster_lineage: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        return (
            f"ResolvedSuspectSet(direct={len(self.direct_suspects)}, "
            f"cluster_inherited={len(self.cluster_inherited_suspects)}, "
            f"confirmed={len(self.confirmed_suspects)})"
        )


def resolve(registry: SuspectRegistry, result: ScreenerResult, *,
            input_features: List[str] | None = None) -> ResolvedSuspectSet:
    """
    Resolve a SuspectRegistry against a completed ScreenerResult.

    Parameters
    ----------
    registry : SuspectRegistry
        The analyst's suspect declaration.
    result : ScreenerResult
        Output of ``FeatureScreener.screen_for_*``. Must have the persistence
        fields populated (stage1_scores, cluster_assignments,
        cluster_representatives, boruta_groups). Results from before the
        plumbing change will return an empty cluster_lineage.
    input_features : list[str] or None
        The full feature list that entered the screener (Stage 0 input).
        Used to find direct suspects that were rejected before Stage 3
        and therefore don't appear in any cluster.
        When None, falls back to the union of cluster members and
        confirmed/tentative/rejected lists — which captures every feature the
        screener knew about and is sufficient for most use.

    Returns
    -------
    ResolvedSuspectSet
    """
    # Build the universe of features the screener saw
    if input_features is not None:
        universe = set(input_features)
    else:
        universe = set(result.confirmed) | set(result.tentative) | set(result.rejected)
        for members in result.cluster_assignments.values():
            universe.update(members)

    direct_suspects = sorted(f for f in universe if registry.matches(f))
    direct_suspect_set = set(direct_suspects)

    # For each cluster, identify suspects and inheritance
    lineage_rows: list[dict] = []
    cluster_inherited: list[str] = []

    for cluster_id in sorted(result.cluster_assignments.keys()):
        members = result.cluster_assignments[cluster_id]
        suspects_here = [m for m in members if m in direct_suspect_set]
        if not suspects_here:
            continue

        survivor = result.cluster_representatives.get(cluster_id, "")
        survivor_is_suspect = survivor in direct_suspect_set

        if survivor and not survivor_is_suspect:
            cluster_inherited.append(survivor)

        lineage_rows.append({
            "cluster_id":          cluster_id,
            "members":             list(members),
            "suspects_in_cluster": suspects_here,
            "survivor":            survivor,
            "survivor_is_suspect": survivor_is_suspect,
        })

    cluster_lineage = pd.DataFrame(lineage_rows)
    if not cluster_lineage.empty:
        cluster_lineage = cluster_lineage.sort_values("cluster_id").reset_index(drop=True)

    confirmed_set = set(result.confirmed)
    confirmed_suspects = sorted(s for s in direct_suspects if s in confirmed_set)

    return ResolvedSuspectSet(
        registry=registry,
        direct_suspects=direct_suspects,
        cluster_inherited_suspects=sorted(set(cluster_inherited)),
        confirmed_suspects=confirmed_suspects,
        cluster_lineage=cluster_lineage,
    )
