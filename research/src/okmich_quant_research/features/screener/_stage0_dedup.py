"""
Stage 0 — Prefix Deduplication
================================
Zero-compute deduplication of features that share the same base name across
different prefix families (e.g. ``tm_rsi`` vs ``feat_rsi``).

When two libraries compute the same indicator, the resulting features are nearly
perfectly correlated (>0.95) and one is pure redundancy. This stage removes the
lower-priority duplicate before any expensive computation runs.

Rules
-----
- Each feature is matched against ``prefix_priority`` in order.
  The first matching prefix determines its family rank (0 = most preferred).
- Features in the same family (same base name, different prefix) are deduplicated:
  the highest-priority prefix wins; the rest are dropped.
- Features that match **no** prefix in the list are always kept.
- Features whose base name appears only once (no cross-prefix duplicate) are
  always kept regardless of prefix.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ._result import StageReport


def stage0_prefix_dedup(
    X: pd.DataFrame,
    prefix_priority: tuple[str, ...] = ("tm_", "feat_"),
    verbose: bool = True,
) -> tuple[pd.DataFrame, StageReport]:
    """
    Remove cross-prefix duplicates by name matching.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    prefix_priority : tuple[str, ...]
        Prefixes in descending priority order. When two features share the same
        base name (suffix after stripping the prefix), only the one from the
        highest-priority (leftmost) prefix is kept.
        Default: ``("tm_", "feat_")`` — prefers ``tm_*`` over ``feat_*``.
    verbose : bool

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    """
    n_before = X.shape[1]

    # Assign each column a (base_name, priority_rank).
    # rank = index in prefix_priority; -1 = no prefix matched (always kept).
    col_meta: dict[str, tuple[str, int]] = {}
    for col in X.columns:
        for rank, prefix in enumerate(prefix_priority):
            if col.startswith(prefix):
                col_meta[col] = (col[len(prefix):], rank)
                break
        else:
            col_meta[col] = (col, -1)  # unmatched — unique base name

    # Group columns by base name
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for col, (base, rank) in col_meta.items():
        groups[base].append((rank, col))

    kept: list[str] = []
    removed: list[str] = []

    for base, members in groups.items():
        prefixed   = [(r, c) for r, c in members if r >= 0]
        unprefixed = [c for r, c in members if r < 0]

        if len(prefixed) <= 1:
            # No cross-prefix duplicate — keep everything in this group
            kept.extend(c for _, c in members)
        else:
            # Multiple prefixed features share the same base → keep best rank
            best_col = min(prefixed, key=lambda x: x[0])[1]
            kept.append(best_col)
            kept.extend(unprefixed)  # unmatched features always survive
            removed.extend(c for _, c in prefixed if c != best_col)

    # Preserve original column order
    kept_ordered = [c for c in X.columns if c in set(kept)]

    if verbose:
        print(f"  Stage 0 Prefix Dedup (priority={list(prefix_priority)}): "
              f"{n_before} -> {len(kept_ordered)} features ({len(removed)} removed)")
        if removed:
            print(f"    Removed duplicates: {removed}")

    report = StageReport(
        stage="Stage0_PrefixDedup",
        n_before=n_before,
        n_after=len(kept_ordered),
        removed=removed,
    )
    return X[kept_ordered], report