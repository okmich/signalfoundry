"""
Stage 0c — Collinearity Diagnostic / Filter (HMM-specific, OPT-IN)
=================================================================
Removes **near-duplicate** features: a feature that is ~a linear copy of ANOTHER SINGLE feature carries no
information the other does not, and feeding the pair into a joint HMM emission ill-conditions the
covariance so the multi-restart max-LL pick becomes a coin flip. The empirical failure that motivated this
stage: two return-driver features at correlation 1.000 (a 2-feature VIF ~59,000, condition number 343,140)
fed to the same 3-state HMM produced a partition that flipped under a 0.24% change in the data window.

The three stage-0 gates are complementary, not redundant:
  * Stage 0  (variance)     removes features with NO information       (dead / near-constant).
  * Stage 0b (persistence)  removes features with NO memory            (white noise in every moment).
  * Stage 0c (collinearity) removes features with NO INDEPENDENT info  (near-duplicate of another one).

WHY PAIRWISE, NOT SET-WISE VIF
------------------------------
This runs as a GLOBAL pre-filter, before subset generation, so it must only remove features whose
redundancy is scope-invariant. Pairwise near-duplication is: if |corr(a, b)| ~ 1 over the whole sample,
then a and b are redundant in EVERY subset that contains both, so dropping one globally never denies the
subset search a valid candidate. Textbook set-wise VIF does NOT have this property and must not be used
here: for independent a, b and sum = a + b, the set {a, b, sum} is perfectly collinear (all set-wise VIF
-> inf) yet {a, sum} and {b, sum} are valid subsets (pairwise VIF ~1.97). A global set-wise filter would
delete `sum` and destroy those subsets. Set-wise VIF is also undefined when n <= p (the regression
saturates, R^2 -> 1 spuriously), so on a wide/short design it removes valid independent features. Pairwise
correlation has neither problem: it needs no regression, is well-defined for any n, and is O(n * p^2)
rather than O(n * p^4) under greedy recomputation.

The knob is still expressed as ``max_vif`` for continuity: for a PAIR, ``VIF = 1/(1 - r^2)``, so
``max_vif`` is equivalent to a correlation ceiling ``|r| >= sqrt(1 - 1/max_vif)`` (``max_vif=10`` <-> 0.949).

WHAT THIS DELIBERATELY DOES NOT CATCH
-------------------------------------
Genuine MULTI-WAY dependencies where no single pair is near-1 (e.g. sum = a + b). Removing such a feature
globally would be wrong (see above); its only harm is inside the *specific* subset that realises the
dependency, which is the per-subset fit's remit (an ill-conditioned emission there should be caught as a
structural/FRAGILE failure), not this global stage's. VIF here is also blind, as ``_persistence.py`` is,
to regime-switching covariance (two features whose correlation flips sign by regime average to ~0 |corr|,
so they are correctly kept) — which is another reason the threshold stays high and removal stays opt-in.

GREEDY REMOVAL AND THE TIE-BREAK
--------------------------------
Iterative: find the strongest remaining pair; if its VIF exceeds ``max_vif`` drop the LESS persistent of
the two (lowest Stage-0b score, NaN treated as -inf), so the survivor of a duplicate cluster is the more
dwell-friendly member; recompute and repeat. A cluster of mutual near-duplicates collapses to its single
most-persistent member.

SAMPLE-SIZE LIMIT
-----------------
Correlations are pairwise-complete with ``min_periods=min_obs``: a pair whose overlap is below ``min_obs``
scores NaN and is never eligible for removal (untrusted). Constant columns yield NaN correlations and are
likewise left to the variance filter.

CALIBRATION
-----------
Default ``max_vif=inf`` = DIAGNOSTIC ONLY (per-feature nearest-duplicate VIF is still scored into the
report's ``detail`` so a threshold can be tuned from a run without removals). ``10`` is the textbook
"severe" line and the FXPIG-M5 setting; it is a threshold on the estimation set, not validated
cross-instrument. This is a FLOOR, never an objective: do not select features toward low mutual VIF —
orthogonality is not edge, and forcing it discards weak-but-independent features that combine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..screener._result import StageReport
from ._persistence import persistence_score

DEFAULT_MIN_OBS = 200
VIF_CAP = 1.0e6  # keep a perfect-duplicate's pairwise VIF finite and comparable


def _abs_corr(X: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Absolute pairwise-complete Pearson correlation with the diagonal blanked. A pair with fewer than
    ``min_obs`` jointly-present rows is NaN (untrusted); a constant column is NaN throughout."""
    C = X.corr(min_periods=min_obs).abs()
    for c in C.columns:
        C.loc[c, c] = np.nan
    return C


def _vif_from_r(r: float) -> float:
    """Pairwise VIF ``1/(1 - r^2)``, capped so an exact duplicate stays finite/comparable."""
    r2 = min(float(r) ** 2, 1.0 - 1.0 / VIF_CAP)
    return 1.0 / (1.0 - r2)


def nearest_duplicate_vif(X: pd.DataFrame, min_obs: int = DEFAULT_MIN_OBS) -> dict[str, float]:
    """Per feature, the pairwise VIF against its single nearest neighbour: ``1/(1 - max_j corr(i,j)^2)``.

    This is the near-duplicate score the filter acts on — NOT the textbook set-wise VIF (see module
    docstring for why set-wise is wrong for a global pre-filter). A lone column scores ``1.0``; a column
    with no sufficiently-overlapping, non-constant partner scores ``NaN``.
    """
    cols = list(X.columns)
    if len(cols) < 2:
        return {c: 1.0 for c in cols}
    C = _abs_corr(X[cols], min_obs)
    out: dict[str, float] = {}
    for c in cols:
        rmax = C[c].max(skipna=True)
        out[c] = float("nan") if rmax != rmax else _vif_from_r(rmax)
    return out


def _persistence_key(series: pd.Series) -> float:
    """Stage-0b persistence, with NaN (too few adjacent pairs) mapped to -inf so an unscoreable duplicate
    is dropped before a scoreable one."""
    s = persistence_score(series)
    return s if s == s else float("-inf")


def stage0c_collinearity_filter(X: pd.DataFrame, max_vif: float = float("inf"),
                                min_obs: int = DEFAULT_MIN_OBS,
                                verbose: bool = True) -> tuple[pd.DataFrame, StageReport]:
    """Score near-duplication; optionally remove near-duplicate features greedily (pairwise).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = bars, cols = features). May contain NaN.
    max_vif : float
        Pairwise VIF ceiling; ``inf`` (default) = DIAGNOSTIC ONLY (score into ``report.detail``, remove
        nothing). Equivalent correlation ceiling is ``|r| >= sqrt(1 - 1/max_vif)``. Must be >= 1.0. Keep
        it HIGH: this removes near-duplicates, it does NOT orthogonalise (see module docstring).
    min_obs : int
        Minimum jointly-present rows for a pair to be trusted; below it the pair is never removed.
    verbose : bool
        Print removals, and the strongest nearest-duplicate VIFs when running as a diagnostic.

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
        ``report.detail["nearest_duplicate_vif"]`` holds the per-feature score (always populated);
        ``report.detail["removed_vif"]`` maps each removed feature to the pair VIF that evicted it.
    """
    if max_vif != max_vif:  # NaN would silently disable every comparison
        raise ValueError("max_vif must be a number (or inf), got NaN")
    if max_vif < 1.0:
        raise ValueError(f"max_vif must be >= 1.0 (VIF's floor for an orthogonal feature), got {max_vif}")
    if min_obs < 3:
        raise ValueError(f"min_obs must be >= 3, got {min_obs}")

    n_before = X.shape[1]
    scores = nearest_duplicate_vif(X, min_obs)  # ALWAYS computed, even in diagnostic mode
    detail: dict = {"nearest_duplicate_vif": {c: (round(v, 3) if v == v else None) for c, v in scores.items()}}

    def _report(cols, removed, removed_vif):
        detail["removed_vif"] = {c: round(v, 1) for c, v in removed_vif.items()}
        return X[cols], StageReport(stage="Stage0c_Collinearity", n_before=n_before,
                                    n_after=len(cols), removed=removed, detail=detail)

    if np.isinf(max_vif) or n_before < 2:
        if verbose:
            strong = sorted(((v, c) for c, v in scores.items() if v == v), reverse=True)[:3]
            if strong:
                print("  Stage 0c (diagnostic, no removal) strongest nearest-duplicate VIF: "
                      + ", ".join(f"{c} ({v:.1f})" for v, c in strong))
        return _report(list(X.columns), [], {})

    cols = list(X.columns)
    removed: list[str] = []
    removed_vif: dict[str, float] = {}
    while len(cols) > 1:
        C = _abs_corr(X[cols], min_obs)          # recompute: dropping a col can widen the others' overlap
        stacked = C.stack()                       # (i, j), i != j, NaN pairs already excluded
        if stacked.empty:
            break
        rmax = float(stacked.max())
        pair_vif = _vif_from_r(rmax)
        if pair_vif <= max_vif:
            break
        i, j = stacked.idxmax()
        drop = min((i, j), key=lambda c: (_persistence_key(X[c]), c))  # drop the less-persistent member
        cols.remove(drop)
        removed.append(drop)
        removed_vif[drop] = pair_vif

    if verbose and removed:
        detail_str = ", ".join(f"{c} (VIF {removed_vif[c]:.0f})" for c in removed)
        print(f"  Stage 0c removed {len(removed)} near-duplicate feature(s) (VIF > {max_vif:g}): {detail_str}")

    return _report(cols, removed, removed_vif)
