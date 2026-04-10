"""Three-way feature partition logic with strict precedence."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ._enums import FeatureStatus


def _significant_conditions(status_row: pd.Series) -> pd.Index:
    """Return conditions where the feature is ACTIVE or NEGATIVE (significant signal)."""
    return status_row[(status_row == FeatureStatus.ACTIVE) | (status_row == FeatureStatus.NEGATIVE)].index


def _computed_conditions(status_row: pd.Series) -> pd.Index:
    """Return conditions where IC was actually computed (not INSUFFICIENT_DATA)."""
    return status_row[status_row != FeatureStatus.INSUFFICIENT_DATA].index


def partition_features(ic_matrix: pd.DataFrame, status_matrix: pd.DataFrame, stability_scores: pd.Series, subperiod_ic_std: pd.DataFrame, ic_threshold: float, stability_threshold: float, dominance_ratio: float = 2.0, subperiod_instability_ratio: float = 0.5, detection_ratio: float = 0.3) -> Tuple[List[str], Dict[str, List[str]], List[str], List[str], bool]:
    """
    Classify every feature into exactly one bucket using strict precedence:
    unclassified -> global_stable -> condition_specific -> conditional_ensemble.

    All significance decisions are driven by the status_matrix (which already
    encodes both ic_threshold and max_pvalue checks from assign_status).

    Parameters
    ----------
    ic_matrix : pd.DataFrame
        Features x conditions, float IC values.
    status_matrix : pd.DataFrame
        Features x conditions, FeatureStatus values.
    stability_scores : pd.Series
        Per-feature stability score.
    subperiod_ic_std : pd.DataFrame
        Features x conditions, IC std across sub-periods.
    ic_threshold : float
        Adaptive noise floor (used only for dominance ratio comparison
        against below-threshold IC magnitudes).
    stability_threshold : float
        Adaptive boundary between stable and variable.
    dominance_ratio : float
        For condition_specific: dominant |IC| must be >= this * next-best.
    subperiod_instability_ratio : float
        Max subperiod_ic_std / |IC| for global_stable eligibility.
    detection_ratio : float
        Fraction of classifiable features that must be condition-dependent.

    Returns
    -------
    global_stable : list[str]
    condition_specific : dict[str, list[str]]
    conditional_ensemble : list[str]
    unclassified : list[str]
    conditional_structure_detected : bool
    """
    features = ic_matrix.index.tolist()

    global_stable: List[str] = []
    condition_specific: Dict[str, List[str]] = {}
    conditional_ensemble: List[str] = []
    unclassified: List[str] = []

    claimed: set[str] = set()

    # --- Step 1: unclassified ---
    # Features with no ACTIVE or NEGATIVE status in any condition
    # (either INACTIVE everywhere, INSUFFICIENT_DATA everywhere, or both).
    for feat in features:
        status_row = status_matrix.loc[feat]
        significant = _significant_conditions(status_row)

        if len(significant) == 0:
            unclassified.append(feat)
            claimed.add(feat)

    # --- Step 2: global_stable ---
    for feat in features:
        if feat in claimed:
            continue

        ic_row = ic_matrix.loc[feat]
        status_row = status_matrix.loc[feat]
        stab = stability_scores.get(feat, np.nan)

        # Need stability score
        if np.isnan(stab):
            continue

        # Stability above threshold
        if stab <= stability_threshold:
            continue

        # Must be ACTIVE or NEGATIVE in every computed condition
        computed = _computed_conditions(status_row)
        if len(computed) < 2:
            continue

        significant = _significant_conditions(status_row)
        if len(significant) != len(computed):
            # Some computed conditions are INACTIVE — not globally active
            continue

        # IC sign consistent across all computed conditions
        computed_ics = ic_row[computed].dropna()
        if len(computed_ics) < 2:
            continue
        signs = np.sign(computed_ics.values)
        if not (np.all(signs > 0) or np.all(signs < 0)):
            continue

        # Sub-period stability check: subperiod_ic_std / |IC| <= ratio
        # in every condition where sub-period stability can be computed
        subperiod_ok = True
        for cond in computed:
            sp_std = subperiod_ic_std.loc[feat, cond] if feat in subperiod_ic_std.index and cond in subperiod_ic_std.columns else np.nan
            ic_val = abs(ic_row[cond])

            if np.isnan(sp_std):
                continue  # Can't compute — don't block

            if ic_val == 0:
                subperiod_ok = False
                break

            if sp_std / ic_val > subperiod_instability_ratio:
                subperiod_ok = False
                break

        if not subperiod_ok:
            continue

        global_stable.append(feat)
        claimed.add(feat)

    # --- Step 3: condition_specific ---
    for feat in features:
        if feat in claimed:
            continue

        ic_row = ic_matrix.loc[feat]
        status_row = status_matrix.loc[feat]

        # Need at least one significant condition (ACTIVE or NEGATIVE)
        significant = _significant_conditions(status_row)
        if len(significant) == 0:
            continue

        # Dominance check: best |IC| >= dominance_ratio * next-best |IC|
        # Compare against ALL computed conditions (including below-threshold)
        computed = _computed_conditions(status_row)
        computed_ics = ic_row[computed].dropna()

        abs_ics = computed_ics.abs().sort_values(ascending=False)
        if len(abs_ics) < 2:
            # Only one condition computed — trivially passes dominance
            dominant_cond = abs_ics.index[0]
            if dominant_cond not in condition_specific:
                condition_specific[dominant_cond] = []
            condition_specific[dominant_cond].append(feat)
            claimed.add(feat)
            continue

        best_ic = abs_ics.iloc[0]
        next_best_ic = abs_ics.iloc[1]

        if next_best_ic == 0:
            # All others are zero — trivially passes
            dominant_cond = abs_ics.index[0]
            if dominant_cond not in condition_specific:
                condition_specific[dominant_cond] = []
            condition_specific[dominant_cond].append(feat)
            claimed.add(feat)
            continue

        if best_ic >= dominance_ratio * next_best_ic:
            dominant_cond = abs_ics.index[0]
            if dominant_cond not in condition_specific:
                condition_specific[dominant_cond] = []
            condition_specific[dominant_cond].append(feat)
            claimed.add(feat)

    # --- Step 4: conditional_ensemble ---
    for feat in features:
        if feat in claimed:
            continue

        status_row = status_matrix.loc[feat]
        significant = _significant_conditions(status_row)

        if len(significant) >= 2:
            conditional_ensemble.append(feat)
            claimed.add(feat)

    # --- Remaining edge cases -> unclassified ---
    for feat in features:
        if feat not in claimed:
            unclassified.append(feat)

    # --- Detection flag ---
    classifiable = len(features) - len(unclassified)
    if classifiable == 0:
        conditional_structure_detected = False
    else:
        n_conditional = len(conditional_ensemble) + sum(len(v) for v in condition_specific.values())
        conditional_structure_detected = n_conditional > detection_ratio * classifiable

    return global_stable, condition_specific, conditional_ensemble, unclassified, conditional_structure_detected