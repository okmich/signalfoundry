"""ConditionalFeatureAnalyzer — public orchestrator for conditional IC analysis."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ._enums import ConditionPass
from ._ic_analysis import check_ic_threshold_diagnostic, compute_conditional_ic, compute_global_ic, compute_ic_threshold
from ._partition import partition_features
from ._result import FeatureConditionMap, assign_status
from ._stability import compute_stability_scores, compute_stability_threshold, compute_subperiod_stability

logger = logging.getLogger(__name__)


class ConditionalFeatureAnalyzer:
    """
    Orchestrates conditional IC analysis across one or more conditioning dimensions.

    This is the public entry point. It runs up to three passes (regime, temporal,
    hierarchical) and returns a FeatureConditionMap for each.

    Parameters
    ----------
    min_observations : int
        Minimum rows per (feature, condition) pair after NaN removal. Default 1000.
    max_pvalue : float
        Maximum Spearman p-value for significance. Default 0.05.
    n_permutations : int
        Number of target shuffles for the null IC distribution. Default 100.
    ic_percentile : float
        Percentile of null |IC| for the noise floor. Default 95.0.
    ic_threshold_override : float or None
        Bypass permutation null with a fixed threshold.
    stability_threshold_override : float or None
        Bypass Jenks natural break with a fixed stability threshold.
    dominance_ratio : float
        For condition_specific: dominant |IC| >= this * next-best. Default 2.0.
    subperiod_instability_ratio : float
        Max subperiod_ic_std / |IC| for global_stable. Default 0.5.
    detection_ratio : float
        Fraction of classifiable features that must be condition-dependent. Default 0.3.
    n_subperiods : int
        Non-overlapping time chunks for sub-period stability. Default 3.
    verbose : bool
        Print progress and adaptive threshold values. Default True.
    """

    def __init__(self, min_observations: int = 1000, max_pvalue: float = 0.05, n_permutations: int = 100, ic_percentile: float = 95.0, ic_threshold_override: float | None = None, stability_threshold_override: float | None = None, dominance_ratio: float = 2.0, subperiod_instability_ratio: float = 0.5, detection_ratio: float = 0.3, n_subperiods: int = 3, verbose: bool = True):
        self.min_observations = min_observations
        self.max_pvalue = max_pvalue
        self.n_permutations = n_permutations
        self.ic_percentile = ic_percentile
        self.ic_threshold_override = ic_threshold_override
        self.stability_threshold_override = stability_threshold_override
        self.dominance_ratio = dominance_ratio
        self.subperiod_instability_ratio = subperiod_instability_ratio
        self.detection_ratio = detection_ratio
        self.n_subperiods = n_subperiods
        self.verbose = verbose

    def analyze_by_regime(self, features_df: pd.DataFrame, target: pd.Series, regime_labels: pd.Series) -> FeatureConditionMap:
        """Pass 1: condition = regime label."""
        return self._run_pass(features_df, target, regime_labels, ConditionPass.REGIME)

    def analyze_by_time(self, features_df: pd.DataFrame, target: pd.Series, time_labels: pd.Series) -> FeatureConditionMap:
        """Pass 2: condition = time-window label."""
        return self._run_pass(features_df, target, time_labels, ConditionPass.TEMPORAL)

    def analyze_hierarchical(self, features_df: pd.DataFrame, target: pd.Series, regime_labels: pd.Series, time_labels: pd.Series) -> FeatureConditionMap:
        """
        Pass 3: condition = (regime, time_window) tuples.

        Should only be called if both Pass 1 and Pass 2 returned
        conditional_structure_detected = True.
        """
        # Validate no | in label values
        for label_series, name in [(regime_labels, "regime_labels"), (time_labels, "time_labels")]:
            unique_vals = label_series.dropna().unique()
            for val in unique_vals:
                if "|" in str(val):
                    raise ValueError(f"Label value '{val}' in {name} contains reserved separator '|'.")

        # Create hierarchical labels — preserve NaN where either label is NaN
        either_nan = regime_labels.isna() | time_labels.isna()
        hierarchical_labels = regime_labels.astype(str) + "|" + time_labels.astype(str)
        hierarchical_labels[either_nan] = np.nan
        hierarchical_labels.name = "hierarchical"

        return self._run_pass(features_df, target, hierarchical_labels, ConditionPass.HIERARCHICAL)

    def full_analysis(self, features_df: pd.DataFrame, target: pd.Series, regime_labels: pd.Series | None = None, time_labels: pd.Series | None = None) -> Dict[ConditionPass, FeatureConditionMap]:
        """
        Run all applicable passes.

        - If only regime_labels: Pass 1 only.
        - If only time_labels: Pass 2 only.
        - If both: Pass 1, Pass 2, and Pass 3 (only if both detect structure).
        - If neither: raises ValueError.
        """
        if regime_labels is None and time_labels is None:
            raise ValueError("At least one of regime_labels or time_labels must be provided.")

        results: Dict[ConditionPass, FeatureConditionMap] = {}

        if regime_labels is not None:
            if self.verbose:
                logger.info("Running Pass 1: regime conditioning...")
            results[ConditionPass.REGIME] = self.analyze_by_regime(features_df, target, regime_labels)
            if self.verbose:
                logger.info("Pass 1 complete. %s", results[ConditionPass.REGIME])

        if time_labels is not None:
            if self.verbose:
                logger.info("Running Pass 2: temporal conditioning...")
            results[ConditionPass.TEMPORAL] = self.analyze_by_time(features_df, target, time_labels)
            if self.verbose:
                logger.info("Pass 2 complete. %s", results[ConditionPass.TEMPORAL])

        # Pass 3 only if both passes ran and both detected structure
        if regime_labels is not None and time_labels is not None:
            regime_detected = results[ConditionPass.REGIME].conditional_structure_detected
            temporal_detected = results[ConditionPass.TEMPORAL].conditional_structure_detected

            if regime_detected and temporal_detected:
                if self.verbose:
                    logger.info("Both passes detected conditional structure. Running Pass 3: hierarchical...")
                results[ConditionPass.HIERARCHICAL] = self.analyze_hierarchical(features_df, target, regime_labels, time_labels)
                if self.verbose:
                    logger.info("Pass 3 complete. %s", results[ConditionPass.HIERARCHICAL])
            elif self.verbose:
                logger.info(
                    "Skipping Pass 3: regime_detected=%s, temporal_detected=%s",
                    regime_detected, temporal_detected,
                )

        return results

    def _run_pass(self, features_df: pd.DataFrame, target: pd.Series, condition_labels: pd.Series, condition_pass: ConditionPass) -> FeatureConditionMap:
        """Execute a single conditioning pass and return a FeatureConditionMap."""

        # 1. Compute condition-stratified IC
        if self.verbose:
            logger.info("  Computing condition-stratified IC...")
        ic_matrix, pvalue_matrix, obs_matrix = compute_conditional_ic(features_df, target, condition_labels, self.min_observations)

        # 2. Compute global IC
        global_ic = compute_global_ic(features_df, target)

        # 3. Determine IC threshold
        if self.ic_threshold_override is not None:
            ic_threshold = self.ic_threshold_override
            if self.verbose:
                logger.info("  Using fixed ic_threshold=%.4f (override)", ic_threshold)
        else:
            if self.verbose:
                logger.info("  Computing adaptive IC threshold via permutation null...")
            ic_threshold = compute_ic_threshold(features_df, target, condition_labels, self.n_permutations, self.ic_percentile, self.min_observations)
            if self.verbose:
                logger.info("  Adaptive ic_threshold=%.4f", ic_threshold)
            # Diagnostic: warn if threshold exceeds median real |IC|
            check_ic_threshold_diagnostic(ic_threshold, ic_matrix)

        # 4. Assign status
        status_matrix = assign_status(ic_matrix, pvalue_matrix, obs_matrix, ic_threshold, self.max_pvalue, self.min_observations)

        # 5. Compute stability scores
        stability_scores = compute_stability_scores(ic_matrix)

        # 6. Determine stability threshold
        if self.stability_threshold_override is not None:
            stability_threshold = self.stability_threshold_override
            if self.verbose:
                logger.info("  Using fixed stability_threshold=%.4f (override)", stability_threshold)
        else:
            stability_threshold = compute_stability_threshold(stability_scores)
            if self.verbose:
                logger.info("  Adaptive stability_threshold=%.4f", stability_threshold)

        # 7. Compute sub-period stability
        if self.verbose:
            logger.info("  Computing sub-period stability...")
        subperiod_ic_std = compute_subperiod_stability(features_df, target, condition_labels, self.n_subperiods, self.min_observations)

        # 8. Partition features
        if self.verbose:
            logger.info("  Partitioning features...")
        global_stable, condition_specific, conditional_ensemble, unclassified, detected = partition_features(
            ic_matrix, status_matrix, stability_scores, subperiod_ic_std,
            ic_threshold, stability_threshold,
            self.dominance_ratio, self.subperiod_instability_ratio, self.detection_ratio,
        )

        # 9. Diagnostics — use median of obs_matrix per condition (post-NaN counts)
        conditions = sorted(condition_labels.dropna().unique())
        condition_sample_counts = obs_matrix.median(axis=0).astype(int)
        condition_sample_counts.name = "condition_sample_counts"
        skipped_conditions = [c for c in conditions if condition_sample_counts.get(c, 0) < self.min_observations]

        return FeatureConditionMap(
            ic_matrix=ic_matrix,
            pvalue_matrix=pvalue_matrix,
            status_matrix=status_matrix,
            global_ic=global_ic,
            stability_scores=stability_scores,
            global_stable=global_stable,
            condition_specific=condition_specific,
            conditional_ensemble=conditional_ensemble,
            unclassified=unclassified,
            ic_threshold=ic_threshold,
            stability_threshold=stability_threshold,
            condition_pass=condition_pass,
            conditional_structure_detected=detected,
            min_observations=self.min_observations,
            max_pvalue=self.max_pvalue,
            dominance_ratio=self.dominance_ratio,
            subperiod_instability_ratio=self.subperiod_instability_ratio,
            subperiod_ic_std=subperiod_ic_std,
            condition_sample_counts=condition_sample_counts,
            skipped_conditions=skipped_conditions,
        )