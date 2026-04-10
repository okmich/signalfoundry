"""FeatureConditionMap result object and status assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ._enums import ConditionPass, FeatureBucket, FeatureStatus


def assign_status(ic_matrix: pd.DataFrame, pvalue_matrix: pd.DataFrame, obs_matrix: pd.DataFrame, ic_threshold: float, max_pvalue: float = 0.05, min_observations: int = 1000) -> pd.DataFrame:
    """
    Convert IC matrix to status matrix.

    Cell rules (applied in order):
    - n_obs < min_observations -> INSUFFICIENT_DATA
    - p_value > max_pvalue -> INACTIVE (not statistically significant)
    - IC > ic_threshold -> ACTIVE
    - IC < -ic_threshold -> NEGATIVE (inversely predictive)
    - otherwise -> INACTIVE

    Parameters
    ----------
    ic_matrix : pd.DataFrame
        Features x conditions, float IC values.
    pvalue_matrix : pd.DataFrame
        Features x conditions, float p-values.
    obs_matrix : pd.DataFrame
        Features x conditions, int observation counts.
    ic_threshold : float
        Adaptive noise floor.
    max_pvalue : float
        Maximum p-value for significance.
    min_observations : int
        Minimum observations for reliable estimation.

    Returns
    -------
    pd.DataFrame
        Features x conditions, FeatureStatus values.
    """
    status = pd.DataFrame(FeatureStatus.INACTIVE, index=ic_matrix.index, columns=ic_matrix.columns)

    for feat in ic_matrix.index:
        for cond in ic_matrix.columns:
            n_obs = obs_matrix.loc[feat, cond]
            if n_obs < min_observations:
                status.loc[feat, cond] = FeatureStatus.INSUFFICIENT_DATA
                continue

            pval = pvalue_matrix.loc[feat, cond]
            if np.isnan(pval) or pval > max_pvalue:
                status.loc[feat, cond] = FeatureStatus.INACTIVE
                continue

            ic = ic_matrix.loc[feat, cond]
            if np.isnan(ic):
                status.loc[feat, cond] = FeatureStatus.INACTIVE
                continue

            if ic > ic_threshold:
                status.loc[feat, cond] = FeatureStatus.ACTIVE
            elif ic < -ic_threshold:
                status.loc[feat, cond] = FeatureStatus.NEGATIVE
            else:
                status.loc[feat, cond] = FeatureStatus.INACTIVE

    return status


@dataclass
class FeatureConditionMap:
    """Primary output of the conditional feature analysis."""

    # -- Core data --
    ic_matrix: pd.DataFrame
    pvalue_matrix: pd.DataFrame
    status_matrix: pd.DataFrame
    global_ic: pd.Series
    stability_scores: pd.Series

    # -- Three-way partition --
    global_stable: List[str]
    condition_specific: Dict[str, List[str]]
    conditional_ensemble: List[str]
    unclassified: List[str]

    # -- Adaptive thresholds (computed, not configured) --
    ic_threshold: float
    stability_threshold: float

    # -- Fixed configuration --
    condition_pass: ConditionPass
    conditional_structure_detected: bool
    min_observations: int
    max_pvalue: float
    dominance_ratio: float
    subperiod_instability_ratio: float

    # -- Sub-period stability --
    subperiod_ic_std: pd.DataFrame

    # -- Diagnostics --
    condition_sample_counts: pd.Series
    skipped_conditions: List[str] = field(default_factory=list)

    def active_features_for(self, condition: str) -> List[str]:
        """Return features with Active status in the given condition."""
        if condition not in self.status_matrix.columns:
            raise KeyError(f"Condition '{condition}' not found. Available: {list(self.status_matrix.columns)}")
        col = self.status_matrix[condition]
        return col[col == FeatureStatus.ACTIVE].index.tolist()

    def inverted_features_for(self, condition: str) -> List[str]:
        """Return features with Negative (inversely predictive) status in the given condition."""
        if condition not in self.status_matrix.columns:
            raise KeyError(f"Condition '{condition}' not found. Available: {list(self.status_matrix.columns)}")
        col = self.status_matrix[condition]
        return col[col == FeatureStatus.NEGATIVE].index.tolist()

    def sign_flipping_features(self) -> List[str]:
        """Return features that are Active in some conditions and Negative in others."""
        result = []
        for feat in self.status_matrix.index:
            row = self.status_matrix.loc[feat]
            has_active = (row == FeatureStatus.ACTIVE).any()
            has_negative = (row == FeatureStatus.NEGATIVE).any()
            if has_active and has_negative:
                result.append(feat)
        return result

    def feature_profile(self, feature: str) -> pd.DataFrame:
        """
        Return IC, p-value, status, sub-period stability, and bucket across
        all conditions for one feature. Highlights sign flips.
        """
        if feature not in self.ic_matrix.index:
            raise KeyError(f"Feature '{feature}' not found.")

        bucket = self._feature_bucket(feature)
        is_sign_flipper = feature in self.sign_flipping_features()

        rows = []
        for cond in self.ic_matrix.columns:
            rows.append({
                "condition": cond,
                "ic": self.ic_matrix.loc[feature, cond],
                "pvalue": self.pvalue_matrix.loc[feature, cond],
                "status": self.status_matrix.loc[feature, cond],
                "subperiod_ic_std": self.subperiod_ic_std.loc[feature, cond] if feature in self.subperiod_ic_std.index \
                                                                                and cond in self.subperiod_ic_std.columns else np.nan,
            })

        df = pd.DataFrame(rows)
        df["bucket"] = bucket
        df["sign_flipping"] = is_sign_flipper
        df["stability_score"] = self.stability_scores.get(feature, np.nan)
        df["global_ic"] = self.global_ic.get(feature, np.nan)
        return df

    def condition_profile(self, condition: str) -> pd.DataFrame:
        """Return IC, status, and bucket for all features in one condition."""
        if condition not in self.ic_matrix.columns:
            raise KeyError(f"Condition '{condition}' not found.")

        rows = []
        for feat in self.ic_matrix.index:
            rows.append({
                "feature": feat,
                "ic": self.ic_matrix.loc[feat, condition],
                "status": self.status_matrix.loc[feat, condition],
                "bucket": self._feature_bucket(feat),
            })

        return pd.DataFrame(rows).sort_values("ic", key=abs, ascending=False).reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """Summary table: n_features per bucket, detection flag, adaptive thresholds."""
        n_cs = sum(len(v) for v in self.condition_specific.values())
        rows = [
            {"metric": "total_features", "value": len(self.ic_matrix.index)},
            {"metric": "global_stable", "value": len(self.global_stable)},
            {"metric": "condition_specific", "value": n_cs},
            {"metric": "conditional_ensemble", "value": len(self.conditional_ensemble)},
            {"metric": "unclassified", "value": len(self.unclassified)},
            {"metric": "conditional_structure_detected", "value": self.conditional_structure_detected},
            {"metric": "ic_threshold", "value": self.ic_threshold},
            {"metric": "stability_threshold", "value": self.stability_threshold},
            {"metric": "n_conditions", "value": len(self.ic_matrix.columns)},
            {"metric": "n_skipped_conditions", "value": len(self.skipped_conditions)},
            {"metric": "condition_pass", "value": self.condition_pass},
        ]
        return pd.DataFrame(rows)

    def to_mask(self) -> pd.DataFrame:
        """Binary mask (features x conditions): 1 if Active, 0 otherwise."""
        return (self.status_matrix == FeatureStatus.ACTIVE).astype(int)

    def to_signed_mask(self) -> pd.DataFrame:
        """Signed mask: +1 if Active, -1 if Negative, 0 otherwise."""
        mask = pd.DataFrame(0, index=self.status_matrix.index, columns=self.status_matrix.columns)
        mask[self.status_matrix == FeatureStatus.ACTIVE] = 1
        mask[self.status_matrix == FeatureStatus.NEGATIVE] = -1
        return mask

    def _feature_bucket(self, feature: str) -> str:
        """Return the bucket name for a feature."""
        lookup = self._bucket_lookup()
        return lookup.get(feature, FeatureBucket.UNCLASSIFIED)

    def _bucket_lookup(self) -> Dict[str, str]:
        """Build (and cache) a feature -> bucket lookup dict."""
        if not hasattr(self, "_cached_bucket_lookup"):
            lookup: Dict[str, str] = {}
            for feat in self.global_stable:
                lookup[feat] = FeatureBucket.GLOBAL_STABLE
            for feats in self.condition_specific.values():
                for feat in feats:
                    lookup[feat] = FeatureBucket.CONDITION_SPECIFIC
            for feat in self.conditional_ensemble:
                lookup[feat] = FeatureBucket.CONDITIONAL_ENSEMBLE
            for feat in self.unclassified:
                lookup[feat] = FeatureBucket.UNCLASSIFIED
            self._cached_bucket_lookup = lookup
        return self._cached_bucket_lookup

    # -- Persistence --

    def save(self, output_dir: str) -> Path:
        """
        Persist the map to a directory.

        Layout:
            {output_dir}/
            |- ic_matrix.parquet
            |- pvalue_matrix.parquet
            |- status_matrix.parquet
            |- subperiod_ic_std.parquet
            |- series.parquet           # global_ic, stability_scores as columns
            |- condition_counts.parquet # condition_sample_counts
            |- metadata.json           # partition, thresholds, config, skipped_conditions
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # DataFrames
        self.ic_matrix.to_parquet(out / "ic_matrix.parquet")
        self.pvalue_matrix.to_parquet(out / "pvalue_matrix.parquet")

        # Status matrix: convert enum values to strings for parquet
        status_str = self.status_matrix.map(lambda x: x.value if isinstance(x, FeatureStatus) else str(x))
        status_str.to_parquet(out / "status_matrix.parquet")

        self.subperiod_ic_std.to_parquet(out / "subperiod_ic_std.parquet")

        # Series bundled into one DataFrame
        series_df = pd.DataFrame({
            "global_ic": self.global_ic,
            "stability_scores": self.stability_scores,
        })
        # Add condition_sample_counts as a separate row-indexed series
        counts_df = self.condition_sample_counts.to_frame(name="condition_sample_counts")
        series_df.to_parquet(out / "series.parquet")
        counts_df.to_parquet(out / "condition_counts.parquet")

        # Metadata as JSON
        metadata = {
            "global_stable": self.global_stable,
            "condition_specific": self.condition_specific,
            "conditional_ensemble": self.conditional_ensemble,
            "unclassified": self.unclassified,
            "ic_threshold": self.ic_threshold,
            "stability_threshold": self.stability_threshold,
            "condition_pass": self.condition_pass.value,
            "conditional_structure_detected": self.conditional_structure_detected,
            "min_observations": self.min_observations,
            "max_pvalue": self.max_pvalue,
            "dominance_ratio": self.dominance_ratio,
            "subperiod_instability_ratio": self.subperiod_instability_ratio,
            "skipped_conditions": self.skipped_conditions,
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return out

    @classmethod
    def load(cls, output_dir: str) -> FeatureConditionMap:
        """
        Reconstruct a FeatureConditionMap from a saved directory.

        Raises FileNotFoundError if the directory or required files are missing.
        """
        out = Path(output_dir)
        if not out.exists():
            raise FileNotFoundError(f"Directory not found: {output_dir}")

        ic_matrix = pd.read_parquet(out / "ic_matrix.parquet")
        pvalue_matrix = pd.read_parquet(out / "pvalue_matrix.parquet")

        status_str = pd.read_parquet(out / "status_matrix.parquet")
        status_matrix = status_str.map(lambda x: FeatureStatus(x))

        subperiod_ic_std = pd.read_parquet(out / "subperiod_ic_std.parquet")

        series_df = pd.read_parquet(out / "series.parquet")
        global_ic = series_df["global_ic"]
        global_ic.name = "global_ic"
        stability_scores = series_df["stability_scores"]
        stability_scores.name = "stability_scores"

        counts_df = pd.read_parquet(out / "condition_counts.parquet")
        condition_sample_counts = counts_df["condition_sample_counts"]

        with open(out / "metadata.json") as f:
            metadata = json.load(f)

        return cls(
            ic_matrix=ic_matrix,
            pvalue_matrix=pvalue_matrix,
            status_matrix=status_matrix,
            global_ic=global_ic,
            stability_scores=stability_scores,
            global_stable=metadata["global_stable"],
            condition_specific=metadata["condition_specific"],
            conditional_ensemble=metadata["conditional_ensemble"],
            unclassified=metadata["unclassified"],
            ic_threshold=metadata["ic_threshold"],
            stability_threshold=metadata["stability_threshold"],
            condition_pass=ConditionPass(metadata["condition_pass"]),
            conditional_structure_detected=metadata["conditional_structure_detected"],
            min_observations=metadata["min_observations"],
            max_pvalue=metadata["max_pvalue"],
            dominance_ratio=metadata["dominance_ratio"],
            subperiod_instability_ratio=metadata["subperiod_instability_ratio"],
            subperiod_ic_std=subperiod_ic_std,
            condition_sample_counts=condition_sample_counts,
            skipped_conditions=metadata.get("skipped_conditions", []),
        )

    def __repr__(self) -> str:
        n_cs = sum(len(v) for v in self.condition_specific.values())
        return (
            f"FeatureConditionMap(pass={self.condition_pass.value}, "
            f"global_stable={len(self.global_stable)}, "
            f"condition_specific={n_cs}, "
            f"conditional_ensemble={len(self.conditional_ensemble)}, "
            f"unclassified={len(self.unclassified)}, "
            f"conditional_structure={'yes' if self.conditional_structure_detected else 'no'})"
        )