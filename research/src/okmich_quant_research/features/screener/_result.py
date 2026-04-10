from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class StageReport:
    """Record of what one stage kept and removed."""
    stage: str
    n_before: int
    n_after: int
    removed: List[str] = field(default_factory=list)

    @property
    def n_removed(self) -> int:
        return self.n_before - self.n_after

    def __repr__(self) -> str:
        return (
            f"StageReport({self.stage}: {self.n_before} -> {self.n_after}, "
            f"removed {self.n_removed})"
        )


@dataclass
class ScreenerResult:
    """
    Output of FeatureScreener.screen_for_regimes() or screen_for_returns().

    Attributes
    ----------
    confirmed : list[str]
        Features Boruta confirmed as better than random.
    tentative : list[str]
        Features Boruta could not decide — include with domain judgment.
    rejected : list[str]
        Features removed at some stage.
    shap_rank : pd.Series
        Mean |SHAP| importance, descending (confirmed + tentative features).
    mda_rank : pd.Series
        Mean Decrease in Accuracy (permutation importance), descending.
    stage_reports : list[StageReport]
        Per-stage reduction record.
    icir_scores : dict[str, float]
        IC-IR computed at Stage 2 for each surviving feature.
    """
    confirmed: List[str]
    tentative: List[str]
    rejected: List[str]
    shap_rank: pd.Series
    mda_rank: pd.Series
    stage_reports: List[StageReport] = field(default_factory=list)
    icir_scores: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising the stage-by-stage reduction."""
        rows = []
        for r in self.stage_reports:
            rows.append({
                "stage":     r.stage,
                "n_before":  r.n_before,
                "n_after":   r.n_after,
                "n_removed": r.n_removed,
            })
        return pd.DataFrame(rows)

    def top_features(self, n: int = 15) -> pd.DataFrame:
        """
        Return a combined ranking of confirmed features.

        Columns: shap_rank, mda_rank, boruta_status
        """
        all_features = set(self.confirmed) | set(self.tentative)
        rows = []
        for feat in all_features:
            rows.append({
                "feature":      feat,
                "shap":         self.shap_rank.get(feat, float("nan")),
                "mda":          self.mda_rank.get(feat, float("nan")),
                "boruta_status": "confirmed" if feat in self.confirmed else "tentative",
            })
        if not rows:
            return pd.DataFrame(columns=["feature", "shap", "mda", "boruta_status"])
        df = pd.DataFrame(rows)
        return df.sort_values("shap", ascending=False, na_position="last").reset_index(drop=True).head(n)

    def audit(self, s1_scores: dict | None = None, boruta_groups: dict | None = None) -> pd.DataFrame:
        """
        Full feature journey audit — one row per feature, showing which stage eliminated it and all available scores.

        Useful for informed overrides: if you want to add a rejected feature back, the audit tells you exactly why it
        was removed and how close it was to surviving.

        Parameters
        ----------
        s1_scores : dict or None
            Stage 1 score dict returned by stage1_regime/stage1_return
            e.g. ``{"mi": pd.Series, "ks": pd.Series}`` or
            ``{"mi": pd.Series, "dcor": pd.Series}``.
            Pass this to include Stage 1 scores in the audit.
        boruta_groups : dict or None
            ``{"confirmed": [...], "tentative": [...], "rejected": [...]}``
            returned by stage4_boruta. Pass to include Boruta decision.

        Returns
        -------
        pd.DataFrame
            Columns: feature, eliminated_at, icir, shap, mda,
            boruta_status, + any Stage 1 score columns provided.
            Sorted: confirmed first, then tentative, then by elimination stage.
        """
        # Build elimination map: feature -> stage name
        eliminated_at: dict[str, str] = {}
        for report in self.stage_reports:
            for feat in report.removed:
                if feat not in eliminated_at:
                    eliminated_at[feat] = report.stage

        all_features = (
            set(self.confirmed)
            | set(self.tentative)
            | set(self.rejected)
            | set(eliminated_at.keys())
        )

        rows = []
        for feat in all_features:
            if feat in self.confirmed:
                status = "confirmed"
                stage  = "survived"
            elif feat in self.tentative:
                status = "tentative"
                stage  = "survived"
            else:
                status = "rejected"
                stage  = eliminated_at.get(feat, "unknown")

            row: dict = {
                "feature":      feat,
                "status":       status,
                "eliminated_at": stage,
                "icir":         self.icir_scores.get(feat, float("nan")),
                "shap":         self.shap_rank.get(feat, float("nan")),
                "mda":          self.mda_rank.get(feat, float("nan")),
            }

            # Stage 1 scores if provided
            if s1_scores:
                for score_name, score_series in s1_scores.items():
                    if isinstance(score_series, pd.Series):
                        row[f"s1_{score_name}"] = score_series.get(feat, float("nan"))

            # Boruta decision if provided
            if boruta_groups:
                if feat in boruta_groups.get("confirmed", []):
                    row["boruta"] = "confirmed"
                elif feat in boruta_groups.get("tentative", []):
                    row["boruta"] = "tentative"
                elif feat in boruta_groups.get("rejected", []):
                    row["boruta"] = "rejected"
                else:
                    row["boruta"] = "not_reached"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort: survived confirmed → survived tentative → rejected (by stage order)
        stage_order = {r.stage: i for i, r in enumerate(self.stage_reports)}
        stage_order["survived"] = len(self.stage_reports)
        stage_order["unknown"]  = len(self.stage_reports) + 1

        status_order = {"confirmed": 0, "tentative": 1, "rejected": 2}
        df["_status_rank"] = df["status"].map(status_order)
        df["_stage_rank"]  = df["eliminated_at"].map(lambda s: stage_order.get(s, 99))
        df = (df.sort_values(["_status_rank", "_stage_rank", "icir"], ascending=[True, True, False])
                .drop(columns=["_status_rank", "_stage_rank"])
                .reset_index(drop=True))

        return df

    def __repr__(self) -> str:
        return (
            f"ScreenerResult(confirmed={len(self.confirmed)}, "
            f"tentative={len(self.tentative)}, rejected={len(self.rejected)})"
        )