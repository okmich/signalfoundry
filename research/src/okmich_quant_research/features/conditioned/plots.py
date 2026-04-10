"""Diagnostic visualisations for FeatureConditionMap."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ._enums import ConditionPass, FeatureBucket, FeatureStatus
from ._result import FeatureConditionMap

# -- Colour Palette (Project Standard) --
POSITIVE = "#06A77D"
NEGATIVE = "#D62828"
PRIMARY = "#2E86AB"
SECONDARY = "#F77F00"
GRAY = "#6C757D"
BG = "#F8F9FA"

_BUCKET_COLOURS = {
    FeatureBucket.GLOBAL_STABLE: POSITIVE,
    FeatureBucket.CONDITION_SPECIFIC: SECONDARY,
    FeatureBucket.CONDITIONAL_ENSEMBLE: PRIMARY,
    FeatureBucket.UNCLASSIFIED: GRAY,
}

_STATUS_COLOURS = {
    FeatureStatus.ACTIVE: POSITIVE,
    FeatureStatus.INACTIVE: GRAY,
    FeatureStatus.NEGATIVE: NEGATIVE,
    FeatureStatus.INSUFFICIENT_DATA: "#FFFFFF",
}


def _save_or_show(fig, save_path: Optional[str] = None, show: bool = False) -> None:
    """Save figure if save_path given, optionally show, otherwise close."""
    import matplotlib.pyplot as plt

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


class ConditionedFeatureVisualizer:
    """Static matplotlib/seaborn plots for FeatureConditionMap diagnostics."""

    @staticmethod
    def plot_ic_heatmap(fcm: FeatureConditionMap, figsize: Tuple[int, int] = (16, 10), save_path: Optional[str] = None) -> "Figure":
        """
        Annotated heatmap of the IC matrix (features x conditions).
        Colour scale: diverging RdBu_r centred at 0.
        Features sorted by stability score (most stable at top).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Sort features by stability score
        order = fcm.stability_scores.sort_values(ascending=False).index
        order = [f for f in order if f in fcm.ic_matrix.index]
        data = fcm.ic_matrix.loc[order]

        # Determine symmetric colour range
        vmax = max(abs(data.min().min()), abs(data.max().max()))
        if np.isnan(vmax) or vmax == 0:
            vmax = 0.1

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        sns.heatmap(data, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax, linewidths=0.5, linecolor="white", mask=data.isna())

        ax.set_title(f"IC Heatmap — {fcm.condition_pass.value} pass", fontsize=14, fontweight="bold")
        ax.set_xlabel("Condition", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_status_grid(fcm: FeatureConditionMap, figsize: Tuple[int, int] = (14, 10), save_path: Optional[str] = None) -> "Figure":
        """
        Categorical heatmap of the status matrix.
        Features grouped by bucket.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        # Order features by bucket
        bucket_order = []
        bucket_order.extend(fcm.global_stable)
        for cond_feats in fcm.condition_specific.values():
            bucket_order.extend(cond_feats)
        bucket_order.extend(fcm.conditional_ensemble)
        bucket_order.extend(fcm.unclassified)

        status_map = {FeatureStatus.ACTIVE: 3, FeatureStatus.NEGATIVE: 2, FeatureStatus.INACTIVE: 1, FeatureStatus.INSUFFICIENT_DATA: 0}
        numeric = fcm.status_matrix.loc[bucket_order].map(lambda x: status_map.get(x, 0) if isinstance(x, FeatureStatus) else status_map.get(FeatureStatus(x), 0))

        cmap = ListedColormap(["#FFFFFF", GRAY, NEGATIVE, POSITIVE])

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)
        ax.imshow(numeric.values, aspect="auto", cmap=cmap, vmin=0, vmax=3)

        ax.set_xticks(range(len(numeric.columns)))
        ax.set_xticklabels(numeric.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(numeric.index)))
        ax.set_yticklabels(numeric.index, fontsize=8)

        legend_elements = [
            Patch(facecolor=POSITIVE, label="Active"),
            Patch(facecolor=NEGATIVE, label="Negative"),
            Patch(facecolor=GRAY, label="Inactive"),
            Patch(facecolor="#FFFFFF", edgecolor="black", label="Insufficient Data"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(f"Status Grid — {fcm.condition_pass.value} pass", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_partition_summary(fcm: FeatureConditionMap, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None) -> "Figure":
        """
        Horizontal stacked bar showing feature counts per bucket.
        Annotates adaptive thresholds and detection flag.
        """
        import matplotlib.pyplot as plt

        n_cs = sum(len(v) for v in fcm.condition_specific.values())
        buckets = ["global_stable", "condition_specific", "conditional_ensemble", "unclassified"]
        counts = [len(fcm.global_stable), n_cs, len(fcm.conditional_ensemble), len(fcm.unclassified)]
        colours = [POSITIVE, SECONDARY, PRIMARY, GRAY]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        left = 0
        for bucket, count, colour in zip(buckets, counts, colours):
            ax.barh(0, count, left=left, color=colour, edgecolor="white", label=f"{bucket} ({count})")
            if count > 0:
                ax.text(left + count / 2, 0, str(count), ha="center", va="center", fontweight="bold", fontsize=12, color="white")
            left += count

        ax.set_yticks([])
        ax.set_xlabel("Number of Features", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)

        detected_str = "YES" if fcm.conditional_structure_detected else "NO"
        ax.set_title(
            f"Feature Partition — {fcm.condition_pass.value} pass\n"
            f"ic_threshold={fcm.ic_threshold:.4f}, stability_threshold={fcm.stability_threshold:.4f}, "
            f"conditional_structure={detected_str}",
            fontsize=13, fontweight="bold",
        )

        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_stability_distribution(fcm: FeatureConditionMap, figsize: Tuple[int, int] = (12, 6), save_path: Optional[str] = None) -> "Figure":
        """
        Histogram of stability_scores, coloured by bucket assignment.
        Vertical line at stability_threshold.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        for bucket_name, colour in [(FeatureBucket.GLOBAL_STABLE, POSITIVE), (FeatureBucket.CONDITION_SPECIFIC, SECONDARY), (FeatureBucket.CONDITIONAL_ENSEMBLE, PRIMARY), (FeatureBucket.UNCLASSIFIED, GRAY)]:
            if bucket_name == FeatureBucket.GLOBAL_STABLE:
                feats = fcm.global_stable
            elif bucket_name == FeatureBucket.CONDITION_SPECIFIC:
                feats = [f for fs in fcm.condition_specific.values() for f in fs]
            elif bucket_name == FeatureBucket.CONDITIONAL_ENSEMBLE:
                feats = fcm.conditional_ensemble
            else:
                feats = fcm.unclassified

            vals = fcm.stability_scores.reindex(feats).dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.6, color=colour, label=bucket_name.value, edgecolor="white")

        ax.axvline(fcm.stability_threshold, color="black", linestyle="--", linewidth=2, label=f"threshold={fcm.stability_threshold:.3f}")
        ax.set_xlabel("Stability Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Stability Score Distribution — {fcm.condition_pass.value} pass", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_feature_profile(fcm: FeatureConditionMap, feature: str, figsize: Tuple[int, int] = (14, 5), save_path: Optional[str] = None) -> "Figure":
        """
        Bar chart of IC across conditions for a single feature.
        Bars coloured by status. Error bars from subperiod_ic_std.
        """
        import matplotlib.pyplot as plt

        profile = fcm.feature_profile(feature)
        conditions = profile["condition"].tolist()
        ics = profile["ic"].values
        statuses = profile["status"].tolist()
        sp_std = profile["subperiod_ic_std"].values

        colours = [_STATUS_COLOURS.get(FeatureStatus(s) if not isinstance(s, FeatureStatus) else s, GRAY) for s in statuses]
        yerr = np.where(np.isnan(sp_std), 0, sp_std)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        bars = ax.bar(range(len(conditions)), ics, color=colours, edgecolor="white", yerr=yerr, capsize=3)
        ax.axhline(profile["global_ic"].iloc[0], color=PRIMARY, linestyle="--", linewidth=1.5, label=f"global IC={profile['global_ic'].iloc[0]:.4f}")
        ax.axhline(0, color="black", linewidth=0.5)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Spearman IC", fontsize=11)

        bucket = profile["bucket"].iloc[0]
        stab = profile["stability_score"].iloc[0]
        sign_flip = profile["sign_flipping"].iloc[0]
        title = f"{feature} — bucket={bucket}, stability={stab:.3f}"
        if sign_flip:
            title += " [SIGN-FLIPPING]"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_condition_profile(fcm: FeatureConditionMap, condition: str, figsize: Tuple[int, int] = (14, 8), save_path: Optional[str] = None) -> "Figure":
        """
        Horizontal bar chart of IC for all features within a single condition.
        Sorted by |IC| descending. Bars coloured by status.
        """
        import matplotlib.pyplot as plt

        profile = fcm.condition_profile(condition)
        features = profile["feature"].tolist()
        ics = profile["ic"].values
        statuses = profile["status"].tolist()

        colours = [_STATUS_COLOURS.get(FeatureStatus(s) if not isinstance(s, FeatureStatus) else s, GRAY) for s in statuses]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        y_pos = range(len(features))
        ax.barh(y_pos, ics, color=colours, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        n_obs = fcm.condition_sample_counts.get(condition, "?")
        ax.set_xlabel("Spearman IC", fontsize=11)
        ax.set_title(f"Condition: {condition} (n={n_obs}) — {fcm.condition_pass.value} pass", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_global_vs_conditional(fcm: FeatureConditionMap, condition: str, figsize: Tuple[int, int] = (10, 10), save_path: Optional[str] = None) -> "Figure":
        """
        Scatter: x = global IC, y = conditional IC for a given condition.
        Diagonal line (y = x). Points coloured by bucket.
        """
        import matplotlib.pyplot as plt

        if condition not in fcm.ic_matrix.columns:
            raise KeyError(f"Condition '{condition}' not found.")

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        for feat in fcm.ic_matrix.index:
            x = fcm.global_ic.get(feat, np.nan)
            y = fcm.ic_matrix.loc[feat, condition]
            if np.isnan(x) or np.isnan(y):
                continue
            bucket = fcm._feature_bucket(feat)
            colour = _BUCKET_COLOURS.get(FeatureBucket(bucket), GRAY)
            ax.scatter(x, y, color=colour, s=40, alpha=0.7, edgecolors="white", linewidth=0.5)

        # Diagonal
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel("Global IC", fontsize=11)
        ax.set_ylabel(f"IC in '{condition}'", fontsize=11)
        ax.set_title(f"Global vs Conditional IC — {condition}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=b.value) for b, c in _BUCKET_COLOURS.items()]
        ax.legend(handles=legend_elements, fontsize=9)

        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_subperiod_stability(fcm: FeatureConditionMap, figsize: Tuple[int, int] = (16, 10), save_path: Optional[str] = None) -> "Figure":
        """
        Heatmap of subperiod_ic_std (features x conditions).
        Sequential colour scale (low std = good, high std = bad).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = fcm.subperiod_ic_std

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5, linecolor="white", mask=data.isna())

        ax.set_title(f"Sub-Period IC Std — {fcm.condition_pass.value} pass", fontsize=14, fontweight="bold")
        ax.set_xlabel("Condition", fontsize=11)
        ax.set_ylabel("Feature", fontsize=11)
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_condition_counts(fcm: FeatureConditionMap, figsize: Tuple[int, int] = (12, 5), save_path: Optional[str] = None) -> "Figure":
        """
        Bar chart of n_observations per condition.
        Horizontal line at min_observations threshold.
        """
        import matplotlib.pyplot as plt

        counts = fcm.condition_sample_counts.sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        colours = [NEGATIVE if c < fcm.min_observations else PRIMARY for c in counts.values]
        ax.bar(range(len(counts)), counts.values, color=colours, edgecolor="white")
        ax.axhline(fcm.min_observations, color="black", linestyle="--", linewidth=1.5, label=f"min_observations={fcm.min_observations}")

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Observations", fontsize=11)
        ax.set_title(f"Condition Sample Counts — {fcm.condition_pass.value} pass", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def plot_pass_comparison(maps: Dict[ConditionPass, FeatureConditionMap], figsize: Tuple[int, int] = (14, 8), save_path: Optional[str] = None) -> "Figure":
        """
        Side-by-side comparison of partition summaries across passes.
        Grouped bars showing bucket proportions.
        """
        import matplotlib.pyplot as plt

        passes = list(maps.keys())
        bucket_names = ["global_stable", "condition_specific", "conditional_ensemble", "unclassified"]
        colours = [POSITIVE, SECONDARY, PRIMARY, GRAY]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(BG)

        x = np.arange(len(passes))
        width = 0.2
        offsets = np.arange(len(bucket_names)) - (len(bucket_names) - 1) / 2

        for k, (bucket, colour) in enumerate(zip(bucket_names, colours)):
            vals = []
            for p in passes:
                m = maps[p]
                if bucket == "global_stable":
                    vals.append(len(m.global_stable))
                elif bucket == "condition_specific":
                    vals.append(sum(len(v) for v in m.condition_specific.values()))
                elif bucket == "conditional_ensemble":
                    vals.append(len(m.conditional_ensemble))
                else:
                    vals.append(len(m.unclassified))

            ax.bar(x + offsets[k] * width, vals, width, color=colour, label=bucket, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([p.value for p in passes], fontsize=11)
        ax.set_ylabel("Number of Features", fontsize=11)
        ax.set_title("Feature Partition Across Passes", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    @staticmethod
    def create_report(fcm: FeatureConditionMap, output_dir: str = "output/conditioned_reports", show: bool = True) -> Dict[str, str]:
        """
        Generate a full diagnostic report: all plots saved as PNG + summary CSV.

        Returns dict of {plot_name: save_path}.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = fcm.condition_pass.value

        paths: Dict[str, str] = {}

        plot_methods = [
            ("ic_heatmap", ConditionedFeatureVisualizer.plot_ic_heatmap),
            ("status_grid", ConditionedFeatureVisualizer.plot_status_grid),
            ("partition_summary", ConditionedFeatureVisualizer.plot_partition_summary),
            ("stability_distribution", ConditionedFeatureVisualizer.plot_stability_distribution),
            ("subperiod_stability", ConditionedFeatureVisualizer.plot_subperiod_stability),
            ("condition_counts", ConditionedFeatureVisualizer.plot_condition_counts),
        ]

        for name, method in plot_methods:
            sp = str(out / f"{prefix}_{name}.png")
            method(fcm, save_path=sp, show=show)
            paths[name] = sp

        # Summary CSV
        summary_path = str(out / f"{prefix}_summary.csv")
        fcm.summary().to_csv(summary_path, index=False)
        paths["summary"] = summary_path

        return paths