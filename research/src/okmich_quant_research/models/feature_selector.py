from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from okmich_quant_research.eda import FeatureEDA


class FeatureSelectionResult:

    def __init__(self, selected_features: List[str], feature_scores: pd.DataFrame,
                 vif_scores: Optional[pd.DataFrame] = None, removed_features: Optional[Dict[str, str]] = None,
                 transformations: Optional[Dict[str, str]] = None, eda_results: Optional[Dict[str, Any]] = None):
        self.selected_features = selected_features
        self.feature_scores = feature_scores
        self.vif_scores = vif_scores
        self.removed_features = removed_features or {}
        self.transformations = transformations or {}
        self.eda_results = eda_results or {}

    def save(self, output_dir: str):
        """Save selection results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save selected features list
        import json

        with open(output_dir / "selected_features.json", "w") as f:
            json.dump(
                {
                    "selected_features": self.selected_features,
                    "n_selected": len(self.selected_features),
                    "removed_features": self.removed_features,
                    "transformations": self.transformations,
                },
                f,
                indent=2,
            )

        # Save scores
        if self.feature_scores is not None:
            self.feature_scores.to_csv(output_dir / "feature_scores.csv", index=False)

        if self.vif_scores is not None:
            self.vif_scores.to_csv(output_dir / "vif_scores.csv", index=False)

    def __repr__(self):
        return (
            f"FeatureSelectionResult(\n"
            f"  selected_features={len(self.selected_features)},\n"
            f"  removed_features={len(self.removed_features)}\n"
            f")"
        )


class FeatureSelector:
    """
    Automated feature selection using FeatureEDA.

    Selects features based on:
    1. Relevance to target
    2. Low multicollinearity (VIF)
    3. Distribution quality
    """

    def __init__(self, features: pd.DataFrame, target: pd.Series, target_type: str = "continuous"):
        self.features = features.copy()
        self.target = target.copy()
        self.target_type = target_type

        # Initialize FeatureEDA
        self.eda = FeatureEDA(features=self.features, target=self.target, target_type=self.target_type)

    def select(self, top_n: int = 15, vif_threshold: float = 10.0, min_importance: float = 0.01,
               correlation_threshold: float = 0.95, methods: Optional[List[str]] = None, save_eda_artifacts: bool = False,
               output_dir: Optional[str] = None) -> FeatureSelectionResult:
        print("=" * 80)
        print("AUTOMATED FEATURE SELECTION")
        print("=" * 80)

        if methods is None:
            methods = ["correlation", "mutual_info", "rf", "vif"]

        # Step 1: Feature relevance analysis
        print("\n[1/4] Analyzing feature relevance...")
        relevance_df = self.eda.analyze_feature_relevance(n_top=top_n * 2)

        # Step 2: Model-based importance (if requested)
        importance_df = None
        if "rf" in methods:
            print("\n[2/4] Computing RF importance...")
            importance_df = self.eda.analyze_model_based_importance(
                model_type="rf", n_top=top_n * 2
            )
        else:
            print("\n[2/4] Skipping RF importance (not in methods)")

        # Step 3: VIF analysis
        vif_df = None
        if "vif" in methods:
            print("\n[3/4] Computing VIF scores...")
            vif_df = self.eda.compute_vif(vif_threshold=vif_threshold)
        else:
            print("\n[3/4] Skipping VIF analysis (not in methods)")

        # Step 4: Correlation analysis
        print("\n[4/4] Analyzing correlations...")
        corr_matrix, high_corr_pairs = self.eda.analyze_correlation(
            threshold=correlation_threshold
        )

        # Selection logic
        print("\n" + "=" * 80)
        print("FEATURE SELECTION")
        print("=" * 80)

        removed_features = {}
        candidate_features = set(self.features.columns)

        # Remove features with high VIF
        if vif_df is not None:
            high_vif = vif_df[vif_df["vif"] > vif_threshold]["feature"].tolist()
            for feat in high_vif:
                if feat in candidate_features:
                    removed_features[feat] = (
                        f"High VIF ({vif_df[vif_df['feature'] == feat]['vif'].values[0]:.2f} > {vif_threshold})"
                    )
                    candidate_features.remove(feat)
            print(f"Removed {len(high_vif)} features due to high VIF")

        # Remove features with high correlation (keep higher importance one)
        if high_corr_pairs:
            for feat1, feat2, corr in high_corr_pairs:
                if feat1 in candidate_features and feat2 in candidate_features:
                    # Keep feature with higher relevance
                    if self.target_type == "continuous":
                        score_col = "abs_pearson"
                    else:
                        score_col = "mutual_info"

                    score1 = relevance_df[relevance_df["feature"] == feat1][
                        score_col
                    ].values
                    score2 = relevance_df[relevance_df["feature"] == feat2][
                        score_col
                    ].values

                    if len(score1) > 0 and len(score2) > 0:
                        if score1[0] < score2[0]:
                            removed_features[feat1] = (
                                f"High correlation with {feat2} ({corr:.3f})"
                            )
                            if feat1 in candidate_features:
                                candidate_features.remove(feat1)
                        else:
                            removed_features[feat2] = (
                                f"High correlation with {feat1} ({corr:.3f})"
                            )
                            if feat2 in candidate_features:
                                candidate_features.remove(feat2)
            print(
                f"Removed features due to high correlation: "
                f"{len([r for r in removed_features.values() if 'correlation' in r])}"
            )

        # Rank remaining features by relevance
        candidate_df = relevance_df[relevance_df["feature"].isin(candidate_features)]

        if self.target_type == "continuous":
            candidate_df = candidate_df.sort_values("abs_pearson", ascending=False)
        else:
            candidate_df = candidate_df.sort_values("mutual_info", ascending=False)

        # Select top N
        selected_features = candidate_df.head(top_n)["feature"].tolist()

        print(
            f"\nSelected {len(selected_features)} features from {len(self.features.columns)} original"
        )
        print(f"Removed {len(removed_features)} features")

        # Get transformation recommendations
        print("\nRecommending transformations...")
        transformation_df = self.eda.recommend_transformations()
        transformations = {}
        for _, row in transformation_df.iterrows():
            if row["feature"] in selected_features:
                if row["transformations"] != "none":
                    transformations[row["feature"]] = row["transformations"]

        # Save EDA artifacts if requested
        if save_eda_artifacts:
            if output_dir is None:
                raise ValueError("output_dir required when save_eda_artifacts=True")

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nSaving EDA artifacts to: {output_dir}")

            # Save plots
            try:
                fig_dist = self.eda.plot_distributions(
                    features_to_plot=selected_features[:12], figsize=(20, 15)
                )
                fig_dist.savefig(
                    output_dir / "distributions.png", dpi=100, bbox_inches="tight"
                )
                print("  - Saved distributions.png")
            except Exception as e:
                print(f"  - Warning: Could not save distributions plot: {e}")

            try:
                fig_qq = self.eda.plot_qq_plots(
                    features_to_plot=selected_features[:12], figsize=(20, 15)
                )
                fig_qq.savefig(
                    output_dir / "qq_plots.png", dpi=100, bbox_inches="tight"
                )
                print("  - Saved qq_plots.png")
            except Exception as e:
                print(f"  - Warning: Could not save QQ plots: {e}")

            try:
                fig_corr = self.eda.plot_correlation_matrix(
                    cluster=True, figsize=(16, 14)
                )
                # Note: clustermap returns a ClusterGrid, not a Figure
                if hasattr(fig_corr, "savefig"):
                    fig_corr.savefig(
                        output_dir / "correlation_matrix.png",
                        dpi=100,
                        bbox_inches="tight",
                    )
                else:
                    fig_corr.fig.savefig(
                        output_dir / "correlation_matrix.png",
                        dpi=100,
                        bbox_inches="tight",
                    )
                print("  - Saved correlation_matrix.png")
            except Exception as e:
                print(f"  - Warning: Could not save correlation matrix: {e}")

            # Save CSV results
            relevance_df.to_csv(output_dir / "relevance_scores.csv", index=False)
            if vif_df is not None:
                vif_df.to_csv(output_dir / "vif_scores.csv", index=False)
            transformation_df.to_csv(
                output_dir / "transformation_recommendations.csv", index=False
            )
            print("  - Saved CSV files")

        # Create result object
        result = FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=relevance_df,
            vif_scores=vif_df,
            removed_features=removed_features,
            transformations=transformations,
            eda_results={
                "correlation_matrix": corr_matrix,
                "high_correlation_pairs": high_corr_pairs,
                "transformation_recommendations": transformation_df,
            },
        )
        print("\n" + "=" * 80)
        print("SELECTION COMPLETE")
        print("=" * 80)
        print(f"Selected features: {len(selected_features)}")
        print(f"Features to transform: {len(transformations)}")
        print()
        return result

    def get_eda(self) -> FeatureEDA:
        """Get the underlying FeatureEDA object for advanced usage."""
        return self.eda
