"""
Comprehensive Feature EDA Framework for Quantitative Features

This module provides tools for exploratory data analysis of features generated from the okmich_quant_features package, including:
- Feature relevance analysis
- Distribution analysis and normality tests
- Correlation analysis and multicollinearity detection
- Transformation recommendations (logit, log, box-cox, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import logit, expit
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")


class FeatureEDA:
    """
    Comprehensive EDA framework for quantitative trading features.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing all features
    target : pd.Series or np.ndarray
        Target variable (returns, labels, etc.)
    target_type : str, default='continuous'
        Type of target: 'continuous' (regression) or 'categorical' (classification)
    feature_names : List[str], optional
        Subset of feature names to analyze. If None, analyzes all features.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        target: Union[pd.Series, np.ndarray],
        target_type: str = "continuous",
        feature_names: Optional[List[str]] = None,
    ):

        self.features = features.copy()
        self.target = (
            pd.Series(target) if isinstance(target, np.ndarray) else target.copy()
        )
        self.target_type = target_type

        # Align features and target
        common_idx = self.features.index.intersection(self.target.index)
        self.features = self.features.loc[common_idx]
        self.target = self.target.loc[common_idx]

        # Drop any rows with NaN in target
        valid_idx = self.target.notna()
        self.features = self.features[valid_idx]
        self.target = self.target[valid_idx]

        # Select features to analyze
        if feature_names is not None:
            self.feature_names = [
                f for f in feature_names if f in self.features.columns
            ]
        else:
            self.feature_names = list(self.features.columns)

        # Results storage
        self.relevance_results = {}
        self.distribution_results = {}
        self.correlation_results = {}
        self.transformation_results = {}

    # ============================================================================
    # 1. FEATURE RELEVANCE ANALYSIS
    # ============================================================================

    def analyze_feature_relevance(self, n_top: int = 20) -> pd.DataFrame:
        """
        Comprehensive feature relevance analysis using multiple methods.

        Parameters
        ----------
        n_top : int, default=20
            Number of top features to highlight

        Returns
        -------
        pd.DataFrame
            DataFrame with relevance scores from different methods
        """
        print("=" * 80)
        print("FEATURE RELEVANCE ANALYSIS")
        print("=" * 80)

        results = []

        for feat in self.feature_names:
            feat_data = self.features[feat].copy()

            # Skip features with too many NaNs (>50%)
            if feat_data.isna().sum() / len(feat_data) > 0.5:
                continue

            # Forward fill and drop remaining NaNs
            feat_data = feat_data.ffill().dropna()

            # Remove infinite values
            feat_data = feat_data.replace([np.inf, -np.inf], np.nan).dropna()

            common_idx = feat_data.index.intersection(self.target.index)
            feat_data = feat_data.loc[common_idx]
            target_data = self.target.loc[common_idx]

            if len(feat_data) < 30:  # Need minimum data
                continue

            feat_results = {"feature": feat}

            # 1. Pearson correlation (for continuous target)
            if self.target_type == "continuous":
                corr, p_val = stats.pearsonr(feat_data, target_data)
                feat_results["pearson_corr"] = corr
                feat_results["pearson_pval"] = p_val
                feat_results["abs_pearson"] = abs(corr)

            # 2. Spearman correlation (rank-based, more robust)
            spearman, sp_pval = stats.spearmanr(feat_data, target_data)
            feat_results["spearman_corr"] = spearman
            feat_results["spearman_pval"] = sp_pval
            feat_results["abs_spearman"] = abs(spearman)

            # 3. Mutual Information
            feat_2d = feat_data.values.reshape(-1, 1)
            if self.target_type == "continuous":
                mi = mutual_info_regression(feat_2d, target_data, random_state=42)[0]
            else:
                mi = mutual_info_classif(feat_2d, target_data, random_state=42)[0]
            feat_results["mutual_info"] = mi

            results.append(feat_results)

        results_df = pd.DataFrame(results)

        # Sort by absolute correlation or mutual information
        if self.target_type == "continuous":
            results_df = results_df.sort_values("abs_pearson", ascending=False)
        else:
            results_df = results_df.sort_values("mutual_info", ascending=False)

        self.relevance_results = results_df

        # Print top features
        print(f"\nTop {n_top} Most Relevant Features:")
        print("-" * 80)
        display_cols = [c for c in results_df.columns if c != "feature"]
        print(results_df.head(n_top).to_string(index=False))

        return results_df

    def analyze_model_based_importance(
        self, model_type: str = "rf", n_top: int = 20
    ) -> pd.DataFrame:
        """
        Feature importance using tree-based or linear models.

        Parameters
        ----------
        model_type : str, default='rf'
            'rf' for Random Forest, 'linear' for Linear/Logistic Regression
        n_top : int, default=20
            Number of top features to display

        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        print("\n" + "=" * 80)
        print(f"MODEL-BASED FEATURE IMPORTANCE ({model_type.upper()})")
        print("=" * 80)

        # Prepare data
        X = self.features[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        y = self.target.copy()

        # Train model
        if model_type == "rf":
            if self.target_type == "continuous":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=50,
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=50,
                    random_state=42,
                    n_jobs=-1,
                )
            model.fit(X, y)
            importance = model.feature_importances_

        else:  # linear
            if self.target_type == "continuous":
                model = LinearRegression()
                model.fit(X, y)
                importance = np.abs(model.coef_)
            else:
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X, y)
                importance = np.abs(model.coef_[0])

        # Create results DataFrame
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print(f"\nTop {n_top} Most Important Features:")
        print("-" * 80)
        print(importance_df.head(n_top).to_string(index=False))

        return importance_df

    # ============================================================================
    # 2. DISTRIBUTION ANALYSIS
    # ============================================================================

    def analyze_distributions(self, n_features: int = 10) -> pd.DataFrame:
        """
        Comprehensive distribution analysis for features.

        Parameters
        ----------
        n_features : int, default=10
            Number of features to analyze in detail (top by relevance)

        Returns
        -------
        pd.DataFrame
            Distribution statistics for all features
        """
        print("\n" + "=" * 80)
        print("FEATURE DISTRIBUTION ANALYSIS")
        print("=" * 80)

        results = []

        for feat in self.feature_names:
            feat_data = self.features[feat].dropna()

            # Remove infinite values
            feat_data = feat_data.replace([np.inf, -np.inf], np.nan).dropna()

            if len(feat_data) < 30:
                continue

            feat_stats = {
                "feature": feat,
                "count": len(feat_data),
                "mean": feat_data.mean(),
                "std": feat_data.std(),
                "min": feat_data.min(),
                "max": feat_data.max(),
                "skewness": stats.skew(feat_data),
                "kurtosis": stats.kurtosis(feat_data),
            }

            # Normality tests
            if len(feat_data) >= 30:
                # Shapiro-Wilk (good for small-medium samples)
                if len(feat_data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(feat_data)
                    feat_stats["shapiro_stat"] = shapiro_stat
                    feat_stats["shapiro_pval"] = shapiro_p
                    feat_stats["is_normal_shapiro"] = shapiro_p > 0.05

                # Jarque-Bera (based on skewness and kurtosis)
                jb_stat, jb_p = stats.jarque_bera(feat_data)
                feat_stats["jarque_bera_stat"] = jb_stat
                feat_stats["jarque_bera_pval"] = jb_p
                feat_stats["is_normal_jb"] = jb_p > 0.05

            # Check if bounded (like ratios, oscillators)
            feat_stats["is_bounded_0_1"] = (feat_data.min() >= 0) and (
                feat_data.max() <= 1
            )
            feat_stats["is_strictly_positive"] = feat_data.min() > 0

            results.append(feat_stats)

        dist_df = pd.DataFrame(results)
        self.distribution_results = dist_df

        print("\nDistribution Summary:")
        print("-" * 80)
        print(f"Total features analyzed: {len(dist_df)}")

        if "is_normal_jb" in dist_df.columns:
            n_normal = dist_df["is_normal_jb"].sum()
            print(
                f"Normal distributions (Jarque-Bera): {n_normal} ({n_normal/len(dist_df)*100:.1f}%)"
            )

        if "is_bounded_0_1" in dist_df.columns:
            n_bounded = dist_df["is_bounded_0_1"].sum()
            print(f"Bounded [0,1] features: {n_bounded}")

        if "is_strictly_positive" in dist_df.columns:
            n_positive = dist_df["is_strictly_positive"].sum()
            print(f"Strictly positive features: {n_positive}")

        # Show features with extreme skewness or kurtosis
        print("\nFeatures with High Skewness (|skew| > 2):")
        high_skew = dist_df[dist_df["skewness"].abs() > 2].sort_values(
            "skewness", key=abs, ascending=False
        )
        if len(high_skew) > 0:
            print(
                high_skew[["feature", "skewness", "kurtosis"]]
                .head(10)
                .to_string(index=False)
            )
        else:
            print("None")

        print("\nFeatures with High Kurtosis (|kurtosis| > 3):")
        high_kurt = dist_df[dist_df["kurtosis"].abs() > 3].sort_values(
            "kurtosis", key=abs, ascending=False
        )
        if len(high_kurt) > 0:
            print(
                high_kurt[["feature", "skewness", "kurtosis"]]
                .head(10)
                .to_string(index=False)
            )
        else:
            print("None")

        return dist_df

    def plot_distributions(
        self,
        features_to_plot: Optional[List[str]] = None,
        n_features: int = 12,
        figsize: Tuple[int, int] = (20, 15),
    ):
        """
        Plot distribution histograms with KDE and Q-Q plots.

        Parameters
        ----------
        features_to_plot : List[str], optional
            Specific features to plot. If None, plots top N by relevance
        n_features : int, default=12
            Number of features to plot if features_to_plot is None
        figsize : Tuple[int, int], default=(20, 15)
            Figure size
        """
        if features_to_plot is None:
            # Get top features by relevance
            if hasattr(self, "relevance_results") and len(self.relevance_results) > 0:
                features_to_plot = self.relevance_results.head(n_features)[
                    "feature"
                ].tolist()
            else:
                features_to_plot = self.feature_names[:n_features]

        n_plots = len(features_to_plot)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Ensure axes is always a 1D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_1d(axes)
        else:
            axes = axes.flatten()

        for idx, feat in enumerate(features_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]
            feat_data = self.features[feat].dropna()

            # Remove infinite values
            feat_data = feat_data.replace([np.inf, -np.inf], np.nan).dropna()

            # Histogram with KDE
            ax.hist(
                feat_data,
                bins=50,
                density=True,
                alpha=0.6,
                color="steelblue",
                edgecolor="black",
            )

            # KDE overlay
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(feat_data)
                x_range = np.linspace(feat_data.min(), feat_data.max(), 100)
                ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
            except:
                pass

            ax.set_title(
                f"{feat}\nSkew: {stats.skew(feat_data):.2f}, Kurt: {stats.kurtosis(feat_data):.2f}",
                fontsize=10,
            )
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.suptitle("Feature Distributions", fontsize=16, y=1.001)
        return fig

    def plot_qq_plots(
        self,
        features_to_plot: Optional[List[str]] = None,
        n_features: int = 12,
        figsize: Tuple[int, int] = (20, 15),
    ):
        """
        Q-Q plots to assess normality.

        Parameters
        ----------
        features_to_plot : List[str], optional
            Specific features to plot
        n_features : int, default=12
            Number of features to plot if features_to_plot is None
        figsize : Tuple[int, int], default=(20, 15)
            Figure size
        """
        if features_to_plot is None:
            if hasattr(self, "relevance_results") and len(self.relevance_results) > 0:
                features_to_plot = self.relevance_results.head(n_features)[
                    "feature"
                ].tolist()
            else:
                features_to_plot = self.feature_names[:n_features]

        n_plots = len(features_to_plot)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Ensure axes is always a 1D array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_1d(axes)
        else:
            axes = axes.flatten()

        for idx, feat in enumerate(features_to_plot):
            if idx >= len(axes):
                break

            ax = axes[idx]
            feat_data = self.features[feat].dropna()

            # Remove infinite values
            feat_data = feat_data.replace([np.inf, -np.inf], np.nan).dropna()

            stats.probplot(feat_data, dist="norm", plot=ax)
            ax.set_title(f"{feat}", fontsize=10)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.suptitle("Q-Q Plots (Normality Assessment)", fontsize=16, y=1.001)
        return fig

    # ============================================================================
    # 3. CORRELATION & MULTICOLLINEARITY ANALYSIS
    # ============================================================================
    def analyze_correlation(
        self, threshold: float = 0.8, method: str = "pearson"
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Correlation analysis and redundant feature detection.

        Parameters
        ----------
        threshold : float, default=0.8
            Correlation threshold for flagging redundant features
        method : str, default='pearson'
            'pearson' or 'spearman'

        Returns
        -------
        Tuple[pd.DataFrame, List[Tuple[str, str, float]]]
            Correlation matrix and list of highly correlated feature pairs
        """
        print("\n" + "=" * 80)
        print(f"CORRELATION ANALYSIS ({method.upper()})")
        print("=" * 80)

        # Compute correlation matrix
        X = self.features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        corr_matrix = X.corr(method=method)

        self.correlation_results["matrix"] = corr_matrix
        self.correlation_results["method"] = method

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                    )

        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
        self.correlation_results["high_corr_pairs"] = high_corr_pairs

        print(
            f"\nFound {len(high_corr_pairs)} highly correlated pairs (|corr| >= {threshold}):"
        )
        print("-" * 80)
        if len(high_corr_pairs) > 0:
            print(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':>12}")
            print("-" * 80)
            for f1, f2, corr in high_corr_pairs[:20]:  # Show top 20
                print(f"{f1:<30} {f2:<30} {corr:>12.4f}")
            if len(high_corr_pairs) > 20:
                print(f"\n... and {len(high_corr_pairs) - 20} more pairs")

        return corr_matrix, high_corr_pairs

    def compute_vif(self, vif_threshold: float = 10.0) -> pd.DataFrame:
        """
        Compute Variance Inflation Factor (VIF) for multicollinearity detection.

        Parameters
        ----------
        vif_threshold : float, default=10.0
            VIF threshold for flagging multicollinearity

        Returns
        -------
        pd.DataFrame
            VIF scores for each feature
        """
        print("\n" + "=" * 80)
        print("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
        print("=" * 80)

        X = self.features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)

        vif_data = []
        for i, feat in enumerate(self.feature_names):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data.append(
                    {
                        "feature": feat,
                        "vif": vif,
                        "high_multicollinearity": vif > vif_threshold,
                    }
                )
            except:
                vif_data.append(
                    {"feature": feat, "vif": np.nan, "high_multicollinearity": False}
                )

        vif_df = pd.DataFrame(vif_data).sort_values("vif", ascending=False)
        self.correlation_results["vif"] = vif_df

        print(f"\nFeatures with VIF > {vif_threshold} (High Multicollinearity):")
        print("-" * 80)
        high_vif = vif_df[vif_df["high_multicollinearity"]]
        if len(high_vif) > 0:
            print(high_vif.to_string(index=False))
        else:
            print("None - all features have acceptable VIF")

        return vif_df

    def plot_correlation_matrix(
        self,
        method: str = "pearson",
        figsize: Tuple[int, int] = (16, 14),
        cluster: bool = True,
    ):
        """
        Plot correlation matrix heatmap.

        Parameters
        ----------
        method : str, default='pearson'
            Correlation method
        figsize : Tuple[int, int], default=(16, 14)
            Figure size
        cluster : bool, default=True
            Whether to cluster features by correlation

        Returns
        -------
        matplotlib.figure.Figure or seaborn.ClusterGrid
            Figure object (regular heatmap) or ClusterGrid (clustered heatmap)
        """
        X = self.features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        X = X.ffill().fillna(0)
        corr_matrix = X.corr(method=method)

        if cluster:
            # clustermap creates its own figure - return the ClusterGrid object
            clustergrid = sns.clustermap(
                corr_matrix,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                annot=False,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Correlation"},
                figsize=figsize,
            )
            clustergrid.fig.suptitle(
                f"Clustered Correlation Matrix ({method.capitalize()})",
                fontsize=16,
                y=0.99,
            )
            return clustergrid
        else:
            # Regular heatmap uses provided axes
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(
                corr_matrix,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                annot=False,
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Correlation"},
                ax=ax,
            )
            plt.title(f"Correlation Matrix ({method.capitalize()})", fontsize=16)
            plt.tight_layout()
            return fig

    # ============================================================================
    # 4. TRANSFORMATION RECOMMENDATIONS
    # ============================================================================

    def recommend_transformations(self) -> pd.DataFrame:
        """
        Recommend appropriate transformations for each feature.

        Returns
        -------
        pd.DataFrame
            Transformation recommendations
        """
        print("\n" + "=" * 80)
        print("TRANSFORMATION RECOMMENDATIONS")
        print("=" * 80)

        recommendations = []

        for feat in self.feature_names:
            feat_data = self.features[feat].dropna()

            # Remove infinite values
            feat_data = feat_data.replace([np.inf, -np.inf], np.nan).dropna()

            if len(feat_data) < 30:
                continue

            rec = {"feature": feat, "transformations": [], "reason": []}

            # Get distribution stats
            is_bounded_0_1 = (feat_data.min() >= 0) and (feat_data.max() <= 1)
            is_strictly_positive = feat_data.min() > 0
            skewness = stats.skew(feat_data)
            kurtosis = stats.kurtosis(feat_data)

            # Rule-based recommendations

            # 1. Logit transformation for bounded [0,1] features
            if is_bounded_0_1:
                # Check if it's truly bounded (not just accidentally)
                range_size = feat_data.max() - feat_data.min()
                if range_size > 0.5:  # Uses significant portion of [0,1] range
                    rec["transformations"].append("logit")
                    rec["reason"].append("Bounded [0,1] ratio/oscillator")

            # 2. Log transformation for right-skewed positive features
            if is_strictly_positive and skewness > 1.0:
                rec["transformations"].append("log")
                rec["reason"].append(f"Right-skewed (skew={skewness:.2f})")

            # 3. Box-Cox/Yeo-Johnson for normalization
            if abs(skewness) > 1.5 or abs(kurtosis) > 3:
                if is_strictly_positive:
                    rec["transformations"].append("box-cox")
                else:
                    rec["transformations"].append("yeo-johnson")
                rec["reason"].append(
                    f"Non-normal (skew={skewness:.2f}, kurt={kurtosis:.2f})"
                )

            # 4. Rank/Quantile transformation for extreme outliers
            q1, q99 = feat_data.quantile([0.01, 0.99])
            iqr = feat_data.quantile(0.75) - feat_data.quantile(0.25)
            if iqr > 0:
                outlier_ratio = ((feat_data < q1) | (feat_data > q99)).sum() / len(
                    feat_data
                )
                if outlier_ratio > 0.05:  # More than 5% outliers
                    rec["transformations"].append("quantile/rank")
                    rec["reason"].append(f"Heavy outliers ({outlier_ratio*100:.1f}%)")

            # 5. Standardization (z-score) for features with very different scales
            if feat_data.std() > 0:
                cv = (
                    feat_data.std() / abs(feat_data.mean())
                    if feat_data.mean() != 0
                    else np.inf
                )
                if cv > 2:  # High coefficient of variation
                    rec["transformations"].append("standardize")
                    rec["reason"].append(f"High variance (CV={cv:.2f})")

            # If no transformation needed
            if len(rec["transformations"]) == 0:
                rec["transformations"].append("none")
                rec["reason"].append("Well-behaved distribution")

            rec["transformations"] = ", ".join(rec["transformations"])
            rec["reason"] = " | ".join(rec["reason"])

            recommendations.append(rec)

        rec_df = pd.DataFrame(recommendations)
        self.transformation_results = rec_df

        # Summary
        print("\nTransformation Summary:")
        print("-" * 80)

        all_transforms = []
        for trans_str in rec_df["transformations"]:
            all_transforms.extend([t.strip() for t in trans_str.split(",")])

        from collections import Counter

        trans_counts = Counter(all_transforms)

        for trans, count in trans_counts.most_common():
            print(f"  {trans:<20}: {count} features")

        # Show specific recommendations
        print("\nFeatures Needing Logit Transformation:")
        logit_feats = rec_df[rec_df["transformations"].str.contains("logit")]
        if len(logit_feats) > 0:
            print(logit_feats[["feature", "reason"]].to_string(index=False))
        else:
            print("  None")

        print("\nFeatures Needing Log Transformation:")
        log_feats = rec_df[
            rec_df["transformations"].str.contains("log")
            & ~rec_df["transformations"].str.contains("logit")
        ]
        if len(log_feats) > 0:
            print(log_feats[["feature", "reason"]].to_string(index=False))
        else:
            print("  None")

        return rec_df

    def apply_transformation(
        self, feature: str, transformation: str, **kwargs
    ) -> pd.Series:
        """
        Apply a specific transformation to a feature.

        Parameters
        ----------
        feature : str
            Feature name
        transformation : str
            Transformation type: 'logit', 'log', 'box-cox', 'yeo-johnson',
            'quantile', 'rank', 'standardize'
        **kwargs
            Additional arguments for the transformation

        Returns
        -------
        pd.Series
            Transformed feature
        """
        feat_data = self.features[feature].copy()

        if transformation == "logit":
            # Clip to avoid infinity
            epsilon = kwargs.get("epsilon", 1e-7)
            feat_data = feat_data.clip(epsilon, 1 - epsilon)
            return pd.Series(
                logit(feat_data), index=feat_data.index, name=f"{feature}_logit"
            )

        elif transformation == "log":
            # Add small constant if needed
            if feat_data.min() <= 0:
                offset = abs(feat_data.min()) + 1
                feat_data = feat_data + offset
            return np.log(feat_data).rename(f"{feature}_log")

        elif transformation == "box-cox":
            from scipy.stats import boxcox

            transformed, lmbda = boxcox(feat_data.dropna() + 1)  # +1 to ensure positive
            return pd.Series(
                transformed, index=feat_data.dropna().index, name=f"{feature}_boxcox"
            )

        elif transformation == "yeo-johnson":
            pt = PowerTransformer(method="yeo-johnson")
            transformed = pt.fit_transform(feat_data.values.reshape(-1, 1)).flatten()
            return pd.Series(
                transformed, index=feat_data.index, name=f"{feature}_yeojohnson"
            )

        elif transformation == "quantile":
            qt = QuantileTransformer(output_distribution="normal")
            transformed = qt.fit_transform(feat_data.values.reshape(-1, 1)).flatten()
            return pd.Series(
                transformed, index=feat_data.index, name=f"{feature}_quantile"
            )

        elif transformation == "rank":
            return feat_data.rank(pct=True).rename(f"{feature}_rank")

        elif transformation == "standardize":
            return ((feat_data - feat_data.mean()) / feat_data.std()).rename(
                f"{feature}_std"
            )

        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    # ============================================================================
    # COMPREHENSIVE REPORT
    # ============================================================================

    def generate_comprehensive_report(
        self, output_path: Optional[str] = None, n_top_features: int = 20
    ) -> Dict:
        """
        Generate comprehensive EDA report.

        Parameters
        ----------
        output_path : str, optional
            Path to save HTML report
        n_top_features : int, default=20
            Number of top features to highlight

        Returns
        -------
        Dict
            Dictionary containing all analysis results
        """
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE EDA REPORT")
        print("=" * 80)

        # Run all analyses
        print("\n[1/5] Analyzing feature relevance...")
        relevance_df = self.analyze_feature_relevance(n_top=n_top_features)

        print("\n[2/5] Analyzing distributions...")
        distribution_df = self.analyze_distributions(n_features=n_top_features)

        print("\n[3/5] Analyzing correlations...")
        corr_matrix, high_corr_pairs = self.analyze_correlation(threshold=0.8)

        print("\n[4/5] Computing VIF...")
        vif_df = self.compute_vif(vif_threshold=10.0)

        print("\n[5/5] Recommending transformations...")
        transformation_df = self.recommend_transformations()

        # Compile report
        report = {
            "relevance": relevance_df,
            "distribution": distribution_df,
            "correlation_matrix": corr_matrix,
            "high_correlation_pairs": high_corr_pairs,
            "vif": vif_df,
            "transformations": transformation_df,
        }

        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE")
        print("=" * 80)

        return report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_eda(
    features: pd.DataFrame,
    target: Union[pd.Series, np.ndarray],
    target_type: str = "continuous",
    n_top: int = 20,
) -> FeatureEDA:
    """
    Quick EDA with default settings.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame
    target : pd.Series or np.ndarray
        Target variable
    target_type : str, default='continuous'
        'continuous' or 'categorical'
    n_top : int, default=20
        Number of top features to analyze

    Returns
    -------
    FeatureEDA
        EDA object with all results
    """
    eda = FeatureEDA(features, target, target_type)
    eda.generate_comprehensive_report(n_top_features=n_top)
    return eda
