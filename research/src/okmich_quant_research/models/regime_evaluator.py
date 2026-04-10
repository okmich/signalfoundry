from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

from okmich_quant_labelling.utils.label_eval_util import all_labels_path_structure_statistics, evaluate_all_labels_regime_returns_potentials
from okmich_quant_labelling.utils.label_util import map_label_to_trend_direction, map_label_to_momentum_score, \
    map_regime_to_volatility_score, map_regime_to_path_structure_score


class RegimeEvaluationResult:
    """
    Results from regime evaluation.

    Attributes
    ----------
    path_structure_stats : pd.DataFrame
        Statistics from path_structure_label_classification
    regime_returns_stats : pd.DataFrame, optional
        Statistics from evaluate_regime_returns_potentials
    label_mapping : dict, optional
        Mapping from cluster IDs to directional labels (unsupervised only)
    metrics_summary : dict
        Summary of key metrics per model
    """

    def __init__(
        self,
        path_structure_stats: pd.DataFrame,
        regime_returns_stats: Optional[pd.DataFrame] = None,
        label_mapping: Optional[Dict[str, Dict[int, int]]] = None,
        metrics_summary: Optional[Dict[str, Any]] = None,
    ):
        self.path_structure_stats = path_structure_stats
        self.regime_returns_stats = regime_returns_stats
        self.label_mapping = label_mapping
        self.metrics_summary = metrics_summary or {}

    def save(self, output_dir: str):
        """Save evaluation results to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save path structure stats (only if not empty)
        if not self.path_structure_stats.empty:
            self.path_structure_stats.to_parquet(
                output_dir / "path_structure_stats.parquet", index=False
            )
            self.path_structure_stats.to_csv(
                output_dir / "path_structure_stats.csv", index=False
            )

        # Save regime returns stats if available
        if self.regime_returns_stats is not None:
            # Convert regime column to string to handle mixed types (int + 'overall')
            regime_stats_copy = self.regime_returns_stats.copy()
            regime_stats_copy["regime"] = regime_stats_copy["regime"].astype(str)

            regime_stats_copy.to_parquet(
                output_dir / "regime_returns_stats.parquet", index=False
            )
            regime_stats_copy.to_csv(
                output_dir / "regime_returns_stats.csv", index=False
            )

        # Save label mapping if available
        if self.label_mapping is not None:
            import json

            with open(output_dir / "label_mapping.json", "w") as f:
                json.dump(self.label_mapping, f, indent=2)

        # Save metrics summary
        if self.metrics_summary:
            import json

            with open(output_dir / "metrics_summary.json", "w") as f:
                json.dump(self.metrics_summary, f, indent=2, default=str)

    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        if model_name in self.metrics_summary:
            return self.metrics_summary[model_name]

        # Compute from dataframes
        metrics = {}

        # From path structure stats
        model_stats = self.path_structure_stats[
            self.path_structure_stats["algo"] == model_name
        ]

        if len(model_stats) > 0:
            metrics["regime_discriminability"] = model_stats[
                "regime_discriminability"
            ].mean()
            metrics["mean_duration"] = model_stats["mean_duration"].mean()
            metrics["regime_stability"] = model_stats["regime_stability"].mean()
            metrics["efficiency_ratio"] = model_stats["efficiency_ratio"].mean()
            metrics["volatility"] = model_stats["volatility"].mean()
            metrics["autocorrelation_lag1"] = model_stats["autocorrelation_lag1"].mean()

        # From regime returns stats if available
        if self.regime_returns_stats is not None:
            returns_stats = self.regime_returns_stats[
                self.regime_returns_stats["algo"] == model_name
            ]

            if len(returns_stats) > 0:
                # Get overall row
                overall = returns_stats[returns_stats["regime"] == "overall"]
                if len(overall) > 0:
                    metrics["win_rate"] = overall["win_rate"].values[0]
                    metrics["regime_purity"] = overall["regime_purity"].values[0]
                    metrics["persistence_score"] = overall["persistence_score"].values[
                        0
                    ]
        return metrics

    def __repr__(self):
        n_models = (
            0
            if self.path_structure_stats.empty
            else len(self.path_structure_stats["algo"].unique())
        )
        has_returns = self.regime_returns_stats is not None
        has_mapping = self.label_mapping is not None

        return (
            f"RegimeEvaluationResult(\n"
            f"  models_evaluated={n_models},\n"
            f"  regime_returns_stats={has_returns},\n"
            f"  label_mapping={has_mapping}\n"
            f")"
        )


class RegimeEvaluator:
    """
    Evaluates regime labels using label_eval_util functions.

    Supports both supervised and unsupervised models.

    Examples
    --------
    >>> evaluator = RegimeEvaluator()
    >>>
    >>> # Supervised evaluation
    >>> result = evaluator.evaluate(
    ...     df=data_df,
    ...     label_cols=['hmm_mm_learn_3', 'hmm_pmgnt_3'],
    ...     returns_col='return',
    ...     price_col='close'
    ... )
    >>>
    >>> # Unsupervised evaluation with trend mapping (default)
    >>> result = evaluator.evaluate_unsupervised(
    ...     df=data_df,
    ...     label_cols=['kmeans_3', 'gmm_3'],
    ...     returns_col='return',
    ...     price_col='close',
    ...     mapping_type='trend',
    ...     trend_method='conservative'
    ... )
    >>>
    >>> # Unsupervised evaluation with momentum mapping
    >>> # Note: momentum_range auto-inferred from n_states
    >>> result = evaluator.evaluate_unsupervised(
    ...     df=data_df,
    ...     label_cols=['kmeans_3', 'gmm_3'],
    ...     returns_col='return',
    ...     mapping_type='momentum'
    ... )
    >>>
    >>> # Unsupervised evaluation with volatility mapping
    >>> result = evaluator.evaluate_unsupervised(
    ...     df=data_df,
    ...     label_cols=['kmeans_3', 'gmm_3'],
    ...     mapping_type='volatility',
    ...     vol_proxy_col='atr'
    ... )
    >>>
    >>> # Unsupervised evaluation with path structure mapping
    >>> # Note: choppiness_range auto-inferred from n_states
    >>> result = evaluator.evaluate_unsupervised(
    ...     df=data_df,
    ...     label_cols=['kmeans_3', 'gmm_3'],
    ...     mapping_type='path_structure',
    ...     lookback=14
    ... )
    """

    def __init__(self):
        """Initialize regime evaluator."""
        pass

    @staticmethod
    def _compute_regime_persistence(labels: pd.Series) -> float:
        """Mean consecutive-bar run length of non-NaN labels."""
        vals = labels.dropna().values
        if len(vals) == 0:
            return np.nan
        run_lengths = []
        run = 1
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
        return float(np.mean(run_lengths))

    @staticmethod
    def _compute_regime_conditioned_sharpe(
        labels: pd.Series,
        returns: pd.Series,
        bars_per_year: int = 252 * 288,
    ) -> Dict[Any, float]:
        """Annualised Sharpe ratio conditioned on each regime state."""
        aligned = pd.concat([labels.rename("label"), returns.rename("ret")], axis=1).dropna()
        result: Dict[Any, float] = {}
        for regime, group in aligned.groupby("label"):
            rets = group["ret"]
            if len(rets) < 10 or rets.std() == 0:
                result[regime] = np.nan
            else:
                result[regime] = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        label_cols: List[str],
        returns_col: str = "return",
        price_col: str = "close",
        include_regime_returns: bool = True,
        progressive_skip: int = 1,
        whipsaw_cost: float = 0.0,
        wrong_direction_penalty: float = 0.0,
        label_sign_map: Optional[Dict[int, int]] = None,
        bars_per_year: int = 252 * 288,
    ) -> RegimeEvaluationResult:
        print("=" * 80)
        print("REGIME EVALUATION")
        print("=" * 80)

        # Validate inputs
        required_cols = [returns_col, price_col] + label_cols
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # 1. Path structure statistics
        print(
            f"\n[1/2] Computing path structure statistics for {len(label_cols)} models..."
        )
        path_structure_stats = all_labels_path_structure_statistics(
            df=df, state_cols=label_cols, returns_col=returns_col, price_col=price_col
        )
        if path_structure_stats is not None:
            print(
                f"   [OK] Computed statistics for {len(path_structure_stats)} regimes"
            )
        else:
            print(
                f"   [SKIP] No label columns provided, skipping path structure statistics"
            )
            path_structure_stats = pd.DataFrame()  # Empty dataframe instead of None

        # 2. Regime returns potentials
        regime_returns_stats = None
        if include_regime_returns and len(label_cols) > 0:
            print(f"\n[2/2] Evaluating regime returns potentials...")

            # Use default label_sign_mapping_method for supervised research
            regime_returns_stats = evaluate_all_labels_regime_returns_potentials(
                df=df,
                labels_cols=label_cols,
                progressive_skip=progressive_skip,
                whipsaw_cost=whipsaw_cost,
                wrong_direction_penalty=wrong_direction_penalty,
                label_sign_mapping_method="simple",  # Use default method
                include_overall=True,
            )

            # Handle None return (empty label_cols)
            if regime_returns_stats is not None:
                # Reset index to get 'algo' as column
                regime_returns_stats = regime_returns_stats.reset_index()
                print(f"   [OK] Evaluated returns for {len(label_cols)} models")
            else:
                print(
                    f"   [SKIP] No label columns provided, skipping regime returns evaluation"
                )

        # Create summary
        metrics_summary = {}
        for label_col in label_cols:
            summary = {}

            # Only compute path structure metrics if we have data
            if not path_structure_stats.empty:
                model_stats = path_structure_stats[
                    path_structure_stats["algo"] == label_col
                ]

                if not model_stats.empty:
                    summary["regime_discriminability"] = float(
                        model_stats["regime_discriminability"].mean()
                    )
                    summary["mean_duration"] = float(
                        model_stats["mean_duration"].mean()
                    )
                    summary["regime_stability"] = float(
                        model_stats["regime_stability"].mean()
                    )
                    summary["efficiency_ratio_mean"] = float(
                        model_stats["efficiency_ratio"].mean()
                    )
                    summary["volatility_mean"] = float(model_stats["volatility"].mean())

            if regime_returns_stats is not None:
                returns_overall = regime_returns_stats[
                    (regime_returns_stats["algo"] == label_col)
                    & (regime_returns_stats["regime"] == "overall")
                ]
                if len(returns_overall) > 0:
                    summary["win_rate"] = float(returns_overall["win_rate"].values[0])
                    summary["regime_purity"] = float(
                        returns_overall["regime_purity"].values[0]
                    )
                    summary["persistence_score"] = float(
                        returns_overall["persistence_score"].values[0]
                    )

            # Regime persistence and regime-conditioned Sharpe
            summary["regime_persistence"] = self._compute_regime_persistence(df[label_col])
            if returns_col in df.columns:
                summary["regime_conditioned_sharpe"] = self._compute_regime_conditioned_sharpe(
                    df[label_col], df[returns_col], bars_per_year=bars_per_year
                )

            metrics_summary[label_col] = summary

        result = RegimeEvaluationResult(
            path_structure_stats=path_structure_stats,
            regime_returns_stats=regime_returns_stats,
            metrics_summary=metrics_summary,
        )

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)

        return result

    def evaluate_unsupervised(
        self,
        df: pd.DataFrame,
        label_cols: List[str],
        returns_col: str = "return",
        price_col: str = "close",
        progressive_skip: int = 1,
        mapping_type: str = "trend",
        vol_proxy_col: Optional[str] = None,
        lookback: int = 14,
        trend_method: str = "conservative",
        cost_threshold: Optional[float] = None,
        min_sharpe: float = 0.3,
        include_regime_returns: bool = True,
    ) -> RegimeEvaluationResult:
        """
        Evaluate unsupervised clustering labels.

        Maps unlabeled clusters to meaningful labels based on market properties,
        then evaluates both cluster quality and the mapped property potential.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price data and cluster labels
        label_cols : list of str
            Column names containing cluster labels
        returns_col : str, default='return'
            Returns column name
        price_col : str, default='close'
            Price column name
        progressive_skip : int, default=1
            Skip periods for evaluation
        mapping_type : str, default='trend'
            Type of mapping to apply. Options:
            - 'trend': Map to trend direction (-1, 0, 1) using statistical tests
            - 'momentum': Map to momentum scores
            - 'volatility': Map to volatility buckets
            - 'path_structure': Map to choppiness/path structure scores
        vol_proxy_col : str, optional
            Column name for volatility proxy (required for 'volatility' mapping)
            Examples: 'realized_vol', 'atr', 'abs_returns'
        lookback : int, default=14
            Lookback window for path structure calculations
            Note: momentum_range and choppiness_range are auto-inferred from n_states
            (number of unique labels in each model)
        trend_method : str, default='conservative'
            Method for trend mapping. Options: 'conservative', 'statistical', 'sharpe', 'simple'
            - 'conservative': Requires statistical significance + Sharpe + magnitude (strictest)
            - 'statistical': Requires statistical significance + magnitude
            - 'sharpe': Requires risk-adjusted returns + magnitude
            - 'simple': Pure ranking by mean (fastest, least statistical)
        cost_threshold : float, optional
            Minimum mean return to exceed transaction costs (trend mapping only)
            If None, auto-estimated from state separation
        min_sharpe : float, default=0.3
            Minimum Sharpe ratio for trend mapping
        include_regime_returns : bool, default=True
            Whether to evaluate regime returns potentials

        Returns
        -------
        RegimeEvaluationResult
            Evaluation results with label mapping
        """
        print("=" * 80)
        print("UNSUPERVISED REGIME EVALUATION")
        print("=" * 80)

        # 1. Path structure statistics (on original cluster labels)
        print(f"\n[1/4] Computing cluster statistics for {len(label_cols)} models...")
        path_structure_stats = all_labels_path_structure_statistics(
            df=df, state_cols=label_cols, returns_col=returns_col, price_col=price_col
        )
        if path_structure_stats is None:
            path_structure_stats = pd.DataFrame()
        print(f"   [OK] Computed statistics for {len(path_structure_stats)} clusters")

        # 2. Map clusters to meaningful labels based on mapping_type
        mapping_type_display = {
            "trend": "trend direction",
            "momentum": "momentum scores",
            "volatility": "volatility buckets",
            "path_structure": "path structure/choppiness scores",
        }.get(mapping_type, mapping_type)

        print(f"\n[2/4] Mapping clusters to {mapping_type_display}...")

        # Validate mapping_type
        valid_types = ["trend", "momentum", "volatility", "path_structure"]
        if mapping_type not in valid_types:
            raise ValueError(
                f"Invalid mapping_type '{mapping_type}'. Choose from: {valid_types}"
            )

        # Validate required columns for specific mapping types
        if mapping_type == "volatility":
            if vol_proxy_col is None:
                raise ValueError(
                    "vol_proxy_col is required for 'volatility' mapping type"
                )
            if vol_proxy_col not in df.columns:
                raise ValueError(f"Column '{vol_proxy_col}' not found in DataFrame")

        if mapping_type == "path_structure":
            required_cols = ["high", "low", "close"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(
                    f"path_structure mapping requires columns {required_cols}, missing: {missing}"
                )

        label_mapping = {}
        df_mapped = df.copy()

        for label_col in label_cols:
            # Auto-infer n_states from the actual number of unique labels
            n_states = int(df[label_col].nunique())

            # Apply the appropriate mapping function
            if mapping_type == "trend":
                mapping = map_label_to_trend_direction(
                    df=df,
                    state_col=label_col,
                    return_col=returns_col,
                    method=trend_method,
                    cost_threshold=cost_threshold,
                    min_sharpe=min_sharpe,
                )
                # Print mapping
                print(f"   {label_col} (n_states={n_states}):")
                for cluster, direction in mapping.items():
                    direction_label = {-1: "bearish", 0: "neutral", 1: "bullish"}.get(
                        direction, "unknown"
                    )
                    print(f"      Cluster {cluster} -> {direction} ({direction_label})")

            elif mapping_type == "momentum":
                # Auto-infer momentum_range from n_states
                # For directional scoring, use range that gives n_states unique values
                momentum_range = (
                    n_states // 2
                )  # e.g., 3 states -> range 1 (-1, 0, 1); 5 states -> range 2 (-2, -1, 0, 1, 2)

                mapping = map_label_to_momentum_score(
                    df=df,
                    regime_col=label_col,
                    ret_col=returns_col,
                    momentum_range=momentum_range,
                    is_directional=True,
                )
                # Print mapping
                print(
                    f"   {label_col} (n_states={n_states}, momentum_range={momentum_range}):"
                )
                for cluster, score in mapping.items():
                    print(f"      Cluster {cluster} -> momentum score {score}")

            elif mapping_type == "volatility":
                mapping = map_regime_to_volatility_score(
                    df=df, regime_col=label_col, vol_proxy_col=vol_proxy_col
                )
                # Print mapping
                print(f"   {label_col} (n_states={n_states}):")
                for cluster, bucket in mapping.items():
                    print(f"      Cluster {cluster} -> volatility bucket {bucket}")

            elif mapping_type == "path_structure":
                # Auto-infer choppiness_range from n_states
                # Range from 0 to (n_states - 1)
                choppiness_range = (0, n_states - 1)

                mapping = map_regime_to_path_structure_score(
                    df=df,
                    regime_col=label_col,
                    price_col=price_col,
                    choppiness_range=choppiness_range,
                    lookback=lookback,
                )
                # Print mapping
                print(
                    f"   {label_col} (n_states={n_states}, choppiness_range={choppiness_range}):"
                )
                for cluster, score in mapping.items():
                    chop_label = (
                        "smooth/trending"
                        if score == choppiness_range[0]
                        else (
                            "choppy/whipsaw"
                            if score == choppiness_range[1]
                            else "moderate"
                        )
                    )
                    print(
                        f"      Cluster {cluster} -> choppiness score {score} ({chop_label})"
                    )

            label_mapping[label_col] = mapping

            # Apply mapping
            mapped_col = f"{label_col}_mapped"
            df_mapped[mapped_col] = df_mapped[label_col].map(mapping)

        # 3. Evaluate mapped directional labels (if enabled)
        regime_returns_stats = None
        if include_regime_returns:
            print(f"\n[3/4] Evaluating directional potential (mapped labels)...")
            mapped_cols = [f"{col}_mapped" for col in label_cols]

            # Use default label_sign_mapping_method since mapping is already done
            regime_returns_stats = evaluate_all_labels_regime_returns_potentials(
                df=df_mapped,
                labels_cols=mapped_cols,
                progressive_skip=progressive_skip,
                whipsaw_cost=0.0,
                wrong_direction_penalty=0.0,
                label_sign_mapping_method="simple",  # Use default method
                include_overall=True,
            )

            # Reset index to get 'algo' as column
            if regime_returns_stats is not None:
                regime_returns_stats = regime_returns_stats.reset_index()
                # Rename algo column to remove '_mapped' suffix for consistency
                regime_returns_stats["algo"] = regime_returns_stats["algo"].str.replace(
                    "_mapped", ""
                )
                print(f"   [OK] Evaluated directional potential")
            else:
                print(f"   [WARN] evaluate_all_labels_regime_returns_potentials returned None; "
                      f"directional potential metrics will be absent from summary")
        else:
            print(f"\n[3/4] Skipping regime returns evaluation (disabled in config)")

        # 4. Create summary
        print(f"\n[4/4] Creating summary...")
        metrics_summary = {}

        for label_col in label_cols:
            summary: Dict[str, Any] = {"label_mapping": label_mapping[label_col]}

            # Cluster quality metrics (only when path_structure_stats is available)
            if not path_structure_stats.empty:
                cluster_stats = path_structure_stats[
                    path_structure_stats["algo"] == label_col
                ]
                if not cluster_stats.empty:
                    summary["n_clusters"] = len(cluster_stats)
                    summary["regime_discriminability"] = float(
                        cluster_stats["regime_discriminability"].mean()
                    )
                    summary["mean_duration"] = float(cluster_stats["mean_duration"].mean())
                    summary["regime_stability"] = float(
                        cluster_stats["regime_stability"].mean()
                    )
                    summary["efficiency_ratio_mean"] = float(
                        cluster_stats["efficiency_ratio"].mean()
                    )
                    summary["volatility_mean"] = float(cluster_stats["volatility"].mean())

            # Directional potential metrics (if available)
            if regime_returns_stats is not None:
                returns_overall = regime_returns_stats[
                    (regime_returns_stats["algo"] == label_col)
                    & (regime_returns_stats["regime"] == "overall")
                ]
                if len(returns_overall) > 0:
                    summary["win_rate"] = float(returns_overall["win_rate"].values[0])
                    summary["regime_purity"] = float(
                        returns_overall["regime_purity"].values[0]
                    )
                    summary["persistence_score"] = float(
                        returns_overall["persistence_score"].values[0]
                    )

            metrics_summary[label_col] = summary

        result = RegimeEvaluationResult(
            path_structure_stats=path_structure_stats,
            regime_returns_stats=regime_returns_stats,
            label_mapping=label_mapping,
            metrics_summary=metrics_summary,
        )

        print("\n" + "=" * 80)
        print("UNSUPERVISED EVALUATION COMPLETE")
        print("=" * 80)

        return result
