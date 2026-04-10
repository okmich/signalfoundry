import json
from dataclasses import asdict
from enum import Enum
from itertools import product
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score, davies_bouldin_score

from okmich_quant_labelling.utils.label_eval_util import label_path_structure_statistics
from okmich_quant_ml.hmm import PomegranateHMM, PomegranateMixtureHMM
from okmich_quant_ml.hmm.pomegranate import DistType
from okmich_quant_research.backtesting.hmm_clustering_comparison_backtesting_pipeline import (
    HMM_ALGOS,
)


def to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (numpy types, arrays, enums, or Python natives)

    Returns:
        JSON-serializable Python native type
    """
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):  # Handle any Enum type (DistType, etc.)
        return obj.name
    return obj


def _create_hmm_model_static(config: Dict[str, Any], hmm_params: Dict[str, Any]):
    """Static factory function to create HMM model (for parallel execution).

    Args:
        config: Configuration dict with model_variant, n_states, and optional n_components
        hmm_params: HMM parameters dict

    Returns:
        HMM model instance
    """
    variant = config["model_variant"]
    n_states = config["n_states"]
    random_state = hmm_params.get("random_state", 42)
    max_iter = hmm_params.get("max_iter", 100)

    # HMMLearn models
    if variant == "hmm_pmgnt":
        return PomegranateHMM(
            distribution_type=DistType.NORMAL,
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_expnt":
        return PomegranateHMM(
            distribution_type=DistType.EXPONENTIAL,
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_gmma":
        return PomegranateHMM(
            distribution_type=DistType.GAMMA,
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_lambda":
        return PomegranateHMM(
            distribution_type=DistType.LAMDA,
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_lognorm":
        return PomegranateHMM(
            distribution_type=DistType.LOGNORMAL,
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_student":
        return PomegranateHMM(
            distribution_type=DistType.STUDENTT,
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
            **hmm_params.get("dist_kwargs", {}),
        )
    # PomegranateMixtureHMM models
    elif variant == "hmm_mm_pmgnt":
        return PomegranateMixtureHMM(
            distribution_type=DistType.NORMAL,
            n_states=n_states,
            n_components=config.get("n_components", 2),

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_mm_expnt":
        return PomegranateMixtureHMM(
            distribution_type=DistType.EXPONENTIAL,
            n_states=n_states,
            n_components=config.get("n_components", 2),

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_mm_gmma":
        return PomegranateMixtureHMM(
            distribution_type=DistType.GAMMA,
            n_states=n_states,
            n_components=config.get("n_components", 2),

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_mm_lambda":
        return PomegranateMixtureHMM(
            distribution_type=DistType.LAMDA,
            n_states=n_states,
            n_components=config.get("n_components", 2),

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_mm_lognorm":
        return PomegranateMixtureHMM(
            distribution_type=DistType.LOGNORMAL,
            n_components=config.get("n_components", 2),
            n_states=n_states,

            random_state=random_state,
            max_iter=max_iter,
        )
    elif variant == "hmm_mm_student":
        return PomegranateMixtureHMM(
            distribution_type=DistType.STUDENTT,
            n_states=n_states,
            n_components=config.get("n_components", 2),

            random_state=random_state,
            max_iter=max_iter,
            **hmm_params.get("dist_kwargs", {}),
        )
    else:
        raise ValueError(f"Unknown HMM variant: {variant}")


def _train_single_hmm_config(
    config: Dict[str, Any],
    train_features: np.ndarray,
    criterion: str,
    hmm_params: Dict[str, Any],
) -> Tuple:
    """Train a single HMM configuration (for parallel execution).

    Args:
        config: HMM configuration dict
        train_features: Training features (2D numpy array)
        criterion: 'aic', 'bic', 'silhouette', or 'combined'
        hmm_params: HMM parameters dict

    Returns:
        Tuple of (config, model, score, aic, bic, silhouette, db_index, error)
        - If successful: (config, model, score, aic, bic, silhouette, db_index, None)
        - If failed: (config, None, np.inf, None, None, None, None, error_str)
    """
    try:
        if isinstance(train_features, pd.DataFrame):
            train_features = train_features.to_numpy(dtype=np.float64)
        train_features = np.asarray(train_features, dtype=np.float64)

        # Create and fit model
        model = _create_hmm_model_static(config, hmm_params)
        model.fit(train_features)

        # Calculate AIC/BIC
        aic, bic = model.get_aic_bic(train_features)

        # Get state predictions for clustering metrics
        state_predictions = model.predict(train_features)

        # Calculate clustering quality metrics
        try:
            # Only calculate if we have at least 2 states and multiple samples per state
            n_states = len(np.unique(state_predictions))
            if n_states >= 2 and len(train_features) >= n_states * 2:
                silhouette = silhouette_score(train_features, state_predictions)
                db_index = davies_bouldin_score(train_features, state_predictions)
            else:
                silhouette = -1.0  # Invalid (lower is worse)
                db_index = np.inf  # Invalid (higher is worse)
        except Exception:
            silhouette = -1.0
            db_index = np.inf

        # Calculate composite score based on criterion
        if criterion == "aic":
            score = aic
        elif criterion == "bic":
            score = bic
        elif criterion == "silhouette":
            score = -silhouette  # Negate so lower is better (consistent with AIC/BIC)
        elif criterion == "combined":
            # Normalize and combine (will be normalized across all models in main function)
            score = (
                aic,
                bic,
                silhouette,
                db_index,
            )  # Return tuple for later normalization
        else:
            score = aic  # Default to AIC

        return config, model, score, aic, bic, silhouette, db_index, None

    except Exception as e:
        return config, None, np.inf, None, None, None, None, str(e)


class CheckpointingMixin:
    """Mixin providing checkpoint save/load functionality for walk-forward optimizers.

    Classes using this mixin must have:
        - self.checkpoint_dir (Path): Directory to store checkpoints
        - self.window_results (list): List of window results
        - self.best_window_idx (Optional[int]): Index of best window
        - self.best_metric_value (float): Best metric value seen
        - self.log: Logger instance with info/debug/warning methods
        - self._create_window_result(**kwargs): Method to create window result from dict
    """

    def _save_checkpoint(self, window_idx: int) -> None:
        """Save checkpoint for resume capability.

        Args:
            window_idx: Current window index being processed
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{window_idx}.json"
            checkpoint_data = {
                "window_idx": window_idx,
                "window_results": [asdict(wr) for wr in self.window_results],
                "best_window_idx": self.best_window_idx,
                "best_metric_value": (
                    float(self.best_metric_value)
                    if self.best_metric_value != -np.inf
                    else None
                ),
                "completed": True,
            }
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=to_serializable)
            self.log.debug(f"Checkpoint saved: {checkpoint_file}", "💾")
        except Exception as e:
            self.log.warning(f"Could not save checkpoint: {e}")

    def _load_checkpoint(self) -> Optional[int]:
        """Load checkpoint to resume interrupted run.

        Returns:
            Last completed window index if checkpoint exists, None otherwise
        """
        if not self.checkpoint_dir.exists():
            return None

        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoint_files:
            return None

        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        try:
            with open(latest_checkpoint, "r") as f:
                checkpoint_data = json.load(f)

            self.window_results = [
                self._create_window_result(**wr)
                for wr in checkpoint_data["window_results"]
            ]
            self.best_window_idx = checkpoint_data.get("best_window_idx")
            self.best_metric_value = checkpoint_data.get("best_metric_value", -np.inf)
            if self.best_metric_value is None:
                self.best_metric_value = -np.inf

            last_window = checkpoint_data["window_idx"]

            self.log.info(f"Loaded checkpoint from window {last_window}", "✓")
            return last_window
        except Exception as e:
            self.log.warning(f"Could not load checkpoint: {e}")
            return None


class WindowGenerationMixin:
    """Mixin providing window calculation functionality for walk-forward optimizers.

    Classes using this mixin must have:
        - self.raw_data: DataFrame with data to split into windows
        - self.train_period (int): Number of periods for training
        - self.test_period (int): Number of periods for testing
        - self.step_period (int): Number of periods to step forward
        - self.anchored (bool): Whether to use anchored (expanding) windows
        - self.log: Logger instance with info method
    """

    def _calculate_windows(self) -> List[Tuple[int, int, int, int]]:
        """Calculate window boundaries for walk-forward analysis.

        Returns:
            List of tuples (train_start_idx, train_end_idx, test_start_idx, test_end_idx)
        """
        total_length = len(self.raw_data)
        windows = []

        if self.anchored:
            min_train_end = self.train_period
            n_windows = (
                total_length - min_train_end - self.test_period
            ) // self.step_period + 1
        else:
            n_windows = (
                total_length - self.train_period - self.test_period
            ) // self.step_period + 1

        for i in range(n_windows):
            if self.anchored:
                train_start_idx = 0
                train_end_idx = self.train_period + (i * self.step_period)
            else:
                train_start_idx = i * self.step_period
                train_end_idx = train_start_idx + self.train_period

            test_start_idx = train_end_idx
            test_end_idx = test_start_idx + self.test_period

            if test_end_idx > total_length:
                break

            windows.append(
                (train_start_idx, train_end_idx, test_start_idx, test_end_idx)
            )

        mode = "anchored" if self.anchored else "rolling"
        self.log.info(f"Calculated {len(windows)} windows ({mode} mode)", "📊")
        return windows


class MetricsTrackingMixin:
    """Mixin providing convergence and metrics tracking for walk-forward optimizers.

    Classes using this mixin must have:
        - self.window_results (list): List of window results
        - self.convergence_window_size (int): Number of windows to use for convergence calculation
        - self.convergence_threshold (float): Threshold for trend detection
        - self.portfolio_convergence_metric (str): Metric to track ('sharpe', 'return', 'drawdown')
          (only for portfolio convergence)
    """

    def _check_portfolio_convergence(self) -> Optional[str]:
        """Check if portfolio performance is converging, stable, or degrading.

        Returns:
            'improving', 'degrading', 'stable', or None if insufficient data
        """
        if len(self.window_results) < self.convergence_window_size:
            return None

        # Get recent metrics based on portfolio_convergence_metric
        recent_metrics = []
        for wr in self.window_results[-self.convergence_window_size :]:
            if self.portfolio_convergence_metric == "sharpe":
                recent_metrics.append(wr.sharpe_ratio)
            elif self.portfolio_convergence_metric == "return":
                recent_metrics.append(wr.total_return)
            elif self.portfolio_convergence_metric == "drawdown":
                recent_metrics.append(-wr.max_drawdown)  # Negative so higher is better
            else:
                recent_metrics.append(wr.sharpe_ratio)

        if len(recent_metrics) < 2:
            return None

        # Calculate trend
        x = np.arange(len(recent_metrics))
        trend = np.polyfit(x, recent_metrics, 1)[0]

        if trend > self.convergence_threshold:
            return "improving"
        elif trend < -self.convergence_threshold:
            return "degrading"
        else:
            return "stable"

    def _check_convergence(self, metric_name: str = "f1_score") -> Optional[str]:
        """Check if performance is converging, stable, or degrading based on a metric.

        Args:
            metric_name: Name of the metric attribute to track (default: 'f1_score')

        Returns:
            'improving', 'degrading', 'stable', or None if insufficient data
        """
        if len(self.window_results) < self.convergence_window_size:
            return None

        recent_metrics = [
            getattr(wr, metric_name)
            for wr in self.window_results[-self.convergence_window_size :]
        ]

        if len(recent_metrics) < 2:
            return None

        x = np.arange(len(recent_metrics))
        trend = np.polyfit(x, recent_metrics, 1)[0]

        if trend > self.convergence_threshold:
            return "improving"
        elif trend < -self.convergence_threshold:
            return "degrading"
        else:
            return "stable"


class BacktestingMixin:
    """Mixin providing VectorBT backtesting functionality for walk-forward optimizers.

    Classes using this mixin must have:
        - self.vbt_params (Dict): Parameters for VectorBT Portfolio.from_signals
        - self.signal_optimization_metric (str): Metric to optimize ('sharpe', 'return', 'win_rate')
        - self.close_col (str): Name of close price column (for backtesting classes)
        - self.log: Logger instance with debug method
    """

    def _run_backtest(
        self, signals: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], prices
    ) -> "vbt.Portfolio":
        """Run VectorBT backtest with ndarray signals.

        Args:
            signals: Tuple of 4 ndarrays (long_entries, long_exits, short_entries, short_exits)
            prices: Price series for backtesting

        Returns:
            VectorBT Portfolio object
        """
        if vbt is None:
            raise ImportError("vectorbt is required for backtesting functionality")

        long_entries, long_exits, short_entries, short_exits = signals
        pf = vbt.Portfolio.from_signals(
            prices,
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            **self.vbt_params,
        )
        return pf

    def _extract_metric(self, pf: "vbt.Portfolio") -> float:
        """Extract optimization metric from portfolio.

        Args:
            pf: VectorBT Portfolio object

        Returns:
            Metric value for optimization
        """
        stats = pf.stats()

        # Support both 'sharpe' and 'sharpe_ratio' for compatibility
        if self.signal_optimization_metric in ("sharpe", "sharpe_ratio"):
            return float(stats.get("Sharpe Ratio", -np.inf))
        elif self.signal_optimization_metric == "return":
            return float(stats.get("Total Return [%]", -np.inf))
        elif self.signal_optimization_metric == "win_rate":
            return float(stats.get("Win Rate [%]", -np.inf))
        else:
            return float(stats.get("Sharpe Ratio", -np.inf))

    def _calculate_trading_metrics(self, pf: "vbt.Portfolio") -> Dict[str, float]:
        """Extract comprehensive trading metrics from portfolio.

        Args:
            pf: VectorBT Portfolio object

        Returns:
            Dict with sharpe, return, drawdown, win_rate, and advanced metrics
        """
        stats = pf.stats()

        # Get returns for additional metrics calculations
        returns = pf.returns()

        metrics = {
            "sharpe_ratio": float(stats.get("Sharpe Ratio", np.nan)),
            "total_return": float(stats.get("Total Return [%]", np.nan)),
            "max_drawdown": float(stats.get("Max Drawdown [%]", np.nan)),
            "win_rate": float(stats.get("Win Rate [%]", np.nan)),
            "num_trades": int(stats.get("Total Trades", 0)),
        }

        try:
            # Sortino ratio (downside risk adjusted) - returns already fetched above
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    metrics["sortino_ratio"] = (
                        returns.mean() / downside_std
                    ) * np.sqrt(252)
                else:
                    metrics["sortino_ratio"] = np.nan
            else:
                metrics["sortino_ratio"] = np.nan

            # Calmar ratio (return / max drawdown)
            if metrics["max_drawdown"] != 0:
                metrics["calmar_ratio"] = metrics["total_return"] / abs(
                    metrics["max_drawdown"]
                )
            else:
                metrics["calmar_ratio"] = np.nan

            # Profit factor (gross profit / gross loss)
            trades = pf.trades.records_readable
            if len(trades) > 0:
                winning_trades = trades[trades["PnL"] > 0]["PnL"].sum()
                losing_trades = abs(trades[trades["PnL"] < 0]["PnL"].sum())
                if losing_trades > 0:
                    metrics["profit_factor"] = winning_trades / losing_trades
                else:
                    metrics["profit_factor"] = np.inf if winning_trades > 0 else np.nan

                # Average trade duration
                if "Duration" in trades.columns:
                    metrics["avg_trade_duration"] = trades["Duration"].mean()
                else:
                    metrics["avg_trade_duration"] = np.nan
            else:
                metrics["profit_factor"] = np.nan
                metrics["avg_trade_duration"] = np.nan

        except Exception as e:
            self.log.debug(f"Could not calculate advanced metrics: {e}")
            metrics["sortino_ratio"] = np.nan
            metrics["calmar_ratio"] = np.nan
            metrics["profit_factor"] = np.nan
            metrics["avg_trade_duration"] = np.nan

        return metrics


class HMMMetricsMixin:
    """Mixin providing HMM state metrics calculation for walk-forward optimizers.

    Classes using this mixin must have:
        - self.log: Logger instance with debug method
    """

    def _calculate_state_durations(
        self, state_predictions: np.ndarray, prices: np.ndarray = None
    ) -> Dict[str, Any]:
        """Calculate state duration statistics using label_path_structure_statistics.

        This uses the trusted label_path_structure_statistics function to calculate
        per-state duration metrics, then extracts both aggregate and per-state statistics.

        Args:
            state_predictions: Array of predicted states
            prices: Optional price array (if not provided, uses synthetic prices)

        Returns:
            Dict with:
            - Aggregate duration statistics (for backward compatibility)
            - per_state_duration_stats: Dict mapping state -> duration metrics
        """
        # Create DataFrame for label_path_structure_statistics
        if prices is None:
            # Use constant synthetic prices so duration statistics are deterministic.
            # (Return-based metrics will all be zero, but duration counts are correct.)
            prices = np.ones(len(state_predictions)) * 100.0

        df = pd.DataFrame({"close": prices, "state": state_predictions})
        df["returns"] = df["close"].pct_change()

        try:
            # Use the trusted method
            stats_df = label_path_structure_statistics(
                df, state_col="state", returns_col="returns", price_col="close"
            )

            # Extract per-state statistics (this is the detailed, trustworthy data)
            per_state_stats = {}
            all_durations = []

            for _, row in stats_df.iterrows():
                state_id = int(row["regime"])
                per_state_stats[state_id] = {
                    "mean_duration": float(row["mean_duration"]),
                    "median_duration": float(row["median_duration"]),
                    "min_duration": float(row["min_duration"]),
                    "q5_duration": float(row["q5_duration"]),
                    "q10_duration": float(row["q10_duration"]),
                    "q25_duration": float(row["q25_duration"]),
                    "q75_duration": float(row["q75_duration"]),
                    "max_duration": float(row["max_duration"]),
                    "frequency_pct": float(row["frequency_pct"]),
                    "n_observations": int(row["n_observations"]),
                    "regime_stability": float(row["regime_stability"]),
                }

                # Collect all durations for aggregate stats (weighted by frequency)
                # Approximate: create list weighted by n_observations
                # This is a pooled approach - all durations from all states
                state_mean_duration = row["mean_duration"]
                n_regimes = (
                    int(row["n_observations"] / state_mean_duration)
                    if state_mean_duration > 0
                    else 0
                )
                all_durations.extend([state_mean_duration] * max(1, n_regimes))

            # Calculate aggregate statistics (pooled across all states)
            # These are for backward compatibility with existing code
            if len(all_durations) > 0:
                aggregate_stats = {
                    "avg_state_duration": float(np.mean(all_durations)),
                    "median_state_duration": float(np.median(all_durations)),
                }
            else:
                aggregate_stats = {
                    "avg_state_duration": np.nan,
                    "median_state_duration": np.nan,
                }

            # Combine aggregate and per-state stats
            result = aggregate_stats.copy()
            result["per_state_duration_stats"] = per_state_stats

            return result

        except Exception as e:
            # Fallback if label_path_structure_statistics fails
            self.log.debug(f"Could not calculate duration statistics: {e}")
            return {
                "avg_state_duration": np.nan,
                "median_state_duration": np.nan,
                "per_state_duration_stats": {},
            }

    def _calculate_state_entropy(self, state_probabilities: np.ndarray) -> float:
        """Calculate average entropy of state probabilities.

        Lower entropy = more confident predictions.

        Args:
            state_probabilities: State probability matrix (n_samples x n_states)

        Returns:
            Average entropy across all samples
        """
        # Avoid log(0)
        probs = np.clip(state_probabilities, 1e-10, 1.0)
        # Calculate entropy for each sample: -sum(p * log(p))
        entropies = -np.sum(probs * np.log(probs), axis=1)
        return float(np.mean(entropies))

    def _calculate_state_stability(self, state_predictions: np.ndarray) -> float:
        """Calculate state stability (proportion of non-transitions).

        Args:
            state_predictions: Array of predicted states

        Returns:
            Proportion of samples where state doesn't change (0-1)
        """
        if len(state_predictions) <= 1:
            return 1.0

        transitions = np.sum(np.diff(state_predictions) != 0)
        stability = 1.0 - (transitions / (len(state_predictions) - 1))
        return float(stability)


class HMMModelSelectionMixin:
    """Mixin providing HMM model selection and creation for walk-forward optimizers.

    Classes using this mixin must have:
        - self.hmm_params (Dict): HMM parameters including model_variants, n_states_range, etc.
        - self.log: Logger instance with info/debug methods
    """

    def _generate_hmm_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of HMM configurations to try.

        Returns:
            List of configuration dicts, each with model_variant, n_states, and type-specific params
        """
        n_states_range = self.hmm_params["n_states_range"]
        model_variants = self.hmm_params.get("model_variants", [])

        # Validate model_variants
        if not isinstance(model_variants, list):
            model_variants = [model_variants]

        for variant in model_variants:
            if variant not in HMM_ALGOS:
                raise ValueError(
                    f"Unknown HMM variant '{variant}'. Must be one of: {HMM_ALGOS}"
                )

        configs = []

        # Grid search: model_variant x n_states
        for variant in model_variants:
            for n_states in n_states_range:
                # For mixture models, also grid search over n_components if provided
                if variant.startswith("hmm_mm_"):
                    n_components_range = self.hmm_params.get("n_components_range", [2])
                    if not isinstance(n_components_range, list):
                        n_components_range = [n_components_range]

                    for n_components in n_components_range:
                        configs.append(
                            {
                                "model_variant": variant,
                                "n_states": n_states,
                                "n_components": n_components,
                            }
                        )
                else:
                    configs.append({"model_variant": variant, "n_states": n_states})

        return configs

    def _create_hmm_model(self, config: Dict[str, Any]):
        """Factory method to create HMM model based on configuration.

        Args:
            config: Configuration dict with model_variant, n_states, and optional n_components

        Returns:
            HMM model instance (GaussianHMM, PomegranateHMM, or PomegranateMixtureHMM)
        """
        variant = config["model_variant"]
        n_states = config["n_states"]
        random_state = self.hmm_params.get("random_state", 42)
        max_iter = self.hmm_params.get("max_iter", 100)

        # PomegranateHMM models (single distribution)
        if variant == "hmm_pmgnt":
            return PomegranateHMM(
                distribution_type=DistType.NORMAL,
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_expnt":
            return PomegranateHMM(
                distribution_type=DistType.EXPONENTIAL,
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_gmma":
            return PomegranateHMM(
                distribution_type=DistType.GAMMA,
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_lambda":
            return PomegranateHMM(
                distribution_type=DistType.LAMDA,
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_lognorm":
            return PomegranateHMM(
                distribution_type=DistType.LOGNORMAL,
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_student":
            return PomegranateHMM(
                distribution_type=DistType.STUDENTT,
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
                **self.hmm_params.get("dist_kwargs", {}),
            )

        # PomegranateMixtureHMM models
        elif variant == "hmm_mm_pmgnt":
            return PomegranateMixtureHMM(
                distribution_type=DistType.NORMAL,
                n_states=n_states,
                n_components=config.get("n_components", 2),

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_mm_expnt":
            return PomegranateMixtureHMM(
                distribution_type=DistType.EXPONENTIAL,
                n_states=n_states,
                n_components=config.get("n_components", 2),

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_mm_gmma":
            return PomegranateMixtureHMM(
                distribution_type=DistType.GAMMA,
                n_states=n_states,
                n_components=config.get("n_components", 2),

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_mm_lambda":
            return PomegranateMixtureHMM(
                distribution_type=DistType.LAMDA,
                n_states=n_states,
                n_components=config.get("n_components", 2),

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_mm_lognorm":
            return PomegranateMixtureHMM(
                distribution_type=DistType.LOGNORMAL,
                n_components=config.get("n_components", 2),
                n_states=n_states,

                random_state=random_state,
                max_iter=max_iter,
            )
        elif variant == "hmm_mm_student":
            return PomegranateMixtureHMM(
                distribution_type=DistType.STUDENTT,
                n_states=n_states,
                n_components=config.get("n_components", 2),

                random_state=random_state,
                max_iter=max_iter,
                **self.hmm_params.get("dist_kwargs", {}),
            )
        else:
            raise ValueError(f"Unknown HMM variant: {variant}")

    def _optimize_hmm_model(self, train_features, train_data_raw=None):
        """Optimize HMM model via PARALLEL grid search over configurations.

        Args:
            train_features: Engineered features for training (numpy array or DataFrame)
            train_data_raw: Raw training data (optional, for state labeling)

        Returns:
            (best_model, best_config): Best HMM and its configuration
        """
        self.log.info("Optimizing HMM model (parallel)...", "🔍")

        # Convert DataFrame to numpy array if needed
        if isinstance(train_features, pd.DataFrame):
            train_features = train_features.to_numpy(dtype=np.float64)

        # Ensure features are 2D and numeric
        train_features = np.asarray(train_features, dtype=np.float64)
        if train_features.ndim == 1:
            train_features = train_features.reshape(-1, 1)

        configs = self._generate_hmm_grid()
        criterion = self.hmm_params["criterion"]

        self.log.info(f"🚀 Parallel grid search: {len(configs)} configurations", "")

        # PARALLEL TRAINING
        results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
            delayed(_train_single_hmm_config)(
                config, train_features, criterion, self.hmm_params
            )
            for config in configs
        )

        # TWO-STAGE SELECTION
        # Stage 1: Group results by variant and find best within each using AIC/BIC
        # Stage 2: Select across variants using clustering metrics (silhouette/davies_bouldin)

        results_by_variant = {}
        successful_count = 0
        failed_configs = []

        # Collect results by variant
        for config, model, score, aic, bic, silhouette, db_index, error in results:
            if error is None and model is not None:
                successful_count += 1
                variant = config["model_variant"]

                if variant not in results_by_variant:
                    results_by_variant[variant] = []

                results_by_variant[variant].append(
                    {
                        "config": config,
                        "model": model,
                        "score": score,
                        "aic": aic,
                        "bic": bic,
                        "silhouette": silhouette,
                        "db_index": db_index,
                    }
                )

                # Log each result
                n_states = config["n_states"]
                debug_msg = f"  ✓ {variant}, n_states={n_states}"
                if "n_components" in config:
                    debug_msg += f", n_components={config['n_components']}"
                debug_msg += f", {criterion.upper()}={score:.2f}, Sil={silhouette:.3f}"
                self.log.debug(debug_msg)
            else:
                failed_configs.append((config, error))
                self.log.debug(
                    f"  ✗ {config['model_variant']} n_states={config['n_states']}: {error}"
                )

        if not results_by_variant:
            raise RuntimeError(f"All {len(configs)} HMM configurations failed to train")

        # STAGE 1: Find best within each variant using AIC/BIC
        best_per_variant = []
        for variant, variant_results in results_by_variant.items():
            best_in_variant = min(variant_results, key=lambda x: x["score"])
            best_per_variant.append(best_in_variant)
            self.log.debug(
                f"  → Best {variant}: n_states={best_in_variant['config']['n_states']}, "
                f"{criterion.upper()}={best_in_variant['score']:.2f}"
            )

        # STAGE 2: Select across variants using clustering metrics
        if len(best_per_variant) > 1:
            # Multiple variants - use clustering metrics to decide
            cross_variant_metric = self.hmm_params.get(
                "cross_variant_metric", "silhouette"
            )
            self.log.debug(
                f"  → Selecting across variants using {cross_variant_metric}..."
            )

            silhouettes = np.array([r["silhouette"] for r in best_per_variant])
            db_indices = np.array([r["db_index"] for r in best_per_variant])

            # Handle edge cases in normalization
            def normalize(arr):
                arr_min, arr_max = arr.min(), arr.max()
                if arr_max - arr_min < 1e-10:
                    return np.zeros_like(arr)
                return (arr - arr_min) / (arr_max - arr_min)

            # Compute clustering scores based on metric choice
            if cross_variant_metric == "silhouette":
                # Use only silhouette (higher is better)
                clustering_scores = silhouettes
            elif cross_variant_metric == "db":
                # Use only Davies-Bouldin (lower is better, so negate)
                clustering_scores = -db_indices
            elif cross_variant_metric == "combined":
                # Use both: silhouette - normalized_db (higher is better)
                norm_db = normalize(db_indices)
                clustering_scores = silhouettes - norm_db
            else:
                # Default to silhouette
                self.log.debug(
                    f"  → Unknown metric '{cross_variant_metric}', defaulting to silhouette"
                )
                clustering_scores = silhouettes

            best_idx = np.argmax(clustering_scores)
            best_result = best_per_variant[best_idx]

            self.log.debug(
                f"  → Selected {best_result['config']['model_variant']} "
                f"(Sil={best_result['silhouette']:.3f}, DB={best_result['db_index']:.3f})"
            )
        else:
            # Only one variant - use it
            best_result = best_per_variant[0]

        # Extract best model and config
        best_model = best_result["model"]
        best_config = best_result["config"].copy()
        best_config["criterion_value"] = best_result["score"]
        best_config["aic"] = best_result["aic"]
        best_config["bic"] = best_result["bic"]
        best_config["silhouette"] = best_result["silhouette"]
        best_config["db_index"] = best_result["db_index"]

        if best_model is None:
            raise RuntimeError(f"All {len(configs)} HMM configurations failed to train")

        # Summary
        self.log.debug(
            f"Completed: {successful_count}/{len(configs)} configs trained successfully"
        )

        # Build info message
        info_msg = f"Best HMM: {best_config['model_variant']}, n_states={best_config['n_states']}"
        if "n_components" in best_config:
            info_msg += f", n_components={best_config['n_components']}"

        # Add metrics
        info_msg += f", {criterion.upper()}={best_config['criterion_value']:.2f}"
        info_msg += (
            f", Sil={best_config['silhouette']:.3f}, DB={best_config['db_index']:.3f}"
        )

        self.log.info(info_msg, "✓")
        return best_model, best_config


class SignalOptimizationMixin:
    """Mixin providing signal parameter optimization for walk-forward backtesting.

    Classes using this mixin must have:
        - self.signal_param_grid (Dict): Parameter grid for signal optimization
        - self.signal_param_optimizer (str): Optimizer type ('grid' or 'bayesian')
        - self.signal_param_n_calls (int): Number of Bayesian optimization calls
        - self.signal_optimization_metric (str): Metric to optimize
        - self.signal_generator_fn: Signal generation function
        - self.track_signal_param_importance (bool): Whether to track parameter importance
        - self.signal_param_importance_history (List): History of parameter importance
        - self.close_col (str): Name of close price column
        - self.log: Logger instance
        - self._run_backtest: Method to run backtest
        - self._extract_metric: Method to extract metric from portfolio
    """

    def _generate_signal_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of signal parameters from grid.

        Returns:
            List of parameter dictionaries
        """
        if not self.signal_param_grid:
            return [{}]

        keys = list(self.signal_param_grid.keys())
        values = [self.signal_param_grid[k] for k in keys]

        # Generate all combinations
        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations

    # Alias for compatibility with Keras classes
    def _generate_postprocess_param_combinations(self) -> List[Dict[str, Any]]:
        """Alias for _generate_signal_param_combinations for compatibility.

        Returns:
            List of parameter dictionaries
        """
        return self._generate_signal_param_combinations()

    def _optimize_signal_params_grid(
        self, predictions: np.ndarray, val_features, val_labels, val_prices
    ) -> Dict[str, Any]:
        """Optimize signal parameters using grid search.

        Args:
            predictions: Model predictions on validation set
            val_features: Validation features DataFrame
            val_labels: Validation labels Series
            val_prices: Validation prices DataFrame

        Returns:
            Best parameter combination
        """
        param_combinations = self._generate_signal_param_combinations()

        if not param_combinations or param_combinations == [{}]:
            self.log.debug("No signal parameter grid provided, using defaults")
            return {}

        best_metric = -np.inf
        best_params = {}
        param_performances = []

        self.log.debug(
            f"Testing {len(param_combinations)} signal parameter combinations"
        )

        for params in param_combinations:
            try:
                # Generate signals with current parameters
                signals = self.signal_generator_fn(
                    predictions, val_prices, val_features, **params
                )

                # Run backtest
                pf = self._run_backtest(signals, val_prices[self.close_col])

                # Extract metric
                metric_value = self._extract_metric(pf)

                # Track performance
                param_performances.append({"params": params, "metric": metric_value})

                # Update best
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()

            except Exception as e:
                self.log.debug(f"Signal params {params} failed: {e}")
                continue

        # Analyze parameter importance if tracking enabled
        if self.track_signal_param_importance and param_performances:
            self._analyze_param_importance(param_performances)

        self.log.debug(
            f"Best signal params: {best_params} ({self.signal_optimization_metric}={best_metric:.4f})"
        )
        return best_params

    def _optimize_signal_params_bayesian(
        self, predictions: np.ndarray, val_features, val_labels, val_prices
    ) -> Dict[str, Any]:
        """Optimize signal parameters using Bayesian optimization.

        Supports integer, float, boolean, and categorical parameters.

        Args:
            predictions: Model predictions on validation set
            val_features: Validation features DataFrame
            val_labels: Validation labels Series
            val_prices: Validation prices DataFrame

        Returns:
            Best parameter combination
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            self.log.warning(
                "scikit-optimize not available, falling back to grid search"
            )
            return self._optimize_signal_params_grid(
                predictions, val_features, val_labels, val_prices
            )

        if not self.signal_param_grid:
            self.log.debug("No signal parameter grid provided")
            return {}

        # Build search space
        space = []
        param_names = []
        param_performances = []

        for param_name, param_values in self.signal_param_grid.items():
            if not param_values:
                continue

            param_names.append(param_name)

            # Determine parameter type and create appropriate space
            if all(isinstance(v, (bool, np.bool_)) for v in param_values):
                # Boolean parameters
                space.append(Categorical(param_values, name=param_name))
            elif all(isinstance(v, (int, np.integer)) for v in param_values):
                # Integer parameters
                space.append(
                    Integer(min(param_values), max(param_values), name=param_name)
                )
            elif all(isinstance(v, (float, np.floating)) for v in param_values):
                # Float parameters
                space.append(
                    Real(min(param_values), max(param_values), name=param_name)
                )
            else:
                # Mixed or string types - use categorical
                space.append(Categorical(param_values, name=param_name))

        if not space:
            return {}

        self.log.debug(
            f"Bayesian optimization: {self.signal_param_n_calls} evaluations..."
        )

        # Define objective function
        def objective(params_list):
            params = dict(zip(param_names, params_list))

            try:
                signals = self.signal_generator_fn(
                    predictions, val_prices, val_features, **params
                )

                pf = self._run_backtest(signals, val_prices[self.close_col])
                metric_value = self._extract_metric(pf)

                # Handle invalid metrics (inf, -inf, nan)
                if not np.isfinite(metric_value):
                    self.log.debug(f"Invalid metric {metric_value} for params {params}")
                    return 1e10  # Large penalty

                # Cap extreme values to avoid numerical issues
                metric_value = np.clip(metric_value, -1e6, 1e6)

                # Store for importance tracking
                param_performances.append({"params": params, "metric": metric_value})

                # Return negative for minimization
                return -metric_value

            except Exception as e:
                self.log.debug(f"Signal params {params} failed: {e}")
                return 1e10  # Large penalty for failed params

        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.signal_param_n_calls,
            random_state=42,
            n_initial_points=min(10, self.signal_param_n_calls // 2),
            verbose=False,
        )

        best_params = dict(zip(param_names, result.x))
        best_metric = -result.fun

        self.log.debug(f"Bayesian optimization complete: best metric={best_metric:.4f}")

        # Analyze parameter importance if tracking enabled
        if self.track_signal_param_importance and len(param_performances) > 1:
            self._analyze_param_importance(param_performances)

        return best_params

    def _analyze_param_importance(
        self, param_performances: List[Dict[str, Any]]
    ) -> None:
        """Analyze which signal parameters have most impact on performance.

        Args:
            param_performances: List of dicts with 'params' and 'metric' keys
        """
        if len(param_performances) < 3:
            return

        # Convert to DataFrame
        df = pd.DataFrame(
            [{**p["params"], "metric": p["metric"]} for p in param_performances]
        )

        # Calculate variance-based importance
        param_importance = {}
        for col in df.columns:
            if col == "metric":
                continue

            # Group by parameter value and calculate mean metric
            grouped = df.groupby(col)["metric"].mean()

            # Importance = variance of mean metrics
            param_importance[col] = grouped.var()

        self.log.debug(f"Signal parameter importance: {param_importance}")

    def _track_signal_param_importance(
        self, window_idx: int, best_params: Dict[str, Any], best_metric: float
    ) -> None:
        """Track signal parameter selections across windows.

        Args:
            window_idx: Current window index
            best_params: Best parameters for this window
            best_metric: Metric value achieved
        """
        if not self.track_signal_param_importance:
            return

        self.signal_param_importance_history.append(
            {
                "window_idx": window_idx,
                "params": best_params.copy(),
                "metric_value": best_metric,
            }
        )


class ResultsAggregationMixin:
    """Mixin providing results aggregation and display for walk-forward optimizers.

    Classes using this mixin must have:
        - self.window_results (List): List of window results
        - self.all_predictions (pd.Series): All predictions
        - self.all_true_labels (pd.Series): All true labels
        - self.checkpoint_dir (Path): Directory for saving results
        - self.log: Logger instance
        - self.best_window_idx (Optional[int]): Best window index
        - self.best_metric_value (float): Best metric value
    """

    def _print_results_header(self) -> None:
        """Print results aggregation header."""
        print("\n" + "=" * 60)
        self.log.info("AGGREGATE RESULTS", "📊")
        print("=" * 60)

    def _create_results_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from window results.

        Returns:
            DataFrame with all window results
        """
        if not self.window_results:
            return pd.DataFrame()

        return pd.DataFrame([asdict(wr) for wr in self.window_results])

    def _print_classification_statistics(self, results_df: pd.DataFrame) -> None:
        """Print per-window classification statistics.

        Args:
            results_df: DataFrame with window results
        """
        print("\nPer-Window Classification Statistics:")
        print(
            f"  Average Accuracy:  {results_df['accuracy'].mean():.4f}, "
            f"Average Precision: {results_df['precision'].mean():.4f}, "
            f"Average Recall:    {results_df['recall'].mean():.4f}, "
            f"Average F1-Score:  {results_df['f1_score'].mean():.4f}, "
            f"Average AUC-ROC:   {results_df['auc_roc'].mean():.4f}"
        )

    def _print_regression_statistics(self, results_df: pd.DataFrame) -> None:
        """Print per-window regression statistics.

        Args:
            results_df: DataFrame with window results
        """
        print("\nPer-Window Regression Statistics:")
        print(
            f"  MSE:   Mean={results_df['mse'].mean():.4f}, Std={results_df['mse'].std():.4f}, "
            f"RMSE:  Mean={results_df['rmse'].mean():.4f}, Std={results_df['rmse'].std():.4f}"
        )
        print(
            f"  MAE:   Mean={results_df['mae'].mean():.4f}, Std={results_df['mae'].std():.4f}, "
            f"R²:    Mean={results_df['r2'].mean():.4f}, Std={results_df['r2'].std():.4f}, "
            f"MAPE:  Mean={results_df['mape'].mean():.2f}%, Std={results_df['mape'].std():.2f}%"
        )

    def _print_trading_statistics(
        self, results_df: pd.DataFrame, detailed: bool = False
    ) -> None:
        """Print per-window trading statistics.

        Args:
            results_df: DataFrame with window results
            detailed: Whether to print advanced trading metrics
        """
        print("\nPer-Window Trading Statistics:")
        print(
            f"  Average Sharpe Ratio: {results_df['sharpe_ratio'].mean():.2f}, "
            f"Average Return:       {results_df['total_return'].mean():.2f}%, "
            f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%, "
            f"Average Win Rate:     {results_df['win_rate'].mean():.2f}%, "
            f"Total Trades:         {results_df['num_trades'].sum()}"
        )

        if detailed and "sortino_ratio" in results_df.columns:
            print("\nAdvanced Trading Statistics:")
            print(
                f"  Average Sortino:      {results_df['sortino_ratio'].mean():.2f}, "
                f"Average Calmar:       {results_df['calmar_ratio'].mean():.2f}, "
                f"Average Profit Factor:{results_df['profit_factor'].mean():.2f}"
            )

    def _print_hmm_statistics(self, results_df: pd.DataFrame) -> None:
        """Print per-window HMM statistics.

        Args:
            results_df: DataFrame with window results
        """
        print("\nPer-Window HMM Statistics:")
        print(
            f"  Average AIC:       {results_df['aic_score'].mean():.2f}, "
            f"Average BIC:       {results_df['bic_score'].mean():.2f}"
        )
        print(
            f"  Average Entropy:   {results_df['avg_state_entropy'].mean():.3f}, "
            f"Average Stability: {results_df['state_stability'].mean():.3f}"
        )

    def _print_ensemble_statistics(self, results_df: pd.DataFrame) -> None:
        """Print ensemble statistics if applicable.

        Args:
            results_df: DataFrame with window results
        """
        if hasattr(self, "ensemble_size") and self.ensemble_size > 1:
            print(f"\nEnsemble Statistics:")
            print(f"  Average ensemble size: {results_df['ensemble_size'].mean():.1f}")

    def _print_early_stopping_statistics(self, results_df: pd.DataFrame) -> None:
        """Print early stopping statistics if applicable.

        Args:
            results_df: DataFrame with window results
        """
        if hasattr(self, "enable_early_stopping") and self.enable_early_stopping:
            early_stopped_count = results_df["early_stopped"].sum()
            print(f"\nEarly Stopping Statistics:")
            print(f"  Windows early stopped: {early_stopped_count}/{len(results_df)}")
            if early_stopped_count > 0:
                avg_convergence = results_df[results_df["early_stopped"]][
                    "convergence_epoch"
                ].mean()
                print(f"  Average convergence epoch: {avg_convergence:.1f}")

    def _print_best_model_info(self) -> None:
        """Print best model selection information."""
        if self.best_window_idx is not None:
            print(f"\nBest Models:")
            if hasattr(self, "best_model_selection"):
                print(f"  Strategy: {self.best_model_selection}")
            print(f"  Selected from window: {self.best_window_idx}")
            if hasattr(self, "best_model_metric"):
                print(f"  Best {self.best_model_metric}: {self.best_metric_value:.4f}")

    def _print_feature_importance_summary(self) -> None:
        """Print feature importance summary if available."""
        if hasattr(self, "track_feature_importance") and self.track_feature_importance:
            if (
                hasattr(self, "feature_importance_history")
                and self.feature_importance_history
            ):
                print(f"\nFeature Importance:")
                print(
                    f"  Tracked across {len(self.feature_importance_history)} windows"
                )
                if hasattr(self, "_summarize_feature_importance"):
                    self._summarize_feature_importance()

    def _print_signal_param_summary(self) -> None:
        """Print signal parameter importance summary if available."""
        if (
            hasattr(self, "track_signal_param_importance")
            and self.track_signal_param_importance
        ):
            if (
                hasattr(self, "signal_param_importance_history")
                and self.signal_param_importance_history
            ):
                print(f"\nSignal Parameter Selection:")
                print(
                    f"  Tracked across {len(self.signal_param_importance_history)} windows"
                )
                if hasattr(self, "_summarize_signal_param_importance"):
                    self._summarize_signal_param_importance()

    def _save_results_files(
        self,
        results_df: pd.DataFrame,
        save_predictions: bool = True,
        save_returns: bool = False,
        save_signals: bool = False,
    ) -> None:
        """Save results to CSV and JSON files.

        Args:
            results_df: DataFrame with window results
            save_predictions: Whether to save predictions file
            save_returns: Whether to save returns file
            save_signals: Whether to save signals
        """
        try:
            # Save main results
            results_file = self.checkpoint_dir / "final_results.csv"
            results_df.to_csv(results_file, index=False)
            self.log.info(f"Results saved to: {results_file}", "💾")

            # Save predictions
            if save_predictions:
                predictions_df = pd.DataFrame(
                    {
                        "predictions": self.all_predictions,
                        "true_labels": self.all_true_labels,
                    }
                )

                # Add signals if available
                if (
                    save_signals
                    and hasattr(self, "all_signals")
                    and self.all_signals is not None
                ):
                    predictions_df = predictions_df.join(self.all_signals, how="left")
                    predictions_file = (
                        self.checkpoint_dir / "predictions_and_signals.csv"
                    )
                else:
                    predictions_file = self.checkpoint_dir / "predictions.csv"

                predictions_df.to_csv(predictions_file)
                self.log.info(f"Predictions saved to: {predictions_file}", "💾")

            # Save returns
            if (
                save_returns
                and hasattr(self, "all_returns")
                and len(self.all_returns) > 0
            ):
                returns_file = self.checkpoint_dir / "returns.csv"
                self.all_returns.to_csv(returns_file)
                self.log.info(f"Returns saved to: {returns_file}", "💾")

            # Save feature importance
            if (
                hasattr(self, "feature_importance_history")
                and self.feature_importance_history
            ):
                importance_file = self.checkpoint_dir / "feature_importance.json"
                with open(importance_file, "w") as f:
                    json.dump(
                        self.feature_importance_history,
                        f,
                        indent=2,
                        default=to_serializable,
                    )
                self.log.info(f"Feature importance saved to: {importance_file}", "💾")

            # Save signal parameter importance
            if (
                hasattr(self, "signal_param_importance_history")
                and self.signal_param_importance_history
            ):
                signal_importance_file = (
                    self.checkpoint_dir / "signal_param_importance.json"
                )
                with open(signal_importance_file, "w") as f:
                    json.dump(
                        self.signal_param_importance_history,
                        f,
                        indent=2,
                        default=to_serializable,
                    )
                self.log.info(
                    f"Signal param importance saved to: {signal_importance_file}", "💾"
                )

        except Exception as e:
            self.log.warning(f"Could not save results: {e}")
