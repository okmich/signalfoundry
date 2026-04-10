import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, Any

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt
import xgboost
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from .hmm_clustering_comparison_backtesting_pipeline import HMM_ALGOS
from .vbt_export import QuantframeExportMixin
from .wfa_plot_utils import HMMVisualizationMixin
from .wfa_utils import to_serializable, CheckpointingMixin, WindowGenerationMixin, MetricsTrackingMixin, \
    BacktestingMixin, SignalOptimizationMixin, HMMMetricsMixin, HMMModelSelectionMixin, ResultsAggregationMixin
from okmich_quant_research.env_utils import UniversalLogger, EnvironmentDetector

# Set default frequency for vectorbt to avoid frequency inference errors
vbt.settings.array_wrapper["freq"] = "m"

# Type aliases for signal generator outputs
SignalArrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class HMMWindowResult:
    """Stores results from a single HMM window."""

    # Window info
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # HMM model info
    n_states: int
    model_variant: str = (
        ""  # HMM variant (e.g., 'hmm_learn', 'hmm_pmgnt', 'hmm_mm_student')
    )
    n_components: int = 0  # For mixture models
    covariance_type: str = (
        ""  # For hmmlearn (deprecated, kept for backward compatibility)
    )
    distribution_type: str = (
        ""  # For pomegranate (deprecated, kept for backward compatibility)
    )
    best_model_criterion_value: float = np.nan  # AIC or BIC value

    # Model selection metrics
    aic_score: float = np.nan
    bic_score: float = np.nan
    log_likelihood: float = np.nan

    # State prediction quality
    avg_state_entropy: float = np.nan
    state_stability: float = np.nan  # Proportion of non-transitions

    # State duration statistics
    avg_state_duration: float = np.nan
    median_state_duration: float = np.nan
    per_state_duration_stats: Optional[Dict[int, Dict[str, float]]] = None

    # Signal parameters
    best_signal_params: Dict = None

    # Trading metrics (from backtest)
    sharpe_ratio: float = np.nan
    total_return: float = np.nan
    max_drawdown: float = np.nan
    win_rate: float = np.nan
    num_trades: int = 0
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    profit_factor: float = np.nan
    avg_trade_duration: float = np.nan

    # State mapping
    state_map: Optional[Dict[int, str]] = None

    # Feature importance
    feature_importance: Optional[Dict] = None

    # Convergence tracking
    metric_trend: Optional[str] = None


class HMMWalkForwardAnalysisBacktestOptimizer(CheckpointingMixin, WindowGenerationMixin, MetricsTrackingMixin,
                                              BacktestingMixin, SignalOptimizationMixin, HMMMetricsMixin,
                                              HMMModelSelectionMixin, ResultsAggregationMixin, HMMVisualizationMixin,
                                              QuantframeExportMixin):
    """
    HMMWalkForwardAnalysisBacktestOptimizer: Advanced walk-forward analysis with HMMs and backtesting.

    This class combines Hidden Markov Model optimization with portfolio backtesting using VectorBT.
    It performs walk-forward analysis where each window:
    1. Optimizes the HMM model (number of states, covariance/distribution type)
    2. Labels states with semantic meaning (via state_labeling_fn)
    3. Optimizes signal generation parameters via backtesting
    4. Evaluates both HMM quality metrics and trading performance
    5. Tracks comprehensive metrics across market regimes

    Key Features:
    - Dual optimization: HMM model parameters + signal parameters
    - Grid search over n_states and covariance_type (hmmlearn) or distribution_type (pomegranate)
    - State labeling: Map arbitrary state IDs to semantic regime names
    - Comprehensive metrics: HMM quality (AIC, BIC, entropy, stability) + trading metrics
    - Feature importance tracking across regimes
    - Signal parameter importance analysis
    - Portfolio convergence monitoring
    - Checkpointing and resume capability

    HMM Model Parameters:
    The hmm_params dict controls model selection. Examples:

    ```python
    # Example 1: Grid search over multiple HMM variants and n_states
    from okmich_quant_research.backtesting.hmm_clustering_comparison_backtesting_pipeline import HMM_ALGOS

    hmm_params = {
        'model_variants': ['hmm_learn', 'hmm_pmgnt', 'hmm_student'],
        'n_states_range': [2, 3, 4, 5],
        'criterion': 'aic',  # or 'bic'
        'random_state': 42,
        'covariance_type': 'full',  # for hmm_learn
        'dist_kwargs': {'dofs': 3}  # for hmm_student
    }

    # Example 2: Mixture models with grid search over n_components
    hmm_params = {
        'model_variants': ['hmm_mm_pmgnt', 'hmm_mm_student'],
        'n_states_range': [2, 3, 4],
        'n_components_range': [2, 3],  # Grid search over n_components
        'criterion': 'bic',
        'random_state': 100,
        'dist_kwargs': {'dofs': 3}
    }

    # Example 3: Compare all available HMM types
    hmm_params = {
        'model_variants': HMM_ALGOS,  # All HMM variants
        'n_states_range': [2, 3, 4],
        'criterion': 'aic'
    }
    ```

    State Labeling Function:
    The state_labeling_fn maps HMM states to semantic regime labels:

    ```python
    def state_labeling_fn(model, train_data_raw, train_features,
                         state_predictions, state_probabilities):
        '''
        Map HMM states to semantic regime labels.

        Args:
            model: Trained HMM (GaussianHMM or PomegranateHMM)
            train_data_raw: Original raw training data (OHLCV) - NOT scaled
            train_features: Engineered features (may be scaled)
            state_predictions: Predicted states on training data (1D array)
            state_probabilities: State probabilities (2D array: n_samples x n_states)

        Returns:
            state_map: Dict mapping state_id -> regime_name
        '''
        state_map = {}
        for state_id in range(model.n_states):
            state_mask = (state_predictions == state_id)
            state_returns = train_data_raw['close'].pct_change()[state_mask]
            state_volatility = state_returns.std()

            if state_volatility < 0.01:
                state_map[state_id] = 'low_volatility'
            else:
                state_map[state_id] = 'high_volatility'

        return state_map
    ```

    Signal Generator Function:
    The signal_generator_fn uses state_map to generate trading signals:

    ```python
    def signal_generator_fn(state_predictions, state_probabilities, prices,
                           features, state_map, **params):
        '''
        Generate trading signals from HMM regime predictions.

        Args:
            state_predictions: Most likely state sequence (1D array)
            state_probabilities: State probabilities (2D array)
            prices: Raw price data (DataFrame)
            features: Engineered features
            state_map: Dict from state_labeling_fn
            **params: Signal parameters

        Returns:
            (long_entries, long_exits, short_entries, short_exits): 4-tuple
        '''
        confidence = params.get('confidence', 0.7)

        # Use semantic labels
        bullish_states = [sid for sid, label in state_map.items()
                         if 'bullish' in label.lower()]

        is_bullish = np.isin(state_predictions, bullish_states)
        is_confident = state_probabilities.max(axis=1) > confidence

        long_entries = is_bullish & is_confident
        long_exits = ~is_bullish
        short_entries = ~is_bullish & is_confident
        short_exits = is_bullish

        return long_entries, long_exits, short_entries, short_exits
    ```

    Typical Usage:
    ```python
    def feature_engineering_fn(train_raw, test_raw, train_labels, test_labels):
        # Engineer features from raw OHLCV data
        return train_features, test_features, train_labels, test_labels

    optimizer = HMMWalkForwardAnalysisBacktestOptimizer(
        raw_data=ohlcv_df,
        train_period=252,
        test_period=63,
        step_period=21,
        feature_engineering_fn=feature_engineering_fn,
        hmm_params={
            'model_variants': ['hmm_learn', 'hmm_pmgnt'],
            'n_states_range': [2, 3, 4],
            'criterion': 'aic',
            'covariance_type': 'full'  # for hmm_learn
        },
        state_labeling_fn=state_labeling_fn,
        signal_generator_fn=signal_generator_fn,
        signal_param_grid={
            'confidence': [0.6, 0.7, 0.8]
        },
        signal_optimization_metric='sharpe_ratio',
        vbt_params={'fees': 0.001},
        checkpoint_dir='./checkpoints'
    )

    results_df, predictions, state_labels = optimizer.run()
    best_models = optimizer.get_best_models()
    ```

    Args:
        raw_data: OHLCV DataFrame with DatetimeIndex (must contain close_col)
        train_period: Number of samples in training window
        test_period: Number of samples in test window
        step_period: Number of samples to step forward each window
        feature_engineering_fn: Feature transformation function
        hmm_params: Dict with HMM configuration (see examples above)
        state_labeling_fn: Optional function to map states to regime names
        signal_generator_fn: Converts predictions to trading signals (4-tuple)
        signal_param_grid: Dict of signal parameters to optimize
        signal_param_optimizer: 'grid', 'random', or 'bayesian'
        signal_param_n_calls: Number of evaluations for bayesian optimization
        signal_optimization_metric: Metric to optimize ('sharpe_ratio', 'return', etc.)
        vbt_params: VectorBT portfolio parameters (fees, slippage, etc.)
        close_col: Column name for close prices in raw_data
        track_feature_importance: Track feature importance across windows
        track_signal_param_importance: Track signal parameter patterns
        checkpoint_dir: Directory for saving checkpoints and results
        anchored: If True, training window grows; if False, rolls forward
        convergence_window_size: Number of windows for convergence detection
        convergence_threshold: Threshold for detecting performance trends
        portfolio_convergence_metric: Metric for convergence ('sharpe'/'return'/'drawdown')
        verbose: Logging verbosity (0=silent, 1=info, 2=debug)

    Outputs:
        - window_results: List[HMMWindowResult] with all metrics
        - all_predictions: pd.Series of state predictions
        - all_true_labels: pd.Series (placeholder for future classification metrics)
        - all_signals: pd.DataFrame of trading signals
        - all_returns: pd.Series of portfolio returns
    """

    def __init__(self, raw_data: pd.DataFrame, train_period: int, test_period: int, step_period: int,
                 feature_engineering_fn: Callable, hmm_params: Dict[str, Any],  # HMM configuration
                 state_labeling_fn: Optional[Callable] = None,  # State labeling
                 signal_generator_fn: Callable = None,  # Signal generation
                 signal_param_grid: Optional[Dict[str, List[Any]]] = None, signal_param_optimizer: str = "grid",
                 signal_param_n_calls: int = 50, signal_optimization_metric: str = "sharpe_ratio",
                 vbt_params: Optional[Dict] = None,  close_col: str = "close",
                 # Tracking and management
                 track_feature_importance: bool = True, track_signal_param_importance: bool = True,
                 checkpoint_dir: Optional[str] = None, anchored: bool = False, convergence_window_size: int = 3,
                 convergence_threshold: float = 0.05, portfolio_convergence_metric: str = "sharpe", verbose: int = 1):
        self.log = UniversalLogger(verbose)
        self.env = EnvironmentDetector()

        # Validate price data
        if close_col not in raw_data.columns:
            raise ValueError(f"raw_data must have '{close_col}' column")

        # Core WFA parameters
        self.raw_data = raw_data
        self.train_period = train_period
        self.test_period = test_period
        self.step_period = step_period
        self.anchored = anchored

        # Feature engineering
        self.feature_engineering_fn = feature_engineering_fn

        # HMM configuration
        self.hmm_params = hmm_params
        self._validate_hmm_params()

        # State labeling
        self.state_labeling_fn = state_labeling_fn

        # Signal generation
        self.signal_generator_fn = signal_generator_fn
        self.signal_param_grid = signal_param_grid or {}
        self.signal_param_optimizer = signal_param_optimizer
        self.signal_param_n_calls = signal_param_n_calls
        self.signal_optimization_metric = signal_optimization_metric

        # Backtesting
        self.vbt_params = vbt_params or {"fees": 0.001, "slippage": 0.001}
        self.close_col = close_col

        # Tracking
        self.track_feature_importance = track_feature_importance
        self.track_signal_param_importance = track_signal_param_importance
        self.convergence_window_size = convergence_window_size
        self.convergence_threshold = convergence_threshold
        self.portfolio_convergence_metric = portfolio_convergence_metric

        # Checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = self.env.get_default_checkpoint_dir(checkpoint_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.window_results: List[HMMWindowResult] = []
        self.all_predictions = pd.Series(dtype=int)  # State predictions
        self.all_state_probabilities = None  # Will be DataFrame
        self.all_true_labels = pd.Series(dtype=float)  # Placeholder for future
        self.all_signals = None  # Will be DataFrame
        self.all_returns = pd.Series(dtype=float)
        self.window_portfolios: List[vbt.Portfolio] = []
        self.window_scalers: List = []  # Feature scalers per window

        # Best model tracking
        self.best_window_idx = None
        self.best_metric_value = -np.inf
        self.best_scaler = None  # Scaler from best window

        # Importance tracking
        self.feature_importance_history: List[Dict] = []
        self.signal_param_importance_history: List[Dict] = []

        self.log.info("HMMWalkForwardAnalysisBacktestOptimizer initialized", "✓")
        self.log.info(f"Environment: {'Colab' if self.env.is_colab() else 'Jupyter' if self.env.is_jupyter() else 'CLI'}")
        self.log.info(f"HMM model variants: {hmm_params.get('model_variants', [])}")
        self.log.info(f"Signal optimizer: {signal_param_optimizer}")

        if self.track_feature_importance:
            self.log.info("Feature importance tracking enabled")
        if self.track_signal_param_importance:
            self.log.info("Signal parameter importance tracking enabled")

    def _validate_hmm_params(self) -> None:
        """Validate HMM parameters."""
        required = ["model_variants", "n_states_range", "criterion"]
        for param in required:
            if param not in self.hmm_params:
                raise ValueError(f"hmm_params must contain '{param}'")

        model_variants = self.hmm_params["model_variants"]
        if not isinstance(model_variants, list):
            raise ValueError(f"model_variants must be a list, got {type(model_variants)}")

        if len(model_variants) == 0:
            raise ValueError("model_variants cannot be empty")

        for variant in model_variants:
            if variant not in HMM_ALGOS:
                raise ValueError(f"Unknown HMM variant '{variant}'. Must be one of: {HMM_ALGOS}")

    def _label_states(self, model, train_data_raw: pd.DataFrame, train_features: np.ndarray) -> Dict[int, str]:
        """
        Label HMM states with semantic regime names.

        Args:
            model: Trained HMM model
            train_data_raw: Original raw training data (OHLCV)
            train_features: Engineered features

        Returns:
            state_map: Dict mapping state_id -> regime_name
        """
        if self.state_labeling_fn is None:
            # Default: just number the states
            return {i: f"state_{i}" for i in range(model.n_states)}

        # Convert DataFrame to numpy array if needed
        if isinstance(train_features, pd.DataFrame):
            train_features = train_features.values

        # Ensure features are 2D
        if train_features.ndim == 1:
            train_features = train_features.reshape(-1, 1)

        # Get predictions on training data
        state_predictions = model.predict(train_features)
        state_probabilities = model.predict_proba(train_features)

        # Call user-defined labeling function
        state_map = self.state_labeling_fn(
            model,
            train_data_raw,
            train_features,
            state_predictions,
            state_probabilities,
        )
        self.log.debug(f"State mapping: {state_map}")
        return state_map

    def _calculate_hmm_metrics(self, model, features: np.ndarray, state_predictions: np.ndarray,
                               state_probabilities: np.ndarray, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive HMM quality metrics.

        Args:
            model: Trained HMM model
            features: Features used for prediction
            state_predictions: Predicted state sequence
            state_probabilities: State probability matrix
            config: Model configuration with AIC/BIC

        Returns:
            Dict with all HMM metrics
        """
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        metrics = {
            "aic_score": config.get("aic", np.nan),
            "bic_score": config.get("bic", np.nan),
            "best_model_criterion_value": config.get("criterion_value", np.nan),
            "avg_state_entropy": self._calculate_state_entropy(state_probabilities),
            "state_stability": self._calculate_state_stability(state_predictions),
        }

        # State duration statistics
        # Extract prices from features or model data if available
        # For now, use synthetic prices - ideally should pass actual price data
        duration_stats = self._calculate_state_durations(state_predictions, prices=None)
        metrics.update(duration_stats)

        # Log likelihood
        try:
            # For hmmlearn
            if hasattr(model, "_model") and hasattr(model._model, "score"):
                log_likelihood = model._model.score(features)
            # For pomegranate
            elif hasattr(model, "_model") and hasattr(model._model, "log_probability"):
                log_likelihood = model._model.log_probability([features]).sum()
            else:
                log_likelihood = np.nan

            metrics["log_likelihood"] = float(log_likelihood)
        except Exception as e:
            self.log.debug(f"Could not calculate log likelihood: {e}")
            metrics["log_likelihood"] = np.nan

        return metrics

    def _create_window_result(self, **kwargs) -> HMMWindowResult:
        """
        Hook for checkpoint loading to create correct dataclass type.
        """
        return HMMWindowResult(**kwargs)

    def _extract_feature_importance(self, model, features: np.ndarray, feature_names: Optional[List[str]] = None,
                                    state_predictions: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Extract feature importance using three methods:
        1. Emission-based: Variance of emission means across states
        2. Random Forest: Feature importance from predicting states
        3. XGBoost: Feature importance from predicting states (if available)

        Args:
            model: Trained HMM model
            features: Feature array
            feature_names: Optional list of feature names
            state_predictions: Predicted states (for tree-based methods)

        Returns:
            Dict with three importance dictionaries and correlation scores:
            {
                'emission_based': {...},
                'random_forest': {...},
                'xgboost': {...} or None,
                'correlation_emission_rf': float,
                'correlation_emission_xgb': float or None,
                'correlation_rf_xgb': float or None
            }
        """
        if not self.track_feature_importance:
            return None

        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        n_features = features.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        result = {}

        # Method 1: Emission-based importance
        try:
            emission_importance = self._emission_based_importance(model, features, feature_names)
            result["emission_based"] = emission_importance
        except Exception as e:
            self.log.debug(f"Could not extract emission-based importance: {e}")
            result["emission_based"] = None

        # Method 2: Random Forest importance
        if state_predictions is not None and len(state_predictions) > 0:
            try:
                rf_importance = self._random_forest_importance(features, state_predictions, feature_names)
                result["random_forest"] = rf_importance
            except Exception as e:
                self.log.debug(f"Could not extract Random Forest importance: {e}")
                result["random_forest"] = None
        else:
            result["random_forest"] = None

        # Method 3: XGBoost importance (if available)
        if state_predictions is not None and len(state_predictions) > 0:
            try:
                xgb_importance = self._xgboost_importance(features, state_predictions, feature_names)
                result["xgboost"] = xgb_importance
            except Exception as e:
                self.log.debug(f"Could not extract XGBoost importance (may not be installed): {e}")
                result["xgboost"] = None
        else:
            result["xgboost"] = None

        # Calculate correlations between methods
        result["correlation_emission_rf"] = self._calculate_importance_correlation(
            result["emission_based"], result["random_forest"])
        result["correlation_emission_xgb"] = self._calculate_importance_correlation(
            result["emission_based"], result["xgboost"])
        result["correlation_rf_xgb"] = self._calculate_importance_correlation(
            result["random_forest"], result["xgboost"])

        return result

    def _emission_based_importance(self, model, features: np.ndarray, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extract feature importance based on emission distribution parameters.
        Features with high variance of means across states are important.
        """
        try:
            # Get emission distribution means
            means, _ = model.means()  # Shape: (n_states, n_features)
            if means is None or len(means) == 0:
                return None

            # Convert to numpy array if needed
            if isinstance(means, list):
                means = np.array(means)

            if means.ndim == 1:
                means = means.reshape(-1, 1)

            n_features = means.shape[1] if means.ndim > 1 else 1

            importance = {}
            for feat_idx, feat_name in enumerate(feature_names):
                if means.ndim > 1 and feat_idx < means.shape[1]:
                    # Variance of this feature's means across states
                    feat_means = means[:, feat_idx]
                    importance[feat_name] = float(np.var(feat_means))
                else:
                    importance[feat_name] = 0.0

            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            return importance
        except Exception as e:
            self.log.debug(f"Emission-based importance failed: {e}")
            return None

    def _random_forest_importance(self, features: np.ndarray, state_predictions: np.ndarray,
                                  feature_names: List[str]) -> Optional[Dict[str, float]]:
        try:
            # Ensure alignment
            n_samples = min(len(features), len(state_predictions))
            X = features[:n_samples]
            y = state_predictions[:n_samples]

            # Need at least 2 samples per class
            if len(np.unique(y)) < 2 or len(y) < 10:
                return None

            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
            rf.fit(X, y)

            # Extract feature importances
            importances = rf.feature_importances_
            importance = {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
            return importance
        except ImportError:
            self.log.debug("sklearn not available for Random Forest importance")
            return None
        except Exception as e:
            self.log.debug(f"Random Forest importance failed: {e}")
            return None

    def _xgboost_importance(self, features: np.ndarray, state_predictions: np.ndarray,
                            feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Extract feature importance using XGBoost classifier.
        Train XGBoost to predict states from features.
        """
        try:
            # Ensure alignment
            n_samples = min(len(features), len(state_predictions))
            X = features[:n_samples]
            y = state_predictions[:n_samples]

            # Need at least 2 samples per class
            if len(np.unique(y)) < 2 or len(y) < 10:
                return None

            # Train XGBoost
            model = xgboost.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
                                          n_jobs=-1, verbosity=0)
            model.fit(X, y)

            importance_dict = model.get_booster().get_score(importance_type="weight")
            importance = {}
            for i, feat_name in enumerate(feature_names):
                # XGBoost uses f0, f1, f2 naming
                xgb_name = f"f{i}"
                importance[feat_name] = float(importance_dict.get(xgb_name, 0.0))

            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            return importance
        except ImportError:
            self.log.debug("xgboost not available for XGBoost importance")
            return None
        except Exception as e:
            self.log.debug(f"XGBoost importance failed: {e}")
            return None

    def _calculate_importance_correlation(self, importance1: Optional[Dict[str, float]],
                                          importance2: Optional[Dict[str, float]]) -> Optional[float]:
        """
        Calculate Pearson correlation between two importance dictionaries.

        Returns:
            Correlation coefficient (0-1) or None if either is None
        """
        if importance1 is None or importance2 is None:
            return None
        try:
            # Get common features
            common_features = set(importance1.keys()) & set(importance2.keys())
            if len(common_features) < 2:
                return None

            # Extract values in same order
            values1 = [importance1[f] for f in sorted(common_features)]
            values2 = [importance2[f] for f in sorted(common_features)]

            # Calculate Pearson correlation
            correlation = np.corrcoef(values1, values2)[0, 1]

            return float(correlation) if np.isfinite(correlation) else None

        except Exception as e:
            self.log.debug(f"Could not calculate importance correlation: {e}")
            return None

    def _optimize_signal_params_grid(self, state_predictions: np.ndarray, state_probabilities: np.ndarray,
                                     val_features: pd.DataFrame, val_prices: pd.DataFrame,
                                     state_map: Dict[int, str]) -> Dict[str, Any]:
        """Grid search over signal parameter combinations."""
        param_combinations = self._generate_signal_param_combinations()

        if len(param_combinations) == 1 and not param_combinations[0]:
            return {}

        self.log.debug(f"Grid search: testing {len(param_combinations)} signal param combinations...")

        best_params = None
        best_metric = -np.inf
        param_performances = []

        for params in param_combinations:
            try:
                signals = self.signal_generator_fn(state_predictions, state_probabilities, val_prices, val_features,
                                                   state_map, **params)

                pf = self._run_backtest(signals, val_prices[self.close_col])
                metric = self._extract_metric(pf)

                param_performances.append({"params": params, "metric": metric})

                if metric > best_metric:
                    best_metric = metric
                    best_params = params

            except Exception as e:
                self.log.debug(f"Failed params {params}: {e}")
                continue

        if self.track_signal_param_importance and len(param_performances) > 1:
            self._analyze_param_importance(param_performances)

        return best_params or {}

    def _optimize_signal_params_bayesian(self, state_predictions: np.ndarray, state_probabilities: np.ndarray,
                                         val_features: pd.DataFrame, val_prices: pd.DataFrame,
                                         state_map: Dict[int, str]) -> Dict[str, Any]:
        """Bayesian optimization using scikit-optimize."""
        if not self.signal_param_grid:
            return {}

        # Build search space
        space = []
        param_names = []
        for param_name, param_values in self.signal_param_grid.items():
            param_names.append(param_name)

            # Determine parameter type and create appropriate space
            if all(isinstance(v, (bool, np.bool_)) for v in param_values):
                space.append(Categorical(param_values, name=param_name))
            elif all(isinstance(v, (int, np.integer)) for v in param_values):
                space.append(Integer(min(param_values), max(param_values), name=param_name))
            elif all(isinstance(v, (float, np.floating)) for v in param_values):
                space.append(Real(min(param_values), max(param_values), name=param_name))
            else:
                # Mixed or string types - use categorical
                space.append(Categorical(param_values, name=param_name))

        self.log.debug(f"Bayesian optimization: {self.signal_param_n_calls} evaluations...")

        # Store performances for importance tracking
        param_performances = []

        # Objective function
        def objective(param_values):
            params = dict(zip(param_names, param_values))

            try:
                signals = self.signal_generator_fn(state_predictions, state_probabilities, val_prices, val_features,
                                                   state_map, **params)

                pf = self._run_backtest(signals, val_prices[self.close_col])
                metric = self._extract_metric(pf)

                # Handle invalid metrics
                if not np.isfinite(metric):
                    self.log.debug(f"Invalid metric {metric} for params {params}")
                    return 1e10

                # Cap extreme values
                metric = np.clip(metric, -1e6, 1e6)

                # Store for importance tracking
                param_performances.append({"params": params, "metric": metric})

                return -metric  # Minimize (negative for maximization)

            except Exception as e:
                self.log.debug(f"Failed params {params}: {e}")
                return 1e10

        # Run optimization
        result = gp_minimize(objective, space, n_calls=self.signal_param_n_calls, random_state=42,
                             verbose=False, n_initial_points=min(10, self.signal_param_n_calls // 2))

        best_params = dict(zip(param_names, result.x))
        best_metric = -result.fun

        self.log.debug(f"Bayesian optimization complete: best metric={best_metric:.4f}")

        if self.track_signal_param_importance and len(param_performances) > 1:
            self._analyze_param_importance(param_performances)

        return best_params

    def _run_window(self, window_idx: int, train_start_idx: int, train_end_idx: int, test_start_idx: int,
                    test_end_idx: int) -> HMMWindowResult:
        """
        Execute training and evaluation for a single window.
        """
        print(f"\n{'=' * 60}")
        self.log.info(f"Window {window_idx}", "🔄")
        print(f"{'=' * 60}")

        # Slice data
        train_data_raw = self.raw_data.iloc[train_start_idx:train_end_idx]
        test_data_raw = self.raw_data.iloc[test_start_idx:test_end_idx]

        print(
            f"Training: {train_data_raw.index[0].date()} to {train_data_raw.index[-1].date()} "
            f"({len(train_data_raw)} samples)"
        )
        print(
            f"Testing:  {test_data_raw.index[0].date()} to {test_data_raw.index[-1].date()} "
            f"({len(test_data_raw)} samples)"
        )

        # Feature engineering (no labels for HMM - placeholder None)
        train_features, test_features, train_metadata, _ = self.feature_engineering_fn(
            train_data_raw, test_data_raw, None, None)
        self.log.debug(f"Transformed shapes - Train: {train_features.shape}, Test: {test_features.shape}")

        # Extract scaler if provided in metadata
        window_scaler = None
        if train_metadata is not None and isinstance(train_metadata, dict):
            window_scaler = train_metadata.get("scaler")

        # Step 1: Optimize HMM model
        best_model, best_config = self._optimize_hmm_model(
            train_features, train_data_raw
        )

        # Step 2: Label states
        state_map = self._label_states(best_model, train_data_raw, train_features)
        self.log.info(f"State mapping: {state_map}", "🏷️")

        # Step 3: Optimize signal parameters (if signal generator provided)
        best_signal_params = {}
        if self.signal_generator_fn:
            # Use validation split
            val_split_idx = int(len(train_features) * 0.8)
            val_features = train_features[val_split_idx:]

            # Get corresponding raw data for validation
            lookback_offset = len(train_data_raw) - len(train_features)
            val_data_start = train_start_idx + lookback_offset + val_split_idx
            val_data_end = train_end_idx
            val_prices = self.raw_data.iloc[val_data_start:val_data_end]

            # Ensure alignment
            if isinstance(val_features, pd.DataFrame):
                val_features_df = val_features
            else:
                if val_features.ndim == 1:
                    val_features = val_features.reshape(-1, 1)
                val_prices = val_prices.iloc[: len(val_features)]
                val_features_df = pd.DataFrame(val_features, index=val_prices.index)

            # Predict states on validation set
            # Convert to numpy if needed
            val_features_np = (
                val_features.values
                if isinstance(val_features, pd.DataFrame)
                else val_features
            )
            val_state_predictions = best_model.predict(val_features_np)
            val_state_probabilities = best_model.predict_proba(val_features_np)

            self.log.info("Step 3: Optimizing signal parameters...", "🎯")

            if self.signal_param_optimizer == "grid":
                best_signal_params = self._optimize_signal_params_grid(val_state_predictions, val_state_probabilities,
                                                                       val_features_df, val_prices, state_map)
            elif self.signal_param_optimizer == "bayesian":
                best_signal_params = self._optimize_signal_params_bayesian(val_state_predictions, val_state_probabilities,
                                                                           val_features_df, val_prices, state_map)
            else:
                self.log.warning(f"Unknown optimizer '{self.signal_param_optimizer}', using grid")
                best_signal_params = self._optimize_signal_params_grid(val_state_predictions, val_state_probabilities,
                                                                       val_features_df, val_prices, state_map)

            self.log.info(f"Best signal params: {best_signal_params}", "✓")

        # Step 4: Predict on test set
        # Convert to numpy if needed
        test_features_np = test_features.values if isinstance(test_features, pd.DataFrame) else test_features
        test_state_predictions = best_model.predict(test_features_np)
        test_state_probabilities = best_model.predict_proba(test_features_np)

        # Calculate HMM metrics
        hmm_metrics = self._calculate_hmm_metrics(best_model, test_features_np, test_state_predictions,
                                                  test_state_probabilities, best_config)

        # Step 5: Generate signals and backtest
        trading_metrics = {
            "sharpe_ratio": np.nan,
            "total_return": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "num_trades": 0,
            "sortino_ratio": np.nan,
            "calmar_ratio": np.nan,
            "profit_factor": np.nan,
            "avg_trade_duration": np.nan,
        }

        if self.signal_generator_fn:
            # Align test data
            if isinstance(test_features, pd.DataFrame):
                test_features_df = test_features
                # Align test_data_raw to match test_features length
                n_samples = len(test_features)
                test_data_raw_aligned = test_data_raw.iloc[-n_samples:]
            else:
                if test_features.ndim == 1:
                    test_features = test_features.reshape(-1, 1)
                n_samples = len(test_features)
                test_data_raw_aligned = test_data_raw.iloc[-n_samples:]
                test_features_df = pd.DataFrame(test_features, index=test_data_raw_aligned.index)

            # Generate signals
            signals = self.signal_generator_fn(test_state_predictions, test_state_probabilities, test_data_raw_aligned,
                                               test_features_df, state_map, **best_signal_params)

            if not (isinstance(signals, tuple) and len(signals) == 4):
                raise ValueError(
                    "signal_generator_fn must return 4-tuple (long_entries, long_exits, short_entries, short_exits)"
                )

            # Run backtest
            test_prices = test_data_raw_aligned[self.close_col]
            pf = self._run_backtest(signals, test_prices)
            trading_metrics = self._calculate_trading_metrics(pf)

            # Track signal param importance
            if best_signal_params:
                self._track_signal_param_importance(window_idx, best_signal_params,
                                                    trading_metrics[self.signal_optimization_metric])

            # Store signals and returns
            signals_df = pd.DataFrame(
                {
                    "long_entries": signals[0].astype(int),
                    "long_exits": signals[1].astype(int),
                    "short_entries": signals[2].astype(int),
                    "short_exits": signals[3].astype(int),
                },
                index=test_data_raw_aligned.index,
            )

            if self.all_signals is None:
                self.all_signals = signals_df
            else:
                self.all_signals = pd.concat([self.all_signals, signals_df])

            window_returns = pf.returns()
            if self.all_returns.empty:
                self.all_returns = window_returns
            else:
                self.all_returns = pd.concat([self.all_returns, window_returns])
            self.window_portfolios.append(pf)

        # Store predictions
        if isinstance(test_features, pd.DataFrame):
            test_index = test_features.index
        else:
            n_samples = len(test_state_predictions)
            test_index = test_data_raw.index[-n_samples:]

        new_predictions = pd.Series(test_state_predictions, index=test_index)
        if self.all_predictions.empty:
            self.all_predictions = new_predictions
        else:
            self.all_predictions = pd.concat([self.all_predictions, new_predictions])

        # Feature importance
        feature_names = None
        if isinstance(train_features, pd.DataFrame):
            feature_names = train_features.columns.tolist()

        # Get train state predictions for tree-based importance
        train_features_np = train_features.values if isinstance(train_features, pd.DataFrame) else train_features
        train_state_predictions = best_model.predict(train_features_np)

        feature_importance = self._extract_feature_importance(
            best_model, train_features_np, feature_names, train_state_predictions)
        if feature_importance:
            # Log correlations if available
            if feature_importance.get("correlation_emission_rf") is not None:
                self.log.debug(f"Correlation (Emission vs RF): {feature_importance['correlation_emission_rf']:.3f}")
            if feature_importance.get("correlation_emission_xgb") is not None:
                self.log.debug(f"Correlation (Emission vs XGB): {feature_importance['correlation_emission_xgb']:.3f}")

            self.feature_importance_history.append({"window_idx": window_idx, "importance": feature_importance})

        # Check convergence
        metric_trend = self._check_portfolio_convergence()
        if metric_trend:
            self.log.info(f"Portfolio trend: {metric_trend}", "📈" if metric_trend == "improving" else "📉")

        # Create window result
        window_result = HMMWindowResult(
            window_idx=window_idx,
            train_start=str(train_data_raw.index[0].date()),
            train_end=str(train_data_raw.index[-1].date()),
            test_start=str(test_data_raw.index[0].date()),
            test_end=str(test_data_raw.index[-1].date()),
            n_states=best_model.n_states,
            model_variant=best_config.get("model_variant", ""),
            n_components=best_config.get("n_components", 0),
            covariance_type=best_config.get(
                "covariance_type", ""
            ),  # Backward compatibility
            distribution_type=str(
                best_config.get("distribution_type", "")
            ),  # Backward compatibility
            best_signal_params=best_signal_params,
            state_map=state_map,
            feature_importance=feature_importance,
            metric_trend=metric_trend,
            **hmm_metrics,
            **trading_metrics,
        )

        self.window_results.append(window_result)
        self.window_scalers.append(window_scaler)  # Store scaler for this window

        # Print results
        print(f"\nWindow {window_idx} HMM Metrics:")
        print(
            f"  States:    {window_result.n_states}, "
            f"AIC:       {window_result.aic_score:.2f}, "
            f"BIC:       {window_result.bic_score:.2f},  "
            f"Entropy:   {window_result.avg_state_entropy:.3f}, "
            f"Stability: {window_result.state_stability:.3f}   "
            f"Model: {str(best_model)}"
        )

        if self.signal_generator_fn:
            print(f"\nWindow {window_idx} Trading Metrics:")
            print(
                f"  Sharpe: {window_result.sharpe_ratio:.2f},  "
                f"Return: {window_result.total_return:.2f}%,  "
                f"Drawdown: {window_result.max_drawdown:.2f}%,  "
                f"Trades: {window_result.num_trades}"
            )

        # Save best models (including scaler)
        self._save_best_models(best_model, window_result, window_idx, window_scaler)
        self._save_checkpoint(window_idx)

        # Memory cleanup
        gc.collect()

        return window_result

    def _save_best_models(self, model, window_result: HMMWindowResult, window_idx: int, scaler=None) -> None:

        try:
            best_models_dir = self.checkpoint_dir / "best_models"
            best_models_dir.mkdir(parents=True, exist_ok=True)

            # Remove old best models and scalers
            for old_model in best_models_dir.glob("model_*.joblib"):
                old_model.unlink()
            for old_scaler in best_models_dir.glob("scaler_*.joblib"):
                old_scaler.unlink()

            # Save model
            model_path = best_models_dir / "model_0.joblib"
            model.save(str(model_path))

            # Save scaler if provided
            if scaler is not None:
                scaler_path = best_models_dir / "scaler_0.joblib"
                joblib.dump(scaler, scaler_path)
                self.best_scaler = scaler
                self.log.debug("Scaler saved alongside model", "💾")

            # Save metadata
            metadata = {
                "window_idx": window_result.window_idx,
                "train_start": window_result.train_start,
                "train_end": window_result.train_end,
                "test_start": window_result.test_start,
                "test_end": window_result.test_end,
                "n_states": window_result.n_states,
                "model_variant": window_result.model_variant,
                "n_components": window_result.n_components,
                "covariance_type": window_result.covariance_type,  # Backward compatibility
                "distribution_type": window_result.distribution_type,  # Backward compatibility
                "signal_parameters": window_result.best_signal_params,
                "state_map": window_result.state_map,
                "hmm_metrics": {
                    "aic": window_result.aic_score,
                    "bic": window_result.bic_score,
                    "log_likelihood": window_result.log_likelihood,
                    "entropy": window_result.avg_state_entropy,
                    "stability": window_result.state_stability,
                },
                "trading_metrics": {
                    "sharpe_ratio": window_result.sharpe_ratio,
                    "total_return": window_result.total_return,
                    "max_drawdown": window_result.max_drawdown,
                    "win_rate": window_result.win_rate,
                    "num_trades": window_result.num_trades,
                },
            }

            metadata_path = best_models_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2, default=to_serializable)

            self.log.debug(f"Best models saved to {best_models_dir}", "💾")
            self.best_window_idx = window_idx

        except Exception as e:
            self.log.warning(f"Could not save best models: {e}")

    def run(self, resume: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Execute full Walk-Forward Analysis."""
        print("\n" + "=" * 60)
        self.log.info("STARTING HMM WALK-FORWARD ANALYSIS", "🚀")
        print("=" * 60)

        start_window = 0
        if resume:
            last_completed = self._load_checkpoint()
            if last_completed is not None:
                start_window = last_completed + 1
                self.log.info(f"Resuming from window {start_window}", "⏩")

        windows = self._calculate_windows()

        for idx, (train_start, train_end, test_start, test_end) in enumerate(
            windows, start=1
        ):
            if idx < start_window:
                continue
            self._run_window(idx, train_start, train_end, test_start, test_end)

        return self._aggregate_results()

    def _aggregate_results(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Aggregate and display final results using ResultsAggregationMixin."""
        self._print_results_header()

        if not self.window_results:
            self.log.warning("No windows completed")
            return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=float)

        results_df = self._create_results_dataframe()

        # HMM-specific statistics
        self._print_hmm_statistics(results_df)

        # Trading statistics if signal generator was provided
        if self.signal_generator_fn:
            self._print_trading_statistics(results_df, detailed=False)

        # Print feature and signal parameter summaries
        self._print_feature_importance_summary()
        self._print_signal_param_summary()

        # Save results with HMM-specific naming for predictions
        self._save_results_files(
            results_df,
            save_predictions=True,
            save_returns=self.signal_generator_fn is not None,
            save_signals=self.all_signals is not None,
        )

        return results_df, self.all_predictions, self.all_true_labels

    def get_best_models(self) -> List:
        """Load and return the best performing model from disk."""
        import joblib

        best_models_dir = self.checkpoint_dir / "best_models"

        if not best_models_dir.exists():
            self.log.warning("No best models found. Run analysis first.")
            return []

        models = []
        model_path = best_models_dir / "model_0.joblib"
        if model_path.exists():
            # Load using joblib - more secure and efficient for HMM models
            model = joblib.load(model_path)
            models.append(model)
            self.log.info(f"Loaded best model from window {self.best_window_idx}", "✓")

        return models

    def get_best_scaler(self):
        """Load and return the best performing scaler from disk."""
        import joblib

        best_models_dir = self.checkpoint_dir / "best_models"

        if not best_models_dir.exists():
            self.log.warning("No best models directory found. Run analysis first.")
            return None

        scaler_path = best_models_dir / "scaler_0.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            self.log.info(f"Loaded best scaler from window {self.best_window_idx}", "✓")
            return scaler
        else:
            self.log.warning(
                "No scaler found. Feature engineering may not include scaling."
            )
            return None

    def get_best_state_map(self) -> Optional[Dict[int, str]]:
        """Get the state mapping from the best window."""
        best_models_dir = self.checkpoint_dir / "best_models"
        metadata_path = best_models_dir / "metadata.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("state_map")
        except Exception as e:
            self.log.warning(f"Could not load state map: {e}")
            return None

    def get_best_portfolio(self) -> Optional[vbt.Portfolio]:
        if not self.window_portfolios:
            self.log.warning("No portfolios available. Run analysis first.")
            return None

        if self.best_window_idx is None:
            self.log.warning("No best window identified.")
            return None

        # best_window_idx is 1-based, convert to 0-based for list indexing
        idx = self.best_window_idx - 1
        if idx < 0 or idx >= len(self.window_portfolios):
            self.log.warning(f"Best window index {self.best_window_idx} out of range")
            return None

        return self.window_portfolios[idx]

    def get_all_portfolios(self) -> List[vbt.Portfolio]:
        return self.window_portfolios
