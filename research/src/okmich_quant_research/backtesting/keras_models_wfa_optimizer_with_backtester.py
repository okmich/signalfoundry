import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, List, Any

import keras_tuner as kt
import numpy as np
import pandas as pd
import vectorbt as vbt
from matplotlib import pyplot as plt

from .keras_models_wfa_optimizer import ModelWalkForwardAnalysisOptimizer, WindowResult
from .vbt_export import QuantframeExportMixin
from .wfa_plot_utils import BacktestVisualizationMixin
from .wfa_utils import to_serializable, BacktestingMixin, SignalOptimizationMixin

# Set default frequency for vectorbt to avoid frequency inference errors
vbt.settings.array_wrapper["freq"] = "h"

# Type aliases for signal generator outputs
SignalArrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass
class WindowBacktestResult(WindowResult):
    """Extends WindowResult with backtesting metrics and enhancements.

    Inherits task_type and optional classification/regression metrics from WindowResult.
    Adds trading-specific metrics.
    """

    sharpe_ratio: float = np.nan
    total_return: float = np.nan
    max_drawdown: float = np.nan
    win_rate: float = np.nan
    num_trades: int = 0
    sortino_ratio: float = np.nan
    calmar_ratio: float = np.nan
    profit_factor: float = np.nan
    avg_trade_duration: float = np.nan


class ModelWalkForwardAnalysisBacktestOptimizer(ModelWalkForwardAnalysisOptimizer, BacktestingMixin,
                                                SignalOptimizationMixin, BacktestVisualizationMixin,
                                                QuantframeExportMixin):
    """
    ModelWalkForwardAnalysisBacktestOptimizer: Advanced walk-forward analysis with integrated backtesting.

    This class combines deep learning model optimization with portfolio backtesting using VectorBT.
    It performs walk-forward analysis where each window:
    1. Optimizes the model hyperparameters
    2. Optimizes signal generation parameters via backtesting
    3. Evaluates both classification metrics and trading performance
    4. Tracks comprehensive metrics across market regimes

    Key Features:
    - Dual optimization: Deep learning hyperparameters + signal parameters
    - Ensemble learning support (best/top-k/weighted models)
    - Transfer learning across windows
    - Early stopping and convergence detection
    - Feature and signal parameter importance tracking
    - Memory management and checkpointing
    - Data leakage detection
    - Signal generation with long/short entries/exits (4-tuple format)

    Signal Generator Format:
    The signal_generator_fn must return signals as a 4-tuple of ndarrays (long_entries, long_exits, short_entries, short_exits):

    ```python
    def signal_generator_fn(predictions, prices, features, **params):
        threshold = params.get('threshold', 0.5)

        # Generate entry/exit signals as numpy boolean arrays
        long_entries = (predictions > threshold + 0.1)
        long_exits = (predictions < threshold - 0.1)
        short_entries = (predictions < threshold - 0.1)
        short_exits = (predictions > threshold + 0.1)

        return long_entries, long_exits, short_entries, short_exits  # All ndarrays
    ```

    Typical Usage:
    ```python
    # Define your components
    def feature_engineering_fn(train_raw, test_raw, train_labels, test_labels):
        # Engineer features from raw OHLCV data
        return train_features, test_features, train_labels, test_labels

    def model_builder(hp):
        # Build Keras model with hyperparameter tuning
        model = Sequential([...])
        return model

    def signal_generator_fn(predictions, prices, features, **params):
        # Return 4-tuple of ndarrays for long/short entries and exits
        threshold = params.get('threshold', 0.5)
        hold_period = params.get('hold_period', 5)

        long_entries = predictions > threshold
        long_exits = predictions < (threshold - 0.1)
        short_entries = predictions < (1 - threshold)
        short_exits = predictions > (1 - threshold + 0.1)

        return long_entries, long_exits, short_entries, short_exits  # All ndarrays

    # Setup optimizer
    optimizer = ModelWalkForwardAnalysisBacktestOptimizer(
        raw_data=ohlcv_df,              # OHLCV DataFrame with DatetimeIndex
        label_data=labels_series,        # Binary classification labels
        train_period=252,                # Training window size (days)
        test_period=63,                  # Test window size (days)
        step_period=21,                  # Step size between windows
        feature_engineering_fn=feature_engineering_fn,
        model_builder_fn=model_builder,
        signal_generator_fn=signal_generator_fn,
        signal_param_grid={              # Parameters to optimize
            'threshold': [0.4, 0.5, 0.6],
            'hold_period': [1, 3, 5]
        },
        signal_param_optimizer='bayesian',  # 'grid', or 'bayesian'
        signal_param_n_calls=50,         # For bayesian optimization
        tuner_params={                   # Keras Tuner config
            'objective': 'val_accuracy',
            'max_trials': 20
        },
        vbt_params={                     # VectorBT backtest settings
            'fees': 0.001,
            'slippage': 0.001
        },
        signal_optimization_metric='sharpe',  # Optimize for Sharpe ratio
        ensemble_size=3,                 # Train ensemble of 3 models
        track_feature_importance=True,   # Track feature importance
        track_signal_param_importance=True,  # Track signal param patterns
        anchored=False,                  # Rolling window (not anchored)
        checkpoint_dir='./checkpoints'
    )

    # Run walk-forward analysis
    results_df, predictions, true_labels = optimizer.run()

    # Access results
    results_df = optimizer.window_results  # Per-window metrics
    predictions = optimizer.all_predictions  # All predictions
    signals = optimizer.all_signals  # All trading signals
    returns = optimizer.all_returns  # Portfolio returns

    # Get best models (inherited from parent)
    best_models = optimizer.get_best_models()
    best_hp = optimizer.get_best_hyperparameters()

    # Get signal-specific info
    signal_params = optimizer.get_best_signal_parameters()
    trading_summary = optimizer.get_trading_performance_summary()

    # Get VectorBT portfolio for advanced analysis
    best_pf = optimizer.get_best_portfolio()
    best_pf.stats()  # Full portfolio statistics
    best_pf.plot().show()  # Interactive equity curve
    best_pf.trades.records_readable  # Detailed trade log

    # Visualize
    optimizer.plot_results()
    optimizer.plot_equity_curve()
    optimizer.plot_rolling_metrics()

    # Export for production
    optimizer.export_for_production('./production')
    ```

    Result Metrics:
    - Classification: accuracy, precision, recall, F1, AUC-ROC
    - Trading: Sharpe, Sortino, Calmar, total return, max drawdown, win rate,
      profit factor, number of trades, avg trade duration
    - Ensemble: ensemble size, model agreement
    - Convergence: early stopping, metric trends (improving/stable/degrading)

    Advanced Features:
    - Portfolio convergence monitoring (sharpe/return/drawdown trends)
    - Signal parameter importance analysis
    - Feature importance tracking across regimes (inherited from parent)
    - Memory management for large datasets
    - Checkpoint resume capability
    - Data leakage detection (inherited from parent)
    - Multiple signal optimization strategies (grid/random/bayesian)

    Args:
        raw_data: OHLCV DataFrame with DatetimeIndex (must contain close_col)
        label_data: Binary classification targets (pd.Series)
        train_period: Number of samples in training window
        test_period: Number of samples in test window
        step_period: Number of samples to step forward each window
        feature_engineering_fn: Feature transformation function (takes and returns 4 items)
        model_builder_fn: Keras model builder with hyperparameter tuning
        signal_generator_fn: Converts predictions to trading signals (4-tuple of ndarrays: long_entries, long_exits, short_entries, short_exits)
        signal_param_grid: Dict of signal parameters to optimize
        signal_param_optimizer: Optimization method ('grid', 'random', 'bayesian')
        signal_param_n_calls: Number of evaluations for random/bayesian optimization
        tuner_params: Keras Tuner configuration
        postprocess_optimization_metric: Metric for signal param optimization
        vbt_params: VectorBT portfolio parameters (fees, slippage, etc.)
        signal_optimization_metric: Metric to optimize ('sharpe', 'return', etc.)
        close_col: Column name for close prices in raw_data
        ensemble_size: Number of models in ensemble (1 = no ensemble)
        ensemble_method: 'best', 'top_k', or 'weighted_avg'
        track_signal_param_importance: Track signal parameter patterns
        portfolio_convergence_metric: Metric for convergence ('sharpe'/'return'/'drawdown')
        anchored: If True, training window grows; if False, rolls forward
        checkpoint_dir: Directory for saving checkpoints and results
        transfer_learning: Use previous window's model as starting point
        track_feature_importance: Extract and track feature importance
        enable_memory_management: Clear Keras sessions between windows
        detect_data_leakage: Check for train/test data leakage
        verbose: Logging verbosity (0=silent, 1=info, 2=debug)

    Outputs:
    - window_results: List[WindowBacktestResult] with all metrics
    - all_predictions: pd.Series of ML predictions
    - all_true_labels: pd.Series of actual labels
    - all_signals: pd.Series or pd.DataFrame of trading signals
    - all_returns: pd.Series of portfolio returns
    - Saved files: CSV results, predictions, returns, importance JSONs, plots

    Inherited Methods (from parent):
    - get_best_models(): Load best trained models
    - get_best_hyperparameters(): Get model hyperparameters + metadata
    - get_best_window_info(): Get info about best window
    - export_best_models(): Export models for production
    - get_all_hyperparameters(): Get all model hyperparameters
    - get_feature_importance_summary(): Get feature importance across windows

    Child-Specific Methods:
    - get_best_signal_parameters(): Get optimal signal parameters
    - get_all_signal_parameters(): Get all signal parameters
    - get_trading_performance_summary(): Get trading metrics summary
    - get_signal_param_importance_summary(): Get signal param importance
    - get_best_portfolio(): Get VectorBT portfolio from best window
    - export_for_production(): Export models + signal parameters
    - plot_equity_curve(): Plot equity curve
    - plot_rolling_metrics(): Plot rolling performance metrics
    """

    def __init__(self, raw_data: pd.DataFrame, label_data: pd.Series, train_period: int, test_period: int, step_period: int,
                 feature_engineering_fn: Callable, model_builder_fn: Callable, signal_generator_fn: Callable[..., SignalArrayTuple],
                 signal_param_grid: Optional[Dict[str, List[Any]]] = None, signal_param_optimizer: str = "grid",
                 signal_param_n_calls: int = 50, tuner_params: Dict = None, postprocess_optimization_metric: str = "f1",
                 anchored: bool = False, checkpoint_dir: Optional[str] = None, transfer_learning: bool = False,
                 tuning_epochs: int = 10, tuning_val_split: float = 0.2, vbt_params: Optional[Dict] = None,
                 signal_optimization_metric: str = "sharpe_ratio", close_col: str = "close", ensemble_size: int = 1,
                 ensemble_method: str = "best", early_stopping_patience: int = 5,
                 early_stopping_min_delta: float = 0.001, track_feature_importance: bool = False,
                 enable_memory_management: bool = True, clear_session_between_windows: bool = True, detect_data_leakage: bool = True,
                 convergence_window_size: int = 3, convergence_threshold: float = 0.05, track_signal_param_importance: bool = True,
                 portfolio_convergence_metric: str = "sharpe", verbose: int = 1):
        super().__init__(raw_data=raw_data, label_data=label_data, train_period=train_period, test_period=test_period,
                         step_period=step_period, feature_engineering_fn=feature_engineering_fn,
                         model_builder_fn=model_builder_fn, postprocess_fn=None,  # We don't use parent's postprocess
                         postprocess_param_grid=signal_param_grid, tuner_params=tuner_params,
                         postprocess_optimization_metric=postprocess_optimization_metric, anchored=anchored,
                         checkpoint_dir=checkpoint_dir, transfer_learning=transfer_learning, tuning_epochs=tuning_epochs,
                         tuning_val_split=tuning_val_split, verbose=verbose, ensemble_size=ensemble_size, ensemble_method=ensemble_method,
                         early_stopping_patience=early_stopping_patience, early_stopping_min_delta=early_stopping_min_delta,
                         track_feature_importance=track_feature_importance, enable_memory_management=enable_memory_management,
                         clear_session_between_windows=clear_session_between_windows, detect_data_leakage=detect_data_leakage,
                         convergence_window_size=convergence_window_size, convergence_threshold=convergence_threshold)
        # Validate price data
        if close_col not in raw_data.columns:
            raise ValueError(f"raw_data must have '{close_col}' column")

        # Backtesting-specific attributes
        self.signal_generator_fn = signal_generator_fn
        self.signal_param_grid = signal_param_grid or {}
        self.signal_param_optimizer = signal_param_optimizer
        self.signal_param_n_calls = signal_param_n_calls
        self.vbt_params = vbt_params or {"fees": 0.001, "slippage": 0.001}
        self.signal_optimization_metric = signal_optimization_metric
        self.close_col = close_col
        self.postprocess_fn = (
            None  # Override postprocess_fn to None since we handle signals differently
        )
        self.all_signals = None
        self.all_returns = pd.Series(dtype=float)

        self.track_signal_param_importance = track_signal_param_importance
        self.signal_param_importance_history: List[Dict] = []
        self.portfolio_convergence_metric = portfolio_convergence_metric

        # Portfolio storage for each window
        self.window_portfolios: List[vbt.Portfolio] = []

        self.log.info("ModelWalkForwardAnalysisBacktestOptimizer initialized", "✓")
        if self.track_signal_param_importance:
            self.log.info("Signal parameter importance tracking enabled")
        if signal_param_optimizer != "grid":
            self.log.info(f"Signal param optimizer: {signal_param_optimizer} (n_calls={signal_param_n_calls})")

    def _create_window_result(self, **kwargs) -> WindowBacktestResult:
        """
        Hook for parent's checkpoint loading to create correct dataclass type.
        Overrides parent's method to create WindowBacktestResult instead of WindowResult.
        """
        return WindowBacktestResult(**kwargs)

    def _optimize_postprocess_parameters(self, predictions: np.ndarray, val_features: pd.DataFrame,
                                         val_labels: pd.Series) -> Dict[str, Any]:
        """
        Override parent to optimize signal parameters using backtesting metrics.
        Supports grid, random, and Bayesian optimization.
        """
        if not self.signal_generator_fn:
            return {}

        # Get validation price data
        if isinstance(val_features, pd.DataFrame):
            val_start_idx = self.raw_data.index.get_loc(val_features.index[0])
            val_end_idx = self.raw_data.index.get_loc(val_features.index[-1]) + 1
            val_prices = self.raw_data.iloc[val_start_idx:val_end_idx]
        else:
            self.log.warning("val_features is not a DataFrame, using last N rows of raw_data")
            val_prices = self.raw_data.iloc[-len(val_features) :]

        # Dispatch to appropriate optimizer
        if self.signal_param_optimizer == "grid":
            best_params = self._optimize_signal_params_grid(predictions, val_features, val_labels, val_prices)
        elif self.signal_param_optimizer == "bayesian":
            best_params = self._optimize_signal_params_bayesian(predictions, val_features, val_labels, val_prices)
        else:
            self.log.warning(f"Unknown optimizer '{self.signal_param_optimizer}', using grid search")
            best_params = self._optimize_signal_params_grid(predictions, val_features, val_labels, val_prices)

        self.log.debug(f"Best signal params: {best_params}")
        return best_params

    def _run_window(self, window_idx: int, train_start_idx: int, train_end_idx: int, test_start_idx: int,
                    test_end_idx: int ) -> WindowBacktestResult:
        """
        Override parent's _run_window to add backtesting with all enhancements.
        """
        print(f"\n{'=' * 60}")
        self.log.info(f"Window {window_idx}", "🔄")
        print(f"{'=' * 60}")

        # Slice data
        train_data_raw = self.raw_data.iloc[train_start_idx:train_end_idx]
        train_labels_raw = self.label_data.iloc[train_start_idx:train_end_idx]
        test_data_raw = self.raw_data.iloc[test_start_idx:test_end_idx]
        test_labels_raw = self.label_data.iloc[test_start_idx:test_end_idx]

        print(
            f"Training: {train_data_raw.index[0].date()} to {train_data_raw.index[-1].date()} "
            f"({len(train_data_raw)} samples)"
        )
        print(
            f"Testing:  {test_data_raw.index[0].date()} to {test_data_raw.index[-1].date()} "
            f"({len(test_data_raw)} samples)"
        )

        # Feature engineering
        train_features, test_features, train_labels, test_labels = (
            self.feature_engineering_fn(
                train_data_raw, test_data_raw, train_labels_raw, test_labels_raw
            )
        )
        self.log.debug(f"Transformed shapes - Train: {train_features.shape}, Test: {test_features.shape}")

        # Data leakage detection (inherited from parent)
        self._detect_data_leakage(train_features, test_features, train_data_raw, test_data_raw)

        # Prepare callbacks
        callbacks = []
        early_stop_callback = self._get_early_stopping_callback()
        if early_stop_callback:
            callbacks.append(early_stop_callback)

        # Step 1: Optimize model hyperparameters (uses parent's tuner)
        self.log.info("Step 1: Optimizing model hyperparameters...", "🔍")

        tuner = kt.BayesianOptimization(self.model_builder_fn, **self.tuner_params,
                                        directory=str(self.checkpoint_dir / "tuning"),
                                        project_name=f"window_{window_idx}", overwrite=True)
        if isinstance(train_labels, pd.Series):
            train_labels_array = train_labels.values
        else:
            train_labels_array = train_labels

        tuner.search(train_features, train_labels_array, epochs=self.tuning_epochs,
                     validation_split=self.tuning_val_split, callbacks=callbacks, verbose=0)

        # Train ensemble (inherited from parent)
        ensemble_models, ensemble_hps = self._train_ensemble(tuner, train_features, train_labels_array, callbacks)
        best_model = ensemble_models[0]
        best_model_hp = ensemble_hps[0]

        self.log.info(f"Best model params: {best_model_hp}", "✓")
        if len(ensemble_models) > 1:
            self.log.info(f"Ensemble of {len(ensemble_models)} models trained", "✓")

        # Feature importance tracking (inherited from parent)
        feature_names = None
        if isinstance(train_features, pd.DataFrame):
            feature_names = train_features.columns.tolist()

        feature_importance = self._extract_feature_importance(best_model, feature_names)
        if feature_importance:
            self.log.debug(f"Feature importance extracted: {len(feature_importance)} features")
            self.feature_importance_history.append({"window_idx": window_idx, "importance": feature_importance})

        # Early stopping detection
        early_stopped = False
        convergence_epoch = None
        if early_stop_callback and hasattr(early_stop_callback, "stopped_epoch"):
            if early_stop_callback.stopped_epoch > 0:
                early_stopped = True
                convergence_epoch = early_stop_callback.stopped_epoch
                self.log.info(f"Early stopped at epoch {convergence_epoch}", "⏸")

        # Transfer learning (inherited from parent)
        if self.transfer_learning:
            self.previous_best_model = best_model

        # Step 2: Optimize signal parameters
        best_signal_params = {}
        if self.signal_generator_fn:
            val_split_idx = int(len(train_features) * 0.8)
            val_features = train_features[val_split_idx:]

            # Ensemble prediction on validation
            val_predictions = self._ensemble_predict(ensemble_models, val_features)

            lookback_offset = len(train_data_raw) - len(train_features)
            val_data_start = train_start_idx + lookback_offset + val_split_idx
            val_data_end = train_end_idx
            val_prices = self.raw_data.iloc[val_data_start:val_data_end]

            if isinstance(train_features, pd.DataFrame):
                val_features_df = train_features.iloc[val_split_idx:]
            else:
                # For 3D LSTM data, convert back to 2D
                val_features_2d = train_features[val_split_idx:]
                if val_features_2d.ndim == 3:
                    val_features_2d = val_features_2d[:, -1, :]  # Last timestep
                # Ensure val_prices has the same length as val_features_2d
                val_prices = val_prices.iloc[: len(val_features_2d)]
                val_features_df = pd.DataFrame(val_features_2d, index=val_prices.index)

            val_labels = train_labels.iloc[val_split_idx:] if isinstance(train_labels, pd.Series) \
                else pd.Series(train_labels[val_split_idx:])

            self.log.info("Step 2: Optimizing signal parameters...", "🎯")
            best_signal_params = self._optimize_postprocess_parameters(val_predictions, val_features_df, val_labels)
            self.log.info(f"Best signal params: {best_signal_params}", "✓")

        # Step 3: Generate predictions on test set
        test_predictions_raw = self._ensemble_predict(ensemble_models, test_features)

        # Step 4: Calculate classification/regression metrics based on task type
        test_predictions_proba, test_predictions_continuous = None, None
        if self.task_type == "classification":
            # Convert to binary for classification metrics
            # Handle 2D softmax output (n_samples, n_classes) vs 1D output (n_samples,)
            if test_predictions_raw.ndim > 1 and test_predictions_raw.shape[1] > 1:
                # Multi-class softmax output: use argmax for class and max for probability
                test_predictions_binary = np.argmax(test_predictions_raw, axis=1)
                test_predictions_proba = np.max(test_predictions_raw, axis=1)
            else:
                # Single output or already 1D
                if test_predictions_raw.ndim > 1:
                    test_predictions_raw_flat = test_predictions_raw.flatten()
                else:
                    test_predictions_raw_flat = test_predictions_raw
                test_predictions_binary = (test_predictions_raw_flat > 0.5).astype(int)
                test_predictions_proba = test_predictions_raw_flat

            # Use the test_labels returned by feature_engineering_fn which should already be aligned
            classification_metrics = self._calculate_metrics(test_labels, test_predictions_binary, test_predictions_proba)
        else:  # regression
            if test_predictions_raw.ndim > 1:
                test_predictions_continuous = test_predictions_raw.flatten()
            else:
                test_predictions_continuous = test_predictions_raw

            classification_metrics = self._calculate_regression_metrics(test_labels, test_predictions_continuous)

        # Step 5: Generate trading signals
        if isinstance(test_features, pd.DataFrame):
            test_features_df = test_features
            test_data_raw_aligned = test_data_raw
        else:
            # For 3D LSTM data, convert to 2D for signal generation
            test_features_2d = test_features
            if test_features_2d.ndim == 3:
                test_features_2d = test_features_2d[:, -1, :]  # Use last timestep

            # Align test_data_raw with sequence-transformed features
            # Sequence transformation reduces sample count, so use the last N samples
            n_samples_after_sequence = len(test_features_2d)
            test_data_raw_aligned = test_data_raw.iloc[-n_samples_after_sequence:]

            test_features_df = pd.DataFrame(test_features_2d, index=test_data_raw_aligned.index)

        # Pass aligned test data to signal generator
        signals = self.signal_generator_fn(test_predictions_raw, test_data_raw_aligned,
                                           test_features_df, **best_signal_params)

        # signals must be a 4-tuple (long_entries, long_exits, short_entries, short_exits)
        if not (isinstance(signals, tuple) and len(signals) == 4):
            raise ValueError(
                "signal_generator_fn must return a 4-tuple (long_entries, long_exits, short_entries, short_exits)"
            )

        # Step 6: Run backtest
        # Signals are ndarrays aligned with test_data_raw_aligned by construction
        # Use the close prices from the aligned test data
        test_prices_aligned = test_data_raw_aligned[self.close_col]
        pf = self._run_backtest(signals, test_prices_aligned)

        # Calculate trading metrics
        trading_metrics = self._calculate_trading_metrics(pf)
        if best_signal_params:
            self._track_signal_param_importance(
                window_idx,
                best_signal_params,
                trading_metrics[self.signal_optimization_metric],
            )

        # Step 7: Store results
        # Use the aligned index from test_data_raw_aligned
        if self.task_type == "classification":
            new_predictions = pd.Series(test_predictions_proba, index=test_data_raw_aligned.index)
        else:  # regression
            new_predictions = pd.Series(test_predictions_continuous, index=test_data_raw_aligned.index)

        if self.all_predictions.empty:
            self.all_predictions = new_predictions
        else:
            self.all_predictions = pd.concat([self.all_predictions, new_predictions])

        new_labels = pd.Series(test_labels, index=test_data_raw_aligned.index)
        if self.all_true_labels.empty:
            self.all_true_labels = new_labels
        else:
            self.all_true_labels = pd.concat([self.all_true_labels, new_labels])

        # Store signals (4-tuple format)
        # Signals are ndarrays, so we need to attach the proper datetime index
        signals_df = pd.DataFrame(
            {
                "long_entries": signals[0].astype(int),
                "long_exits": signals[1].astype(int),
                "short_entries": signals[2].astype(int),
                "short_exits": signals[3].astype(int),
            },
            index=test_data_raw_aligned.index)
        if self.all_signals is None:
            self.all_signals = signals_df
        else:
            self.all_signals = pd.concat([self.all_signals, signals_df])

        # Store returns
        window_returns = pf.returns()
        if self.all_returns.empty:
            self.all_returns = window_returns
        else:
            self.all_returns = pd.concat([self.all_returns, window_returns])

        # Store portfolio for this window
        self.window_portfolios.append(pf)

        # Check portfolio convergence
        metric_trend = self._check_portfolio_convergence()
        if metric_trend:
            self.log.info(f"Portfolio trend: {metric_trend}", "📈" if metric_trend == "improving" else "📉")

        # Create window result with all metrics
        window_result = WindowBacktestResult(
            window_idx=window_idx,
            train_start=str(train_data_raw.index[0].date()),
            train_end=str(train_data_raw.index[-1].date()),
            test_start=str(test_data_raw.index[0].date()),
            test_end=str(test_data_raw.index[-1].date()),
            best_model_hp=best_model_hp,
            best_postprocess_params=best_signal_params,
            task_type=self.task_type,
            **classification_metrics,  # Unpack classification or regression metrics
            ensemble_size=len(ensemble_models),
            feature_importance=feature_importance,
            early_stopped=early_stopped,
            convergence_epoch=convergence_epoch,
            metric_trend=metric_trend,
            **trading_metrics,
        )

        self.window_results.append(window_result)

        # Print results
        if self.task_type == "classification":
            print(f"\nWindow {window_idx} Classification Metrics:")
            print(
                f"  Accuracy:  {window_result.accuracy:.4f}, "
                f"Precision: {window_result.precision:.4f}, "
                f"Recall:    {window_result.recall:.4f}, "
                f"F1:        {window_result.f1_score:.4f}")
        else:  # regression
            print(f"\nWindow {window_idx} Regression Metrics:")
            print(
                f"  MSE:       {window_result.mse:.4f}, "
                f"RMSE:      {window_result.rmse:.4f}, "
                f"MAE:       {window_result.mae:.4f}, "
                f"R²:        {window_result.r2:.4f}")

        print(f"\nWindow {window_idx} Trading Metrics:")
        print(
            f"  Sharpe:    {window_result.sharpe_ratio:.2f}, "
            f"Return:    {window_result.total_return:.2f}%, "
            f"Drawdown:  {window_result.max_drawdown:.2f}%, "
            f"Win Rate:  {window_result.win_rate:.2f}%, "
            f"Trades:    {window_result.num_trades}"
        )
        print(
            f"  Sortino:   {window_result.sortino_ratio:.2f}, "
            f"Calmar:    {window_result.calmar_ratio:.2f}, "
            f"PF:        {window_result.profit_factor:.2f}"
        )

        if window_result.ensemble_size > 1:
            print(f"  Ensemble:  {window_result.ensemble_size} models")
        if window_result.metric_trend:
            print(f"  Trend:     {window_result.metric_trend}")

        # Check if this window should be saved as best (inherited from parent)
        if self._should_update_best_models(window_idx, window_result):
            current_metric = getattr(window_result, self.best_model_metric)
            self.best_metric_value = current_metric
            self.best_window_idx = window_idx

            self._save_best_models(ensemble_models, window_result)
            self.log.info(f"New best models saved! {self.best_model_metric}={current_metric:.4f}", "🏆")

        # Save checkpoint (inherited from parent)
        self._save_checkpoint(window_idx)

        # Clean up tuner directory to save disk space
        if self.enable_memory_management:
            tuner_dir = self.checkpoint_dir / "tuning" / f"window_{window_idx}"
            if tuner_dir.exists():
                try:
                    shutil.rmtree(tuner_dir)
                    self.log.debug(f"Cleaned tuner directory: {tuner_dir}")
                except Exception as e:
                    self.log.debug(f"Could not clean tuner directory: {e}")

        # Memory management (inherited from parent)
        self._clear_memory()
        return window_result

    def _aggregate_results(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Override to include comprehensive trading metrics using ResultsAggregationMixin."""
        self._print_results_header()

        if not self.window_results:
            self.log.warning("No windows completed")
            return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

        results_df = self._create_results_dataframe()

        # Print classification and trading statistics
        self._print_classification_statistics(results_df)
        self._print_trading_statistics(results_df, detailed=True)

        # Print ensemble and early stopping statistics
        self._print_ensemble_statistics(results_df)
        self._print_early_stopping_statistics(results_df)

        # Print best model info
        self._print_best_model_info()

        # Print feature and signal parameter summaries
        self._print_feature_importance_summary()
        self._print_signal_param_summary()

        # Overall out-of-sample classification metrics
        if len(self.all_predictions) > 0:
            all_preds_binary = (self.all_predictions > 0.5).astype(int)
            overall_class_metrics = self._calculate_metrics(
                self.all_true_labels.values,
                all_preds_binary.values,
                self.all_predictions.values,
            )

            print("\nOverall Out-of-Sample Classification:")
            print(
                f"  Accuracy:  {overall_class_metrics['accuracy']:.4f}, "
                f"Precision: {overall_class_metrics['precision']:.4f}, "
                f"Recall:    {overall_class_metrics['recall']:.4f}, "
                f"F1-Score:  {overall_class_metrics['f1_score']:.4f}, "
                f"AUC-ROC:   {overall_class_metrics['auc_roc']:.4f}"
            )

        # Overall out-of-sample trading metrics
        if len(self.all_returns) > 0:
            returns = self.all_returns.dropna()
            if not returns.empty:
                # Ensure the index has a frequency set for vectorbt
                if returns.index.freq is None and len(returns) > 1:
                    try:
                        # Try to infer frequency from the index
                        inferred_freq = pd.infer_freq(returns.index)
                        if inferred_freq is not None:
                            returns = pd.Series(
                                returns.values,
                                index=pd.DatetimeIndex(returns.index, freq=inferred_freq),
                                name=returns.name,
                            )
                        else:
                            returns = pd.Series(
                                returns.values,
                                index=pd.date_range(start=returns.index[0], periods=len(returns), freq="m"),
                                name=returns.name,
                            )
                    except (ValueError, TypeError) as e:
                        self.log.warning(f"Could not infer frequency, using hourly: {e}")
                        returns = pd.Series(
                            returns.values,
                            index=pd.date_range(start=returns.index[0], periods=len(returns), freq="H"),
                            name=returns.name,
                        )

                ret_obj = returns.vbt.returns
                overall_sharpe = ret_obj.sharpe_ratio()
                overall_return = ret_obj.total()
                overall_dd = ret_obj.max_drawdown()

                # Calculate overall Sortino
                downside_returns = self.all_returns[self.all_returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        overall_sortino = (
                            self.all_returns.mean() / downside_std
                        ) * np.sqrt(252)
                    else:
                        overall_sortino = np.nan
                else:
                    overall_sortino = np.nan
            else:
                overall_sharpe = overall_sortino = overall_return = overall_dd = np.nan
        else:
            overall_sharpe = overall_sortino = overall_return = overall_dd = np.nan

        print("\nOverall Out-of-Sample Trading:")
        print(
            f"  Sharpe Ratio:   {overall_sharpe:.2f}, "
            f"Sortino Ratio:  {overall_sortino:.2f}, "
            f"Total Return:   {overall_return * 100:.2f}%, "
            f"Max Drawdown:   {overall_dd * 100:.2f}%"
        )
        # Save results
        self._save_results_files(results_df, save_predictions=True, save_returns=True, save_signals=True)

        return results_df, self.all_predictions, self.all_true_labels

    def _summarize_signal_param_importance(self) -> None:
        """
        Summarize signal parameter selection patterns across windows.
        Shows which parameters are consistently chosen.
        """
        if not self.signal_param_importance_history:
            return

        # Collect all parameter names
        all_param_names = set()
        for window_data in self.signal_param_importance_history:
            all_param_names.update(window_data["params"].keys())

        # Analyze selection frequency and impact
        print(f"  Most frequently selected values:")
        for param_name in sorted(all_param_names):
            value_counts = {}
            for window_data in self.signal_param_importance_history:
                value = window_data["params"].get(param_name)
                if value is not None:
                    value_str = str(value)
                    value_counts[value_str] = value_counts.get(value_str, 0) + 1

            if value_counts:
                most_common = max(value_counts.items(), key=lambda x: x[1])
                total = len(self.signal_param_importance_history)
                print(
                    f"    {param_name}: {most_common[0]} "
                    f"({most_common[1]}/{total} = {most_common[1] / total:.1%})"
                )

    def get_best_signal_parameters(self, window_idx: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get optimal signal parameters from specific window.

        Args:
            window_idx: Window index. If None, returns from best window.

        Returns:
            Dict of signal parameter names and values, or None if not found.

        Example:
            # >>> signal_params = optimizer.get_best_signal_parameters()
            # >>> print(f"Best threshold: {signal_params.get('threshold')}")
        """
        if not self.window_results:
            return None

        if window_idx is None:
            # Use best_window_idx if available, otherwise use last window
            if self.best_window_idx is not None:
                # Find the window result with matching window_idx
                for wr in self.window_results:
                    if wr.window_idx == self.best_window_idx:
                        return wr.best_postprocess_params
                # If not found, fall back to last window
                window_idx = len(self.window_results) - 1
            else:
                window_idx = len(self.window_results) - 1

        # At this point window_idx is an array index (0-based)
        if window_idx < 0 or window_idx >= len(self.window_results):
            self.log.warning(f"Invalid window_idx {window_idx}")
            return None
        return self.window_results[window_idx].best_postprocess_params

    def get_all_signal_parameters(self) -> pd.DataFrame:
        """
        Get all signal parameters across windows.

        Returns:
            DataFrame with window indices and signal parameter values.
        """
        if not self.window_results:
            return pd.DataFrame()

        data = []
        for wr in self.window_results:
            row = {"window_idx": wr.window_idx}
            row.update(wr.best_postprocess_params or {})
            data.append(row)
        return pd.DataFrame(data)

    def get_trading_performance_summary(self) -> Optional[pd.DataFrame]:
        """
        Get summary of trading performance across all windows.

        Returns:
            DataFrame with aggregated trading metrics.
        """
        if not self.window_results:
            return None

        results_df = pd.DataFrame([asdict(wr) for wr in self.window_results])
        summary = {
            "avg_sharpe": results_df["sharpe_ratio"].mean(),
            "std_sharpe": results_df["sharpe_ratio"].std(),
            "avg_return": results_df["total_return"].mean(),
            "std_return": results_df["total_return"].std(),
            "avg_drawdown": results_df["max_drawdown"].mean(),
            "avg_win_rate": results_df["win_rate"].mean(),
            "total_trades": results_df["num_trades"].sum(),
            "avg_sortino": results_df["sortino_ratio"].mean(),
            "avg_calmar": results_df["calmar_ratio"].mean(),
            "avg_profit_factor": results_df["profit_factor"].mean(),
            "best_sharpe": results_df["sharpe_ratio"].max(),
            "worst_drawdown": results_df["max_drawdown"].min(),
            "median_return": results_df["total_return"].median(),
        }
        return pd.DataFrame([summary])

    def get_signal_param_importance_summary(self) -> Optional[pd.DataFrame]:
        """
        Get signal parameter importance summary across all windows.

        Returns:
            DataFrame showing parameter selection patterns and stability.
        """
        if not self.signal_param_importance_history:
            return None

        # Collect all parameter names
        all_param_names = set()
        for window_data in self.signal_param_importance_history:
            all_param_names.update(window_data["params"].keys())

        # Build summary
        summary_data = []
        for param_name in sorted(all_param_names):
            values = []
            for window_data in self.signal_param_importance_history:
                value = window_data["params"].get(param_name)
                if value is not None:
                    values.append(value)

            if values:
                # For numeric parameters
                try:
                    numeric_values = [float(v) for v in values]
                    summary_data.append(
                        {
                            "parameter": param_name,
                            "type": "numeric",
                            "mean_value": np.mean(numeric_values),
                            "std_value": np.std(numeric_values),
                            "min_value": np.min(numeric_values),
                            "max_value": np.max(numeric_values),
                            "stability": 1
                            - (
                                np.std(numeric_values)
                                / (np.mean(numeric_values) + 1e-10)
                            ),
                        }
                    )
                except (ValueError, TypeError):
                    # For categorical parameters
                    value_counts = pd.Series(values).value_counts()
                    most_common = value_counts.index[0]
                    frequency = value_counts.iloc[0] / len(values)
                    summary_data.append(
                        {
                            "parameter": param_name,
                            "type": "categorical",
                            "most_common": most_common,
                            "frequency": frequency,
                            "unique_values": len(value_counts),
                            "stability": frequency,
                        }
                    )
        if summary_data:
            df = pd.DataFrame(summary_data)
            df = df.sort_values("stability", ascending=False)
            return df

        return None

    def get_best_portfolio(self) -> Optional[vbt.Portfolio]:
        """
        Get the VectorBT portfolio from the best window (same window as best models).

        Returns:
            VectorBT Portfolio object from the best performing window, or None if not available

        Example:
            # >>> instance.run()
            # >>> best_pf = optimizer.get_best_portfolio()
            # >>> best_pf.stats()  # Full VectorBT stats
            # >>> best_pf.plot().show()  # Plot equity curve
            # >>> best_pf.drawdowns.plot().show()  # Plot drawdowns
            # >>> best_pf.trades.records_readable  # Access trade details
        """
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

    # ==================== PRODUCTION EXPORT ====================

    def export_for_production(self, export_dir: str, include_signal_params: bool = True, include_metadata: bool = True) -> None:
        """
        Export complete production package: models + signal parameters + metadata.

        Extends parent's export_best_models() with signal-specific information.

        Args:
            export_dir: Directory to export to
            include_signal_params: Include signal parameters JSON
            include_metadata: Include comprehensive metadata
        """
        # Call parent's export to handle model files
        self.export_best_models(export_dir)

        if not include_signal_params and not include_metadata:
            return

        export_path = Path(export_dir)

        # Add signal parameters
        if include_signal_params:
            metadata = self.get_best_hyperparameters()
            if metadata and metadata.get("postprocess_parameters"):
                signal_params_file = export_path / "signal_parameters.json"
                with open(signal_params_file, "w") as f:
                    json.dump(metadata["postprocess_parameters"], f, indent=2, default=to_serializable)

                self.log.info("Signal parameters exported", "✓")

        # Add comprehensive metadata for backtesting
        if include_metadata:
            trading_summary = self.get_trading_performance_summary()
            if trading_summary is not None:
                trading_file = export_path / "trading_performance.json"
                with open(trading_file, "w") as f:
                    json.dump(trading_summary.to_dict(orient="records")[0], f, indent=2, default=to_serializable)

                self.log.info("Trading performance exported", "✓")

        # Update README with signal generator usage
        readme_path = export_path / "README.md"
        if readme_path.exists():
            with open(readme_path, "a") as f:
                f.write("\n\n## Signal Parameters\n")
                if include_signal_params:
                    metadata = self.get_best_hyperparameters()
                    f.write(
                        f"Optimal signal parameters: {metadata.get('postprocess_parameters', {})}\n"
                    )
                    f.write("See signal_parameters.json for details.\n\n")

                if include_metadata:
                    f.write("## Trading Performance\n")
                    f.write("See trading_performance.json for comprehensive metrics.\n")

    def plot_parameter_evolution(self, param_name: str, figsize=(18, 6), save_path: Optional[str] = None) -> None:
        """
        Plot how a specific signal parameter evolved across windows.

        Args:
            param_name: Name of the parameter to plot
            figsize: Figure size
            save_path: Path to save plot
        """
        try:
            signal_df = self.get_all_signal_parameters()
            if signal_df.empty or param_name not in signal_df.columns:
                self.log.warning(f"Parameter '{param_name}' not found")
                return

            fig, ax = plt.subplots(figsize=figsize)

            ax.plot(
                signal_df["window_idx"],
                signal_df[param_name],
                marker="o",
                linewidth=2,
                markersize=8,
                color="#2E86AB",
            )
            ax.set_title(
                f"Evolution of {param_name} Across Windows",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel("Window", fontsize=12)
            ax.set_ylabel(param_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#F8F9FA")

            plt.tight_layout()

            if save_path is None:
                save_path = self.checkpoint_dir / f"param_evolution_{param_name}.png"
            if self.env.is_interactive():
                plt.show()

            try:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                self.log.info(f"Parameter evolution saved to: {save_path}", "💾")
            except Exception as e:
                self.log.warning(f"Could not save plot: {e}")

            plt.close()

        except ImportError:
            self.log.warning("matplotlib not available")
