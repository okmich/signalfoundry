import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from itertools import product
from pathlib import Path
import ast
from typing import Dict, List, Any, Callable, Optional, Tuple, Protocol, Union

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqdm.auto import tqdm

from .vbt_export import QuantframeExportMixin

logger = logging.getLogger(__name__)

# Constants
MIN_FILE_SIZE_BYTES = 100
MAX_PARAM_COMBINATIONS = 100000
DEFAULT_MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N windows
MAX_FAILURE_RATE = 0.5  # Stop if more than 50% of windows fail
MIN_WINDOWS_FOR_EARLY_STOP = 5  # Minimum windows before early stopping kicks in


@dataclass
class WindowConfig:
    type: str  # 'rolling' or 'expanding'
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    step_size: Optional[int] = None
    min_train_size: Optional[int] = None


@dataclass
class EvaluationResult:
    params: Dict[str, Any]
    stats: Dict[str, Any]
    params_hash: str
    error: Optional[str] = None
    evaluation_time: float = 0.0


@dataclass
class PerformanceMetrics:
    total_windows: int = 0
    completed_windows: int = 0
    failed_windows: int = 0
    total_param_evaluations: int = 0
    total_time: float = 0.0
    avg_window_time: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    window_times: List[float] = field(default_factory=list)


@dataclass
class BacktesterConfig:
    """Complete configuration for the backtester"""

    param_grid: Dict[str, List[Any]]
    window_config: Dict[str, Any]
    vbt_config: Dict[str, Any]
    criteria: Dict[str, Any]
    n_top_params: int
    n_workers: int
    results_dir: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config_hash: str = ""

    def compute_hash(self) -> str:
        """Compute deterministic hash for configuration"""
        config_dict = {
            "param_grid": sorted(self.param_grid.items()),
            "window_config": sorted(self.window_config.items()),
            "vbt_config": sorted(self.vbt_config.items()),
            "criteria": sorted(self.criteria.items()),
            "n_top_params": self.n_top_params,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()


class SignalGenerator(Protocol):
    """Protocol defining the signal generator interface"""

    def generate(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate trading signals from data"""
        ...


class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


def atomic_write(file_path: Path, content: str, mode: str = "w"):
    """
    Atomic file write to prevent corruption.
    Writes to temp file first, then renames.
    """
    temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
    try:
        with open(temp_file, mode) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        temp_file.replace(file_path)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise e


class ParameterEvaluator:
    """Handles evaluation of parameter combinations on data"""

    # Cache for parameter hashes to improve performance
    _hash_cache: Dict[str, str] = {}

    def __init__(self, signal_class: Callable, vbt_config: Dict[str, Any]):
        self.signal_class = signal_class
        self.vbt_config = vbt_config

    @staticmethod
    def compute_params_hash(params: Dict[str, Any]) -> str:
        """Compute deterministic hash for parameters with caching"""
        # Create cache key
        cache_key = str(sorted(params.items()))

        if cache_key in ParameterEvaluator._hash_cache:
            return ParameterEvaluator._hash_cache[cache_key]

        # Compute hash
        sorted_items = sorted(params.items())
        param_str = json.dumps(sorted_items, sort_keys=True)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()

        # Cache it
        ParameterEvaluator._hash_cache[cache_key] = param_hash
        return param_hash

    def evaluate(self, params: Dict[str, Any], data: pd.DataFrame, max_retries: int = DEFAULT_MAX_RETRIES) -> EvaluationResult:
        """Evaluate a single parameter combination on given data with retry logic"""
        params_hash = self.compute_params_hash(params)
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                # Create signal generator with parameters
                signal_generator = self.signal_class(**params)

                # Generate signals
                long_entries, long_exits, short_entries, short_exits = (
                    signal_generator.generate(data)
                )

                # Run backtest
                portfolio = vbt.Portfolio.from_signals(
                    data["close"],
                    entries=long_entries,
                    exits=long_exits,
                    short_entries=short_entries,
                    short_exits=short_exits,
                    **self.vbt_config,
                )
                stats = portfolio.stats()

                eval_time = time.time() - start_time

                return EvaluationResult(
                    params=params,
                    stats=stats.to_dict(),
                    params_hash=params_hash,
                    evaluation_time=eval_time,
                )
            except KeyError as e:
                # Data structure error - don't retry
                error_msg = f"Missing required column in data: {str(e)}"
                logger.error(
                    f"Parameter evaluation failed (no retry): {error_msg}, params: {params}"
                )
                return EvaluationResult(
                    params=params,
                    stats={},
                    params_hash=params_hash,
                    error=error_msg,
                    evaluation_time=time.time() - start_time,
                )
            except Exception as e:
                error_msg = f"Error evaluating parameters: {str(e)}"
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for params {params}: {error_msg}"
                    )
                else:
                    logger.error(
                        f"Failed after {max_retries} attempts for params {params}: {error_msg}"
                    )
                    return EvaluationResult(
                        params=params,
                        stats={},
                        params_hash=params_hash,
                        error=error_msg,
                        evaluation_time=time.time() - start_time,
                    )

        return EvaluationResult(
            params=params,
            stats={},
            params_hash=params_hash,
            error="Max retries exceeded",
            evaluation_time=time.time() - start_time,
        )


class ResultFilter:
    """Handles filtering and ranking of evaluation results"""

    def __init__(self, criteria: Dict[str, Any]):
        self.criteria = criteria
        self._validate_criteria()

    def _validate_criteria(self):
        if "primary_metric" not in self.criteria:
            raise ValidationError("Criteria must contain 'primary_metric'")

        filters = self.criteria.get("filters", {})
        for filter_name, filter_value in filters.items():
            if isinstance(filter_value, tuple):
                if len(filter_value) != 2:
                    raise ValidationError(
                        f"Tuple filter for '{filter_name}' must have exactly 2 elements (min, max)"
                    )
                if filter_value[0] > filter_value[1]:
                    raise ValidationError(
                        f"Invalid range for '{filter_name}': min ({filter_value[0]}) > max ({filter_value[1]})"
                    )

    def filter_and_rank(self, results: List[EvaluationResult], n_top: int) -> List[EvaluationResult]:
        """Filter and rank results based on criteria"""
        valid_results = []

        for result in results:
            if result.error is not None or not result.stats:
                continue

            if self._passes_filters(result.stats):
                valid_results.append(result)

        primary_metric = self.criteria["primary_metric"]
        reverse = self.criteria.get("higher_is_better", True)

        valid_results.sort(
            key=lambda x: x.stats.get(primary_metric, -float("inf") if reverse else float("inf")),
            reverse=reverse,
        )
        return valid_results[:n_top]

    def _passes_filters(self, stats: Dict[str, Any]) -> bool:
        """Check if stats pass all filters"""
        filters = self.criteria.get("filters", {})

        for filter_name, filter_value in filters.items():
            if filter_name not in stats:
                logger.warning(f"Filter metric '{filter_name}' not found in stats")
                return False

            stat_value = stats[filter_name]

            if isinstance(filter_value, tuple) and len(filter_value) == 2:
                # Range filter (min, max)
                min_val, max_val = filter_value
                if not (min_val <= stat_value <= max_val):
                    return False
            elif isinstance(filter_value, dict):
                # Support for min/max separately
                if "min" in filter_value and stat_value < filter_value["min"]:
                    return False
                if "max" in filter_value and stat_value > filter_value["max"]:
                    return False
            else:
                # Minimum value filter
                if stat_value < filter_value:
                    return False
        return True


class CheckpointManager:
    """Manages checkpointing and progress persistence"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = checkpoint_dir / "progress.json"

    def save_checkpoint(
        self,
        completed_window_ids: List[str],
        total_windows: int,
        config_hash: str,
        performance_metrics: Optional[PerformanceMetrics] = None,
    ):
        """Save progress checkpoint with configuration validation"""
        checkpoint_data = {
            "completed_window_ids": completed_window_ids,
            "total_windows": total_windows,
            "completed_count": len(completed_window_ids),
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
        }

        if performance_metrics:
            checkpoint_data["performance_metrics"] = asdict(performance_metrics)

        try:
            content = json.dumps(checkpoint_data, indent=2)
            atomic_write(self.checkpoint_file, content)
            logger.info(
                f"Checkpoint saved: {len(completed_window_ids)}/{total_windows} windows"
            )
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(
        self, expected_config_hash: str
    ) -> Optional[Tuple[List[str], Optional[PerformanceMetrics]]]:
        """Load progress checkpoint and validate configuration"""
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            # Validate configuration hasn't changed
            saved_hash = checkpoint_data.get("config_hash", "")
            if saved_hash != expected_config_hash:
                logger.warning(
                    f"Configuration has changed since checkpoint was created. "
                    f"Checkpoint will be ignored. Delete checkpoint file to start fresh."
                )
                return None

            completed = checkpoint_data.get("completed_window_ids", [])

            # Load performance metrics if available
            perf_metrics = None
            if "performance_metrics" in checkpoint_data:
                perf_data = checkpoint_data["performance_metrics"]
                perf_metrics = PerformanceMetrics(**perf_data)

            logger.info(
                f"Resuming from checkpoint: {len(completed)} windows already completed"
            )
            return completed, perf_metrics
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        """Clear checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                logger.info("Checkpoint cleared")
            except Exception as e:
                logger.error(f"Failed to clear checkpoint: {e}")


# Optimized hash computation with caching
_window_hash_cache: Dict[str, str] = {}


def compute_window_hash(train_range: Tuple, test_range: Tuple, param_combinations_hash: str) -> str:
    """Compute deterministic hash for a window configuration (optimized)"""
    # Create cache key
    cache_key = f"{train_range[0]}_{train_range[1]}_{test_range[0]}_{test_range[1]}_{param_combinations_hash}"

    if cache_key in _window_hash_cache:
        return _window_hash_cache[cache_key]

    # Compute hash
    window_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    _window_hash_cache[cache_key] = window_hash
    return window_hash


def process_window_worker(args):
    """
    Standalone worker function that can be pickled.
    This function contains all the logic needed to process a window.
    """
    (
        window_idx, window_data, signal_class, param_combinations, vbt_config, criteria,
        n_top_params, temp_dir, param_combinations_hash) = args

    # Setup logging for worker
    worker_logger = logging.getLogger(f"{__name__}.worker_{window_idx}")

    window_start_time = time.time()

    # Create a unique hash for this window
    window_hash = compute_window_hash(window_data["train_range"], window_data["test_range"], param_combinations_hash)

    temp_file = Path(temp_dir) / f"window_{window_hash}.json"
    # Check if already processed
    if temp_file.exists():
        worker_logger.info(f"Window {window_idx} already processed (hash: {window_hash[:8]}...)")
        return window_hash, 0.0  # Return hash and time

    worker_logger.info(f"Processing window {window_idx} (hash: {window_hash[:8]}...)")

    window_results = {
        "window_id": window_hash,
        "window_idx": window_idx,
        "train_range": [
            str(window_data["train_range"][0]),
            str(window_data["train_range"][1]),
        ],
        "test_range": [
            str(window_data["test_range"][0]),
            str(window_data["test_range"][1]),
        ],
        "train_results": [],
        "test_results": [],
        "errors": [],
        "metrics": {
            "n_params_evaluated": 0,
            "n_params_passed_filters": 0,
            "processing_time": 0.0,
        },
    }

    try:
        # Initialize evaluator and filter
        evaluator = ParameterEvaluator(signal_class, vbt_config)
        result_filter = ResultFilter(criteria)

        # Evaluate all parameters on training data
        worker_logger.info(f"Evaluating {len(param_combinations)} parameter combinations on training data")
        train_results = []
        for params in param_combinations:
            result = evaluator.evaluate(params, window_data["train"])
            train_results.append(result)
            window_results["metrics"]["n_params_evaluated"] += 1

            if result.error:
                window_results["errors"].append({"phase": "train", "params": params, "error": result.error})

        # Filter and rank training results
        top_params = result_filter.filter_and_rank(train_results, n_top_params)
        window_results["metrics"]["n_params_passed_filters"] = len(top_params)
        worker_logger.info(f"Selected {len(top_params)} top parameters from training")

        if not top_params:
            worker_logger.warning(f"No valid parameters found for window {window_idx}")
            window_results["metrics"]["processing_time"] = time.time() - window_start_time

            # Still save the file to mark as processed
            content = json.dumps(window_results, indent=2, default=str)
            atomic_write(temp_file, content)
            return window_hash, window_results["metrics"]["processing_time"]

        # Evaluate top parameters on test data
        worker_logger.info(f"Evaluating {len(top_params)} top parameters on test data")
        for train_result in top_params:
            test_result = evaluator.evaluate(train_result.params, window_data["test"])

            if test_result.error:
                window_results["errors"].append(
                    {
                        "phase": "test",
                        "params": train_result.params,
                        "error": test_result.error,
                    }
                )
            else:
                # Convert EvaluationResult to dict for JSON serialization
                window_results["train_results"].append(
                    {
                        "params": train_result.params,
                        "stats": train_result.stats,
                        "params_hash": train_result.params_hash,
                        "evaluation_time": train_result.evaluation_time,
                    }
                )
                window_results["test_results"].append(
                    {
                        "params": test_result.params,
                        "stats": test_result.stats,
                        "params_hash": test_result.params_hash,
                        "evaluation_time": test_result.evaluation_time,
                    }
                )

        # Calculate total processing time
        window_results["metrics"]["processing_time"] = time.time() - window_start_time

        # Save to temporary JSON file atomically
        content = json.dumps(window_results, indent=2, default=str)
        atomic_write(temp_file, content)

        worker_logger.info(
            f"Window {window_idx} completed in {window_results['metrics']['processing_time']:.2f}s"
        )
        return window_hash, window_results["metrics"]["processing_time"]

    except Exception as e:
        error_msg = f"Fatal error processing window {window_idx}: {str(e)}"
        worker_logger.error(error_msg)
        window_results["errors"].append({"phase": "fatal", "error": error_msg})
        window_results["metrics"]["processing_time"] = time.time() - window_start_time

        # Save partial results
        try:
            content = json.dumps(window_results, indent=2, default=str)
            atomic_write(temp_file, content)
        except Exception as save_error:
            worker_logger.error(f"Failed to save error state: {save_error}")

        raise  # Re-raise to be caught by executor


class VectobtWalkForwardBacktester(QuantframeExportMixin):
    """
    Walk-forward backtesting framework using vectorbt.

    This class implements a robust walk-forward optimization and backtesting system that evaluates trading strategies
    across multiple time windows with parallel processing, checkpointing, comprehensive error handling, and performance tracking.

    Example:
        >>> backtester = VectobtWalkForwardBacktester(
        ...     signal_class=MySignalGenerator,
        ...     param_grid={'fast': [10, 20], 'slow': [50, 100]},
        ...     vbt_config={'init_cash': 10000, 'fees': 0.001},
        ...     window_config={'type': 'rolling', 'train_size': 252, 'test_size': 63},
        ...     criteria={'primary_metric': 'Sharpe Ratio', 'higher_is_better': True},
        ...     n_top_params=5,
        ...     n_workers=4
        ... )
        >>> results_df = backtester.run(data)
        >>> summary = backtester.get_performance_summary()
    """

    def __init__(self, signal_class: Callable, param_grid: Dict[str, List[Any]], vbt_config: Dict[str, Any],
                 window_config: Dict[str, Any], criteria: Dict[str, Any], results_dir: str = "./wf_results",
                 n_top_params: int = 5, n_workers: int = 4, param_filter: Callable = None):
        """
        Initialize the walk-forward backtester.

        Args:
            signal_class: Class or factory function that creates signal generators.
                         Must implement generate(data) -> (long_entries, long_exits, short_entries, short_exits)
            param_grid: Dictionary mapping parameter names to lists of values.
                       Example: {'fast_period': [10, 20, 30], 'slow_period': [50, 100]}
            vbt_config: Configuration dictionary for vectorbt Portfolio.
                       Example: {'init_cash': 10000, 'fees': 0.001, 'slippage': 0.001}
            window_config: Configuration for window creation.
                          For rolling: {'type': 'rolling', 'train_size': 252, 'test_size': 63, 'step_size': 21}
                          For expanding: {'type': 'expanding', 'min_train_size': 252, 'test_size': 63}
            criteria: Dictionary defining filtering and ranking criteria.
                     Example: {
                         'primary_metric': 'Sharpe Ratio',
                         'higher_is_better': True,
                         'filters': {
                             'Total Return [%]': 0,  # Minimum value
                             'Max Drawdown [%]': (0, 50),  # Range (min, max)
                             'Win Rate [%]': {'min': 40, 'max': 100}  # Dict format
                         }
                     }
            results_dir: Directory to save results
            n_top_params: Number of top parameters to select from each training window
            n_workers: Number of parallel workers for processing
            param_filter: Optional callable(params: dict) -> bool that returns True to keep
                         a parameter combination. Useful for excluding invalid combos such as
                         ma_fast >= ma_slow before any evaluation is attempted.
        """
        self.signal_class = signal_class
        self.param_grid = param_grid
        self.param_filter = param_filter
        self.vbt_config = vbt_config
        self.window_config = self._parse_window_config(window_config)
        self.criteria = criteria
        self.results_dir = Path(results_dir)
        self.n_top_params = n_top_params
        self.n_workers = n_workers
        self.temp_dir = None
        self.checkpoint_manager = None
        self.performance_metrics = PerformanceMetrics()
        self._validate_inputs()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.param_combinations = self._generate_param_combinations()
        self.param_combinations_hash = self._compute_param_combinations_hash()
        self._validate_param_grid_size()
        self.config = BacktesterConfig(param_grid=param_grid, window_config=window_config, vbt_config=vbt_config,
                                       criteria=criteria, n_top_params=n_top_params, n_workers=n_workers,
                                       results_dir=str(results_dir))
        self.config.config_hash = self.config.compute_hash()

        logger.info(
            f"Initialized VectobtWalkForwardBacktester with {len(self.param_combinations)} "
            f"parameter combinations and {n_workers} workers"
        )
        self._save_config()

    def _compute_param_combinations_hash(self) -> str:
        """Compute hash for all parameter combinations"""
        # def to_natural_key(k):
        #     if isinstance(k, (np.integer, np.floating)):
        #         return k.item()
        #     return k
        params_str = json.dumps(
            [sorted(p.items()) for p in self.param_combinations],
            sort_keys=True,
            default=lambda k: k.item() if isinstance(k, (np.integer, np.floating)) else k,
        )
        return hashlib.sha256(params_str.encode()).hexdigest()

    def _save_config(self):
        """Save configuration to file"""
        try:
            config_file = self.results_dir / "config.json"
            content = json.dumps(asdict(self.config), indent=2, default=str)
            atomic_write(config_file, content)
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.warning(f"Failed to save configuration: {e}")

    def _parse_window_config(self, config: Dict[str, Any]) -> WindowConfig:
        """Parse and validate window configuration"""
        return WindowConfig(
            type=config.get("type"),
            train_size=config.get("train_size"),
            test_size=config.get("test_size"),
            step_size=config.get("step_size"),
            min_train_size=config.get("min_train_size")
        )

    def _validate_inputs(self):
        """Validate all input parameters"""
        if not callable(self.signal_class):
            raise ValidationError("signal_class must be callable")

        if not self.param_grid:
            raise ValidationError("param_grid cannot be empty")

        for param_name, values in self.param_grid.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValidationError(f"param_grid['{param_name}'] must be a non-empty list")

        if self.window_config.type not in ["rolling", "expanding"]:
            raise ValidationError("window_config['type'] must be 'rolling' or 'expanding'")

        if self.window_config.type == "rolling":
            if not self.window_config.train_size or not self.window_config.test_size:
                raise ValidationError("window_config must contain 'train_size' and 'test_size' for rolling windows")
            if self.window_config.train_size <= 0 or self.window_config.test_size <= 0:
                raise ValidationError("train_size and test_size must be positive")

        elif self.window_config.type == "expanding":
            if not self.window_config.min_train_size or not self.window_config.test_size:
                raise ValidationError("window_config must contain 'min_train_size' and 'test_size' for expanding windows")

            if self.window_config.min_train_size <= 0 or self.window_config.test_size <= 0:
                raise ValidationError("min_train_size and test_size must be positive")

        if not isinstance(self.vbt_config, dict):
            raise ValidationError("vbt_config must be a dictionary")

        if not isinstance(self.criteria, dict):
            raise ValidationError("criteria must be a dictionary")

        try:
            ResultFilter(self.criteria)
        except ValidationError as e:
            raise ValidationError(f"Invalid criteria: {e}")

        if self.n_top_params <= 0:
            raise ValidationError("n_top_params must be positive")

        if self.n_workers <= 0:
            raise ValidationError("n_workers must be positive")

    def _validate_param_grid_size(self):
        """Validate that parameter grid won't create too many combinations"""
        n_combinations = len(self.param_combinations)
        if n_combinations > MAX_PARAM_COMBINATIONS:
            raise ValidationError(
                f"Parameter grid creates {n_combinations} combinations, "
                f"which exceeds the maximum of {MAX_PARAM_COMBINATIONS}. "
                f"Please reduce the parameter grid size."
            )

        if n_combinations > 10000:
            logger.warning(
                f"Parameter grid creates {n_combinations} combinations. This may take a long time to process.")

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(product(*values))

        all_combos = [dict(zip(keys, combo)) for combo in combinations]
        if self.param_filter is not None:
            before = len(all_combos)
            all_combos = [p for p in all_combos if self.param_filter(p)]
            logger.info(f"param_filter removed {before - len(all_combos)} invalid combinations "
                        f"({len(all_combos)} remaining)")
        return all_combos

    def _validate_data(self, data: pd.DataFrame):
        """Validate input data structure"""
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("data must be a pandas DataFrame")

        if len(data) == 0:
            raise ValidationError("data cannot be empty")

        # Check for required columns
        required_columns = ["close"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(f"data is missing required columns: {missing_columns}")

        # Check for valid index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("data index is not DatetimeIndex. This may cause issues.")

        # Check for NaN values
        if data["close"].isna().any():
            raise ValidationError("data contains NaN values in 'close' column")

        # Validate sufficient data for windows
        if self.window_config.type == "rolling":
            min_required = self.window_config.train_size + self.window_config.test_size
            if len(data) < min_required:
                raise ValidationError(
                    f"data has {len(data)} rows but rolling windows require at least "
                    f"{min_required} rows (train_size + test_size)"
                )
        elif self.window_config.type == "expanding":
            min_required = self.window_config.min_train_size + self.window_config.test_size
            if len(data) < min_required:
                raise ValidationError(
                    f"data has {len(data)} rows but expanding windows require at least "
                    f"{min_required} rows (min_train_size + test_size)"
                )

    def _create_windows(self, data: pd.DataFrame) -> List[Dict[str, pd.DataFrame]]:
        """Create in-sample and out-of-sample windows based on config"""
        windows = []

        if self.window_config.type == "rolling":
            train_size = self.window_config.train_size
            test_size = self.window_config.test_size
            step = (
                self.window_config.step_size
                if self.window_config.step_size
                else test_size
            )

            for i in range(0, len(data) - train_size - test_size + 1, step):
                train_data = data.iloc[i : i + train_size]
                test_data = data.iloc[i + train_size : i + train_size + test_size]

                if len(train_data) == train_size and len(test_data) == test_size:
                    windows.append(
                        {
                            "train": train_data,
                            "test": test_data,
                            "train_range": (train_data.index[0], train_data.index[-1]),
                            "test_range": (test_data.index[0], test_data.index[-1]),
                        }
                    )

        elif self.window_config.type == "expanding":
            min_train_size = self.window_config.min_train_size
            test_size = self.window_config.test_size
            step = (
                self.window_config.step_size
                if self.window_config.step_size
                else test_size
            )

            for i in range(min_train_size, len(data) - test_size + 1, step):
                train_data = data.iloc[:i]
                test_data = data.iloc[i : i + test_size]

                if len(test_data) == test_size:
                    windows.append(
                        {
                            "train": train_data,
                            "test": test_data,
                            "train_range": (train_data.index[0], train_data.index[-1]),
                            "test_range": (test_data.index[0], test_data.index[-1]),
                        }
                    )

        if not windows:
            raise ValidationError("No valid windows could be created from the data")

        logger.info(f"Created {len(windows)} windows")
        return windows

    @contextmanager
    def _managed_resources(self):
        """Context manager for proper resource cleanup"""
        _completed_successfully = False
        try:
            # Create temporary directory
            self.temp_dir = Path(
                tempfile.mkdtemp(prefix="wf_temp_", dir=self.results_dir)
            )
            logger.info(f"Created temporary directory: {self.temp_dir}")

            # Initialize checkpoint manager
            self.checkpoint_manager = CheckpointManager(
                self.results_dir / "checkpoints"
            )

            yield

            _completed_successfully = True

        finally:
            # Always remove the temporary directory; only clear the checkpoint
            # when the run finished normally so that resume=True works after a crash.
            self._cleanup(clear_checkpoint=_completed_successfully)

    def run(self, data: pd.DataFrame, resume: bool = True) -> pd.DataFrame:
        """
        Run walk-forward backtest on provided data.

        Args:
            data: DataFrame containing OHLCV data with DatetimeIndex
            resume: Whether to resume from checkpoint if available

        Returns:
            DataFrame containing all results across windows

        Raises:
            ValidationError: If input data or configuration is invalid
            RuntimeError: If no results were generated
        """
        # Validate data
        self._validate_data(data)

        # Create windows
        windows = self._create_windows(data)
        self.performance_metrics.total_windows = len(windows)
        self.performance_metrics.start_time = time.time()

        logger.info(f"Starting walk-forward backtest with {len(windows)} windows")

        # Use context manager for resource management
        with self._managed_resources():
            # Check for existing progress
            completed_window_ids = []
            if resume:
                checkpoint_result = self.checkpoint_manager.load_checkpoint(self.config.config_hash)
                if checkpoint_result:
                    completed_window_ids, loaded_metrics = checkpoint_result
                    if loaded_metrics:
                        self.performance_metrics = loaded_metrics

            # Prepare arguments for each window
            window_args = []
            for i, window in enumerate(windows):
                window_hash = compute_window_hash(window["train_range"], window["test_range"], self.param_combinations_hash)

                # Skip if already completed
                if window_hash in completed_window_ids:
                    logger.info(f"Skipping window {i} (already completed)")
                    continue

                window_args.append(
                    (
                        i,
                        window,
                        self.signal_class,
                        self.param_combinations,
                        self.vbt_config,
                        self.criteria,
                        self.n_top_params,
                        str(self.temp_dir),
                        self.param_combinations_hash,
                    )
                )

            if not window_args:
                logger.info("All windows already completed. Loading results...")
            else:
                logger.info(f"Processing {len(window_args)} windows ({len(completed_window_ids)} already completed)")

                # Process windows in parallel
                completed_count = len(completed_window_ids)
                failed_windows = []

                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    # Submit all tasks
                    future_to_idx = {
                        executor.submit(process_window_worker, args): args[0]
                        for args in window_args
                    }

                    # Process completed tasks with progress bar
                    with tqdm(total=len(window_args), desc="Processing windows", initial=0) as pbar:
                        for future in as_completed(future_to_idx):
                            window_idx = future_to_idx[future]
                            try:
                                window_hash, window_time = future.result()
                                if window_hash:
                                    completed_window_ids.append(window_hash)
                                    completed_count += 1
                                    self.performance_metrics.completed_windows += 1
                                    self.performance_metrics.total_param_evaluations += len(self.param_combinations)
                                    self.performance_metrics.window_times.append(
                                        window_time
                                    )

                                    # Save checkpoint periodically
                                    if completed_count % CHECKPOINT_INTERVAL == 0:
                                        self.checkpoint_manager.save_checkpoint(
                                            completed_window_ids,
                                            len(windows),
                                            self.config.config_hash,
                                            self.performance_metrics,
                                        )
                            except Exception as e:
                                logger.error(f"Window {window_idx} failed with error: {e}")
                                failed_windows.append((window_idx, str(e)))
                                self.performance_metrics.failed_windows += 1

                                # Early stopping if failure rate is too high
                                if self._should_stop_early(completed_count, len(failed_windows)):
                                    logger.error(f"Stopping early: failure rate exceeded {MAX_FAILURE_RATE * 100}%")
                                    pbar.close()
                                    executor.shutdown(wait=False, cancel_futures=True)
                                    break
                            finally:
                                pbar.update(1)

                # Final checkpoint
                self.checkpoint_manager.save_checkpoint(
                    completed_window_ids,
                    len(windows),
                    self.config.config_hash,
                    self.performance_metrics,
                )

                # Report failures
                if failed_windows:
                    logger.error(f"{len(failed_windows)} windows failed:")
                    for idx, error in failed_windows[:10]:  # Limit output
                        logger.error(f"  Window {idx}: {error}")
                    if len(failed_windows) > 10:
                        logger.error(f"  ... and {len(failed_windows) - 10} more")

            # Load and process results
            logger.info("Loading and formatting results...")
            all_results = self._load_temp_results()

            if not all_results:
                raise RuntimeError("No results were generated. Check logs for errors.")
            results_df = self._stream_format_results(all_results)

            # Update final performance metrics
            self.performance_metrics.end_time = time.time()
            self.performance_metrics.total_time = self.performance_metrics.end_time - self.performance_metrics.start_time
            if self.performance_metrics.window_times:
                self.performance_metrics.avg_window_time = sum(self.performance_metrics.window_times) / len(self.performance_metrics.window_times)

            # Save final results
            self._save_final_results(results_df, all_results)

        logger.info(f"Walk-forward backtest completed. Results shape: {results_df.shape}")
        logger.info(
            f"Total time: {self.performance_metrics.total_time:.2f}s, "
            f"Avg window time: {self.performance_metrics.avg_window_time:.2f}s"
        )
        return results_df

    def _should_stop_early(self, completed: int, failed: int) -> bool:
        """Determine if early stopping should be triggered"""
        total = completed + failed
        if total < MIN_WINDOWS_FOR_EARLY_STOP:
            return False

        failure_rate = failed / total
        return failure_rate > MAX_FAILURE_RATE

    def _stream_format_results(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Stream format results to reduce memory usage"""
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        dfs = []

        current_rows = []
        for window in all_results:
            # Skip windows with no results in either train or test
            if not window.get("train_results") or not window.get("test_results"):
                logger.warning(f"Window {window.get('window_id', 'unknown')} has no valid results")
                continue

            for train_result, test_result in zip(window["train_results"], window["test_results"]):
                row = {
                    "window_id": window["window_id"],
                    "train_start": window["train_range"][0],
                    "train_end": window["train_range"][1],
                    "test_start": window["test_range"][0],
                    "test_end": window["test_range"][1],
                    "params_hash": train_result["params_hash"],
                }

                # Add parameter values
                for key, value in train_result["params"].items():
                    row[f"param_{key}"] = value

                # Add training stats
                for key, value in train_result["stats"].items():
                    row[f"train_{key}"] = value

                # Add test stats
                for key, value in test_result["stats"].items():
                    row[f"test_{key}"] = value

                current_rows.append(row)

                # Process chunk
                if len(current_rows) >= chunk_size:
                    dfs.append(pd.DataFrame(current_rows))
                    current_rows = []

        # Process remaining rows
        if current_rows:
            dfs.append(pd.DataFrame(current_rows))

        if not dfs:
            logger.warning("No valid results to format")
            return pd.DataFrame()

        # Concatenate all chunks
        return pd.concat(dfs, ignore_index=True)

    def _load_temp_results(self) -> List[Dict[str, Any]]:
        """Load all results from temporary JSON files"""
        all_results = []
        if self.temp_dir and self.temp_dir.exists():
            json_files = list(self.temp_dir.glob("window_*.json"))
            logger.info(f"Loading {len(json_files)} result files...")

            for temp_file in tqdm(json_files, desc="Loading results"):
                try:
                    file_size = os.path.getsize(temp_file)
                    if file_size > MIN_FILE_SIZE_BYTES:
                        with open(temp_file, "r") as f:
                            result = json.load(f)
                            all_results.append(result)
                    else:
                        logger.warning(
                            f"Skipping small/empty file: {temp_file} ({file_size} bytes)"
                        )
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON file {temp_file}: {e}")
                except Exception as e:
                    logger.error(f"Error loading temp file {temp_file}: {e}")

        logger.info(f"Loaded {len(all_results)} result sets")
        return all_results

    def _save_final_results(self, results_df: pd.DataFrame, all_results: List[Dict[str, Any]]) -> None:
        """Save final results and performance metrics"""
        logger.info("Saving final results...")

        # Save DataFrame in multiple formats
        results_df.to_parquet(self.results_dir / "final_results.parquet")
        results_df.to_csv(self.results_dir / "final_results.csv", index=False)
        logger.info(f"Saved results DataFrame ({results_df.shape[0]} rows)")

        # Save detailed results as JSON
        with open(self.results_dir / "detailed_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Create structured JSON output
        json_data = []
        for _, row in results_df.iterrows():
            json_row = {
                "window_id": row["window_id"],
                "params_hash": row["params_hash"],
                "train_period": f"{row['train_start']} to {row['train_end']}",
                "test_period": f"{row['test_start']} to {row['test_end']}",
                "parameters": {
                    k.replace("param_", ""): v
                    for k, v in row.items()
                    if k.startswith("param_")
                },
                "train_stats": {
                    k.replace("train_", ""): v
                    for k, v in row.items()
                    if k.startswith("train_")
                },
                "test_stats": {
                    k.replace("test_", ""): v
                    for k, v in row.items()
                    if k.startswith("test_")
                },
            }
            json_data.append(json_row)

        with open(self.results_dir / "final_results.json", "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        # Save performance metrics
        self._save_performance_metrics()

        logger.info(f"Saved all results to {self.results_dir}")

    def _save_performance_metrics(self):
        """Save performance metrics to file"""
        try:
            metrics_file = self.results_dir / "performance_metrics.json"
            content = json.dumps(asdict(self.performance_metrics), indent=2, default=str)
            atomic_write(metrics_file, content)
            logger.info(f"Performance metrics saved to {metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to save performance metrics: {e}")

    def _cleanup(self, clear_checkpoint: bool = False):
        # Always clean up temporary directory (it holds only transient data)
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to remove temporary directory: {e}")

        # Only clear the checkpoint when the run finished successfully.
        # Leaving it intact on failure allows resume=True to pick up where it
        # left off on the next invocation.
        if clear_checkpoint and self.checkpoint_manager:
            self.checkpoint_manager.clear_checkpoint()

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.

        Returns:
            Dictionary containing performance statistics
        """
        return {
            "total_windows": self.performance_metrics.total_windows,
            "completed_windows": self.performance_metrics.completed_windows,
            "failed_windows": self.performance_metrics.failed_windows,
            "success_rate": (
                self.performance_metrics.completed_windows
                / self.performance_metrics.total_windows
                if self.performance_metrics.total_windows > 0
                else 0
            ),
            "total_time_seconds": self.performance_metrics.total_time,
            "avg_window_time_seconds": self.performance_metrics.avg_window_time,
            "total_param_evaluations": self.performance_metrics.total_param_evaluations,
        }

    def analyze_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze walk-forward results to identify best parameters.

        Args:
            results_df: Results DataFrame from run()

        Returns:
            DataFrame with aggregated statistics per parameter combination
        """
        if results_df.empty:
            logger.warning("No results to analyze")
            return pd.DataFrame()

        # Get parameter columns
        param_cols = [col for col in results_df.columns if col.startswith("param_")]

        # Coerce unhashable param values (e.g. dicts passed as fixed grid values)
        # to their string representation so pandas groupby can hash them.
        results_df = results_df.copy()
        for col in param_cols:
            if results_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                results_df[col] = results_df[col].apply(
                    lambda x: str(x) if isinstance(x, (dict, list)) else x
                )

        # Group by parameters
        grouped = results_df.groupby(param_cols)

        # Calculate aggregate statistics
        primary_metric = self.criteria["primary_metric"]
        test_metric = f"test_{primary_metric}"

        if test_metric not in results_df.columns:
            logger.error(f"Primary metric '{primary_metric}' not found in results")
            return pd.DataFrame()

        analysis = grouped.agg(
            {
                test_metric: ["mean", "std", "min", "max", "count"],
                "params_hash": "first",
            }
        ).reset_index()

        # Flatten column names
        analysis.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in analysis.columns.values
        ]

        # Sort by mean performance
        higher_is_better = self.criteria.get("higher_is_better", True)
        analysis = analysis.sort_values(
            f"{test_metric}_mean", ascending=not higher_is_better
        ).reset_index(drop=True)

        logger.info(f"Analysis complete: {len(analysis)} unique parameter combinations")
        return analysis

    def get_best_params(self, results_df: pd.DataFrame, n: int = 1) -> List[Dict[str, Any]]:
        """
        Get the best performing parameter combinations.

        Args:
            results_df: Results DataFrame from run()
            n: Number of top parameter combinations to return

        Returns:
            List of dictionaries containing parameter combinations
        """
        analysis = self.analyze_results(results_df)

        if analysis.empty:
            return []

        param_cols = [col for col in analysis.columns if col.startswith("param_")]
        best_params = []

        for _, row in analysis.head(n).iterrows():
            params = {col.replace("param_", ""): row[col] for col in param_cols}
            best_params.append(params)

        return best_params

    @staticmethod
    def _coerce_param_value(value: Any) -> Any:
        """Parse string-encoded dicts/lists back to their original Python types.

        When unhashable param values (e.g. a ``regime_direction`` dict) are coerced to strings for groupby inside
        ``analyze_results``, the string representation is what ends up in the ``get_best_params`` output.
        This method reverses that coercion so the value can be passed directly to the signal-class constructor.

        JSON serialisation converts integer dict keys to strings (e.g. ``{0: 1}`` → ``{'0': 1}``).
        If all keys of a parsed dict look like integers they are converted back to ``int`` so that
        e.g. ``regime.map({0: 1, 1: -1, 2: 0})`` works correctly.
        """
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    # Restore integer keys that were stringified by JSON serialisation
                    if all(isinstance(k, str) and k.lstrip('-').isdigit() for k in parsed):
                        parsed = {int(k): v for k, v in parsed.items()}
                    return parsed
                if isinstance(parsed, (list, tuple)):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return value

    def reconstruct_portfolio(self, data: pd.DataFrame, params: Dict[str, Any]) -> Any:
        """Reconstruct a ``vbt.Portfolio`` for an arbitrary parameter set.

        ``VectobtWalkForwardBacktester`` runs each window in a worker process, so portfolio objects cannot be returned
        across process boundaries — they are discarded after ``.stats()`` is extracted.  Use this method to rebuild a
        portfolio from any parameter dict (e.g. from ``get_best_params()``).

        Args:
            data: DataFrame passed to ``signal_class.generate()``.  Must
                contain whatever columns the signal needs (e.g. ``'close'``, ``'regime'``).
            params: Parameter dict, e.g. ``{'ma_fast': 10, 'ma_slow': 30}``.
                String-encoded dict/list values produced by
                ``analyze_results``'s groupby-safe serialisation are automatically parsed back to their original types.

        Returns:
            ``vbt.Portfolio`` instance.
        """
        import vectorbt as vbt

        coerced = {k: self._coerce_param_value(v) for k, v in params.items()}
        signal = self.signal_class(**coerced)
        long_entries, long_exits, short_entries, short_exits = signal.generate(data)

        return vbt.Portfolio.from_signals(
            data["close"],
            entries=long_entries,
            exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            **self.vbt_config,
        )

    def get_best_portfolio(self, data: pd.DataFrame, results_df: pd.DataFrame, n: int = 1) -> Union[Any, List[Any]]:
        """Reconstruct portfolio(s) for the top-n OOS parameter combinations.

        Convenience wrapper: ``analyze_results`` → ``get_best_params`` → ``reconstruct_portfolio``.

        Args:
            data: DataFrame for signal generation (full dataset or any window).
            results_df: Walk-forward results DataFrame from ``run()``.
            n: Number of top parameter combinations to reconstruct.

        Returns:
            A single ``vbt.Portfolio`` when ``n == 1``, otherwise a list.
        """
        best_params_list = self.get_best_params(results_df, n=n)
        portfolios = [self.reconstruct_portfolio(data, p) for p in best_params_list]
        return portfolios[0] if n == 1 else portfolios

    def analyze_best_by_regime(
        self,
        data: pd.DataFrame,
        results_df: pd.DataFrame,
        regime_labels: pd.Series,
        regime_names: Optional[Dict[str, Any]] = None,
        annualization_factor: int = 252,
        n: int = 1,
    ) -> Union[Any, List[Any]]:
        """Reconstruct best portfolio(s) from results_df and analyze by regime.

        Wraps ``get_best_portfolio()`` to produce one or more
        ``RegimePerformanceAnalyzer`` instances.

        Args:
            data: Full dataset used for portfolio reconstruction.
            results_df: Walk-forward results from ``run()``.
            regime_labels: DatetimeIndex Series of integer regime IDs.
            regime_names: Optional mapping from regime ID to display name.
            annualization_factor: Bars per year (252 daily, 8760 hourly, etc.).
            n: Number of top parameter combinations to analyze.

        Returns:
            A single ``RegimePerformanceAnalyzer`` when ``n == 1``, otherwise a list.
        """
        from .regime_performance_analyzer import RegimePerformanceAnalyzer

        portfolios = self.get_best_portfolio(data, results_df, n=n)
        if n == 1:
            return RegimePerformanceAnalyzer(
                portfolios, regime_labels, regime_names, annualization_factor
            )
        return [
            RegimePerformanceAnalyzer(pf, regime_labels, regime_names, annualization_factor)
            for pf in portfolios
        ]
