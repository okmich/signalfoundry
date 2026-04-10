import json
import shutil
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.backtesting.vectorbt_walkforward_backtester import (
    VectobtWalkForwardBacktester,
    WindowConfig,
    EvaluationResult,
    PerformanceMetrics,
    BacktesterConfig,
    ValidationError,
    ParameterEvaluator,
    ResultFilter,
    CheckpointManager,
    compute_window_hash,
    process_window_worker,
    atomic_write,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    data = pd.DataFrame(
        {
            "close": np.random.randn(500).cumsum() + 100,
            "open": np.random.randn(500).cumsum() + 100,
            "high": np.random.randn(500).cumsum() + 105,
            "low": np.random.randn(500).cumsum() + 95,
            "volume": np.random.randint(100, 1000, 500),
        },
        index=dates,
    )
    return data


@pytest.fixture
def basic_param_grid():
    """Basic parameter grid for testing"""
    return {"fast_period": [10, 20], "slow_period": [30, 50]}


@pytest.fixture
def basic_vbt_config():
    """Basic vectorbt configuration"""
    return {"init_cash": 10000, "fees": 0.001, "slippage": 0.001}


@pytest.fixture
def rolling_window_config():
    """Rolling window configuration"""
    return {"type": "rolling", "train_size": 100, "test_size": 50, "step_size": 50}


@pytest.fixture
def expanding_window_config():
    """Expanding window configuration"""
    return {
        "type": "expanding",
        "min_train_size": 100,
        "test_size": 50,
        "step_size": 50,
    }


@pytest.fixture
def basic_criteria():
    """Basic filtering criteria"""
    return {
        "primary_metric": "Sharpe Ratio",
        "higher_is_better": True,
        "filters": {"Total Return [%]": 0, "Max Drawdown [%]": (-50, 0)},
    }


@pytest.fixture
def temp_results_dir(tmp_path):
    """Temporary directory for results"""
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    yield results_dir
    # Cleanup
    if results_dir.exists():
        shutil.rmtree(results_dir)


class MockSignalGenerator:
    """Mock signal generator for testing"""

    def __init__(self, **params):
        self.params = params

    def generate(
        self, data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate dummy signals"""
        n = len(data)
        long_entries = pd.Series([False] * n, index=data.index)
        long_exits = pd.Series([False] * n, index=data.index)
        short_entries = pd.Series([False] * n, index=data.index)
        short_exits = pd.Series([False] * n, index=data.index)

        # Create some simple signals
        if n > 10:
            long_entries.iloc[5] = True
            long_exits.iloc[15] = True if n > 15 else False

        return long_entries, long_exits, short_entries, short_exits


# ============================================================================
# Data Classes Tests
# ============================================================================


class TestWindowConfig:
    """Test WindowConfig dataclass"""

    def test_rolling_window_config_creation(self):
        """Test creation of rolling window config"""
        config = WindowConfig(
            type="rolling", train_size=100, test_size=50, step_size=25
        )
        assert config.type == "rolling"
        assert config.train_size == 100
        assert config.test_size == 50
        assert config.step_size == 25

    def test_expanding_window_config_creation(self):
        """Test creation of expanding window config"""
        config = WindowConfig(type="expanding", min_train_size=100, test_size=50)
        assert config.type == "expanding"
        assert config.min_train_size == 100
        assert config.test_size == 50


class TestEvaluationResult:
    """Test EvaluationResult dataclass"""

    def test_successful_evaluation_result(self):
        """Test creation of successful evaluation result"""
        result = EvaluationResult(
            params={"fast": 10, "slow": 20},
            stats={"Sharpe Ratio": 1.5, "Return": 0.1},
            params_hash="abc123",
            evaluation_time=1.5,
        )
        assert result.params == {"fast": 10, "slow": 20}
        assert result.stats["Sharpe Ratio"] == 1.5
        assert result.error is None
        assert result.evaluation_time == 1.5

    def test_failed_evaluation_result(self):
        """Test creation of failed evaluation result"""
        result = EvaluationResult(
            params={"fast": 10},
            stats={},
            params_hash="abc123",
            error="Division by zero",
            evaluation_time=0.1,
        )
        assert result.error == "Division by zero"
        assert result.stats == {}


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_metrics_initialization(self):
        """Test performance metrics initialization"""
        metrics = PerformanceMetrics()
        assert metrics.total_windows == 0
        assert metrics.completed_windows == 0
        assert metrics.failed_windows == 0
        assert metrics.window_times == []

    def test_metrics_accumulation(self):
        """Test accumulating metrics"""
        metrics = PerformanceMetrics()
        metrics.total_windows = 10
        metrics.completed_windows = 8
        metrics.failed_windows = 2
        metrics.window_times = [1.0, 2.0, 3.0]

        assert metrics.total_windows == 10
        assert metrics.completed_windows == 8
        assert metrics.failed_windows == 2
        assert len(metrics.window_times) == 3


class TestBacktesterConfig:
    """Test BacktesterConfig dataclass"""

    def test_config_creation(
        self, basic_param_grid, rolling_window_config, basic_vbt_config, basic_criteria
    ):
        """Test backtester configuration creation"""
        config = BacktesterConfig(
            param_grid=basic_param_grid,
            window_config=rolling_window_config,
            vbt_config=basic_vbt_config,
            criteria=basic_criteria,
            n_top_params=5,
            n_workers=4,
            results_dir="./results",
        )

        assert config.param_grid == basic_param_grid
        assert config.n_top_params == 5
        assert config.n_workers == 4
        assert config.results_dir == "./results"

    def test_config_hash_computation(
        self, basic_param_grid, rolling_window_config, basic_vbt_config, basic_criteria
    ):
        """Test configuration hash computation"""
        config = BacktesterConfig(
            param_grid=basic_param_grid,
            window_config=rolling_window_config,
            vbt_config=basic_vbt_config,
            criteria=basic_criteria,
            n_top_params=5,
            n_workers=4,
            results_dir="./results",
        )

        hash1 = config.compute_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex digest

        # Same config should produce same hash
        hash2 = config.compute_hash()
        assert hash1 == hash2

    def test_config_hash_changes_with_params(
        self, basic_param_grid, rolling_window_config, basic_vbt_config, basic_criteria
    ):
        """Test that hash changes when configuration changes"""
        config1 = BacktesterConfig(
            param_grid=basic_param_grid,
            window_config=rolling_window_config,
            vbt_config=basic_vbt_config,
            criteria=basic_criteria,
            n_top_params=5,
            n_workers=4,
            results_dir="./results",
        )

        # Change parameter grid
        modified_grid = basic_param_grid.copy()
        modified_grid["fast_period"] = [15, 25]

        config2 = BacktesterConfig(
            param_grid=modified_grid,
            window_config=rolling_window_config,
            vbt_config=basic_vbt_config,
            criteria=basic_criteria,
            n_top_params=5,
            n_workers=4,
            results_dir="./results",
        )

        assert config1.compute_hash() != config2.compute_hash()


# ============================================================================
# Utility Functions Tests
# ============================================================================


class TestAtomicWrite:
    """Test atomic file writing"""

    def test_atomic_write_success(self, tmp_path):
        """Test successful atomic write"""
        file_path = tmp_path / "test.txt"
        content = "test content"

        atomic_write(file_path, content)

        assert file_path.exists()
        assert file_path.read_text() == content

    def test_atomic_write_no_temp_file_left(self, tmp_path):
        """Test that temp file is cleaned up"""
        file_path = tmp_path / "test.txt"
        content = "test content"

        atomic_write(file_path, content)

        # Check no temp files left
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0


class TestComputeWindowHash:
    """Test window hash computation"""

    def test_hash_consistency(self):
        """Test that same inputs produce same hash"""
        train_range = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
        test_range = (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-12-31"))
        param_hash = "abc123"

        hash1 = compute_window_hash(train_range, test_range, param_hash)
        hash2 = compute_window_hash(train_range, test_range, param_hash)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64

    def test_hash_changes_with_ranges(self):
        """Test that different ranges produce different hashes"""
        train_range1 = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
        train_range2 = (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-07-30"))
        test_range = (pd.Timestamp("2020-07-01"), pd.Timestamp("2020-12-31"))
        param_hash = "abc123"

        hash1 = compute_window_hash(train_range1, test_range, param_hash)
        hash2 = compute_window_hash(train_range2, test_range, param_hash)

        assert hash1 != hash2


# ============================================================================
# ParameterEvaluator Tests
# ============================================================================


class TestParameterEvaluator:
    """Test ParameterEvaluator class"""

    def test_params_hash_computation(self):
        """Test parameter hash computation"""
        params = {"fast": 10, "slow": 20}
        hash1 = ParameterEvaluator.compute_params_hash(params)
        hash2 = ParameterEvaluator.compute_params_hash(params)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64

    def test_params_hash_caching(self):
        """Test that hash computation uses cache"""
        params = {"fast": 10, "slow": 20}

        # Clear cache
        ParameterEvaluator._hash_cache.clear()

        hash1 = ParameterEvaluator.compute_params_hash(params)
        assert len(ParameterEvaluator._hash_cache) == 1

        hash2 = ParameterEvaluator.compute_params_hash(params)
        assert hash1 == hash2
        assert len(ParameterEvaluator._hash_cache) == 1  # Still only one entry

    @patch("vectorbt.Portfolio.from_signals")
    def test_successful_evaluation(self, mock_portfolio, sample_data, basic_vbt_config):
        """Test successful parameter evaluation"""
        # Mock portfolio stats
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {"Sharpe Ratio": 1.5, "Return": 0.1}
        mock_portfolio.return_value.stats.return_value = mock_stats

        evaluator = ParameterEvaluator(MockSignalGenerator, basic_vbt_config)
        params = {"fast_period": 10, "slow_period": 30}

        result = evaluator.evaluate(params, sample_data)

        assert result.error is None
        assert result.stats["Sharpe Ratio"] == 1.5
        assert result.params == params
        assert result.evaluation_time >= 0

    @patch("vectorbt.Portfolio.from_signals")
    def test_evaluation_with_keyerror(
        self, mock_portfolio, sample_data, basic_vbt_config
    ):
        """Test evaluation handles KeyError (missing column)"""
        evaluator = ParameterEvaluator(MockSignalGenerator, basic_vbt_config)
        params = {"fast_period": 10}

        # Create data without 'close' column
        bad_data = pd.DataFrame({"open": [1, 2, 3]})

        result = evaluator.evaluate(params, bad_data)

        assert result.error is not None
        assert "Missing required column" in result.error
        assert result.stats == {}

    @patch("vectorbt.Portfolio.from_signals")
    def test_evaluation_with_retry(self, mock_portfolio, sample_data, basic_vbt_config):
        """Test evaluation retry logic"""
        # First call raises exception, second succeeds
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {"Sharpe Ratio": 1.5}

        mock_portfolio.side_effect = [
            Exception("Temporary error"),
            Mock(stats=Mock(return_value=mock_stats)),
        ]

        evaluator = ParameterEvaluator(MockSignalGenerator, basic_vbt_config)
        params = {"fast_period": 10}

        result = evaluator.evaluate(params, sample_data, max_retries=3)

        # Should succeed after retry
        assert result.error is None or result.stats != {}


# ============================================================================
# ResultFilter Tests
# ============================================================================


class TestResultFilter:
    """Test ResultFilter class"""

    def test_valid_criteria(self):
        """Test filter initialization with valid criteria"""
        criteria = {
            "primary_metric": "Sharpe Ratio",
            "higher_is_better": True,
            "filters": {"Return": 0.05, "Max Drawdown": (-0.2, 0)},
        }

        filter_obj = ResultFilter(criteria)
        assert filter_obj.criteria == criteria

    def test_missing_primary_metric(self):
        """Test that missing primary_metric raises error"""
        criteria = {"filters": {"Return": 0.05}}

        with pytest.raises(ValidationError, match="must contain 'primary_metric'"):
            ResultFilter(criteria)

    def test_invalid_tuple_filter(self):
        """Test that invalid tuple filter raises error"""
        criteria = {
            "primary_metric": "Sharpe Ratio",
            "filters": {"Return": (0.05,)},  # Only one element
        }

        with pytest.raises(ValidationError, match="must have exactly 2 elements"):
            ResultFilter(criteria)

    def test_invalid_range_filter(self):
        """Test that invalid range raises error"""
        criteria = {
            "primary_metric": "Sharpe Ratio",
            "filters": {"Return": (0.2, 0.1)},  # min > max
        }

        with pytest.raises(ValidationError, match="Invalid range"):
            ResultFilter(criteria)

    def test_filter_and_rank(self):
        """Test filtering and ranking results"""
        criteria = {
            "primary_metric": "Sharpe Ratio",
            "higher_is_better": True,
            "filters": {"Return": 0.05},
        }

        results = [
            EvaluationResult(
                params={"fast": 10},
                stats={"Sharpe Ratio": 1.5, "Return": 0.1},
                params_hash="hash1",
                evaluation_time=1.0,
            ),
            EvaluationResult(
                params={"fast": 20},
                stats={"Sharpe Ratio": 1.2, "Return": 0.08},
                params_hash="hash2",
                evaluation_time=1.0,
            ),
            EvaluationResult(
                params={"fast": 30},
                stats={"Sharpe Ratio": 0.8, "Return": 0.02},  # Fails filter
                params_hash="hash3",
                evaluation_time=1.0,
            ),
        ]

        filter_obj = ResultFilter(criteria)
        top_results = filter_obj.filter_and_rank(results, n_top=2)

        # Should return 2 results (third fails filter)
        assert len(top_results) == 2
        # Should be sorted by Sharpe Ratio (descending)
        assert top_results[0].stats["Sharpe Ratio"] == 1.5
        assert top_results[1].stats["Sharpe Ratio"] == 1.2

    def test_range_filter(self):
        """Test range-based filtering"""
        criteria = {
            "primary_metric": "Sharpe Ratio",
            "higher_is_better": True,
            "filters": {"Max Drawdown": (-0.2, -0.05)},
        }

        results = [
            EvaluationResult(
                params={"fast": 10},
                stats={"Sharpe Ratio": 1.5, "Max Drawdown": -0.1},  # Passes
                params_hash="hash1",
                evaluation_time=1.0,
            ),
            EvaluationResult(
                params={"fast": 20},
                stats={"Sharpe Ratio": 1.2, "Max Drawdown": -0.3},  # Fails (too low)
                params_hash="hash2",
                evaluation_time=1.0,
            ),
        ]

        filter_obj = ResultFilter(criteria)
        top_results = filter_obj.filter_and_rank(results, n_top=5)

        assert len(top_results) == 1
        assert top_results[0].params["fast"] == 10


# ============================================================================
# CheckpointManager Tests
# ============================================================================


class TestCheckpointManager:
    """Test CheckpointManager class"""

    def test_checkpoint_creation(self, tmp_path):
        """Test checkpoint directory creation"""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        assert checkpoint_dir.exists()
        assert manager.checkpoint_file == checkpoint_dir / "progress.json"

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoint"""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        completed_ids = ["window1", "window2", "window3"]
        config_hash = "abc123"

        manager.save_checkpoint(completed_ids, total_windows=5, config_hash=config_hash)

        # Load checkpoint
        result = manager.load_checkpoint(config_hash)
        assert result is not None

        loaded_ids, loaded_metrics = result
        assert loaded_ids == completed_ids

    def test_checkpoint_config_validation(self, tmp_path):
        """Test that checkpoint validates config hash"""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        # Save with one hash
        manager.save_checkpoint(["window1"], total_windows=5, config_hash="hash1")

        # Try to load with different hash
        result = manager.load_checkpoint("different_hash")
        assert result is None

    def test_checkpoint_with_performance_metrics(self, tmp_path):
        """Test checkpoint with performance metrics"""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        metrics = PerformanceMetrics(
            total_windows=10,
            completed_windows=5,
            failed_windows=1,
            window_times=[1.0, 2.0, 3.0],
        )

        manager.save_checkpoint(
            ["window1", "window2"],
            total_windows=10,
            config_hash="abc123",
            performance_metrics=metrics,
        )

        loaded_ids, loaded_metrics = manager.load_checkpoint("abc123")

        assert loaded_metrics is not None
        assert loaded_metrics.total_windows == 10
        assert loaded_metrics.completed_windows == 5
        assert loaded_metrics.failed_windows == 1

    def test_clear_checkpoint(self, tmp_path):
        """Test checkpoint clearing"""
        checkpoint_dir = tmp_path / "checkpoints"
        manager = CheckpointManager(checkpoint_dir)

        manager.save_checkpoint(["window1"], total_windows=5, config_hash="abc")
        assert manager.checkpoint_file.exists()

        manager.clear_checkpoint()
        assert not manager.checkpoint_file.exists()


# ============================================================================
# VectobtWalkForwardBacktester Tests
# ============================================================================


class TestVectobtWalkForwardBacktester:
    """Test VectobtWalkForwardBacktester class"""

    def test_initialization(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test backtester initialization"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
            n_top_params=3,
            n_workers=2,
        )

        assert backtester.signal_class == MockSignalGenerator
        assert backtester.n_top_params == 3
        assert backtester.n_workers == 2
        assert backtester.results_dir == temp_results_dir

        # Check config file was created
        assert (temp_results_dir / "config.json").exists()

    def test_param_combinations_generation(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test parameter combinations generation"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Should generate 4 combinations (2 fast_period × 2 slow_period)
        assert len(backtester.param_combinations) == 4

        # Verify all combinations exist
        param_tuples = [
            (combo["fast_period"], combo["slow_period"])
            for combo in backtester.param_combinations
        ]
        assert (10, 30) in param_tuples
        assert (10, 50) in param_tuples
        assert (20, 30) in param_tuples
        assert (20, 50) in param_tuples

    def test_invalid_signal_class(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test that non-callable signal class raises error"""
        with pytest.raises(ValidationError, match="signal_class must be callable"):
            VectobtWalkForwardBacktester(
                signal_class="not_callable",
                param_grid=basic_param_grid,
                vbt_config=basic_vbt_config,
                window_config=rolling_window_config,
                criteria=basic_criteria,
                results_dir=str(temp_results_dir),
            )

    def test_empty_param_grid(
        self, rolling_window_config, basic_vbt_config, basic_criteria, temp_results_dir
    ):
        """Test that empty param grid raises error"""
        with pytest.raises(ValidationError, match="param_grid cannot be empty"):
            VectobtWalkForwardBacktester(
                signal_class=MockSignalGenerator,
                param_grid={},
                vbt_config=basic_vbt_config,
                window_config=rolling_window_config,
                criteria=basic_criteria,
                results_dir=str(temp_results_dir),
            )

    def test_invalid_window_type(
        self, basic_param_grid, basic_vbt_config, basic_criteria, temp_results_dir
    ):
        """Test that invalid window type raises error"""
        invalid_config = {"type": "invalid_type"}

        with pytest.raises(ValidationError, match="must be 'rolling' or 'expanding'"):
            VectobtWalkForwardBacktester(
                signal_class=MockSignalGenerator,
                param_grid=basic_param_grid,
                vbt_config=basic_vbt_config,
                window_config=invalid_config,
                criteria=basic_criteria,
                results_dir=str(temp_results_dir),
            )

    def test_rolling_windows_creation(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
        sample_data,
    ):
        """Test rolling windows creation"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        windows = backtester._create_windows(sample_data)

        assert len(windows) > 0

        for window in windows:
            assert "train" in window
            assert "test" in window
            assert "train_range" in window
            assert "test_range" in window

            # Verify sizes
            assert len(window["train"]) == rolling_window_config["train_size"]
            assert len(window["test"]) == rolling_window_config["test_size"]

            # Verify train comes before test
            assert window["train"].index[-1] < window["test"].index[0]

    def test_expanding_windows_creation(
        self,
        basic_param_grid,
        expanding_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
        sample_data,
    ):
        """Test expanding windows creation"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=expanding_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        windows = backtester._create_windows(sample_data)

        assert len(windows) > 0

        # Verify train size increases
        train_sizes = [len(window["train"]) for window in windows]
        assert all(
            train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1)
        )

        # Verify test size is constant
        test_sizes = [len(window["test"]) for window in windows]
        assert all(size == expanding_window_config["test_size"] for size in test_sizes)

    def test_data_validation_missing_close(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test data validation catches missing 'close' column"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Data without 'close' column
        bad_data = pd.DataFrame({"open": [1, 2, 3]})

        with pytest.raises(ValidationError, match="missing required columns"):
            backtester._validate_data(bad_data)

    def test_data_validation_insufficient_data(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test data validation catches insufficient data"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Data too short for windows
        short_data = pd.DataFrame(
            {"close": [1, 2, 3]}, index=pd.date_range("2020-01-01", periods=3)
        )

        with pytest.raises(ValidationError, match="require at least"):
            backtester._validate_data(short_data)

    def test_stream_format_results(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test results formatting"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Mock window results
        all_results = [
            {
                "window_id": "window1",
                "train_range": ["2020-01-01", "2020-04-10"],
                "test_range": ["2020-04-11", "2020-05-30"],
                "train_results": [
                    {
                        "params": {"fast_period": 10, "slow_period": 30},
                        "stats": {"Sharpe Ratio": 1.5, "Return": 0.1},
                        "params_hash": "hash1",
                    }
                ],
                "test_results": [
                    {
                        "params": {"fast_period": 10, "slow_period": 30},
                        "stats": {"Sharpe Ratio": 1.2, "Return": 0.08},
                        "params_hash": "hash1",
                    }
                ],
            }
        ]

        results_df = backtester._stream_format_results(all_results)

        assert len(results_df) == 1
        assert "window_id" in results_df.columns
        assert "params_hash" in results_df.columns
        assert "param_fast_period" in results_df.columns
        assert "train_Sharpe Ratio" in results_df.columns
        assert "test_Sharpe Ratio" in results_df.columns

        assert results_df.iloc[0]["param_fast_period"] == 10
        assert results_df.iloc[0]["train_Sharpe Ratio"] == 1.5
        assert results_df.iloc[0]["test_Sharpe Ratio"] == 1.2

    def test_analyze_results(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test results analysis"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Create mock results dataframe
        results_df = pd.DataFrame(
            {
                "param_fast_period": [10, 10, 20],
                "param_slow_period": [30, 30, 50],
                "params_hash": ["hash1", "hash1", "hash2"],
                "test_Sharpe Ratio": [1.5, 1.6, 1.2],
            }
        )

        analysis = backtester.analyze_results(results_df)

        assert len(analysis) == 2  # Two unique parameter combinations
        assert "test_Sharpe Ratio_mean" in analysis.columns
        assert "test_Sharpe Ratio_std" in analysis.columns

    def test_get_best_params(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test getting best parameters"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Create mock results dataframe
        results_df = pd.DataFrame(
            {
                "param_fast_period": [10, 10, 20, 20],
                "param_slow_period": [30, 30, 50, 50],
                "params_hash": ["hash1", "hash1", "hash2", "hash2"],
                "test_Sharpe Ratio": [1.5, 1.6, 1.2, 1.3],
            }
        )

        best_params = backtester.get_best_params(results_df, n=1)

        assert len(best_params) == 1
        assert best_params[0]["fast_period"] == 10
        assert best_params[0]["slow_period"] == 30

    def test_performance_summary(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test performance summary generation"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Set some metrics
        backtester.performance_metrics.total_windows = 10
        backtester.performance_metrics.completed_windows = 8
        backtester.performance_metrics.failed_windows = 2

        summary = backtester.get_performance_summary()

        assert summary["total_windows"] == 10
        assert summary["completed_windows"] == 8
        assert summary["failed_windows"] == 2
        assert summary["success_rate"] == 0.8

    def test_should_stop_early(
        self,
        basic_param_grid,
        rolling_window_config,
        basic_vbt_config,
        basic_criteria,
        temp_results_dir,
    ):
        """Test early stopping logic"""
        backtester = VectobtWalkForwardBacktester(
            signal_class=MockSignalGenerator,
            param_grid=basic_param_grid,
            vbt_config=basic_vbt_config,
            window_config=rolling_window_config,
            criteria=basic_criteria,
            results_dir=str(temp_results_dir),
        )

        # Not enough windows to stop early
        assert not backtester._should_stop_early(completed=2, failed=2)

        # High failure rate after minimum windows
        assert backtester._should_stop_early(completed=3, failed=4)

        # Low failure rate
        assert not backtester._should_stop_early(completed=8, failed=2)


# ============================================================================
# Integration Tests
# ============================================================================


class TestProcessWindowWorker:
    """Test process_window_worker function"""

    @patch("vectorbt.Portfolio.from_signals")
    def test_window_processing_success(self, mock_portfolio, tmp_path, sample_data):
        """Test successful window processing"""
        # Setup mock
        mock_stats = Mock()
        mock_stats.to_dict.return_value = {
            "Sharpe Ratio": 1.5,
            "Return": 0.1,
            "Total Return [%]": 10.0,
        }
        mock_portfolio.return_value.stats.return_value = mock_stats

        # Prepare arguments
        window_data = {
            "train": sample_data.iloc[:100],
            "test": sample_data.iloc[100:150],
            "train_range": (sample_data.index[0], sample_data.index[99]),
            "test_range": (sample_data.index[100], sample_data.index[149]),
        }

        param_combinations = [
            {"fast_period": 10, "slow_period": 30},
            {"fast_period": 20, "slow_period": 50},
        ]

        vbt_config = {"init_cash": 10000, "fees": 0.001}
        criteria = {
            "primary_metric": "Sharpe Ratio",
            "higher_is_better": True,
            "filters": {"Total Return [%]": 0},
        }

        param_hash = "test_hash"
        args = (
            0,
            window_data,
            MockSignalGenerator,
            param_combinations,
            vbt_config,
            criteria,
            1,
            str(tmp_path),
            param_hash,
        )

        # Execute
        window_hash, processing_time = process_window_worker(args)

        # Verify
        assert window_hash is not None
        assert processing_time >= 0

        # Check file was created
        result_file = tmp_path / f"window_{window_hash}.json"
        assert result_file.exists()

        # Verify content
        with open(result_file) as f:
            result_data = json.load(f)

        assert result_data["window_id"] == window_hash
        assert len(result_data["train_results"]) > 0

    def test_window_processing_skips_existing(self, tmp_path, sample_data):
        """Test that worker skips already processed windows"""
        window_data = {
            "train": sample_data.iloc[:100],
            "test": sample_data.iloc[100:150],
            "train_range": (sample_data.index[0], sample_data.index[99]),
            "test_range": (sample_data.index[100], sample_data.index[149]),
        }

        param_hash = "test_hash"
        window_hash = compute_window_hash(
            window_data["train_range"], window_data["test_range"], param_hash
        )

        # Create existing file
        result_file = tmp_path / f"window_{window_hash}.json"
        result_file.write_text('{"already": "processed"}')

        args = (
            0,
            window_data,
            MockSignalGenerator,
            [],
            {},
            {},
            1,
            str(tmp_path),
            param_hash,
        )

        returned_hash, processing_time = process_window_worker(args)

        # Should return without processing
        assert returned_hash == window_hash
        assert processing_time == 0.0

        # File should remain unchanged
        assert result_file.read_text() == '{"already": "processed"}'
