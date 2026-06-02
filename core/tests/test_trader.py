"""
Tests for Trader class with single strategy.
"""

from datetime import datetime, timedelta

from okmich_quant_core.base_strategy import BaseStrategy
from okmich_quant_core.config import StrategyConfig
from okmich_quant_core.logging import BaseEventLogger, RunnerIdentity
from okmich_quant_core.signal import BaseSignal
from okmich_quant_core.trader import Trader


_bar_seq = [0]


def _bar_dt() -> datetime:
    """Monotonically increasing bar timestamps spaced 1 minute apart.

    BaseStrategy.run()'s sealed template carries a 1-second dup-guard; consecutive
    ``datetime.now()`` calls would be coalesced. Spaced timestamps simulate distinct bars.
    """
    _bar_seq[0] += 1
    return datetime(2026, 1, 1) + timedelta(minutes=_bar_seq[0])


class _RecordingLogger(BaseEventLogger):
    """In-memory inference-log sink (no thread, no disk) so dispatch tests stay fast + isolated."""

    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def drain(self, timeout=None):
        pass

    def close(self):
        pass


class MockStrategy(BaseStrategy):
    """Mock strategy for testing. Drives the real sealed template via on_new_bar()/is_new_bar()."""

    def __init__(self, name="MockStrategy"):
        super().__init__(StrategyConfig(name=name, symbol="EURUSD", timeframe=1, magic=1),
                         BaseSignal(), inference_logger=_RecordingLogger())
        self.bind_runner_identity(RunnerIdentity(runner_id="test-runner", runner_start_token="tok",
                                                 broker="test", account_id="0", broker_session_id=None))
        self.name = name
        self.run_count = 0
        self.position_check_count = 0
        self.should_fail = False
        self.cleanup_called = False

    def is_new_bar(self, run_dt: datetime) -> bool:
        return True

    def on_new_bar(self):
        self.run_count += 1
        if self.should_fail:
            raise ValueError(f"Intentional error in {self.name}")

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> int:
        self.position_check_count += 1
        # Raise only in the position-check path (flag=False); the new-bar path (flag=True) must
        # reach on_new_bar so the heartbeat + run-count reflect the cycle.
        if self.should_fail and not flag:
            raise ValueError(f"Position check error in {self.name}")
        return 0

    def cleanup(self):
        self.cleanup_called = True


class TestTrader:
    """Test Trader class with single strategy."""

    def test_initialization(self):
        """Test trader initialization."""
        strategy = MockStrategy("TestStrategy")
        trader = Trader(strategy, max_consecutive_errors=3)

        assert trader.strategy == strategy
        assert trader.health.strategy_name == "TestStrategy"
        assert trader.health.max_consecutive_errors == 3
        assert trader.health.is_enabled is True

    def test_successful_run(self):
        """Test successful strategy execution."""
        strategy = MockStrategy()
        trader = Trader(strategy)

        trader.run(_bar_dt())

        assert strategy.run_count == 1
        assert trader.health.total_runs == 1
        assert trader.health.successful_runs == 1
        assert trader.health.consecutive_errors == 0
        assert trader.health.is_enabled is True

    def test_failed_run(self):
        """Test strategy execution with error."""
        strategy = MockStrategy()
        strategy.should_fail = True
        trader = Trader(strategy, max_consecutive_errors=3)

        trader.run(_bar_dt())

        assert strategy.run_count == 1
        assert trader.health.total_runs == 1
        assert trader.health.successful_runs == 0
        assert trader.health.consecutive_errors == 1
        assert trader.health.total_errors == 1
        assert trader.health.is_enabled is True  # Not disabled yet

    def test_circuit_breaker_trips(self):
        """Test circuit breaker disables strategy after max errors."""
        strategy = MockStrategy()
        strategy.should_fail = True
        trader = Trader(strategy, max_consecutive_errors=3)

        trader.run(_bar_dt())
        trader.run(_bar_dt())
        trader.run(_bar_dt())

        assert trader.health.consecutive_errors == 3
        assert trader.health.is_enabled is False

        # Further runs should be skipped
        trader.run(_bar_dt())
        assert strategy.run_count == 3  # No additional run

    def test_circuit_breaker_resets_on_success(self):
        """Test that success resets consecutive error count."""
        strategy = MockStrategy()
        trader = Trader(strategy, max_consecutive_errors=3)

        strategy.should_fail = True
        trader.run(_bar_dt())
        trader.run(_bar_dt())
        assert trader.health.consecutive_errors == 2

        strategy.should_fail = False
        trader.run(_bar_dt())
        assert trader.health.consecutive_errors == 0
        assert trader.health.is_enabled is True

    def test_successful_position_check(self):
        """Test successful position check."""
        strategy = MockStrategy()
        trader = Trader(strategy)

        trader.check_positions(datetime.now())

        assert strategy.position_check_count == 1
        assert trader.health.total_position_checks == 1

    def test_failed_position_check(self):
        """Test position check with error (doesn't trip circuit breaker)."""
        strategy = MockStrategy()
        strategy.should_fail = True
        trader = Trader(strategy, max_consecutive_errors=2)

        # Position check errors don't count toward circuit breaker
        trader.check_positions(datetime.now())
        trader.check_positions(datetime.now())
        trader.check_positions(datetime.now())

        assert trader.health.is_enabled is True  # Still enabled
        assert trader.health.consecutive_errors == 0

    def test_disabled_strategy_skips_execution(self):
        """Test that disabled strategy doesn't execute."""
        strategy = MockStrategy()
        trader = Trader(strategy)

        trader.disable()
        trader.run(_bar_dt())

        assert strategy.run_count == 0
        assert trader.health.total_runs == 0

    def test_manual_enable_disable(self):
        """Test manual enable/disable."""
        strategy = MockStrategy()
        trader = Trader(strategy)

        trader.disable()
        assert trader.health.is_enabled is False

        trader.enable()
        assert trader.health.is_enabled is True

    def test_get_health_status(self):
        """Test health status retrieval."""
        strategy = MockStrategy("TestStrat")
        trader = Trader(strategy)

        trader.run(_bar_dt())

        status = trader.get_health_status()
        assert status["strategy"] == "TestStrat"
        assert status["total_runs"] == 1
        assert status["successful_runs"] == 1
        assert status["enabled"] is True

    def test_close_calls_cleanup(self):
        """Test that close() calls strategy cleanup."""
        strategy = MockStrategy()
        trader = Trader(strategy)

        trader.run(_bar_dt())
        trader.close()

        assert strategy.cleanup_called is True

    def test_close_handles_cleanup_error(self):
        """Test that close() handles cleanup errors gracefully."""
        strategy = MockStrategy()

        def failing_cleanup():
            raise RuntimeError("Cleanup failed")

        strategy.cleanup = failing_cleanup
        trader = Trader(strategy)

        # Should not raise exception
        trader.close()
