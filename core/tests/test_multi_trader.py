"""
Tests for MultiTrader class with multiple strategies.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock
from okmich_quant_core.multi_trader import MultiTrader
from okmich_quant_core.base_strategy import BaseStrategy


class MockStrategyConfig:
    """Mock strategy config for testing."""
    def __init__(self, name):
        self.name = name


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, name="MockStrategy"):
        self.strategy_config = MockStrategyConfig(name)
        self.notifier = None
        self.run_count = 0
        self.position_check_count = 0
        self.should_fail = False
        self.cleanup_called = False

    def is_new_bar(self, run_dt: datetime) -> bool:
        """Mock implementation - always returns True for testing."""
        return True

    def on_new_bar(self):
        pass

    def run(self, run_dt: datetime):
        self.run_count += 1
        if self.should_fail:
            raise ValueError(f"Intentional error in {self.name}")

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> int:
        self.position_check_count += 1
        if self.should_fail:
            raise ValueError(f"Position check error in {self.name}")
        return 0

    def cleanup(self):
        self.cleanup_called = True


class TestMultiTrader:
    """Test MultiTrader class with multiple strategies."""

    def test_initialization(self):
        """Test multi-trader initialization."""
        strategies = [MockStrategy("Strat1"), MockStrategy("Strat2")]
        multi_trader = MultiTrader(strategies, max_consecutive_errors=3)

        assert len(multi_trader.strategies) == 2
        assert len(multi_trader.health_trackers) == 2
        assert "Strat1" in multi_trader.health_trackers
        assert "Strat2" in multi_trader.health_trackers

    def test_all_strategies_run_successfully(self):
        """Test all strategies execute successfully."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        multi_trader = MultiTrader([strat1, strat2])

        multi_trader.run(datetime.now())

        assert strat1.run_count == 1
        assert strat2.run_count == 1
        assert multi_trader.health_trackers["Strat1"].successful_runs == 1
        assert multi_trader.health_trackers["Strat2"].successful_runs == 1

    def test_error_isolation_between_strategies(self):
        """Test that one strategy's error doesn't affect others."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        strat1.should_fail = True  # Only first fails

        multi_trader = MultiTrader([strat1, strat2])
        multi_trader.run(datetime.now())

        # Strat1 failed, Strat2 succeeded
        assert strat1.run_count == 1
        assert strat2.run_count == 1
        assert multi_trader.health_trackers["Strat1"].total_errors == 1
        assert multi_trader.health_trackers["Strat2"].successful_runs == 1

    def test_individual_circuit_breakers(self):
        """Test that each strategy has independent circuit breaker."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        strat1.should_fail = True

        multi_trader = MultiTrader([strat1, strat2], max_consecutive_errors=2)

        multi_trader.run(datetime.now())
        multi_trader.run(datetime.now())

        # Strat1 should be disabled, Strat2 still enabled
        assert multi_trader.health_trackers["Strat1"].is_enabled is False
        assert multi_trader.health_trackers["Strat2"].is_enabled is True

        # Next run: only Strat2 executes
        multi_trader.run(datetime.now())
        assert strat1.run_count == 2  # Not incremented
        assert strat2.run_count == 3  # Still running

    def test_check_positions_all_strategies(self):
        """Test position checks for all strategies."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        multi_trader = MultiTrader([strat1, strat2])

        multi_trader.check_positions(datetime.now())

        assert strat1.position_check_count == 1
        assert strat2.position_check_count == 1

    def test_get_health_status(self):
        """Test retrieving health status for all strategies."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        multi_trader = MultiTrader([strat1, strat2])

        multi_trader.run(datetime.now())

        status = multi_trader.get_health_status()
        assert "Strat1" in status
        assert "Strat2" in status
        assert status["Strat1"]["total_runs"] == 1
        assert status["Strat2"]["total_runs"] == 1

    def test_get_aggregate_stats(self):
        """Test aggregate statistics calculation."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        strat1.should_fail = True

        multi_trader = MultiTrader([strat1, strat2])
        multi_trader.run(datetime.now())

        agg_stats = multi_trader.get_aggregate_stats()
        assert agg_stats["total_strategies"] == 2
        assert agg_stats["enabled_strategies"] == 2
        assert agg_stats["disabled_strategies"] == 0
        assert agg_stats["total_runs"] == 2
        assert agg_stats["total_successes"] == 1
        assert agg_stats["total_errors"] == 1
        assert agg_stats["overall_success_rate"] == "50.00%"

    def test_enable_disable_specific_strategy(self):
        """Test enabling/disabling specific strategies."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        multi_trader = MultiTrader([strat1, strat2])

        multi_trader.disable_strategy("Strat1")
        assert multi_trader.health_trackers["Strat1"].is_enabled is False
        assert multi_trader.health_trackers["Strat2"].is_enabled is True

        multi_trader.enable_strategy("Strat1")
        assert multi_trader.health_trackers["Strat1"].is_enabled is True

    def test_enable_disable_all(self):
        """Test enabling/disabling all strategies."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        multi_trader = MultiTrader([strat1, strat2])

        multi_trader.disable_all()
        assert multi_trader.health_trackers["Strat1"].is_enabled is False
        assert multi_trader.health_trackers["Strat2"].is_enabled is False

        multi_trader.enable_all()
        assert multi_trader.health_trackers["Strat1"].is_enabled is True
        assert multi_trader.health_trackers["Strat2"].is_enabled is True

    def test_close_calls_all_cleanups(self):
        """Test that close() calls cleanup on all strategies."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        multi_trader = MultiTrader([strat1, strat2])

        multi_trader.run(datetime.now())
        multi_trader.close()

        assert strat1.cleanup_called is True
        assert strat2.cleanup_called is True

    def test_close_handles_individual_cleanup_errors(self):
        """Test that close() continues even if one cleanup fails."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")

        def failing_cleanup():
            raise RuntimeError("Cleanup failed")

        strat1.cleanup = failing_cleanup

        multi_trader = MultiTrader([strat1, strat2])
        multi_trader.close()

        # Strat2 cleanup should still be called
        assert strat2.cleanup_called is True

    def test_get_strategy_name_helper(self):
        """Test _get_strategy_name helper method."""
        strat = MockStrategy("TestName")
        multi_trader = MultiTrader([strat])

        strategy_name = multi_trader._get_strategy_name(strat)
        assert strategy_name == "TestName"

    def test_multiple_strategies_with_same_class(self):
        """Test that strategies of same class but different names are tracked separately."""
        strat1 = MockStrategy("Strategy_A")
        strat2 = MockStrategy("Strategy_B")
        strat3 = MockStrategy("Strategy_C")

        multi_trader = MultiTrader([strat1, strat2, strat3])

        assert len(multi_trader.health_trackers) == 3
        assert "Strategy_A" in multi_trader.health_trackers
        assert "Strategy_B" in multi_trader.health_trackers
        assert "Strategy_C" in multi_trader.health_trackers

    def test_error_in_one_doesnt_prevent_others_from_running(self):
        """Test that error in first strategy doesn't prevent subsequent strategies from running."""
        strat1 = MockStrategy("Strat1")
        strat2 = MockStrategy("Strat2")
        strat3 = MockStrategy("Strat3")

        strat1.should_fail = True

        multi_trader = MultiTrader([strat1, strat2, strat3])
        multi_trader.run(datetime.now())

        # All strategies should have attempted to run
        assert strat1.run_count == 1
        assert strat2.run_count == 1
        assert strat3.run_count == 1

        # Only strat1 should have error
        assert multi_trader.health_trackers["Strat1"].total_errors == 1
        assert multi_trader.health_trackers["Strat2"].total_errors == 0
        assert multi_trader.health_trackers["Strat3"].total_errors == 0

    def test_disable_nonexistent_strategy_logs_warning(self):
        """Test that disabling nonexistent strategy logs warning."""
        strat = MockStrategy("Strat1")
        multi_trader = MultiTrader([strat])

        # Should log warning but not crash
        multi_trader.disable_strategy("NonExistent")
        # If we get here without exception, test passes

    def test_enable_nonexistent_strategy_logs_warning(self):
        """Test that enabling nonexistent strategy logs warning."""
        strat = MockStrategy("Strat1")
        multi_trader = MultiTrader([strat])

        # Should log warning but not crash
        multi_trader.enable_strategy("NonExistent")
        # If we get here without exception, test passes