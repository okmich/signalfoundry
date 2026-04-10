"""
Tests for StrategyHealth tracking and metrics.
"""

import pytest
from datetime import datetime
from okmich_quant_core.health import StrategyHealth


class TestStrategyHealth:
    """Test StrategyHealth class functionality."""

    def test_initialization(self):
        """Test health tracker initialization."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=3)

        assert health.strategy_name == "TestStrategy"
        assert health.max_consecutive_errors == 3
        assert health.is_enabled is True
        assert health.consecutive_errors == 0
        assert health.total_errors == 0
        assert health.total_runs == 0
        assert health.successful_runs == 0
        assert health.success_rate == 0.0
        assert health.average_execution_time_ms == 0.0

    def test_record_success(self):
        """Test recording successful execution."""
        health = StrategyHealth("TestStrategy")

        health.record_success(100.0)

        assert health.total_runs == 1
        assert health.successful_runs == 1
        assert health.consecutive_errors == 0
        assert health.total_errors == 0
        assert health.success_rate == 100.0
        assert health.average_execution_time_ms == 100.0
        assert health.last_success_time is not None
        assert health.last_run_time is not None

    def test_record_error(self):
        """Test recording failed execution."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=5)

        health.record_error(50.0)

        assert health.total_runs == 1
        assert health.successful_runs == 0
        assert health.consecutive_errors == 1
        assert health.total_errors == 1
        assert health.success_rate == 0.0
        assert health.average_execution_time_ms == 50.0
        assert health.last_error_time is not None
        assert health.is_enabled is True  # Not disabled yet

    def test_consecutive_errors_reset_on_success(self):
        """Test that consecutive errors reset after success."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=3)

        health.record_error(10.0)
        health.record_error(10.0)
        assert health.consecutive_errors == 2

        health.record_success(10.0)
        assert health.consecutive_errors == 0
        assert health.total_errors == 2  # Still tracked

    def test_circuit_breaker_trips_after_max_errors(self):
        """Test circuit breaker disables strategy after max errors."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=3)

        health.record_error(10.0)
        assert health.is_enabled is True

        health.record_error(10.0)
        assert health.is_enabled is True

        health.record_error(10.0)
        # Circuit breaker should trip
        assert health.is_enabled is False
        assert health.consecutive_errors == 3

    def test_circuit_breaker_does_not_trip_on_success(self):
        """Test that success resets circuit breaker."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=3)

        health.record_error(10.0)
        health.record_error(10.0)
        health.record_success(10.0)  # Reset
        health.record_error(10.0)
        health.record_error(10.0)

        # Should still be enabled (only 2 consecutive)
        assert health.is_enabled is True
        assert health.consecutive_errors == 2

    def test_success_rate_calculation(self):
        """Test success rate percentage calculation."""
        health = StrategyHealth("TestStrategy")

        # 3 successes, 1 error = 75%
        health.record_success(10.0)
        health.record_success(10.0)
        health.record_success(10.0)
        health.record_error(10.0)

        assert health.success_rate == 75.0
        assert health.total_runs == 4

    def test_average_execution_time(self):
        """Test average execution time calculation."""
        health = StrategyHealth("TestStrategy")

        health.record_success(100.0)
        health.record_success(200.0)
        health.record_error(300.0)

        # Average: (100 + 200 + 300) / 3 = 200
        assert health.average_execution_time_ms == 200.0

    def test_record_position_check(self):
        """Test position check recording."""
        health = StrategyHealth("TestStrategy")

        health.record_position_check()
        health.record_position_check()
        health.record_position_check()

        assert health.total_position_checks == 3

    def test_manual_enable(self):
        """Test manually re-enabling disabled strategy."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=2)

        # Disable via circuit breaker
        health.record_error(10.0)
        health.record_error(10.0)
        assert health.is_enabled is False

        # Manually re-enable
        health.enable()
        assert health.is_enabled is True
        assert health.consecutive_errors == 0  # Should reset

    def test_manual_disable(self):
        """Test manually disabling strategy."""
        health = StrategyHealth("TestStrategy")

        assert health.is_enabled is True

        health.disable()
        assert health.is_enabled is False

    def test_get_status_summary(self):
        """Test status summary generation."""
        health = StrategyHealth("TestStrategy", max_consecutive_errors=5)

        health.record_success(100.0)
        health.record_error(50.0)
        health.record_position_check()

        summary = health.get_status_summary()

        assert summary["strategy"] == "TestStrategy"
        assert summary["enabled"] is True
        assert summary["total_runs"] == 2
        assert summary["successful_runs"] == 1
        assert summary["total_errors"] == 1
        assert summary["consecutive_errors"] == 1
        assert summary["success_rate"] == "50.00%"
        assert summary["avg_execution_time_ms"] == "75.00"
        assert summary["total_position_checks"] == 1
        assert summary["last_run"] is not None
        assert summary["last_success"] is not None
        assert summary["last_error"] is not None

    def test_zero_division_safety(self):
        """Test that calculations handle zero runs safely."""
        health = StrategyHealth("TestStrategy")

        # No runs yet
        assert health.success_rate == 0.0
        assert health.average_execution_time_ms == 0.0

        summary = health.get_status_summary()
        assert summary["success_rate"] == "0.00%"
        assert summary["avg_execution_time_ms"] == "0.00"