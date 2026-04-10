import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class StrategyHealth:
    """Tracks health and performance metrics for a strategy."""

    def __init__(self, strategy_name: str, max_consecutive_errors: int = 5):
        self.strategy_name = strategy_name
        self.max_consecutive_errors = max_consecutive_errors

        # Health status
        self.is_enabled = True
        self.consecutive_errors = 0
        self.total_errors = 0
        self.total_runs = 0
        self.total_position_checks = 0

        # Timing
        self.last_run_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.last_error_time: Optional[datetime] = None
        self.total_execution_time_ms = 0.0

        # Performance
        self.successful_runs = 0

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100

    @property
    def average_execution_time_ms(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_execution_time_ms / self.total_runs

    def record_success(self, execution_time_ms: float):
        self.consecutive_errors = 0
        self.successful_runs += 1
        self.total_runs += 1
        self.total_execution_time_ms += execution_time_ms
        self.last_run_time = datetime.now()
        self.last_success_time = datetime.now()

    def record_error(self, execution_time_ms: float):
        self.consecutive_errors += 1
        self.total_errors += 1
        self.total_runs += 1
        self.total_execution_time_ms += execution_time_ms
        self.last_run_time = datetime.now()
        self.last_error_time = datetime.now()

        # Circuit breaker: disable after too many consecutive errors
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.is_enabled = False
            logger.critical(
                f"Strategy '{self.strategy_name}' DISABLED after "
                f"{self.consecutive_errors} consecutive errors (circuit breaker tripped)"
            )

    def record_position_check(self):
        self.total_position_checks += 1

    def enable(self):
        if not self.is_enabled:
            self.is_enabled = True
            self.consecutive_errors = 0
            logger.info(f"Strategy '{self.strategy_name}' manually re-enabled")

    def disable(self):
        if self.is_enabled:
            self.is_enabled = False
            logger.warning(f"Strategy '{self.strategy_name}' manually disabled")

    def get_status_summary(self) -> dict:
        """
        Get summary of strategy health metrics.

        Returns:
            Dictionary with health status and performance metrics
        """
        return {
            "strategy": self.strategy_name,
            "enabled": self.is_enabled,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "total_errors": self.total_errors,
            "consecutive_errors": self.consecutive_errors,
            "success_rate": f"{self.success_rate:.2f}%",
            "avg_execution_time_ms": f"{self.average_execution_time_ms:.2f}",
            "total_position_checks": self.total_position_checks,
            "last_run": self.last_run_time.isoformat() if self.last_run_time else None,
            "last_success": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
            "last_error": (
                self.last_error_time.isoformat() if self.last_error_time else None
            ),
        }