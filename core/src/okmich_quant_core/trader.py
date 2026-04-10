import logging
import traceback
from datetime import datetime
from time import time

from .base_strategy import BaseStrategy
from .health import StrategyHealth

logger = logging.getLogger(__name__)


class Trader:
    """
    Orchestrates a single trading strategy with error isolation, health tracking, and performance monitoring.
    """

    def __init__(self, strategy: BaseStrategy, max_consecutive_errors: int = 5):

        self.strategy = strategy
        strategy_name = strategy.strategy_config.name

        self.health = StrategyHealth(
            strategy_name=strategy_name,
            max_consecutive_errors=max_consecutive_errors
        )
        logger.info(
            f"Trader initialized for strategy: {strategy_name} "
            f"(circuit breaker: {max_consecutive_errors} errors)")

    def check_positions(self, run_dt: datetime):
        if not self.health.is_enabled:
            logger.debug(f"Strategy '{self.health.strategy_name}' is disabled, skipping position check")
            return

        try:
            self.strategy.manage_positions(run_dt, False)
            self.health.record_position_check()
        except Exception as e:
            logger.error(f"Error in position check for '{self.health.strategy_name}': {e}", exc_info=True)

    def run(self, run_dt: datetime):
        if not self.health.is_enabled:
            logger.warning( f"Strategy '{self.health.strategy_name}' is disabled, skipping execution")
            return

        start_time = time()
        try:
            self.strategy.run(run_dt)
            execution_time_ms = (time() - start_time) * 1000
            self.health.record_success(execution_time_ms)

            logger.debug(
                f"Strategy '{self.health.strategy_name}' executed successfully "
                f"in {execution_time_ms:.2f}ms")
        except Exception as e:
            execution_time_ms = (time() - start_time) * 1000
            self.health.record_error(execution_time_ms)

            logger.error(
                f"Error in strategy '{self.health.strategy_name}' "
                f"(error #{self.health.consecutive_errors}/{self.health.max_consecutive_errors}): {e}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")

            # Notify via strategy notifier (parity with MultiTrader behaviour)
            if self.strategy.notifier:
                self.strategy.notifier.on_error(self.health.strategy_name, str(e))

            if not self.health.is_enabled:
                logger.critical(
                    f"Circuit breaker tripped for '{self.health.strategy_name}' - "
                    f"strategy is now DISABLED"
                )
                if self.strategy.notifier:
                    self.strategy.notifier.on_circuit_breaker_tripped(
                        self.health.strategy_name, self.health.consecutive_errors
                    )

    def get_health_status(self) -> dict:
        return self.health.get_status_summary()

    def enable(self):
        self.health.enable()

    def disable(self):
        self.health.disable()

    def close(self):
        logger.info(
            f"Closing trader for strategy '{self.health.strategy_name}' - "
            f"Final stats: {self.health.total_runs} runs, {self.health.successful_runs} successful, "
            f"{self.health.total_errors} errors, success rate: {self.health.success_rate:.2f}%"
        )

        # Call strategy-specific cleanup if available
        if hasattr(self.strategy, "cleanup"):
            try:
                self.strategy.cleanup()
                logger.info(f"Strategy '{self.health.strategy_name}' cleanup completed")
            except Exception as e:
                logger.error(f"Error during strategy cleanup for '{self.health.strategy_name}': {e}")
