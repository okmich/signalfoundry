import logging
import traceback
from datetime import datetime
from time import time
from typing import List, Dict

from .base_strategy import BaseStrategy
from .health import StrategyHealth

logger = logging.getLogger(__name__)


class MultiTrader:
    """
    Orchestrates multiple trading strategies with error isolation, health tracking, and performance monitoring per strategy.
    """

    def __init__(self, strategies: List[BaseStrategy], max_consecutive_errors: int = 5, *args, **kwargs):
        """
        Initialize multi-trader with multiple strategies.

        Args:
            strategies: List of trading strategies to execute
            max_consecutive_errors: Max consecutive errors before circuit breaker trips (per strategy)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.strategies = strategies
        self.args = args
        self.kwargs = kwargs

        # Enforce unique strategy identity before creating health trackers. Keying by name alone would silently
        # overwrite a tracker when 2 strategies share the same name (e.g. same signal on different symbols or magic
        # numbers).  Use (name, symbol, magic) as the uniqueness key.
        seen: Dict[tuple, BaseStrategy] = {}
        for strategy in strategies:
            cfg = strategy.strategy_config
            key = (cfg.name, cfg.symbol, cfg.magic)
            if key in seen:
                raise ValueError(
                    f"Duplicate strategy identity (name='{cfg.name}', symbol='{cfg.symbol}', "
                    f"magic={cfg.magic}). Each strategy registered with MultiTrader must have "
                    "a unique (name, symbol, magic) combination."
                )
            seen[key] = strategy

        # Create health tracker for each strategy, keyed by name.
        # Uniqueness above guarantees names are distinct within this MultiTrader.
        self.health_trackers: Dict[str, StrategyHealth] = {}
        for strategy in strategies:
            strategy_name = strategy.strategy_config.name
            self.health_trackers[strategy_name] = StrategyHealth(
                strategy_name=strategy_name,
                max_consecutive_errors=max_consecutive_errors
            )

        logger.info(
            f"MultiTrader initialized with {len(strategies)} strategies "
            f"(circuit breaker: {max_consecutive_errors} errors per strategy)"
        )

    def _get_strategy_name(self, strategy: BaseStrategy) -> str:
        """Extract strategy name from strategy config."""
        return strategy.strategy_config.name

    def check_positions(self, run_dt: datetime):
        for strategy in self.strategies:
            strategy_name = self._get_strategy_name(strategy)
            health = self.health_trackers[strategy_name]

            if not health.is_enabled:
                logger.debug(f"Strategy '{strategy_name}' is disabled, skipping position check")
                continue

            try:
                strategy.manage_positions(run_dt)
                health.record_position_check()
            except Exception as e:
                logger.error(f"Error in position check for '{strategy_name}': {e}", exc_info=True,)

    def run(self, run_dt: datetime):
        for strategy in self.strategies:
            strategy_name = self._get_strategy_name(strategy)
            health = self.health_trackers[strategy_name]

            if not health.is_enabled:
                logger.warning(f"Strategy '{strategy_name}' is disabled, skipping execution")
                continue

            start_time = time()
            try:
                strategy.run(run_dt)
                execution_time_ms = (time() - start_time) * 1000
                health.record_success(execution_time_ms)

                logger.debug(f"Strategy '{strategy_name}' executed successfully in {execution_time_ms:.2f}ms")
            except Exception as e:
                execution_time_ms = (time() - start_time) * 1000
                health.record_error(execution_time_ms)

                logger.error(
                    f"Error in strategy '{strategy_name}' "
                    f"(error #{health.consecutive_errors}/{health.max_consecutive_errors}): {e}"
                )
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")

                if strategy.notifier:
                    strategy.notifier.on_error(strategy_name, str(e))

                if not health.is_enabled:
                    logger.critical(f"Circuit breaker tripped for '{strategy_name}' - strategy is now DISABLED")
                    if strategy.notifier:
                        strategy.notifier.on_circuit_breaker_tripped(strategy_name, health.consecutive_errors)

    def get_health_status(self) -> Dict[str, dict]:
        return {
            name: health.get_status_summary()
            for name, health in self.health_trackers.items()
        }

    def get_aggregate_stats(self) -> dict:
        total_strategies = len(self.strategies)
        enabled_strategies = sum(1 for h in self.health_trackers.values() if h.is_enabled)
        disabled_strategies = total_strategies - enabled_strategies

        total_runs = sum(h.total_runs for h in self.health_trackers.values())
        total_successes = sum(h.successful_runs for h in self.health_trackers.values())
        total_errors = sum(h.total_errors for h in self.health_trackers.values())

        overall_success_rate = (
            (total_successes / total_runs * 100) if total_runs > 0 else 0.0
        )

        avg_exec_time = (
            sum(h.total_execution_time_ms for h in self.health_trackers.values())
            / total_runs
            if total_runs > 0
            else 0.0
        )

        return {
            "total_strategies": total_strategies,
            "enabled_strategies": enabled_strategies,
            "disabled_strategies": disabled_strategies,
            "total_runs": total_runs,
            "total_successes": total_successes,
            "total_errors": total_errors,
            "overall_success_rate": f"{overall_success_rate:.2f}%",
            "avg_execution_time_ms": f"{avg_exec_time:.2f}",
        }

    def enable_strategy(self, strategy_name: str):
        if strategy_name in self.health_trackers:
            self.health_trackers[strategy_name].enable()
        else:
            logger.warning(f"Strategy '{strategy_name}' not found")

    def disable_strategy(self, strategy_name: str):
        if strategy_name in self.health_trackers:
            self.health_trackers[strategy_name].disable()
        else:
            logger.warning(f"Strategy '{strategy_name}' not found")

    def enable_all(self):
        for health in self.health_trackers.values():
            if not health.is_enabled:
                health.enable()

    def disable_all(self):
        for health in self.health_trackers.values():
            if health.is_enabled:
                health.disable()

    def close(self):
        logger.info("=" * 60)
        logger.info("MultiTrader Shutdown - Final Statistics")
        logger.info("=" * 60)

        # Log aggregate stats
        agg_stats = self.get_aggregate_stats()
        logger.info(f"Total Strategies: {agg_stats['total_strategies']}")
        logger.info(f"Enabled: {agg_stats['enabled_strategies']}, Disabled: {agg_stats['disabled_strategies']}")
        logger.info(f"Total Runs: {agg_stats['total_runs']}")
        logger.info(f"Successes: {agg_stats['total_successes']}, Errors: {agg_stats['total_errors']}")
        logger.info(f"Overall Success Rate: {agg_stats['overall_success_rate']}")
        logger.info(f"Average Execution Time: {agg_stats['avg_execution_time_ms']}ms")
        logger.info("-" * 60)

        # Log per-strategy stats
        for strategy_name, health in self.health_trackers.items():
            status = health.get_status_summary()
            logger.info(
                f"Strategy '{strategy_name}': "
                f"{status['total_runs']} runs, "
                f"{status['successful_runs']} successful, "
                f"{status['total_errors']} errors, "
                f"success rate: {status['success_rate']}"
            )

        # Call cleanup on each strategy
        for strategy in self.strategies:
            strategy_name = self._get_strategy_name(strategy)

            if hasattr(strategy, "cleanup"):
                try:
                    strategy.cleanup()
                    logger.info(f"Strategy '{strategy_name}' cleanup completed")
                except Exception as e:
                    logger.error(f"Error during cleanup for '{strategy_name}': {e}")

        logger.info("=" * 60)