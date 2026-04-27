import logging
import traceback
from collections import Counter
from datetime import datetime
from time import time
from typing import Dict, List, Optional, Tuple

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

        name_counts = Counter(self._get_strategy_name(strategy) for strategy in strategies)
        seen: Dict[Tuple[str, str, int], BaseStrategy] = {}
        self._strategy_keys: Dict[BaseStrategy, str] = {}
        self._strategy_names_by_key: Dict[str, str] = {}

        for strategy in strategies:
            identity = self._get_strategy_identity(strategy)
            if identity in seen:
                name, symbol, magic = identity
                raise ValueError(
                    f"Duplicate strategy identity (name='{name}', symbol='{symbol}', "
                    f"magic={magic}). Each strategy registered with MultiTrader must have "
                    "a unique (name, symbol, magic) combination."
                )
            seen[identity] = strategy
            key = self._format_strategy_key(identity, name_counts[identity[0]])
            self._strategy_keys[strategy] = key
            self._strategy_names_by_key[key] = identity[0]

        # Health trackers are keyed by a collision-free strategy key. For unique names the key is just the name; for
        # duplicate names it includes symbol and magic to avoid silently overwriting health state.
        self.health_trackers: Dict[str, StrategyHealth] = {}
        for strategy in strategies:
            strategy_key = self._get_strategy_key(strategy)
            self.health_trackers[strategy_key] = StrategyHealth(
                strategy_name=strategy_key,
                max_consecutive_errors=max_consecutive_errors
            )

        logger.info(
            f"MultiTrader initialized with {len(strategies)} strategies "
            f"(circuit breaker: {max_consecutive_errors} errors per strategy)"
        )

    def _get_strategy_name(self, strategy: BaseStrategy) -> str:
        """Extract strategy name from strategy config."""
        return strategy.strategy_config.name

    def _get_strategy_identity(self, strategy: BaseStrategy) -> Tuple[str, str, int]:
        """Extract the collision-resistant strategy identity from strategy config."""
        cfg = strategy.strategy_config
        return cfg.name, cfg.symbol, cfg.magic

    def _get_strategy_key(self, strategy: BaseStrategy) -> str:
        """Return the health-tracker key assigned to a strategy."""
        return self._strategy_keys[strategy]

    def _format_strategy_key(self, identity: Tuple[str, str, int], name_count: int) -> str:
        name, symbol, magic = identity
        if name_count == 1:
            return name
        return f"{name}[{symbol}:{magic}]"

    def _resolve_strategy_key(self, strategy_name_or_key: str) -> Optional[str]:
        if strategy_name_or_key in self.health_trackers:
            return strategy_name_or_key

        matches = [
            key
            for key, strategy_name in self._strategy_names_by_key.items()
            if strategy_name == strategy_name_or_key
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.warning(
                f"Strategy name '{strategy_name_or_key}' is ambiguous; use one of: {', '.join(matches)}"
            )
            return None

        logger.warning(f"Strategy '{strategy_name_or_key}' not found")
        return None

    def check_positions(self, run_dt: datetime):
        for strategy in self.strategies:
            strategy_key = self._get_strategy_key(strategy)
            health = self.health_trackers[strategy_key]

            if not health.is_enabled:
                logger.debug(f"Strategy '{strategy_key}' is disabled, skipping position check")
                continue

            try:
                strategy.manage_positions(run_dt)
                health.record_position_check()
            except Exception as e:
                logger.error(f"Error in position check for '{strategy_key}': {e}", exc_info=True,)

    def run(self, run_dt: datetime):
        for strategy in self.strategies:
            strategy_key = self._get_strategy_key(strategy)
            health = self.health_trackers[strategy_key]

            if not health.is_enabled:
                logger.warning(f"Strategy '{strategy_key}' is disabled, skipping execution")
                continue

            start_time = time()
            try:
                strategy.run(run_dt)
                execution_time_ms = (time() - start_time) * 1000
                health.record_success(execution_time_ms)

                logger.debug(f"Strategy '{strategy_key}' executed successfully in {execution_time_ms:.2f}ms")
            except Exception as e:
                execution_time_ms = (time() - start_time) * 1000
                health.record_error(execution_time_ms)

                logger.error(
                    f"Error in strategy '{strategy_key}' "
                    f"(error #{health.consecutive_errors}/{health.max_consecutive_errors}): {e}"
                )
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")

                if strategy.notifier:
                    strategy.notifier.on_error(strategy_key, str(e))

                if not health.is_enabled:
                    logger.critical(f"Circuit breaker tripped for '{strategy_key}' - strategy is now DISABLED")
                    if strategy.notifier:
                        strategy.notifier.on_circuit_breaker_tripped(strategy_key, health.consecutive_errors)

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
        strategy_key = self._resolve_strategy_key(strategy_name)
        if strategy_key is not None:
            self.health_trackers[strategy_key].enable()

    def disable_strategy(self, strategy_name: str):
        strategy_key = self._resolve_strategy_key(strategy_name)
        if strategy_key is not None:
            self.health_trackers[strategy_key].disable()

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
        for strategy_key, health in self.health_trackers.items():
            status = health.get_status_summary()
            logger.info(
                f"Strategy '{strategy_key}': "
                f"{status['total_runs']} runs, "
                f"{status['successful_runs']} successful, "
                f"{status['total_errors']} errors, "
                f"success rate: {status['success_rate']}"
            )

        # Call cleanup on each strategy
        for strategy in self.strategies:
            strategy_key = self._get_strategy_key(strategy)

            if hasattr(strategy, "cleanup"):
                try:
                    strategy.cleanup()
                    logger.info(f"Strategy '{strategy_key}' cleanup completed")
                except Exception as e:
                    logger.error(f"Error during cleanup for '{strategy_key}': {e}")

        logger.info("=" * 60)
