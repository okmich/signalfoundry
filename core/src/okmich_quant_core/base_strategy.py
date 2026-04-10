import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from .config import StrategyConfig
from .notification.base import BaseNotifier
from .signal import BaseSignal

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig, signal: BaseSignal, notifier: Optional[BaseNotifier] = None,
                 *args, **kwargs):
        self.strategy_config = config
        self.signal_generator = signal
        self.notifier = notifier
        self.args = args
        self.kwargs = kwargs
        self.latest_run_dt = None
        self.previous_run_dt = None
        self.prev_position_chk_dt = None
        self.open_position_count = 0

    def manage_positions(self, run_dt: datetime, flag: bool = False) -> int:
        """
        Search and runs position management based on the instance's position manager implementation and returns
        the number of open positions

        :param run_dt:   - datetime this call was initiated
        :param flag:bool - indicates this was called on a new bar
        """
        return 0

    @abstractmethod
    def on_new_bar(self):
        """
        Runs the complete strategy defined by the implementation. This at minimum should include
        - fetching data
        - generate signals
        - manage positions or possibly exiting positions based on signals
        - open new positions based on signals
        """
        pass

    @abstractmethod
    def is_new_bar(self, run_dt: datetime) -> bool:
        """
        Check if the given datetime represents a new bar for this strategy's timeframe.
        Subclasses must implement broker-specific logic.

        Args:
            run_dt: The datetime to check

        Returns:
            True if this is a new bar, False otherwise
        """
        pass

    def run(self, run_dt: datetime):
        """
        Template method that defines the strategy execution workflow.

        Prevents duplicate runs, manages positions, and calls on_new_bar() when appropriate.

        Args:
            run_dt: The datetime this run was initiated
        """
        # Prevent duplicate runs within 1 second
        if self.previous_run_dt and abs((run_dt - self.previous_run_dt).total_seconds()) <= 1:
            return

        # Evaluate new-bar status once to avoid double side effects and pass
        # the correct flag to manage_positions (flag=True only on a new bar).
        is_new = self.is_new_bar(run_dt)

        # Manage existing positions, passing new-bar status so position manager
        # can distinguish between intra-bar and new-bar calls.
        self.open_position_count = self.manage_positions(run_dt, is_new)

        # Execute strategy logic only on a new bar
        if is_new:
            self.latest_run_dt = run_dt
            self.on_new_bar()

        self.previous_run_dt = run_dt

    def cleanup(self):
        """
        Called by MultiTrader.close() on shutdown. Flushes and closes the notifier if one is configured. Subclasses may
        override to add their own teardown — remember to call super().cleanup() to ensure the notifier is closed.
        """
        if self.notifier:
            self.notifier.close()
