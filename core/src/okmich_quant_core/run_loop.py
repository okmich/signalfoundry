import logging
import sys
import time
from datetime import datetime
from typing import Optional, Union

from .config import RunLoopConfig
from .multi_trader import MultiTrader
from .trader import Trader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RunLoop:
    def __init__(self, config: RunLoopConfig, trader: Union[Trader, MultiTrader]):
        if trader is None:
            raise ValueError("Trader cannot be None")
        self.config = config
        self.trader = trader
        # Per-second idempotency guards: store the last datetime at which each
        # action was dispatched.  With sleep_interval < 1s, the same second
        # bucket can be visited multiple times; these prevent duplicate calls.
        self._last_run_dt: Optional[datetime] = None
        self._last_chk_dt: Optional[datetime] = None

    def run(self):
        """Start the event loop with simulated clock."""
        while True:
            try:
                now_dt = datetime.now().replace(microsecond=0)
                if now_dt.second == 0:
                    if now_dt != self._last_run_dt:
                        self._last_run_dt = now_dt
                        self.trader.run(now_dt)
                elif now_dt.second % self.config.chk_position_interval == 0:
                    if now_dt != self._last_chk_dt:
                        self._last_chk_dt = now_dt
                        self.trader.check_positions(now_dt)

                time.sleep(self.config.sleep_interval)
            except AttributeError as e:
                logging.error(f"Trading error: {e}")
                time.sleep(self.config.sleep_interval)
            except KeyboardInterrupt:
                logging.info("Event loop stopped by user")
                self.trader.close()
                logging.info("Shutting down application....")
                sys.exit(0)
            except Exception as e:
                logging.info(f"Error: {e}")
                time.sleep(self.config.sleep_interval)
