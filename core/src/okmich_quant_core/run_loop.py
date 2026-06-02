import atexit
import logging
import signal
import sys
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Optional, Union

from .broker_session import BrokerSession
from .config import RunLoopConfig
from .logging import RunnerIdentity, RunnerStatus
from .logging.identity import runner_strategy_root
from .multi_trader import MultiTrader
from .trader import Trader

logger = logging.getLogger(__name__)


def _library_versions() -> dict:
    out: dict = {}
    try:
        out["okmich-quant-core"] = version("okmich-quant-core")
    except PackageNotFoundError:
        pass
    return out


class RunLoop:
    """Polled runner lifecycle owner (MT5). Binds the runner identity onto every strategy before the
    first bar, and writes the runner-lifecycle **status file** at startup/shutdown (read directly by
    the Supervisor — not a JSONL the Supervisor must scan).

    The runner script owns the broker handle, so it injects a :class:`BrokerSession` (and may inject
    a runner identity); the loop uses it for the status-file identity and the **proven** disconnect at
    shutdown. Without one, identity is degraded and ``broker_disconnected`` is ``False``.
    """

    def __init__(self, config: RunLoopConfig, trader: Union[Trader, MultiTrader], *,
                 broker_session: Optional[BrokerSession] = None,
                 runner_identity: Optional[RunnerIdentity] = None,
                 runner_name: str = "mt5_runner", log_base=None):
        if trader is None:
            raise ValueError("Trader cannot be None")
        self.config = config
        self.trader = trader
        # Per-second idempotency guards: store the last datetime at which each
        # action was dispatched.  With sleep_interval < 1s, the same second
        # bucket can be visited multiple times; these prevent duplicate calls.
        self._last_run_dt: Optional[datetime] = None
        self._last_chk_dt: Optional[datetime] = None

        self._broker_session = broker_session
        self._runner_identity = runner_identity
        self._runner_name = runner_name
        self._log_base = log_base
        self._runner_status: Optional[RunnerStatus] = None
        self._started = False
        self._shut_down = False
        self._stop_requested = False

    def _startup(self) -> None:
        """Build the runner identity, bind every strategy, and write the running status file."""
        if self._started:
            return
        if self._broker_session is not None:
            broker, account, session_id = (self._broker_session.broker, self._broker_session.account_id,
                                           self._broker_session.broker_session_id)
        else:
            logger.warning("RunLoop: no BrokerSession injected — runner identity degraded and shutdown "
                           "cannot prove broker_disconnected")
            broker, account, session_id = "unknown", "unknown", None
        if self._runner_identity is None:
            self._runner_identity = RunnerIdentity.generate(name=self._runner_name, broker=broker,
                                                            account_id=account, broker_session_id=session_id)

        # The runner-root strategy is the plain code for a single Trader, or ``<strategy>-multi`` for a
        # MultiTrader (statutory, framework-derived — the developer never authors the suffix). Applied at
        # bind so inference paths, record identity, AND the single status file all reflect it (§7.1/§10).
        multi = isinstance(self.trader, MultiTrader)
        for strategy in self.trader.strategies:
            runner_strategy = runner_strategy_root(strategy.log_binding.logical.strategy, multi=multi)
            strategy.bind_runner_identity(self._runner_identity, runner_strategy=runner_strategy)

        self._runner_status = RunnerStatus(
            self._runner_identity, [s.log_binding.logical for s in self.trader.strategies],
            log_base=self._log_base, library_versions=_library_versions())
        self._runner_status.mark_started()
        # Mark completed ONLY after the running status is durably written: a partial startup (a raise before
        # this point) leaves _started False, so the failure is not masked as "already started".
        self._started = True

    def _shutdown(self, reason: str) -> None:
        """Drain + close strategies, disconnect (proven), then mark the status file stopped (§7.4).

        ``clean`` reflects whether the FULL graceful path ran: the logger drain/close succeeded AND
        the broker disconnect was proven. It is NOT hardcoded true — a failed close or an unproven
        disconnect yields ``clean=False`` so the Supervisor's clean-stop proof is honest (§7.4).
        """
        if self._shut_down:
            return
        self._shut_down = True
        close_ok = True
        try:
            self.trader.close()  # drains + closes per-strategy loggers (bars flushed before disconnect, F2)
        except Exception:
            close_ok = False
            logger.exception("RunLoop: error during trader.close()")

        if self._broker_session is not None:
            broker_disconnected = self._broker_session.disconnect()
        else:
            broker_disconnected = False

        clean = close_ok and broker_disconnected
        if self._runner_status is not None:
            try:
                self._runner_status.mark_stopped(broker_disconnected=broker_disconnected, clean=clean, reason=reason)
            except Exception:
                logger.exception("RunLoop: failed to write shutdown status")

    def _install_signal_handlers(self) -> None:
        """Register a SIGTERM stop-request + an atexit backstop so the runner shutdown record + proven
        disconnect are emitted on the normal kill paths (systemd/supervisor SIGTERM, process exit),
        not only on KeyboardInterrupt (LOGGING_CONTRACT §7.4).

        The SIGTERM handler only sets a flag — it does NOT run the (lock-taking, thread-joining)
        shutdown itself, which is unsafe from a signal handler and could deadlock if it fired mid-write.
        atexit covers a SystemExit / fatal escape; ``_shutdown`` is idempotent so multiple paths are safe.
        """
        atexit.register(self._shutdown, "atexit")
        try:
            signal.signal(signal.SIGTERM, self._request_stop)
        except (ValueError, OSError):  # not the main thread / platform without SIGTERM
            logger.debug("RunLoop: could not install SIGTERM handler (non-main-thread or unsupported)")

    def _request_stop(self, *_args) -> None:
        self._stop_requested = True

    def run(self):
        """Start the event loop with simulated clock.

        ``_startup()`` is inside the ``try`` so a startup failure still runs the ``finally``'s
        ``_shutdown`` (broker disconnect + status write) — and then propagates (the runner script
        sees the original exception, mirroring ``IBEventLoop.run``). A clean loop exit reaches
        ``sys.exit(0)``. ``reason`` distinguishes the stop cause for the status file (§7.4 audit).
        """
        reason = "operator_stop"
        try:
            self._startup()
            self._install_signal_handlers()
            while not self._stop_requested:
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
                except KeyboardInterrupt:
                    logging.info("Event loop stopped by user")
                    reason = "keyboard_interrupt"
                    break
                except Exception as e:
                    # Per-iteration resilience: a transient error must NOT kill the runner — log and
                    # keep looping (unchanged from the original design).
                    logging.error(f"Trading loop error: {e}")
                    time.sleep(self.config.sleep_interval)
        finally:
            # Single, idempotent shutdown for every exit path (SIGTERM flag, KeyboardInterrupt,
            # startup failure). On a startup failure the original exception propagates AFTER this.
            self._shutdown(reason)
            logging.info("Shutting down application....")
        sys.exit(0)
