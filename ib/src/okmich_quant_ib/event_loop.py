"""IB-native asyncio event loop owning one or more strategies."""
import asyncio
import logging
import signal
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from ib_async import IB

from okmich_quant_core import RunnerIdentity, RunnerStatus
from okmich_quant_core.broker_session import BrokerSession

from .broker_session import IBBrokerSession
from .functions.ib import cancel_all_pending_orders, connect_ib

logger = logging.getLogger(__name__)


def _library_versions() -> dict:
    out: dict = {}
    for pkg in ("okmich-quant-core", "okmich-quant-ib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


class IBEventLoop:
    """IB-native async event loop. Replaces the polled MultiTrader for IB.

    All activity is coroutine-driven; ``ib_async`` event handlers are async and
    scheduled by asyncio. Entry point is ``start()``, which calls
    ``asyncio.run(self.run())``.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1, *,
                 broker_session: Optional[BrokerSession] = None,
                 runner_identity: Optional[RunnerIdentity] = None,
                 runner_name: str = "ib_runner", log_base=None):
        self.ib = IB()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._strategies: list = []
        self._closing = False
        self._shutdown_event: Optional[asyncio.Event] = None
        # Runner lifecycle status (LOGGING_CONTRACT runner-lifecycle channel). The broker session +
        # runner identity may be injected by the runner script, or built here from the live session.
        self._broker_session = broker_session
        self._runner_identity = runner_identity
        self._runner_name = runner_name
        self._log_base = log_base
        self._runner_status: Optional[RunnerStatus] = None

    def add_strategy(self, strategy) -> None:
        self._strategies.append(strategy)

    def start(self) -> None:
        """Synchronous entry point. Creates the asyncio event loop and blocks until shutdown."""
        asyncio.run(self.run())

    async def run(self) -> None:
        self._shutdown_event = asyncio.Event()
        try:
            await connect_ib(self.ib, self._host, self._port, self._client_id)
            logger.info(
                f"Connected to IB at {self._host}:{self._port} (clientId={self._client_id})"
            )

            # Build the runner identity now that the session is up (managedAccounts/clientId are
            # available). Bind each strategy BEFORE its _bootstrap (which subscribes to bars) so the
            # inference-log envelope is complete before any heartbeat can fire (risk C / first-bar race).
            if self._broker_session is None:
                self._broker_session = IBBrokerSession(self.ib, self._host, self._port, self._client_id)
            if self._runner_identity is None:
                self._runner_identity = RunnerIdentity.generate(
                    name=self._runner_name, broker=self._broker_session.broker,
                    account_id=self._broker_session.account_id,
                    broker_session_id=self._broker_session.broker_session_id)

            for s in self._strategies:
                s.bind_runner_identity(self._runner_identity)
                await s._bootstrap(self.ib)
                logger.info(f"Bootstrapped {s.strategy_config.name}")

            # Runner-lifecycle status file: state=running across each logical system brought up.
            self._runner_status = RunnerStatus(
                self._runner_identity, [s.log_binding.logical for s in self._strategies],
                log_base=self._log_base, library_versions=_library_versions())
            self._runner_status.mark_started()

            self.ib.disconnectedEvent += self._on_disconnect

            # Signal handlers only REQUEST stop (set the event); close() is driven from run() below.
            # add_signal_handler is POSIX-only and runs on the loop; on Windows we fall back to
            # signal.signal, whose handler may fire on an arbitrary thread — so it must NOT touch the
            # loop directly (ensure_future is not async-signal-safe and needs a running loop in the
            # calling thread). call_soon_threadsafe is the safe cross-thread wake.
            loop = asyncio.get_running_loop()
            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                def _request_stop(*_args):
                    loop.call_soon_threadsafe(self._shutdown_event.set)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, _request_stop)

            await self._shutdown_event.wait()
            # Normal stop path: the event was set (signal or _reconnect giving up). Drive the one
            # graceful close from here on the loop thread (idempotent via _closing).
            await self.close()
        except Exception:
            # A failure during connect / bootstrap / startup (or an unexpected error while waiting) must
            # not leak the broker session, loggers, or the shutdown proof. close() is idempotent.
            logger.exception("IBEventLoop.run: setup/run failed — running cleanup")
            await self.close()
            raise

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        logger.info("Shutting down IBEventLoop")

        cleanup_ok = True
        for s in self._strategies:
            try:
                await cancel_all_pending_orders(
                    self.ib, s.strategy_config.symbol, s.strategy_config.magic,
                    con_id=s.contract.conId if s.contract else None,
                )
            except Exception as e:
                logger.error(f"Failed to cancel orders for {s.strategy_config.name}: {e}")
            try:
                await s._unsubscribe(self.ib)
            except Exception as e:
                logger.error(f"Failed to unsubscribe {s.strategy_config.name}: {e}")
            try:
                s.cleanup()  # flushes + closes the strategy's inference logger (drains queued bars, §10/F2)
            except Exception as e:
                cleanup_ok = False
                logger.error(f"Failed to cleanup {s.strategy_config.name}: {e}")

        await asyncio.sleep(1.0)  # let order cancellations settle before releasing the session

        # Disconnect FIRST, then mark the status file stopped with the PROVEN result
        # (LOGGING_CONTRACT §7.4 ordering; risk F2/F4). _closing gates this to once per runner.
        if self._broker_session is not None:
            broker_disconnected = self._broker_session.disconnect()
        else:
            try:
                self.ib.disconnect()
            except Exception:
                logger.exception("IBEventLoop: ib.disconnect() failed")
            broker_disconnected = not self.ib.isConnected()

        # clean reflects the real outcome (logger drains succeeded AND disconnect proven), not a
        # hardcoded true — so the Supervisor's clean-stop proof is honest (LOGGING_CONTRACT §7.4).
        clean = cleanup_ok and broker_disconnected
        if self._runner_status is not None:
            try:
                self._runner_status.mark_stopped(broker_disconnected=broker_disconnected, clean=clean,
                                                 reason="shutdown")
            except Exception:
                logger.exception("IBEventLoop: failed to write shutdown status")

        if self._shutdown_event:
            self._shutdown_event.set()

    async def _on_disconnect(self):
        if self._closing:
            return
        logger.warning("IB disconnected — attempting reconnect")
        for s in self._strategies:
            try:
                await s._unsubscribe(self.ib)
            except Exception as e:
                logger.error(f"Unsubscribe during disconnect failed: {e}")
        await self._reconnect_with_backoff()

    async def _reconnect_with_backoff(self, max_attempts: int = 10):
        delay = 2.0
        for attempt in range(max_attempts):
            if self._closing:  # a shutdown began mid-backoff — stop trying to resurrect the session
                return
            try:
                await connect_ib(self.ib, self._host, self._port, self._client_id)
                if self._closing:  # shutdown raced the reconnect — don't resubscribe into a closing loop
                    return
                for s in self._strategies:
                    await s._resubscribe(self.ib)
                logger.info(f"Reconnected after {attempt + 1} attempt(s)")
                return
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        logger.critical("Reconnection failed after max attempts — initiating shutdown")
        # Request stop via the event; run() drives the single graceful close() on the loop thread.
        if self._shutdown_event is not None:
            self._shutdown_event.set()
