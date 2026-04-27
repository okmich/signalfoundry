"""IB-native asyncio event loop owning one or more strategies."""
import asyncio
import logging
import signal
from typing import Optional

from ib_async import IB

from .functions.ib import cancel_all_pending_orders, connect_ib

logger = logging.getLogger(__name__)


class IBEventLoop:
    """IB-native async event loop. Replaces the polled MultiTrader for IB.

    All activity is coroutine-driven; ``ib_async`` event handlers are async and
    scheduled by asyncio. Entry point is ``start()``, which calls
    ``asyncio.run(self.run())``.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 4002, client_id: int = 1):
        self.ib = IB()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._strategies: list = []
        self._closing = False
        self._shutdown_event: Optional[asyncio.Event] = None

    def add_strategy(self, strategy) -> None:
        self._strategies.append(strategy)

    def start(self) -> None:
        """Synchronous entry point. Creates the asyncio event loop and blocks until shutdown."""
        asyncio.run(self.run())

    async def run(self) -> None:
        self._shutdown_event = asyncio.Event()
        await connect_ib(self.ib, self._host, self._port, self._client_id)
        logger.info(
            f"Connected to IB at {self._host}:{self._port} (clientId={self._client_id})"
        )

        for s in self._strategies:
            try:
                await s._bootstrap(self.ib)
                logger.info(f"Bootstrapped {s.strategy_config.name}")
            except Exception as e:
                logger.exception(f"Failed to bootstrap {s.strategy_config.name}: {e}")
                raise

        self.ib.disconnectedEvent += self._on_disconnect

        # add_signal_handler is POSIX-only; fall back to signal.signal on Windows.
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.close()))
        except NotImplementedError:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: asyncio.ensure_future(self.close()))

        await self._shutdown_event.wait()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        logger.info("Shutting down IBEventLoop")

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
                s.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup {s.strategy_config.name}: {e}")

        await asyncio.sleep(1.0)
        self.ib.disconnect()
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
            try:
                await connect_ib(self.ib, self._host, self._port, self._client_id)
                for s in self._strategies:
                    await s._resubscribe(self.ib)
                logger.info(f"Reconnected after {attempt + 1} attempt(s)")
                return
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)
        logger.critical("Reconnection failed after max attempts — initiating shutdown")
        asyncio.ensure_future(self.close())
