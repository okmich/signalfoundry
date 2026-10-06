"""CCXT-native asyncio event loop owning one exchange account and the strategies trading on it.

Mirrors ``IBEventLoop``: one connection, strategies bound to the runner identity before they bootstrap, a runner
status file, signal handlers that only REQUEST a stop, and one idempotent ``close()``.

Crypto-specific:

* Strategy isolation is enforced in :meth:`CryptoEventLoop.add_strategy`, before anything connects: at most one
  strategy per (exchange, environment, account, market type, symbol).
* Order / fill / position streams are ACCOUNT-wide on every venue, so they are consumed once here and dispatched by
  symbol - never one subscription per strategy competing for the same messages.
* A stream outage is followed by a REST reconciliation of every strategy's book before trading resumes.
* Shutdown cancels only working ENTRY / EXIT orders. Protective stops stay on the venue: cancelling them on the way
  down would leave positions naked exactly while nothing is watching them.
"""
import asyncio
import logging
import signal
from importlib.metadata import PackageNotFoundError, version
from typing import Awaitable, Callable, Optional

from okmich_quant_core import RunnerIdentity, RunnerStatus
from okmich_quant_core.broker_session import BrokerSession
from okmich_quant_core.logging.identity import runner_strategy_root

from .broker_session import CryptoBrokerSession
from .config import CryptoVenueConfig, check_isolation
from .enums import MarginModeScope, MarketType
from .functions.crypto import connect_exchange, load_credentials
from .models import Credentials
from .resilience import ErrorClass, classify_ccxt_error
from .strategy import BaseCryptoStrategy, VenueContext
from .timeframe_utils import utc_now_ms
from .venue.registry import resolve_profile

logger = logging.getLogger(__name__)

ExchangeFactory = Callable[..., Awaitable]


def _library_versions() -> dict:
    out: dict = {}
    for pkg in ("okmich-quant-core", "okmich-quant-crypto", "ccxt"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


class CryptoEventLoop:
    """Async runner for one venue account. Entry point is :meth:`start` (``asyncio.run(self.run())``).

    ``multi`` sets the runner-root log folder exactly as core's ``RunLoop`` does: ``<strategy>`` for a single trader,
    ``<strategy>-multi`` for a multi-trader (one process, N symbols, one venue session). It is where the per-symbol
    inference logs and the one ``status.json`` go - the paths the Fleet Supervisor tails - so pass
    ``multi=bool(system.strategies)`` to match a ``strategies[]`` config (core's text log and the Supervisor classify
    the same way). ``None`` = multi when more than one strategy is added. Sleeves of a multi-trader share their
    strategy name, as in MT5 / IB systems.
    """

    def __init__(self, venue: CryptoVenueConfig, *, credentials: Optional[Credentials] = None,
                 exchange_factory: Optional[ExchangeFactory] = None, broker_session: Optional[BrokerSession] = None,
                 runner_identity: Optional[RunnerIdentity] = None, runner_name: str = "crypto_runner", log_base=None,
                 multi: Optional[bool] = None, clock: Callable[[], int] = utc_now_ms, sleep: Callable = asyncio.sleep):
        self.venue = venue
        self._credentials = credentials
        self._exchange_factory = exchange_factory
        self._broker_session = broker_session
        self._runner_identity = runner_identity
        self._runner_name = runner_name
        self._log_base = log_base
        self._multi = multi
        self._clock = clock
        self._sleep = sleep
        self._strategies: list[BaseCryptoStrategy] = []
        self.exchange = None
        self.profile = None
        self._stream_tasks: list[asyncio.Task] = []
        self._closing = False
        self._shutdown_event: Optional[asyncio.Event] = None
        self._runner_status: Optional[RunnerStatus] = None

    # ------------------------------------------------------------------ setup
    def add_strategy(self, strategy: BaseCryptoStrategy) -> None:
        """Register a strategy. Raises before anything connects if it breaks the isolation rule."""
        if not isinstance(strategy, BaseCryptoStrategy):
            raise TypeError(f"CryptoEventLoop runs BaseCryptoStrategy subclasses (got {type(strategy).__name__})")
        check_isolation(self.venue, [s.strategy_config for s in self._strategies] + [strategy.strategy_config])
        self._strategies.append(strategy)

    @property
    def strategies(self) -> list[BaseCryptoStrategy]:
        return list(self._strategies)

    def start(self) -> None:
        """Synchronous entry point: creates the event loop and blocks until shutdown."""
        asyncio.run(self.run())

    async def run(self) -> None:
        self._shutdown_event = asyncio.Event()
        try:
            await self._startup()
            self._install_signal_handlers()
            await self._shutdown_event.wait()
            await self.close()
        except Exception:
            logger.exception("CryptoEventLoop.run: setup/run failed - running cleanup")
            await self.close()
            raise

    async def _startup(self) -> None:
        if not self._strategies:
            raise ValueError("CryptoEventLoop has no strategies; call add_strategy() first")
        self.profile = resolve_profile(self.venue.exchange_id)
        credentials = self._credentials or load_credentials(self.venue)
        if not credentials.present:
            raise ValueError(f"no API credentials: set the environment variables {self.venue.api_key_env!r} and "
                             f"{self.venue.secret_env!r} (e.g. via core.env_loader) before starting")
        if self._exchange_factory is not None:
            self.exchange = await self._exchange_factory(self.profile, self.venue, credentials)
        else:
            self.exchange = await connect_exchange(self.profile, self.venue, credentials)
        logger.info("Connected to %s (%s) - %r", self.venue.exchange_id, self.venue.environment.value, self.profile)

        if self._broker_session is None:
            account_id = await self.profile.account_uid(self.exchange) or credentials.fingerprint() or "unknown"
            self._broker_session = CryptoBrokerSession(self.exchange, self.venue.exchange_id, self.venue.environment,
                                                       account_id)
        if self._runner_identity is None:
            self._runner_identity = RunnerIdentity.generate(
                name=self._runner_name, broker=self._broker_session.broker,
                account_id=self._broker_session.account_id, broker_session_id=self._broker_session.broker_session_id)

        has_perps = any(s.strategy_config.market_type is MarketType.LINEAR_PERP for s in self._strategies)
        account_scope = self.profile.margin_mode_scope is MarginModeScope.ACCOUNT
        if self.venue.margin_mode is not None and has_perps and account_scope:
            logger.warning("Setting margin mode %s for the WHOLE %s account", self.venue.margin_mode.value,
                           self.venue.exchange_id)
            await self.profile.apply_margin_mode(self.exchange, self.venue.margin_mode, None)

        ctx = VenueContext(exchange=self.exchange, profile=self.profile, venue=self.venue, clock=self._clock,
                           sleep=self._sleep)
        multi = self._multi if self._multi is not None else len(self._strategies) > 1
        for s in self._strategies:
            # The runner root re-points the logical identity, so inference paths, records and status.json agree.
            runner_strategy = runner_strategy_root(s.log_binding.logical.strategy, multi=multi)
            # Bind BEFORE bootstrap so the envelope is complete before any heartbeat can fire (first-bar race).
            s.bind_runner_identity(self._runner_identity, runner_strategy=runner_strategy)
            await s._bootstrap(ctx)
            logger.info("Bootstrapped %s", s.strategy_config.name)

        self._runner_status = RunnerStatus(self._runner_identity, [s.log_binding.logical for s in self._strategies],
                                           log_base=self._log_base, library_versions=_library_versions())
        self._runner_status.mark_started()

        stream_account = any(s.caps.stream_account for s in self._strategies)
        if stream_account:
            self._start_account_streams()
        for s in self._strategies:
            s.start(stream_account)

    def _install_signal_handlers(self) -> None:
        # Handlers only REQUEST stop. add_signal_handler is POSIX-only; on Windows signal.signal may fire on another
        # thread, so it must only wake the loop via call_soon_threadsafe.
        loop = asyncio.get_running_loop()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._shutdown_event.set)
        except NotImplementedError:
            def _request_stop(*_args):
                loop.call_soon_threadsafe(self._shutdown_event.set)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, _request_stop)

    def request_stop(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    # ------------------------------------------------------------------ account streams
    def _start_account_streams(self) -> None:
        streaming = [s for s in self._strategies if s.caps.stream_account]
        perp_symbols = [s.spec.symbol for s in streaming if s.strategy_config.market_type is MarketType.LINEAR_PERP]
        spot = any(s.strategy_config.market_type is MarketType.SPOT for s in streaming)
        calls = self.profile.account_stream_calls(self.exchange, spot=spot, perp_symbols=perp_symbols)
        for i, (kind, watch) in enumerate(calls):
            self._stream_tasks.append(asyncio.create_task(self._account_stream(kind, watch),
                                                          name=f"account:{kind}:{i}"))

    async def _account_stream(self, kind: str, watch: Callable[[], Awaitable[list]]) -> None:
        backoff = 1.0
        offline_since: Optional[int] = None
        while not self._closing:
            try:
                items = await watch()
                if offline_since is not None:
                    logger.warning("account %s stream recovered; reconciling every strategy over REST", kind)
                    if await self._reconcile_all(offline_since):
                        offline_since = None  # kept on failure: the next delivery retries the reconciliation
                for item in items or []:
                    await self._dispatch(kind, item)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_class = classify_ccxt_error(exc, self.profile)
                if offline_since is None:
                    offline_since = self._clock()
                if error_class is ErrorClass.BANNED:
                    logger.critical("account %s stream: venue refused the account (%s)", kind, exc)
                    for s in self._strategies:
                        s._on_banned(exc)
                    backoff = 60.0
                else:
                    logger.warning("account %s stream failed (%s: %s); retrying in %.0fs", kind, type(exc).__name__,
                                   exc, backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _dispatch(self, kind: str, item: dict) -> None:
        symbol = item.get("symbol")
        for s in self._strategies:
            if s.spec is None or s.spec.symbol != symbol or not s.caps.stream_account:
                continue
            try:
                if kind == "orders":
                    await s.on_venue_order(item)
                elif kind == "fills":
                    await s.on_venue_fill(item)
                else:
                    await s.on_venue_position(item)
            except Exception:
                logger.exception("%s: handling %s update failed", s.strategy_config.name, kind)

    async def _reconcile_all(self, offline_since: Optional[int]) -> bool:
        """Reconcile every strategy after an outage. True only if ALL succeeded."""
        ok = True
        for s in self._strategies:
            try:
                await s.on_reconnected(offline_since)
            except Exception:
                ok = False
                logger.exception("%s: reconciliation after reconnect failed", s.strategy_config.name)
        return ok

    # ------------------------------------------------------------------ shutdown
    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        logger.info("Shutting down CryptoEventLoop")
        for task in self._stream_tasks:
            task.cancel()
        for task in self._stream_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._stream_tasks = []

        cleanup_ok = True
        for s in self._strategies:
            try:
                await s.stop()
            except Exception as e:
                logger.error("Failed to stop %s: %s", s.strategy_config.name, e)
            if s.spec is not None and self.exchange is not None:
                try:
                    cancelled = await s.cancel_working_orders()
                    if cancelled:
                        logger.info("%s: cancelled %d working entry/exit order(s); protective stops kept",
                                    s.strategy_config.name, cancelled)
                except Exception as e:
                    logger.error("Failed to cancel working orders for %s: %s", s.strategy_config.name, e)
            try:
                s.cleanup()  # settles held closes, drains the inference logger, closes the notifier
            except Exception as e:
                cleanup_ok = False
                logger.error("Failed to cleanup %s: %s", s.strategy_config.name, e)

        broker_disconnected = await self._release_session()
        clean = cleanup_ok and broker_disconnected
        if self._runner_status is not None:
            try:
                self._runner_status.mark_stopped(broker_disconnected=broker_disconnected, clean=clean,
                                                 reason="shutdown")
            except Exception:
                logger.exception("CryptoEventLoop: failed to write shutdown status")
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    async def _release_session(self) -> bool:
        if isinstance(self._broker_session, CryptoBrokerSession):
            return await self._broker_session.aclose()
        if self._broker_session is not None:
            return self._broker_session.disconnect()
        if self.exchange is not None:
            try:
                await self.exchange.close()
            except Exception:
                logger.exception("CryptoEventLoop: exchange.close() failed")
        return False
