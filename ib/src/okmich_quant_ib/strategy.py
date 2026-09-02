"""IB strategy lifecycle (bootstrap / subscribe / unsubscribe / resubscribe) and
the ``GenericBasicIBStrategy`` execution loop fired on completed-bar events.
"""
import asyncio
import logging
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Optional, Union

import pandas as pd
from ib_async import IB, Contract

from okmich_quant_core import (
    BarOutcome, BaseSignal, BaseStrategy, ClosedTrade, CloseReason, OrderType, PositionSizingType, StrategyConfig,
    StrategyHealth,
)
from okmich_quant_core.notification.base import BaseNotifier
from okmich_quant_core.price_buffer import PriceBuffer

from .bar_aggregator import BarAggregator
from .contract import (
    DEFAULT_USE_RTH, DEFAULT_WHAT_TO_SHOW, IBContractConfig, resolve_contract,
)
from .filters import create_filter
from .functions.ib import (
    close_position as ib_close_position, fetch_contract_info,
    fetch_historical_bars_paginated, get_pending_orders, place_bracket_order,
    place_limit_order, place_market_order, place_stop_order,
)
from .position_cache import IBPositionCache
from .position_manager import get_position_manager
from .resilience import (
    ErrorClass, IBConnectionError, IBPermanentError, IBTransientError, classify_ib_error,
)
from .timeframe_utils import bar_size_to_minutes

logger = logging.getLogger(__name__)


class BaseIBStrategy(BaseStrategy):
    #: The async per-bar seam is sealed: it owns the inference-log heartbeat (§5.1).
    _IB_SEALED = frozenset({"_on_bar_close"})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # also runs BaseStrategy's seal (run / _emit_bar_record / bind)
        for name in BaseIBStrategy._IB_SEALED:
            if name in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} may not override sealed BaseIBStrategy.{name}() — the inference-log "
                    f"heartbeat lives there (LOGGING_CONTRACT §5.1). Implement on_new_bar() instead."
                )

    def __init__(self, config: StrategyConfig, signal: BaseSignal, contract_cfg: IBContractConfig,
                 notifier: Optional[BaseNotifier] = None, *, max_consecutive_errors: int = 5, **kwargs):
        tf_min = bar_size_to_minutes(config.timeframe)
        if tf_min >= 1440:
            raise ValueError(
                f"Timeframe '{config.timeframe}' is not supported in real-time bar mode. "
                "Daily bars cannot be assembled from 5-second real-time bars; override "
                "_bootstrap and subscribe via reqHistoricalDataAsync polling instead."
            )
        super().__init__(config, signal, notifier, timeframe_minutes=tf_min, **kwargs)
        # Per-strategy circuit breaker — IB has no MultiTrader, so the breaker lives here to bring
        # IB to MT5 near-par for circuit_breaker_tripped / strategy_reenabled / skipped_disabled (§7.3).
        self._ib_health = StrategyHealth(config.name, max_consecutive_errors)
        self.contract_cfg = contract_cfg
        self.max_number_of_open_positions = config.max_number_of_open_positions

        self.ib: Optional[IB] = None
        self.contract: Optional[Contract] = None
        self.contract_info: dict = {}
        self._position_cache: Optional[IBPositionCache] = None
        #: conId -> the fill that last reduced that position. IB reports the realised P/L on the fill's
        #: commission report, and a position that has already left the cache cannot be asked for it.
        self._last_fills: dict[str, object] = {}
        self._bar_aggregator: Optional[BarAggregator] = None
        self._rt_bars = None
        self._ticker = None

        self.price_buffer = PriceBuffer(
            symbol=config.symbol,
            timeframe=config.timeframe,
            buffer_size=config.bars_to_copy,
            exclude_columns=getattr(config, "exclude_columns", None),
            timeframe_minutes=tf_min,
        )
        self.filter_chain = create_filter(config)
        self.position_manager = None
        self._bracket_trades: dict[int, list] = {}
        self._bar_lock = asyncio.Lock()

    @property
    def health(self) -> StrategyHealth:
        """The per-strategy circuit-breaker health tracker (IB's MultiTrader-equivalent)."""
        return self._ib_health

    # ---- Lifecycle ----

    async def _bootstrap(self, ib: IB) -> None:
        self.ib = ib
        self.contract = await resolve_contract(ib, self.strategy_config.symbol, self.contract_cfg)
        self.contract_info = await fetch_contract_info(ib, self.contract)

        seed = await fetch_historical_bars_paginated(
            ib, self.contract, bar_size=self.strategy_config.timeframe,
            bars_to_copy=self.strategy_config.bars_to_copy,
            what_to_show=DEFAULT_WHAT_TO_SHOW[self.contract_cfg.sec_type],
            use_rth=DEFAULT_USE_RTH[self.contract_cfg.sec_type],
        )
        self.price_buffer.update(seed, datetime.now(tz=timezone.utc))

        self._position_cache = IBPositionCache(
            self.strategy_config.symbol, self.contract.conId, self.strategy_config.magic
        )
        await self._position_cache.resync(ib)

        tf_min = bar_size_to_minutes(self.strategy_config.timeframe)
        _what_to_show = DEFAULT_WHAT_TO_SHOW[self.contract_cfg.sec_type]
        self._bar_aggregator = BarAggregator(
            target_minutes=tf_min,
            on_bar_close=self._on_bar_close,
            gap_reset_seconds=max(60, tf_min * 60),
            track_volume=(_what_to_show == "TRADES"),
        )

        if self.strategy_config.position_manager:
            self.position_manager = get_position_manager(
                ib, self.contract, self.strategy_config,
                price_buffer=self.price_buffer,
                contract_info=self.contract_info,
            )

        await self._subscribe(ib)

    async def _subscribe(self, ib: IB) -> None:
        self._rt_bars = ib.reqRealTimeBars(
            self.contract, 5,
            DEFAULT_WHAT_TO_SHOW[self.contract_cfg.sec_type],
            DEFAULT_USE_RTH[self.contract_cfg.sec_type],
        )
        self._rt_bars.updateEvent += self._bar_aggregator.on_realtime_bar
        self._ticker = ib.reqMktData(self.contract, "", snapshot=False, regulatorySnapshot=False)
        ib.fillEvent += self._position_cache.on_fill
        ib.fillEvent += self._on_position_fill          # AFTER the cache: dispatch is registration-ordered
        ib.positionEvent += self._position_cache.on_position
        ib.execDetailsEvent += self._on_fill
        ib.errorEvent += self._on_error

    async def _unsubscribe(self, ib: IB) -> None:
        try:
            if self._rt_bars is not None:
                try:
                    ib.cancelRealTimeBars(self._rt_bars)
                except Exception:
                    pass
                try:
                    self._rt_bars.updateEvent -= self._bar_aggregator.on_realtime_bar
                except Exception:
                    pass
                self._rt_bars = None
            if self._ticker is not None:
                try:
                    ib.cancelMktData(self.contract)
                except Exception:
                    pass
                self._ticker = None
        finally:
            for _evt, _handler in [
                (ib.fillEvent, self._position_cache.on_fill if self._position_cache else None),
                (ib.fillEvent, self._on_position_fill),
                (ib.positionEvent, self._position_cache.on_position if self._position_cache else None),
                (ib.execDetailsEvent, self._on_fill),
                (ib.errorEvent, self._on_error),
            ]:
                if _handler is None:
                    continue
                try:
                    _evt -= _handler
                except Exception:
                    pass

    async def _resubscribe(self, ib: IB) -> None:
        """Post-reconnect — rewire events, resync position cache, reset aggregator.

        Does NOT re-seed PriceBuffer (the live data subscription resumes; gaps
        are handled by PriceBuffer's stale-buffer detection on next update).
        """
        self.ib = ib
        self._bar_aggregator._reset()
        await self._position_cache.resync(ib)
        # Detach first. ib_async keeps handlers on the IB object across a reconnect, so subscribing again
        # without this leaves every handler registered twice — and each reconnect adds another copy, so one
        # fill is applied to the position cache N times.
        await self._unsubscribe(ib)
        await self._subscribe(ib)

    # ---- Event handlers ----

    async def _on_bar_close(self, completed_bar: dict) -> None:
        """SEALED IB seam — the un-bypassable per-bar inference-log heartbeat (§5.1, design A).

        Mirrors the sync ``BaseStrategy.run()`` template for the event-driven IB path: derive
        ``asof_bar_ts`` from the framework BarAggregator boundary, wrap ``on_new_bar()``, emit the
        Tier 0 ``bar`` record (+ best-effort Tier 1), then drive the IB circuit breaker. Unlike MT5
        it does NOT re-raise (no MultiTrader above it) — the breaker is the bookkeeping target.
        """
        if completed_bar.get("partial", False):
            logger.debug(f"Skipping partial bar for {self.strategy_config.symbol}")
            return
        asof_bar_ts = completed_bar["time"]  # framework-derived UTC bar boundary (not developer data)
        # The disabled-check, the cycle, and the breaker update all run UNDER the bar lock so the
        # breaker state is read+written atomically w.r.t. the cycle. Otherwise two queued bar callbacks
        # could both pass an outside-the-lock enabled-check and the second would run on_new_bar() after
        # the first already tripped the breaker.
        async with self._bar_lock:
            if not self._ib_health.is_enabled:
                # Circuit-broken: skip the cycle but emit a skipped heartbeat so ops sees why this
                # system went quiet while siblings keep trading (§7.3).
                self._emit_bar_record(asof_bar_ts=asof_bar_ts, outcome=BarOutcome.SKIPPED_DISABLED)
                return
            error: Optional[Exception] = None
            try:
                self._append_bar(completed_bar)
                self.latest_run_dt = completed_bar["time"]
                await self.on_new_bar()
            except Exception as e:
                error = e
                logger.exception(f"Error in on_new_bar for {self.strategy_config.symbol}: {e}")
                if self.notifier:
                    self.notifier.on_error(self.strategy_config.name, str(e))
            self._emit_bar_record(asof_bar_ts=asof_bar_ts,
                                  outcome=BarOutcome.ERROR if error is not None else BarOutcome.OK)
            self._record_bar_health(error)

    def _record_bar_health(self, error: Optional[Exception]) -> None:
        """Drive the IB circuit breaker after a bar, mirroring MultiTrader.run() semantics (§7.3)."""
        was_enabled = self._ib_health.is_enabled
        if error is None:
            self._ib_health.record_success(0.0)
            return
        self._ib_health.record_error(0.0)
        if was_enabled and not self._ib_health.is_enabled:  # just tripped
            self.emit_circuit_breaker_tripped(consecutive_errors=self._ib_health.consecutive_errors,
                                              last_error=str(error))
            if self.notifier:
                self.notifier.on_circuit_breaker_tripped(self.strategy_config.name,
                                                         self._ib_health.consecutive_errors)

    def reenable(self) -> None:
        """Re-enable a circuit-broken strategy and emit strategy_reenabled (parity with MT5, §7.3)."""
        if not self._ib_health.is_enabled:
            self._ib_health.enable()
            self.emit_strategy_reenabled(reason="manual")

    def _append_bar(self, completed_bar: dict) -> None:
        """Append a single completed OHLCV bar to the PriceBuffer.

        Uses ``datetime.now(tz=utc)`` rather than the bar's own start timestamp:
        ``PriceBuffer.update`` filters out any bar at or beyond the wall-clock
        forming bar, and a completed bar emitted at its own start time would be
        rejected as not-yet-closed. By the time ``_emit`` fires we are already
        inside the next 5-second period, so wall clock is a safe ceiling.
        """
        ts = completed_bar["time"]
        df = pd.DataFrame(
            {
                "open": [completed_bar["open"]],
                "high": [completed_bar["high"]],
                "low": [completed_bar["low"]],
                "close": [completed_bar["close"]],
                "volume": [completed_bar["volume"]],
            },
            index=pd.DatetimeIndex([ts], name="date"),
        )
        self.price_buffer.update(df, datetime.now(tz=timezone.utc))

    def _on_fill(self, trade, fill) -> None:
        is_partial = trade.orderStatus.status == "PartiallyFilled"
        logger.info(
            f"{'Partial fill' if is_partial else 'Fill'}: "
            f"{fill.execution.side} {fill.execution.shares} @ {fill.execution.price} "
            f"(order {trade.order.orderId}, ref {fill.execution.orderRef})"
        )
        if self.notifier and not is_partial:
            self.notifier.on_trade_filled(
                symbol=self.strategy_config.symbol,
                order_id=trade.order.orderId,
                qty=trade.orderStatus.filled,
                avg_price=trade.orderStatus.avgFillPrice,
            )

    def _on_position_fill(self, trade, fill) -> None:
        """Close detection for the event-driven path.

        The polled MT5 path notices a close by missing the position from a sweep; here IB tells us. Both funnel
        into the same ``observe_open_positions`` diff, so attribution, announcement and the once-only guarantee
        are shared — only the trigger differs, because forcing IB to poll for something it already pushes would
        be strictly worse.

        LIMITATION: a reversal (long 100, sell 150 -> short 50) never takes the cached position to zero, so no
        close is reported for the leg that ended. IB netting makes that one position, not two, and unpicking it
        would mean inventing a close the broker never described.
        """
        if self._position_cache is None:
            return
        try:
            # fillEvent is connection-wide, not per-strategy. Filter exactly as the cache does (orderRef AND
            # conId) so another strategy's fill on the same connection neither overwrites our stored fill nor
            # attributes its execution to our position.
            if (fill.execution.orderRef != self._position_cache.order_ref
                    or fill.contract.conId != self._position_cache.con_id):
                return
            # Both are kept: the realised P/L is on the fill, but the order type that produced it is on the
            # TRADE. ib_async's Fill is (contract, execution, commissionReport, time) — it has no .order.
            self._last_fills[str(fill.contract.conId)] = (trade, fill)
            self.observe_open_positions(self.get_open_positions())
        except Exception:
            logger.exception("Closed-position reconciliation failed on fill")

    def resolve_closed_trade(self, key: str, last_seen: dict) -> Optional[ClosedTrade]:
        """Describe a flattened position from the fill that flattened it.

        IB books realised P/L on the commission report attached to the fill, so nothing has to be queried — the
        payload is already in hand, which is why the shared handler takes a finished ClosedTrade rather than
        calling back into the broker (that call would have to be awaitable here and plain on MT5).
        """
        entry = self._last_fills.pop(key, None)
        if entry is None:
            return None
        trade, fill = entry
        realised = self._realised_pnl(getattr(fill, "commissionReport", None))
        commission = self._finite(getattr(getattr(fill, "commissionReport", None), "commission", None))
        return ClosedTrade(
            key=key, symbol=self.strategy_config.symbol, magic=getattr(self.strategy_config, "magic", None),
            reason=self._infer_close_reason(trade), volume=abs(float(fill.execution.shares)),
            entry_price=float(last_seen.get("avg_cost") or 0.0), exit_price=float(fill.execution.price),
            profit=realised if realised is not None else 0.0,
            commission=commission if commission is not None else 0.0,
            closed_at=getattr(fill, "time", None), last_seen=last_seen,
            resolved=realised is not None)

    #: IB's "this field was never populated" sentinel. It arrives as a real float, so an unguarded read reports
    #: a profit of 1.8e308 — a number that poisons every downstream sum rather than failing visibly.
    _UNSET_DOUBLE = 1.7976931348623157e+308

    @classmethod
    def _finite(cls, value) -> Optional[float]:
        """A usable float, or None if IB left the field unset."""
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return None if abs(value) >= cls._UNSET_DOUBLE else value

    @classmethod
    def _realised_pnl(cls, report) -> Optional[float]:
        """Realised P/L from a commission report, or None when IB did not supply one.

        ``CommissionReport.realizedPNL`` defaults to ``0.0`` in ib_async and is set to ``UNSET_DOUBLE`` when the
        broker does not report it, so neither a plain ``is not None`` nor a truthiness test distinguishes "flat
        trade" from "not reported". A missing report means the trade is described but marked UNRESOLVED, which
        is honest; reporting 0.0 as if it were realised would silently understate the record.
        """
        if report is None:
            return None
        return cls._finite(getattr(report, "realizedPNL", None))

    @staticmethod
    def _infer_close_reason(trade) -> CloseReason:
        """Best-effort cause from the ORDER that produced the fill.

        IB does not label an execution the way MT5 labels a deal, so this reads the bracket child's order type:
        the protective leg is a stop, the target leg a limit. It is a heuristic and says so — a strategy-recorded
        intent refines it in ``_build_closed_trade``.
        """
        order_type = str(getattr(getattr(trade, "order", None), "orderType", "") or "").upper()
        if order_type.startswith("STP"):
            return CloseReason.STOP_LOSS
        if order_type == "LMT":
            return CloseReason.TAKE_PROFIT
        return CloseReason.UNKNOWN

    def _on_error(self, reqId, code, msg, _advanced) -> None:
        cls = classify_ib_error(code)
        if cls == ErrorClass.WARNING:
            logger.warning(f"IB warning {code}: {msg}")
        elif cls == ErrorClass.TRANSIENT:
            logger.error(f"IB transient {code}: {msg}")
        elif cls == ErrorClass.PERMANENT:
            logger.error(f"IB permanent {code}: {msg}")
            if code == 201 and reqId in self._bracket_trades:
                asyncio.ensure_future(self._on_bracket_rejection(reqId))
        else:
            logger.error(f"IB unknown {code}: {msg}")

    async def _on_bracket_rejection(self, rejected_order_id: int) -> None:
        """Cancel remaining live legs when a bracket leg is asynchronously rejected."""
        trades = self._bracket_trades.get(rejected_order_id, [])
        for t in trades:
            self._bracket_trades.pop(t.order.orderId, None)
        logger.error(
            f"Bracket rejection for orderId={rejected_order_id} — cancelling {len(trades)} sibling leg(s)"
        )
        for t in trades:
            if t.orderStatus.status not in ("Filled", "Cancelled", "ApiCancelled", "Inactive"):
                try:
                    self.ib.cancelOrder(t.order)
                except Exception as e:
                    logger.warning(
                        f"Could not cancel bracket sibling orderId={t.order.orderId}: {e}"
                    )

    # ---- BaseStrategy ABC ----

    def is_new_bar(self, run_dt) -> bool:
        """Event-driven — bar boundary detected by BarAggregator, not polled."""
        return False

    @abstractmethod
    async def on_new_bar(self):
        """Read PriceBuffer, generate signal, manage positions, place orders."""

    # ---- Convenience API ----

    def fetch_price_bars(self):
        return self.price_buffer.get_data()

    def get_open_positions(self) -> list[dict]:
        return self._position_cache.get_open() if self._position_cache else []

    def _current_tick_info(self) -> Optional[dict]:
        if self._ticker is None:
            return None
        return {
            "bid": self._ticker.bid or 0.0,
            "ask": self._ticker.ask or 0.0,
            "last": self._ticker.last or 0.0,
            "bid_size": self._ticker.bidSize or 0.0,
            "ask_size": self._ticker.askSize or 0.0,
        }

    def _notify_trade_failed(self, direction: str, reason: str, retcode: int = None) -> None:
        if not self.notifier:
            return
        self.notifier.on_trade_failed(
            symbol=self.strategy_config.symbol, direction=direction, reason=reason,
            retcode=retcode, context={"strategy_name": self.strategy_config.name},
        )

    async def open_position(self, action: str, quantity: Optional[float] = None) -> bool:
        qty = quantity if quantity is not None else self.calculate_quantity()
        try:
            await place_market_order(self.ib, self.contract, action, qty,
                                     self.strategy_config.magic)
            return True
        except (IBTransientError, IBConnectionError) as e:
            logger.error(f"Failed to open position after retries: {e}")
            self._notify_trade_failed(action, str(e), getattr(e, "code", None))
            return False
        except IBPermanentError as e:
            logger.error(f"Failed to open position (permanent): {e}")
            self._notify_trade_failed(action, str(e), e.code)
            return False

    async def place_order(self, order_type: Union[str, OrderType], price: float = 0.0,
                          sl: float = 0.0, tp: float = 0.0,
                          quantity: Optional[float] = None) -> bool:
        ot = order_type.value if isinstance(order_type, OrderType) else order_type
        ot = ot.lower()
        qty = quantity if quantity is not None else self.calculate_quantity()

        try:
            if sl or tp:
                raise NotImplementedError(
                    "place_order() does not attach SL/TP. Use place_bracket() for "
                    "atomic entry+TP+SL, or a position manager for post-fill stops."
                )
            if ot in ("buy", "sell"):
                await place_market_order(self.ib, self.contract, ot.upper(), qty,
                                         self.strategy_config.magic)
            elif ot in ("buy_limit", "sell_limit"):
                await place_limit_order(self.ib, self.contract, ot.split("_")[0].upper(),
                                        qty, price, self.strategy_config.magic)
            elif ot in ("buy_stop", "sell_stop"):
                await place_stop_order(self.ib, self.contract, ot.split("_")[0].upper(),
                                       qty, price, self.strategy_config.magic)
            else:
                raise ValueError(f"Invalid order_type: {order_type}")
            return True
        except (IBTransientError, IBConnectionError) as e:
            logger.error(f"Failed to place {order_type}: {e}")
            self._notify_trade_failed(ot, str(e), getattr(e, "code", None))
            return False
        except IBPermanentError as e:
            logger.error(f"Failed to place {order_type} (permanent): {e}")
            self._notify_trade_failed(ot, str(e), e.code)
            return False

    async def close_position(self, position: dict, reason: str = "strategy_close") -> bool:
        """Close a position. ``reason`` is recorded as intent, not announced: the fill that actually flattens
        the position is what reports it, so the announcement carries the realised fill instead of a guess."""
        key = self._position_key(position)
        try:
            if key is not None:
                self.note_close_intent(key, reason)
            await ib_close_position(self.ib, position, self.strategy_config.magic)
            return True
        except (IBTransientError, IBConnectionError) as e:
            logger.error(f"Close failed after retries: {e}")
            self._notify_trade_failed("CLOSE", str(e), getattr(e, "code", None))
        except IBPermanentError as e:
            logger.error(f"Close failed (permanent): {e}")
            self._notify_trade_failed("CLOSE", str(e), e.code)
        # The close did not go through: withdraw the intent so it cannot outlive this call and relabel a
        # later close by someone else as ours.
        if key is not None:
            self.clear_close_intent(key)
        return False

    async def place_bracket(self, action: str, take_profit: float, stop_loss: float,
                            limit_price: Optional[float] = None,
                            quantity: Optional[float] = None) -> list:
        """Submit an atomic bracket (entry + TP + SL). Returns three Trades, or [] on failure."""
        qty = quantity if quantity is not None else self.calculate_quantity()
        try:
            trades = await place_bracket_order(
                self.ib, self.contract, action, qty, limit_price,
                take_profit, stop_loss, self.strategy_config.magic,
            )
            self._register_bracket(trades)
            return trades
        except (IBTransientError, IBConnectionError) as e:
            logger.error(f"Bracket order failed: {e}")
            self._notify_trade_failed(f"BRACKET {action}", str(e), getattr(e, "code", None))
            return []
        except IBPermanentError as e:
            logger.error(f"Bracket order rejected (permanent): {e}")
            self._notify_trade_failed(f"BRACKET {action}", str(e), e.code)
            return []

    def _register_bracket(self, trades: list) -> None:
        for t in trades:
            self._bracket_trades[t.order.orderId] = trades

    def calculate_quantity(self) -> float:
        """Compute instrument-aware size, rounded to ``size_increment``."""
        sizing = self.strategy_config.position_sizing
        if sizing.type is PositionSizingType.FIXED:
            assert sizing.units is not None
            raw = sizing.units
        else:
            raise NotImplementedError(
                f"Position sizing '{sizing.type.value}' is declared but not implemented "
                f"for IB. Override calculate_quantity in a subclass."
            )

        increment = self.contract_info.get("size_increment", 1.0)
        min_size = self.contract_info.get("min_size", 1.0)
        return max(min_size, round(raw / increment) * increment)


class GenericBasicIBStrategy(BaseIBStrategy):
    """Event-driven counterpart of MT5 GenericBasicStrategy."""

    async def on_new_bar(self):
        _symbol = self.strategy_config.symbol
        _magic = self.strategy_config.magic
        logger.info(
            f"GenericBasicIBStrategy ({self.strategy_config}) on_new_bar @ {self.latest_run_dt}"
        )

        # Exceptions propagate to the sealed _on_bar_close seam, which records the outcome=error
        # heartbeat, notifies, and drives the IB circuit breaker — so this body must NOT swallow them.
        price_bars = self.fetch_price_bars()
        if price_bars is None or len(price_bars) < self.strategy_config.bars_to_copy:
            logger.warning(f"Insufficient price data for {_symbol}")
            return

        entries_long, exits_long, entries_short, exits_short = self.signal_generator.generate(price_bars)
        entries_long, exits_long, entries_short, exits_short = (
            entries_long[-1].item(), exits_long[-1].item(),
            entries_short[-1].item(), exits_short[-1].item(),
        )
        logger.info(
            f"Signal {_symbol}({_magic}): "
            f"L={entries_long},{exits_long} S={entries_short},{exits_short}"
        )

        positions = self.get_open_positions()
        managed_closing_con_ids: set[int] = set()

        if self.position_manager and positions:
            managed_closing_con_ids = await self.position_manager.manage_positions(positions)
            positions = self.get_open_positions()

        signal_closed_con_ids: set[int] = set()

        if positions and (exits_long or exits_short):
            for pos in positions:
                if pos["contract"].conId in managed_closing_con_ids:
                    logger.info(
                        f"Skipping signal close for {_symbol}: manager already submitted "
                        f"a close for conId={pos['contract'].conId}"
                    )
                    continue
                is_long = pos["position"] > 0
                is_short = pos["position"] < 0
                if (exits_long and is_long) or (exits_short and is_short):
                    if self.position_manager:
                        stop_cancelled = await self.position_manager.cancel_protective_stop(pos)
                        if not stop_cancelled:
                            logger.critical(
                                f"Aborting close for {_symbol}: protective stop could not be "
                                "confirmed cancelled. Will retry on next bar."
                            )
                            continue
                    submitted = await self.close_position(pos)
                    if submitted:
                        signal_closed_con_ids.add(pos["contract"].conId)
                        logger.info(f"Close order submitted for {_symbol} position {pos['position']}")
                    else:
                        logger.error(f"Close order failed for {_symbol} position {pos['position']}")

        all_closing_con_ids = managed_closing_con_ids | signal_closed_con_ids
        if all_closing_con_ids:
            logger.info(
                f"Skipping entry for {_symbol}: close submitted on this bar for conIds={all_closing_con_ids}"
            )
            return

        if entries_long and entries_short:
            logger.warning(
                f"Ambiguous signal for {_symbol}: both long and short entry on same bar — skipping entry"
            )
            return

        if entries_long or entries_short:
            direction = "BUY" if entries_long else "SELL"
            positions = self.get_open_positions()
            pending = get_pending_orders(
                self.ib, _symbol, _magic,
                con_id=self.contract.conId, action_filter=direction,
            )
            if len(positions) + len(pending) >= self.max_number_of_open_positions:
                logger.info(
                    f"Entry signal ignored — max positions ({len(positions)} open + "
                    f"{len(pending)} pending) reached for {_symbol}"
                )
                return

            filter_context = {
                "datetime": self.latest_run_dt,
                "contract_info": self.contract_info,
                "open_positions": len(positions),
                "signal_type": "long" if entries_long else "short",
                "tick_info": self._current_tick_info(),
            }
            if not self.filter_chain(filter_context):
                logger.info(f"Filter chain blocked entry for {_symbol}")
                return

            submitted = await self.open_position(direction)
            if submitted:
                logger.info(f"Entry order submitted for {_symbol} direction={direction}")
            else:
                logger.error(f"Entry order failed for {_symbol} direction={direction}")
