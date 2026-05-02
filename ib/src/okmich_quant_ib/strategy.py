"""IB strategy lifecycle (bootstrap / subscribe / unsubscribe / resubscribe) and
the ``GenericBasicIBStrategy`` execution loop fired on completed-bar events.
"""
import asyncio
import logging
import traceback
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Optional, Union

from ib_async import IB, Contract

from okmich_quant_core import BaseSignal, BaseStrategy, OrderType, PositionSizingType, StrategyConfig
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
    def __init__(self, config: StrategyConfig, signal: BaseSignal, contract_cfg: IBContractConfig,
                 notifier: Optional[BaseNotifier] = None, **kwargs):
        super().__init__(config, signal, notifier, **kwargs)
        self.contract_cfg = contract_cfg
        self.max_number_of_open_positions = config.max_number_of_open_positions

        self.ib: Optional[IB] = None
        self.contract: Optional[Contract] = None
        self.contract_info: dict = {}
        self._position_cache: Optional[IBPositionCache] = None
        self._bar_aggregator: Optional[BarAggregator] = None
        self._rt_bars = None
        self._ticker = None

        _tf_min = bar_size_to_minutes(config.timeframe)
        if _tf_min >= 1440:
            raise ValueError(
                f"Timeframe '{config.timeframe}' is not supported in real-time bar mode. "
                "Daily bars cannot be assembled from 5-second real-time bars; override "
                "_bootstrap and subscribe via reqHistoricalDataAsync polling instead."
            )
        self.price_buffer = PriceBuffer(
            symbol=config.symbol,
            timeframe=config.timeframe,
            buffer_size=config.bars_to_copy,
            exclude_columns=getattr(config, "exclude_columns", None),
            timeframe_minutes=_tf_min,
        )
        self.filter_chain = create_filter(config)
        self.position_manager = None
        self._bracket_trades: dict[int, list] = {}
        self._bar_lock = asyncio.Lock()

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
        await self._subscribe(ib)

    # ---- Event handlers ----

    async def _on_bar_close(self, completed_bar: dict) -> None:
        if completed_bar.get("partial", False):
            logger.debug(f"Skipping partial bar for {self.strategy_config.symbol}")
            return
        async with self._bar_lock:
            try:
                self._append_bar(completed_bar)
                self.latest_run_dt = completed_bar["time"]
                await self.on_new_bar()
            except Exception as e:
                logger.exception(f"Error in on_new_bar for {self.strategy_config.symbol}: {e}")
                if self.notifier:
                    self.notifier.on_error(self.strategy_config.name, str(e))

    def _append_bar(self, completed_bar: dict) -> None:
        """Append a single completed OHLCV bar to the PriceBuffer.

        Uses ``datetime.now(tz=utc)`` rather than the bar's own start timestamp:
        ``PriceBuffer.update`` filters out any bar at or beyond the wall-clock
        forming bar, and a completed bar emitted at its own start time would be
        rejected as not-yet-closed. By the time ``_emit`` fires we are already
        inside the next 5-second period, so wall clock is a safe ceiling.
        """
        import pandas as pd
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

    async def open_position(self, action: str, quantity: Optional[float] = None) -> bool:
        qty = quantity if quantity is not None else self.calculate_quantity()
        try:
            await place_market_order(self.ib, self.contract, action, qty,
                                     self.strategy_config.magic)
            return True
        except (IBTransientError, IBConnectionError) as e:
            logger.error(f"Failed to open position after retries: {e}")
            return False
        except IBPermanentError as e:
            logger.error(f"Failed to open position (permanent): {e}")
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
            return False
        except IBPermanentError as e:
            logger.error(f"Failed to place {order_type} (permanent): {e}")
            return False

    async def close_position(self, position: dict) -> bool:
        try:
            await ib_close_position(self.ib, position, self.strategy_config.magic)
            return True
        except (IBTransientError, IBConnectionError) as e:
            logger.error(f"Close failed after retries: {e}")
            return False
        except IBPermanentError as e:
            logger.error(f"Close failed (permanent): {e}")
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
            return []
        except IBPermanentError as e:
            logger.error(f"Bracket order rejected (permanent): {e}")
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

        try:
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
        except Exception as e:
            traceback.print_exc()
            if self.notifier:
                self.notifier.on_error(self.strategy_config.name, str(e))
