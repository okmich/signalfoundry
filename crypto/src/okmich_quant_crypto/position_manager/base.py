"""Crypto position managers: MT5's semantics on IB's async shape.

Management runs as ``manage_positions(positions, apply_levels, close)``: each manager evaluates a position to
``(should_close, new_sl, new_tp)`` and the strategy's stop controller applies the levels - natively on the venue or as
managed levels - so a manager never needs to know which stop mode is active.

The level logic mirrors the MT5 managers it is named after:

* FIXED - set the initial SL/TP once, from the entry price;
* WITH_TRAILING - once in profit, trail the SL behind the current price (never loosening it);
* WITH_BREAK_EVEN - once price has moved ``break_even`` in favour, move the SL to max(entry, trailing level);
* DYNAMIC - trail only once the trailing level is beyond the entry.

Differences from MT5, on purpose: an unset level is ``None`` (MT5 uses 0.0, which breaks the comparisons for shorts),
and the initial levels are also exposed as :meth:`initial_levels` so risk-based sizing can know the stop distance
BEFORE the entry is sent.
"""
import logging
from abc import abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from okmich_quant_core import BasePositionManager, PositionManagerType, StrategyConfig

logger = logging.getLogger(__name__)

ApplyLevels = Callable[[dict, Optional[float], Optional[float]], Awaitable[bool]]
ClosePosition = Callable[[dict, str], Awaitable[bool]]
Levels = tuple[Optional[float], Optional[float]]

_TRAILING = {PositionManagerType.FIXED_POINT_WITH_TRAILING, PositionManagerType.FIXED_PERCENT_WITH_TRAILING,
             PositionManagerType.FIXED_ATR_WITH_TRAILING}
_BREAK_EVEN = {PositionManagerType.FIXED_POINT_WITH_BREAK_EVEN, PositionManagerType.FIXED_PERCENT_WITH_BREAK_EVEN,
               PositionManagerType.FIXED_ATR_WITH_BREAK_EVEN}
_DYNAMIC = {PositionManagerType.DYNAMIC_POINT, PositionManagerType.DYNAMIC_PERCENT, PositionManagerType.DYNAMIC_ATR}


class BaseCryptoPositionManager(BasePositionManager):
    """Async manager base. Positions come from the strategy (venue position or spot ledger), not from here."""

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config)
        self.strategy_config = strategy_config
        self.price_buffer = kwargs.get("price_buffer")
        self.kwargs = kwargs

    @abstractmethod
    def _evaluate_position(self, position: dict) -> tuple[bool, Optional[float], Optional[float]]:
        """Return ``(should_close, new_sl, new_tp)``; ``None`` leaves a level unchanged."""

    def initial_levels(self, price_open: float, is_long: bool, quantity: Optional[float] = None) -> Levels:
        """The SL/TP this manager would set on a fresh position entered at ``price_open``."""
        return None, None

    async def manage_positions(self, positions: list[dict], apply_levels: ApplyLevels,
                               close: ClosePosition) -> set[str]:
        """Evaluate every position; returns the position ids a close was submitted for."""
        closing: set[str] = set()
        for position in positions:
            try:
                should_close, sl, tp = self._evaluate_position(position)
            except Exception:
                logger.exception("%s: evaluating position %s failed", self.symbol, position.get("position_id"))
                continue
            if should_close:
                if await close(position, "position_manager"):
                    closing.add(position["position_id"])
            elif sl is not None or tp is not None:
                new_sl = sl if sl is not None else position.get("stop_loss")
                new_tp = tp if tp is not None else position.get("take_profit")
                await apply_levels(position, new_sl, new_tp)
        return closing

    def _latest_close(self) -> Optional[float]:
        if self.price_buffer is None or self.price_buffer.is_empty():
            return None
        return float(self.price_buffer.get_data()["close"].iloc[-1])

    # ---- core BasePositionManager abstracts: the polled MT5 API is not used here ----
    def get_open_positions(self) -> List[Dict[str, Any]]:
        return []

    def close_position(self, position: Dict[str, Any]) -> bool:
        raise NotImplementedError("crypto position managers close through manage_positions(..., close)")

    def modify_position(self, position: Dict[str, Any], sl: float = None, tp: float = None) -> bool:
        raise NotImplementedError("crypto position managers set levels through manage_positions(..., apply_levels)")

    def manage_long_position(self, position: Dict[str, Any], flag: bool):
        raise NotImplementedError("crypto position managers dispatch via manage_positions(positions, ...)")

    def manage_short_position(self, position: Dict[str, Any], flag: bool):
        raise NotImplementedError("crypto position managers dispatch via manage_positions(positions, ...)")


class DistanceBasedPositionManager(BaseCryptoPositionManager):
    """Shared FIXED / TRAILING / BREAK_EVEN / DYNAMIC logic; subclasses define what a distance unit is."""

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        cfg = strategy_config.position_manager
        self.manager_type = cfg.type
        self.sl_units = cfg.sl or 0.0
        self.tp_units = cfg.tp or 0.0
        self.trailing_units = cfg.trailing
        self.break_even_units = cfg.break_even
        needs_trailing = self.manager_type in _TRAILING | _BREAK_EVEN | _DYNAMIC
        if needs_trailing and self.trailing_units is None:
            raise ValueError(f"{type(self).__name__} ({self.manager_type.value}) requires 'trailing'")
        if self.manager_type in _BREAK_EVEN and self.break_even_units is None:
            raise ValueError(f"{type(self).__name__} ({self.manager_type.value}) requires 'break_even'")

    @abstractmethod
    def _distance(self, units: float, reference_price: float) -> Optional[float]:
        """Price distance of ``units`` (points, percent of ``reference_price``, or ATR multiples)."""

    def initial_levels(self, price_open: float, is_long: bool, quantity: Optional[float] = None) -> Levels:
        sl = tp = None
        if self.sl_units > 0:
            d = self._distance(self.sl_units, price_open)
            if d is not None:
                sl = price_open - d if is_long else price_open + d
        if self.tp_units > 0:
            d = self._distance(self.tp_units, price_open)
            if d is not None:
                tp = price_open + d if is_long else price_open - d
        if sl is not None and sl <= 0:
            logger.error("%s: computed SL %s <= 0; SL will not be set", self.symbol, sl)
            sl = None
        if tp is not None and tp <= 0:
            logger.error("%s: computed TP %s <= 0; TP will not be set", self.symbol, tp)
            tp = None
        return sl, tp

    def _evaluate_position(self, position: dict) -> tuple[bool, Optional[float], Optional[float]]:
        is_long = position["position"] > 0
        price_open = float(position["price_open"])
        price_current = float(position.get("price_current") or price_open)
        sl, tp = position.get("stop_loss"), position.get("take_profit")
        new_sl = new_tp = None

        init_sl, init_tp = self.initial_levels(price_open, is_long, abs(position["position"]))
        if sl is None and init_sl is not None:
            new_sl = sl = init_sl
        if tp is None and init_tp is not None:
            new_tp = tp = init_tp

        candidate = self._trailing_candidate(is_long, price_open, price_current, sl)
        if candidate is not None and (sl is None or (candidate > sl if is_long else candidate < sl)):
            new_sl = candidate
        return False, new_sl, new_tp

    def _trailing_candidate(self, is_long: bool, price_open: float, price_current: float,
                            sl: Optional[float]) -> Optional[float]:
        if self.manager_type in _TRAILING:
            in_profit = price_current > price_open if is_long else price_current < price_open
            if not in_profit:
                return None
            d = self._distance(self.trailing_units, price_current)
            return None if d is None else (price_current - d if is_long else price_current + d)
        if self.manager_type in _BREAK_EVEN:
            threshold = self._distance(self.break_even_units, price_open)
            d = self._distance(self.trailing_units, price_current)
            if threshold is None or d is None:
                return None
            if is_long:
                reached = price_current >= price_open + threshold and (sl is None or sl < price_open)
                return max(price_current - d, price_open) if reached else None
            reached = price_current <= price_open - threshold and (sl is None or sl > price_open)
            return min(price_current + d, price_open) if reached else None
        if self.manager_type in _DYNAMIC:
            in_profit = price_current > price_open if is_long else price_current < price_open
            d = self._distance(self.trailing_units, price_current)
            if not in_profit or d is None:
                return None
            candidate = price_current - d if is_long else price_current + d
            beyond_entry = candidate > price_open if is_long else candidate < price_open
            return candidate if beyond_entry else None
        return None


class MaxLossAmountPositionManager(BaseCryptoPositionManager):
    """Close when the unrealised loss (quote currency) reaches ``max_loss_amount``."""

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.max_loss_amount = strategy_config.position_manager.max_loss_amount or 0.0

    def _evaluate_position(self, position: dict) -> tuple[bool, Optional[float], Optional[float]]:
        price = position.get("price_current") or self._latest_close()
        avg = float(position.get("avg_cost") or 0.0)
        qty = float(position["position"])
        if self.max_loss_amount <= 0 or not price or avg <= 0 or qty == 0:
            return False, None, None
        pnl = (float(price) - avg) * qty
        if pnl < 0 and abs(pnl) >= self.max_loss_amount:
            logger.info("%s: max-loss close, loss %.2f >= %.2f", self.symbol, abs(pnl), self.max_loss_amount)
            return True, None, None
        return False, None, None


class MaxLossStopLossPositionManager(BaseCryptoPositionManager):
    """Set a stop once, at the price where the loss would equal ``max_loss_amount`` (quote currency).

    distance = max_loss_amount / |quantity in base units|; linear contracts make one base unit worth one quote unit
    per unit of price, so no tick value is needed (unlike MT5).
    """

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.max_loss_amount = strategy_config.position_manager.max_loss_amount or 0.0

    def initial_levels(self, price_open: float, is_long: bool, quantity: Optional[float] = None) -> Levels:
        if not quantity or self.max_loss_amount <= 0:
            return None, None
        distance = self.max_loss_amount / abs(quantity)
        sl = price_open - distance if is_long else price_open + distance
        if sl <= 0:
            logger.error("%s: max-loss SL %s <= 0; max_loss_amount too large for this size", self.symbol, sl)
            return None, None
        return sl, None

    def _evaluate_position(self, position: dict) -> tuple[bool, Optional[float], Optional[float]]:
        if position.get("stop_loss") is not None:
            return False, None, None
        sl, _ = self.initial_levels(float(position["price_open"]), position["position"] > 0, abs(position["position"]))
        return False, sl, None
