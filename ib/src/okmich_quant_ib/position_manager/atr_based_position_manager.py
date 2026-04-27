"""ATR-trailing IB position manager.

Reads recent OHLC from a PriceBuffer (passed via kwargs) to compute the rolling
ATR, then trails the protective stop at ``sl_multiplier * ATR`` from the
current close. Never widens the stop in the unfavorable direction.
"""
import logging
from typing import Optional

from ib_async import IB, Contract

from okmich_quant_core import StrategyConfig

from .base import BaseIBPositionManager

logger = logging.getLogger(__name__)


class ATRBasedIBPositionManager(BaseIBPositionManager):
    """Trail SL at ``sl_multiplier × ATR`` from the latest close.

    Required ``StrategyConfig.position_manager`` fields:
        sl          — ATR multiplier for the stop distance.
        atr_period  — lookback for the ATR calculation (e.g. 14).

    Optional kwargs:
        price_buffer — okmich_quant_core.price_buffer.PriceBuffer reference
                       (provided by BaseIBStrategy._bootstrap). Required at
                       runtime; raises if missing.
    """

    def __init__(self, ib: IB, contract: Contract, strategy_config: StrategyConfig, **kwargs):
        super().__init__(ib, contract, strategy_config, **kwargs)
        cfg = strategy_config.position_manager
        self.sl_multiplier = cfg.sl if cfg.sl else 0.0
        self.atr_period = cfg.atr_period
        self.price_buffer = kwargs.get("price_buffer")
        if self.price_buffer is None:
            logger.warning(
                f"ATRBasedIBPositionManager for {self.symbol}: no price_buffer provided; "
                "ATR-based trailing will be a no-op until one is supplied."
            )

    def _current_atr(self) -> Optional[float]:
        if self.price_buffer is None or self.price_buffer.is_empty():
            return None
        df = self.price_buffer.get_data()
        if len(df) < self.atr_period + 1:
            return None
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close_prev = df["close"].astype(float).shift(1)
        tr = (high - low).combine((high - close_prev).abs(), max).combine(
            (low - close_prev).abs(), max
        )
        atr = tr.rolling(self.atr_period).mean().iloc[-1]
        if atr is None or atr != atr:  # NaN guard
            return None
        return float(atr)

    def _latest_close(self) -> Optional[float]:
        if self.price_buffer is None or self.price_buffer.is_empty():
            return None
        return float(self.price_buffer.get_data()["close"].iloc[-1])

    def _evaluate_position(
        self, position: dict
    ) -> tuple[bool, Optional[float], Optional[float]]:
        close = self._latest_close()
        atr = self._current_atr()
        if close is None or atr is None or self.sl_multiplier <= 0:
            return False, None, None

        is_long = position["position"] > 0
        distance = self.sl_multiplier * atr
        candidate_sl = close - distance if is_long else close + distance

        existing_trade = self._sl_trades.get(position["contract"].conId)
        if existing_trade is not None:
            current_stop = existing_trade.order.auxPrice
            # Only trail in the favorable direction.
            if is_long and candidate_sl <= current_stop:
                return False, None, None
            if (not is_long) and candidate_sl >= current_stop:
                return False, None, None

        return False, candidate_sl, None
