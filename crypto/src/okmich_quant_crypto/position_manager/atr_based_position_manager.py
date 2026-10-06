"""ATR-based managers: distances are ``units * ATR``.

ATR is Wilder's (the TA-Lib ``ATR`` the MT5 managers use): seeded with the mean of the first ``period`` true ranges,
then ``ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period``. It is computed on CLOSED bars from the strategy's price
buffer only; MT5 includes the forming bar, which makes the stop depend on intra-bar noise.
"""
from typing import Optional

import numpy as np

from okmich_quant_core import StrategyConfig

from .base import DistanceBasedPositionManager


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Optional[float]:
    if period <= 0 or len(close) < period + 1:
        return None
    prev_close = close[:-1]
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)])
    atr = float(np.mean(tr[:period]))
    for value in tr[period:]:
        atr = (atr * (period - 1) + float(value)) / period
    return atr if np.isfinite(atr) and atr > 0 else None


class AtrBasedPositionManager(DistanceBasedPositionManager):

    def __init__(self, strategy_config: StrategyConfig, **kwargs):
        super().__init__(strategy_config, **kwargs)
        self.atr_period = strategy_config.position_manager.atr_period

    def current_atr(self) -> Optional[float]:
        if self.price_buffer is None or self.price_buffer.is_empty():
            return None
        df = self.price_buffer.get_data()
        return wilder_atr(df["high"].to_numpy(dtype=float), df["low"].to_numpy(dtype=float),
                          df["close"].to_numpy(dtype=float), self.atr_period)

    def _distance(self, units: float, reference_price: float) -> Optional[float]:
        atr = self.current_atr()
        return None if atr is None else units * atr
