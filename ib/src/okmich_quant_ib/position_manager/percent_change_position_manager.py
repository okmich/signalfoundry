"""Close on cumulative P&L percentage threshold (loss or gain)."""
import logging
from typing import Optional

from ib_async import IB, Contract

from okmich_quant_core import StrategyConfig

from .base import BaseIBPositionManager

logger = logging.getLogger(__name__)


class PercentChangeIBPositionManager(BaseIBPositionManager):
    """Close when realised+unrealised P&L percent of cost basis crosses thresholds.

    Required ``StrategyConfig.position_manager`` fields:
        sl — loss threshold percent (e.g. 0.02 = 2% loss).
        tp — profit threshold percent (e.g. 0.05 = 5% gain).

    Optional kwargs:
        price_buffer — used to read the latest mark price for unrealised P&L.
    """

    def __init__(self, ib: IB, contract: Contract, strategy_config: StrategyConfig, **kwargs):
        super().__init__(ib, contract, strategy_config, **kwargs)
        cfg = strategy_config.position_manager
        self.loss_pct = cfg.sl if cfg.sl else 0.0
        self.gain_pct = cfg.tp if cfg.tp else 0.0
        self.price_buffer = kwargs.get("price_buffer")

    def _latest_close(self) -> Optional[float]:
        if self.price_buffer is None or self.price_buffer.is_empty():
            return None
        return float(self.price_buffer.get_data()["close"].iloc[-1])

    def _evaluate_position(
        self, position: dict
    ) -> tuple[bool, Optional[float], Optional[float]]:
        close = self._latest_close()
        avg_cost = position.get("avg_cost", 0.0)
        if close is None or avg_cost <= 0:
            return False, None, None
        is_long = position["position"] > 0
        pnl_pct = (close - avg_cost) / avg_cost if is_long else (avg_cost - close) / avg_cost
        if self.loss_pct > 0 and pnl_pct <= -self.loss_pct:
            logger.info(
                f"PercentChange close: {self.symbol} P&L {pnl_pct:.4f} <= -{self.loss_pct}"
            )
            return True, None, None
        if self.gain_pct > 0 and pnl_pct >= self.gain_pct:
            logger.info(
                f"PercentChange close: {self.symbol} P&L {pnl_pct:.4f} >= {self.gain_pct}"
            )
            return True, None, None
        return False, None, None
