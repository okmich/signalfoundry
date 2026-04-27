"""Close on dollar-loss threshold."""
import logging
from typing import Optional

from ib_async import IB, Contract

from okmich_quant_core import StrategyConfig

from .base import BaseIBPositionManager

logger = logging.getLogger(__name__)


class MaxLossAmountIBPositionManager(BaseIBPositionManager):
    """Close when unrealised dollar loss reaches ``max_loss_amount``.

    Required ``StrategyConfig.position_manager`` fields:
        max_loss_amount — close when |loss| (account currency) >= this value.

    Optional kwargs:
        price_buffer  — latest close for mark price.
        contract_info — dict from ``fetch_contract_info`` (used for multiplier).
    """

    def __init__(self, ib: IB, contract: Contract, strategy_config: StrategyConfig, **kwargs):
        super().__init__(ib, contract, strategy_config, **kwargs)
        cfg = strategy_config.position_manager
        self.max_loss_amount = cfg.max_loss_amount or 0.0
        self.price_buffer = kwargs.get("price_buffer")
        self.contract_info = kwargs.get("contract_info") or {}

    def _latest_close(self) -> Optional[float]:
        if self.price_buffer is None or self.price_buffer.is_empty():
            return None
        return float(self.price_buffer.get_data()["close"].iloc[-1])

    def _evaluate_position(
        self, position: dict
    ) -> tuple[bool, Optional[float], Optional[float]]:
        if self.max_loss_amount <= 0:
            return False, None, None
        close = self._latest_close()
        avg_cost = position.get("avg_cost", 0.0)
        size = position["position"]
        if close is None or avg_cost <= 0 or size == 0:
            return False, None, None
        multiplier = float(self.contract_info.get("multiplier", 1.0))
        # Unrealised P&L in account currency.
        pnl = (close - avg_cost) * size * multiplier
        if pnl < 0 and abs(pnl) >= self.max_loss_amount:
            logger.info(
                f"MaxLossAmount close: {self.symbol} loss ${abs(pnl):.2f} "
                f">= ${self.max_loss_amount:.2f}"
            )
            return True, None, None
        return False, None, None
