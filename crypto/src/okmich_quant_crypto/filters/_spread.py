"""Spread filter on bid/ask as a fraction of mid (same semantics as the IB filter, kept local: no ib dependency)."""
import logging
from typing import Any, Dict, Optional

from okmich_quant_core import BaseFilter

logger = logging.getLogger(__name__)


class SpreadFilter(BaseFilter):
    """Block entry when ``(ask - bid) / mid`` exceeds ``max_spread_pct``. Fails closed on missing quotes unless
    ``allow_on_missing`` is set."""

    def __init__(self, max_spread_pct: float, name: Optional[str] = None, allow_on_missing: bool = False):
        super().__init__(name or "SpreadFilter")
        if not 0 <= max_spread_pct <= 1:
            raise ValueError(f"max_spread_pct must be in [0, 1], got {max_spread_pct}")
        self.max_spread_pct = max_spread_pct
        self.allow_on_missing = allow_on_missing

    def do_filter(self, context: Dict[str, Any]) -> bool:
        tick = context.get("tick_info")
        if not tick:
            return self.allow_on_missing
        bid, ask = tick.get("bid", 0.0), tick.get("ask", 0.0)
        if bid <= 0 or ask <= 0:
            return self.allow_on_missing
        mid = (bid + ask) / 2.0
        ratio = (ask - bid) / mid
        if ratio > self.max_spread_pct:
            logger.info("Filter '%s': spread %.6f exceeds max %s", self.name, ratio, self.max_spread_pct)
            return False
        return True
