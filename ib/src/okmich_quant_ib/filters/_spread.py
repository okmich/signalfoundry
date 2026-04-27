"""IB-specific spread filter using bid/ask percentage of mid-price."""
import logging
from typing import Any, Dict, Optional

from okmich_quant_core import BaseFilter

logger = logging.getLogger(__name__)


class SpreadFilter(BaseFilter):
    """Block entry when (ask - bid) / mid exceeds ``max_spread_pct``.

    Fails closed when bid/ask data is unavailable; pass ``allow_on_missing=True``
    to permit entries when market data is temporarily absent (e.g. pre-market or
    immediately after subscribe before the first tick).
    """

    def __init__(self, max_spread_pct: float, name: Optional[str] = None,
                 allow_on_missing: bool = False):
        super().__init__(name or "SpreadFilter")
        if not 0 <= max_spread_pct <= 1:
            raise ValueError(f"max_spread_pct must be in [0, 1], got {max_spread_pct}")
        self.max_spread_pct = max_spread_pct
        self.allow_on_missing = allow_on_missing

    def do_filter(self, context: Dict[str, Any]) -> bool:
        tick = context.get("tick_info")
        if not tick:
            return self.allow_on_missing
        bid = tick.get("bid", 0.0)
        ask = tick.get("ask", 0.0)
        if bid <= 0 or ask <= 0:
            return self.allow_on_missing
        mid = (bid + ask) / 2.0
        ratio = (ask - bid) / mid
        if ratio > self.max_spread_pct:
            logger.info(
                f"Filter '{self.name}': spread {ratio:.4f} exceeds max {self.max_spread_pct}"
            )
            return False
        return True
