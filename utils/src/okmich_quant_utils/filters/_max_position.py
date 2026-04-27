import logging
from typing import Any, Dict, Optional

from okmich_quant_core import BaseFilter

logger = logging.getLogger(__name__)


class MaxPositionsFilter(BaseFilter):
    """Filter based on current number of open positions."""

    def __init__(self, max_positions: int, name: Optional[str] = None):
        super().__init__(name or "MaxPositionsFilter")
        self.max_positions = max_positions

    def do_filter(self, context: Dict[str, Any]) -> bool:
        open_positions = context.get("open_positions", 0)
        if open_positions >= self.max_positions:
            logger.info(
                f"Filter '{self.name}': Max positions {self.max_positions} reached"
            )
            return False
        return True
