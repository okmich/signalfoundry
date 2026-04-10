import logging
from typing import Optional, Dict, Any

from okmich_quant_core import BaseFilter

logger = logging.getLogger(__name__)


class SpreadFilter(BaseFilter):
    """
    Filter based on bid-ask spread.
    Blocks trading when spread exceeds threshold.
    """

    def __init__(self, max_spread_points: float, name: Optional[str] = None):
        """
        :param max_spread_points: Maximum allowed spread in points
        :param name: Optional filter name
        """
        super().__init__(name or "SpreadFilter")
        self.max_spread_points = max_spread_points

    def do_filter(self, context: Dict[str, Any]) -> bool:
        spread = context.get("spread")
        if spread is None:
            logger.warning(f"Filter '{self.name}': No spread in context")
            return False

        if spread >= self.max_spread_points:
            logger.info(
                f"Filter '{self.name}': Spread {spread:.1f} exceeds or equals max {self.max_spread_points}"
            )
            return False

        return True
