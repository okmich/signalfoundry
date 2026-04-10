import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from okmich_quant_core import BaseFilter

logger = logging.getLogger(__name__)


class DayTimeFilter(BaseFilter):
    """
    Filter based on date and time ranges.
    Allows trading only during specified days of week and time ranges.
    """

    def __init__(
        self,
        allowed_days: Optional[List[int]] = None,
        allowed_time_ranges: Optional[List[tuple]] = None,
        name: Optional[str] = None,
    ):
        """
        :param allowed_days: List of allowed weekdays (0=Monday, 6=Sunday). None means all days allowed.
        :param allowed_time_ranges: List of (from_time, to_time) tuples where each time is datetime.time object.
                                   None means all times allowed.
        :param name: Optional filter name

        Example:
            # Only trade Mon-Fri, 9:00-17:00
            DateTimeFilter(
                allowed_days=[0, 1, 2, 3, 4],
                allowed_time_ranges=[(time(9, 0), time(17, 0))]
            )
        """
        super().__init__(name or "DateTimeFilter")
        self.allowed_days = allowed_days
        self.allowed_time_ranges = allowed_time_ranges

    def do_filter(self, context: Dict[str, Any]) -> bool:
        dt = context.get("datetime", datetime.now())
        if not isinstance(dt, datetime):
            logger.warning(f"Filter '{self.name}': No valid datetime in context")
            return False

        # Check day of week
        if self.allowed_days is not None:
            if dt.weekday() not in self.allowed_days:
                logger.debug(
                    f"Filter '{self.name}': Day {dt.weekday()} not in allowed days"
                )
                return False

        # Check time ranges
        if self.allowed_time_ranges is not None:
            current_time = dt.time()
            time_allowed = any(
                from_time <= current_time <= to_time
                for from_time, to_time in self.allowed_time_ranges
            )
            if not time_allowed:
                logger.debug(
                    f"Filter '{self.name}': Time {current_time} not in allowed ranges"
                )
                return False

        return True
