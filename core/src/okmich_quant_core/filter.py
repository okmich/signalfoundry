#
# {
#     filters: []
# }

# spread filter
# datetime filter

import logging
from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """
    Abstract base class for all filters.
    Filters return True to allow trading operations, False to block them.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def do_filter(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the filter condition.

        :param context: Dictionary containing relevant information for filtering
                       (e.g., 'datetime', 'symbol_info', 'tick_info', 'signal_type')
        :return: True if operation should proceed, False if it should be blocked
        """
        pass

    def __call__(self, context: Dict[str, Any]) -> bool:
        """Allow filter to be called directly."""
        result = self.do_filter(context)
        logger.debug(f"Filter '{self.name}' returned: {result}")
        return result


class FilterChain(BaseFilter):
    """
    Composite filter that chains multiple filters together.
    Returns False as soon as any filter returns False (short-circuit evaluation).
    Returns True only if all filters return True.
    """

    def __init__(self, filters: List[BaseFilter], name: Optional[str] = None):
        """
        :param filters: List of filters to chain
        :param name: Optional name for the chain
        """
        super().__init__(name or "FilterChain")
        self.filters = filters

    def do_filter(self, context: Dict[str, Any]) -> bool:
        if not self.filters:
            return True

        for f in self.filters:
            if not f(context):
                logger.info(f"FilterChain '{self.name}' blocked by filter '{f.name}'")
                return False
        logger.debug(f"FilterChain '{self.name}' passed all filters")
        return True

    def add_filter(self, filter_obj: BaseFilter):
        self.filters.append(filter_obj)

    def remove_filter(self, filter_name: str):
        self.filters = [f for f in self.filters if f.name != filter_name]
