"""Filter chain factory for IB strategies.

DayTimeFilter and MaxPositionsFilter are broker-agnostic and live in
``okmich_quant_utils.filters``; both this module and the MT5 module import them
from there. The IB package has no dependency on the MT5 package.
"""
from datetime import datetime, time

from okmich_quant_core import BaseFilter, FilterChain, StrategyConfig
from okmich_quant_utils.filters import DayTimeFilter, MaxPositionsFilter

from ._spread import SpreadFilter


def _parse_time(time_str: str) -> time:
    try:
        return datetime.strptime(time_str, "%H:%M").time()
    except ValueError as e:
        raise ValueError(f"Time must be in 'HH:MM' format, got '{time_str}': {e}")


def create_filter(strategy_config: StrategyConfig) -> FilterChain:
    """Build a FilterChain from ``strategy_config.filters``."""
    filters: list[BaseFilter] = []
    for filter_config in strategy_config.filters:
        params = filter_config.params

        if filter_config.type == "datetime":
            time_ranges = []
            for tr in params["allowed_time_ranges"]:
                time_ranges.append((_parse_time(tr["from"]), _parse_time(tr["to"])))
            filters.append(DayTimeFilter(
                allowed_days=params["allowed_days"],
                allowed_time_ranges=time_ranges,
                name=params.get("name"),
            ))
        elif filter_config.type == "spread":
            filters.append(SpreadFilter(
                max_spread_pct=params["max_spread_pct"],
                name=params.get("name"),
                allow_on_missing=params.get("allow_on_missing", False),
            ))
        elif filter_config.type == "max_positions":
            filters.append(MaxPositionsFilter(
                max_positions=params["max_positions"],
                name=params.get("name"),
            ))
        else:
            raise ValueError(f"Unknown filter type: {filter_config.type}")

    return FilterChain(filters)
