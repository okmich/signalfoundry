from datetime import time, datetime

from okmich_quant_core import FilterChain, BaseFilter, StrategyConfig
from ._max_position import MaxPositionsFilter
from ._daytime import DayTimeFilter
from ._spread import SpreadFilter


def create_filter(strategy_config: StrategyConfig) -> FilterChain:
    """
    Create a filter instance from configuration dictionary.

    :param strategy_config: strategy configuration holding the filter configuration
    :return: FilterChain instance
    """
    filters = []
    for filter_config in strategy_config.filters:
        filter_type = filter_config.type
        _filter_obj = None
        params = filter_config.params

        if filter_type == "datetime":

            def _parse_time(time_str: str) -> time:
                try:
                    dt = datetime.strptime(time_str, "%H:%M")
                    return dt.time()
                except ValueError as e:
                    raise ValueError(
                        f"Time must be in 'HH:MM' format, got '{time_str}': {e}"
                    )

            # Parse time ranges
            time_ranges = []
            for time_range in params["allowed_time_ranges"]:
                from_time = _parse_time(time_range["from"])
                to_time = _parse_time(time_range["to"])
                time_ranges.append((from_time, to_time))

            _filter_obj = DayTimeFilter(
                allowed_days=params["allowed_days"],
                allowed_time_ranges=time_ranges,
                name=params.get("name"),
            )
        elif filter_type == "spread":
            _filter_obj = SpreadFilter(
                name=params.get("name"), max_spread_points=params["max_spread_points"]
            )
        elif filter_type == "max_positions":
            _filter_obj = MaxPositionsFilter(
                name=params.get("name"), max_positions=params["max_positions"]
            )
        else:
            raise ValueError(f"Encountered unknown filter type: {filter_type}")

        filters.append(_filter_obj)

    return FilterChain(filters)
