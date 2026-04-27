"""``okmich_quant_ib`` — Interactive Brokers integration."""
from .bar_aggregator import BarAggregator
from .contract import (
    DEFAULT_USE_RTH, DEFAULT_WHAT_TO_SHOW, IBContractConfig, SecType,
    make_order_ref, resolve_contract,
)
from .event_loop import IBEventLoop
from .functions.ib import (
    cancel_all_pending_orders, cancel_order, close_position, connect_ib,
    disconnect_ib, fetch_contract_info, fetch_historical_bars,
    fetch_historical_bars_paginated, fetch_tick, get_account_summary,
    get_pending_orders, get_positions, modify_order, place_bracket_order,
    place_limit_order, place_market_order, place_stop_order,
)
from .position_cache import IBPositionCache
from .resilience import (
    ConnectionMonitor, ErrorClass, IBConnectionError, IBPermanentError,
    IBTransientError, classify_ib_error, is_ib_connected, with_retry,
)
from .strategy import BaseIBStrategy, GenericBasicIBStrategy
from .timeframe_utils import (
    BAR_SIZE_MINUTES, MAX_DURATION_DAYS, bar_size_to_minutes, required_duration,
)

__all__ = [
    # contract
    "IBContractConfig", "SecType", "make_order_ref", "resolve_contract",
    "DEFAULT_WHAT_TO_SHOW", "DEFAULT_USE_RTH",
    # resilience
    "IBConnectionError", "IBTransientError", "IBPermanentError", "ErrorClass",
    "classify_ib_error", "with_retry", "ConnectionMonitor", "is_ib_connected",
    # timeframe utils
    "bar_size_to_minutes", "required_duration", "BAR_SIZE_MINUTES",
    "MAX_DURATION_DAYS",
    # bar aggregation
    "BarAggregator",
    # position cache
    "IBPositionCache",
    # event loop
    "IBEventLoop",
    # strategy
    "BaseIBStrategy", "GenericBasicIBStrategy",
    # functions
    "connect_ib", "disconnect_ib", "fetch_contract_info",
    "fetch_historical_bars", "fetch_historical_bars_paginated", "fetch_tick",
    "place_market_order", "place_limit_order", "place_stop_order",
    "place_bracket_order", "close_position", "modify_order", "cancel_order",
    "get_positions", "get_pending_orders", "cancel_all_pending_orders",
    "get_account_summary",
]
