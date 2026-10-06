"""``okmich_quant_crypto`` - crypto exchange integration via CCXT (spot and USDT-margined linear perpetuals)."""
from .bar_aggregator import CandleCloseDetector
from .broker_session import CryptoBrokerSession, is_exchange_closed
from .capabilities import ResolvedCapabilities, resolve_capabilities
from .client_order_id import ClientIdRule, client_order_prefix, is_ours, magic_of, make_client_order_id
from .config import (
    CryptoStrategyConfig, CryptoSystemConfig, CryptoVenueConfig, check_isolation, derive_log_symbol, isolation_key,
)
from .enums import (
    FeedMode, FillKind, MarginMode, MarginModeScope, MarketType, OrderRole, OrderSide, PositionMode, SizingUnit,
    StopMode, StopTrigger, VenueEnvironment,
)
from .event_loop import CryptoEventLoop
from .feed import BarReconciler, BarSequencer, ClosedBarSource, PollBarSource, StreamBarSource
from .functions.crypto import (
    cancel_order_safe, connect_exchange, fetch_free_balance, fetch_quote_equity, fetch_tick_info, get_open_orders,
    load_credentials, place_order_idempotent,
)
from .markets import MarketSpec, OrderSizeError
from .models import ClosedBar, Credentials, Fill, FundingPayment
from .orders import OrderRegistry
from .position_cache import CryptoPositionCache, EndedLifecycle
from .resilience import (
    CryptoBannedError, CryptoError, CryptoPermanentError, CryptoTransientError, ErrorClass, OrderStateUnknownError,
    VenueUnsupportedError, classify_ccxt_error, with_retry,
)
from .spot_ledger import SpotInventoryLedger
from .state_store import StateStore
from .stops import ManagedStopController, NativeStopController, StopController
from .strategy import BaseCryptoStrategy, GenericBasicCryptoStrategy, VenueContext
from .timeframe_utils import (
    MAX_TIMEFRAME_MINUTES, timeframe_to_minutes, timeframe_to_ms, timeframe_to_seconds, validate_venue_timeframe,
)
from .venue import (
    BinanceProfile, BybitProfile, StopCapabilities, VenueProfile, get_profile, is_supported, resolve_profile,
    supported_venues,
)

__all__ = [
    # enums
    "FeedMode", "FillKind", "MarginMode", "MarginModeScope", "MarketType", "OrderRole", "OrderSide", "PositionMode",
    "SizingUnit", "StopMode", "StopTrigger", "VenueEnvironment",
    # config
    "CryptoStrategyConfig", "CryptoSystemConfig", "CryptoVenueConfig", "check_isolation", "derive_log_symbol",
    "isolation_key",
    # venue profiles + capabilities
    "VenueProfile", "BinanceProfile", "BybitProfile", "StopCapabilities", "supported_venues", "is_supported",
    "get_profile", "resolve_profile",
    "ResolvedCapabilities", "resolve_capabilities",
    # resilience
    "CryptoError", "CryptoTransientError", "CryptoPermanentError", "CryptoBannedError", "OrderStateUnknownError",
    "VenueUnsupportedError", "ErrorClass", "classify_ccxt_error", "with_retry",
    # markets, ids, models
    "MarketSpec", "OrderSizeError", "ClientIdRule", "make_client_order_id", "client_order_prefix", "magic_of",
    "is_ours", "ClosedBar", "Credentials", "Fill", "FundingPayment",
    # feeds
    "CandleCloseDetector", "BarReconciler", "BarSequencer", "ClosedBarSource", "StreamBarSource", "PollBarSource",
    # book + stops
    "CryptoPositionCache", "EndedLifecycle", "SpotInventoryLedger", "OrderRegistry", "StateStore", "StopController",
    "NativeStopController", "ManagedStopController",
    # runtime
    "CryptoBrokerSession", "is_exchange_closed", "CryptoEventLoop", "BaseCryptoStrategy", "GenericBasicCryptoStrategy",
    "VenueContext",
    # functions
    "connect_exchange", "load_credentials", "fetch_tick_info", "fetch_quote_equity", "fetch_free_balance",
    "place_order_idempotent", "cancel_order_safe", "get_open_orders",
    # timeframes
    "MAX_TIMEFRAME_MINUTES", "timeframe_to_minutes", "timeframe_to_ms", "timeframe_to_seconds",
    "validate_venue_timeframe",
]
