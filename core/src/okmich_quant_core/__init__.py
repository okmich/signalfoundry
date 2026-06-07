from .base_strategy import BaseStrategy
from .broker_session import BrokerSession
from .filter import BaseFilter, FilterChain
from .health import StrategyHealth
from .logging import (
    BarOutcome,
    BarRecord,
    BaseEventLogger,
    CircuitBreakerTrippedRecord,
    JsonlEventLogger,
    LogBinding,
    LogEventType,
    LogicalSystemIdentity,
    RunnerIdentity,
    RunnerStatus,
    StrategyReenabledRecord,
    SystemRecordFactory,
    setup_text_logger,
    text_log_dir,
)
from .multi_trader import MultiTrader
from .notification import BaseNotifier, Telegram, TelegramNotifier
from .position_manager import BasePositionManager
from .config import  OrderType, PositionManagerType, PositionManagerConfig, PositionSizingType, PositionSizingConfig, \
    RunLoopConfig, StrategyConfig, SystemConfig
from .run_loop import RunLoop
from .signal import BaseSignal
from .trader import Trader

#: Public names removed in 0.7.0 (LOGGING_CONTRACT v1.0.0). No behavior-preserving alias is possible
#: (the event-typed API has different constructors); this just turns the bare ImportError into a
#: migration hint for any out-of-repo caller.
_RETIRED_IN_0_7_0 = {
    "InferenceLogRecord": "BarRecord (constructed via SystemRecordFactory)",
    "BaseInferenceLogger": "BaseEventLogger",
    "JsonlInferenceLogger": "JsonlEventLogger",
}


def __getattr__(name: str):
    if name in _RETIRED_IN_0_7_0:
        raise ImportError(
            f"okmich_quant_core.{name} was removed in 0.7.0 (LOGGING_CONTRACT v1.0.0): the inference "
            f"log is now event-typed. Use {_RETIRED_IN_0_7_0[name]} instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
