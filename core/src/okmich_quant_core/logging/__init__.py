from .base import (
    LOG_SCHEMA_VERSION,
    BarOutcome,
    BarRecord,
    BaseEventLogger,
    CircuitBreakerTrippedRecord,
    Envelope,
    LogBinding,
    LogEventType,
    LogRecord,
    StrategyReenabledRecord,
    SystemRecordFactory,
    UnknownSchemaMajorError,
    load_schema,
    record_from_dict,
    schema_path,
)
from .identity import LogicalSystemIdentity, LogRootConfigError, RunnerIdentity
from .jsonl import JsonlEventLogger
from .runner_status import RunnerStatus
from .text_log import setup_text_logger, text_log_dir

__all__ = [
    # LOGGING_CONTRACT v1.0.0 — inference-log channel (per-bar heartbeat + per-strategy breaker)
    "LOG_SCHEMA_VERSION",
    "LogEventType",
    "BarOutcome",
    "Envelope",
    "LogRecord",
    "BarRecord",
    "CircuitBreakerTrippedRecord",
    "StrategyReenabledRecord",
    "SystemRecordFactory",
    "LogBinding",
    "BaseEventLogger",
    "JsonlEventLogger",
    "LogRootConfigError",
    "RunnerIdentity",
    "LogicalSystemIdentity",
    "record_from_dict",
    "load_schema",
    "schema_path",
    "UnknownSchemaMajorError",
    # runner-lifecycle channel (status file, read directly by the Supervisor)
    "RunnerStatus",
    # per-process text log (human-readable; resolves LOG_BASE, else the config dir)
    "setup_text_logger",
    "text_log_dir",
]

#: Names removed in 0.7.0 (LOGGING_CONTRACT v1.0.0) — turn a bare ImportError into a migration hint.
_RETIRED_IN_0_7_0 = {
    "InferenceLogRecord": "BarRecord (constructed via SystemRecordFactory)",
    "BaseInferenceLogger": "BaseEventLogger",
    "JsonlInferenceLogger": "JsonlEventLogger",
    "build_runner_logger": "RunnerStatus (runner lifecycle is now a status file, not a JSONL log)",
    "RunnerRecordFactory": "RunnerStatus",
    "StartupRecord": "RunnerStatus.mark_started()",
    "ShutdownRecord": "RunnerStatus.mark_stopped()",
}


def __getattr__(name: str):
    if name in _RETIRED_IN_0_7_0:
        raise ImportError(
            f"okmich_quant_core.logging.{name} was removed in 0.7.0 (LOGGING_CONTRACT v1.0.0): the "
            f"inference log is now event-typed and runner lifecycle is a status file. "
            f"Use {_RETIRED_IN_0_7_0[name]} instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
