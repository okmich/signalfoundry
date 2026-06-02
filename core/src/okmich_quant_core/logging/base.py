"""LOGGING_CONTRACT v1.0.0 — event-typed inference-log records, factory, schema, sink interface.

The inference-log channel carries the per-bar heartbeat + per-strategy breaker events (runner
lifecycle is a status file — see ``runner_status.py``). Every record carries the common
:class:`Envelope` (§6) and is one of the per-event record types keyed by :class:`LogEventType` (§7).
Records are constructed ONLY via the framework-owned :class:`SystemRecordFactory` — call sites MUST
NOT hand-assemble record dicts (§5.1). The free-form ``extras`` sub-record is serialised verbatim;
readers (e.g. ``okmich_quant_ml.posterior_inference.monitoring_io``) pull family-specific keys (HMM
``probs``/``loglik``) back out. The authoritative definition of each event is the JSON Schema shipped
under ``logging/schema/`` (§13).
"""

from __future__ import annotations

import enum
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .identity import LogicalSystemIdentity, RunnerIdentity


LOG_SCHEMA_VERSION = "1.0.0"

_SCHEMA_DIR = Path(__file__).resolve().parent / "schema"


class LogEventType(enum.StrEnum):
    # The inference-log channel carries the per-bar heartbeat + per-strategy breaker events.
    # Runner lifecycle (startup/shutdown) is NOT a log event — it is the runner status file
    # (runner_status.py), read directly by the Supervisor rather than scanned from this stream.
    BAR = "bar"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    STRATEGY_REENABLED = "strategy_reenabled"


class BarOutcome(enum.StrEnum):
    OK = "ok"                               # on_new_bar() completed
    ERROR = "error"                         # on_new_bar() raised; cycle failed
    SKIPPED_DISABLED = "skipped_disabled"   # strategy circuit-broken; cycle not run


class UnknownSchemaMajorError(ValueError):
    """A record's ``log_schema_version`` major exceeds what this consumer understands (§13)."""


def _iso_utc(ts: Any) -> str | None:
    """Coerce a timestamp-like value to an ISO-8601 string with explicit UTC offset.

    ``None`` passes through. Naive timestamps are *assumed UTC* (the framework derives
    ``asof_bar_ts`` in UTC; this is a guard, not a localiser).
    """
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    return t.isoformat()


def _check_schema_major(version: Any) -> None:
    major = int(str(version).split(".")[0])
    ours = int(LOG_SCHEMA_VERSION.split(".")[0])
    if major > ours:
        raise UnknownSchemaMajorError(
            f"record log_schema_version {version!r} (major {major}) exceeds supported major {ours}"
        )


@dataclass(frozen=True)
class Envelope:
    """The common envelope carried by every record on the inference-log channel (§6)."""

    log_schema_version: str
    event: LogEventType
    wall_clock_utc: str
    runner_id: str
    runner_start_token: str
    logical_system_id: str
    strategy: str
    symbol: str | None
    timeframe: int | None
    broker: str
    account_id: str
    broker_session_id: str | None
    order_tag: str | int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_schema_version": self.log_schema_version,
            "event": str(self.event),
            "wall_clock_utc": self.wall_clock_utc,
            "runner_id": self.runner_id,
            "runner_start_token": self.runner_start_token,
            "logical_system_id": self.logical_system_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "broker": self.broker,
            "account_id": self.account_id,
            "broker_session_id": self.broker_session_id,
            "order_tag": self.order_tag,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Envelope":
        return cls(
            log_schema_version=payload["log_schema_version"],
            event=LogEventType(payload["event"]),
            wall_clock_utc=payload["wall_clock_utc"],
            runner_id=payload["runner_id"],
            runner_start_token=payload["runner_start_token"],
            logical_system_id=payload["logical_system_id"],
            strategy=payload["strategy"],
            symbol=payload.get("symbol"),
            timeframe=payload.get("timeframe"),
            broker=payload["broker"],
            account_id=payload["account_id"],
            broker_session_id=payload.get("broker_session_id"),
            order_tag=payload.get("order_tag"),
        )


@dataclass(frozen=True)
class LogRecord(ABC):
    """Base for every event record: an :class:`Envelope` plus event-specific fields."""

    envelope: Envelope

    def to_dict(self) -> dict[str, Any]:
        return {**self.envelope.to_dict(), **self._event_fields()}

    def _event_fields(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class BarRecord(LogRecord):
    """Per-bar heartbeat (§7.2): Tier 0 floor (mandatory) + Tier 1 content (best-effort)."""

    asof_bar_ts: str | None = None
    outcome: BarOutcome = BarOutcome.OK
    bar_close: float | None = None
    label_bar_ts: str | None = None
    direction: int | None = None
    confidence: float | None = None
    features: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)
    tier1_error: str | None = None  # §8 isolation annotation when Tier 1 capture/serialise failed

    def _event_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "asof_bar_ts": self.asof_bar_ts,
            "outcome": str(self.outcome),
            "bar_close": None if self.bar_close is None else float(self.bar_close),
            "label_bar_ts": self.label_bar_ts,
            "direction": None if self.direction is None else int(self.direction),
            "confidence": None if self.confidence is None else float(self.confidence),
            "features": dict(self.features),
            "extras": dict(self.extras),
        }
        if self.tier1_error is not None:
            out["tier1_error"] = self.tier1_error
        return out


@dataclass(frozen=True)
class CircuitBreakerTrippedRecord(LogRecord):
    """Breaker trip (§7.3). Per-strategy; the ops-channel twin of the existing Telegram alert."""

    consecutive_errors: int = 0
    last_error: str | None = None

    def _event_fields(self) -> dict[str, Any]:
        return {"consecutive_errors": int(self.consecutive_errors), "last_error": self.last_error}


@dataclass(frozen=True)
class StrategyReenabledRecord(LogRecord):
    """Strategy re-enable (§7.3). Per-strategy."""

    reason: str | None = None

    def _event_fields(self) -> dict[str, Any]:
        return {"reason": self.reason}


_RECORD_BY_EVENT: dict[LogEventType, type[LogRecord]] = {
    LogEventType.BAR: BarRecord,
    LogEventType.CIRCUIT_BREAKER_TRIPPED: CircuitBreakerTrippedRecord,
    LogEventType.STRATEGY_REENABLED: StrategyReenabledRecord,
}


def record_from_dict(payload: Mapping[str, Any]) -> LogRecord:
    """Reconstruct an inference-log record from a parsed JSONL line, rejecting an unknown major (§13).

    Raises :class:`UnknownSchemaMajorError` if the record's major exceeds ours, ``ValueError``
    if required keys / the ``event`` discriminator are missing or unrecognised. Callers that
    want quarantine-and-continue (the reader/tailer) catch these per line. (Runner lifecycle is a
    status file, not a log record — it is not reconstructed here.)
    """
    for required in ("event", "log_schema_version"):
        if required not in payload:
            raise ValueError(f"record_from_dict: missing required key '{required}'")
    _check_schema_major(payload["log_schema_version"])
    event = LogEventType(payload["event"])  # ValueError on an unrecognised event string
    envelope = Envelope.from_dict(payload)
    cls = _RECORD_BY_EVENT[event]
    if cls is BarRecord:
        return BarRecord(envelope=envelope, asof_bar_ts=payload.get("asof_bar_ts"),
                         outcome=BarOutcome(payload["outcome"]), bar_close=payload.get("bar_close"),
                         label_bar_ts=payload.get("label_bar_ts"), direction=payload.get("direction"),
                         confidence=payload.get("confidence"), features=dict(payload.get("features", {})),
                         extras=dict(payload.get("extras", {})), tier1_error=payload.get("tier1_error"))
    if cls is CircuitBreakerTrippedRecord:
        return CircuitBreakerTrippedRecord(envelope=envelope,
                                           consecutive_errors=int(payload.get("consecutive_errors", 0)),
                                           last_error=payload.get("last_error"))
    return StrategyReenabledRecord(envelope=envelope, reason=payload.get("reason"))


def schema_path(name: LogEventType | str) -> Path:
    """Filesystem path of a shipped JSON Schema (§13). Accepts a :class:`LogEventType` (the event's
    record schema) or a plain name like ``"runner_status"`` (the status-file schema)."""
    stem = name.value if isinstance(name, LogEventType) else str(name)
    return _SCHEMA_DIR / f"{stem}.json"


def load_schema(name: LogEventType | str) -> dict[str, Any]:
    """Load a shipped JSON Schema — the authoritative definition the prose mirrors."""
    with open(schema_path(name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_envelope(event: LogEventType, *, runner: RunnerIdentity, logical_system_id: str,
                    strategy: str, symbol: str | None, timeframe: int | None,
                    order_tag: str | int | None, wall_clock: Any) -> Envelope:
    return Envelope(log_schema_version=LOG_SCHEMA_VERSION, event=event, wall_clock_utc=_iso_utc(wall_clock),
                    runner_id=runner.runner_id, runner_start_token=runner.runner_start_token,
                    logical_system_id=logical_system_id, strategy=strategy, symbol=symbol, timeframe=timeframe,
                    broker=runner.broker, account_id=runner.account_id,
                    broker_session_id=runner.broker_session_id, order_tag=order_tag)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SystemRecordFactory:
    """The single constructor for per-logical-system records (bar / breaker / reenabled, §5.1).

    Built from a bound :class:`RunnerIdentity` + :class:`LogicalSystemIdentity`; stamps
    ``log_schema_version`` and ``wall_clock_utc`` in one place so the envelope never drifts.
    """

    def __init__(self, runner: RunnerIdentity, logical: LogicalSystemIdentity,
                 order_tag: str | int | None = None, clock=_utc_now):
        self._runner = runner
        self._logical = logical
        self._order_tag = order_tag
        self._clock = clock

    def _env(self, event: LogEventType, wall_clock: Any) -> Envelope:
        return _build_envelope(event, runner=self._runner, logical_system_id=self._logical.logical_system_id,
                               strategy=self._logical.strategy, symbol=self._logical.symbol,
                               timeframe=self._logical.timeframe_minutes, order_tag=self._order_tag,
                               wall_clock=wall_clock if wall_clock is not None else self._clock())

    def bar(self, *, asof_bar_ts: Any, outcome: BarOutcome | str, bar_close: float | None = None,
            label_bar_ts: Any = None, direction: int | None = None, confidence: float | None = None,
            features: Mapping[str, Any] | None = None, extras: Mapping[str, Any] | None = None,
            tier1_error: str | None = None, wall_clock: Any = None) -> BarRecord:
        return BarRecord(envelope=self._env(LogEventType.BAR, wall_clock), asof_bar_ts=_iso_utc(asof_bar_ts),
                         outcome=BarOutcome(outcome), bar_close=bar_close, label_bar_ts=_iso_utc(label_bar_ts),
                         direction=direction, confidence=confidence, features=dict(features or {}),
                         extras=dict(extras or {}), tier1_error=tier1_error)

    def circuit_breaker_tripped(self, *, consecutive_errors: int, last_error: str | None = None,
                                wall_clock: Any = None) -> CircuitBreakerTrippedRecord:
        return CircuitBreakerTrippedRecord(envelope=self._env(LogEventType.CIRCUIT_BREAKER_TRIPPED, wall_clock),
                                           consecutive_errors=consecutive_errors, last_error=last_error)

    def strategy_reenabled(self, *, reason: str | None = None, wall_clock: Any = None) -> StrategyReenabledRecord:
        return StrategyReenabledRecord(envelope=self._env(LogEventType.STRATEGY_REENABLED, wall_clock), reason=reason)


class LogBinding:
    """Two-phase identity holder owned by a strategy (LOGGING_CONTRACT §5, design C).

    Phase 1 (strategy ``__init__``): logical identity + a fail-closed logger are known, so the
    binding is constructible immediately. Phase 2 (runner startup): :meth:`bind` supplies the
    runner identity, completing the envelope before the first bar fires. :meth:`system_factory`
    refuses to emit while unbound, so a record can never be written with a missing ``runner_id``.
    """

    def __init__(self, logical: LogicalSystemIdentity, logger: "BaseEventLogger",
                 *, order_tag: str | int | None = None):
        self.logical = logical
        self.logger = logger
        self.order_tag = order_tag
        self._runner: RunnerIdentity | None = None
        self._factory: SystemRecordFactory | None = None

    @property
    def is_bound(self) -> bool:
        return self._runner is not None

    @property
    def runner(self) -> RunnerIdentity | None:
        return self._runner

    def bind(self, runner: RunnerIdentity) -> None:
        self._runner = runner
        # Build the per-system factory once at bind (identity is fixed thereafter) so the per-bar
        # emission path allocates no factory.
        self._factory = SystemRecordFactory(runner, self.logical, order_tag=self.order_tag)

    def system_factory(self) -> SystemRecordFactory:
        if self._factory is None:
            raise RuntimeError(
                "LogBinding.system_factory: runner identity not bound — call bind() at startup "
                "before emitting records (the envelope's runner_id MUST be present, §6)."
            )
        return self._factory


class BaseEventLogger(ABC):
    """Abstract sink for contract v1.0.0 records (§5/§10).

    ``write`` dispatches by event criticality (lifecycle records synchronously fsync-durable;
    ``bar`` heartbeats through a bounded non-blocking writer). ``drain`` flushes pending bar
    records (used before a clean shutdown so the last heartbeats are not lost). ``close`` drains
    then releases handles.
    """

    @abstractmethod
    def write(self, record: LogRecord) -> None:
        ...

    @abstractmethod
    def drain(self, timeout: float | None = None) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
