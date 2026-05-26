"""Abstract inference-logger interface + universal record schema.

The trading loop emits one :class:`InferenceLogRecord` per decision cycle to a :class:`BaseInferenceLogger`. The record
schema is intentionally generic across signal families (HMM, NN, rule-based) — universal fields (timestamps, features,
direction, confidence) sit at the top; family-specific data lives in ``extras``. HMM consumers put ``probs`` and ``loglik``
in extras; NN consumers put softmax + temperature; rule-based strategies emit ``extras={}``.

The logger has no opinion on extras — it serialises whatever's given. Readers that care about a specific family
(e.g. the HMM streaming-monitor reader in ``okmich_quant_ml.posterior_inference.monitoring_io``) pull the relevant keys
back out of ``extras`` on their own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def _to_iso(ts: Any) -> str | None:
    """Coerce a timestamp-like value to ISO string. None passes through."""
    if ts is None:
        return None
    return pd.Timestamp(ts).isoformat()


def _from_iso(value: Any) -> pd.Timestamp | None:
    """Coerce an ISO string back to pd.Timestamp. None passes through."""
    if value is None:
        return None
    return pd.Timestamp(value)


@dataclass(frozen=True)
class InferenceLogRecord:
    """One inference cycle's worth of state, written verbatim by the logger.

    Universal fields cover any signal-producing strategy. ``extras`` is the free-form sub-record where
    model/strategy-family-specific data lives — the logger serialises it as nested JSON and never inspects its contents.

    Strategy identity lives on the :class:`BaseInferenceLogger` (used for filename + read-time attribution); the record
    itself does not carry it, so the same record type works unchanged across every strategy.

    All timestamps round-trip via ISO 8601 strings in JSON; ``to_dict`` and ``from_dict`` handle the conversion both ways.

    **``extras`` contract — JSON-serialisable values only.** Values must be composed of the primitive JSON types
    (``str``, ``int``, ``float``, ``bool``, ``None``, ``list``, ``dict``); the logger calls ``json.dumps`` on the
    serialised record verbatim and does NOT recursively transform extras.
    Passing ``numpy`` arrays, ``pd.Timestamp``, ``datetime``, or other non-JSON types will raise ``TypeError`` at write
    time, deep in the call stack. Convert at the producer (e.g. ``[float(p) for p in probs]``).
    """
    wall_clock_utc: pd.Timestamp
    asof_bar_ts: pd.Timestamp
    label_bar_ts: pd.Timestamp | None
    bar_close: float | None
    features: Mapping[str, Any]
    direction: int | None
    confidence: float | None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_clock_utc": _to_iso(self.wall_clock_utc),
            "asof_bar_ts": _to_iso(self.asof_bar_ts),
            "label_bar_ts": _to_iso(self.label_bar_ts),
            "bar_close": None if self.bar_close is None else float(self.bar_close),
            "features": dict(self.features),
            "direction": None if self.direction is None else int(self.direction),
            "confidence": None if self.confidence is None else float(self.confidence),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InferenceLogRecord:
        required = ("wall_clock_utc", "asof_bar_ts", "features")
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"InferenceLogRecord.from_dict: missing required keys {missing}")
        bar_close_raw = payload.get("bar_close")
        direction_raw = payload.get("direction")
        confidence_raw = payload.get("confidence")
        return cls(wall_clock_utc=_from_iso(payload["wall_clock_utc"]),
                   asof_bar_ts=_from_iso(payload["asof_bar_ts"]),
                   label_bar_ts=_from_iso(payload.get("label_bar_ts")),
                   bar_close=None if bar_close_raw is None else float(bar_close_raw),
                   features=dict(payload["features"]),
                   direction=None if direction_raw is None else int(direction_raw),
                   confidence=None if confidence_raw is None else float(confidence_raw),
                   extras=dict(payload.get("extras", {})))


class BaseInferenceLogger(ABC):
    """Abstract inference-logger interface.

    Concrete implementations decide where records go (JSONL on disk, Parquet, in-memory ring, database, etc.). The trading
    loop holds a single instance per strategy and calls :meth:`write` once per inference cycle; :meth:`close` is called on
    shutdown to flush any open handles.

    Sync semantics by default — implementations must guarantee that a successful return from ``write`` means the record is
    durable on the chosen medium. The trading loop tolerates the per-write cost; the monitoring system cannot tolerate lost records.
    """

    @abstractmethod
    def write(self, record: InferenceLogRecord) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
