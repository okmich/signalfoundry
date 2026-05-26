from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from .base import BaseInferenceLogger, InferenceLogRecord


_log = logging.getLogger(__name__)


class JsonlInferenceLogger(BaseInferenceLogger):
    """Append-only JSONL logger with daily UTC rotation and per-write fsync.

    Writes one JSON record per line to ``<log_dir>/inference_<strategy_name>_<YYYYMMDD>.jsonl`` where the date is the
    current UTC date at write time. When the UTC date flips between writes, the previous handle is closed and a new file
    is opened — readers can pick up the relevant day(s) by globbing on the date suffix.

    **Durability:** each :meth:`write` performs ``write`` + ``flush`` + ``os.fsync`` before returning. This is the
    durability contract the monitoring system relies on (data loss would silently undermine drift detection). On Windows
    the fsync cost is non-trivial; for M5-cadence strategies the per-bar overhead is negligible (~12 writes/hour). For
    higher-frequency strategies, consider whether durability is worth the cost.

    **Thread-safety:** not safe for concurrent writes from multiple threads on the same instance. Use one logger per strategy;
    instantiate per-thread if multiple producers exist (rare for HMM-style per-bar pipelines).
    """

    def __init__(self, log_dir: str | Path, strategy_name: str, *, fsync: bool = True):
        if not strategy_name:
            raise ValueError("JsonlInferenceLogger: strategy_name must be non-empty")
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._strategy_name = str(strategy_name)
        self._fsync = bool(fsync)
        self._current_date: str | None = None
        self._current_handle: IO[str] | None = None

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def strategy_name(self) -> str:
        return self._strategy_name

    def _path_for_date(self, date_str: str) -> Path:
        return self._log_dir / f"inference_{self._strategy_name}_{date_str}.jsonl"

    def _ensure_handle(self) -> IO[str]:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._current_date:
            if self._current_handle is not None:
                try:
                    self._current_handle.close()
                except Exception as exc:
                    _log.warning(f"JsonlInferenceLogger: error closing prior handle on date flip: {exc}")
            self._current_handle = open(self._path_for_date(today), "a", encoding="utf-8")
            self._current_date = today
        return self._current_handle

    def write(self, record: InferenceLogRecord) -> None:
        handle = self._ensure_handle()
        handle.write(json.dumps(record.to_dict()) + "\n")
        try:
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        except OSError as exc:
            # Durability contract broken — handle state is unclear (partial line in OS buffer
            # may or may not have reached disk). Close + reset so the next call reopens to a
            # known-good state, then re-raise so the caller knows this record is not guaranteed
            # durable.
            _log.error(f"JsonlInferenceLogger: flush/fsync failed; closing handle to reset: {exc}")
            try:
                handle.close()
            except Exception:
                pass
            self._current_handle = None
            self._current_date = None
            raise

    def close(self) -> None:
        if self._current_handle is not None:
            try:
                self._current_handle.flush()
                if self._fsync:
                    os.fsync(self._current_handle.fileno())
                self._current_handle.close()
            finally:
                self._current_handle = None
                self._current_date = None
