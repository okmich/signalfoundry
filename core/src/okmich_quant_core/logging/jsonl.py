from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from .base import BaseEventLogger, LogEventType, LogRecord
from .identity import LogicalSystemIdentity, _resolve_log_base


_log = logging.getLogger(__name__)

#: Behavioral events written synchronously (write + flush + fsync) on the calling thread. Runner
#: lifecycle (startup/shutdown) is NOT in this stream — it is a status file (see runner_status.py).
#: Everything else (``bar``) routes through the bounded writer thread (LOGGING_CONTRACT §10).
_LIFECYCLE_EVENTS = frozenset({
    LogEventType.CIRCUIT_BREAKER_TRIPPED, LogEventType.STRATEGY_REENABLED,
})

#: Default bound for the bar-record writer queue. Large enough that hitting it means a
#: genuine prolonged disk stall (not a momentary spike), at which point back-pressure is
#: surfaced rather than the trading loop blocked.
_DEFAULT_BAR_QUEUE_MAXSIZE = 10_000


_WRITER_SENTINEL = object()


class JsonlEventLogger(BaseEventLogger):
    """Conformant inference-log sink for LOGGING_CONTRACT v1.0.0 records.

    **Path (§10, OPS §7).** ``<log_base>\\<strategy>\\<symbol>\\<timeframe>\\inference\\
    inference_<YYYYMMDD>.jsonl`` where ``<log_base>`` is supplied explicitly or via
    ``OKMICH_QUANT_LOG_BASE``. Missing/blank roots fail fast; deployment examples such as
    ``D:\\quant_logs`` are never used as source-code fallbacks. Strategy / symbol / timeframe live
    in the **path**, not the filename. Append-only; the handle rotates when the UTC date flips. One
    logger per logical system → one file per symbol.

    **Durability split (§10).** Lifecycle records (``startup``/``shutdown``/
    ``circuit_breaker_tripped``/``strategy_reenabled``) are written **synchronously durable**
    (write + flush + ``os.fsync`` before :meth:`write` returns) — rare, ops-critical, the fsync
    cost paid willingly. ``bar`` heartbeats are handed off **non-blocking** to a bounded
    single-consumer writer thread that performs the fsync **off the trading loop**, so a disk
    spike never stalls trading. Both paths serialise through one file handle behind a lock, so
    lifecycle and bar writes never interleave a torn line.

    **Back-pressure (§10).** If the bar queue is full (writer can't keep up / disk stalled), the
    record is dropped but the condition is **surfaced** — :attr:`backpressure_dropped` is
    incremented and an alert-worthy error logged — never a *silent* drop, so a stalled writer is
    itself observable. The handoff is always non-blocking; the trading loop is never blocked.

    **Errors (§10).** A failed write/fsync on the lifecycle path is re-raised (the caller must
    know the durability contract broke). On the bar path the writer thread logs + counts
    (:attr:`write_errors`) and resets the handle rather than dying, so one bad write doesn't
    silently stop all subsequent heartbeats.
    """

    def __init__(self, logical: LogicalSystemIdentity, log_base: str | Path | None = None, *,
                 fsync: bool = True, bar_queue_maxsize: int = _DEFAULT_BAR_QUEUE_MAXSIZE):
        self._logical = logical
        base = _resolve_log_base(log_base)
        strategy, symbol, timeframe = logical.path_parts()
        self._dir = base / strategy / symbol / timeframe / "inference"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fsync = bool(fsync)

        self._lock = threading.Lock()
        self._current_date: str | None = None
        self._current_handle: IO[str] | None = None

        self._queue: queue.Queue = queue.Queue(maxsize=bar_queue_maxsize)
        self._closed = False
        self.backpressure_dropped = 0
        self.write_errors = 0
        self.post_close_dropped = 0

        self._writer = threading.Thread(target=self._writer_loop, name=f"jsonl-writer:{logical.logical_system_id}",
                                        daemon=True)
        self._writer.start()

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def logical(self) -> LogicalSystemIdentity:
        return self._logical

    def _path_for_date(self, date_str: str) -> Path:
        return self._dir / f"inference_{date_str}.jsonl"

    def _write_line_locked(self, line: str) -> None:
        """Append one JSONL line + flush (+ fsync). Caller MUST hold ``self._lock``.

        Handles UTC daily rotation. On OSError the handle is closed/reset (so the next write
        reopens to a known-good state) and the error re-raised.
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today != self._current_date:
            if self._current_handle is not None:
                try:
                    self._current_handle.close()
                except Exception as exc:
                    _log.warning(f"JsonlEventLogger: error closing prior handle on date flip: {exc}")
            self._current_handle = open(self._path_for_date(today), "a", encoding="utf-8")
            self._current_date = today
        handle = self._current_handle
        try:
            handle.write(line + "\n")
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        except OSError as exc:
            _log.error(f"JsonlEventLogger: write/flush/fsync failed; resetting handle: {exc}")
            try:
                handle.close()
            except Exception:
                pass
            self._current_handle = None
            self._current_date = None
            raise

    def _write_sync(self, record: LogRecord) -> None:
        # allow_nan=False: never write the bare NaN/Infinity tokens (invalid JSON for strict readers).
        # Tier 1 is already guarded at the producer; this is defense-in-depth for any other field.
        line = json.dumps(record.to_dict(), allow_nan=False)
        with self._lock:
            self._write_line_locked(line)

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _WRITER_SENTINEL:
                self._queue.task_done()
                break
            try:
                self._write_sync(item)
            except Exception as exc:  # writer thread must survive a bad write (see class docstring)
                self.write_errors += 1
                _log.error(f"JsonlEventLogger: bar-record write failed in writer thread: {exc}")
            finally:
                self._queue.task_done()

    def write(self, record: LogRecord) -> None:
        """Route a record by event criticality (lifecycle = sync-durable; bar = bounded queue).

        After :meth:`close` the writer thread is gone, so a late write is rejected (surfaced via
        :attr:`post_close_dropped` + an error log) rather than silently queued to a dead consumer or
        resurrecting a closed handle.
        """
        if self._closed:
            self.post_close_dropped += 1
            _log.error(
                f"JsonlEventLogger: write() after close() — dropping {record.envelope.event} record for "
                f"{self._logical.logical_system_id} (post_close_dropped={self.post_close_dropped})"
            )
            return
        if record.envelope.event in _LIFECYCLE_EVENTS:
            self._write_sync(record)
            return
        # bar heartbeat — non-blocking handoff
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self.backpressure_dropped += 1
            _log.error(
                f"JsonlEventLogger: bar queue full (maxsize={self._queue.maxsize}) for "
                f"{self._logical.logical_system_id} — writer stalled, dropping a heartbeat "
                f"(backpressure_dropped={self.backpressure_dropped}). A stalled writer is now observable."
            )

    def drain(self, timeout: float | None = None) -> None:
        """Block until every queued bar record has been written to disk.

        Called before a clean shutdown so the last heartbeats aren't lost. With ``timeout`` set, gives
        up after that many seconds (polling, no helper thread) and logs how many records remain — so a
        stalled disk can never hang shutdown forever.
        """
        if timeout is None:
            self._queue.join()
            return
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                _log.error(
                    f"JsonlEventLogger.drain: timed out after {timeout:.1f}s with ~{self._queue.unfinished_tasks} "
                    f"bar record(s) still queued for {self._logical.logical_system_id}"
                )
                return
            time.sleep(0.02)

    def close(self, drain_timeout: float | None = 30.0) -> None:
        """Drain queued bars (bounded), stop the writer thread, then durably close the handle.

        ``drain_timeout`` caps how long the final drain waits so a stalled disk cannot hang shutdown
        forever (the contract's clean-stop must not block indefinitely).
        """
        if self._closed:
            return
        self._closed = True
        self.drain(timeout=drain_timeout)
        # Signal the writer to stop. Non-blocking: if the queue is still full the writer is wedged
        # (stalled disk) and a blocking put would hang shutdown forever — the daemon writer dies with
        # the process instead.
        try:
            self._queue.put_nowait(_WRITER_SENTINEL)
        except queue.Full:
            _log.error(f"JsonlEventLogger.close: bar queue full — writer wedged, abandoning it (daemon) "
                       f"for {self._logical.logical_system_id}")
        self._writer.join(timeout=5.0)
        if self._writer.is_alive():
            _log.error(f"JsonlEventLogger.close: writer thread did not exit within 5s for "
                       f"{self._logical.logical_system_id}")
        # Acquire the handle lock with a timeout: a wedged writer may hold it mid-fsync, and shutdown
        # must not block forever. If we can't get it, leave the handle for OS cleanup at process exit.
        if self._lock.acquire(timeout=5.0):
            try:
                if self._current_handle is not None:
                    try:
                        self._current_handle.flush()
                        if self._fsync:
                            os.fsync(self._current_handle.fileno())
                        self._current_handle.close()
                    finally:
                        self._current_handle = None
                        self._current_date = None
            finally:
                self._lock.release()
        else:
            _log.error(f"JsonlEventLogger.close: could not acquire handle lock within 5s (writer wedged) "
                       f"for {self._logical.logical_system_id}; leaving handle for OS cleanup")
