"""Runner lifecycle status file (LOGGING_CONTRACT runner-lifecycle channel).

Runner startup/shutdown is **state**, not a stream event: "is this system running or cleanly
stopped, and did it disconnect?" is answered by reading one small file, not by tailing+filtering a
bar JSONL. So lifecycle does NOT go on the inference-log channel. Instead the runner writes a single
``status.json`` **atomically** (write tmp → fsync → ``os.replace``, last-write-wins) at the runner root:

``<log_base>/<strategy>/status.json``   (``<strategy>-multi`` for a multi-trader)

The Supervisor reads it directly for its stop/restart loop (clean-stop proof = ``state == "stopped"``
+ ``broker_disconnected: true``; restart detection = a changed ``runner_start_token``). Runner lifecycle
is one process, so there is exactly ONE status file at the runner root — NOT mirrored per logical system.
The runner's logical systems are enumerated inside it (``logical_systems[]``), which is how the Supervisor
maps the runner to its symbols.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .base import LOG_SCHEMA_VERSION, _iso_utc
from .identity import LogicalSystemIdentity, RunnerIdentity, _path_safe, _resolve_log_base

_log = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically + durably (tmp → fsync → os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name so two runners overlapping on the same status path (e.g. two multi-traders of
    # one strategy) don't clobber each other's temp file; os.replace stays atomic on the final name.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, allow_nan=False))  # strict JSON, matching the JSONL records
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic on the same directory (the clean-stop proof must not tear)


class RunnerStatus:
    """Writes the single runner-root ``status.json`` at startup and at shutdown.

    Construct once at runner startup with the bound :class:`RunnerIdentity` and the runner's logical
    systems. Call :meth:`mark_started` after binding (state ``running``) and :meth:`mark_stopped`
    after the proven broker disconnect (state ``stopped`` + the disconnect proof).
    """

    def __init__(self, runner: RunnerIdentity, logical_systems: Iterable[LogicalSystemIdentity], *,
                 log_base: str | Path | None = None, pid: int | None = None,
                 library_versions: Optional[Mapping[str, Any]] = None):
        self._runner = runner
        self._systems = list(logical_systems)
        self._base = _resolve_log_base(log_base)
        self._pid = os.getpid() if pid is None else pid
        self._library_versions = dict(library_versions or {})
        self._started_at: str | None = None

    def _runner_root_strategy(self) -> str:
        """The single runner-root folder: the (already runner-suffixed) strategy shared by the runner's
        logical systems. A homogeneous runner (the strategy-root layout's assumption) has exactly one; a
        heterogeneous one (different strategies in one process) falls back to the first, with a warning."""
        strategies = {s.strategy for s in self._systems}
        if len(strategies) == 1:
            return next(iter(strategies))
        root = self._systems[0].strategy if self._systems else "unknown"
        _log.warning("RunnerStatus: runner spans multiple strategies %s; the single status.json is written "
                     "under %r (the strategy-root layout assumes one strategy per runner).",
                     sorted(strategies), root)
        return root

    @property
    def status_path(self) -> Path:
        """The ONE runner-scoped status file at the runner root: ``<log_base>/<strategy>/status.json``
        (``<strategy>-multi`` for a multi-trader). Runner lifecycle is one process → one file, NOT
        mirrored per logical system (LOGGING_CONTRACT §7.1)."""
        return self._base / _path_safe(self._runner_root_strategy()) / "status.json"

    def _payload(self, state: str, *, broker_disconnected: bool | None = None, clean: bool | None = None,
                 reason: str | None = None, stopped_at: str | None = None) -> dict[str, Any]:
        return {
            "log_schema_version": LOG_SCHEMA_VERSION,
            "state": state,
            "runner_id": self._runner.runner_id,
            "runner_start_token": self._runner.runner_start_token,
            "pid": self._pid,
            "broker": self._runner.broker,
            "account_id": self._runner.account_id,
            "broker_session_id": self._runner.broker_session_id,
            "logical_systems": [{"logical_system_id": s.logical_system_id, "symbol": s.symbol,
                                 "timeframe": s.timeframe_minutes} for s in self._systems],
            "library_versions": self._library_versions,
            "started_at": self._started_at,
            "stopped_at": stopped_at,
            "broker_disconnected": broker_disconnected,
            "clean": clean,
            "reason": reason,
        }

    def _write(self, payload: Mapping[str, Any]) -> None:
        _atomic_write_json(self.status_path, payload)

    def mark_started(self) -> None:
        self._started_at = _iso_utc(datetime.now(timezone.utc))
        self._write(self._payload("running"))

    def mark_stopped(self, *, broker_disconnected: bool, clean: bool, reason: str | None = None) -> None:
        self._write(self._payload("stopped", broker_disconnected=bool(broker_disconnected),
                                  clean=bool(clean), reason=reason,
                                  stopped_at=_iso_utc(datetime.now(timezone.utc))))
