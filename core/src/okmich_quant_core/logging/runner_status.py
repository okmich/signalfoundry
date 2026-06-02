"""Runner lifecycle status file (LOGGING_CONTRACT runner-lifecycle channel).

Runner startup/shutdown is **state**, not a stream event: "is this system running or cleanly
stopped, and did it disconnect?" is answered by reading one small file, not by tailing+filtering a
bar JSONL. So lifecycle does NOT go on the inference-log channel. Instead each logical system gets a
single ``status.json`` written **atomically** (write tmp → fsync → ``os.replace``, last-write-wins)
beside its inference dir:

``<log_base>/<strategy>/<symbol>/<timeframe>/status.json``

The Supervisor reads it directly for its stop/restart loop (clean-stop proof = ``state == "stopped"``
+ ``broker_disconnected: true``; restart detection = a changed ``runner_start_token``). A
``multi-trader`` writes the same runner status to each of its logical systems' paths, so the
Supervisor finds it at whichever system path it controls — it never has to scan or wait for N.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .base import LOG_SCHEMA_VERSION, _iso_utc
from .identity import LogicalSystemIdentity, RunnerIdentity, _resolve_log_base


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as JSON to ``path`` atomically + durably (tmp → fsync → os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic on the same directory (the clean-stop proof must not tear)


class RunnerStatus:
    """Writes the per-logical-system ``status.json`` at startup and at shutdown.

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

    def status_path(self, logical: LogicalSystemIdentity) -> Path:
        strategy, symbol, timeframe = logical.path_parts()
        return self._base / strategy / symbol / timeframe / "status.json"

    @property
    def status_paths(self) -> list[Path]:
        return [self.status_path(s) for s in self._systems]

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

    def _write_all(self, payload: Mapping[str, Any]) -> None:
        for logical in self._systems:
            _atomic_write_json(self.status_path(logical), payload)

    def mark_started(self) -> None:
        self._started_at = _iso_utc(datetime.now(timezone.utc))
        self._write_all(self._payload("running"))

    def mark_stopped(self, *, broker_disconnected: bool, clean: bool, reason: str | None = None) -> None:
        self._write_all(self._payload("stopped", broker_disconnected=bool(broker_disconnected),
                                      clean=bool(clean), reason=reason,
                                      stopped_at=_iso_utc(datetime.now(timezone.utc))))
