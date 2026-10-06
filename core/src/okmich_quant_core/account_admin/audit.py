"""The Admin's audit trail (ACCOUNT_ADMIN_SPEC §7.7): ``audit\\admin_<YYYYMMDD>.jsonl`` under its log root, one JSON
object per line, UTC-dated. It is the Admin's own channel, not the inference log, so it MAY carry balance, equity and
P&L: the Supervisor never reads it. It is the input to the policy replay test."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .enums import AdminEvent
from .timeutil import iso_z, utc_now

logger = logging.getLogger(__name__)

AUDIT_DIR = "audit"


def _jsonable(value: Any) -> Any:
    """Enums as their values, datetimes as ISO-Z, containers recursively; everything else as is."""
    if isinstance(value, datetime):
        return iso_z(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), (str, int)):
        return value.value
    return value


class AuditLog:
    """Append-only writer. Every record carries ``event``, ``wall_clock_utc``, ``account``, ``task``, ``sequence`` and
    ``episode``. A write failure is logged, never raised: auditing must not stop a cycle."""

    def __init__(self, directory: str | Path, account: str):
        self.directory = Path(directory)
        self.account = account

    def path_for(self, when: datetime) -> Path:
        """``admin_<YYYYMMDD>.jsonl`` by the UTC date of ``when``."""
        return self.directory / f"admin_{iso_z(when)[:10].replace('-', '')}.jsonl"

    def write(self, event: AdminEvent, *, task: str | None = None, sequence: int | None = None,
              episode: int | None = None, now: datetime | None = None, **fields: Any) -> dict[str, Any]:
        now = now if now is not None else utc_now()
        record = {"event": str(event), "wall_clock_utc": iso_z(now), "account": self.account, "task": task,
                  "sequence": sequence, "episode": episode, **_jsonable(fields)}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with open(self.path_for(now), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception:
            logger.exception("audit write failed (%s)", event)
        return record
