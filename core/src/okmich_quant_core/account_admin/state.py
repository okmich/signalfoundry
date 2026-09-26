"""The Admin's private state file: one slot per task (ACCOUNT_ADMIN_SPEC §7.3).

The Admin is its only reader and writer. A slot belongs to one task kind; a slot whose task is no longer configured is
carried over untouched, so removing a task and adding it back loses nothing.
"""

from __future__ import annotations

import enum
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json, read_json_retry
from .timeutil import iso_z

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"
STATE_SCHEMA_VERSION = 1


class StateLoad(enum.StrEnum):
    OK = "ok"
    MISSING = "missing"
    CORRUPT = "corrupt"


def load_state(path: str | Path) -> tuple[dict[str, dict[str, Any]], StateLoad]:
    """The task slots, and whether the file was there and sound. A missing or corrupt file yields no slots; each task
    decides what a lost slot means for it (``prop_guard`` trips ``state_lost``)."""
    try:
        payload = read_json_retry(path)
    except FileNotFoundError:
        return {}, StateLoad.MISSING
    except (OSError, ValueError) as exc:
        logger.error("state file %s is unreadable: %s", path, exc)
        return {}, StateLoad.CORRUPT
    if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA_VERSION \
            or not isinstance(payload.get("tasks"), dict):
        logger.error("state file %s does not follow schema %s", path, STATE_SCHEMA_VERSION)
        return {}, StateLoad.CORRUPT
    slots = {str(k): v for k, v in payload["tasks"].items() if isinstance(v, dict)}
    return slots, StateLoad.OK


def save_state(path: str | Path, slots: dict[str, dict[str, Any]], now: datetime) -> None:
    """Atomically replace the state file. Raises ``OSError`` when the replace kept failing."""
    atomic_write_json(path, {"schema_version": STATE_SCHEMA_VERSION, "written_utc": iso_z(now), "tasks": slots})
