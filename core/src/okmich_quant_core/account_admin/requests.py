"""The operator request inbox (ACCOUNT_ADMIN_SPEC §8.2).

An operator command writes one request file atomically into ``_account_admin\\requests\\``. The Admin consumes them at
its next cycle, oldest first: the host checks the envelope and routes each to the task it names, the task applies or
rejects it, and the host moves it to ``done\\`` with its outcome. The command never touches a task's output or the
state file.
"""

from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .atomic import atomic_write_json, read_json_retry
from .enums import RequestOutcome
from .timeutil import iso_z, parse_utc, utc_now

logger = logging.getLogger(__name__)

REQUESTS_DIR = "requests"
DONE_DIR = "done"


class RequestFormatError(ValueError):
    """A request whose envelope is malformed. It is rejected, never guessed at (spec §8.2)."""


@dataclass(frozen=True)
class AdminRequest:
    """One operator request. ``fields`` is the whole request as written, for the task to read its own keys from."""

    request_id: str
    created_utc: datetime
    operator: str
    task: str
    kind: str
    fields: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Any) -> "AdminRequest":
        if not isinstance(payload, Mapping):
            raise RequestFormatError("request is not a JSON object")
        try:
            request_id = str(payload["request_id"]).strip()
            operator = str(payload["operator"]).strip()
            task = str(payload["task"]).strip()
            kind = str(payload["kind"]).strip()
            created = parse_utc(payload["created_utc"])
        except KeyError as exc:
            raise RequestFormatError(f"missing field {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise RequestFormatError(f"malformed field: {exc}") from exc
        for name, value in (("request_id", request_id), ("operator", operator), ("task", task), ("kind", kind)):
            if not value:
                raise RequestFormatError(f"field {name!r} is empty")
        return cls(request_id=request_id, created_utc=created, operator=operator, task=task, kind=kind,
                   fields=dict(payload))


@dataclass(frozen=True)
class PendingRequest:
    """A file waiting in the inbox: its parsed request, or why it could not be parsed."""

    path: Path
    raw: Any
    request: AdminRequest | None
    error: str | None


def new_request_id(now: datetime | None = None) -> str:
    """``20260924T131522Z-7f3a``: sortable by creation time, unique enough for one operator's inbox."""
    now = now if now is not None else utc_now()
    return f"{iso_z(now).replace('-', '').replace(':', '')}-{secrets.token_hex(2)}"


def submit_request(inbox: str | Path, *, operator: str, task: str, kind: str, now: datetime | None = None,
                   **fields: Any) -> Path:
    """Write one request atomically into ``inbox`` and return its path. Used by the operator command."""
    now = now if now is not None else utc_now()
    request_id = new_request_id(now)
    payload = {"request_id": request_id, "created_utc": iso_z(now), "operator": operator, "task": task, "kind": kind,
               **fields}
    AdminRequest.from_dict(payload)  # refuse to write what the Admin would reject as malformed
    path = Path(inbox) / f"{request_id}.json"
    atomic_write_json(path, payload)
    return path


class RequestInbox:
    """The inbox folder and its ``done\\`` archive."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.done_directory = self.directory / DONE_DIR

    def pending(self) -> list[PendingRequest]:
        """Every request waiting, oldest first (by ``created_utc``, then file name; unparseable ones first by name so
        they are rejected promptly). Temp files of an in-flight submit are not requests yet."""
        if not self.directory.is_dir():
            return []
        items: list[PendingRequest] = []
        for path in self.directory.glob("*.json"):
            if not path.is_file():
                continue
            try:
                raw = read_json_retry(path)
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                items.append(PendingRequest(path, None, None, f"unreadable: {exc}"))
                continue
            try:
                items.append(PendingRequest(path, raw, AdminRequest.from_dict(raw), None))
            except RequestFormatError as exc:
                items.append(PendingRequest(path, raw, None, str(exc)))

        def order(item: PendingRequest):
            created = item.request.created_utc.timestamp() if item.request is not None else float("-inf")
            return created, item.path.name

        return sorted(items, key=order)

    def complete(self, item: PendingRequest, outcome: RequestOutcome, reason: str, now: datetime) -> Path:
        """Record the outcome and move the request to ``done\\``. The original content is kept under ``request``."""
        target = self.done_directory / item.path.name
        record = {"request": item.raw, "outcome": str(outcome), "reason": reason, "consumed_utc": iso_z(now)}
        atomic_write_json(target, record)
        try:
            os.remove(item.path)
        except FileNotFoundError:
            pass
        return target
