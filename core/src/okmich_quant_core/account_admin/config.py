"""The Admin's ``config.json`` (ACCOUNT_ADMIN_SPEC §6.1) and its stage-1 validation (§6.7).

The host validates its own sections (``kind``, ``runner``, ``clock``, ``requests``, ``tasks[]``) and hands each task
entry to its registered task class to validate. Every problem found is reported, and all of it happens before the Admin
touches the terminal: a config that cannot run never costs a terminal IPC slot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._validate import Problems, integer, section, text
from .clock import ClockConfig, parse_clock
from .directive import ADMIN_FOLDER
from .enums import AdminTaskKind
from .tasks.base import TASK_REGISTRY, AdminTask

CONFIG_KIND = "account_admin"
#: ``broker`` is the broker library's own section (e.g. the MT5 server clock); core passes it through untouched and the
#: broker library validates it in stage 1.
_HOST_KEYS = frozenset({"kind", "runner", "clock", "requests", "tasks", "name", "broker"})


class AdminConfigError(ValueError):
    """The config cannot run. ``problems`` lists every issue found."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("invalid Account Admin config:\n  - " + "\n  - ".join(self.problems))


@dataclass(frozen=True)
class AdminConfig:
    clock: ClockConfig
    request_ttl_s: int
    tasks: tuple[AdminTask, ...]
    raw: Mapping[str, Any]

    @property
    def governing_task(self) -> AdminTask | None:
        return next((t for t in self.tasks if t.governs_directive), None)


def parse_admin_config(raw: Any, registry: Mapping[AdminTaskKind, type[AdminTask]] | None = None) -> AdminConfig:
    """Validate ``raw`` completely and build the config with its task instances. Raises :class:`AdminConfigError`."""
    registry = TASK_REGISTRY if registry is None else registry
    p = Problems()
    if not isinstance(raw, Mapping):
        raise AdminConfigError(["config.json must be a JSON object"])
    if raw.get("kind") != CONFIG_KIND:
        p.add("kind", f"must be {CONFIG_KIND!r}, got {raw.get('kind')!r}")
    runner = text(raw, "runner", p)
    if runner is not None and runner != ADMIN_FOLDER:
        p.add("runner", f"must be {ADMIN_FOLDER!r}, got {runner!r}")
    if "strategies" in raw:
        p.add("strategies", "must not be present: the Supervisor would read it as a multi-trader (spec §6.1)")
    for key in raw:
        if key not in _HOST_KEYS and key != "strategies":
            p.add(key, "unknown key")
    clock = parse_clock(section(raw, "clock", p), p.child("clock"))
    requests = section(raw, "requests", p)
    ttl = integer(requests, "request_ttl_s", p.child("requests"), minimum=1) if requests is not None else None

    tasks: list[AdminTask] = []
    entries = raw.get("tasks")
    if not isinstance(entries, list) or not entries:
        p.add("tasks", "must be a non-empty list of task entries")
        entries = []
    seen: set[AdminTaskKind] = set()
    for i, entry in enumerate(entries):
        where = f"tasks[{i}]"
        if not isinstance(entry, Mapping):
            p.add(where, "must be an object")
            continue
        try:
            kind = AdminTaskKind(entry.get("kind"))
        except ValueError:
            known = ", ".join(k.value for k in registry)
            p.add(f"{where}.kind", f"unknown task kind {entry.get('kind')!r} (registered: {known})")
            continue
        if kind not in registry:
            p.add(f"{where}.kind", f"task kind {kind.value!r} is not registered in this build")
            continue
        if kind in seen:
            p.add(f"{where}.kind", f"task kind {kind.value!r} appears more than once")
            continue
        seen.add(kind)
        cls = registry[kind]
        task_problems = cls.validate(entry, where)
        if task_problems:
            p.items.extend(task_problems)
        else:
            tasks.append(cls(entry))
    if p or clock is None or ttl is None:
        raise AdminConfigError(p.items or ["invalid config"])
    return AdminConfig(clock=clock, request_ttl_s=ttl, tasks=tuple(tasks), raw=raw)


def load_admin_config(path: str | Path, registry: Mapping[AdminTaskKind, type[AdminTask]] | None = None) -> AdminConfig:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise AdminConfigError([f"{path}: not found"]) from exc
    except (OSError, ValueError) as exc:
        raise AdminConfigError([f"{path}: not readable JSON ({exc})"]) from exc
    return parse_admin_config(raw, registry)
