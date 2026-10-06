"""Who owns a magic number on this account (ACCOUNT_ADMIN_SPEC §11 "Naming the owner", §12.3 ``orphaned_only``).

The Admin reads, read-only, the ``config.json`` of every system deployed in its own account folder: those are exactly
the systems meant to trade this account. A magic no config there claims is **not this account's**. Whether an owner is
running comes from its ``status.json`` under the account's log folder and a check that its process is alive.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .directive import ADMIN_FOLDER

logger = logging.getLogger(__name__)

MANUAL_MAGIC = 0


@dataclass(frozen=True)
class MagicOwner:
    magic: int
    system_id: str          # <account>/<strategy>/<symbol>/<timeframe>, <account>/<strategy>-multi, or <account>/<runner>
    runner_root: str        # the folder under <log_base>\<account> holding its status.json
    running: bool


def pid_alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists. Conservative: an error reads as alive, so a system is never declared
    stopped (and its orders orphaned) on a failed check."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5             # ERROR_ACCESS_DENIED: it exists, we may not look
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                        # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _runner_running(log_account_dir: Path | None, runner_root: str) -> bool:
    if log_account_dir is None:
        return False
    status = _read_json(log_account_dir / runner_root / "status.json")
    if not isinstance(status, dict) or status.get("state") != "running":
        return False
    try:
        return pid_alive(int(status.get("pid") or 0))
    except (TypeError, ValueError):
        return False


def magic_owners(account_dir: str | Path, log_account_dir: str | Path | None, account: str) -> dict[int, MagicOwner]:
    """Every magic claimed by a system deployed under ``account_dir``, with whether that system is running.

    Tolerant by design: an unreadable or foreign config is skipped with a warning; it only costs a nicer name. A magic
    claimed twice keeps the first claim and warns, since two systems sharing a magic is its own defect.
    """
    root = Path(account_dir)
    logs = Path(log_account_dir) if log_account_dir is not None else None
    owners: dict[int, MagicOwner] = {}
    if not root.is_dir():
        return owners
    for run_py in root.rglob("run.py"):
        folder = run_py.parent
        rel = folder.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or (rel.parts and rel.parts[0] == ADMIN_FOLDER):
            continue
        config = _read_json(folder / "config.json")
        if not isinstance(config, dict):
            continue
        strategies = config.get("strategies")
        entries = strategies if isinstance(strategies, list) and strategies else [config.get("strategy") or config]
        multi = isinstance(strategies, list) and bool(strategies)
        parts = rel.parts
        if multi:
            # Named as the Supervisor names it: after the first strategy's name, with the statutory -multi suffix.
            code = str(strategies[0].get("name") or folder.name) if isinstance(strategies[0], dict) else folder.name
            runner_root = code if code.endswith("-multi") else f"{code}-multi"
            system_id = f"{account}/{runner_root}"
        elif len(parts) == 3:
            runner_root, system_id = parts[0], f"{account}/{'/'.join(parts)}"
        else:
            runner_root = parts[0] if parts else folder.name
            system_id = f"{account}/{runner_root}"
        running = _runner_running(logs, runner_root)
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("magic") is None:
                continue
            try:
                magic = int(entry["magic"])
            except (TypeError, ValueError):
                continue
            if magic in owners:
                logger.warning("magic %s is claimed by both %s and %s", magic, owners[magic].system_id, system_id)
                continue
            owners[magic] = MagicOwner(magic=magic, system_id=system_id, runner_root=runner_root, running=running)
    return owners


def describe_owner(magic: int, owners: dict[int, MagicOwner]) -> str:
    """A readable owner label for alerts."""
    if magic == MANUAL_MAGIC:
        return "manual"
    owner = owners.get(magic)
    if owner is None:
        return "not this account's"
    return owner.system_id if owner.running else f"{owner.system_id} (stopped)"
