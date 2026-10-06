"""Small atomic JSON persistence for the state that must survive a restart.

What lives here: position lifecycle ids (the closed-trade key, see ``position_cache``), the order-id -> role map
(close attribution and what shutdown may cancel), managed stop levels (a managed stop that forgets its level on
restart is no stop at all), and the spot inventory ledger.

One file per (venue, strategy) namespace. Writes go to a temp file in the same directory and are swapped in with
``os.replace`` so a crash mid-write leaves the previous state, never a torn file.
"""
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(*parts: str) -> str:
    """Join identity parts into one filesystem-safe file stem."""
    return "__".join(_UNSAFE.sub("_", str(p)).strip("_") or "_" for p in parts)


class StateStore:
    """A JSON document on disk with atomic save. ``data`` is a plain dict the owner mutates then ``save()``s."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {}

    @classmethod
    def open(cls, state_dir: str | Path, *name_parts: str) -> "StateStore":
        store = cls(Path(state_dir) / f"{safe_name(*name_parts)}.json")
        store.load()
        return store

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.data = {}
            return self.data
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8")) or {}
        except Exception:
            # A corrupt state file must not stop the system from starting, but it must be loud: the lifecycle ids
            # and managed stop levels in it are gone.
            logger.exception("state file %s is unreadable; starting from empty state (kept as .corrupt)", self.path)
            try:
                os.replace(self.path, self.path.with_suffix(".corrupt"))
            except OSError:
                pass
            self.data = {}
        return self.data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=1, sort_keys=True, default=str)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def section(self, key: str) -> dict[str, Any]:
        """A named sub-dict, created on first use."""
        value = self.data.get(key)
        if not isinstance(value, dict):
            value = {}
            self.data[key] = value
        return value
