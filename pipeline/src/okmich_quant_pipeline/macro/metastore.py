"""Metastore for the macro data store — a JSON sidecar mirroring MT5's ``_metadata.json``.

One file per store folder, keyed by canonical series name. Tracks each series' FRED id,
channel, availability policy, coverage span, observation count, and the last update run.
Written atomically (temp + ``os.replace``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

METADATA_FILENAME = "_metadata.json"


class MacroMetastore:
    """Read/update the per-series metadata JSON for a macro store directory."""

    def __init__(self, folder: Path, filename: str = METADATA_FILENAME):
        self.path = Path(folder) / filename

    def read(self) -> dict[str, Any]:
        """Return the metadata dict, or ``{}`` if absent/unreadable."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    def write(self, metadata: dict[str, Any]) -> None:
        """Persist the full metadata dict atomically (temp + replace)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(metadata, indent=4, default=str))
        os.replace(tmp, self.path)

    def update_series(self, series: str, updates: dict[str, Any]) -> None:
        """Merge ``updates`` into one series' record and persist atomically."""
        metadata = self.read()
        metadata.setdefault(series, {}).update(updates)
        self.write(metadata)

    def last_obs(self, series: str) -> str | None:
        """Return the stored last observation date (ISO string) for a series, if any."""
        return self.read().get(series, {}).get("last_obs")
