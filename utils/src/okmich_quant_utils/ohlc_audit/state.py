"""The tool's state folder: staging copies, quality flags, run manifests, snapshots (and the P4 hole ledger).

Never inside the market-data tree. Resolution: the ``state_dir`` argument, else OKMICH_QUANT_DATA_STATE_DIR, else fail.
The folder must already exist, so a typo can't silently create a new tree.
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import os
from pathlib import Path

#: Env var naming the tool's state folder.
STATE_DIR_VAR = "OKMICH_QUANT_DATA_STATE_DIR"


class StateDirNotConfiguredError(RuntimeError):
    """No usable state folder: none given, it doesn't exist, or it sits inside the market-data tree."""


def resolve_state_dir(state_dir: str | Path | None = None, market_data_root: Path | None = None) -> Path:
    value = str(state_dir).strip() if state_dir else os.environ.get(STATE_DIR_VAR, "").strip()
    source = "--state-dir" if state_dir else STATE_DIR_VAR
    if not value:
        raise StateDirNotConfiguredError(f"no state folder: pass --state-dir or set {STATE_DIR_VAR}")
    path = Path(value)
    if not path.is_dir():
        raise StateDirNotConfiguredError(f"{source}={value} is not an existing folder")
    if market_data_root is not None:
        p, root = path.resolve(), Path(market_data_root).resolve()
        if p == root or root in p.parents:
            raise StateDirNotConfiguredError(f"{source}={value} is inside the market-data tree {root}; tool state must live elsewhere")
    return path


@dc.dataclass(frozen=True)
class StateLayout:
    root: Path
    broker: str
    bar_minutes: int

    @property
    def base(self) -> Path:
        return self.root / self.broker / str(self.bar_minutes)

    @property
    def staging(self) -> Path:
        return self.base / "staging"

    @property
    def quality(self) -> Path:
        return self.base / "quality"

    @property
    def runs(self) -> Path:
        return self.base / "runs"

    @property
    def snapshots(self) -> Path:
        return self.base / "snapshots"

    @property
    def fills(self) -> Path:
        """Verified broker bars from refetches, one parquet per symbol file (the raw schema plus provenance columns)."""
        return self.base / "fills"

    @property
    def ledger(self) -> Path:
        """The hole ledger: one row per refetch target, persistent across runs."""
        return self.base / "holes.parquet"

    def ensure(self) -> StateLayout:
        for d in (self.staging, self.quality, self.runs, self.snapshots, self.fills):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def latest_snapshot(self) -> Path | None:
        if not self.snapshots.is_dir():
            return None
        snaps = sorted(p for p in self.snapshots.iterdir() if (p / "snapshot.json").is_file())
        return snaps[-1] if snaps else None


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
