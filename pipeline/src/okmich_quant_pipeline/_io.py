"""Shared atomic-write helpers for the pipeline's on-disk data assets.

A crash mid-write must never leave a half-written file under its final name. Every persisted
artifact (per-series macro parquet, the materialized feature store, the news-calendar parquet)
writes to a *unique* temp sibling, fsyncs the data to disk, then ``os.replace`` — an atomic rename
on both POSIX and Windows. The temp name is unique (``mkstemp``) so two writers targeting the same
final path can't trample each other's in-flight temp file before the rename.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def _mktemp(path: Path) -> Path:
    """Create a unique empty temp file in ``path``'s directory and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    return Path(name)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically (unique temp + fsync + replace).

    fsync before the replace so a crash can't commit the rename while the data blocks are still
    unflushed (which would leave a zero-length/partial parquet under the final name).
    """
    path = Path(path)
    tmp = _mktemp(path)
    try:
        with open(tmp, "wb") as f:
            df.to_parquet(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(text: str, path: Path) -> None:
    """Write ``text`` to ``path`` atomically (unique temp + fsync + replace)."""
    path = Path(path)
    tmp = _mktemp(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
