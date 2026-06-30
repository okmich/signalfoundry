"""Shared atomic-write helper for the macro store.

A crash mid-write must never leave a half-written file under its final name. Every persisted
artifact (per-series parquet, the materialized feature store) writes to a temp sibling, fsyncs the
data to disk, then ``os.replace`` — an atomic rename on both POSIX and Windows.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically (temp + fsync + replace).

    fsync before the replace so a crash can't commit the rename while the data blocks are still
    unflushed (which would leave a zero-length/partial parquet under the final name).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        df.to_parquet(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    """Write ``text`` to ``path`` atomically (temp + fsync + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
