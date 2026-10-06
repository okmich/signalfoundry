"""Parquet storage shared by the history downloader and the live recorder.

* single-file datasets (candles, funding, open interest, long/short ratio): merged with what is on disk and replaced
  atomically (temp file + ``os.replace``), so a crash never leaves a torn file;
* partitioned datasets (trades, recorder streams): one directory per day (UTC). The recorder appends small
  ``part-*`` files while an hour is open and compacts them into ``HH.parquet`` once the hour has passed.
"""
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_symbol(symbol: str) -> str:
    """A CCXT symbol as one path component: ``BTC/USDT:USDT`` -> ``BTC_USDT_USDT``."""
    return _UNSAFE.sub("_", symbol).strip("_")


def save_atomically(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    try:
        os.close(fd)
        df.to_parquet(tmp, compression="snappy")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_existing(path: str | Path) -> Optional[pd.DataFrame]:
    return pd.read_parquet(path) if os.path.exists(path) else None


def merge_on_index(existing: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """Union of two time-indexed frames; the newer download wins on duplicate timestamps."""
    if existing is None or existing.empty:
        return new.sort_index()
    if new.empty:
        return existing.sort_index()
    merged = pd.concat([existing, new])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def merge_on_key(existing: Optional[pd.DataFrame], new: pd.DataFrame, key: str) -> pd.DataFrame:
    """Union of two frames that may share timestamps (trades), de-duplicated on ``key``. Rows WITHOUT a key are never
    treated as duplicates of each other - two trades without an id are two trades."""
    if existing is None or existing.empty:
        merged = new
    elif new.empty:
        merged = existing
    else:
        merged = pd.concat([existing, new])
    has_key = merged[key].notna()
    merged = merged[~(has_key & merged[key].duplicated(keep="last"))]
    return merged.sort_index(kind="stable")


def day_of(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d")


def write_daily_partitions(df: pd.DataFrame, directory: str | Path, key: str) -> list[Path]:
    """Merge ``df`` (time-indexed, UTC) into ``directory/YYYY-MM-DD.parquet`` files, de-duplicated on ``key``."""
    written = []
    if df.empty:
        return written
    for day, part in df.groupby(df.index.tz_convert("UTC").strftime("%Y-%m-%d")):
        path = Path(directory) / f"{day}.parquet"
        save_atomically(merge_on_key(load_existing(path), part, key), path)
        written.append(path)
    return written


def latest_partition_ts(directory: str | Path) -> Optional[pd.Timestamp]:
    """Newest index timestamp across ``directory/*.parquet`` daily partitions (None when there are none)."""
    files = sorted(Path(directory).glob("*.parquet")) if Path(directory).exists() else []
    for path in reversed(files):
        df = pd.read_parquet(path)
        if not df.empty:
            return df.index.max()
    return None


def utc_ms_to_ts(ms: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(ms / 1000, tz=timezone.utc))
