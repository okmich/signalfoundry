"""Fill store: verified broker bars from refetches, kept in ``<state>/<broker>/<tf>/fills/<file>.parquet``.

Every remediation run rebuilds staging from ``raw/``, so bars won by a refetch must persist outside ``raw/`` until P5
promotes them. Each row is in the raw file's schema, on naive raw stamps, plus provenance columns:
- ``_kind``: ``fill`` (a bar inserted where raw has none) or ``replace`` (a spike the broker now serves differently).
- ``_target``, ``_shift_minutes`` (handshake offset), ``_fetched_at``.
- ``_old_open`` .. ``_old_close``: the raw bar a replacement swaps out. It is applied only while raw still holds exactly
  that bar, so a new export that changed the bar wins.
The store only grows; ``compose`` decides per run which rows apply.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .handshake import PRICE_COLUMNS

FILL, REPLACE = "fill", "replace"
OLD_COLUMNS = [f"_old_{c}" for c in PRICE_COLUMNS]
META_COLUMNS = ["_kind", "_target", "_shift_minutes", "_fetched_at", *OLD_COLUMNS]


def like_index(naive: pd.DatetimeIndex, like: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive raw stamps re-labelled the way ``like`` (a raw file's index) is: same tz label, unit and name."""
    idx = pd.DatetimeIndex(naive)
    if like.tz is not None:
        idx = idx.tz_localize(like.tz, ambiguous="raise", nonexistent="raise")
    return pd.DatetimeIndex(idx.as_unit(like.unit), name=like.name)


def conform(bars: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame | str:
    """``bars`` in the raw file's columns and dtypes, or the reason they can't be."""
    missing = [c for c in like.columns if c not in bars.columns]
    if missing:
        return f"fetched bars lack column(s) {missing}"
    try:
        return bars[list(like.columns)].astype(like.dtypes.to_dict())
    except (ValueError, TypeError) as e:
        return f"fetched bars don't fit the raw dtypes: {e}"


class FillStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def path(self, file: str) -> Path:
        return self.folder / file

    def load(self, file: str) -> pd.DataFrame:
        p = self.path(file)
        return pd.read_parquet(p) if p.is_file() else pd.DataFrame()

    def save(self, file: str, rows: pd.DataFrame) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=self.folder)
        os.close(fd)
        try:
            rows.to_parquet(tmp)
            os.replace(tmp, self.path(file))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def merge_rows(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Store contents after a run: the new rows win on the same stamp. Sorted, so equal content gives equal files."""
    if old.empty:
        out = new
    elif new.empty:
        out = old
    else:
        both = pd.concat([old, new])
        out = both[~both.index.duplicated(keep="last")]
    return out.sort_index(kind="stable")


def compose(orig: pd.DataFrame, wall: pd.DatetimeIndex, keep: np.ndarray, store: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The staged frame: raw rows at ``keep`` plus the store rows that apply, in raw-stamp order.

    - A ``fill`` row applies where the kept raw rows have no bar at its stamp.
    - A ``replace`` row applies where the kept raw bar still has exactly the logged old prices.
    Returns (staged frame with the raw index labelling, the store rows used, on naive stamps).
    """
    kept_wall = wall[keep]
    if store.empty:
        return orig.iloc[keep], store
    present = store.index.isin(kept_wall)
    fill = store[(store["_kind"] == FILL) & ~present]
    rep = store[(store["_kind"] == REPLACE) & present]
    if len(rep):
        pos = kept_wall.get_indexer(rep.index)
        raw_px = orig.iloc[keep[pos]][PRICE_COLUMNS].to_numpy(dtype=float)
        rep = rep[(raw_px == rep[OLD_COLUMNS].to_numpy(dtype=float)).all(axis=1)]
    used = pd.concat([fill, rep]).sort_index(kind="stable") if len(rep) else fill
    if used.empty:
        return orig.iloc[keep], used
    keep_mask = ~kept_wall.isin(rep.index)
    base = orig.iloc[keep[keep_mask]]
    add = used[list(orig.columns)].astype(orig.dtypes.to_dict())
    add.index = like_index(used.index, orig.index)
    stamps = np.r_[kept_wall[keep_mask].asi8, used.index.as_unit("ns").asi8]
    order = np.argsort(stamps, kind="stable")
    return pd.concat([base, add]).iloc[order], used
