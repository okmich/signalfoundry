"""Snapshots of a broker's raw files (the pre-promotion backup, SPEC §6.8) and D12, the history-rewrite diff.

A snapshot is a hash-verified copy of ``raw/<broker>/<tf>/*.parquet`` (+ ``_metadata.json``) in
``<state>/<broker>/<tf>/snapshots/<UTC time>/``, with ``snapshot.json`` listing every file and its sha256.

D12 compares today's raw files with a snapshot. Bars that already existed must not change: a changed OHLC, a bar that
vanished, or a bar inserted inside the snapshot's span means the export rewrote history (the 2026-10-29 splice was
that).
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .clock import wall_clock
from .enums import Action, FindingKind, Severity
from .findings import Finding
from .profile import BrokerProfile
from .state import StateLayout, sha256

PRICE_COLUMNS = ("open", "high", "low", "close")


def snapshot_broker(profile: BrokerProfile, layout: StateLayout, symbols: Iterable[str] | None = None) -> Path:
    names = sorted(symbols or profile.symbols)
    stamp = pd.Timestamp.now(tz="UTC").floor("s")
    folder = layout.snapshots / f"{stamp:%Y%m%dT%H%M%SZ}"
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    sources = [profile.path(s) for s in names]
    meta = profile.data_dir / "_metadata.json"
    if meta.is_file():
        sources.append(meta)
    for src in sources:
        dst = folder / src.name
        shutil.copy2(src, dst)
        h_src, h_dst = sha256(src), sha256(dst)
        if h_src != h_dst:
            raise OSError(f"snapshot copy of {src} does not match its source")
        files[src.name] = h_dst
    (folder / "snapshot.json").write_text(json.dumps({"broker": profile.name, "bar_minutes": profile.bar_minutes, "created_utc": str(stamp),
                                                      "source": str(profile.data_dir), "files": files}, indent=2), encoding="utf-8")
    return folder


def _wall(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = wall_clock(pd.DatetimeIndex(df.index)).as_unit("ns")
    return df[~df.index.duplicated(keep="first")].sort_index()


class SnapshotCoverageError(ValueError):
    """A symbol asked for is not in the snapshot: comparing nothing must not read as "no rewrite"."""


def history_diff(profile: BrokerProfile, snapshot_dir: Path, symbols: Iterable[str] | None = None) -> list[Finding]:
    """D12 findings: per symbol, windows where bars inside the snapshot's span changed, vanished or were inserted.

    Every requested symbol (default: all of the profile's) must be in the snapshot, else SnapshotCoverageError. Bars
    outside the snapshot's span, newer data or an older backfill, are not rewrites.
    """
    out = []
    bar = pd.Timedelta(minutes=profile.bar_minutes)
    names = sorted(symbols or profile.symbols)
    absent = [s for s in names if not (Path(snapshot_dir) / f"{profile.symbol(s).file_stem}.parquet").is_file()]
    if absent:
        held = sorted(s for s in profile.symbols if (Path(snapshot_dir) / f"{profile.symbol(s).file_stem}.parquet").is_file())
        raise SnapshotCoverageError(f"{len(absent)} requested symbol(s) not in snapshot {Path(snapshot_dir).name}: {', '.join(absent)}. "
                                    f"It holds {', '.join(held) or 'no symbol'}; pass those with --symbols, or take a new snapshot.")
    for s in names:
        snap_file = Path(snapshot_dir) / f"{profile.symbol(s).file_stem}.parquet"
        old = _wall(pd.read_parquet(snap_file, columns=list(PRICE_COLUMNS)))
        new = _wall(pd.read_parquet(profile.path(s), columns=list(PRICE_COLUMNS)))
        new = new[(new.index >= old.index.min()) & (new.index <= old.index.max())]   # older backfill / newer data: not rewrites
        both = old.index.intersection(new.index)
        a, b = old.loc[both].to_numpy(), new.loc[both].to_numpy()
        changed = both[~((a == b) | (np.isnan(a) & np.isnan(b))).all(axis=1)]
        vanished = old.index.difference(new.index)
        inserted = new.index.difference(old.index)
        events = pd.concat([pd.Series("changed", index=changed, dtype=object), pd.Series("vanished", index=vanished, dtype=object),
                            pd.Series("inserted", index=inserted, dtype=object)]).sort_index()
        if events.empty:
            continue
        t = events.index.asi8
        breaks = np.where(np.diff(t) > 12 * bar.value)[0] + 1          # a new window after 12 bars without events
        for seg in np.split(np.arange(len(t)), breaks):
            sel = events.iloc[seg]
            r0, r1 = sel.index.min(), sel.index.max() + bar
            u = profile.clock.to_utc(pd.DatetimeIndex([r0, r1]), s)
            counts = sel.value_counts().to_dict()
            out.append(Finding(s, FindingKind.HISTORY_REWRITE, Severity.CRITICAL, "D12", r0, r1, u[0], u[1], len(sel), Action.DROP,
                               {**{k: int(v) for k, v in counts.items()}, "snapshot": Path(snapshot_dir).name}))
    return out
