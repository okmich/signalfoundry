"""Known bad windows per broker (append-only CSV, versioned in git next to the profile)."""
from __future__ import annotations

import dataclasses as dc
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .enums import WindowAction, WindowKind

ALL = "ALL"
COLUMNS = ["id", "symbols", "from_raw", "to_raw", "kind", "action", "evidence", "found_on", "export_date"]


@dc.dataclass(frozen=True)
class BadWindow:
    id: str
    symbols: frozenset[str] | None       # None = every symbol of the broker
    from_raw: pd.Timestamp               # inclusive, raw wall-clock
    to_raw: pd.Timestamp                 # exclusive
    kind: WindowKind
    action: WindowAction
    evidence: str
    found_on: dt.date
    export_date: dt.date

    def applies_to(self, symbol: str) -> bool:
        return self.symbols is None or symbol in self.symbols

    def mask(self, raw: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((raw >= self.from_raw) & (raw < self.to_raw))


def load_registry(path: Path) -> list[BadWindow]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate ids {df.loc[df['id'].duplicated(), 'id'].tolist()}")
    out = []
    for r in df.itertuples(index=False):
        syms = None if r.symbols.strip() == ALL else frozenset(s.strip() for s in r.symbols.split(";") if s.strip())
        w = BadWindow(r.id, syms, pd.Timestamp(r.from_raw), pd.Timestamp(r.to_raw), WindowKind(r.kind), WindowAction(r.action),
                      r.evidence, dt.date.fromisoformat(r.found_on), dt.date.fromisoformat(r.export_date))
        if w.to_raw <= w.from_raw:
            raise ValueError(f"{path}: window {w.id} has to_raw <= from_raw")
        out.append(w)
    return out


def windows_for(registry: list[BadWindow], symbol: str, actions: set[WindowAction] | None = None) -> list[BadWindow]:
    return [w for w in registry if w.applies_to(symbol) and (actions is None or w.action in actions)]


def in_windows(raw: pd.DatetimeIndex, windows: list[BadWindow]) -> np.ndarray:
    hit = np.zeros(len(raw), dtype=bool)
    for w in windows:
        hit |= w.mask(raw)
    return hit
