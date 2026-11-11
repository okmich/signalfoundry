"""Broker profiles: everything broker-specific the core needs, as data.

A profile says how to read a broker's bars. It holds no run settings: the market-data root, the bad-window registry,
the blind-after cutoff and the report folder are all inputs to a run.
"""
from __future__ import annotations

import dataclasses as dc
import functools
import os
from pathlib import Path
from collections.abc import Mapping

import pandas as pd

from .clock import ClockModel, wall_clock
from .enums import AssetClass

#: Env var naming the market-data root: the folder that contains ``raw/`` (e.g. ``D:\\data_dump\\market_data``).
MARKET_DATA_DIR_VAR = "OKMICH_QUANT_MARKET_DATA_DIR"


class MarketDataDirNotConfiguredError(RuntimeError):
    """Neither a data-dir argument nor OKMICH_QUANT_MARKET_DATA_DIR names an existing market-data root."""


def resolve_market_data_dir(data_dir: str | Path | None = None) -> Path:
    """The market-data root: the ``data_dir`` argument if given, else ``OKMICH_QUANT_MARKET_DATA_DIR``; else fail.

    The argument wins, so a run aimed at a copy or staging tree never silently falls back to the real one.
    """
    value = str(data_dir).strip() if data_dir else os.environ.get(MARKET_DATA_DIR_VAR, "").strip()
    source = "--data-dir" if data_dir else MARKET_DATA_DIR_VAR
    if not value:
        raise MarketDataDirNotConfiguredError(f"no market-data root: pass --data-dir or set {MARKET_DATA_DIR_VAR} "
                                              f"(the folder that contains raw/)")
    path = Path(value)
    if not path.is_dir():
        raise MarketDataDirNotConfiguredError(f"{source}={value} is not an existing folder")
    return path


@dc.dataclass(frozen=True)
class SymbolSpec:
    name: str                 # short name used everywhere in the tool, e.g. "EURUSD"
    file_stem: str            # parquet stem and MT5 symbol, e.g. "EURUSD.r"
    asset_class: AssetClass
    session: str              # key into BrokerProfile.sessions
    positive_prices: bool = True   # False for series that legitimately sit at or below 0 (premium index, basis, spreads)


@dc.dataclass(frozen=True)
class FetchSource:
    """How to reach the broker for refetches: an env file in OKMICH_QUANT_ENV_DIR, named .env.<broker>.<account>."""

    env_broker: str
    env_account: str


@dc.dataclass(frozen=True)
class ReferenceSpec:
    """An independent series in true UTC to check a symbol against (D13), e.g. Binance klines for broker crypto."""

    path: Path
    column: str = "close"
    note: str = ""

    def load(self) -> pd.Series:
        """The reference in UTC, deduplicated and sorted. Cached per file version (refetch verification asks often)."""
        return _load_reference(Path(self.path), self.column, Path(self.path).stat().st_mtime_ns).copy()


@functools.lru_cache(maxsize=8)
def _load_reference(path: Path, column: str, mtime_ns: int) -> pd.Series:
    s = pd.read_parquet(path, columns=[column])[column]
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return s[~s.index.duplicated(keep="last")].sort_index()


@dc.dataclass(frozen=True)
class BrokerProfile:
    name: str                                 # folder under <raw root>, e.g. "FXPIG-Server"
    bar_minutes: int
    clock: ClockModel
    symbols: Mapping[str, SymbolSpec]
    sessions: Mapping[str, object]            # session name -> SessionTemplate (calendar.py)
    fetch: FetchSource
    raw_root: Path | None = None              # <market-data root>/raw; None when bars are supplied in memory
    references: Mapping[str, ReferenceSpec] = dc.field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        if self.raw_root is None:
            raise MarketDataDirNotConfiguredError(f"{self.name}: no market-data root; pass --data-dir or set {MARKET_DATA_DIR_VAR}")
        return self.raw_root / self.name / str(self.bar_minutes)

    def symbol(self, name: str) -> SymbolSpec:
        try:
            return self.symbols[name]
        except KeyError:
            raise KeyError(f"{name!r} is not a {self.name} symbol; known: {sorted(self.symbols)}") from None

    def path(self, name: str) -> Path:
        return self.data_dir / f"{self.symbol(name).file_stem}.parquet"

    def load_raw(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        """The parquet as stored, with the index reduced to raw wall-clock digits (tz label dropped)."""
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(f"{self.name} {name}: no bar file at {path}")
        df = pd.read_parquet(path, columns=columns)
        if isinstance(df.index, pd.DatetimeIndex):       # anything else is rejected by symbol_data.check_bar_schema
            df.index = wall_clock(df.index)
            df.index.name = "raw"
        return df

    def symbols_in(self, asset_class: AssetClass) -> list[str]:
        return sorted(s.name for s in self.symbols.values() if s.asset_class is asset_class)
