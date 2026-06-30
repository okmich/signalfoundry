r"""fetch-macro-data — update the macro data store and its metastore.

Mirrors the fetch-mt5-data pattern: per-series parquet + a ``_metadata.json`` metastore, merged
(keep-last, absorbing FRED revisions) and written atomically on each run. One invocation updates
every registered series.

    fetch-macro-data                        # incremental: refresh the tail of every series
    fetch-macro-data --full                 # re-fetch full history from --start
    fetch-macro-data --out <dir> --overlap-days 60
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from okmich_quant_pipeline.macro._types import SERIES, MacroSeries
from okmich_quant_pipeline.macro.fetchers import fred
from okmich_quant_pipeline.macro.metastore import MacroMetastore

logger = logging.getLogger(__name__)

DEFAULT_STORE = r"D:\data_dump\macro_data\daily"
DEFAULT_START = dt.date(2010, 1, 1)
DEFAULT_OVERLAP_DAYS = 60


def _merge(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Concat + dedupe on ``date`` keeping the newest row (absorbs FRED revisions)."""
    combined = new if existing is None or existing.empty else pd.concat([existing, new], ignore_index=True)
    return combined.sort_values("date").drop_duplicates(subset="date", keep="last").reset_index(drop=True)


def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file then atomically replace (never leave a half-written series)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def update_series(series: MacroSeries, store_dir: Path, metastore: MacroMetastore, *,
                  full: bool, start: dt.date, end: dt.date, overlap_days: int) -> dict:
    """Fetch (incremental or full), merge with existing, atomic-write, update metastore."""
    spec = SERIES[series]
    path = store_dir / f"{series.value}.parquet"
    existing = pd.read_parquet(path) if path.exists() else None

    if full or existing is None:
        cosd = start
    else:
        last_obs = pd.to_datetime(existing["date"]).max().date()
        cosd = max(start, last_obs - dt.timedelta(days=overlap_days))

    new = fred.fetch(series, cosd, end)
    merged = _merge(None if full else existing, new)
    _atomic_write(merged, path)

    dates = pd.to_datetime(merged["date"])
    record = {
        "fred_id": spec.fred_id,
        "channel": spec.channel.value,
        "availability": repr(spec.availability),
        "first_obs": dates.min().date().isoformat(),
        "last_obs": dates.max().date().isoformat(),
        "n_obs": int(len(merged)),
        "last_update": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    metastore.update_series(series.value, record)
    return record


def update_all(store_dir: Path, *, full: bool = False, start: dt.date = DEFAULT_START,
               end: dt.date | None = None, overlap_days: int = DEFAULT_OVERLAP_DAYS) -> dict[str, bool]:
    """Update every registered series; returns ``{series: success}``."""
    store_dir = Path(store_dir)
    end = end or dt.date.today()
    metastore = MacroMetastore(store_dir)
    logger.info(f"Macro store update [{'FULL' if full else f'incremental, overlap {overlap_days}d'}] -> {store_dir}")
    results: dict[str, bool] = {}
    for series in MacroSeries:
        try:
            rec = update_series(series, store_dir, metastore, full=full, start=start, end=end, overlap_days=overlap_days)
            logger.info(f"  {series.value:<14} {rec['fred_id']:<14} {rec['n_obs']:5d} obs  {rec['first_obs']} .. {rec['last_obs']}")
            results[series.value] = True
        except Exception as exc:  # one bad series must not abort the rest
            logger.error(f"  {series.value:<14} FAILED: {exc}")
            results[series.value] = False
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="Update the macro data store (per-series parquet + _metadata.json metastore).")
    parser.add_argument("--out", default=DEFAULT_STORE, help=f"Store directory (default: {DEFAULT_STORE})")
    parser.add_argument("--full", action="store_true", help="Re-fetch full history (ignore existing); default is incremental tail refresh.")
    parser.add_argument("--start", type=lambda s: dt.date.fromisoformat(s), default=DEFAULT_START, help="Earliest observation date (default 2010-01-01).")
    parser.add_argument("--end", type=lambda s: dt.date.fromisoformat(s), default=dt.date.today(), help="Latest observation date (default today).")
    parser.add_argument("--overlap-days", type=int, default=DEFAULT_OVERLAP_DAYS, help="Incremental: re-fetch this many days before last_obs to absorb revisions.")
    args = parser.parse_args()

    results = update_all(Path(args.out), full=args.full, start=args.start, end=args.end, overlap_days=args.overlap_days)
    ok, total = sum(results.values()), len(results)
    print(f"\n{ok}/{total} series updated -> {args.out}")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
