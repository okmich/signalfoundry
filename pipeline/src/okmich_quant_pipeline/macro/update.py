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
import sys
from pathlib import Path

import pandas as pd

from okmich_quant_pipeline._io import atomic_write_parquet
from okmich_quant_pipeline.macro._types import SERIES, FredSource, MacroSeries
from okmich_quant_pipeline.macro.fetchers import alfred, fred
from okmich_quant_pipeline.macro.fred_key import load_fred_key
from okmich_quant_pipeline.macro.metastore import MacroMetastore

logger = logging.getLogger(__name__)

DEFAULT_STORE = r"E:\data_dump\macro_data\daily"
DEFAULT_START = dt.date(2010, 1, 1)
DEFAULT_OVERLAP_DAYS = 60


def _merge(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Concat + dedupe on ``date`` keeping the newest row (absorbs FRED revisions).

    Dedupe BEFORE sort: ``new`` is concatenated after ``existing``, so on the concat row order
    ``keep="last"`` deterministically keeps the revised row. Sorting first would reorder ties
    (``sort_values`` is not stable), which could retain the stale ``existing`` value instead.
    """
    combined = new if existing is None or existing.empty else pd.concat([existing, new], ignore_index=True)
    return combined.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)


def update_series(series: MacroSeries, store_dir: Path, metastore: MacroMetastore, *,
                  full: bool, start: dt.date, end: dt.date, overlap_days: int, allow_shrink: bool = False) -> dict:
    """Fetch (incremental or full), merge with existing, atomic-write, update metastore."""
    spec = SERIES[series]
    path = store_dir / f"{series.value}.parquet"
    existing = pd.read_parquet(path) if path.exists() else None

    if full or existing is None:
        cosd = start
    else:
        last_obs = pd.to_datetime(existing["date"]).max().date()
        cosd = max(start, last_obs - dt.timedelta(days=overlap_days))

    if spec.source is FredSource.API:
        # Keyed JSON: output_type=4 for first-print vintages (idempotent — a settled obs always
        # returns the same first release, so the keep-last merge never clobbers it), else latest.
        new = alfred.fetch(series, cosd, end, api_key=load_fred_key(), output_type=4 if spec.vintage else 1)
    else:
        new = fred.fetch(series, cosd, end)
    merged = _merge(None if full else existing, new)
    if merged.empty:
        raise ValueError(f"no observations for {series.value} ({spec.fred_id}) in {cosd}..{end}")

    # Full re-fetch discards `existing`, so a partial provider response or a too-late `--start`
    # could replace a good historical asset with a truncated one (the empty guard above only
    # catches a *zero*-row fetch). Refuse to narrow the date span unless explicitly allowed.
    # Incremental runs merge with `existing` and so can never shrink — only `full` needs this.
    if full and existing is not None and not existing.empty and not allow_shrink:
        ex = pd.to_datetime(existing["date"])
        new_dates = pd.to_datetime(merged["date"])
        if new_dates.min() > ex.min() or new_dates.max() < ex.max():
            raise ValueError(
                f"{series.value}: full re-fetch would shrink coverage "
                f"[{ex.min().date()}..{ex.max().date()}] -> [{new_dates.min().date()}..{new_dates.max().date()}]; "
                f"pass --allow-shrink to override"
            )
    atomic_write_parquet(merged, path)

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
               end: dt.date | None = None, overlap_days: int = DEFAULT_OVERLAP_DAYS,
               allow_shrink: bool = False) -> dict[str, bool]:
    """Update every registered series; returns ``{series: success}``."""
    store_dir = Path(store_dir)
    end = end or dt.date.today()
    metastore = MacroMetastore(store_dir)
    logger.info(f"Macro store update [{'FULL' if full else f'incremental, overlap {overlap_days}d'}] -> {store_dir}")
    results: dict[str, bool] = {}
    for series in MacroSeries:
        try:
            rec = update_series(series, store_dir, metastore, full=full, start=start, end=end,
                                overlap_days=overlap_days, allow_shrink=allow_shrink)
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
    parser.add_argument("--allow-shrink", action="store_true", help="Permit a --full re-fetch to narrow a series' date span (e.g. an intentional narrower --start).")
    args = parser.parse_args()

    results = update_all(Path(args.out), full=args.full, start=args.start, end=args.end,
                         overlap_days=args.overlap_days, allow_shrink=args.allow_shrink)
    ok, total = sum(results.values()), len(results)
    print(f"\n{ok}/{total} series updated -> {args.out}")
    sys.exit(0 if ok == total else 1)


if __name__ == "__main__":
    main()
