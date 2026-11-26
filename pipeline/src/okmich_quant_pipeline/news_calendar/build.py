"""fetch-news-calendar — orchestrator: joins every fetcher and writes the calendar parquet.

Run via:

    fetch-news-calendar                                  # default window/out
    fetch-news-calendar --year-min 2024 --year-max 2026 --out <path>
    python -m okmich_quant_pipeline.news_calendar.build  # equivalent module form

Pipeline
--------
1. Call ``fetch(year_min, year_max)`` on each fetcher in ``_DATE_FETCHERS`` and
   ``forexfactory.fetch_with_recorded``. Each returns a DataFrame with columns
   ``(release_date, event_name, source, impact_tier)``; ForexFactory also returns the instants it recorded.
2. Concatenate all fetcher frames.
3. For each row, derive ``timestamp_utc`` from ``release_times.to_utc(release_date, event_name)``
   — fetchers don't deal with times, the release-time tables do.
4. Drop duplicates on ``(timestamp_utc, event_name)``.
5. Check every row ForexFactory also lists against the instant it recorded, and the per-year coverage. Either
   failure raises before the write.
6. Sort by ``timestamp_utc`` and write atomically to ``--out``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

from okmich_quant_pipeline._io import atomic_write_parquet
from okmich_quant_pipeline.news_calendar._types import EventName, ImpactTier, Source
from okmich_quant_pipeline.news_calendar.fetchers import fomc, forexfactory, fred
from okmich_quant_pipeline.news_calendar.reader import DEFAULT_CALENDAR_PATH
from okmich_quant_pipeline.news_calendar.release_times import to_utc

logger = logging.getLogger(__name__)

# Fetchers that return dates only; ``forexfactory.fetch_with_recorded`` is called separately for its recorded instants.
_DATE_FETCHERS = (fomc, fred)

# Conservative per-year lower bounds per event family (~75% of nominal cadence). A healthy scrape
# clears these comfortably; a partial scrape (one fetcher returning a fraction of its rows) trips
# the check below before the truncated asset can overwrite a good one.
_MIN_EVENTS_PER_YEAR: dict[EventName, int] = {
    EventName.US_FOMC_STATEMENT: 6,      # 8/yr nominal
    EventName.US_NFP: 9,                 # ~12/yr
    EventName.US_CPI: 9,                 # ~12/yr
    EventName.EU_ECB_RATE_DECISION: 6,   # 8/yr (one FF request per month lists every meeting)
    EventName.UK_BOE_RATE_DECISION: 6,   # 8/yr
}


def _default_years() -> tuple[int, int]:
    """Default window: last year through next year, so a re-run after New Year rolls forward.

    The upper bound is ``today.year + 1`` because the Fed posts FOMC dates ~18 months ahead, so
    next year's schedule is already public; the lower bound keeps a year of history for context.
    Static years would silently exclude the current year once the calendar rolls past them.
    """
    y = dt.date.today().year
    return y - 1, y + 1


def _validate_coverage(final: pd.DataFrame, year_min: int, year_max: int, today_year: int | None = None) -> None:
    """Raise if any tracked event family is below its conservative minimum (partial-scrape guard).

    Runs *before* the write so a half-scraped calendar can never replace a good asset on disk.
    The floor is applied to **each fully-completed** year (strictly before the current year), keyed on
    ``release_date``. Checking each year rather than the window total catches a fetcher that silently
    drops whole years: two missing years inside a long window can otherwise hide behind the bonus rows
    of the others. Publication timing makes the current and future years legitimately partial — the Fed
    posts FOMC dates ~18 months ahead, but BLS/BEA/Census (NFP/CPI) and ForexFactory (ECB/BoE) only
    publish ~1 year out, and the current year is itself only part-elapsed. Those years still contribute
    their (bonus) rows; they're just not *required*, so the guard never false-trips on the window
    boundary. A window with no completed year must hold one year's floor in total.
    """
    today_year = today_year if today_year is not None else dt.date.today().year
    complete_years = range(year_min, min(year_max, today_year - 1) + 1)
    years = pd.to_datetime(final["release_date"]).dt.year
    counts = final.groupby([final["event_name"], years]).size()
    totals = final["event_name"].value_counts()
    shortfalls: list[str] = []
    for event, per_year in _MIN_EVENTS_PER_YEAR.items():
        if not complete_years:
            if int(totals.get(event.value, 0)) < per_year:
                shortfalls.append(f"{event.value}: {int(totals.get(event.value, 0))} < {per_year} expected")
            continue
        for year in complete_years:
            n = int(counts.get((event.value, year), 0))
            if n < per_year:
                shortfalls.append(f"{event.value} {year}: {n} < {per_year} expected")
    if shortfalls:
        raise ValueError(
            "news-calendar coverage check failed (partial scrape?) — refusing to overwrite:\n  "
            + "\n  ".join(shortfalls)
        )


def _check_release_times(final: pd.DataFrame, recorded: pd.DataFrame, now: pd.Timestamp | None = None) -> None:
    """Raise if a stamped row disagrees with the instant ForexFactory recorded for it (before the write).

    Rows ForexFactory does not list on the same date (FOMC press conferences it skipped, ALFRED revision dates) go
    unchecked; the log reports how many rows of each event were checked.
    """
    pairs = final[["event_name", "release_date", "timestamp_utc"]].merge(recorded, on=["event_name", "release_date"])
    forexfactory.check_recorded_times(pairs, now=now)
    checked = pairs.drop_duplicates(["event_name", "release_date"]).groupby("event_name").size()
    table = pd.DataFrame({"checked": checked, "rows": final.groupby("event_name").size()}).fillna(0).astype(int)
    logger.info("Release times checked against ForexFactory:\n" + table.to_string())


def build(year_min: int, year_max: int, out_path: Path) -> pd.DataFrame:
    """Run every fetcher, apply release times, dedupe, check times and coverage, write parquet (atomic)."""
    frames: list[pd.DataFrame] = []
    for mod in _DATE_FETCHERS:
        name = mod.__name__.rsplit(".", 1)[-1]
        df = mod.fetch(year_min, year_max)
        logger.info(f"  {name:<14} -> {len(df):4d} rows")
        frames.append(df)
    ff_rows, recorded = forexfactory.fetch_with_recorded(year_min, year_max)
    logger.info(f"  {'forexfactory':<14} -> {len(ff_rows):4d} rows")
    frames.append(ff_rows)
    combined = pd.concat(frames, ignore_index=True)

    # Apply release times. ``to_utc`` raises KeyError if an event has no
    # release-time convention — that's a programming error (new EventName
    # without a release_times.py entry) — and ValueError for a date before
    # the event's first checked release time. Surface both loudly.
    combined["timestamp_utc"] = combined.apply(
        lambda row: pd.Timestamp(to_utc(row["release_date"], row["event_name"])), axis=1
    )

    # Stringify the enum columns before writing — parquet has no native enum
    # type, and round-tripping enum-valued columns is brittle. Readers can
    # cast back to the enum if they need the typed form.
    combined["event_name"] = combined["event_name"].map(lambda e: e.value if isinstance(e, EventName) else str(e))
    combined["source"] = combined["source"].map(lambda s: s.value if isinstance(s, Source) else str(s))
    combined["impact_tier"] = combined["impact_tier"].map(lambda i: int(i) if isinstance(i, ImpactTier) else int(i))

    deduped = combined.drop_duplicates(subset=["timestamp_utc", "event_name"])
    final = deduped[["timestamp_utc", "event_name", "source", "impact_tier", "release_date"]].sort_values("timestamp_utc").reset_index(drop=True)

    _check_release_times(final, recorded)
    _validate_coverage(final, year_min, year_max)
    atomic_write_parquet(final, Path(out_path))
    logger.info(f"Wrote {len(final)} rows -> {out_path}")
    logger.info("By event_name:\n" + final.event_name.value_counts().to_string())
    logger.info("By source:\n" + final.source.value_counts().to_string())
    return final


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="Build the high-impact economic news calendar parquet.")
    default_min, default_max = _default_years()
    parser.add_argument("--year-min", type=int, default=default_min)
    parser.add_argument("--year-max", type=int, default=default_max)
    parser.add_argument("--out", default=str(DEFAULT_CALENDAR_PATH), help=f"Output parquet (default: {DEFAULT_CALENDAR_PATH})")
    args = parser.parse_args()
    build(args.year_min, args.year_max, Path(args.out))


if __name__ == "__main__":
    main()
