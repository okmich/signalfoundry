"""``fetch-crypto-data`` CLI - historical datasets for ANY CCXT exchange through CCXT's unified REST methods.

Deliberately venue-agnostic and not limited to the supported-exchange list: it only reads public data, so research
data can come from venues this package does not trade on (yet). See ``okmich_quant_crypto.data.history`` for the
datasets, their on-disk schemas and what happens when a venue does not serve the requested history.

Candles are stored as the exchange returns them: every bar (weekends included - crypto trades 24/7), index ``date`` =
bar OPEN time in UTC, columns ``open, high, low, close, volume``. There is no ``tick_volume``: that is an MT5 concept.
Pagination uses ``since`` + an explicit page size (CCXT's ``paginate=True`` silently truncates long ranges without
``until``). The forming candle is never written; venue gaps are reported, never filled.
``okmich_quant_pipeline``'s dataset builder expects ``tick_volume`` and drops weekend bars, so it cannot consume these
files unchanged.

Single-file datasets resume from the last stored record and merge; trades are written as daily partition files
(``<output>/<YYYY-MM-DD>.parquet``) because a few weeks of trades on a liquid symbol is tens of millions of rows.
"""
import argparse
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..data.exchange import make_public_exchange
from ..data.history import (
    OHLCV_COLUMNS, candles_to_frame, check_supported, count_missing_candles, fetch_candles_range,
    fetch_records_range, ohlcv_page_limit, records_to_frame, spec_for,
)
from ..data.storage import latest_partition_ts, load_existing, merge_on_index, save_atomically, write_daily_partitions
from ..enums import Dataset, VenueEnvironment
from ..resilience import call_with_retry
from ..timeframe_utils import timeframe_to_ms

logger = logging.getLogger(__name__)

__all__ = ["OHLCV_COLUMNS", "fetch_and_save", "fetch_dataset", "fetch_trades", "main", "rows_to_frame"]

#: Backwards-compatible name for the candle frame builder.
rows_to_frame = candles_to_frame


def _utc_ms(dt: datetime, name: str) -> int:
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC); a naive datetime would be read as local time")
    return int(dt.timestamp() * 1000)


async def fetch_trades(symbol: str, exchange_id: str, start: datetime, end: datetime, output: str, *,
                       environment: VenueEnvironment = VenueEnvironment.LIVE, resume: bool = True,
                       page_limit: Optional[int] = None, max_start_gap_minutes: float = 60.0) -> int:
    """Public trades into daily partition files under ``output``, written as each UTC day completes (a long range
    needs memory for one day, and what was fetched before a failure stays on disk). Returns the trades written."""
    spec = spec_for(Dataset.TRADES, None)
    start_ms, end_ms = _utc_ms(start, "start"), _utc_ms(end, "end")
    written = {"rows": 0, "files": set()}

    def sink(batch: list[dict]) -> None:
        frame = records_to_frame(spec, batch)
        written["files"].update(write_daily_partitions(frame, output, spec.key))
        written["rows"] += len(frame)

    exchange = make_public_exchange(exchange_id, environment)
    try:
        await call_with_retry(exchange.load_markets, what=f"{exchange_id} load_markets", max_retries=5)
        check_supported(exchange, spec, symbol, None)
        if resume:
            last = latest_partition_ts(output)
            if last is not None and int(last.timestamp() * 1000) > start_ms:
                start_ms = int(last.timestamp() * 1000)
                logger.info("%s trades: resuming from %s", symbol, last)
        await fetch_records_range(exchange, spec, symbol, None, start_ms, end_ms, page_limit,
                                  int(max_start_gap_minutes * 60_000), sink=sink)
    finally:
        await exchange.close()
    logger.info("%s: %d trades written to %d daily file(s) under %s", symbol, written["rows"], len(written["files"]),
                output)
    return written["rows"]


async def fetch_dataset(dataset: Dataset, symbol: str, exchange_id: str, start: datetime, end: datetime,
                        output: str, *, timeframe: Optional[str] = None,
                        environment: VenueEnvironment = VenueEnvironment.LIVE, resume: bool = True,
                        page_limit: Optional[int] = None, max_start_gap_minutes: float = 60.0) -> pd.DataFrame:
    """Download one single-file dataset for ``symbol`` into ``output``; returns the merged frame on disk.

    ``resume=False`` (``--full-refetch``) re-downloads the whole ``start..end`` range but still MERGES with the file:
    stored history outside the range is never thrown away. Trades go through :func:`fetch_trades`.
    """
    if dataset is Dataset.TRADES:
        raise ValueError("trades are written as daily partitions; use fetch_trades()")
    spec = spec_for(dataset, timeframe)
    start_ms, end_ms = _utc_ms(start, "start"), _utc_ms(end, "end")
    existing = load_existing(output)
    exchange = make_public_exchange(exchange_id, environment)
    try:
        await call_with_retry(exchange.load_markets, what=f"{exchange_id} load_markets", max_retries=5)
        check_supported(exchange, spec, symbol, timeframe)
        if resume and existing is not None and not existing.empty:
            last_ms = int(existing.index[-1].timestamp() * 1000)
            if last_ms > start_ms:
                logger.info("%s %s: resuming from %s (existing data in %s)", symbol, dataset.value, existing.index[-1],
                            output)
                start_ms = last_ms
        if spec.candles:
            tf_ms = timeframe_to_ms(timeframe)
            start_ms = start_ms // tf_ms * tf_ms
            rows = await fetch_candles_range(exchange, spec.method, symbol, timeframe, start_ms, end_ms,
                                             page_limit or ohlcv_page_limit(exchange))
            new = candles_to_frame(rows)
        else:
            records = await fetch_records_range(exchange, spec, symbol, timeframe, start_ms, end_ms, page_limit,
                                                int(max_start_gap_minutes * 60_000))
            new = records_to_frame(spec, records)
    finally:
        await exchange.close()

    merged = merge_on_index(existing, new)
    if merged.empty:
        logger.error("%s %s: no data fetched - nothing written", symbol, dataset.value)
        return merged
    if spec.candles:
        tf_ms = timeframe_to_ms(timeframe)
        misaligned = (merged.index.asi8 // 1_000_000) % tf_ms != 0
        if misaligned.any():
            raise RuntimeError(f"{symbol}: {int(misaligned.sum())} candles are off the {timeframe} UTC grid - refusing "
                               f"to write data core's bar labels would disagree with")
        missing = count_missing_candles(merged, tf_ms)
        if missing:
            logger.warning("%s %s %s: %d candles missing in the venue's history (maintenance gaps) - not filled",
                           symbol, dataset.value, timeframe, missing)
    save_atomically(merged, output)
    logger.info("%s %s: saved %d rows (%s .. %s) to %s", symbol, dataset.value, len(merged), merged.index[0],
                merged.index[-1], output)
    return merged


async def fetch_and_save(symbol: str, exchange_id: str, timeframe: str, start: datetime, end: datetime,
                         output_path: str, *, environment: VenueEnvironment = VenueEnvironment.LIVE,
                         resume: bool = True) -> pd.DataFrame:
    """Candles for ``symbol`` (the original ``fetch-crypto-data`` behaviour)."""
    return await fetch_dataset(Dataset.OHLCV, symbol, exchange_id, start, end, output_path, timeframe=timeframe,
                               environment=environment, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fetch-crypto-data",
                                     description="Download historical market data (CCXT unified API) to Parquet.")
    parser.add_argument("symbol", help="CCXT unified symbol, e.g. BTC/USDT (spot) or BTC/USDT:USDT (USDT perp)")
    parser.add_argument("--dataset", default=Dataset.OHLCV.value, choices=[d.value for d in Dataset],
                        help="What to download (default: ohlcv)")
    parser.add_argument("--exchange", default="bybit", help="Any CCXT exchange id (default: bybit)")
    parser.add_argument("--timeframe", default=None,
                        help="Candle timeframe (up to 1d) or the period of open_interest / long_short_ratio, e.g. 5m")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=None, metavar="YYYY-MM-DD", help="Default: now")
    parser.add_argument("--output", required=True, help="Destination .parquet file (a directory for --dataset trades)")
    parser.add_argument("--environment", default=VenueEnvironment.LIVE.value,
                        choices=[e.value for e in VenueEnvironment], help="Venue copy to read (default: live)")
    parser.add_argument("--page-limit", type=int, default=None,
                        help="Records per request (default: venue maximum for candles, venue default otherwise)")
    parser.add_argument("--max-start-gap-minutes", type=float, default=60.0,
                        help="Trades: stop if the first page starts this much later than --start (default: 60)")
    parser.add_argument("--full-refetch", action="store_true", default=False,
                        help="Re-fetch the full --start..--end range instead of resuming (still merged with the file)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        # The whole --end day, to its last millisecond.
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999_000,
                                                              tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
    if args.timeframe is None and Dataset(args.dataset) in (Dataset.OHLCV, Dataset.MARK_OHLCV, Dataset.INDEX_OHLCV,
                                                            Dataset.PREMIUM_INDEX_OHLCV):
        parser.error(f"--timeframe is required for --dataset {args.dataset}")
    environment = VenueEnvironment(args.environment)
    if Dataset(args.dataset) is Dataset.TRADES:
        if args.timeframe:
            parser.error("--dataset trades has no timeframe")
        asyncio.run(fetch_trades(args.symbol, args.exchange, start, end, args.output, environment=environment,
                                 resume=not args.full_refetch, page_limit=args.page_limit,
                                 max_start_gap_minutes=args.max_start_gap_minutes))
        return
    asyncio.run(fetch_dataset(Dataset(args.dataset), args.symbol, args.exchange, start, end, args.output,
                              timeframe=args.timeframe, environment=environment, resume=not args.full_refetch,
                              page_limit=args.page_limit, max_start_gap_minutes=args.max_start_gap_minutes))
