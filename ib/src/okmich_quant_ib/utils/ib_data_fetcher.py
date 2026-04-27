"""``fetch-ib-data`` CLI — bulk historical OHLCV downloader for Interactive Brokers.

Paginates ``reqHistoricalDataAsync`` backward in chunks. Enforces the three IB
historical-data pacing rules via ``PacingTracker``:

  1. Global rate     — ≤60 requests / 600 s rolling window.
  2. Identical reqs  — ≥15 s between requests sharing the same identity key.
  3. Per-contract    — ≤5 same-contract requests / 2 s (IB triggers at ≥6).
"""
import argparse
import asyncio
import logging
import math
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from ib_async import IB, util

from ..contract import (
    DEFAULT_USE_RTH, DEFAULT_WHAT_TO_SHOW, IBContractConfig, SecType,
)
from ..functions.ib import connect_ib, resolve_contract
from ..timeframe_utils import BAR_SIZE_MINUTES, MAX_DURATION_DAYS

logger = logging.getLogger(__name__)

_PACING_WINDOW = 600
_MAX_TOTAL = 60
_IDENTICAL_COOLDOWN = 15.0
_BURST_WINDOW = 2.0
_BURST_LIMIT = 5
_MIN_SLEEP = 2.0


class PacingTracker:
    """Rolling-window pacing enforcer for the three IB historical-data rules."""

    def __init__(self):
        self._all: deque[float] = deque()
        self._last_identical: dict[tuple, float] = {}
        self._by_contract: dict[tuple, deque[float]] = {}

    @staticmethod
    def make_identical_key(symbol: str, sec_type: str, exchange: str, bar_size: str,
                           what_to_show: str, use_rth: bool, end_dt: datetime) -> tuple:
        bar_minutes = BAR_SIZE_MINUTES[bar_size]
        bucket = int(end_dt.timestamp()) // (bar_minutes * 60)
        return (symbol, sec_type, exchange, bar_size, what_to_show, use_rth, bucket)

    @staticmethod
    def make_contract_key(symbol: str, sec_type: str, exchange: str) -> tuple:
        return (symbol, sec_type, exchange)

    def _prune(self, now: float) -> None:
        global_cutoff = now - _PACING_WINDOW
        while self._all and self._all[0] < global_cutoff:
            self._all.popleft()
        burst_cutoff = now - _BURST_WINDOW
        for dq in self._by_contract.values():
            while dq and dq[0] < burst_cutoff:
                dq.popleft()

    async def wait_if_needed(self, identical_key: tuple, contract_key: tuple) -> None:
        """Sleep until all three pacing rules allow the next request."""
        while True:
            now = asyncio.get_event_loop().time()
            self._prune(now)

            if len(self._all) >= _MAX_TOTAL:
                await asyncio.sleep(2.0)
                continue

            last = self._last_identical.get(identical_key, 0.0)
            since_last = now - last
            if since_last < _IDENTICAL_COOLDOWN:
                await asyncio.sleep(_IDENTICAL_COOLDOWN - since_last + 0.1)
                continue

            if len(self._by_contract.get(contract_key, deque())) >= _BURST_LIMIT:
                await asyncio.sleep(0.5)
                continue

            break

    def record(self, identical_key: tuple, contract_key: tuple) -> None:
        now = asyncio.get_event_loop().time()
        self._all.append(now)
        self._last_identical[identical_key] = now
        self._by_contract.setdefault(contract_key, deque()).append(now)


async def fetch_and_save(
    symbol: str, contract_cfg: IBContractConfig, bar_size: str,
    start_date: datetime, end_date: datetime, output_path: str,
    what_to_show: Optional[str] = None, use_rth: Optional[bool] = None,
    host: str = "127.0.0.1", port: int = 4002, client_id: int = 10,
    allow_gaps: bool = False,
) -> None:
    """Fetch OHLCV bars from IB and write to Parquet.

    Paginates ``reqHistoricalDataAsync`` backward from ``end_date`` to ``start_date``.
    Reconnects with backoff on disconnect and resumes from the interrupted chunk.

    ``allow_gaps`` (default False) raises ``RuntimeError`` when any chunk fails
    after all retries, aborting before writing output to prevent silent gaps.
    """
    _what = what_to_show or DEFAULT_WHAT_TO_SHOW[contract_cfg.sec_type]
    _rth = use_rth if use_rth is not None else DEFAULT_USE_RTH[contract_cfg.sec_type]
    max_chunk_days = MAX_DURATION_DAYS[bar_size]
    pacing = PacingTracker()

    ib = IB()
    chunks: list[pd.DataFrame] = []
    end_utc = end_date.astimezone(timezone.utc)
    start_utc = start_date.astimezone(timezone.utc)
    cursor = end_utc

    try:
        await connect_ib(ib, host, port, client_id)
        contract = await resolve_contract(ib, symbol, contract_cfg)

        while cursor > start_utc:
            chunk_start = max(cursor - timedelta(days=max_chunk_days), start_utc)
            end_str = cursor.strftime("%Y%m%d %H:%M:%S UTC")
            duration = f"{max(1, math.ceil((cursor - chunk_start).total_seconds() / 86400))} D"
            identical_key = PacingTracker.make_identical_key(
                symbol, str(contract_cfg.sec_type), contract_cfg.exchange,
                bar_size, _what, _rth, cursor,
            )
            contract_key = PacingTracker.make_contract_key(
                symbol, str(contract_cfg.sec_type), contract_cfg.exchange
            )

            bars = None
            for attempt in range(5):
                # Pacing + min sleep are inside the retry loop so every attempt
                # respects all three rules — without this, retries after a 1s
                # backoff would violate the 15s identical-request cooldown.
                await pacing.wait_if_needed(identical_key, contract_key)
                await asyncio.sleep(_MIN_SLEEP)
                try:
                    pacing.record(identical_key, contract_key)
                    bars = await ib.reqHistoricalDataAsync(
                        contract, endDateTime=end_str, durationStr=duration,
                        barSizeSetting=bar_size, whatToShow=_what,
                        useRTH=_rth, formatDate=2, keepUpToDate=False,
                    )
                    break
                except Exception as e:
                    logger.warning(f"Chunk {end_str} attempt {attempt + 1} failed: {e}")
                    if not ib.isConnected():
                        await connect_ib(ib, host, port, client_id)
                        contract = await resolve_contract(ib, symbol, contract_cfg)
                    await asyncio.sleep(min(2.0 ** attempt, 30.0))

            if bars:
                df = util.df(bars)
                df["date"] = pd.to_datetime(df["date"], utc=True)
                chunks.append(df.set_index("date")[["open", "high", "low", "close", "volume"]])
                logger.info(f"Fetched {len(df)} bars ending {end_str}")
            else:
                msg = f"Chunk {end_str} failed after 5 attempts"
                if allow_gaps:
                    logger.error(f"{msg} — skipping (allow_gaps=True, output may have gaps)")
                else:
                    raise RuntimeError(
                        f"{msg}. Aborting to prevent silent data gaps. "
                        "Pass allow_gaps=True to continue."
                    )

            cursor = chunk_start
    finally:
        if ib.isConnected():
            ib.disconnect()

    if not chunks:
        logger.error("No data fetched — output not written")
        return

    result = pd.concat(chunks).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result[(result.index >= start_utc) & (result.index <= end_utc)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, compression="snappy")
    logger.info(f"Wrote {len(result)} bars → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fetch-ib-data",
        description="Download IB historical bars to a Parquet file.",
    )
    parser.add_argument("symbol", help="Ticker symbol (e.g. AAPL, EURUSD)")
    parser.add_argument("--sec-type", default="STK", choices=[t.value for t in SecType])
    parser.add_argument("--exchange", default="SMART")
    parser.add_argument("--currency", default="USD")
    parser.add_argument("--primary-exchange", default="",
                        help="Tiebreaker for SMART (e.g. NASDAQ)")
    parser.add_argument("--bar-size", required=True, choices=list(BAR_SIZE_MINUTES),
                        metavar="BAR_SIZE", help="e.g. '1 min', '5 mins', '1 hour'")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end", default=None, metavar="YYYY-MM-DD",
                        help="Default: today")
    parser.add_argument("--what-to-show", default=None,
                        help="Default: TRADES (STK/FUT/OPT), MIDPOINT (CASH/CFD)")
    parser.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=None,
                        help="--use-rth / --no-use-rth. Default: True for STK/OPT, False for rest")
    parser.add_argument("--output", required=True, help="Destination .parquet path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=10,
                        help="IB client ID (default 10, avoids collision with live IBEventLoop)")
    parser.add_argument("--allow-gaps", action="store_true", default=False,
                        help="Continue after failed chunks (default: abort to prevent silent gaps)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    contract_cfg = IBContractConfig(
        sec_type=SecType(args.sec_type),
        exchange=args.exchange,
        currency=args.currency,
        primary_exchange=args.primary_exchange,
    )
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc,
        )
    else:
        end = datetime.now(tz=timezone.utc)

    asyncio.run(fetch_and_save(
        symbol=args.symbol, contract_cfg=contract_cfg, bar_size=args.bar_size,
        start_date=start, end_date=end, output_path=args.output,
        what_to_show=args.what_to_show, use_rth=args.use_rth,
        host=args.host, port=args.port, client_id=args.client_id,
        allow_gaps=args.allow_gaps,
    ))
