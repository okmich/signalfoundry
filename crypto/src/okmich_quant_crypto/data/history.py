"""Historical datasets through CCXT's unified REST methods - venue-agnostic.

Every dataset is checked against ``exchange.has`` before the first request, paged forward from ``since`` and saved
in the package's parquet layout. What history exists is the VENUE's business: some venues serve years of funding or
trades, some only the latest page. The downloader never pretends otherwise:

* candles (last / mark / index / premium-index): the forming candle is never written; off-grid candles are refused;
* funding, open interest, long/short ratio: a series that starts later than requested is reported (listing date or
  the venue's retention), never padded;
* trades: some venues ignore ``since`` and return only their latest trades. If the first page starts more than
  ``max_start_gap`` after the requested start, the download stops with :class:`HistoryNotServedError` instead of
  writing a file that looks like history but is not. Raise the tolerance for illiquid symbols or a start before
  listing.

On-disk schemas (index ``date`` = UTC; values exactly as CCXT reports them, nothing renamed into MT5 terms):

==================  =========================================================================
candles             open, high, low, close, volume              (index: candle OPEN time)
funding_rate        funding_rate                                 (index: funding time)
open_interest       open_interest_amount, open_interest_value    (index: reading time)
long_short_ratio    long_short_ratio                             (index: reading time)
trades              trade_id, side (aggressor), price, amount, cost   (daily partition files)
==================  =========================================================================
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from ..enums import Dataset
from ..resilience import CryptoPermanentError, call_with_retry
from ..timeframe_utils import timeframe_to_ms, timeframe_to_seconds
from .exchange import require_capability

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
#: Page size for candles when CCXT's ``features`` does not state the venue maximum.
DEFAULT_OHLCV_PAGE = 100


class HistoryNotServedError(CryptoPermanentError):
    """The venue does not serve history for the requested range (it ignored ``since``)."""


@dataclass(frozen=True)
class DatasetSpec:
    dataset: Dataset
    has_key: str
    method: str
    what: str
    needs_timeframe: bool
    candles: bool = False
    columns: tuple = ()
    parse: Optional[Callable[[dict], dict]] = None
    #: Records that can share a timestamp are de-duplicated on this column instead of the index.
    key: Optional[str] = None


def _funding(r: dict) -> dict:
    return {"funding_rate": _num(r.get("fundingRate"))}


def _open_interest(r: dict) -> dict:
    return {"open_interest_amount": _num(r.get("openInterestAmount")),
            "open_interest_value": _num(r.get("openInterestValue"))}


def _long_short(r: dict) -> dict:
    return {"long_short_ratio": _num(r.get("longShortRatio"))}


def _trade(r: dict) -> dict:
    side = r.get("side")
    return {"trade_id": str(r.get("id")) if r.get("id") is not None else None,
            "side": str(side).lower() if side else None, "price": _num(r.get("price")),
            "amount": _num(r.get("amount")), "cost": _num(r.get("cost"))}


SPECS: dict[Dataset, DatasetSpec] = {
    Dataset.OHLCV: DatasetSpec(Dataset.OHLCV, "fetchOHLCV", "fetch_ohlcv", "candles", True, candles=True,
                               columns=tuple(OHLCV_COLUMNS)),
    Dataset.MARK_OHLCV: DatasetSpec(Dataset.MARK_OHLCV, "fetchMarkOHLCV", "fetch_mark_ohlcv", "mark-price candles",
                                    True, candles=True, columns=tuple(OHLCV_COLUMNS)),
    Dataset.INDEX_OHLCV: DatasetSpec(Dataset.INDEX_OHLCV, "fetchIndexOHLCV", "fetch_index_ohlcv",
                                     "index-price candles", True, candles=True, columns=tuple(OHLCV_COLUMNS)),
    Dataset.PREMIUM_INDEX_OHLCV: DatasetSpec(Dataset.PREMIUM_INDEX_OHLCV, "fetchPremiumIndexOHLCV",
                                             "fetch_premium_index_ohlcv", "premium-index candles", True, candles=True,
                                             columns=tuple(OHLCV_COLUMNS)),
    Dataset.FUNDING_RATE: DatasetSpec(Dataset.FUNDING_RATE, "fetchFundingRateHistory", "fetch_funding_rate_history",
                                      "funding-rate history", False, columns=("funding_rate",), parse=_funding),
    Dataset.OPEN_INTEREST: DatasetSpec(Dataset.OPEN_INTEREST, "fetchOpenInterestHistory",
                                       "fetch_open_interest_history", "open-interest history", True,
                                       columns=("open_interest_amount", "open_interest_value"), parse=_open_interest),
    Dataset.LONG_SHORT_RATIO: DatasetSpec(Dataset.LONG_SHORT_RATIO, "fetchLongShortRatioHistory",
                                          "fetch_long_short_ratio_history", "long/short-ratio history", True,
                                          columns=("long_short_ratio",), parse=_long_short),
    Dataset.TRADES: DatasetSpec(Dataset.TRADES, "fetchTrades", "fetch_trades", "public trades", False,
                                columns=("trade_id", "side", "price", "amount", "cost"), parse=_trade,
                                key="trade_id"),
}


def spec_for(dataset: Dataset, timeframe: Optional[str]) -> DatasetSpec:
    spec = SPECS[dataset]
    if spec.needs_timeframe and not timeframe:
        raise ValueError(f"dataset {dataset.value} needs --timeframe")
    if not spec.needs_timeframe and timeframe:
        raise ValueError(f"dataset {dataset.value} has no timeframe; drop --timeframe")
    return spec


def ohlcv_page_limit(exchange) -> int:
    """The venue's maximum candles per request from CCXT ``features`` (passed explicitly: CCXT defaults are small)."""
    features = getattr(exchange, "features", None) or {}
    blocks = [features.get("spot"), (features.get("swap") or {}).get("linear"), features.get("default")]
    for block in blocks:
        limit = ((block or {}).get("fetchOHLCV") or {}).get("limit")
        if limit:
            return int(limit)
    return DEFAULT_OHLCV_PAGE


# ---------------------------------------------------------------------- candles

def candles_to_frame(rows: list) -> pd.DataFrame:
    """CCXT OHLCV rows -> the package's candle schema (missing volume - e.g. index candles - stays NaN)."""
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="date"), dtype=float)
    df = pd.DataFrame([list(r[:6]) + [None] * (6 - len(r[:6])) for r in rows], columns=["ts"] + OHLCV_COLUMNS)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True), name="date")
    df = df.astype(float)
    return df[~df.index.duplicated(keep="last")].sort_index()


async def fetch_candles_range(exchange, method: str, symbol: str, timeframe: str, start_ms: int, end_ms: int,
                              page_limit: int, now_ms: Optional[int] = None) -> list:
    """Closed candles with open time in ``[start_ms, end_ms]``, paged forward with an explicit page size."""
    tf_ms = timeframe_to_ms(timeframe)
    now_ms = now_ms if now_ms is not None else exchange.milliseconds()
    end_ms = min(end_ms, (now_ms // tf_ms) * tf_ms - tf_ms)  # last CLOSED candle
    fetch = getattr(exchange, method)
    out: dict[int, list] = {}
    cursor = start_ms
    while cursor <= end_ms:
        rows = await call_with_retry(lambda: fetch(symbol, timeframe, cursor, page_limit),
                                     what=f"{symbol} {timeframe} {method} @{cursor}", max_retries=5)
        if not rows:
            logger.info("%s %s: venue returned no candles from %d; stopping", symbol, timeframe, cursor)
            break
        for row in rows:
            ts = int(row[0])
            if start_ms <= ts <= end_ms:
                out[ts] = row
        newest = max(int(r[0]) for r in rows)
        if newest < cursor:
            break
        cursor = newest + tf_ms
    return [out[ts] for ts in sorted(out)]


# ---------------------------------------------------------------------- records

def records_to_frame(spec: DatasetSpec, records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=list(spec.columns), index=pd.DatetimeIndex([], tz="UTC", name="date"))
    rows = [{"ts": int(r["timestamp"]), **spec.parse(r)} for r in records if r.get("timestamp") is not None]
    df = pd.DataFrame(rows, columns=["ts"] + list(spec.columns))
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), unit="ms", utc=True), name="date")
    if spec.key is None:
        df = df[~df.index.duplicated(keep="last")]
    return df.sort_index(kind="stable")


async def _page(exchange, spec: DatasetSpec, symbol: str, timeframe: Optional[str], since: int,
                page_limit: Optional[int]) -> list[dict]:
    fetch = getattr(exchange, spec.method)
    if spec.needs_timeframe:
        page = await call_with_retry(lambda: fetch(symbol, timeframe, since, page_limit),
                                     what=f"{symbol} {spec.method} @{since}", max_retries=5)
    else:
        page = await call_with_retry(lambda: fetch(symbol, since, page_limit),
                                     what=f"{symbol} {spec.method} @{since}", max_retries=5)
    return sorted((r for r in page or [] if r.get("timestamp") is not None), key=lambda r: int(r["timestamp"]))


#: How much earlier the second probe of a trades download asks for.
_TRADES_PROBE_MS = 6 * 3_600_000
#: A page whose newest record is this close to "now" is the venue's latest page.
_LATEST_PAGE_MS = 15 * 60_000


async def _assert_trade_history_served(exchange, spec: DatasetSpec, symbol: str, start_ms: int,
                                       page_limit: Optional[int]) -> None:
    """Probe with two different ``since`` values. A venue that honours ``since`` answers the earlier one with earlier
    trades; one that ignores it returns the same latest page twice. Catches the case the start-gap check cannot: a
    start only minutes ago, where the latest page covers most - but not all - of the range."""
    page_a = await _page(exchange, spec, symbol, None, start_ms, page_limit)
    if not page_a:
        return
    page_b = await _page(exchange, spec, symbol, None, start_ms - _TRADES_PROBE_MS, page_limit)
    same_page = bool(page_b) and page_a[0].get("id") == page_b[0].get("id") \
        and int(page_a[0]["timestamp"]) == int(page_b[0]["timestamp"])
    latest = int(page_a[-1]["timestamp"]) >= exchange.milliseconds() - _LATEST_PAGE_MS
    if same_page and latest and int(page_a[0]["timestamp"]) > start_ms:
        first_ts = pd.Timestamp(int(page_a[0]["timestamp"]), unit="ms", tz="UTC")
        raise HistoryNotServedError(
            f"{exchange.id} does not serve public trade history through CCXT: asked for trades from two different "
            f"start times, it returned the same latest page (from {first_ts}). Nothing was written.")


async def fetch_records_range(exchange, spec: DatasetSpec, symbol: str, timeframe: Optional[str], start_ms: int,
                              end_ms: int, page_limit: Optional[int], max_start_gap_ms: int,
                              sink: Optional[Callable[[list[dict]], None]] = None,
                              sink_rows: int = 200_000) -> list[dict]:
    """Records with timestamps in ``[start_ms, end_ms]``, paged forward on ``since``.

    With a ``sink``, records are handed over in batches - every completed UTC day, or every ``sink_rows`` - and not
    kept: a long trades download then needs memory for one day, and what was fetched before a failure is on disk.
    Returns the records only when there is no sink.

    An EMPTY page is not the end: several CCXT methods ask the venue for a fixed window after ``since`` (e.g. one
    hour of trades on Binance, ``period x limit`` of open interest on Bybit), so a quiet window or a start before
    listing comes back empty. The cursor then steps by one such window until data appears or the range ends.
    """
    if spec.dataset is Dataset.TRADES:
        await _assert_trade_history_served(exchange, spec, symbol, start_ms, page_limit)
    unit_ms = timeframe_to_seconds(timeframe) * 1000 if timeframe else 3_600_000
    empty_step = 3_600_000 if spec.dataset is Dataset.TRADES else unit_ms * max(1, page_limit or 50)
    out: dict[Any, dict] = {}
    cursor = start_ms
    first_page = True
    empty_pages = 0
    while cursor <= end_ms:
        page = await _page(exchange, spec, symbol, timeframe, cursor, page_limit)
        if not page:
            empty_pages += 1
            if empty_pages % 100 == 0:
                logger.info("%s %s: %d empty windows so far (cursor %s)", symbol, spec.dataset.value, empty_pages,
                            pd.Timestamp(cursor, unit="ms", tz="UTC"))
            cursor += empty_step
            continue
        earliest = int(page[0]["timestamp"])
        if first_page and earliest - start_ms > max_start_gap_ms:
            first_ts = pd.Timestamp(earliest, unit="ms", tz="UTC")
            message = (f"{exchange.id} {spec.what} for {symbol} starts at {first_ts}, "
                       f"{(earliest - start_ms) / 3_600_000:.1f}h after the requested start")
            if spec.dataset is Dataset.TRADES:
                raise HistoryNotServedError(message + ". The venue appears not to serve trade history for this range "
                                                      "(it returned its latest trades); nothing was written. Raise "
                                                      "--max-start-gap-minutes for an illiquid symbol or a start "
                                                      "before listing.")
            newest_first = int(page[-1]["timestamp"])
            if newest_first >= exchange.milliseconds() - max(3_600_000, newest_first - earliest):
                # The very first page already reaches "now": the venue (or CCXT's method for it) returned its LATEST
                # records instead of the ones from ``since`` - e.g. CCXT's Bybit long/short ratio passes no start time.
                logger.warning("%s: it returned only its most recent records - the venue (or CCXT) ignores `since` "
                               "for this dataset, or the market is newer than the requested start. Older history is "
                               "not available this way.", message)
            else:
                logger.warning("%s (listing date, or the venue's retention limit)", message)
        first_page = False
        for r in page:
            ts = int(r["timestamp"])
            if start_ms <= ts <= end_ms:
                key = r.get("id") if spec.key else ts
                out[key if key is not None else (ts, r.get("price"), r.get("amount"))] = r
        newest = int(page[-1]["timestamp"])
        if sink is not None:
            _drain(out, sink, sink_rows, newest)
        if newest < cursor:
            break
        # Trades can share a millisecond: re-read from the newest one (de-duplicated by id) unless that stalls.
        next_cursor = newest if spec.key else newest + 1
        cursor = next_cursor if next_cursor > cursor else cursor + 1
    if sink is not None:
        _drain(out, sink, 0, None)
        return []
    return sorted(out.values(), key=lambda r: int(r["timestamp"]))


def _drain(out: dict, sink: Callable[[list[dict]], None], sink_rows: int, newest_ms: Optional[int]) -> None:
    """Hand records of completed UTC days (or everything, past ``sink_rows`` / at the end) to ``sink``."""
    if not out:
        return
    if newest_ms is None or len(out) >= sink_rows > 0:
        batch_keys = list(out)
    else:
        day_start = (newest_ms // 86_400_000) * 86_400_000
        batch_keys = [k for k, r in out.items() if int(r["timestamp"]) < day_start]
    if not batch_keys:
        return
    batch = sorted((out.pop(k) for k in batch_keys), key=lambda r: int(r["timestamp"]))
    sink(batch)


def check_supported(exchange, spec: DatasetSpec, symbol: str, timeframe: Optional[str]) -> None:
    require_capability(exchange, spec.has_key, spec.what)
    market = exchange.market(symbol)
    if spec.dataset in (Dataset.MARK_OHLCV, Dataset.INDEX_OHLCV, Dataset.PREMIUM_INDEX_OHLCV, Dataset.FUNDING_RATE,
                        Dataset.OPEN_INTEREST, Dataset.LONG_SHORT_RATIO) and not market.get("contract"):
        raise ValueError(f"{spec.dataset.value} exists for derivatives only; {symbol} is a {market.get('type')} market")
    if spec.candles and timeframe and getattr(exchange, "timeframes", None) and timeframe not in exchange.timeframes:
        raise ValueError(f"{exchange.id} does not offer timeframe {timeframe!r}; offered: "
                         f"{sorted(exchange.timeframes)}")


def count_missing_candles(df: pd.DataFrame, tf_ms: int) -> int:
    if len(df) < 2:
        return 0
    expected = (df.index[-1] - df.index[0]).total_seconds() * 1000 / tf_ms + 1
    return int(round(expected)) - len(df)


def _num(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
