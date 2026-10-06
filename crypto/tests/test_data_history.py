"""Historical datasets through CCXT's unified REST methods: capability checks, paging, schemas, trades safeguards."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from okmich_quant_crypto.data import history as h
from okmich_quant_crypto.data.history import HistoryNotServedError
from okmich_quant_crypto.enums import Dataset
from okmich_quant_crypto.resilience import VenueUnsupportedError
from okmich_quant_crypto.utils import crypto_data_fetcher as f

from .fakes import FakeClock, FakeExchange

H = 3_600_000
PERP, SPOT = "BTC/USDT:USDT", "BTC/USDT"
D0 = int(datetime(2026, 6, 5, tzinfo=timezone.utc).timestamp() * 1000)


class FakeDataExchange(FakeExchange):
    """Adds the unified history methods, each served in pages of ``page`` records from ``since`` on."""

    def __init__(self, clock, page=3, **kw):
        super().__init__(clock, **kw)
        self.has.update({"fetchMarkOHLCV": True, "fetchFundingRateHistory": True, "fetchOpenInterestHistory": True,
                         "fetchLongShortRatioHistory": True, "fetchTrades": True})
        self.page = page
        self.funding, self.oi, self.lsr, self.public_trades = [], [], [], []
        self.trades_ignore_since = False
        self.requests = []

    def _serve(self, rows, since, limit):
        rows = sorted(rows, key=lambda r: r["timestamp"])
        if since is not None:
            rows = [r for r in rows if r["timestamp"] >= since]
        return [dict(r) for r in rows[: min(limit or self.page, self.page)]]

    async def fetch_funding_rate_history(self, symbol=None, since=None, limit=None, params=None):
        self.requests.append(("funding", since, limit))
        return self._serve(self.funding, since, limit)

    async def fetch_open_interest_history(self, symbol, timeframe="1h", since=None, limit=None, params=None):
        self.requests.append(("oi", timeframe, since, limit))
        return self._serve(self.oi, since, limit)

    async def fetch_long_short_ratio_history(self, symbol=None, timeframe=None, since=None, limit=None, params=None):
        self.requests.append(("lsr", timeframe, since, limit))
        return self._serve(self.lsr, since, limit)

    async def fetch_trades(self, symbol, since=None, limit=None, params=None):
        self.requests.append(("trades", since, limit))
        if self.trades_ignore_since:
            return [dict(r) for r in sorted(self.public_trades, key=lambda r: r["timestamp"])[-self.page:]]
        return self._serve(self.public_trades, since, limit)

    async def fetch_mark_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
        rows = await self.fetch_ohlcv(symbol + "#mark", timeframe, since, limit)
        return rows


def _dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


@pytest.fixture
def dex(monkeypatch):
    clock = FakeClock(D0 + 72 * H)
    ex = FakeDataExchange(clock)
    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    return ex


async def test_funding_history_is_paged_and_resumed(dex, tmp_path):
    dex.funding = [{"timestamp": D0 + i * 8 * H, "fundingRate": 0.0001 * (i + 1)} for i in range(7)]
    out = str(tmp_path / "funding.parquet")
    df = await f.fetch_dataset(Dataset.FUNDING_RATE, PERP, "x", _dt(D0), _dt(D0 + 48 * H), out)
    assert list(df.columns) == ["funding_rate"] and df.index.name == "date" and str(df.index.tz) == "UTC"
    assert len(df) == 7 and df["funding_rate"].iloc[-1] == pytest.approx(0.0007)
    assert len([r for r in dex.requests if r[0] == "funding"]) >= 3          # pages of 3
    dex.funding.append({"timestamp": D0 + 56 * H, "fundingRate": 0.0008})
    dex.requests.clear()
    df2 = await f.fetch_dataset(Dataset.FUNDING_RATE, PERP, "x", _dt(D0), _dt(D0 + 60 * H), out)
    assert len(df2) == 8 and dex.requests[0][1] == D0 + 48 * H                # resumed from the last stored record


async def test_open_interest_and_long_short_need_a_period(dex, tmp_path):
    dex.oi = [{"timestamp": D0 + i * H, "openInterestAmount": 100 + i, "openInterestValue": 1e6 + i} for i in range(5)]
    dex.lsr = [{"timestamp": D0 + i * H, "longShortRatio": 1.0 + i / 10} for i in range(5)]
    with pytest.raises(ValueError, match="needs --timeframe"):
        await f.fetch_dataset(Dataset.OPEN_INTEREST, PERP, "x", _dt(D0), _dt(D0 + 10 * H), str(tmp_path / "oi.pq"))
    oi = await f.fetch_dataset(Dataset.OPEN_INTEREST, PERP, "x", _dt(D0), _dt(D0 + 10 * H), str(tmp_path / "oi.pq"),
                               timeframe="1h")
    assert list(oi.columns) == ["open_interest_amount", "open_interest_value"] and len(oi) == 5
    assert all(r[1] == "1h" for r in dex.requests if r[0] == "oi")
    lsr = await f.fetch_dataset(Dataset.LONG_SHORT_RATIO, PERP, "x", _dt(D0), _dt(D0 + 10 * H),
                                str(tmp_path / "lsr.pq"), timeframe="1h")
    assert lsr["long_short_ratio"].tolist() == pytest.approx([1.0, 1.1, 1.2, 1.3, 1.4])


async def test_derivative_datasets_refuse_spot_symbols(dex, tmp_path):
    with pytest.raises(ValueError, match="derivatives only"):
        await f.fetch_dataset(Dataset.FUNDING_RATE, SPOT, "x", _dt(D0), _dt(D0 + H), str(tmp_path / "x.pq"))


async def test_unsupported_dataset_fails_before_any_request(dex, tmp_path):
    dex.has["fetchLongShortRatioHistory"] = False
    with pytest.raises(VenueUnsupportedError, match="fetchLongShortRatioHistory"):
        await f.fetch_dataset(Dataset.LONG_SHORT_RATIO, PERP, "x", _dt(D0), _dt(D0 + H), str(tmp_path / "x.pq"),
                              timeframe="1h")
    assert not [r for r in dex.requests if r[0] == "lsr"]


def test_timeframe_rules():
    with pytest.raises(ValueError, match="has no timeframe"):
        h.spec_for(Dataset.FUNDING_RATE, "1h")
    with pytest.raises(ValueError, match="needs --timeframe"):
        h.spec_for(Dataset.MARK_OHLCV, None)


async def test_mark_candles_use_the_mark_method(dex, tmp_path):
    rows = [[D0 + i * H, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, None] for i in range(4)]
    dex.add_candles(PERP + "#mark", "1h", rows)
    df = await f.fetch_dataset(Dataset.MARK_OHLCV, PERP, "x", _dt(D0), _dt(D0 + 10 * H), str(tmp_path / "m.pq"),
                               timeframe="1h")
    assert len(df) == 4 and df["volume"].isna().all()                          # mark candles carry no volume


def _trade(i, ts, side="buy"):
    return {"id": f"t{i}", "timestamp": ts, "side": side, "price": 100.0 + i, "amount": 0.01, "cost": 1.0 + i}


async def test_trades_are_written_as_daily_partitions_and_dedupe_on_resume(dex, tmp_path):
    # Three trades share one millisecond across a page boundary - none may be lost.
    ts = [D0 + 10, D0 + 20, D0 + 20, D0 + 20, D0 + 30, D0 + 26 * H, D0 + 26 * H + 5]
    dex.public_trades = [_trade(i, t, "sell" if i % 2 else "buy") for i, t in enumerate(ts)]
    out = tmp_path / "trades"
    written = await f.fetch_trades(PERP, "x", _dt(D0), _dt(D0 + 30 * H), str(out))
    stored = pd.concat(pd.read_parquet(p) for p in out.glob("*.parquet"))
    assert written == 7 and sorted(stored["trade_id"]) == sorted(t["id"] for t in dex.public_trades)
    files = sorted(p.name for p in out.glob("*.parquet"))
    assert files == ["2026-06-05.parquet", "2026-06-06.parquet"]
    day1 = pd.read_parquet(out / "2026-06-05.parquet")
    assert list(day1.columns) == ["trade_id", "side", "price", "amount", "cost"] and len(day1) == 5
    # Resume: re-reads from the newest stored trade; duplicates collapse on trade_id.
    dex.public_trades.append(_trade(99, D0 + 27 * H))
    await f.fetch_trades(PERP, "x", _dt(D0), _dt(D0 + 30 * H), str(out))
    day2 = pd.read_parquet(out / "2026-06-06.parquet")
    assert sorted(day2["trade_id"]) == ["t5", "t6", "t99"]


async def test_venue_that_ignores_since_for_trades_writes_nothing(dex, tmp_path):
    dex.public_trades = [_trade(i, D0 + 70 * H + i) for i in range(5)]   # only recent trades exist / are served
    dex.trades_ignore_since = True
    out = tmp_path / "trades"
    with pytest.raises(HistoryNotServedError, match="serve .*trade history"):
        await f.fetch_trades(PERP, "x", _dt(D0), _dt(D0 + 71 * H), str(out))
    assert not out.exists() or not list(out.glob("*.parquet"))


async def test_trades_start_gap_tolerance_is_configurable(dex, tmp_path):
    dex.public_trades = [_trade(i, D0 + 2 * H + i) for i in range(3)]       # illiquid: first trade 2h after start
    written = await f.fetch_trades(PERP, "x", _dt(D0), _dt(D0 + 3 * H), str(tmp_path / "t"),
                                   max_start_gap_minutes=180)
    assert written == 3


def test_ohlcv_page_limit_from_features(exchange):
    assert h.ohlcv_page_limit(exchange) == 3
    exchange.features = {}
    assert h.ohlcv_page_limit(exchange) == h.DEFAULT_OHLCV_PAGE
