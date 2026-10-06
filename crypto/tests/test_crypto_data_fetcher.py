"""fetch-crypto-data: on-disk schema, paging, forming-bar exclusion, weekends kept, resume / merge."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from okmich_quant_crypto.data import history as h
from okmich_quant_crypto.utils import crypto_data_fetcher as f

from .fakes import FakeClock, FakeExchange

H = 3_600_000
SYM = "BTC/USDT:USDT"
# Friday 2026-06-05 00:00 UTC: the range below spans a full weekend.
FRI = int(datetime(2026, 6, 5, tzinfo=timezone.utc).timestamp() * 1000)


def _rows(start, n):
    return [[start + i * H, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 3.0 + i] for i in range(n)]


def test_schema_is_the_exchange_candles_with_volume_and_utc_date_index():
    df = f.rows_to_frame(_rows(FRI, 3) + [_rows(FRI, 1)[0]])
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert "tick_volume" not in df.columns
    assert df.index.name == "date" and str(df.index.tz) == "UTC" and len(df) == 3
    assert df.index[0] == pd.Timestamp(FRI, unit="ms", tz="UTC")


async def test_paging_explicit_limit_weekends_kept_and_forming_bar_dropped():
    clock = FakeClock(FRI + 72 * H + 10 * 60_000)                    # Monday 00:10, the 00:00 bar is forming
    ex = FakeExchange(clock, ohlcv_limit=5)
    ex.add_candles(SYM, "1h", _rows(FRI, 72))
    ex.set_forming(SYM, "1h", [FRI + 72 * H, 1, 1, 1, 1, 0.5])
    rows = await h.fetch_candles_range(ex, "fetch_ohlcv", SYM, "1h", FRI, FRI + 100 * H, page_limit=5)
    assert len(rows) == 72 and rows[-1][0] == FRI + 71 * H           # Saturday and Sunday included, forming excluded
    assert {r[3] for r in ex.ohlcv_requests} == {5}
    df = f.rows_to_frame(rows)
    assert set(df.index.dayofweek) == {4, 5, 6}                      # Fri, Sat, Sun
    assert h.count_missing_candles(df, H) == 0


async def test_fetch_and_save_resumes_and_merges(tmp_path, monkeypatch):
    clock = FakeClock(FRI + 10 * H + 60_000)
    ex = FakeExchange(clock, ohlcv_limit=4)
    ex.add_candles(SYM, "1h", _rows(FRI, 10))

    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    out = str(tmp_path / "btc_1h.parquet")
    start = datetime.fromtimestamp(FRI / 1000, tz=timezone.utc)
    end = datetime.fromtimestamp((FRI + 5 * H) / 1000, tz=timezone.utc)
    first = await f.fetch_and_save(SYM, "bybit", "1h", start, end, out)
    assert len(first) == 6
    # Second run: later end, resumes from the last stored bar only.
    ex.ohlcv_requests.clear()
    end2 = datetime.fromtimestamp((FRI + 9 * H) / 1000, tz=timezone.utc)
    merged = await f.fetch_and_save(SYM, "bybit", "1h", start, end2, out)
    assert len(merged) == 10 and merged.index.is_monotonic_increasing and not merged.index.duplicated().any()
    assert ex.ohlcv_requests[0][2] == FRI + 5 * H
    on_disk = pd.read_parquet(out)
    assert on_disk.equals(merged)


async def test_misaligned_candles_are_refused(tmp_path, monkeypatch):
    clock = FakeClock(FRI + 10 * H)
    ex = FakeExchange(clock)
    ex.add_candles(SYM, "1h", [[FRI + 1_000, 1, 1, 1, 1, 1]])

    monkeypatch.setattr(f, "make_public_exchange", lambda *a, **k: ex)
    start = datetime.fromtimestamp(FRI / 1000, tz=timezone.utc)
    with pytest.raises(RuntimeError, match="off the 1h UTC grid"):
        await f.fetch_and_save(SYM, "bybit", "1h", start, start.replace(hour=5), str(tmp_path / "x.parquet"))
