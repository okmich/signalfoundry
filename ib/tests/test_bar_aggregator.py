"""Unit tests for BarAggregator."""
from types import SimpleNamespace

import pytest

from okmich_quant_ib.bar_aggregator import BarAggregator


def _bar(epoch, o, h, l, c, v=10.0):
    return SimpleNamespace(time=epoch, open_=o, high=h, low=l, close=c, volume=v)


class _Recorder:
    def __init__(self):
        self.bars = []

    async def __call__(self, bar):
        self.bars.append(bar)


@pytest.mark.asyncio
async def test_emits_one_bar_per_5min_period():
    rec = _Recorder()
    agg = BarAggregator(target_minutes=5, on_bar_close=rec)
    # 60 5-sec bars fully cover one 5-minute period (300s); the 61st triggers emit.
    base = 1_700_000_000
    base = base - (base % 300)  # align to 5-min boundary
    for i in range(60):
        await agg.on_realtime_bar([_bar(base + i * 5, 100, 101, 99, 100.5)], True)
    assert rec.bars == []
    # The next bar opens period N+1 → emit period N's accumulator.
    await agg.on_realtime_bar([_bar(base + 60 * 5, 102, 103, 101, 102.5)], True)
    assert len(rec.bars) == 1
    emitted = rec.bars[0]
    assert emitted["partial"] is False
    assert emitted["sample_count"] == 60
    assert emitted["high"] == 101
    assert emitted["low"] == 99


@pytest.mark.asyncio
async def test_partial_flag_set_when_subscribe_mid_period():
    rec = _Recorder()
    agg = BarAggregator(target_minutes=5, on_bar_close=rec)
    base = 1_700_000_000
    base = base - (base % 300)
    # Subscribe mid-period — feed the LAST 30 of 60 5-sec bars (positions 30..59)
    # so consecutive arrivals never trip the gap-reset guard.
    for i in range(30, 60):
        await agg.on_realtime_bar([_bar(base + i * 5, 100, 101, 99, 100.5)], True)
    await agg.on_realtime_bar([_bar(base + 60 * 5, 102, 103, 101, 102.5)], True)
    assert len(rec.bars) == 1
    assert rec.bars[0]["partial"] is True
    assert rec.bars[0]["sample_count"] == 30


@pytest.mark.asyncio
async def test_duplicate_bar_dropped_silently():
    rec = _Recorder()
    agg = BarAggregator(target_minutes=5, on_bar_close=rec)
    base = 1_700_000_000
    base = base - (base % 300)
    await agg.on_realtime_bar([_bar(base, 100, 101, 99, 100.5)], True)
    await agg.on_realtime_bar([_bar(base, 100, 101, 99, 100.5)], True)
    assert agg._sample_count == 1


@pytest.mark.asyncio
async def test_large_gap_resets_state():
    rec = _Recorder()
    agg = BarAggregator(target_minutes=5, on_bar_close=rec, gap_reset_seconds=30)
    base = 1_700_000_000
    base = base - (base % 300)
    await agg.on_realtime_bar([_bar(base, 100, 101, 99, 100.5)], True)
    # 120s gap > 30s threshold → reset
    await agg.on_realtime_bar([_bar(base + 120, 200, 201, 199, 200.5)], True)
    assert agg._sample_count == 1
    assert agg._open == 200


@pytest.mark.asyncio
async def test_has_new_bar_false_ignored():
    rec = _Recorder()
    agg = BarAggregator(target_minutes=5, on_bar_close=rec)
    await agg.on_realtime_bar([_bar(1_700_000_000, 100, 101, 99, 100.5)], False)
    assert agg._sample_count == 0


@pytest.mark.asyncio
async def test_track_volume_false_drops_volume():
    rec = _Recorder()
    agg = BarAggregator(target_minutes=5, on_bar_close=rec, track_volume=False)
    base = 1_700_000_000
    base = base - (base % 300)
    for i in range(60):
        await agg.on_realtime_bar([_bar(base + i * 5, 100, 101, 99, 100.5, v=99)], True)
    await agg.on_realtime_bar([_bar(base + 60 * 5, 102, 103, 101, 102.5)], True)
    assert rec.bars[0]["volume"] == 0.0


def test_daily_target_rejected():
    with pytest.raises(ValueError, match="daily or longer"):
        BarAggregator(target_minutes=1440, on_bar_close=lambda b: None)
