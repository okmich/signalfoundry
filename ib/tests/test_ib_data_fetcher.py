"""Unit tests for ib_data_fetcher: PacingTracker semantics."""
import asyncio
from datetime import datetime, timezone

import pytest

from okmich_quant_ib.utils.ib_data_fetcher import PacingTracker


def test_identical_key_buckets_by_bar_minutes():
    end1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end2 = datetime(2024, 1, 1, 12, 4, 59, tzinfo=timezone.utc)  # same 5-min bucket
    end3 = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)   # next 5-min bucket
    k1 = PacingTracker.make_identical_key("AAPL", "STK", "SMART", "5 mins", "TRADES", True, end1)
    k2 = PacingTracker.make_identical_key("AAPL", "STK", "SMART", "5 mins", "TRADES", True, end2)
    k3 = PacingTracker.make_identical_key("AAPL", "STK", "SMART", "5 mins", "TRADES", True, end3)
    assert k1 == k2
    assert k1 != k3


def test_contract_key_excludes_bar_size():
    k1 = PacingTracker.make_contract_key("AAPL", "STK", "SMART")
    k2 = PacingTracker.make_contract_key("AAPL", "STK", "NASDAQ")
    assert k1 != k2


@pytest.mark.asyncio
async def test_record_increments_all_three_buckets():
    tracker = PacingTracker()
    ikey = PacingTracker.make_identical_key(
        "AAPL", "STK", "SMART", "5 mins", "TRADES", True,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    ckey = PacingTracker.make_contract_key("AAPL", "STK", "SMART")
    tracker.record(ikey, ckey)
    assert len(tracker._all) == 1
    assert ikey in tracker._last_identical
    assert len(tracker._by_contract[ckey]) == 1


@pytest.mark.asyncio
async def test_wait_returns_immediately_when_uncongested():
    tracker = PacingTracker()
    ikey = PacingTracker.make_identical_key(
        "AAPL", "STK", "SMART", "5 mins", "TRADES", True,
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    ckey = PacingTracker.make_contract_key("AAPL", "STK", "SMART")
    # Should not block when no prior requests have been recorded.
    await asyncio.wait_for(tracker.wait_if_needed(ikey, ckey), timeout=1.0)
