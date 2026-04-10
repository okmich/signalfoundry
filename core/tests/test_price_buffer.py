import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from okmich_quant_core.price_buffer import PriceBuffer

UTC = timezone.utc
TF = 5  # default timeframe minutes


def make_bars(start_dt: datetime, n: int, tf_minutes: int = TF) -> pd.DataFrame:
    """Create n OHLCV bars with a UTC-aware DatetimeIndex."""
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=UTC)
    dates = pd.date_range(start=start_dt, periods=n, freq=f"{tf_minutes}min")
    return pd.DataFrame(
        {
            "open":        np.random.uniform(1.08, 1.09, n),
            "high":        np.random.uniform(1.09, 1.10, n),
            "low":         np.random.uniform(1.07, 1.08, n),
            "close":       np.random.uniform(1.08, 1.09, n),
            "tick_volume": np.random.randint(100, 1000, n).astype(float),
            "spread":      np.random.randint(1, 5, n).astype(float),
        },
        index=dates,
    )


def next_bar_open(data: pd.DataFrame, tf_minutes: int = TF) -> datetime:
    """Return the open time of the bar AFTER the last bar in data.

    Passing this as current_dt to update() makes data.index[-1] the stable boundary,
    so all bars in data are accepted.
    """
    return data.index[-1] + timedelta(minutes=tf_minutes)


class TestPriceBuffer(unittest.TestCase):

    def setUp(self):
        self.symbol = "EURUSD"
        self.buffer_size = 100
        self.buffer = PriceBuffer(self.symbol, TF, self.buffer_size, timeframe_minutes=TF)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def test_initialization(self):
        self.assertEqual(self.buffer.symbol, self.symbol)
        self.assertEqual(self.buffer.buffer_size, self.buffer_size)
        self.assertEqual(self.buffer.timeframe_minutes, TF)
        self.assertTrue(self.buffer.is_empty())
        self.assertIsNone(self.buffer._data_tz)
        self.assertIsNone(self.buffer._stable_boundary)

    def test_requires_timeframe_minutes(self):
        with self.assertRaises(ValueError):
            PriceBuffer("X", 5, 100)  # missing timeframe_minutes

    # ------------------------------------------------------------------
    # Timezone — inferred from first data, never pre-set
    # ------------------------------------------------------------------

    def test_timezone_inferred_from_first_data(self):
        """_data_tz is None until the first update(); thereafter it matches the data's tz."""
        self.assertIsNone(self.buffer._data_tz)
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        self.buffer.update(data, next_bar_open(data))
        self.assertIsNotNone(self.buffer._data_tz)
        self.assertEqual(str(self.buffer._data_tz), str(UTC))

    def test_timezone_preserved_across_reset(self):
        """_data_tz survives reset() — the broker feed timezone doesn't change on reconnect."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        self.buffer.update(data, next_bar_open(data))
        self.buffer.reset()
        self.assertIsNotNone(self.buffer._data_tz)
        self.assertTrue(self.buffer.is_empty())

    def test_naive_data_index_raises(self):
        """A timezone-naive data index must raise ValueError — no silent localization."""
        naive_data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        naive_data.index = naive_data.index.tz_localize(None)
        with self.assertRaises(ValueError):
            self.buffer.update(naive_data, datetime(2025, 1, 1, 1, 0, tzinfo=UTC))

    def test_timezone_mismatch_converts_incoming_data(self):
        """Data with a different tz is converted to the buffer's established tz."""
        data_utc = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        self.buffer.update(data_utc, next_bar_open(data_utc))

        # Supply data in a different tz
        eastern = timezone(timedelta(hours=-5))
        data_eastern = make_bars(datetime(2025, 1, 1, 2, 0, tzinfo=eastern), 5)
        result = self.buffer.update(data_eastern, next_bar_open(data_eastern).astimezone(UTC))
        self.assertTrue(result)
        # Buffer index should all be in UTC
        self.assertEqual(str(self.buffer.data.index.tz), str(UTC))

    # ------------------------------------------------------------------
    # Stable boundary — forming bar is never stored or returned
    # ------------------------------------------------------------------

    def test_stable_boundary_strips_forming_bar(self):
        """If current_dt puts the last data bar in the 'forming' window, it is excluded."""
        # 10 bars from 00:00 → last bar 00:45
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        last_complete = data.index[-2]  # 00:40
        forming_bar  = data.index[-1]   # 00:45

        # current_dt = 00:45:30 → boundary = 00:40 → 00:45 bar is forming and stripped
        current_dt = forming_bar + timedelta(seconds=30)
        self.buffer.update(data, current_dt)

        self.assertEqual(self.buffer.last_bar_time, last_complete)
        self.assertNotIn(forming_bar, self.buffer.data.index)

    def test_get_data_only_returns_stable_bars(self):
        """get_data() never includes a bar beyond the stable boundary at update time."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        stable_boundary = data.index[-2]  # 00:40

        # current_dt within the 00:45 bar → boundary is 00:40
        current_dt = data.index[-1] + timedelta(seconds=1)
        self.buffer.update(data, current_dt)

        returned = self.buffer.get_data()
        self.assertTrue((returned.index <= stable_boundary).all())

    def test_update_all_bars_beyond_boundary_is_not_an_error(self):
        """If broker sends data from the future (all beyond boundary), update returns True,
        buffer contents are unchanged, but last_update_dt and _stable_boundary are refreshed."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        self.buffer.update(data, next_bar_open(data))
        size_before = len(self.buffer.data)
        last_bar_before = self.buffer.last_bar_time

        # Simulate broker sending only the forming bar
        forming = make_bars(data.index[-1] + timedelta(minutes=TF), 1)
        current_dt = forming.index[0] + timedelta(seconds=1)  # still inside forming bar
        result = self.buffer.update(forming, current_dt)

        self.assertTrue(result)
        self.assertEqual(len(self.buffer.data), size_before)
        self.assertEqual(self.buffer.last_bar_time, last_bar_before)
        self.assertEqual(self.buffer.last_update_dt, current_dt)  # monitoring timestamp updated

    def test_naive_datetime_raises_in_get_fetch_params(self):
        """Timezone-naive current_dt must raise ValueError — no silent guessing."""
        with self.assertRaises(ValueError):
            self.buffer.get_fetch_params(datetime(2025, 1, 1, 12, 0))  # no tzinfo

    def test_naive_datetime_raises_in_update(self):
        """Timezone-naive current_dt must raise ValueError in update()."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        with self.assertRaises(ValueError):
            self.buffer.update(data, datetime(2025, 1, 1, 1, 0))  # no tzinfo

    def test_count_zero_when_buffer_already_current(self):
        """When last_bar_time is already at the stable boundary, count=0 — no fetch needed."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 10)
        self.buffer.update(data, next_bar_open(data))

        # Call get_fetch_params at a time where boundary == last_bar_time
        # i.e. we are exactly one bar ahead of last stored bar (forming bar is open)
        current_dt = data.index[-1] + timedelta(minutes=TF, seconds=30)
        params = self.buffer.get_fetch_params(current_dt)

        self.assertEqual(params["count"], 0)

    # ------------------------------------------------------------------
    # get_fetch_params — full reload
    # ------------------------------------------------------------------

    def test_empty_buffer_requests_full_reload(self):
        current_dt = datetime(2025, 1, 1, 1, 0, tzinfo=UTC)
        params = self.buffer.get_fetch_params(current_dt)
        self.assertIsNone(params["start_dt"])
        self.assertEqual(params["count"], self.buffer_size + 1)

    def test_stale_buffer_requests_full_reload(self):
        """Gap > 2× timeframe between boundary and last_bar triggers full reload."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        self.buffer.update(data, next_bar_open(data))

        # Jump 30 minutes ahead — gap (30 min) >> 2×5 min
        stale_dt = data.index[-1] + timedelta(minutes=30)
        params = self.buffer.get_fetch_params(stale_dt)
        self.assertIsNone(params["start_dt"])
        self.assertEqual(params["count"], self.buffer_size + 1)

    # ------------------------------------------------------------------
    # get_fetch_params — incremental fetch
    # ------------------------------------------------------------------

    def test_incremental_fetch_start_is_next_bar_after_last_stored(self):
        """start_dt must be last_bar_time + 1 timeframe, not last_bar_time itself."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        self.buffer.update(data, next_bar_open(data))

        last_stored = self.buffer.last_bar_time
        expected_start = last_stored + timedelta(minutes=TF)

        # current_dt two bars after last stored → boundary = last_stored + 1 bar
        current_dt = last_stored + timedelta(minutes=TF * 2)
        params = self.buffer.get_fetch_params(current_dt)

        self.assertEqual(params["start_dt"], expected_start)

    def test_incremental_fetch_count_one_new_bar(self):
        """One bar has elapsed → count=1."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        self.buffer.update(data, next_bar_open(data))

        last_stored = self.buffer.last_bar_time
        # Two bars after last_stored = boundary is one bar ahead
        current_dt = last_stored + timedelta(minutes=TF * 2)
        params = self.buffer.get_fetch_params(current_dt)

        self.assertEqual(params["count"], 1)

    def test_incremental_fetch_count_two_missed_bars(self):
        """Two bars have elapsed (exactly at the 2×TF stale threshold) → count=2.

        Note: missing 3+ bars (gap > 2×TF) triggers a full reload by design,
        so the incremental path only covers up to 2 missed bars.
        """
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        self.buffer.update(data, next_bar_open(data))

        last_stored = self.buffer.last_bar_time
        # 3 bars after last_stored → boundary = last_stored + 2*TF (exactly at threshold)
        current_dt = last_stored + timedelta(minutes=TF * 3)
        params = self.buffer.get_fetch_params(current_dt)

        # next_bar_start = last_stored + TF
        # boundary       = last_stored + 2*TF
        # gap            = TF → count = 1+1 = 2
        self.assertIsNotNone(params["start_dt"])  # incremental, not full reload
        self.assertEqual(params["count"], 2)

    def test_incremental_fetch_boundary_reflects_clock(self):
        """rounded_current_dt in the result is the last complete bar boundary."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        self.buffer.update(data, next_bar_open(data))

        last_stored = self.buffer.last_bar_time
        # Exactly 2 bars after last_stored: boundary = last_stored + 1 bar
        current_dt = last_stored + timedelta(minutes=TF * 2)
        params = self.buffer.get_fetch_params(current_dt)

        expected_boundary = last_stored + timedelta(minutes=TF)
        self.assertEqual(params["rounded_current_dt"], expected_boundary)

    def test_all_strategies_get_same_fetch_params_with_microsecond_variance(self):
        """Microsecond differences in current_dt at the same bar boundary all produce
        the same start_dt and count — critical for multi-strategy consistency."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        last_stored = data.index[-1]

        buffers = [PriceBuffer(f"S{i}", TF, self.buffer_size, timeframe_minutes=TF) for i in range(4)]
        for buf in buffers:
            buf.update(data, next_bar_open(data))

        base_dt = last_stored + timedelta(minutes=TF * 2)
        delays_us = [441000, 469000, 482000, 496000]
        results = [
            buf.get_fetch_params(base_dt.replace(microsecond=d))
            for buf, d in zip(buffers, delays_us)
        ]

        for r in results[1:]:
            self.assertEqual(r["start_dt"], results[0]["start_dt"])
            self.assertEqual(r["count"], results[0]["count"])
            self.assertEqual(r["rounded_current_dt"], results[0]["rounded_current_dt"])

    # ------------------------------------------------------------------
    # update() — merge and deduplication
    # ------------------------------------------------------------------

    def test_initial_update_fills_buffer(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        result = self.buffer.update(data, next_bar_open(data))
        self.assertTrue(result)
        self.assertFalse(self.buffer.is_empty())
        self.assertEqual(len(self.buffer.get_data()), 101)

    def test_buffer_trims_to_buffer_size_plus_one(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 200)
        self.buffer.update(data, next_bar_open(data))
        self.assertEqual(len(self.buffer.get_data()), self.buffer_size + 1)

    def test_incremental_update_appends_and_trims(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 101)
        self.buffer.update(data, next_bar_open(data))

        new_data = make_bars(data.index[-1] + timedelta(minutes=TF), 3)
        result = self.buffer.update(new_data, next_bar_open(new_data))

        self.assertTrue(result)
        self.assertEqual(len(self.buffer.get_data()), self.buffer_size + 1)
        self.assertEqual(self.buffer.last_bar_time, new_data.index[-1])

    def test_deduplication_new_data_wins(self):
        """On overlapping timestamps the most recently supplied values are kept."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        self.buffer.update(data, next_bar_open(data))

        # Overlap: resend last 5 bars with different close prices
        overlap = data.iloc[-5:].copy()
        sentinel = 9.99999
        overlap["close"] = sentinel
        result = self.buffer.update(overlap, next_bar_open(data))

        self.assertTrue(result)
        # The new (sentinel) values must have replaced the old ones
        tail = self.buffer.get_data().iloc[-5:]
        self.assertTrue((tail["close"] == sentinel).all())

    def test_deduplication_no_duplicate_timestamps_in_buffer(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        self.buffer.update(data, next_bar_open(data))
        # Re-send 10 overlapping + 5 new bars
        overlap_and_new = make_bars(data.index[-10], 15)
        self.buffer.update(overlap_and_new, next_bar_open(overlap_and_new))
        self.assertFalse(self.buffer.get_data().index.duplicated().any())

    # ------------------------------------------------------------------
    # update() — validation
    # ------------------------------------------------------------------

    def test_validation_rejects_nan_values(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        data.loc[data.index[25], "close"] = np.nan
        result = self.buffer.update(data, next_bar_open(data))
        self.assertFalse(result)
        self.assertTrue(self.buffer.is_empty())

    def test_validation_rejects_sub_timeframe_gap(self):
        """A bar with a 2-minute gap from its predecessor (in a 5-min buffer) fails.

        The bad gap is injected mid-data so it falls within the stable boundary
        and is not filtered before validation.
        """
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        # Replace bar[25]'s timestamp so it's only 2 min after bar[24]
        new_index = data.index.tolist()
        new_index[25] = new_index[24] + timedelta(minutes=2)
        data.index = pd.DatetimeIndex(new_index)
        result = self.buffer.update(data, next_bar_open(data))
        self.assertFalse(result)
        self.assertTrue(self.buffer.is_empty())

    def test_validation_accepts_large_session_gaps(self):
        """Weekend / holiday gaps (e.g. 60+ hours for H1) are accepted."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        # 3-day gap — simulates a weekly open
        weekend_start = data.index[-1] + timedelta(days=3)
        after_weekend = make_bars(weekend_start, 3)
        all_data = pd.concat([data, after_weekend])
        result = self.buffer.update(all_data, next_bar_open(after_weekend))
        self.assertTrue(result)

    def test_duplicate_rows_deduplicated_before_validation(self):
        """A DataFrame with a duplicate row still passes — dedup runs before validation."""
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        dupe = pd.concat([data, data.iloc[25:26]])
        result = self.buffer.update(dupe, next_bar_open(data))
        self.assertTrue(result)
        self.assertFalse(self.buffer.get_data().index.duplicated().any())

    # ------------------------------------------------------------------
    # _last_complete_bar_boundary
    # ------------------------------------------------------------------

    def test_boundary_exact_bar_open_time(self):
        """At exactly 12:00:00 the 12:00 bar is forming — boundary = 11:55."""
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        result = self.buffer._last_complete_bar_boundary(dt)
        self.assertEqual(result, datetime(2025, 1, 1, 11, 55, tzinfo=UTC))

    def test_boundary_milliseconds_after_bar_open(self):
        """12:00:00.441 — still inside the 12:00 bar — boundary = 11:55."""
        dt = datetime(2025, 1, 1, 12, 0, 0, 441000, tzinfo=UTC)
        result = self.buffer._last_complete_bar_boundary(dt)
        self.assertEqual(result, datetime(2025, 1, 1, 11, 55, tzinfo=UTC))

    def test_boundary_middle_of_bar(self):
        """12:03:47 — inside the 12:00 bar — boundary = 11:55."""
        dt = datetime(2025, 1, 1, 12, 3, 47, tzinfo=UTC)
        result = self.buffer._last_complete_bar_boundary(dt)
        self.assertEqual(result, datetime(2025, 1, 1, 11, 55, tzinfo=UTC))

    def test_boundary_just_before_next_bar(self):
        """12:04:59.999 — still inside 12:00 bar — boundary = 11:55."""
        dt = datetime(2025, 1, 1, 12, 4, 59, 999999, tzinfo=UTC)
        result = self.buffer._last_complete_bar_boundary(dt)
        self.assertEqual(result, datetime(2025, 1, 1, 11, 55, tzinfo=UTC))

    def test_boundary_top_of_hour(self):
        """At 13:00:00 the 13:00 bar is forming — boundary = 12:55."""
        dt = datetime(2025, 1, 1, 13, 0, 0, tzinfo=UTC)
        result = self.buffer._last_complete_bar_boundary(dt)
        self.assertEqual(result, datetime(2025, 1, 1, 12, 55, tzinfo=UTC))

    def test_boundary_m1_timeframe(self):
        buf = PriceBuffer("X", 1, 100, timeframe_minutes=1)
        dt = datetime(2025, 1, 1, 1, 10, 30, tzinfo=UTC)
        self.assertEqual(
            buf._last_complete_bar_boundary(dt),
            datetime(2025, 1, 1, 1, 9, tzinfo=UTC),
        )

    def test_boundary_m15_timeframe(self):
        buf = PriceBuffer("X", 15, 100, timeframe_minutes=15)
        dt = datetime(2025, 1, 1, 1, 22, 0, tzinfo=UTC)
        self.assertEqual(
            buf._last_complete_bar_boundary(dt),
            datetime(2025, 1, 1, 1, 0, tzinfo=UTC),
        )

    def test_boundary_h1_timeframe(self):
        buf = PriceBuffer("X", 60, 100, timeframe_minutes=60)
        dt = datetime(2025, 1, 1, 3, 45, 0, tzinfo=UTC)
        self.assertEqual(
            buf._last_complete_bar_boundary(dt),
            datetime(2025, 1, 1, 2, 0, tzinfo=UTC),
        )

    # ------------------------------------------------------------------
    # reset() and get_data()
    # ------------------------------------------------------------------

    def test_reset_clears_data_and_boundary(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        self.buffer.update(data, next_bar_open(data))
        self.buffer.reset()
        self.assertTrue(self.buffer.is_empty())
        self.assertIsNone(self.buffer.last_bar_time)
        self.assertIsNone(self.buffer._stable_boundary)

    def test_reset_triggers_full_reload_on_next_get_fetch_params(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        self.buffer.update(data, next_bar_open(data))
        self.buffer.reset()
        params = self.buffer.get_fetch_params(datetime(2025, 1, 2, tzinfo=UTC))
        self.assertIsNone(params["start_dt"])

    def test_get_data_returns_independent_copy(self):
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        self.buffer.update(data, next_bar_open(data))
        copy1 = self.buffer.get_data()
        copy1.loc[copy1.index[0], "close"] = 999.0
        copy2 = self.buffer.get_data()
        self.assertNotEqual(copy2.loc[copy2.index[0], "close"], 999.0)

    # ------------------------------------------------------------------
    # exclude_columns
    # ------------------------------------------------------------------

    def test_exclude_columns_dropped_on_arrival(self):
        buf = PriceBuffer("X", TF, 100, exclude_columns=["open", "spread"], timeframe_minutes=TF)
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        buf.update(data, next_bar_open(data))
        cols = buf.get_data().columns
        self.assertNotIn("open", cols)
        self.assertNotIn("spread", cols)
        self.assertIn("close", cols)

    def test_exclude_nonexistent_columns_is_harmless(self):
        buf = PriceBuffer("X", TF, 100, exclude_columns=["does_not_exist"], timeframe_minutes=TF)
        data = make_bars(datetime(2025, 1, 1, tzinfo=UTC), 50)
        result = buf.update(data, next_bar_open(data))
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()