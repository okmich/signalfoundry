"""
Tests for MT5 filter implementations (DayTimeFilter, SpreadFilter, MaxPositionsFilter).
"""

import pytest
from datetime import datetime, time
from okmich_quant_mt5.filters import DayTimeFilter, SpreadFilter, MaxPositionsFilter


class TestDayTimeFilter:
    """Test DayTimeFilter for date and time-based filtering."""

    def test_no_restrictions_always_passes(self):
        """Test that filter with no restrictions always passes."""
        filter = DayTimeFilter(allowed_days=None, allowed_time_ranges=None)
        context = {"datetime": datetime(2025, 2, 12, 14, 30)}  # Wednesday 14:30
        assert filter(context) is True

    def test_allowed_days_weekday(self):
        """Test filtering by allowed weekdays."""
        # Monday-Friday only (0-4)
        filter = DayTimeFilter(allowed_days=[0, 1, 2, 3, 4])

        # Wednesday (weekday 2) - should pass
        context_wed = {"datetime": datetime(2025, 2, 12, 14, 30)}
        assert filter(context_wed) is True

        # Saturday (weekday 5) - should fail
        context_sat = {"datetime": datetime(2025, 2, 15, 14, 30)}
        assert filter(context_sat) is False

        # Sunday (weekday 6) - should fail
        context_sun = {"datetime": datetime(2025, 2, 16, 14, 30)}
        assert filter(context_sun) is False

    def test_allowed_days_weekend_only(self):
        """Test filtering for weekend only."""
        # Saturday-Sunday only (5-6)
        filter = DayTimeFilter(allowed_days=[5, 6])

        # Wednesday - should fail
        context_wed = {"datetime": datetime(2025, 2, 12, 14, 30)}
        assert filter(context_wed) is False

        # Saturday - should pass
        context_sat = {"datetime": datetime(2025, 2, 15, 14, 30)}
        assert filter(context_sat) is True

    def test_allowed_time_single_range(self):
        """Test filtering by single time range."""
        # 9:00 AM to 5:00 PM
        filter = DayTimeFilter(
            allowed_time_ranges=[(time(9, 0), time(17, 0))]
        )

        # 10:00 AM - should pass
        context_morning = {"datetime": datetime(2025, 2, 12, 10, 0)}
        assert filter(context_morning) is True

        # 5:00 PM - should pass (inclusive)
        context_close = {"datetime": datetime(2025, 2, 12, 17, 0)}
        assert filter(context_close) is True

        # 8:00 AM - should fail (before range)
        context_early = {"datetime": datetime(2025, 2, 12, 8, 0)}
        assert filter(context_early) is False

        # 6:00 PM - should fail (after range)
        context_late = {"datetime": datetime(2025, 2, 12, 18, 0)}
        assert filter(context_late) is False

    def test_allowed_time_multiple_ranges(self):
        """Test filtering by multiple time ranges."""
        # 9:00-12:00 and 14:00-17:00 (lunch break excluded)
        filter = DayTimeFilter(
            allowed_time_ranges=[
                (time(9, 0), time(12, 0)),
                (time(14, 0), time(17, 0)),
            ]
        )

        # 10:00 AM - should pass
        context_morning = {"datetime": datetime(2025, 2, 12, 10, 0)}
        assert filter(context_morning) is True

        # 3:00 PM - should pass
        context_afternoon = {"datetime": datetime(2025, 2, 12, 15, 0)}
        assert filter(context_afternoon) is True

        # 1:00 PM (lunch) - should fail
        context_lunch = {"datetime": datetime(2025, 2, 12, 13, 0)}
        assert filter(context_lunch) is False

    def test_combined_day_and_time_restrictions(self):
        """Test filtering by both days and time ranges."""
        # Monday-Friday, 9:00 AM - 5:00 PM
        filter = DayTimeFilter(
            allowed_days=[0, 1, 2, 3, 4],
            allowed_time_ranges=[(time(9, 0), time(17, 0))],
        )

        # Wednesday 10:00 AM - should pass
        context_pass = {"datetime": datetime(2025, 2, 12, 10, 0)}
        assert filter(context_pass) is True

        # Wednesday 8:00 AM - should fail (wrong time)
        context_fail_time = {"datetime": datetime(2025, 2, 12, 8, 0)}
        assert filter(context_fail_time) is False

        # Saturday 10:00 AM - should fail (wrong day)
        context_fail_day = {"datetime": datetime(2025, 2, 15, 10, 0)}
        assert filter(context_fail_day) is False

        # Sunday 8:00 PM - should fail (both wrong)
        context_fail_both = {"datetime": datetime(2025, 2, 16, 20, 0)}
        assert filter(context_fail_both) is False

    def test_missing_datetime_in_context(self):
        """Test handling of missing datetime in context."""
        filter = DayTimeFilter(allowed_days=[0, 1, 2, 3, 4])

        # Should use datetime.now() if not provided
        context_no_dt = {}
        result = filter(context_no_dt)
        # Result depends on current day, but shouldn't crash
        assert isinstance(result, bool)

    def test_invalid_datetime_in_context(self):
        """Test handling of invalid datetime in context."""
        filter = DayTimeFilter(allowed_days=[0, 1, 2, 3, 4])

        # Non-datetime object
        context_invalid = {"datetime": "not a datetime"}
        assert filter(context_invalid) is False

    def test_boundary_conditions(self):
        """Test exact boundary times."""
        filter = DayTimeFilter(
            allowed_time_ranges=[(time(9, 0), time(17, 0))]
        )

        # Exactly 9:00:00 - should pass
        context_start = {"datetime": datetime(2025, 2, 12, 9, 0, 0)}
        assert filter(context_start) is True

        # Exactly 17:00:00 - should pass
        context_end = {"datetime": datetime(2025, 2, 12, 17, 0, 0)}
        assert filter(context_end) is True

        # One second before 9:00 - should fail
        context_before = {"datetime": datetime(2025, 2, 12, 8, 59, 59)}
        assert filter(context_before) is False

        # One second after 17:00 - should fail
        context_after = {"datetime": datetime(2025, 2, 12, 17, 0, 1)}
        assert filter(context_after) is False


class TestSpreadFilter:
    """Test SpreadFilter for bid-ask spread filtering."""

    def test_spread_within_limit(self):
        """Test that acceptable spread passes."""
        filter = SpreadFilter(max_spread_points=50)
        context = {"spread": 30}
        assert filter(context) is True

    def test_spread_equals_limit(self):
        """Test that spread exactly at limit fails."""
        filter = SpreadFilter(max_spread_points=50)
        context = {"spread": 50}
        assert filter(context) is False

    def test_spread_exceeds_limit(self):
        """Test that excessive spread blocks."""
        filter = SpreadFilter(max_spread_points=50)
        context = {"spread": 100}
        assert filter(context) is False

    def test_zero_spread(self):
        """Test that zero spread passes."""
        filter = SpreadFilter(max_spread_points=50)
        context = {"spread": 0}
        assert filter(context) is True

    def test_missing_spread_in_context(self):
        """Test handling of missing spread value."""
        filter = SpreadFilter(max_spread_points=50)
        context = {}
        # Should fail if spread is missing
        assert filter(context) is False

    def test_none_spread_in_context(self):
        """Test handling of None spread value."""
        filter = SpreadFilter(max_spread_points=50)
        context = {"spread": None}
        # Should fail if spread is None
        assert filter(context) is False

    def test_very_tight_spread_limit(self):
        """Test with very tight spread limit."""
        filter = SpreadFilter(max_spread_points=5)

        context_pass = {"spread": 2}
        assert filter(context_pass) is True

        context_fail = {"spread": 10}
        assert filter(context_fail) is False

    def test_custom_name(self):
        """Test filter with custom name."""
        filter = SpreadFilter(max_spread_points=50, name="TightSpread")
        assert filter.name == "TightSpread"


class TestMaxPositionsFilter:
    """Test MaxPositionsFilter for position count limiting."""

    def test_under_limit(self):
        """Test that position count under limit passes."""
        filter = MaxPositionsFilter(max_positions=3)
        context = {"open_positions": 2}
        assert filter(context) is True

    def test_at_limit(self):
        """Test that position count at limit blocks."""
        filter = MaxPositionsFilter(max_positions=3)
        context = {"open_positions": 3}
        assert filter(context) is False

    def test_over_limit(self):
        """Test that position count over limit blocks."""
        filter = MaxPositionsFilter(max_positions=3)
        context = {"open_positions": 5}
        assert filter(context) is False

    def test_zero_positions(self):
        """Test with zero open positions."""
        filter = MaxPositionsFilter(max_positions=3)
        context = {"open_positions": 0}
        assert filter(context) is True

    def test_single_position_limit(self):
        """Test with limit of 1 position."""
        filter = MaxPositionsFilter(max_positions=1)

        context_zero = {"open_positions": 0}
        assert filter(context_zero) is True

        context_one = {"open_positions": 1}
        assert filter(context_one) is False

    def test_missing_positions_in_context(self):
        """Test handling of missing open_positions value."""
        filter = MaxPositionsFilter(max_positions=3)
        context = {}
        # Should default to 0 and pass
        assert filter(context) is True

    def test_custom_name(self):
        """Test filter with custom name."""
        filter = MaxPositionsFilter(max_positions=3, name="MaxPos3")
        assert filter.name == "MaxPos3"


class TestFilterIntegration:
    """Test multiple filters working together."""

    def test_all_filters_combined(self):
        """Test realistic scenario with all three filter types."""
        from okmich_quant_core.filter import FilterChain

        # Create a filter chain: weekday + trading hours + spread + max positions
        chain = FilterChain(
            [
                DayTimeFilter(
                    allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
                    allowed_time_ranges=[(time(9, 0), time(17, 0))],
                ),
                SpreadFilter(max_spread_points=50),
                MaxPositionsFilter(max_positions=3),
            ]
        )

        # Scenario 1: All conditions met - should pass
        context_pass = {
            "datetime": datetime(2025, 2, 12, 10, 0),  # Wednesday 10 AM
            "spread": 25,
            "open_positions": 1,
        }
        assert chain(context_pass) is True

        # Scenario 2: Weekend - should fail
        context_weekend = {
            "datetime": datetime(2025, 2, 15, 10, 0),  # Saturday 10 AM
            "spread": 25,
            "open_positions": 1,
        }
        assert chain(context_weekend) is False

        # Scenario 3: High spread - should fail
        context_spread = {
            "datetime": datetime(2025, 2, 12, 10, 0),  # Wednesday 10 AM
            "spread": 100,
            "open_positions": 1,
        }
        assert chain(context_spread) is False

        # Scenario 4: Too many positions - should fail
        context_positions = {
            "datetime": datetime(2025, 2, 12, 10, 0),  # Wednesday 10 AM
            "spread": 25,
            "open_positions": 3,
        }
        assert chain(context_positions) is False

        # Scenario 5: Before trading hours - should fail
        context_time = {
            "datetime": datetime(2025, 2, 12, 8, 0),  # Wednesday 8 AM
            "spread": 25,
            "open_positions": 1,
        }
        assert chain(context_time) is False