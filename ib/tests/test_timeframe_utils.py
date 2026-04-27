"""Unit tests for timeframe_utils."""
import pytest

from okmich_quant_ib.timeframe_utils import (
    bar_size_to_minutes, required_duration,
)


class TestBarSizeToMinutes:
    def test_known_values(self):
        assert bar_size_to_minutes("1 min") == 1
        assert bar_size_to_minutes("5 mins") == 5
        assert bar_size_to_minutes("1 hour") == 60
        assert bar_size_to_minutes("1 day") == 1440

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported bar size"):
            bar_size_to_minutes("3 mins")


class TestRequiredDuration:
    def test_one_min_rth_within_max(self):
        # 200 bars at 1 min RTH (390 min/day): ceil(200/390 * 1.3) = 1 day
        assert required_duration("1 min", 200, use_rth=True) == "1 D"

    def test_one_min_rth_exceeds_max(self):
        # 500 bars * 1.3 = 1.66 days, but the math here ceil(500/390*1.3)=2 → exceeds MAX=1
        with pytest.raises(ValueError):
            required_duration("1 min", 500, use_rth=True)

    def test_one_min_no_rth_within_max(self):
        # 500 bars at 1 min 24h: ceil(500/1440 * 1.3) = 1 day
        assert required_duration("1 min", 500, use_rth=False) == "1 D"

    def test_five_mins_no_rth(self):
        # 500 bars at 5 mins 24h: 500/(1440/5) = 1.736; *1.3 = 2.26 → 3 days
        assert required_duration("5 mins", 500, use_rth=False) == "3 D"

    def test_one_day_one_year(self):
        # 252 trading days → ceil(252*365/252) = 365 calendar days → 1 Y
        assert required_duration("1 day", 252, use_rth=True) == "1 Y"

    def test_one_day_exceeds_max(self):
        # 253 trading days → 367 calendar days > MAX 365
        with pytest.raises(ValueError):
            required_duration("1 day", 253, use_rth=True)
