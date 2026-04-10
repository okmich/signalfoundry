"""
Validation tests for TemporalFeature.market_session timezone handling (fix #7).

Covers:
  - Naive index without input_tz raises ValueError
  - Naive index with input_tz='UTC' converts correctly to market timezone
  - tz-aware index converts without needing input_tz
  - Session labels are correct after conversion (spot-check known UTC times)
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.temporal import TemporalFeature


def _make_df(index):
    return pd.DataFrame({"close": np.ones(len(index))}, index=index)


class TestMarketSessionTimezone:

    def test_naive_index_without_input_tz_raises(self):
        """Fix #7: naive index must raise ValueError, not silently mislabel sessions."""
        idx = pd.date_range("2024-01-02 14:30", periods=5, freq="1h")  # naive
        assert idx.tz is None
        tf = TemporalFeature(_make_df(idx))
        with pytest.raises(ValueError, match="input_tz"):
            tf.market_session(market="US")

    def test_naive_index_with_input_tz_utc_converts(self):
        """
        14:30 UTC = 09:30 New York (EST in Jan) — must label as 'open'.
        Without fix, tz_localize('America/New_York') on a UTC naive timestamp
        would interpret 14:30 as 14:30 local → 'after_hours'.
        """
        idx = pd.date_range("2024-01-02 14:30", periods=1, freq="1h")  # 14:30 UTC
        tf = TemporalFeature(_make_df(idx))
        result = tf.market_session(market="US", input_tz="UTC")
        # 14:30 UTC = 09:30 EST = market open
        assert result.iloc[0] == "open", (
            f"14:30 UTC should be 'open' in US market (09:30 EST), got '{result.iloc[0]}'"
        )

    def test_tz_aware_index_needs_no_input_tz(self):
        """tz-aware index should be converted without requiring input_tz."""
        idx = pd.date_range("2024-01-02 14:30", periods=3, freq="1h", tz="UTC")
        tf = TemporalFeature(_make_df(idx))
        result = tf.market_session(market="US")  # no input_tz needed
        assert isinstance(result, pd.Series)
        assert result.iloc[0] == "open"   # 14:30 UTC = 09:30 EST

    def test_tz_aware_already_in_market_tz(self):
        """Index already in America/New_York should label correctly."""
        idx = pd.date_range("2024-01-02 09:30", periods=1, freq="1h",
                            tz="America/New_York")
        tf = TemporalFeature(_make_df(idx))
        result = tf.market_session(market="US")
        assert result.iloc[0] == "open"

    def test_pre_market_label(self):
        """09:00 EST = pre_market for US."""
        idx = pd.date_range("2024-01-02 09:00", periods=1, freq="1h",
                            tz="America/New_York")
        tf = TemporalFeature(_make_df(idx))
        result = tf.market_session(market="US")
        assert result.iloc[0] == "pre_market"

    def test_after_hours_label(self):
        """16:30 EST = after_hours for US."""
        idx = pd.date_range("2024-01-02 16:30", periods=1, freq="1h",
                            tz="America/New_York")
        tf = TemporalFeature(_make_df(idx))
        result = tf.market_session(market="US")
        assert result.iloc[0] == "after_hours"

    def test_naive_input_tz_roundtrip_eu(self):
        """08:00 UTC = 08:00 Europe/London (no DST in Jan) = market open for EU."""
        idx = pd.date_range("2024-01-02 08:00", periods=1, freq="1h")
        tf = TemporalFeature(_make_df(idx))
        result = tf.market_session(market="EU", input_tz="UTC")
        assert result.iloc[0] == "open"
