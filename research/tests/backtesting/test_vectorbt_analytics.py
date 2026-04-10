from pathlib import Path

import pandas as pd
import pytest

from okmich_quant_research.backtesting.vectorbt_analytics import (
    VbtTradeAnalytics,
    _get_session,
)


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Entry Index": [
                "2024-01-02 09:00:00",
                "2024-01-02 10:00:00",
            ],
            "Exit Index": [
                "2024-01-02 10:00:00",
                "2024-01-02 12:00:00",
            ],
            "PnL": [10.0, -5.0],
            "Direction": ["Long", "Short"],
        }
    )


class TestConstructionValidation:
    def test_raises_on_missing_required_columns(self, sample_trades_df: pd.DataFrame):
        bad_df = sample_trades_df.drop(columns=["PnL"])
        with pytest.raises(ValueError, match="missing required columns"):
            VbtTradeAnalytics(bad_df)

    def test_allows_missing_optional_columns(self, sample_trades_df: pd.DataFrame):
        optional_missing = sample_trades_df.drop(columns=["Direction"])
        ta = VbtTradeAnalytics(optional_missing)
        assert (ta._entry_df["direction"] == "Unknown").all()


class TestTimezoneAndSessionHandling:
    def test_uses_utc_for_time_dimensions(self):
        # 09:00 US/Eastern in January is 14:00 UTC
        df = pd.DataFrame(
            {
                "Entry Index": ["2024-01-02 09:00:00"],
                "Exit Index": ["2024-01-02 10:00:00"],
                "PnL": [1.0],
                "Direction": ["Long"],
            }
        )
        ta = VbtTradeAnalytics(df, source_tz="US/Eastern")
        row = ta._entry_df.iloc[0]

        assert row["hour"] == 14
        assert row["dow"] == "Tuesday"
        assert row["session"] == "NY–London OL"

    def test_handles_dst_nonexistent_times_as_unknown_session(self):
        # 2024-03-10 02:30 does not exist in US/Eastern due to DST spring-forward.
        df = pd.DataFrame(
            {
                "Entry Index": ["2024-03-10 01:30:00", "2024-03-10 02:30:00"],
                "Exit Index": ["2024-03-10 03:30:00", "2024-03-10 04:30:00"],
                "PnL": [1.0, -1.0],
                "Direction": ["Long", "Short"],
            }
        )

        ta = VbtTradeAnalytics(df, source_tz="US/Eastern")

        assert ta._entry_df["session"].iloc[1] == "Unknown"
        assert pd.isna(ta._entry_df["hour"].iloc[1])

    def test_get_session_maps_nan_to_unknown(self):
        assert _get_session(float("nan")) == "Unknown"


class TestAggregationsAndDashboard:
    def test_count_heatmap_counts_rows_even_with_nan_pnl(self):
        df = pd.DataFrame(
            {
                "Entry Index": ["2024-01-01 09:00:00", "2024-01-01 09:30:00"],
                "Exit Index": ["2024-01-01 10:00:00", "2024-01-01 10:30:00"],
                "PnL": [1.0, None],
                "Direction": ["Long", "Short"],
            }
        )

        ta = VbtTradeAnalytics(df, source_tz="UTC")
        cnt = ta._count_heatmap(ta._entry_df)

        assert cnt.loc["Monday", 9] == 2

    def test_show_dashboard_handles_empty_trades(self):
        empty_df = pd.DataFrame(columns=["Entry Index", "Exit Index", "PnL"])
        ta = VbtTradeAnalytics(empty_df)

        out = Path("projects/research/tests/backtesting/vectorbt_analytics_dashboard_test.html")
        try:
            fig = ta.show_dashboard(output_html=str(out), height=800)
            assert fig is not None
            assert out.exists()
        finally:
            if out.exists():
                out.unlink()
