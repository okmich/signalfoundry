import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.backtesting.temporal_performance_analyzer import (
    TemporalPerformanceAnalyzer,
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


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Hourly OHLC so that trades land across multiple sessions."""
    rng = np.random.default_rng(7)
    n = 240
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = (rng.standard_normal(n).cumsum() + 100).clip(min=10)
    return pd.DataFrame(
        {
            "open": close * (1 + rng.standard_normal(n) * 0.001),
            "high": close * (1 + rng.uniform(0, 0.005, n)),
            "low": close * (1 - rng.uniform(0, 0.005, n)),
            "close": close,
        },
        index=dates,
    )


def _ma_signal(data: pd.DataFrame) -> pd.Series:
    """Signed position series ({-1, 0, +1}) from a fast/slow MA crossover."""
    close = data["close"]
    fast = close.rolling(5).mean()
    slow = close.rolling(20).mean()
    pos = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return pd.Series(pos, index=close.index, dtype=float)


class TestConstructionValidation:
    def test_raises_on_missing_required_columns(self, sample_trades_df: pd.DataFrame):
        bad_df = sample_trades_df.drop(columns=["PnL"])
        with pytest.raises(ValueError, match="missing required columns"):
            TemporalPerformanceAnalyzer(bad_df)

    def test_allows_missing_optional_columns(self, sample_trades_df: pd.DataFrame):
        optional_missing = sample_trades_df.drop(columns=["Direction"])
        ta = TemporalPerformanceAnalyzer(optional_missing)
        assert (ta._entry_df["direction"] == "Unknown").all()


class TestTimezoneAndSessionHandling:
    def test_hour_dow_use_source_tz_session_uses_utc(self):
        # 09:00 US/Eastern in January is 14:00 UTC.
        # hour/dow stay in source_tz so analytics align with the broker clock the
        # live system reads; only `session` is UTC-anchored since session
        # boundaries are defined against universal market hours.
        df = pd.DataFrame(
            {
                "Entry Index": ["2024-01-02 09:00:00"],
                "Exit Index": ["2024-01-02 10:00:00"],
                "PnL": [1.0],
                "Direction": ["Long"],
            }
        )
        ta = TemporalPerformanceAnalyzer(df, source_tz="US/Eastern")
        row = ta._entry_df.iloc[0]

        assert row["hour"] == 9
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

        ta = TemporalPerformanceAnalyzer(df, source_tz="US/Eastern")

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

        ta = TemporalPerformanceAnalyzer(df, source_tz="UTC")
        cnt = ta._count_heatmap(ta._entry_df)

        assert cnt.loc["Monday", 9] == 2

    def test_show_dashboard_handles_empty_trades(self, tmp_path):
        empty_df = pd.DataFrame(columns=["Entry Index", "Exit Index", "PnL"])
        ta = TemporalPerformanceAnalyzer(empty_df)

        out = tmp_path / "temporal_performance_dashboard_test.html"
        fig = ta.show_dashboard(output_html=str(out), height=800)
        assert fig is not None
        assert out.exists()


class TestDualModeConstructors:
    def test_from_signal_builds_from_values_and_fn(self, sample_price_data: pd.DataFrame, tmp_path):
        ta = TemporalPerformanceAnalyzer.from_signal(sample_price_data, _ma_signal, freq="1h")
        assert isinstance(ta, TemporalPerformanceAnalyzer)
        # required time dimensions were derived from the generated trades
        assert {"hour", "session", "direction"}.issubset(ta._entry_df.columns)
        # render a NON-empty dashboard so bar-colouring (and other per-trade paths) execute
        assert not ta._entry_df.empty, "signal should have produced trades to exercise the dashboard"
        out = tmp_path / "from_signal_dashboard.html"
        assert ta.show_dashboard(output_html=str(out), height=800) is not None
        assert out.exists()

    def test_from_signal_missing_close_column_raises(self, sample_price_data: pd.DataFrame):
        no_close = sample_price_data.drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            TemporalPerformanceAnalyzer.from_signal(no_close, _ma_signal, freq="1h")

    def test_from_portfolio_matches_records_readable(self, sample_price_data: pd.DataFrame):
        from okmich_quant_research.backtesting.signal_adapter import signal_to_portfolio

        pf = signal_to_portfolio(sample_price_data, _ma_signal, freq="1h")
        ta = TemporalPerformanceAnalyzer.from_portfolio(pf)
        assert len(ta.raw) == len(pf.trades.records_readable)


class TestBackwardCompatAlias:
    def test_old_import_path_still_works(self):
        from okmich_quant_research.backtesting.vectorbt_analytics import VbtTradeAnalytics

        assert VbtTradeAnalytics is TemporalPerformanceAnalyzer

    def test_package_level_alias(self):
        import okmich_quant_research.backtesting as bt

        assert bt.VbtTradeAnalytics is bt.TemporalPerformanceAnalyzer
