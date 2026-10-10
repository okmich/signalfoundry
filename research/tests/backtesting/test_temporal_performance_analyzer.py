import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.backtesting.temporal_performance_analyzer import (
    DEFAULT_SESSIONS,
    SessionOpen,
    TemporalPerformanceAnalyzer,
    _get_session,
    assign_sessions,
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
        ta = TemporalPerformanceAnalyzer(optional_missing, source_tz="UTC")
        assert (ta._entry_df["direction"] == "Unknown").all()


class TestTimezoneAndSessionHandling:
    def test_hour_dow_use_source_tz_session_uses_market_clocks(self):
        # 09:00 US/Eastern in January is 14:00 UTC.
        # hour/dow stay in source_tz so analytics align with the broker clock the
        # live system reads; `session` opens on each market's own clock.
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
        with pytest.warns(DeprecationWarning):
            assert _get_session(float("nan")) == "Unknown"


def _trades_at(stamps, **kwargs) -> TemporalPerformanceAnalyzer:
    stamps = pd.DatetimeIndex(stamps)
    df = pd.DataFrame({"Entry Timestamp": stamps, "Exit Timestamp": stamps + pd.Timedelta(minutes=30),
                       "PnL": np.ones(len(stamps))})
    return TemporalPerformanceAnalyzer(df, **kwargs)


class TestDaylightSavingSessions:
    def test_winter_matches_the_old_utc_table_hour_by_hour(self):
        stamps = pd.date_range("2024-01-08 00:30", periods=24 * 5, freq="1h", tz="UTC")
        got = assign_sessions(stamps)
        with pytest.warns(DeprecationWarning):
            legacy = [_get_session(h) for h in stamps.hour]
        assert list(got) == legacy

    @pytest.mark.parametrize("utc_time, expected", [
        ("2024-07-10 07:30", "London"),          # London 08:30 BST; the fixed UTC table said Asian
        ("2024-07-10 12:30", "NY–London OL"),    # New York 08:30 EDT; the fixed table said London
        ("2024-07-10 15:30", "New York"),        # London 16:30 BST, closed; the fixed table said overlap
        ("2024-07-10 20:30", "Off-hours"),       # New York 16:30 EDT, closed; the fixed table said New York
        ("2024-07-10 23:30", "Off-hours"),       # Tokyo 08:30 JST, not open yet
        ("2024-07-11 00:30", "Asian"),           # Tokyo 09:30 JST
        ("2024-03-20 12:30", "NY–London OL"),    # US already on EDT, UK still on GMT: London 12:30, NY 08:30
        ("2024-03-20 16:30", "New York"),        # London closed at 16:00 GMT
    ])
    def test_boundaries_follow_each_market_clock(self, utc_time, expected):
        assert assign_sessions(pd.DatetimeIndex([utc_time], tz="UTC"))[0] == expected

    def test_same_instant_same_session_whatever_the_data_timezone(self):
        stamps = pd.date_range("2024-03-01", "2024-11-30", freq="37min", tz="UTC")
        base = assign_sessions(stamps)
        for tz in ("America/New_York", "Europe/Moscow", "Asia/Tokyo", "Etc/GMT-3"):
            assert (assign_sessions(stamps.tz_convert(tz)) == base).all()

    def test_naive_timestamps_require_a_timezone(self):
        with pytest.raises(ValueError, match="tz-aware"):
            assign_sessions(pd.DatetimeIndex(["2024-01-01 10:00"]))


class TestDataTimezone:
    def test_tz_aware_trades_use_their_own_clock(self):
        ta = _trades_at(pd.DatetimeIndex(["2024-07-10 08:30"], tz="America/New_York"))
        row = ta._entry_df.iloc[0]
        assert row["hour"] == 8
        assert row["session"] == "NY–London OL"
        assert str(ta.tz) == "America/New_York"

    def test_source_tz_labels_naive_trades(self):
        ta = _trades_at(["2024-07-10 08:30"], source_tz="America/New_York")
        assert ta._entry_df["hour"].iloc[0] == 8
        assert ta._entry_df["session"].iloc[0] == "NY–London OL"

    def test_naive_trades_without_source_tz_warn_and_assume_utc(self):
        with pytest.warns(UserWarning, match="assuming UTC"):
            ta = _trades_at(["2024-07-10 08:30"])
        assert ta._entry_df["session"].iloc[0] == "London"

    def test_conflicting_source_tz_raises(self):
        with pytest.raises(ValueError, match="disagrees"):
            _trades_at(pd.DatetimeIndex(["2024-07-10 08:30"], tz="America/New_York"), source_tz="UTC")

    def test_equivalent_source_tz_alias_is_accepted(self):
        ta = _trades_at(pd.DatetimeIndex(["2024-07-10 08:30"], tz="America/New_York"), source_tz="US/Eastern")
        assert ta._entry_df["hour"].iloc[0] == 8

    def test_custom_sessions_read_in_the_data_timezone(self):
        ny_clock = (SessionOpen("Asia", 17), SessionOpen("London", 3), SessionOpen("NY AM", 8),
                    SessionOpen("NY PM", 12))
        stamps = pd.DatetimeIndex(["2024-01-10 02:00", "2024-01-10 05:00", "2024-07-10 09:00", "2024-07-10 13:00",
                                   "2024-07-10 18:00"], tz="America/New_York")
        ta = _trades_at(stamps, sessions=ny_clock)
        assert list(ta._entry_df["session"]) == ["Asia", "London", "NY AM", "NY PM", "Asia"]
        assert ta._session_order == ["Asia", "London", "NY AM", "NY PM", "Unknown"]

    def test_duplicate_session_names_raise(self):
        with pytest.raises(ValueError, match="unique"):
            _trades_at(pd.DatetimeIndex(["2024-07-10 08:30"], tz="UTC"),
                       sessions=(SessionOpen("A", 1), SessionOpen("A", 5)))

    def test_from_signal_carries_the_index_timezone(self, sample_price_data: pd.DataFrame):
        data = sample_price_data.tz_localize("America/New_York")
        ta = TemporalPerformanceAnalyzer.from_signal(data, _ma_signal, freq="1h")
        assert str(ta.tz) == "America/New_York"
        expected = assign_sessions(pd.DatetimeIndex(ta._entry_df["Entry Timestamp"]))
        assert (ta._entry_df["session"].to_numpy() == expected).all()

    def test_default_sessions_partition_the_day(self):
        stamps = pd.date_range("2024-01-01", "2024-12-31", freq="15min", tz="UTC")
        got = pd.Series(assign_sessions(stamps))
        assert set(got) == {s.name for s in DEFAULT_SESSIONS}


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
        ta = TemporalPerformanceAnalyzer.from_signal(sample_price_data, _ma_signal, source_tz="UTC", freq="1h")
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
        ta = TemporalPerformanceAnalyzer.from_portfolio(pf, source_tz="UTC")
        assert len(ta.raw) == len(pf.trades.records_readable)
