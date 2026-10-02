import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from okmich_quant_research.backtesting.excursion_lens import EntryFill, ExcursionConfig, ExcursionLens

WARMUP = 40  # flat bars so ATR(14) settles at exactly 1.0 before the scenario starts


def _bars(closes: list[float], opens: dict[int, float] | None = None, start: str = "2024-01-02 00:00") -> pd.DataFrame:
    """Flat warm-up at 100 then ``closes``; open = close unless overridden, and the bar spans open..close +/- 0.5."""
    close = np.r_[np.full(WARMUP, 100.0), np.asarray(closes, dtype=float)]
    open_ = close.copy()
    for position, value in (opens or {}).items():
        open_[position] = value
    high = np.maximum(close, open_) + 0.5
    low = np.minimum(close, open_) - 0.5
    index = pd.date_range(start, periods=len(close), freq="5min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=index)


def _trades(ohlc: pd.DataFrame, rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """(entry bar, exit bar, side) -> a trade frame filled at the bars' closes."""
    return pd.DataFrame({"entry_time": [ohlc.index[e] for e, _, _ in rows],
                         "exit_time": [ohlc.index[x] for _, x, _ in rows],
                         "side": [s for _, _, s in rows],
                         "entry_price": [ohlc["close"].iloc[e] for e, _, _ in rows],
                         "exit_price": [ohlc["close"].iloc[x] for _, x, _ in rows]})


class TestPerTradeExcursions:
    """Hand-built path: entry 100, runs to 103.5 high, exits at 102, the move keeps going to 108.5, then retraces 3."""

    @pytest.fixture
    def lens(self) -> ExcursionLens:
        ohlc = _bars([101, 103, 102, 105, 108, 106, 106, 106, 106, 106])
        e = WARMUP - 1  # entry at the close of the last flat bar (100)
        return ExcursionLens(_trades(ohlc, [(e, e + 3, 1)]), ohlc, n_null=5)

    def test_atr_is_one_and_taken_before_entry(self, lens: ExcursionLens):
        assert lens.trades.loc[0, "atr_entry"] == pytest.approx(1.0)

    def test_mfe_mae_giveback(self, lens: ExcursionLens):
        trade = lens.trades.iloc[0]
        assert trade["realized"] == pytest.approx(2.0)
        assert trade["mfe"] == pytest.approx(3.5)  # high of the 103 bar
        assert trade["mae"] == pytest.approx(0.0)  # every in-trade low stayed above 100
        assert trade["giveback"] == pytest.approx(1.5)

    def test_potential_follows_the_move_past_the_exit_until_a_k_atr_retrace(self, lens: ExcursionLens):
        trade = lens.trades.iloc[0]
        assert trade["potential"] == pytest.approx(8.5)  # high of the 108 bar; the 106 bar retraces 3 >= 2 ATR
        assert trade["extension"] == pytest.approx(5.0)
        assert trade["missed"] == pytest.approx(6.5)
        assert trade["missed"] == pytest.approx(trade["giveback"] + trade["extension"])

    def test_trailing_stop_benchmark(self, lens: ExcursionLens):
        # best 3.5 after the 103 bar -> trailing level 1.5; the 102 bar's low (101.5) touches it
        assert lens.trades.loc[0, "trailing"] == pytest.approx(1.5)

    def test_units(self, lens: ExcursionLens):
        trade = lens.trades.iloc[0]
        assert trade["potential_atr"] == pytest.approx(8.5)
        assert trade["realized_bp"] == pytest.approx(200.0)
        assert np.isnan(trade["realized_r"])  # no stop column -> no R units


class TestShortSide:
    def test_short_excursions_use_the_mirrored_path(self):
        ohlc = _bars([99, 97, 98, 98, 98])
        e = WARMUP - 1
        lens = ExcursionLens(_trades(ohlc, [(e, e + 3, -1)]), ohlc, n_null=5)
        trade = lens.trades.iloc[0]
        assert trade["realized"] == pytest.approx(2.0)
        assert trade["mfe"] == pytest.approx(3.5)  # low of the 97 bar
        assert trade["mae"] == pytest.approx(0.0)

    def test_spread_is_added_to_the_short_path(self):
        ohlc = _bars([99, 97, 98, 98, 98])
        e = WARMUP - 1
        lens = ExcursionLens(_trades(ohlc, [(e, e + 3, -1)]), ohlc, spread=pd.Series(0.2, index=ohlc.index), n_null=5)
        assert lens.trades.loc[0, "mfe"] == pytest.approx(3.3)


class TestStops:
    @pytest.fixture
    def ohlc(self) -> pd.DataFrame:
        # long from 100: the next bar opens at 100, dips to 97.5 (adverse 2.5), then the trade recovers to 101
        return _bars([98, 101, 101, 101], opens={WARMUP: 100.0})

    def test_stop_outcomes_are_resimulated_on_the_path(self, ohlc: pd.DataFrame):
        e = WARMUP - 1
        lens = ExcursionLens(_trades(ohlc, [(e, e + 2, 1)]), ohlc, n_null=5, stop_grid_atr=(1.0, 2.0, 3.0))
        outcome = lens._stop_out_atr[0]
        assert outcome[0] == pytest.approx(-1.0)  # stopped at -1 ATR
        assert outcome[1] == pytest.approx(-2.0)
        assert outcome[2] == pytest.approx(1.0)  # 3 ATR never touched: the original result stands

    def test_a_gap_through_the_stop_fills_at_the_open(self, ohlc: pd.DataFrame):
        gapped = _bars([98, 101, 101, 101], opens={WARMUP: 97.0})
        e = WARMUP - 1
        lens = ExcursionLens(_trades(gapped, [(e, e + 2, 1)]), gapped, n_null=5, stop_grid_atr=(1.0, 2.0))
        assert lens._stop_out_atr[0, 0] == pytest.approx(-3.0)
        assert lens._stop_out_atr[0, 1] == pytest.approx(-3.0)

    def test_stop_column_gives_r_units_and_sets_the_move_end_distance(self, ohlc: pd.DataFrame):
        e = WARMUP - 1
        trades = _trades(ohlc, [(e, e + 2, 1)]).assign(stop=97.0)
        lens = ExcursionLens(trades, ohlc, stop_col="stop", n_null=5)
        trade = lens.trades.iloc[0]
        assert trade["k_leg_atr"] == pytest.approx(3.0)
        assert trade["realized_r"] == pytest.approx(1.0 / 3.0)

    def test_move_end_column_overrides_the_stop_distance_but_keeps_r(self, ohlc: pd.DataFrame):
        e = WARMUP - 1
        trades = _trades(ohlc, [(e, e + 2, 1)]).assign(stop=97.0, move_end=6.0)
        lens = ExcursionLens(trades, ohlc, stop_col="stop", move_end_col="move_end", n_null=5)
        trade = lens.trades.iloc[0]
        assert trade["k_leg_atr"] == pytest.approx(6.0)
        assert trade["realized_r"] == pytest.approx(1.0 / 3.0)


def _random_walk(n: int, seed: int, drift_at: np.ndarray | None = None, drift_bars: int = 24,
                 drift: float = 0.35) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n)
    if drift_at is not None:
        for start in drift_at:
            steps[start + 1:start + 1 + drift_bars] += drift
    close = 100 + np.cumsum(steps) * 0.01
    spread_noise = np.abs(rng.standard_normal(n)) * 0.004
    open_ = np.r_[close[0], close[:-1]]
    index = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread_noise,
                         "low": np.minimum(open_, close) - spread_noise, "close": close}, index=index)


class TestPlantedEdge:
    """A planted 24-bar drift after each entry must show up as entry edge; exiting after 4 bars must read as EARLY."""

    @pytest.fixture(scope="class")
    def planted(self) -> ExcursionLens:
        # irregular spacing: a strictly periodic schedule realigns with itself under whole-day shifts
        starts = 300 + np.cumsum(np.random.default_rng(11).integers(120, 181, 130))
        starts = starts[starts < 19_000]
        ohlc = _random_walk(20_000, seed=1, drift_at=starts)
        trades = _trades(ohlc, [(s, s + 4, 1) for s in starts])
        return ExcursionLens(trades, ohlc, n_null=60, cap_at_day_end=False, horizons=(2, 4, 8, 16, 24))

    def test_entries_beat_random(self, planted: ExcursionLens):
        by_h, by_k = planted.entry_quality()
        assert (by_h["mfe_mae_ratio"] > by_h["random_hi"]).all()
        assert (by_k["p_up_first"] > by_k["random_hi"]).any()
        sig = planted.significance()
        assert sig.loc["entry_mfe_mae", "p_beat"] <= 0.025
        assert sig.loc["entry_first_passage", "p_beat"] <= 0.025

    def test_exits_read_as_early(self, planted: ExcursionLens):
        summary, drift = planted.exit_quality()
        assert summary["extension_atr"] > 1.0
        assert (drift["drift_atr"] > drift["random_hi"]).iloc[1:].all()
        assert planted.significance().loc["after_exit_drift", "p_beat"] <= 0.025
        assert any("EARLY" in line for line in planted.verdicts())


class TestNoEdge:
    def test_random_entries_stay_inside_the_band(self):
        ohlc = _random_walk(20_000, seed=2)
        rng = np.random.default_rng(3)
        starts = np.sort(rng.choice(np.arange(100, 19_900), 120, replace=False))
        trades = _trades(ohlc, [(s, s + 6, int(rng.choice([-1, 1]))) for s in starts])
        lens = ExcursionLens(trades, ohlc, n_null=100, cap_at_day_end=False)
        by_h, _ = lens.entry_quality()
        outside = (by_h["mfe_mae_ratio"] > by_h["random_hi"]) | (by_h["mfe_mae_ratio"] < by_h["random_lo"])
        assert outside.sum() <= 1


class TestIngestionAndDashboard:
    @pytest.fixture
    def ohlc(self) -> pd.DataFrame:
        return _random_walk(3_000, seed=4)

    def test_unknown_timestamps_raise(self, ohlc: pd.DataFrame):
        trades = _trades(ohlc, [(100, 110, 1)])
        trades.loc[0, "entry_time"] = pd.Timestamp("1999-01-01")
        with pytest.raises(ValueError, match="not bars of `ohlc`"):
            ExcursionLens(trades, ohlc)

    def test_open_trades_are_dropped(self, ohlc: pd.DataFrame):
        trades = _trades(ohlc, [(100, 110, 1), (200, 210, -1)]).assign(Status=["Closed", "Open"])
        with pytest.warns(UserWarning, match="open trade"):
            lens = ExcursionLens(trades, ohlc, n_null=5)
        assert len(lens.trades) == 1

    def test_vectorbt_names_and_open_fill(self, ohlc: pd.DataFrame):
        trades = pd.DataFrame({"Entry Timestamp": [ohlc.index[100]], "Exit Timestamp": [ohlc.index[110]],
                               "Avg Entry Price": [ohlc["open"].iloc[100]], "Avg Exit Price": [ohlc["open"].iloc[110]],
                               "Direction": ["Short"], "Status": ["Closed"]})
        lens = ExcursionLens(trades, ohlc, n_null=5, entry_fill=EntryFill.OPEN)
        assert lens.trades.loc[0, "side"] == -1

    def test_from_signal_and_dashboard(self, ohlc: pd.DataFrame, tmp_path):
        def signal(data: pd.DataFrame) -> pd.Series:
            return pd.Series(np.sign(data["close"].diff(12)).fillna(0.0), index=data.index)

        lens = ExcursionLens.from_signal(ohlc, signal, config=ExcursionConfig(n_null=10))
        assert len(lens.trades) > 10
        target = tmp_path / "excursions.html"
        fig = lens.show_dashboard(output_html=str(target))
        assert isinstance(fig, go.Figure)
        assert target.exists()

    def test_walk_forward_has_one_row_per_later_fold(self, ohlc: pd.DataFrame):
        rng = np.random.default_rng(5)
        starts = np.sort(rng.choice(np.arange(100, 2_900), 60, replace=False))
        lens = ExcursionLens(_trades(ohlc, [(s, s + 5, 1) for s in starts]), ohlc, n_null=5, n_stop_folds=4)
        assert len(lens.stop_analysis()["walk_forward"]) == 3


class TestCalibration:
    """A random walk has no edge. Back-to-back trades that keep repeating a side must not read as skill: the old
    independent-random-bar baseline put a quarter of such cells outside its 95% band."""

    def test_back_to_back_book_on_a_random_walk_reads_as_random(self):
        flags, tests = 0, 0
        for seed in range(6):
            ohlc = _random_walk(20_000, seed=50 + seed)
            rng = np.random.default_rng(seed)
            starts = np.arange(200, 19_800, 6)
            side = np.cumprod(np.where(rng.random(starts.size) < 0.1, -1, 1))
            trades = _trades(ohlc, [(s, s + 6, int(d)) for s, d in zip(starts, side)])
            sig = ExcursionLens(trades, ohlc, n_null=60, cap_at_day_end=False, horizons=(6, 24, 48)).significance()
            flags += int((sig[["p_beat", "p_worse"]].min(axis=1) <= 0.025).sum())
            tests += len(sig)
        assert flags <= 2, f"{flags}/{tests} family-wise verdicts fired on a random walk"


class TestSpreadInput:
    def test_scalar_array_and_series_spread_agree(self):
        ohlc = _bars([99, 97, 98, 98, 98])
        trades = _trades(ohlc, [(WARMUP - 1, WARMUP + 2, -1)])
        results = [ExcursionLens(trades, ohlc, spread=spread, n_null=5).trades.loc[0, "mfe"]
                   for spread in (pd.Series(0.2, index=ohlc.index), 0.2, np.full(len(ohlc), 0.2))]
        assert results == pytest.approx([3.3, 3.3, 3.3])

    def test_an_array_of_the_wrong_length_raises(self):
        ohlc = _bars([99, 97, 98])
        with pytest.raises(ValueError, match="one value per ohlc row"):
            ExcursionLens(_trades(ohlc, [(WARMUP - 1, WARMUP + 2, -1)]), ohlc, spread=np.zeros(3), n_null=5)

    def test_a_series_on_other_timestamps_raises(self):
        ohlc = _bars([99, 97, 98])
        stray = pd.Series(0.2, index=ohlc.index + pd.Timedelta(days=400))
        with pytest.raises(ValueError, match="no timestamps"):
            ExcursionLens(_trades(ohlc, [(WARMUP - 1, WARMUP + 2, -1)]), ohlc, spread=stray, n_null=5)

    def test_window_statistics_charge_the_spread_to_both_sides(self):
        ohlc = _random_walk(3_000, seed=8)
        trades = _trades(ohlc, [(100, 110, 1)])
        base = ExcursionLens(trades, ohlc, n_null=5)
        wide = ExcursionLens(trades, ohlc, spread=0.01, n_null=5)
        cost = 0.01 / base._atr_ref
        before, after = base._bar_excursions[0], wide._bar_excursions[0]
        for mfe, mae in ((0, 1), (2, 3)):  # long, then short
            room = before[mfe] > 2 * cost  # MFE well clear of its floor at zero
            np.testing.assert_allclose(after[mfe][room], (before[mfe] - cost)[room], rtol=1e-9)
            hurt = before[mae] > 0
            np.testing.assert_allclose(after[mae][hurt], (before[mae] + cost)[hurt], rtol=1e-9)


class TestBenchmarksAcrossDays:
    def test_entry_on_the_last_bar_of_a_day_gets_a_benchmark(self):
        ohlc = _random_walk(3_000, seed=9)
        e = int(np.flatnonzero((ohlc.index.hour == 23) & (ohlc.index.minute == 55))[0])
        trade = ExcursionLens(_trades(ohlc, [(e, e + 30, 1)]), ohlc, n_null=5).trades.iloc[0]
        assert trade["trailing_exit_time"] > ohlc.index[e]

    def test_benchmarks_cover_an_overnight_trade(self):
        ohlc = _random_walk(3_000, seed=7)
        e = int(np.flatnonzero((ohlc.index.hour == 22) & (ohlc.index.minute == 0))[0])
        close = ohlc["close"].to_numpy().copy()
        close[e:] = close[e] + np.r_[0.0, np.cumsum(np.full(len(close) - e - 1, 0.01))]
        close[e + 60:] = close[e + 60]  # a steady five-hour climb from 22:00, then flat
        open_ = np.r_[close[0], close[:-1]]
        trend = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.002,
                              "low": np.minimum(open_, close) - 0.002, "close": close}, index=ohlc.index)
        trade = ExcursionLens(_trades(trend, [(e, e + 30, 1)]), trend, n_null=5).trades.iloc[0]  # exits 00:30
        assert trade["trailing_exit_time"] >= trade["exit_time"]  # not forced out at 23:55
        assert trade["extension_atr"] > 1.0  # the climb after the exit is followed into the new day

    def test_random_exits_may_run_over_a_break_like_real_ones(self):
        ohlc = _random_walk(3_000, seed=10)
        stamps = ohlc.index.to_numpy().copy()
        stamps[1500:] = stamps[1500:] + np.timedelta64(2, "D")  # a weekend between bars 1499 and 1500
        ohlc.index = pd.DatetimeIndex(stamps)
        lens = ExcursionLens(_trades(ohlc, [(1450, 1550, 1), (1300, 1310, 1)]), ohlc, n_null=5)
        rng = np.random.default_rng(0)
        draws = np.concatenate([lens._random_exit_bars(rng) for _ in range(50)])
        assert (draws >= 1500).any()


class TestOpenFillDrift:
    def test_real_and_shifted_exits_measure_the_same_bars(self):
        n = 20_000
        rng = np.random.default_rng(12)
        close = 100 + np.cumsum(0.004 + rng.standard_normal(n) * 0.01)  # the same up-drift on every bar
        open_ = np.r_[close[0], close[:-1]]
        noise = np.abs(rng.standard_normal(n)) * 0.004
        ohlc = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + noise,
                             "low": np.minimum(open_, close) - noise, "close": close},
                            index=pd.date_range("2024-01-01", periods=n, freq="5min"))
        starts = np.sort(rng.choice(np.arange(200, n - 200), 150, replace=False))
        trades = pd.DataFrame({"entry_time": ohlc.index[starts], "exit_time": ohlc.index[starts + 8], "side": 1,
                               "entry_price": open_[starts], "exit_price": open_[starts + 8]})
        lens = ExcursionLens(trades, ohlc, entry_fill=EntryFill.OPEN, n_null=60, cap_at_day_end=False)
        assert lens.significance().loc["after_exit_drift", "p_beat"] > 0.025  # no "you exit EARLY" from fill timing


class TestStopsBeyondOwnStop:
    def test_levels_wider_than_a_hit_own_stop_are_not_simulated(self):
        ohlc = _bars([98, 101, 101, 101], opens={WARMUP: 100.0})  # dips to 97.5 on the first held bar
        e = WARMUP - 1
        trades = pd.DataFrame({"entry_time": [ohlc.index[e]], "exit_time": [ohlc.index[e + 1]], "side": [1],
                               "entry_price": [100.0], "exit_price": [98.5], "stop": [98.5]})  # stopped at 1.5 ATR
        lens = ExcursionLens(trades, ohlc, stop_col="stop", stop_grid_atr=(1.0, 2.0, 3.0), n_null=5)
        outcome = lens._stop_out_atr[0]
        assert outcome[0] == pytest.approx(-1.0)
        assert np.isnan(outcome[1]) and np.isnan(outcome[2])  # a wider stop would have kept it open: unknown
        stops = lens.stop_analysis()
        assert stops["curve"]["simulable_share"].tolist() == [1.0, 0.0, 0.0]
        assert stops["best_stop_atr"] == pytest.approx(1.0)
        assert stops["best_stop_gain_atr"] == pytest.approx(0.5)


class TestValidation:
    @pytest.mark.parametrize("overrides", [{"n_stop_folds": 1}, {"horizons": (0, 4)}, {"barriers_atr": (0.0,)},
                                           {"n_null": 0}, {"stop_grid_atr": ()}, {"entry_fill": "middle"},
                                           {"atr_period": 0}, {"break_gap_multiple": 1.0}])
    def test_bad_config_raises(self, overrides):
        with pytest.raises(ValueError):
            ExcursionConfig(**overrides)

    def test_entry_fill_accepts_a_string(self):
        assert ExcursionConfig(entry_fill="open").entry_fill is EntryFill.OPEN

    def test_no_usable_trades_raises(self):
        ohlc = _random_walk(500, seed=3)
        with pytest.raises(ValueError, match="no closed trades"):
            ExcursionLens(_trades(ohlc, [(2, 10, 1)]), ohlc, n_null=5)  # inside the ATR warm-up

    def test_trades_without_prices_are_dropped(self):
        ohlc = _random_walk(3_000, seed=4)
        trades = _trades(ohlc, [(100, 110, 1), (200, 210, 1)])
        trades.loc[1, "exit_price"] = np.nan
        with pytest.warns(UserWarning, match="missing"):
            lens = ExcursionLens(trades, ohlc, n_null=5)
        assert len(lens.trades) == 1

    def test_zero_atr_trades_are_dropped(self):
        flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
                            index=pd.date_range("2024-01-01", periods=80, freq="5min"))
        with pytest.raises(ValueError, match="no closed trades"):
            ExcursionLens(_trades(flat, [(40, 50, 1)]), flat, n_null=5)

    def test_trades_from_several_portfolio_columns_raise(self):
        ohlc = _random_walk(3_000, seed=4)
        trades = _trades(ohlc, [(100, 110, 1), (200, 210, 1)]).assign(Column=[0, 1])
        with pytest.raises(ValueError, match="portfolio columns"):
            ExcursionLens(trades, ohlc, n_null=5)

    def test_fewer_trades_than_folds_still_reports(self):
        ohlc = _random_walk(3_000, seed=4)
        lens = ExcursionLens(_trades(ohlc, [(100, 110, 1)]), ohlc, n_null=5)
        assert lens.stop_analysis()["walk_forward"].empty
        assert any(line.startswith("STOP") for line in lens.verdicts())
