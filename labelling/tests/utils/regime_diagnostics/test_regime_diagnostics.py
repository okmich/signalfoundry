"""Unit contracts for regime_diagnostics: the V5 alignment canary, the V6 provenance guard and the building blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.regime_diagnostics import (Axis, LabelVerdict, TargetKey, build_clock,
                                                            conditional_analysis, forward_separability,
                                                            forward_targets, regime_diagnostics, seasonal_cost)
from okmich_quant_labelling.utils.regime_diagnostics import synthetic as syn
from okmich_quant_labelling.utils.regime_diagnostics.folds import walk_forward_folds
from okmich_quant_labelling.utils.regime_diagnostics.null import DayLattice, episode_starts, mean_dwell
from okmich_quant_labelling.utils.regime_diagnostics.stats import Moments, Studentized, holm, quantiles_from_histogram


def _ny(*stamps):
    return pd.DatetimeIndex(pd.to_datetime(list(stamps))).tz_localize("America/New_York").tz_convert("UTC")


class TestClock:
    def test_requires_tz_aware_index(self):
        with pytest.raises(ValueError, match="tz-aware"):
            build_clock(pd.date_range("2024-01-02", periods=10, freq="5min"))

    def test_trading_day_rolls_at_17_new_york(self):
        clock = build_clock(_ny("2024-03-04 16:50", "2024-03-04 16:55", "2024-03-04 17:00", "2024-03-04 17:05"))
        assert clock.trading_day[1] != clock.trading_day[2]
        assert clock.trading_day[2] == clock.trading_day[3]
        assert clock.slot[2] == 0

    def test_hour_of_week_starts_sunday_17_new_york_in_both_dst_seasons(self):
        for sunday in ("2024-01-07 17:00", "2024-07-07 17:00"):
            clock = build_clock(_ny(sunday, pd.Timestamp(sunday) + pd.Timedelta("5min"),
                                    pd.Timestamp(sunday) + pd.Timedelta("10min")))
            assert clock.hour_of_week[0] == 0

    def test_default_exclusion_is_the_new_york_roll_window(self):
        clock = build_clock(_ny("2024-03-04 15:55", "2024-03-04 16:00", "2024-03-04 19:55", "2024-03-04 20:00"))
        assert clock.excluded.tolist() == [False, True, True, False]

    def test_gap_detection(self):
        index = syn.fx_index(2)
        clock = build_clock(index)
        assert clock.gap_before.sum() == 0
        clock = build_clock(index.delete(slice(100, 110)))
        assert clock.gap_before.sum() == 1


class TestTargets:
    def setup_method(self):
        self.index = syn.fx_index(3)
        rng = np.random.default_rng(0)
        self.close = pd.Series(np.exp(np.cumsum(rng.normal(0, 1e-3, len(self.index)))), index=self.index)
        self.clock = build_clock(self.index, exclude_windows=())

    def test_forward_window_is_t_plus_1_to_t_plus_h(self):
        h, i = 6, 100
        out = forward_targets(self.close, self.clock, tuple(Axis), [h], [12])
        log_close = np.log(self.close.to_numpy())
        r = np.diff(log_close)
        assert out[TargetKey(Axis.DIRECTION, h)][i] == pytest.approx(log_close[i + h] - log_close[i])
        assert out[TargetKey(Axis.VOLATILITY, h)][i] == pytest.approx(0.5 * np.log((r[i:i + h] ** 2).sum()))
        move = np.sign(log_close[i] - log_close[i - 12])
        assert out[TargetKey(Axis.PATH, h, 12)][i] == pytest.approx(move * (log_close[i + h] - log_close[i]))

    def test_windows_running_past_the_data_or_before_the_lookback_are_nan(self):
        out = forward_targets(self.close, self.clock, tuple(Axis), [6], [12])
        assert np.isnan(out[TargetKey(Axis.DIRECTION, 6)][-6:]).all()
        assert np.isnan(out[TargetKey(Axis.PATH, 6, 12)][:12]).all()

    def test_gap_and_excluded_window_mask_the_target(self):
        index = self.index.delete(slice(200, 210))
        close = self.close.drop(self.index[200:210])
        clock = build_clock(index)
        d = forward_targets(close, clock, (Axis.DIRECTION,), [6], [12])[TargetKey(Axis.DIRECTION, 6)]
        assert np.isnan(d[194:200]).all() and np.isfinite(d[190:194]).all()
        excluded = np.flatnonzero(clock.excluded)
        assert np.isnan(d[excluded]).all()  # a decision close inside the roll window is not an entry price


class TestFoldsAndStats:
    def test_quarterly_folds_and_purge(self):
        index = syn.fx_index(300)
        folds = walk_forward_folds(index, "2019-10-01", min_train="180D")
        assert index[folds[0].test_start] >= pd.Timestamp("2019-10-01", tz="UTC")
        assert all(f.test_end == g.test_start for f, g in zip(folds[:-1], folds[1:]))
        assert folds[0].train_stop(24) == folds[0].test_start - 24

    def test_a_stub_last_fold_merges_into_the_previous_one(self):
        index = syn.fx_index(300)  # cut at 2020-01-08 below: a one-week stub after the 2020-01-01 boundary
        folds = walk_forward_folds(index[:index.searchsorted(pd.Timestamp("2020-01-08", tz="UTC"))], "2019-07-01",
                                   min_train="180D")
        lengths = [f.test_end - f.test_start for f in folds]
        assert min(lengths) >= 0.25 * sorted(lengths)[len(lengths) // 2]
        assert folds[-1].test_end == index.searchsorted(pd.Timestamp("2020-01-08", tz="UTC"))

    def test_moments_match_direct_computation(self):
        rng = np.random.default_rng(1)
        codes = rng.integers(-1, 3, 500)
        y = rng.normal(size=500)
        y[::17] = np.nan
        m = Moments.of(codes, y, 3)
        ok = (codes >= 0) & np.isfinite(y)
        grand = y[ok].mean()
        ssb = sum(ok[codes == s].sum() * (y[ok & (codes == s)].mean() - grand) ** 2 for s in range(3))
        assert m.eta2() == pytest.approx(ssb / ((y[ok] - grand) ** 2).sum())
        assert m.one_vs_rest()[1] == pytest.approx(y[ok & (codes == 1)].mean() - y[ok & (codes != 1)].mean())

    def test_studentized_matches_a_day_clustered_direct_computation(self):
        rng = np.random.default_rng(4)
        n_days, per_day = 40, 30
        day = np.repeat(np.arange(n_days), per_day)
        codes = rng.integers(0, 2, day.size)
        y = rng.normal(size=day.size) + 0.3 * codes
        st = Studentized.of(codes, y, day, 2, n_days)
        in1 = codes == 1
        m1, m0 = y[in1].mean(), y[~in1].mean()
        influence = np.array([((y - m1) * in1)[day == d].sum() / in1.sum() - ((y - m0) * ~in1)[day == d].sum()
                              / (~in1).sum() for d in range(n_days)])
        assert st.effect[1] == pytest.approx(m1 - m0)
        assert st.t[1] == pytest.approx((m1 - m0) / np.sqrt((influence ** 2).sum()))
        assert st.t[0] == pytest.approx(-st.t[1])
        assert st.omnibus == pytest.approx(2 * st.t[1] ** 2)

    def test_histogram_quantiles_recover_exact_quantiles(self):
        rng = np.random.default_rng(6)
        v = rng.standard_t(3, 100_000)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 2001)))
        counts = np.bincount(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, edges.size - 2),
                             minlength=edges.size - 1)
        levels = (0.01, 0.25, 0.5, 0.9, 0.99)
        got = quantiles_from_histogram(np.cumsum(counts) / counts.sum(), edges, levels)
        assert got == pytest.approx(np.quantile(v, levels), abs=0.02)

    def test_holm(self):
        assert holm(np.array([0.01, 0.04, 0.03, np.nan])).tolist()[:3] == pytest.approx([0.03, 0.06, 0.06])

    def test_day_rotation_keeps_the_intraday_slot(self):
        index = syn.fx_index(10)
        clock = build_clock(index)
        codes = (clock.slot % 3).astype(np.int64)  # a label that is a pure function of the clock
        lattice = DayLattice.build(clock.trading_day, clock.slot, codes)
        assert np.array_equal(lattice.rotate(3), codes)

    def test_episode_helpers(self):
        codes = np.array([0, 0, 1, 1, 1, -1, 1, 0])
        assert episode_starts(codes).tolist() == [True, False, True, False, False, False, True, True]
        assert mean_dwell(codes) == pytest.approx(7 / 4)


class TestAlignmentCanary:
    """V5: a causal sign(r_t) label must show exactly the AR(1) forward effect; a look-ahead label a huge one."""

    def test_causal_and_look_ahead_labels(self):
        rng = np.random.default_rng(2)
        index = syn.fx_index(1500)
        phi, h = -0.2, 3
        close = syn.simulate_close(index, rng, states=np.zeros(len(index), dtype=int), ar=[phi], season=False,
                                   logvol_sd=0.0)
        clock = build_clock(index, exclude_windows=())
        d = forward_targets(close, clock, (Axis.DIRECTION,), [h], [])[TargetKey(Axis.DIRECTION, h)]
        r = np.r_[np.nan, np.diff(np.log(close.to_numpy()))]
        up = r > 0
        predicted = np.nanmean(r[up]) * phi * (1 - phi ** h) / (1 - phi)
        causal_effect = np.nanmean(d[up])
        assert causal_effect == pytest.approx(predicted, rel=0.2)

        ahead = np.r_[r[1:], np.nan] > 0
        look_ahead_effect = np.nanmean(d[ahead])
        assert look_ahead_effect > 3 * abs(causal_effect) and np.sign(look_ahead_effect) != np.sign(causal_effect)


def _small_case(vol_mult=(1.0, 1.0), seed=3):
    rng = np.random.default_rng(seed)
    index = syn.fx_index(320)
    states = syn.sticky_label(len(index), 2, 36, rng)
    close = syn.simulate_close(index, rng, states=states, vol_mult=list(vol_mult))
    return close, pd.Series(states.astype(float), index=index)


class TestForwardSeparability:
    def test_provenance_guard(self):
        close, label = _small_case()
        kwargs = dict(first_test_start="2019-10-01", min_train="180D", horizons=(6,), n_null=19, n_boot=50)
        with pytest.raises(ValueError, match="out-of-sample"):
            forward_separability(close, label, label_fit_end="2020-01-15", **kwargs)
        result = forward_separability(close, label, label_fit_end="2020-01-15", diagnostic=True, **kwargs)
        assert result.diagnostic and "DIAGNOSTIC" in repr(result)
        clean = forward_separability(close, label, label_fit_end="2019-06-30", **kwargs)
        assert not clean.diagnostic

    def test_planted_volatility_effect_is_found_and_direction_is_flat(self):
        close, label = _small_case(vol_mult=(1.0, 1.8))
        result = forward_separability(close, label, first_test_start="2019-10-01", min_train="180D", horizons=(3,),
                                      n_null=99, n_boot=200)
        assert result.label_verdicts[Axis.VOLATILITY] != LabelVerdict.FLAT
        assert result.label_verdicts[Axis.DIRECTION] == LabelVerdict.FLAT
        vol = result.per_state[result.per_state["axis"] == "volatility"].set_index("state")
        assert vol.loc[1.0, "action_value"] > vol.loc[0.0, "action_value"]

        q = result.quantiles
        raw = q[(q["target"] == "volatility_h3") & (q["layer"] == "L0_raw")]
        column = result.values["volatility_h3|raw"]
        known = result.values["state"] >= 0
        pooled_median = raw[(raw["state"] == "all") & (raw["quantile"] == 0.5)]["value"].iloc[0]
        assert pooled_median == pytest.approx(column[known].median())
        high = raw[(raw["state"].astype(str) == "1.0") & (raw["quantile"] == 0.5)].iloc[0]
        assert high["shift_lower"] > 0 and high["shift_lower"] <= high["shift"] <= high["shift_upper"]

    def test_per_state_mean_uses_the_same_alignment_as_the_targets(self):
        close, label = _small_case()
        result = forward_separability(close, label, first_test_start="2019-10-01", min_train="180D",
                                      axes=(Axis.DIRECTION,), horizons=(6,), n_null=9, n_boot=20)
        clock = build_clock(close.index)
        d = forward_targets(close, clock, (Axis.DIRECTION,), [6], [])[TargetKey(Axis.DIRECTION, 6)]
        oos = close.index >= close.index[close.index.searchsorted(pd.Timestamp("2019-10-01", tz="UTC"))]
        expected = np.nanmean(d[oos & (label.to_numpy() == 1.0)])
        got = result.per_state.set_index("state").loc[1.0, "mean"]
        assert got == pytest.approx(expected)


class TestUnmeasurableTargets:
    """Horizons count bars. On 1h bars the 5m-sized defaults include spans that always cross the 4-hour roll."""

    @staticmethod
    def _hourly_case(seed=3):
        rng = np.random.default_rng(seed)
        index = syn.fx_index(320, spacing="1h")
        states = syn.sticky_label(len(index), 2, 12, rng)
        close = syn.simulate_close(index, rng, states=states, vol_mult=[1.0, 1.5])
        return close, pd.Series(states.astype(float), index=index)

    def test_targets_with_no_measurable_bar_are_dropped_with_a_warning(self):
        close, label = self._hourly_case()
        with pytest.warns(RuntimeWarning, match="Dropped"):
            result = forward_separability(close, label, first_test_start="2019-10-01", min_train="180D", n_null=9,
                                          n_boot=20)
        tested = set(result.summary["target"])
        assert {"direction_h24", "volatility_h24", "path_h12_L12", "path_h24_L12"}.isdisjoint(tested)
        assert {"direction_h3", "volatility_h3", "path_h3_L12"} <= tested
        assert any(w.startswith("Dropped") for w in result.meta["warnings"])

    def test_the_factory_writes_its_report_on_hourly_bars(self, tmp_path):
        close, label = self._hourly_case()
        with pytest.warns(RuntimeWarning, match="Dropped"):
            diag = regime_diagnostics(close, label, first_test_start="2019-10-01", min_train="180D", n_null=9,
                                      n_boot=20, out_dir=tmp_path)
        assert diag.report_path.exists() and "Dropped" in diag.story

    def test_no_measurable_target_at_all_raises(self):
        close, label = self._hourly_case()
        with pytest.raises(ValueError, match="No forward target is measurable"):
            forward_separability(close, label, first_test_start="2019-10-01", min_train="180D", axes=(Axis.DIRECTION,),
                                 horizons=(24,), n_null=9, n_boot=20)


def test_within_condition_leaves_thin_cells_out_of_the_statistic():
    rng = np.random.default_rng(5)
    index = syn.fx_index(320)
    states = syn.sticky_label(len(index), 3, 36, rng)
    close = syn.simulate_close(index, rng, states=states, vol_mult=[1.0, 1.3, 1.8])
    label = pd.Series(states.astype(float), index=index)
    in_b = np.zeros(len(index), dtype=bool)
    in_b[states < 2] = rng.random(int((states < 2).sum())) < 0.5
    in_b[np.flatnonzero(states == 2)[-20:]] = True  # stratum B holds only 20 bars of state 2: a thin cell
    condition = pd.Series(np.where(in_b, "B", "A"), index=index, name="cond")

    result = forward_separability(close, label, first_test_start="2019-10-01", min_train="180D",
                                  axes=(Axis.VOLATILITY,), horizons=(3,), n_null=19, n_boot=20)
    within = conditional_analysis(result, condition, n_null=19)
    cells = within.within_states
    thin = cells[(cells["condition"] == "cond=B") & (cells["state"] == 2.0)].iloc[0]
    assert thin["n"] < 300 and not thin["tested"] and np.isnan(thin["t"])
    tested = cells[cells["tested"]]
    assert len(tested) == 5  # stratum A: three states; stratum B: states 0 and 1
    row = within.within.set_index("target").loc["volatility_h3"]
    assert row["cells_tested"] == 5 and row["statistic"] == pytest.approx((tested["t"] ** 2).sum())


def test_seasonal_cost_profile_uses_only_the_fit_window():
    index = syn.fx_index(20)
    spread = pd.Series(1e-4, index=index)
    spread.iloc[len(index) // 2:] = 9e-4  # a later regime the profile must not see
    cost = seasonal_cost(spread, fit_until=index[len(index) // 2])
    assert np.allclose(cost.to_numpy(), 1e-4)
    naive = index[len(index) // 2].tz_convert("UTC").tz_localize(None).isoformat()  # a plain string, read as UTC
    assert np.allclose(seasonal_cost(spread, fit_until=naive).to_numpy(), 1e-4)
