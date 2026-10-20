"""forward_separability(): the walk-forward forward-separability test of a causal regime label (see README.md)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from .baselines import ClockBaseline, HarBaseline, TailThresholds, winsor_limits
from .clock import DEFAULT_EXCLUDE_WINDOWS, build_clock
from .folds import Fold, walk_forward_folds
from .null import DayLattice, draw_offsets, episode_starts, mean_dwell, p_value
from .stats import Moments, Studentized, block_bootstrap_means, block_day_weights, holm, quantiles_from_histogram
from .targets import Axis, TargetKey, forward_targets, har_regressors


class Layer(StrEnum):
    RAW = "L0_raw"
    CLOCK = "L1_beyond_clock"
    HAR = "L2_beyond_har"
    VOL_STANDARDIZED = "L1v_vol_standardized"  # direction/path divided by the HAR forecast of forward vol
    TAIL_UP_95 = "L2_tail_up_q95"  # vol axis: exceeds the train q95 of the HAR residual given the forecast level
    TAIL_UP_99 = "L2_tail_up_q99"
    TAIL_DOWN_05 = "L2_tail_down_q05"
    TAIL_DOWN_01 = "L2_tail_down_q01"


TAIL_LAYERS = {Layer.TAIL_UP_95: (0.95, True), Layer.TAIL_UP_99: (0.99, True), Layer.TAIL_DOWN_05: (0.05, False),
               Layer.TAIL_DOWN_01: (0.01, False)}
DESCRIPTIVE_ONLY_LAYERS = {Layer.VOL_STANDARDIZED}  # read as distributions, not tested


class View(StrEnum):
    GATE = "gate"
    ENTRIES = "entries"


class StateVerdict(StrEnum):
    INSUFFICIENT = "insufficient"
    FLAT = "flat"
    UNSTABLE = "unstable"
    SEPARABLE = "separable"
    ACTIONABLE = "actionable"


class LabelVerdict(StrEnum):
    KEEP = "KEEP"
    SEPARABLE_ONLY = "SEPARABLE-ONLY"
    FLAT = "FLAT"


PRIMARY_LAYER = {Axis.DIRECTION: Layer.CLOCK, Axis.PATH: Layer.CLOCK, Axis.VOLATILITY: Layer.HAR}


@dataclass(frozen=True)
class Thresholds:
    alpha: float = 0.05
    confidence: float = 0.90  # one-sided bootstrap bound for the action layer
    vol_up: float = 1.25
    vol_down: float = 0.80
    min_bars: int = 500
    min_episodes: int = 30
    min_share: float = 0.02  # state minimums scale with the out-of-sample size
    fold_min_bars: int = 50
    min_consistency: float = 2 / 3
    max_concentration: float = 0.5
    block_days: int = 5
    cluster_days: int = 5  # studentization clusters: consecutive out-of-sample trading days per cluster


@dataclass
class ForwardSeparabilityResult:
    summary: pd.DataFrame
    per_state: pd.DataFrame
    per_fold: pd.DataFrame
    quantiles: pd.DataFrame  # per state x axis x horizon x layer: quantiles and shifts vs pooled, with bands
    tails: pd.DataFrame  # vol axis: per state tail-exceedance rates beyond HAR, studentized rotation tests
    label_verdicts: dict[Axis, LabelVerdict]
    meta: dict
    null_draws: dict[str, np.ndarray] = field(repr=False)
    values: pd.DataFrame = field(repr=False)  # out-of-sample bars: state code, fold, raw and primary-layer targets
    diagnostic: bool = False

    def __repr__(self) -> str:
        stamp = "DIAGNOSTIC — not tradeable; " if self.diagnostic else ""
        verdicts = ", ".join(f"{axis}: {verdict}" for axis, verdict in self.label_verdicts.items())
        return f"ForwardSeparabilityResult({stamp}{verdicts})"


def forward_separability(close: pd.Series, label: pd.Series, *, first_test_start, axes=tuple(Axis),
                         horizons=(3, 6, 12, 18, 24), lookbacks=(12,), label_fit_end=None, diagnostic: bool = False,
                         cost=None, exclude_windows=DEFAULT_EXCLUDE_WINDOWS, step: str = "QS", train: str = "expanding",
                         min_train: str = "365D", har_windows=("1h", "4h", "1D"), n_null: int = 999,
                         n_boot: int = 1000, thresholds: Thresholds = Thresholds(), entries_view: bool = True,
                         action: bool = True, quantile_bands: bool = True, seed: int = 0) -> ForwardSeparabilityResult:
    """Does the forward law of each axis's target differ across the states of ``label``, beyond the clock (and
    trailing volatility), out of sample, stably through time, and by enough to act on? See README.md.

    ``close`` and ``label`` share one tz-aware true-clock index; ``label`` holds the state known at the close of each
    bar (NaN = unknown). ``label_fit_end`` (timestamp, or per-bar Series for a walk-forward-refit label) guards
    against a label fitted on out-of-sample data. ``cost`` is the round-trip cost per decision bar in log-return
    units (None, scalar or Series).
    """
    if not isinstance(close, pd.Series) or not isinstance(label, pd.Series):
        raise TypeError("close and label must be pandas Series.")
    if not label.index.equals(close.index):
        raise ValueError("label must share close's index exactly.")
    axes = tuple(Axis(a) for a in axes)
    rng = np.random.default_rng(seed)

    clock = build_clock(close.index, exclude_windows)
    targets = forward_targets(close, clock, axes, horizons, lookbacks)
    vol_targets = forward_targets(close, clock, (Axis.VOLATILITY,), horizons, [])  # also scales direction / path
    har = har_regressors(close, clock, har_windows)
    folds = walk_forward_folds(close.index, first_test_start, step, train, min_train)
    provenance_violation = _check_provenance(label_fit_end, close.index, folds, diagnostic)
    diagnostic = diagnostic or provenance_violation

    categorical = pd.Categorical(label)
    states = list(categorical.categories)
    codes = categorical.codes.astype(np.int64)
    n_states = len(states)
    if n_states < 2:
        raise ValueError("label needs at least two states.")

    oos = np.concatenate([np.arange(f.test_start, f.test_end) for f in folds])
    targets, axes, dropped_note = _drop_unmeasurable(targets, axes, oos, clock.spacing)
    fold_slices = _fold_slices(folds)
    fold_oos = np.concatenate([np.full(f.test_end - f.test_start, f.number) for f in folds])
    codes_oos = codes[oos]
    lattice = DayLattice.build(clock.trading_day[oos], clock.slot[oos], codes_oos)

    dwell = mean_dwell(codes_oos)
    reach = max(key.horizon for key in targets) + 2 * (dwell if np.isfinite(dwell) else 1)
    day_offsets = draw_offsets(lattice.n_days, max(1, int(np.ceil(reach / clock.bars_per_day))), n_null, rng)
    bar_offsets = draw_offsets(oos.size, max(int(np.ceil(reach)), clock.bars_per_day), n_null, rng)

    weights = block_day_weights(lattice.n_days, n_boot, thresholds.block_days, rng)
    prepared = {key: _prepare_layers(key, y, vol_targets[TargetKey(Axis.VOLATILITY, key.horizon)], har,
                                     clock.hour_of_week, codes, folds, n_states, oos.size)
                for key, y in targets.items()}
    cost_oos = _cost_array(cost, close.index)[oos]

    observed, null = _observed_and_null(prepared, codes_oos, lattice, day_offsets, bar_offsets, n_states,
                                        entries_view, thresholds.cluster_days)
    summary = _summary(prepared, observed, null, day_offsets.size, bar_offsets.size, thresholds.alpha)

    per_state_rows, per_fold_rows, label_verdicts = [], [], {axis: LabelVerdict.FLAT for axis in axes}
    episode_count = np.bincount(codes_oos[episode_starts(codes_oos)], minlength=n_states)
    for key, prep in prepared.items():
        primary = PRIMARY_LAYER[key.axis]
        stability, fold_rows = _stability(key, prep, codes_oos, fold_slices, folds, close.index, n_states,
                                          thresholds)
        per_fold_rows.extend(fold_rows)
        bounds = _action_bounds(key, prep, codes_oos, fold_oos, lattice, weights, thresholds) if action else None
        row = summary[(summary["target"] == str(key)) & (summary["layer"] == str(primary))
                      & (summary["view"] == str(View.GATE))].iloc[0]
        states_df = _state_rows(key, prep, observed, null, states, codes_oos, cost_oos, episode_count, stability,
                                bounds, row, thresholds)
        per_state_rows.append(states_df)
        label_verdicts[key.axis] = _merge_label_verdict(label_verdicts[key.axis], states_df["verdict"])

    family = int((summary["primary"]).sum())
    meta = {
        "states": states, "n_folds": len(folds), "n_oos_bars": int(oos.size), "n_oos_days": lattice.n_days,
        "mean_dwell": dwell, "B_day_rotations": int(day_offsets.size), "B_bar_rotations": int(bar_offsets.size),
        "family_size": family, "min_attainable_p": 1 / (1 + day_offsets.size),
        "required_B_for_holm": int(np.ceil(family / thresholds.alpha)) - 1,
        "provenance_violation": provenance_violation, "first_test_start": close.index[folds[0].test_start],
    }
    meta["warnings"] = [] if day_offsets.size >= meta["required_B_for_holm"] else [
        f"Only {day_offsets.size} day rotations; Holm over {family} tests needs about {meta['required_B_for_holm']}. "
        "The best attainable p cannot clear the family-wise threshold: raise n_null, or lengthen the out-of-sample "
        f"span ({lattice.n_days} trading days caps the rotations)."]
    if dropped_note:
        meta["warnings"].append(dropped_note)
    per_state = pd.concat(per_state_rows, ignore_index=True)
    values = pd.DataFrame({"state": codes_oos, "fold": fold_oos}, index=close.index[oos])
    for key, prep in prepared.items():
        values[f"{key}|raw"] = prep.layers[Layer.RAW]
        values[f"{key}|primary"] = prep.layers[PRIMARY_LAYER[key.axis]]
        if Layer.VOL_STANDARDIZED in prep.layers:
            values[f"{key}|volstd"] = prep.layers[Layer.VOL_STANDARDIZED]
    quantiles = _quantile_table(prepared, codes_oos, lattice, states, weights if quantile_bands else None,
                                thresholds)
    return ForwardSeparabilityResult(summary=summary, per_state=per_state, per_fold=pd.DataFrame(per_fold_rows),
                                     quantiles=quantiles, tails=_tail_rows(prepared, observed, null, summary, states),
                                     label_verdicts=label_verdicts, meta=meta, null_draws=null,
                                     values=values, diagnostic=diagnostic)


# ---------------------------------------------------------------------------------------------------------------
# Preparation: per-fold baselines, winsorised statistic vectors and train references
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class _Prepared:
    layers: dict[Layer, np.ndarray]  # out-of-sample layer values (unclipped)
    stat: dict[Layer, np.ndarray]  # out-of-sample values clipped at each fold's train winsor limits
    ref_raw: np.ndarray  # (folds, K) sign of each state's train one-vs-rest effect on the raw target
    ref_primary: np.ndarray  # (folds, K) same on the primary layer


def _prepare_layers(key: TargetKey, y: np.ndarray, vol_y: np.ndarray, har: np.ndarray, buckets: np.ndarray,
                    codes: np.ndarray, folds: list[Fold], n_states: int, n_oos: int) -> _Prepared:
    """Per fold, on train only: the clock baseline, the HAR forecast of forward vol at this horizon, the derived
    layers (beyond HAR and tail exceedances for vol; vol-standardized for direction/path), winsor limits and the
    reference signs; applied to the fold's test bars."""
    vol_axis = key.axis == Axis.VOLATILITY
    layer_list = [Layer.RAW, Layer.CLOCK] + ([Layer.HAR, *TAIL_LAYERS] if vol_axis else [Layer.VOL_STANDARDIZED])
    layers = {layer: np.full(n_oos, np.nan) for layer in layer_list}
    stat = {layer: np.full(n_oos, np.nan) for layer in layer_list if layer not in DESCRIPTIVE_ONLY_LAYERS}
    ref_raw = np.zeros((len(folds), n_states))
    ref_primary = np.zeros((len(folds), n_states))
    primary = PRIMARY_LAYER[key.axis]
    levels = [level for level, _ in TAIL_LAYERS.values()]

    offset = 0
    for fold in folds:
        train = slice(fold.train_start, fold.train_stop(key.horizon))
        test = slice(fold.test_start, fold.test_end)
        out = slice(offset, offset + fold.test_end - fold.test_start)
        offset = out.stop

        clock_fit = ClockBaseline.fit(y[train], buckets[train])
        values = {Layer.RAW: (y[train], y[test]),
                  Layer.CLOCK: (clock_fit.residual(y[train], buckets[train]),
                                clock_fit.residual(y[test], buckets[test]))}
        har_fit = HarBaseline.fit(vol_y[train], har[train], buckets[train])
        forecast = (har_fit.predict(har[train], buckets[train]), har_fit.predict(har[test], buckets[test]))
        if vol_axis:
            residual = (y[train] - forecast[0], y[test] - forecast[1])
            values[Layer.HAR] = residual
            tails = TailThresholds.fit(residual[0], forecast[0], levels)
            for i, (layer, (_, upper)) in enumerate(TAIL_LAYERS.items()):
                values[layer] = (tails.exceed(residual[0], forecast[0], i, upper),
                                 tails.exceed(residual[1], forecast[1], i, upper))
        else:
            values[Layer.VOL_STANDARDIZED] = (y[train] / np.exp(forecast[0]), y[test] / np.exp(forecast[1]))

        for layer, (train_values, test_values) in values.items():
            layers[layer][out] = test_values
            if layer in stat:
                lo, hi = winsor_limits(train_values)
                stat[layer][out] = np.clip(test_values, lo, hi)
        ref_raw[fold.number] = _signs(Moments.of(codes[train], values[Layer.RAW][0], n_states).one_vs_rest())
        ref_primary[fold.number] = _signs(Moments.of(codes[train], values[primary][0], n_states).one_vs_rest())
    return _Prepared(layers, stat, ref_raw, ref_primary)


def _tail_rows(prepared, observed, null, summary: pd.DataFrame, states) -> pd.DataFrame:
    """Vol axis: per state, the out-of-sample tail-exceedance rate beyond HAR vs the rest, the studentized one-vs-rest
    t, its rotation p (Holm across states), and the omnibus p with its Holm adjustment over the tail family."""
    rows = []
    for key in prepared:
        if key.axis != Axis.VOLATILITY:
            continue
        for layer, (level, upper) in TAIL_LAYERS.items():
            name = _null_key(key, layer, View.GATE)
            moments, studentized = observed[name]
            line = summary[(summary["target"] == str(key)) & (summary["layer"] == str(layer))
                           & (summary["view"] == "gate")].iloc[0]
            null_t = np.abs(null[name][:, 1:])
            p_states = np.array([p_value(abs(studentized.t[s]), null_t[:, s]) for s in range(len(states))])
            p_states_holm = holm(p_states)
            for s, state in enumerate(states):
                rest_n = moments.n.sum() - moments.n[s]
                rows.append({"target": str(key), "horizon": key.horizon, "tail": str(layer),
                             "side": "upper" if upper else "lower", "nominal_rate": 1 - level if upper else level,
                             "state": state, "n": int(moments.n[s]), "rate": moments.means[s],
                             "rate_rest": (moments.s.sum() - moments.s[s]) / rest_n if rest_n else np.nan,
                             "t": studentized.t[s], "p_state": p_states[s], "p_state_holm": p_states_holm[s],
                             "p_omnibus": line["p"], "p_omnibus_holm": line["p_holm"]})
    return pd.DataFrame(rows)


def _signs(effects: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(effects), np.sign(effects), 0.0)


# ---------------------------------------------------------------------------------------------------------------
# Observed statistics and the rotation nulls
# ---------------------------------------------------------------------------------------------------------------

def _null_key(key: TargetKey, layer: Layer, view: View) -> str:
    return f"{key}|{layer}|{view}"


def _observed_and_null(prepared: dict[TargetKey, _Prepared], codes_oos: np.ndarray, lattice: DayLattice,
                       day_offsets: np.ndarray, bar_offsets: np.ndarray, n_states: int, entries_view: bool,
                       cluster_days: int):
    """Observed (moments, studentized) per test, and null tables of [omnibus, t of every state] per rotation."""
    cluster = lattice.day_pos // cluster_days
    n_clusters = int(cluster.max()) + 1

    def measure(codes: np.ndarray, values: np.ndarray) -> Studentized:
        return Studentized.of(codes, values, cluster, n_states, n_clusters)

    observed: dict[str, tuple[Moments, Studentized]] = {}
    day_jobs, bar_jobs = [], []
    entry_codes = np.where(episode_starts(codes_oos), codes_oos, -1)
    for key, prep in prepared.items():
        for layer, values in prep.stat.items():
            name = _null_key(key, layer, View.GATE)
            observed[name] = (Moments.of(codes_oos, values, n_states), measure(codes_oos, values))
            (bar_jobs if layer == Layer.RAW else day_jobs).append((name, values, False))
        primary = PRIMARY_LAYER[key.axis]
        if entries_view:
            name = _null_key(key, primary, View.ENTRIES)
            observed[name] = (Moments.of(entry_codes, prep.stat[primary], n_states),
                              measure(entry_codes, prep.stat[primary]))
            day_jobs.append((name, prep.stat[primary], True))

    null = {name: np.full((offsets.size, n_states + 1), np.nan)
            for jobs, offsets in ((day_jobs, day_offsets), (bar_jobs, bar_offsets)) for name, _, _ in jobs}
    for b, offset in enumerate(day_offsets):
        rotated = lattice.rotate(int(offset))
        rotated_entries = np.where(episode_starts(rotated), rotated, -1)
        for name, values, entries in day_jobs:
            _store(null[name], b, measure(rotated_entries if entries else rotated, values))
    for b, offset in enumerate(bar_offsets):
        rotated = np.roll(codes_oos, int(offset))
        for name, values, _ in bar_jobs:
            _store(null[name], b, measure(rotated, values))
    return observed, null


def _store(table: np.ndarray, row: int, studentized: Studentized) -> None:
    """Null row = [studentized omnibus, t of every state]."""
    table[row, 0] = studentized.omnibus
    table[row, 1:] = studentized.t


def _summary(prepared, observed, null, n_day, n_bar, alpha) -> pd.DataFrame:
    rows = []
    for key, prep in prepared.items():
        primary = PRIMARY_LAYER[key.axis]
        for layer in prep.stat:
            views = [View.GATE] + ([View.ENTRIES] if _null_key(key, layer, View.ENTRIES) in observed else [])
            for view in views:
                name = _null_key(key, layer, view)
                moments, studentized = observed[name]
                rows.append({"target": str(key), "axis": str(key.axis), "horizon": key.horizon,
                             "lookback": key.lookback, "layer": str(layer), "view": str(view),
                             "primary": layer == primary and view == View.GATE, "eta2": moments.eta2(),
                             "statistic": studentized.omnibus, "B": n_bar if layer == Layer.RAW else n_day,
                             "p": p_value(studentized.omnibus, null[name][:, 0]), "n": int(moments.n.sum())})
    summary = pd.DataFrame(rows)
    summary["tail"] = summary["layer"].isin([str(layer) for layer in TAIL_LAYERS]) & (summary["view"] == "gate")
    summary["p_holm"] = np.nan
    for family in ("primary", "tail"):  # the tail tests are their own Holm family
        mask = summary[family]
        summary.loc[mask, "p_holm"] = holm(summary.loc[mask, "p"].to_numpy())
    summary["significant"] = summary["p_holm"] < alpha
    return summary


# ---------------------------------------------------------------------------------------------------------------
# Robustness through time, the action layer, and verdicts
# ---------------------------------------------------------------------------------------------------------------

def _fold_slices(folds: list[Fold]) -> list[slice]:
    slices, offset = [], 0
    for fold in folds:
        slices.append(slice(offset, offset + fold.test_end - fold.test_start))
        offset += fold.test_end - fold.test_start
    return slices


def _stability(key, prep, codes_oos, fold_slices, folds, index, n_states, th: Thresholds):
    primary = PRIMARY_LAYER[key.axis]
    values = prep.stat[primary]
    hits, eligible = np.zeros(n_states), np.zeros(n_states)
    between, eta2_by_fold, rows = [], [], []
    for fold, part in zip(folds, fold_slices):
        moments = Moments.of(codes_oos[part], values[part], n_states)
        effect = moments.one_vs_rest()
        ok = (moments.n >= max(th.fold_min_bars, th.min_share * moments.n.sum())) & np.isfinite(effect)
        ok &= prep.ref_primary[fold.number] != 0
        hit = ok & (np.sign(effect) == prep.ref_primary[fold.number])
        hits += hit
        eligible += ok
        between.append(moments.between_ss())
        eta2_by_fold.append(moments.eta2())
        for s in range(n_states):
            rows.append({"target": str(key), "fold": fold.number, "fold_start": index[fold.test_start], "state": s,
                         "n": int(moments.n[s]), "effect": effect[s], "train_sign": prep.ref_primary[fold.number, s],
                         "eligible": bool(ok[s]), "hit": bool(hit[s]), "fold_eta2": eta2_by_fold[-1]})
    between = np.asarray(between)
    eta2_by_fold = np.asarray(eta2_by_fold)
    finite = np.isfinite(eta2_by_fold)
    drift = np.nan
    if finite.sum() > 2:  # slope of the per-fold eta2 per year of quarterly folds: is the effect decaying?
        drift = float(np.polyfit(np.arange(len(folds))[finite] / 4, eta2_by_fold[finite], 1)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        stability = {"consistency": np.where(eligible > 0, hits / eligible, np.nan), "eligible_folds": eligible,
                     "concentration": float(between.max() / between.sum()) if between.sum() > 0 else np.nan,
                     "concentration_limit": max(th.max_concentration, 2 / len(folds)), "eta2_drift_per_year": drift}
    return stability, rows


def _action_bounds(key, prep, codes_oos, fold_oos, lattice: DayLattice, weights: np.ndarray, th: Thresholds):
    """Observed value and bootstrap bounds per state: sign-adjusted mean (direction/path) or forward-RV ratio (vol)."""
    raw = prep.layers[Layer.RAW]
    known = (codes_oos >= 0) & np.isfinite(raw)
    n_states = prep.ref_raw.shape[1]
    columns = n_states + 1  # states, then all bars
    if key.axis == Axis.VOLATILITY:
        values = raw
    else:
        values = np.where(known, prep.ref_raw[fold_oos, np.maximum(codes_oos, 0)] * raw, np.nan)
    cell = lattice.day_pos[known] * columns + codes_oos[known]
    size = lattice.n_days * columns
    day_sum = np.bincount(cell, weights=values[known], minlength=size).reshape(lattice.n_days, columns)
    day_count = np.bincount(cell, minlength=size).astype(float).reshape(lattice.n_days, columns)
    day_sum[:, -1] = day_sum[:, :-1].sum(axis=1)
    day_count[:, -1] = day_count[:, :-1].sum(axis=1)

    draws = block_bootstrap_means(day_sum, day_count, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = day_sum.sum(axis=0) / day_count.sum(axis=0)
    tail = 1 - th.confidence
    if key.axis == Axis.VOLATILITY:
        ratio_draws = np.exp(draws[:, :-1] - draws[:, [-1]])
        return {"value": np.exp(point[:-1] - point[-1]), "lower": np.nanquantile(ratio_draws, tail, axis=0),
                "upper": np.nanquantile(ratio_draws, 1 - tail, axis=0)}
    return {"value": point[:-1], "lower": np.nanquantile(draws[:, :-1], tail, axis=0),
            "upper": np.nanquantile(draws[:, :-1], 1 - tail, axis=0)}


QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
HISTOGRAM_BINS = 2000
BAND = (0.05, 0.95)  # 90% bootstrap band on each quantile shift


def _quantile_table(prepared, codes_oos, lattice: DayLattice, states, weights, th: Thresholds) -> pd.DataFrame:
    """Per state and pooled ("all"): quantiles of every target layer, the shift of each state quantile from the pooled
    one, and a day-block bootstrap band on that shift. Bands come from per-day histograms on a fine pooled grid,
    resampled by the bootstrap day weights in one matrix product."""
    rows = []
    for key, prep in prepared.items():
        for layer, values in prep.layers.items():
            rows.extend(_layer_quantiles(key, layer, values, codes_oos, lattice, states, weights))
    return pd.DataFrame(rows)


def _layer_quantiles(key, layer, values, codes_oos, lattice: DayLattice, states, weights) -> list[dict]:
    ok = (codes_oos >= 0) & np.isfinite(values)
    v, c, d = values[ok], codes_oos[ok], lattice.day_pos[ok]
    if v.size == 0:
        return []
    n_states = len(states)
    point_all = np.quantile(v, QUANTILE_LEVELS)
    point = [np.quantile(v[c == s], QUANTILE_LEVELS) if (c == s).any() else np.full(len(QUANTILE_LEVELS), np.nan)
             for s in range(n_states)]
    lower = upper = np.full((n_states, len(QUANTILE_LEVELS)), np.nan)
    if weights is not None:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, HISTOGRAM_BINS + 1)))
        n_bins = edges.size - 1
        bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)
        groups = n_states + 1  # the states, then all bars pooled
        cells = np.r_[(d * groups + c) * n_bins + bins, (d * groups + n_states) * n_bins + bins]
        counts = np.bincount(cells, minlength=lattice.n_days * groups * n_bins).reshape(lattice.n_days, -1)
        drawn = (weights @ counts).reshape(weights.shape[0], groups, n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            cumulative = np.cumsum(drawn, axis=-1) / drawn.sum(axis=-1, keepdims=True)
        q_draws = quantiles_from_histogram(cumulative, edges, QUANTILE_LEVELS)
        shifts = q_draws[:, :n_states, :] - q_draws[:, [n_states], :]
        lower, upper = np.nanquantile(shifts, BAND[0], axis=0), np.nanquantile(shifts, BAND[1], axis=0)

    base = {"target": str(key), "axis": str(key.axis), "horizon": key.horizon, "lookback": key.lookback,
            "layer": str(layer)}
    rows = [{**base, "state": "all", "n": int(v.size), "quantile": q, "value": point_all[i], "shift": np.nan,
             "shift_lower": np.nan, "shift_upper": np.nan} for i, q in enumerate(QUANTILE_LEVELS)]
    for s, state in enumerate(states):
        rows.extend({**base, "state": state, "n": int((c == s).sum()), "quantile": q, "value": point[s][i],
                     "shift": point[s][i] - point_all[i], "shift_lower": lower[s, i], "shift_upper": upper[s, i]}
                    for i, q in enumerate(QUANTILE_LEVELS))
    return rows


def _state_rows(key, prep, observed, null, states, codes_oos, cost_oos, episode_count, stability, bounds, summary_row,
                th: Thresholds) -> pd.DataFrame:
    primary = PRIMARY_LAYER[key.axis]
    name = _null_key(key, primary, View.GATE)
    moments, studentized = observed[name]
    effects = studentized.effect
    null_t = np.abs(null[name][:, 1:])
    p_states = np.array([p_value(abs(studentized.t[s]), null_t[:, s]) for s in range(len(states))])
    p_states_holm = holm(p_states)

    raw = prep.layers[Layer.RAW]
    n_valid = moments.n
    enough = (n_valid >= max(th.min_bars, th.min_share * n_valid.sum()))
    enough &= episode_count >= max(th.min_episodes, th.min_share * episode_count.sum())
    omnibus_pass = bool(summary_row["p_holm"] < th.alpha)

    rows = []
    for s, state in enumerate(states):
        in_state = (codes_oos == s) & np.isfinite(raw)
        values = raw[in_state]
        consistency = stability["consistency"][s]
        row = {"target": str(key), "axis": str(key.axis), "horizon": key.horizon, "lookback": key.lookback,
               "state": state, "n_bars": int(n_valid[s]), "n_episodes": int(episode_count[s]),
               "mean": values.mean() if values.size else np.nan,
               "median": np.median(values) if values.size else np.nan,
               "std": values.std() if values.size > 1 else np.nan,
               **{f"q{int(q * 100):02d}": (np.quantile(values, q) if values.size else np.nan)
                  for q in (0.05, 0.25, 0.75, 0.95)},
               "primary_effect": effects[s], "t_state": studentized.t[s], "p_state": p_states[s],
               "p_state_holm": p_states_holm[s],
               "consistency": consistency, "eligible_folds": int(stability["eligible_folds"][s]),
               "concentration": stability["concentration"], "concentration_limit": stability["concentration_limit"],
               "eta2_drift_per_year": stability["eta2_drift_per_year"]}
        if key.axis != Axis.VOLATILITY:
            row["mean_bps"] = row["mean"] * 1e4
            costs = cost_oos[in_state]
            row["cost_mean"] = float(costs[np.isfinite(costs)].mean()) if np.isfinite(costs).any() else np.nan
        action_pass = None
        if bounds is not None:
            row.update({"action_value": bounds["value"][s], "action_lower": bounds["lower"][s],
                        "action_upper": bounds["upper"][s]})
            if key.axis == Axis.VOLATILITY:
                action_pass = bool(bounds["lower"][s] >= th.vol_up or bounds["upper"][s] <= th.vol_down)
            elif np.isfinite(row["cost_mean"]):
                action_pass = bool(bounds["lower"][s] > row["cost_mean"])

        if not enough[s]:
            verdict = StateVerdict.INSUFFICIENT
        elif not (omnibus_pass and p_states_holm[s] < th.alpha):
            verdict = StateVerdict.FLAT
        elif not (consistency >= th.min_consistency and stability["concentration"] < stability["concentration_limit"]):
            verdict = StateVerdict.UNSTABLE
        elif action_pass:
            verdict = StateVerdict.ACTIONABLE
        else:
            verdict = StateVerdict.SEPARABLE
        row["action_pass"] = action_pass
        row["verdict"] = str(verdict)
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_label_verdict(current: LabelVerdict, verdicts: pd.Series) -> LabelVerdict:
    if current == LabelVerdict.KEEP or (verdicts == StateVerdict.ACTIONABLE).any():
        return LabelVerdict.KEEP
    if current == LabelVerdict.SEPARABLE_ONLY or (verdicts == StateVerdict.SEPARABLE).any():
        return LabelVerdict.SEPARABLE_ONLY
    return LabelVerdict.FLAT


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------

def _drop_unmeasurable(targets: dict[TargetKey, np.ndarray], axes: tuple, oos: np.ndarray,
                       spacing: pd.Timedelta) -> tuple[dict[TargetKey, np.ndarray], tuple, str | None]:
    """Drop, with a warning, every target that has no measurable out-of-sample bar, and every axis left without a
    target (so it gets no verdict rather than a FLAT it was never tested for). Returns the kept targets and axes and
    the warning text (None when nothing was dropped).

    Horizons and lookbacks count bars, and a span that always crosses an excluded clock window or a data gap is never
    measurable: on 1h bars, a 24-bar horizon (or a 12-bar lookback plus a 12-bar horizon) always crosses the 4-hour
    New York roll."""
    empty = [key for key, y in targets.items() if not np.isfinite(y[oos]).any()]
    if not empty:
        return targets, axes, None
    kept = {key: y for key, y in targets.items() if key not in empty}
    names = ", ".join(map(str, empty))
    if not kept:
        raise ValueError(f"No forward target is measurable out of sample at {spacing} bars ({names}). Horizons and "
                         "lookbacks count bars: pass shorter ones for this bar size.")
    note = (f"Dropped {len(empty)} forward target(s) with no measurable out-of-sample bar at {spacing} bars: {names}. "
            "Horizons and lookbacks count bars, and every such span crosses an excluded clock window or a data gap; "
            "pass shorter ones for this bar size.")
    warnings.warn(note, RuntimeWarning, stacklevel=3)
    return kept, tuple(axis for axis in axes if any(key.axis == axis for key in kept)), note


def _check_provenance(label_fit_end, index: pd.DatetimeIndex, folds: list[Fold], diagnostic: bool) -> bool:
    """True when the label was fitted on data at or after an out-of-sample fold start; raises unless diagnostic."""
    if label_fit_end is None:
        return False
    if isinstance(label_fit_end, pd.Series):
        fit_end = pd.to_datetime(label_fit_end.reindex(index))
        violation = any((fit_end.iloc[f.test_start:f.test_end] > index[f.test_start]).any() for f in folds)
    else:
        stamp = pd.Timestamp(label_fit_end)
        stamp = stamp.tz_localize(index.tz) if stamp.tz is None else stamp.tz_convert(index.tz)
        violation = stamp > index[folds[0].test_start]
    if violation and not diagnostic:
        raise ValueError("The label was fitted on out-of-sample data (label_fit_end after a test fold start). "
                         "Refit it walk-forward, move first_test_start, or pass diagnostic=True.")
    return bool(violation)


def _cost_array(cost, index: pd.DatetimeIndex) -> np.ndarray:
    if cost is None:
        return np.full(len(index), np.nan)
    if isinstance(cost, pd.Series):
        return cost.reindex(index).to_numpy(dtype=float)
    return np.full(len(index), float(cost))
