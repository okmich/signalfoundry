"""Conditioning on another regime dimension: the expectation table and the within-condition forward test.

- Expectation table: for every label state x condition cell (the condition may itself combine several dimensions,
  e.g. vol state x session), the out-of-sample forward target's mean and quantiles with day-block bootstrap bands.
  This is "what do we expect next, given where we are".
- Within-condition test: does the label still separate the forward law *inside* each condition stratum? Label effects
  are studentized within each stratum and summed; the null rotates the label by whole trading days while the
  condition stays fixed, so a label that only proxies the condition is not credited.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clock import build_clock
from .null import DayLattice, draw_offsets, episode_starts, mean_dwell, p_value
from .stats import Studentized, block_bootstrap_means, block_day_weights, holm, quantiles_from_histogram

EXPECTATION_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
HISTOGRAM_BINS = 1000
BAND = (0.05, 0.95)


@dataclass
class ConditionalResult:
    expectations: pd.DataFrame  # cell x target x layer: n, episodes, mean and quantiles with 90% bands
    within: pd.DataFrame  # per target: within-condition omnibus T, rotation p, Holm p
    within_states: pd.DataFrame  # per target x condition stratum x label state: effect, t, p


def combine_conditions(condition, index: pd.DatetimeIndex) -> pd.Series:
    """One categorical Series from a Series or a {name: Series} of regime dimensions (NaN where any is unknown)."""
    if condition is None:
        return pd.Series("all", index=index, name="condition")
    parts = condition if isinstance(condition, dict) else {condition.name or "condition": condition}
    frame = pd.DataFrame({name: series.reindex(index) for name, series in parts.items()})
    known = frame.notna().all(axis=1)
    cells = None
    for name in frame.columns:
        part = f"{name}=" + frame[name].astype(str)
        cells = part if cells is None else cells + " & " + part
    return cells.where(known).rename("condition")


def conditional_analysis(result, condition, *, n_null: int = 999, n_boot: int = 1000, block_days: int = 5,
                         cluster_days: int = 5, min_bars: int = 300, within_test: bool = True,
                         seed: int = 0) -> ConditionalResult:
    """Expectation table and (optionally) the within-condition test on a ForwardSeparabilityResult's out-of-sample
    values. With ``condition=None`` the table is per label state only."""
    rng = np.random.default_rng(seed)
    values = result.values
    states = result.meta["states"]
    label_codes = values["state"].to_numpy()
    cond = combine_conditions(condition, values.index)
    cond_cat = pd.Categorical(cond)
    cond_codes = cond_cat.codes.astype(np.int64)
    cond_names = list(cond_cat.categories)

    clock = build_clock(values.index, exclude_windows=())
    lattice = DayLattice.build(clock.trading_day, clock.slot, label_codes)
    weights = block_day_weights(lattice.n_days, n_boot, block_days, rng)
    targets = sorted({column.split("|")[0] for column in values.columns if "|" in column},
                     key=lambda t: (t.split("_")[0], int(t.split("_h")[1].split("_")[0])))

    cells = np.where((label_codes >= 0) & (cond_codes >= 0), cond_codes * len(states) + label_codes, -1)
    cell_names = [(c, s) for c in cond_names for s in states]
    expectations = []
    for target in targets:
        for layer in ("raw", "volstd", "primary"):
            column = f"{target}|{layer}"
            if column in values.columns:
                expectations.extend(_cell_rows(target, layer, values[column].to_numpy(), cells, cell_names,
                                               label_codes, lattice, weights, min_bars))

    within, within_states = pd.DataFrame(), pd.DataFrame()
    if within_test:
        within, within_states = _within_condition(values, targets, label_codes, cond_codes, cond_names, states,
                                                  lattice, cluster_days, n_null, min_bars, rng)
    return ConditionalResult(pd.DataFrame(expectations), within, within_states)


def _cell_rows(target, layer, y, cells, cell_names, label_codes, lattice: DayLattice, weights, min_bars) -> list[dict]:
    ok = (cells >= 0) & np.isfinite(y)
    if not ok.any():
        return []
    n_cells = len(cell_names)
    v, c, d = y[ok], cells[ok], lattice.day_pos[ok]
    starts = episode_starts(label_codes)

    size = lattice.n_days * n_cells
    day_sum = np.bincount(d * n_cells + c, weights=v, minlength=size).reshape(lattice.n_days, n_cells)
    day_n = np.bincount(d * n_cells + c, minlength=size).astype(float).reshape(lattice.n_days, n_cells)
    mean_draws = block_bootstrap_means(day_sum, day_n, weights)

    edges = np.unique(np.quantile(v, np.linspace(0, 1, HISTOGRAM_BINS + 1)))
    n_bins = edges.size - 1
    bins = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)
    counts = np.bincount((d * n_cells + c) * n_bins + bins, minlength=size * n_bins).reshape(lattice.n_days, -1)
    drawn = (weights @ counts).reshape(weights.shape[0], n_cells, n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_draws = quantiles_from_histogram(np.cumsum(drawn, axis=-1) / drawn.sum(axis=-1, keepdims=True), edges,
                                           EXPECTATION_LEVELS)

    rows = []
    for k, (cond_name, state) in enumerate(cell_names):
        in_cell = c == k
        n = int(in_cell.sum())
        row = {"target": target, "layer": layer, "condition": cond_name, "state": state, "n": n,
               "episodes": int((starts[ok] & in_cell).sum()), "enough": n >= min_bars}
        if n:
            cell_values = v[in_cell]
            row.update({"mean": cell_values.mean(), "mean_lower": np.nanquantile(mean_draws[:, k], BAND[0]),
                        "mean_upper": np.nanquantile(mean_draws[:, k], BAND[1])})
            for i, level in enumerate(EXPECTATION_LEVELS):
                tag = f"q{int(round(level * 100)):02d}"
                row[tag] = np.quantile(cell_values, level)
                row[f"{tag}_lower"] = np.nanquantile(q_draws[:, k, i], BAND[0])
                row[f"{tag}_upper"] = np.nanquantile(q_draws[:, k, i], BAND[1])
        rows.append(row)
    return rows


def _within_condition(values, targets, label_codes, cond_codes, cond_names, states, lattice: DayLattice,
                      cluster_days, n_null, min_bars, rng):
    n_states, n_cond = len(states), len(cond_names)
    cluster = lattice.day_pos // cluster_days
    n_clusters = int(cluster.max()) + 1
    horizon_max = max(int(t.split("_h")[1].split("_")[0]) for t in targets)
    dwell = mean_dwell(label_codes)
    reach = horizon_max + 2 * (dwell if np.isfinite(dwell) else 1)
    bars_per_day = int(np.ceil(len(label_codes) / lattice.n_days))
    offsets = draw_offsets(lattice.n_days, max(1, int(np.ceil(reach / bars_per_day))), n_null, rng)

    def stratified(codes: np.ndarray, y: np.ndarray) -> list[Studentized | None]:
        out = []
        for c in range(n_cond):
            in_stratum = (cond_codes == c) & (codes >= 0) & np.isfinite(y)
            enough = np.bincount(codes[in_stratum], minlength=n_states)
            out.append(Studentized.of(np.where(cond_codes == c, codes, -1), y, cluster, n_states, n_clusters)
                       if (enough >= min_bars).sum() >= 2 else None)
        return out

    rows, state_rows = [], []
    columns = [f"{t}|primary" for t in targets if f"{t}|primary" in values.columns]
    observed = {col: stratified(label_codes, values[col].to_numpy()) for col in columns}
    null = {col: np.full((offsets.size, 1 + n_cond * n_states), np.nan) for col in columns}
    for b, offset in enumerate(offsets):
        rotated = lattice.rotate(int(offset))
        for col in columns:
            parts = stratified(rotated, values[col].to_numpy())
            null[col][b] = _flatten(parts, n_states)

    for col in columns:
        flat = _flatten(observed[col], n_states)
        p_omnibus = p_value(flat[0], null[col][:, 0])
        rows.append({"target": col.split("|")[0], "strata_tested": sum(p is not None for p in observed[col]),
                     "statistic": flat[0], "B": int(offsets.size), "p": p_omnibus})
        p_cells = np.array([p_value(abs(flat[1 + i]), np.abs(null[col][:, 1 + i])) for i in range(n_cond * n_states)])
        p_cells_holm = holm(p_cells)
        for c, cond_name in enumerate(cond_names):
            part = observed[col][c]
            for s, state in enumerate(states):
                i = c * n_states + s
                state_rows.append({"target": col.split("|")[0], "condition": cond_name, "state": state,
                                   "effect": part.effect[s] if part is not None else np.nan, "t": flat[1 + i],
                                   "p": p_cells[i], "p_holm": p_cells_holm[i]})
    within = pd.DataFrame(rows)
    if not within.empty:
        within["p_holm"] = holm(within["p"].to_numpy())
    return within, pd.DataFrame(state_rows)


def _flatten(parts: list, n_states: int) -> np.ndarray:
    """[sum of t^2 over strata and states, then t per (stratum, state)]; strata that were not tested are NaN."""
    ts = np.concatenate([p.t if p is not None else np.full(n_states, np.nan) for p in parts])
    finite = np.isfinite(ts)
    return np.r_[(ts[finite] ** 2).sum() if finite.any() else np.nan, ts]
