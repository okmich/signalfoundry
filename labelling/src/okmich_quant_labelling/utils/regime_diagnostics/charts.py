"""Charts for a ForwardSeparabilityResult: the Trade-Visualization Doctrine applied to a label test.
The distributions are the product; verdicts only summarise them.

- states_on_price           close with state shading for sample weeks: where the states actually occur
- quantile_profile_primary  shift of each state's quantiles vs all bars beyond the baseline, with bootstrap bands
- quantile_profile_raw      the same on the raw forward target
- distribution_curves       per-state quantile function of the raw forward target, axes x horizons
- fold_effects              spread of each state's per-fold out-of-sample effect (one point per quarter)
- robustness                per-fold effect through time, sign hits vs train
- distributions_by_fold     per-state forward target by fold (quantile boxes), middle horizon
- null                      rotation null of the studentized omnibus with the observed value

States use the first three validated categorical slots (they validate all-pairs); more states fold into grey.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
STATE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
OTHER = "#8e8d88"
NULL_FILL = "#b9b8b2"


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=INK)), height=height, paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, font=dict(color=INK_SECONDARY, size=11), hovermode="closest",
                      legend=dict(orientation="h", y=1.06, x=0), margin=dict(l=60, r=20, t=90, b=40))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=INK_SECONDARY, zerolinewidth=1, linecolor=GRID)
    return fig


def _color(code: int) -> str:
    return STATE_COLORS[code] if code < len(STATE_COLORS) else OTHER


def _grid(result):
    primary = result.summary[result.summary["primary"]]
    axes = list(dict.fromkeys(primary["axis"]))
    targets = {(r.axis, r.horizon): r for r in primary.itertuples()}
    horizons = sorted({h for _, h in targets})
    return axes, horizons, targets


def robustness_chart(result) -> go.Figure:
    axes, horizons, targets = _grid(result)
    states = result.meta["states"]
    titles = [f"{a} h={h}" for a in axes for h in horizons]
    fig = make_subplots(rows=len(axes), cols=len(horizons), subplot_titles=titles, shared_xaxes=True,
                        vertical_spacing=0.08, horizontal_spacing=0.04)
    folds = result.per_fold
    for i, axis in enumerate(axes):
        for j, h in enumerate(horizons):
            row = targets.get((axis, h))
            if row is None:
                continue
            part = folds[folds["target"] == row.target]
            for code, state in enumerate(states):
                s = part[part["state"] == code]
                fig.add_trace(go.Scatter(x=s["fold_start"], y=s["effect"], mode="lines+markers", name=str(state),
                                         legendgroup=str(state), showlegend=(i == 0 and j == 0),
                                         line=dict(color=_color(code), width=2),
                                         marker=dict(size=8, color=_color(code), line=dict(color=SURFACE, width=2),
                                                     symbol=np.where(s["hit"], "circle", "x")),
                                         hovertemplate=f"state {state}<br>%{{x|%Y-%m}}<br>effect %{{y:.3g}}"
                                                       "<extra></extra>"),
                              row=i + 1, col=j + 1)
    return _layout(fig, "Out-of-sample one-vs-rest effect per fold (primary layer; ● = sign matches train, × = not)",
                   260 * len(axes) + 80)


def null_chart(result) -> go.Figure:
    axes, horizons, targets = _grid(result)
    titles = []
    for a in axes:
        for h in horizons:
            row = targets.get((a, h))
            titles.append(f"{a} h={h}  p_holm={row.p_holm:.3f}" if row is not None else "")
    fig = make_subplots(rows=len(axes), cols=len(horizons), subplot_titles=titles, vertical_spacing=0.1,
                        horizontal_spacing=0.04)
    for i, axis in enumerate(axes):
        for j, h in enumerate(horizons):
            row = targets.get((axis, h))
            if row is None:
                continue
            draws = result.null_draws[f"{row.target}|{row.layer}|gate"][:, 0]
            fig.add_trace(go.Histogram(x=draws, marker=dict(color=NULL_FILL, line=dict(color=SURFACE, width=1)),
                                       showlegend=False, hovertemplate="T %{x:.2f}<br>%{y} rotations<extra></extra>"),
                          row=i + 1, col=j + 1)
            fig.add_vline(x=row.statistic, line=dict(color=INK, width=2), row=i + 1, col=j + 1)
    return _layout(fig, "Rotation null of the studentized omnibus T = Σ t² (grey) vs observed (black line), primary "
                        "layer", 240 * len(axes) + 80)


def distributions_chart(result) -> go.Figure:
    axes, horizons, targets = _grid(result)
    h = horizons[len(horizons) // 2]
    states = result.meta["states"]
    values = result.values
    fig = make_subplots(rows=len(axes), cols=1, subplot_titles=[f"{a} h={h} (raw forward target)" for a in axes],
                        vertical_spacing=0.08)
    for i, axis in enumerate(axes):
        row = targets.get((axis, h))
        if row is None:
            continue
        column = f"{row.target}|raw"
        for code, state in enumerate(states):
            part = values.loc[values["state"] == code, ["fold", column]].dropna()
            if part.empty:  # a state with no measurable bar at this horizon has no box to draw
                continue
            q = part.groupby("fold")[column].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
            starts = result.per_fold.drop_duplicates("fold").set_index("fold")["fold_start"].reindex(q.index)
            fig.add_trace(go.Box(x=starts, q1=q[0.25], median=q[0.5], q3=q[0.75], lowerfence=q[0.05],
                                 upperfence=q[0.95], name=str(state), legendgroup=str(state), showlegend=(i == 0),
                                 marker=dict(color=_color(code)), line=dict(width=2), offsetgroup=str(code)),
                          row=i + 1, col=1)
    fig.update_layout(boxmode="group")
    return _layout(fig, "Forward target by state and fold (5/25/50/75/95%)", 300 * len(axes) + 80)


def states_on_price_chart(close: pd.Series, label: pd.Series, result, n_weeks: int = 3) -> go.Figure:
    oos = result.values.index
    starts = pd.DatetimeIndex([oos[int(k)] for k in np.linspace(0, len(oos) - 2016, n_weeks)])
    states = result.meta["states"]
    fig = make_subplots(rows=n_weeks, cols=1, subplot_titles=[f"week from {s:%Y-%m-%d}" for s in starts],
                        vertical_spacing=0.08)
    codes = pd.Series(pd.Categorical(label, categories=states).codes, index=label.index)
    for i, start in enumerate(starts):
        window = slice(start, start + pd.Timedelta(days=7))
        c, k = close.loc[window], codes.loc[window]
        run = (k != k.shift()).cumsum()
        for _, block in k.groupby(run):
            code = int(block.iloc[0])
            if code < 0:
                continue
            fig.add_vrect(x0=block.index[0], x1=block.index[-1], fillcolor=_color(code), opacity=0.18, line_width=0,
                          row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=c.index, y=c, mode="lines", line=dict(color=INK, width=1.5), showlegend=False,
                                 hovertemplate="%{x|%a %H:%M}<br>%{y:.5f}<extra></extra>"), row=i + 1, col=1)
    for code, state in enumerate(states):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"state {state}",
                                 marker=dict(size=10, color=_color(code), symbol="square")), row=1, col=1)
    return _layout(fig, "Close with state shading (sample out-of-sample weeks)", 260 * n_weeks + 80)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _layer_name(row, layer: str) -> str:
    return {"primary": row.layer, "raw": "L0_raw", "volstd": "L1v_vol_standardized"}[layer]


def quantile_profile_chart(result, layer: str = "primary") -> go.Figure:
    """Shift of each state's quantiles from the pooled quantiles, with the 90% day-block bootstrap band: does a
    state move the whole distribution, or only a tail?"""
    axes, horizons, targets = _grid(result)
    states = result.meta["states"]
    table = result.quantiles
    titles = [f"{a} h={h}" for a in axes for h in horizons]
    fig = make_subplots(rows=len(axes), cols=len(horizons), subplot_titles=titles, shared_xaxes=True,
                        vertical_spacing=0.08, horizontal_spacing=0.04)
    for i, axis in enumerate(axes):
        for j, h in enumerate(horizons):
            row = targets.get((axis, h))
            if row is None:
                continue
            part = table[(table["target"] == row.target) & (table["layer"] == _layer_name(row, layer))]
            for code, state in enumerate(states):
                s = part[part["state"].astype(str) == str(state)].sort_values("quantile")
                x = (s["quantile"] * 100).to_numpy()
                fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[s["shift_upper"], s["shift_lower"][::-1]],
                                         fill="toself", fillcolor=_rgba(_color(code), 0.15), line=dict(width=0),
                                         hoverinfo="skip", showlegend=False), row=i + 1, col=j + 1)
                fig.add_trace(go.Scatter(x=x, y=s["shift"], mode="lines+markers", name=f"state {state}",
                                         legendgroup=str(state), showlegend=(i == 0 and j == 0),
                                         line=dict(color=_color(code), width=2),
                                         marker=dict(size=8, color=_color(code), line=dict(color=SURFACE, width=2)),
                                         hovertemplate=f"state {state}<br>q%{{x:.0f}}<br>shift %{{y:.3g}}"
                                                       "<extra></extra>"),
                              row=i + 1, col=j + 1)
    what = {"primary": "beyond the baseline (primary layer)", "raw": "raw forward target",
            "volstd": "vol-standardized (direction / path only)"}[layer]
    return _layout(fig, f"Quantile shift of each state vs all bars, {what}; band = 90% day-block bootstrap; "
                        "x = quantile level (%)", 250 * len(axes) + 90)


def distribution_curves_chart(result, layer: str = "raw") -> go.Figure:
    """Per-state distribution of the forward target as a quantile function (value at each cumulative share)."""
    axes, horizons, targets = _grid(result)
    states = result.meta["states"]
    values = result.values
    levels = np.linspace(0.01, 0.99, 99)
    titles = [f"{a} h={h}" for a in axes for h in horizons]
    fig = make_subplots(rows=len(axes), cols=len(horizons), subplot_titles=titles, vertical_spacing=0.08,
                        horizontal_spacing=0.04)
    for i, axis in enumerate(axes):
        for j, h in enumerate(horizons):
            row = targets.get((axis, h))
            if row is None:
                continue
            column = f"{row.target}|{layer}"
            if column not in values.columns:
                continue
            for code, state in enumerate(states):
                v = values.loc[values["state"] == code, column].dropna().to_numpy()
                if v.size == 0:
                    continue
                fig.add_trace(go.Scatter(x=np.quantile(v, levels), y=levels, mode="lines", name=f"state {state}",
                                         legendgroup=str(state), showlegend=(i == 0 and j == 0),
                                         line=dict(color=_color(code), width=2),
                                         hovertemplate=f"state {state}<br>value %{{x:.3g}}<br>share %{{y:.2f}}"
                                                       "<extra></extra>"),
                              row=i + 1, col=j + 1)
    what = {"raw": "raw forward target", "primary": "forward target beyond the baseline",
            "volstd": "vol-standardized forward target (direction / path)"}[layer]
    return _layout(fig, f"Per-state distribution of the {what} (cumulative share, 1-99%)", 250 * len(axes) + 90)


def fold_effects_chart(result) -> go.Figure:
    """Spread of each state's per-fold out-of-sample effect on the primary layer: how the effect is distributed
    through time, not only how often its sign held."""
    axes, horizons, targets = _grid(result)
    states = result.meta["states"]
    folds = result.per_fold
    fig = make_subplots(rows=len(axes), cols=1, subplot_titles=[f"{a}" for a in axes], vertical_spacing=0.1)
    for i, axis in enumerate(axes):
        for code, state in enumerate(states):
            xs, ys = [], []
            for h in horizons:
                row = targets.get((axis, h))
                if row is None:
                    continue
                part = folds[(folds["target"] == row.target) & (folds["state"] == code) & folds["eligible"]]
                xs += [f"h={h}"] * len(part)
                ys += part["effect"].tolist()
            fig.add_trace(go.Box(x=xs, y=ys, name=f"state {state}", legendgroup=str(state), showlegend=(i == 0),
                                 marker=dict(color=_color(code), size=8), line=dict(width=2), boxpoints="all",
                                 jitter=0.3, pointpos=0, offsetgroup=str(code)), row=i + 1, col=1)
    fig.update_layout(boxmode="group")
    return _layout(fig, "Per-fold out-of-sample one-vs-rest effect, primary layer (each point = one quarter)",
                   280 * len(axes) + 80)


def build_charts(result, close: pd.Series, label: pd.Series) -> dict[str, go.Figure]:
    return {
        "states_on_price": states_on_price_chart(close, label, result),
        "quantile_profile_primary": quantile_profile_chart(result, "primary"),
        "quantile_profile_volstd": quantile_profile_chart(result, "volstd"),
        "distribution_curves_volstd": distribution_curves_chart(result, "volstd"),
        "quantile_profile_raw": quantile_profile_chart(result, "raw"),
        "distribution_curves": distribution_curves_chart(result, "raw"),
        "fold_effects": fold_effects_chart(result),
        "robustness": robustness_chart(result),
        "distributions_by_fold": distributions_chart(result),
        "null": null_chart(result),
    }


def save_charts(result, close: pd.Series, label: pd.Series, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in build_charts(result, close, label).items():
        fig.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn")
