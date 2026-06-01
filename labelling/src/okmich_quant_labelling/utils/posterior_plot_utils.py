"""Posterior-native plotting analogue of ``plot_price_multiple_labels_with_subplots``.

The hard-label plot paints discrete colored bands behind price. A posterior has no discrete band, so the analogue is a
**stacked-area gamma ribbon** under price: the K state probabilities stacked to 1 at every bar, optionally ordered by a
market axis so "low vol" (or "short momentum", etc.) is always the bottom band and is comparable across variants.
Price can additionally be shaded by confidence (``alpha ~ 1 - H(gamma_t)/log K``) so ambiguous bars wash out.

Styling follows the lab default: ``plotly_dark`` template, range slider on the bottom axis, per-subplot legends at the
top, light line colors over the dark background.
"""

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from okmich_quant_labelling.utils.posterior_eval_util import axis_ranks, discover_posterior_variants,\
    extract_posteriors, posterior_weighted_mean


def _state_order(df: pd.DataFrame, probs: np.ndarray, axis_col: Optional[str], ascending_axis: bool) -> List[int]:
    """State indices ordered for stacking — by axis rank if an axis is given, else raw."""
    K = probs.shape[1]
    if axis_col is None:
        return list(range(K))
    if axis_col not in df.columns:
        raise ValueError(f"axis_col '{axis_col}' not found in DataFrame.")
    axis_scores = posterior_weighted_mean(probs, df[axis_col].to_numpy(dtype=float))
    ranks = axis_ranks(axis_scores, ascending=ascending_axis)
    return list(np.argsort(ranks, kind="stable"))


def _confidence_alpha(probs: np.ndarray) -> np.ndarray:
    """Per-bar confidence in ``[0, 1]`` as ``1 - H(gamma)/log K`` (1 = certain)."""
    K = probs.shape[1]
    eps = 1e-12
    ent = -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)
    return 1.0 - ent / np.log(K) if K > 1 else np.ones(probs.shape[0])


def plot_price_multiple_posteriors_with_subplots(df: pd.DataFrame, variants: Optional[Sequence[str]] = None,
                                                 price_col: str = "close", axis_col: Optional[str] = None,
                                                 ascending_axis: bool = True,
                                                 rank_labels: Optional[Sequence[str]] = None,
                                                 color_price_by_confidence: bool = False,
                                                 title: Optional[str] = None) -> go.Figure:
    """Per-variant price + stacked posterior ribbon, stacked vertically across variants.

    For each variant two rows are drawn: the price series on top and the stacked ``gamma`` area below
    (bands ordered/labelled by ``axis_col`` when supplied). Set ``color_price_by_confidence`` to overlay
    confidence-shaded price markers (opacity ``= 1 - H(gamma)/log K``) so ambiguous bars fade out.

    Returns the :class:`plotly.graph_objects.Figure` (also calls ``.show()``).
    """
    discovered = discover_posterior_variants(df)
    selected = list(variants) if variants is not None else list(discovered)
    if not selected:
        raise ValueError("No posterior variants found (expected post_{variant}_s{k} columns).")
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not found in DataFrame.")
    missing = [v for v in selected if v not in discovered]
    if missing:
        raise ValueError(f"No post_*_s* columns for variant(s): {missing}.")

    n_rows = 2 * len(selected)
    subplot_titles = []
    for variant in selected:
        subplot_titles.extend([f"{variant} — price", f"{variant} — posterior gamma"])

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=subplot_titles,
                        row_heights=[1.4 if r % 2 == 0 else 1.0 for r in range(n_rows)])

    palette = pcolors.qualitative.Set2
    price = df[price_col]

    for v_idx, variant in enumerate(selected):
        probs = extract_posteriors(df, discovered[variant])
        price_row = 2 * v_idx + 1
        ribbon_row = 2 * v_idx + 2

        fig.add_trace(go.Scatter(x=df.index, y=price, mode="lines", line=dict(color="#ccc", width=1),
                                 showlegend=False, hoverinfo="skip"), row=price_row, col=1)

        if color_price_by_confidence:
            alpha = _confidence_alpha(probs)
            # per-point opacity = confidence, so ambiguous (high-entropy) bars literally fade out
            fig.add_trace(go.Scatter(x=df.index, y=price, mode="markers",
                                     marker=dict(size=4, color="#7FDBFF", opacity=alpha),
                                     customdata=alpha, name=f"{variant} confidence", showlegend=False,
                                     hovertemplate="conf=%{customdata:.2f}<extra></extra>"),
                          row=price_row, col=1)

        order = _state_order(df, probs, axis_col, ascending_axis)
        for stack_pos, k in enumerate(order):
            if axis_col is not None and rank_labels is not None and stack_pos < len(rank_labels):
                band_name = f"{variant}: {rank_labels[stack_pos]}"
            else:
                band_name = f"{variant}: s{k}"
            color = palette[stack_pos % len(palette)]
            fig.add_trace(go.Scatter(x=df.index, y=probs[:, k], mode="lines", line=dict(width=0.5, color=color),
                                     stackgroup=f"gamma_{v_idx}", name=band_name, legendgroup=f"variant_{v_idx}",
                                     legend=f"legend{v_idx + 1}",
                                     hovertemplate=f"<b>{band_name}</b><br>p=%{{y:.3f}}<extra></extra>"),
                          row=ribbon_row, col=1)

        fig.update_yaxes(title_text="price", row=price_row, col=1)
        fig.update_yaxes(title_text="P(state)", range=[0, 1], row=ribbon_row, col=1)

    layout = dict(title=title or f"Posterior ribbons: {', '.join(selected)}", template="plotly_dark",
                  height=230 * n_rows, autosize=True, showlegend=True, hovermode="x unified",
                  margin=dict(l=50, r=50, t=80, b=50))
    for v_idx in range(len(selected)):
        y_pos = 1 - (2 * v_idx / n_rows) - 0.01
        layout[f"legend{v_idx + 1}"] = dict(x=1.02, y=y_pos, xanchor="left", yanchor="top",
                                            bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
    fig.update_layout(**layout)
    fig.update_xaxes(rangeslider=dict(visible=True), rangeslider_thickness=0.04, row=n_rows, col=1)
    fig.show()
    return fig


def plot_price_with_posterior(df: pd.DataFrame, variant: str, price_col: str = "close", axis_col: Optional[str] = None,
                              ascending_axis: bool = True, rank_labels: Optional[Sequence[str]] = None,
                              color_price_by_confidence: bool = False, title: Optional[str] = None) -> go.Figure:
    """Single-variant convenience wrapper (mirror of ``plot_price_with_labels``)."""
    return plot_price_multiple_posteriors_with_subplots(df, variants=[variant], price_col=price_col, axis_col=axis_col,
                                                        ascending_axis=ascending_axis, rank_labels=rank_labels,
                                                        color_price_by_confidence=color_price_by_confidence,
                                                        title=title or f"{variant} — price + posterior")
