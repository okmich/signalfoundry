"""Visualizations for triple-barrier labels and meta-labels."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_barriers(close: pd.Series, events: pd.Series, labels: pd.DataFrame, pt_sl: List[float],
                  volatility: pd.Series, max_events: int = 20, figsize=(14, 6)) -> plt.Figure:
    pt, sl = float(pt_sl[0]), float(pt_sl[1])
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(close.index, close.values, color="black", linewidth=0.8, alpha=0.6, label="close")

    # Match get_labels' vol alignment policy.
    vol_aligned = volatility.reindex(close.index, method="ffill")
    plotted = events.iloc[:max_events]
    color_by_label = {1: "tab:green", -1: "tab:red", 0: "tab:gray"}

    for t0, t1 in plotted.items():
        if t0 not in close.index or t0 not in labels.index:
            continue
        entry = float(close.loc[t0])
        vol = float(vol_aligned.loc[t0]) if t0 in vol_aligned.index else np.nan
        if not np.isfinite(vol) or vol <= 0:
            continue
        upper = entry * (1.0 + pt * vol) if pt > 0 else None
        lower = entry * (1.0 - sl * vol) if sl > 0 else None
        if upper is not None:
            ax.hlines(upper, t0, t1, colors="green", linestyles="dashed", alpha=0.5)
        if lower is not None:
            ax.hlines(lower, t0, t1, colors="red", linestyles="dashed", alpha=0.5)
        ax.axvline(t1, color="gray", linestyle="dotted", alpha=0.3)

        seg = close.loc[t0:t1]
        label = int(labels.loc[t0, "label"])
        ax.plot(seg.index, seg.values, color=color_by_label.get(label, "black"), linewidth=1.5)

    ax.set_title("Triple Barrier Method — barriers and realized paths")
    ax.set_ylabel("price")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_label_distribution(labels: pd.DataFrame, title: str = "Label Distribution") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    color_by_value = {-1: "tab:red", 0: "tab:gray", 1: "tab:green"}
    counts = labels["label"].value_counts().sort_index()
    bar_colors = [color_by_value.get(int(v), "black") for v in counts.index]
    axes[0].bar(counts.index.astype(str), counts.values, color=bar_colors)
    axes[0].set_title(f"{title} — counts")
    axes[0].set_xlabel("label")
    axes[0].set_ylabel("count")

    rolling_balance = (labels["label"] == 1).rolling(50, min_periods=10).mean()
    axes[1].plot(rolling_balance.index, rolling_balance.values, color="tab:blue")
    axes[1].axhline(0.5, color="gray", linestyle="dashed", alpha=0.5)
    axes[1].set_title("Rolling proportion of +1 labels (50-event window)")
    axes[1].set_ylabel("proportion")

    fig.tight_layout()
    return fig


def plot_meta_label_overlap(triple_barrier_labels: pd.DataFrame, meta_labels: pd.Series,
                            primary_predictions: pd.Series) -> plt.Figure:
    common = primary_predictions.index.intersection(triple_barrier_labels.index).intersection(meta_labels.index)
    primary = np.sign(primary_predictions.loc[common]).astype(int)
    tb = triple_barrier_labels.loc[common, "label"].astype(int)
    meta = meta_labels.loc[common].astype(int)

    primary_dirs = [-1, 1]
    tb_outcomes = [-1, 0, 1]
    grid = np.full((len(primary_dirs), len(tb_outcomes)), np.nan)

    for i, p in enumerate(primary_dirs):
        for j, o in enumerate(tb_outcomes):
            mask = (primary == p) & (tb == o)
            if mask.any():
                grid[i, j] = meta[mask].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tb_outcomes)))
    ax.set_xticklabels([str(o) for o in tb_outcomes])
    ax.set_yticks(range(len(primary_dirs)))
    ax.set_yticklabels([str(p) for p in primary_dirs])
    ax.set_xlabel("triple-barrier outcome")
    ax.set_ylabel("primary direction")
    ax.set_title("Meta-label proportion (primary × outcome)")

    for i in range(len(primary_dirs)):
        for j in range(len(tb_outcomes)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.3 < v < 0.7 else "white", fontsize=10)

    fig.colorbar(im, ax=ax, label="P(meta=1)")
    fig.tight_layout()
    return fig
