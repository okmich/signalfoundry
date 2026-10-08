"""
Look-ahead-free (prior-window) statistics.

Several features once normalised or binned with a statistic of the WHOLE series — a 99th percentile, ``pd.qcut``
volume bins, a per-bin mean, a min/max. Every such statistic lets bar ``t`` see bars after ``t``: the value at ``t``
changes when later data is appended, so a backtest reads information a live system never has. These helpers are the
causal replacements. Each statistic at bar ``t`` is computed from a rolling window of the ``lookback`` bars BEFORE
``t`` (bar ``t`` itself excluded), so:

* the value at ``t`` never changes when later bars arrive (truncation-invariant);
* the value does not depend on where the loaded history starts once the window is full;
* backtest and live streaming compute the same number with bounded memory.

During warm-up (fewer than ``min_periods`` prior observations) the statistic is NaN — an honest "not yet known",
never a value borrowed from the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Default prior window, in bars, for features that need a distributional statistic of their own history.
DEFAULT_LOOKBACK = 500


def _min_periods(lookback: int, min_periods: int | None) -> int:
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    return max(2, lookback // 4) if min_periods is None else max(1, int(min_periods))


def prior_rolling_quantile(x: pd.Series, q: float, lookback: int = DEFAULT_LOOKBACK,
                           min_periods: int | None = None) -> pd.Series:
    """Quantile ``q`` of the ``lookback`` observations strictly before each bar (NaN values are skipped)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    mp = _min_periods(lookback, min_periods)
    return x.shift(1).rolling(lookback, min_periods=mp).quantile(q)


def prior_rolling_edges(x: pd.Series, quantiles: list[float], lookback: int = DEFAULT_LOOKBACK,
                        min_periods: int | None = None) -> list[pd.Series]:
    """One prior-window quantile series per entry of ``quantiles`` (sorted ascending)."""
    return [prior_rolling_quantile(x, q, lookback, min_periods) for q in sorted(quantiles)]


def count_edges_at_or_below(x: pd.Series, edges: list[pd.Series]) -> pd.Series:
    """Per bar, how many of the (time-varying) ``edges`` are <= ``x`` — the ``np.digitize`` convention.

    NaN where ``x`` or any edge is NaN (warm-up), so a bar is never assigned to a bin it cannot yet be placed in.
    """
    if not edges:
        raise ValueError("at least one edge is required")
    count = sum((x >= e).astype(float) for e in edges)
    known = x.notna()
    for e in edges:
        known &= e.notna()
    return count.where(known)


def prior_rolling_bins(x: pd.Series, n_bins: int, lookback: int = DEFAULT_LOOKBACK,
                       min_periods: int | None = None) -> pd.Series:
    """Equal-frequency bin (0 … ``n_bins``-1) of ``x_t`` against the quantile edges of the prior ``lookback`` bars.

    The causal counterpart of ``pd.qcut(x, n_bins)``: the edges at ``t`` come from bars before ``t`` only. A value
    exactly on an edge goes to the upper bin (``np.digitize`` convention).
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    edges = prior_rolling_edges(x, [k / n_bins for k in range(1, n_bins)], lookback, min_periods)
    return count_edges_at_or_below(x, edges)


def prior_rolling_group_mean(x: pd.Series, group: pd.Series, n_groups: int, lookback: int = DEFAULT_LOOKBACK,
                             min_periods: int = 5) -> pd.Series:
    """Mean of ``x`` over the prior ``lookback`` bars that fell in the SAME group as bar ``t``.

    The causal counterpart of ``x.groupby(group).transform("mean")``. ``group`` holds integer labels
    0 … ``n_groups``-1 (NaN = unassigned). Requires ``min_periods`` same-group observations in the window.
    """
    out = pd.Series(np.nan, index=x.index, dtype=float)
    for g in range(n_groups):
        in_g = group == g
        mean_g = x.where(in_g).shift(1).rolling(lookback, min_periods=min_periods).mean()
        out[in_g] = mean_g[in_g]
    return out
