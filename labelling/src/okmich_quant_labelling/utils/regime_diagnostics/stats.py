"""Per-state moments, eta-squared, one-vs-rest effects, Holm, and the moving-block bootstrap over trading days.

Labels are always categorical codes 0..K-1 (-1 = unknown); nothing here treats a code as a number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Moments:
    n: np.ndarray  # (K,) count per state
    s: np.ndarray  # (K,) sum per state
    q: np.ndarray  # (K,) sum of squares per state

    @classmethod
    def of(cls, codes: np.ndarray, y: np.ndarray, k: int) -> "Moments":
        """Moments of ``y`` by state; bars with an unknown state or a non-finite ``y`` are left out."""
        ok = (codes >= 0) & np.isfinite(y)
        c, v = codes[ok], y[ok]
        return cls(np.bincount(c, minlength=k).astype(float), np.bincount(c, weights=v, minlength=k),
                   np.bincount(c, weights=v * v, minlength=k))

    @property
    def means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.s / self.n

    def eta2(self) -> float:
        """Between-state share of the total sum of squares."""
        n_total = self.n.sum()
        if n_total < 2:
            return np.nan
        grand = self.s.sum() / n_total
        sst = self.q.sum() - n_total * grand ** 2
        present = self.n > 0
        ssb = (self.s[present] ** 2 / self.n[present]).sum() - n_total * grand ** 2
        return float(ssb / sst) if sst > 0 else np.nan

    def one_vs_rest(self) -> np.ndarray:
        """Mean of each state minus the mean of all other states (NaN when either side is empty)."""
        n_total, s_total = self.n.sum(), self.s.sum()
        rest_n = n_total - self.n
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((self.n > 0) & (rest_n > 0), self.s / self.n - (s_total - self.s) / rest_n, np.nan)

    def between_ss(self) -> float:
        n_total = self.n.sum()
        if n_total == 0:
            return 0.0
        present = self.n > 0
        return float((self.s[present] ** 2 / self.n[present]).sum() - self.s.sum() ** 2 / n_total)


@dataclass(frozen=True)
class Studentized:
    """One-vs-rest effects divided by a day-clustered standard error, and their sum of squares as the omnibus.

    Rotation and permutation tests of raw mean differences are not robust when the label lines up with the
    target's heteroskedasticity or its state-dependent autocorrelation: the rotation breaks that alignment and the
    null comes out too narrow (synthetic V3: z-spread 1.3, 3x the nominal size). Studentizing makes the statistic
    close to pivotal, so the rotation null calibrates it (Chung & Romano 2013). Clustering by trading day absorbs
    the intraday autocorrelation of overlapping forward windows.
    """
    effect: np.ndarray  # (K,) state mean minus the mean of all other states
    t: np.ndarray  # (K,) effect / day-clustered standard error

    @property
    def omnibus(self) -> float:
        finite = np.isfinite(self.t)
        return float((self.t[finite] ** 2).sum()) if finite.any() else np.nan

    @classmethod
    def of(cls, codes: np.ndarray, y: np.ndarray, day: np.ndarray, k: int, n_days: int) -> "Studentized":
        ok = (codes >= 0) & np.isfinite(y)
        cell = day[ok] * k + codes[ok]
        size = n_days * k
        day_sum = np.bincount(cell, weights=y[ok], minlength=size).reshape(n_days, k)
        day_n = np.bincount(cell, minlength=size).astype(float).reshape(n_days, k)
        n_state, s_state = day_n.sum(axis=0), day_sum.sum(axis=0)
        n_rest, s_rest = n_state.sum() - n_state, s_state.sum() - s_state
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_state, mean_rest = s_state / n_state, s_rest / n_rest
            effect = mean_state - mean_rest
            # Each day's contribution to the effect's estimation error (influence function summed over the day)
            rest_sum = day_sum.sum(axis=1, keepdims=True) - day_sum
            rest_n = day_n.sum(axis=1, keepdims=True) - day_n
            influence = (day_sum - day_n * mean_state) / n_state - (rest_sum - rest_n * mean_rest) / n_rest
            se = np.sqrt((influence ** 2).sum(axis=0))
            t = np.where((n_state > 0) & (n_rest > 0) & (se > 0), effect / se, np.nan)
        return cls(np.where(np.isfinite(t), effect, np.nan), t)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values; NaN inputs stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    m = finite.size
    if m == 0:
        return out
    order = finite[np.argsort(p[finite])]
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out[order] = adjusted
    return out


def block_day_weights(n_days: int, n_draws: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap of trading days as weights: (n_draws, n_days) = times each day is drawn.

    Any per-day total (sums, counts, histograms) is resampled by one matrix product with these weights.
    """
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n_days, size=(n_draws, n_blocks))
    days = ((starts[:, :, None] + np.arange(block)) % n_days).reshape(n_draws, -1)[:, :n_days]
    flat = (days + np.arange(n_draws)[:, None] * n_days).ravel()
    return np.bincount(flat, minlength=n_draws * n_days).reshape(n_draws, n_days).astype(float)


def block_bootstrap_means(day_sum: np.ndarray, day_count: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Means per column under day-block bootstrap ``weights``: (n_draws, columns) from (days, columns) totals."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ day_sum) / (weights @ day_count)


def quantiles_from_histogram(cumulative: np.ndarray, edges: np.ndarray, levels) -> np.ndarray:
    """Quantiles from cumulative shares at each bin's upper edge (..., bins), linear within the bin: (..., levels)."""
    n_bins = cumulative.shape[-1]
    out = np.empty(cumulative.shape[:-1] + (len(levels),))
    for i, level in enumerate(levels):
        idx = np.minimum((cumulative < level).sum(axis=-1, keepdims=True), n_bins - 1)
        upper = np.take_along_axis(cumulative, idx, axis=-1)
        lower = np.where(idx > 0, np.take_along_axis(cumulative, np.maximum(idx - 1, 0), axis=-1), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(upper > lower, (level - lower) / (upper - lower), 0.5)
        out[..., i] = (edges[idx] + frac * (edges[idx + 1] - edges[idx]))[..., 0]
    return out
