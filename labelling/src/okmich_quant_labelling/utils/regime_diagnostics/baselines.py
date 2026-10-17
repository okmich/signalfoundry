"""Per-fold baselines, fit on train only: hour-of-week means (L1), HAR beyond the clock (L2), winsor limits.

Hour-of-week buckets with fewer than ``min_count`` train values fall back to their hour-of-day mean, then to the
global mean. The L2 fit is a regression with hour-of-week fixed effects done by within-bucket demeaning
(Frisch-Waugh): target and HAR regressors are both demeaned by their train bucket means, then OLS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BUCKETS = 168
MIN_BUCKET_COUNT = 50
WINSOR_QUANTILES = (0.005, 0.995)


def bucket_means(values: np.ndarray, buckets: np.ndarray, min_count: int = MIN_BUCKET_COUNT) -> np.ndarray:
    """Mean of ``values`` per hour-of-week bucket (168,), with the hour-of-day and global fallbacks."""
    ok = np.isfinite(values)
    vals, bks = values[ok], buckets[ok]
    if vals.size == 0:
        return np.zeros(N_BUCKETS)
    count = np.bincount(bks, minlength=N_BUCKETS).astype(float)
    total = np.bincount(bks, weights=vals, minlength=N_BUCKETS)
    hour_count = count.reshape(7, 24).sum(axis=0)
    hour_total = total.reshape(7, 24).sum(axis=0)
    hour_mean = np.where(hour_count >= min_count, hour_total / np.maximum(hour_count, 1), vals.mean())
    return np.where(count >= min_count, total / np.maximum(count, 1), np.tile(hour_mean, 7))


@dataclass(frozen=True)
class ClockBaseline:
    means: np.ndarray  # (168,)

    @classmethod
    def fit(cls, y: np.ndarray, buckets: np.ndarray) -> "ClockBaseline":
        return cls(bucket_means(y, buckets))

    def residual(self, y: np.ndarray, buckets: np.ndarray) -> np.ndarray:
        return y - self.means[buckets]


HAR_BINS = 10


@dataclass(frozen=True)
class HarBaseline:
    """Flexible HAR with hour-of-week fixed effects: each trailing-vol regressor enters linearly and as train-decile
    steps, so any monotone or curved function of trailing vol is absorbed, not just a straight line. A label that
    merely restates trailing vol must not pass as information beyond this baseline (validation V3, see README.md)."""
    edges: list[np.ndarray]  # inner decile edges per regressor, from train
    y_means: np.ndarray  # (168,)
    design_means: np.ndarray  # (168, columns)
    beta: np.ndarray  # (columns,)

    @classmethod
    def fit(cls, y: np.ndarray, x: np.ndarray, buckets: np.ndarray) -> "HarBaseline":
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        quantiles = np.linspace(0, 1, HAR_BINS + 1)[1:-1]
        edges = [np.unique(np.quantile(x[ok, j], quantiles)) if ok.any() else np.array([]) for j in range(x.shape[1])]
        design = _har_design(x, edges)
        y_means = bucket_means(np.where(ok, y, np.nan), buckets)
        design_means = np.column_stack([bucket_means(np.where(ok, design[:, j], np.nan), buckets)
                                        for j in range(design.shape[1])])
        y_dm = y[ok] - y_means[buckets[ok]]
        design_dm = design[ok] - design_means[buckets[ok]]
        columns = design.shape[1]
        beta = np.linalg.lstsq(design_dm, y_dm, rcond=None)[0] if ok.sum() > columns else np.zeros(columns)
        return cls(edges, y_means, design_means, beta)

    def predict(self, x: np.ndarray, buckets: np.ndarray) -> np.ndarray:
        """Forecast of the target (log forward RV) from trailing vol and the clock; NaN where a regressor is NaN."""
        design = _har_design(x, self.edges)
        return self.y_means[buckets] + (design - self.design_means[buckets]) @ self.beta

    def residual(self, y: np.ndarray, x: np.ndarray, buckets: np.ndarray) -> np.ndarray:
        return y - self.predict(x, buckets)


@dataclass(frozen=True)
class TailThresholds:
    """Train quantiles of the HAR residual within deciles of the HAR forecast. A residual beyond its own bin's
    threshold is a tail event *given* the forecast level, so a label that only restates trailing vol carries no
    exceedance information beyond it."""
    edges: np.ndarray  # inner decile edges of the train forecast
    thresholds: np.ndarray  # (bins, levels)

    @classmethod
    def fit(cls, residual: np.ndarray, forecast: np.ndarray, levels) -> "TailThresholds":
        ok = np.isfinite(residual) & np.isfinite(forecast)
        edges = np.array([])
        if ok.any():
            edges = np.unique(np.quantile(forecast[ok], np.linspace(0, 1, HAR_BINS + 1)[1:-1]))
        bins = np.searchsorted(edges, forecast[ok], side="right")
        table = np.full((edges.size + 1, len(levels)), np.nan)
        for b in range(edges.size + 1):
            in_bin = residual[ok][bins == b]
            if in_bin.size:
                table[b] = np.quantile(in_bin, levels)
        return cls(edges, table)

    def exceed(self, residual: np.ndarray, forecast: np.ndarray, level_index: int, upper: bool) -> np.ndarray:
        """1.0 where the residual is beyond its bin's threshold, 0.0 where not, NaN where either input is NaN."""
        bins = np.searchsorted(self.edges, forecast, side="right")
        threshold = self.thresholds[np.minimum(bins, self.thresholds.shape[0] - 1), level_index]
        hit = residual > threshold if upper else residual < threshold
        return np.where(np.isfinite(residual) & np.isfinite(forecast), hit.astype(float), np.nan)


def _har_design(x: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """Linear terms plus one-hot decile steps (first bin dropped) per regressor; NaN rows stay NaN."""
    columns = [x]
    for j, inner in enumerate(edges):
        bins = np.searchsorted(inner, x[:, j], side="right")
        steps = (bins[:, None] == np.arange(1, inner.size + 1)).astype(float)
        steps[~np.isfinite(x[:, j])] = np.nan
        columns.append(steps)
    return np.hstack(columns)


def winsor_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -np.inf, np.inf
    lo, hi = np.quantile(finite, WINSOR_QUANTILES)
    return float(lo), float(hi)
