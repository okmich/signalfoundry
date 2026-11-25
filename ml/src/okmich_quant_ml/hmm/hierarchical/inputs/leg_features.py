"""Per-leg features for event-time models, and an encoder that combines discrete ones into one symbol.

Discrete features (``DiscreteLegFeature``) give every leg a level in 0..n_levels-1:

- ``LegDirection``: down / up.
- ``TayalTrend`` (Tayal 2009, f1, p. 58): market structure over the last five turning points. Up-trend if
  e(n-4) < e(n-2) < e(n) and e(n-3) < e(n-1) (higher highs and higher lows), down-trend for the mirror, else none.
  It does not depend on the leg's direction or size.
- ``TayalVolume`` (f2, p. 59): is volume strengthening or weakening along the leg? phi(n) = average volume per second
  over leg n (end pivots included); the ratios phi(n)/phi(n-1), phi(n)/phi(n-2), phi(n-1)/phi(n-2) are each
  discretised at +/- alpha (25%), then grouped into strengthens / weakens / indeterminate.
- ``QuantileLegFeature``: any continuous ``LegScore`` cut into equal-mass bins frozen on the training legs, e.g.
  ``LegSizeScore`` (|leg| / sigma), ``BOCPDBreakScore`` (changepoint intensity) or ``SignedVolumeScore``.

All features are causal: a leg's value uses bars up to its extremum (``end_bar``), which precedes its confirmation.
Features that look back over earlier legs (Tayal's) need those legs, so compute them on the full leg list and slice
afterwards (``legs_confirmed_from``), not on a sliced list.
"""
from __future__ import annotations

import abc
import copy
from enum import StrEnum
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from okmich_quant_features.microstructure.order_flow import vir
from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector, NormalInverseGammaModel

from .zigzag import MarketData, Zigzag, ZigzagDirection, check_legs_align, realized_vol

_DEFAULT_VOL_WINDOW = pd.Timedelta("24h")


class VolumeRateUnit(StrEnum):
    """Denominator of a leg's average volume."""

    SECOND = "second"   # Tayal: volume / (seconds spanned + 1)
    BAR = "bar"         # volume / bars spanned: immune to weekend and session gaps inside a leg


class ScoreReducer(StrEnum):
    """How a per-bar score is reduced over a leg's bars."""

    MEAN = "mean"
    MAX = "max"


# ======================================================================================================================
# Discretisation
# ======================================================================================================================
def quantile_boundaries(scores: np.ndarray, n_bins: int, rel_tol: float = 1e-9) -> np.ndarray:
    """Interior cut points of ``n_bins`` equal-mass bins (``n_bins - 1`` strictly increasing values).

    Degenerate scores (empty, or a spread within ``rel_tol`` of their scale, e.g. a constant feature carrying only
    floating-point noise) put every value in the middle bin instead of letting rounding noise pick a bin.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    mid = (n_bins - 1) // 2
    if finite.size == 0:
        return np.arange(1, n_bins, dtype=np.float64) - mid - 0.5
    centre = float(np.median(finite))
    scale = max(float(np.max(np.abs(finite))), np.finfo(float).tiny)
    if float(np.ptp(finite)) <= rel_tol * scale:
        step = max(scale * 1e-6, np.finfo(float).tiny)
        return np.array([centre + (j - mid + 0.5) * step for j in range(n_bins - 1)], dtype=np.float64)
    bounds = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    for j in range(1, len(bounds)):                 # strictly increasing for np.digitize
        if bounds[j] <= bounds[j - 1]:
            bounds[j] = np.nextafter(bounds[j - 1], np.inf)
    return bounds


def bucketize(scores: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Bin index in ``[0, len(boundaries)]`` of each score (higher score -> higher bin)."""
    return np.digitize(np.asarray(scores, dtype=np.float64), np.asarray(boundaries, dtype=np.float64)).astype(np.int64)


# ======================================================================================================================
# Raw per-leg quantities
# ======================================================================================================================
def leg_signs(legs: Sequence[Zigzag]) -> np.ndarray:
    """+1 for an up leg, -1 for a down leg."""
    return np.array([1 if z.direction is ZigzagDirection.UP else -1 for z in legs], dtype=np.int64)


def pivot_prices(legs: Sequence[Zigzag]) -> np.ndarray:
    """Turning-point prices: the first leg's start, then every leg's end (len = n_legs + 1)."""
    if not legs:
        return np.zeros(0)
    return np.array([legs[0].start_price] + [z.end_price for z in legs], dtype=np.float64)


def tayal_structure(legs: Sequence[Zigzag]) -> np.ndarray:
    """Tayal f1 per leg: +1 up-trend, -1 down-trend, 0 none (also 0 for the first three legs, which lack history)."""
    e = pivot_prices(legs)
    out = np.zeros(len(legs), dtype=np.int64)
    for n in range(3, len(legs)):
        e4, e3, e2, e1, e0 = e[n - 3], e[n - 2], e[n - 1], e[n], e[n + 1]   # e(n-4) .. e(n)
        if e4 < e2 < e0 and e3 < e1:
            out[n] = 1
        elif e4 > e2 > e0 and e3 > e1:
            out[n] = -1
    return out


def volume_rate(legs: Sequence[Zigzag], market_data: MarketData,
                unit: VolumeRateUnit = VolumeRateUnit.SECOND) -> np.ndarray:
    """phi per leg: total volume from the opening to the closing pivot (both included) per second (+1) or per bar."""
    market_data.require("volume_rate", "volume")
    check_legs_align(legs, market_data.close.index, "volume_rate")
    unit = VolumeRateUnit(unit)
    vol = np.nan_to_num(market_data.volume.to_numpy(dtype=np.float64))
    csum = np.concatenate([[0.0], np.cumsum(vol)])
    index = market_data.close.index
    out = np.empty(len(legs))
    for k, z in enumerate(legs):
        total = csum[z.end_bar + 1] - csum[z.start_bar]
        if unit is VolumeRateUnit.SECOND:
            span = (index[z.end_bar] - index[z.start_bar]).total_seconds()
            out[k] = total / (span + 1.0)
        else:
            out[k] = total / (z.end_bar - z.start_bar + 1)
    return out


def _ratio_signal(num: np.ndarray, den: np.ndarray, alpha: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = num / den
    out = np.zeros(len(theta), dtype=np.int64)
    ok = np.isfinite(theta) & (den > 0)
    out[ok & (theta - 1.0 > alpha)] = 1
    out[ok & (1.0 - theta > alpha)] = -1
    return out


def tayal_volume_signal(legs: Sequence[Zigzag], market_data: MarketData, alpha: float = 0.25,
                        unit: VolumeRateUnit = VolumeRateUnit.SECOND) -> np.ndarray:
    """Tayal f2 per leg: +1 volume strengthens, -1 weakens, 0 indeterminate (also 0 for the first two legs)."""
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    phi = volume_rate(legs, market_data, unit)
    out = np.zeros(len(legs), dtype=np.int64)
    if len(legs) < 3:
        return out
    cur, prev, prev2 = phi[2:], phi[1:-1], phi[:-2]
    t1, t2, t3 = _ratio_signal(cur, prev, alpha), _ratio_signal(cur, prev2, alpha), _ratio_signal(prev, prev2, alpha)
    strengthens = (t1 == 1) & (t2 > -1) & (t3 < 1)
    weakens = (t1 == -1) & (t2 < 1) & (t3 > -1)
    out[2:] = np.where(strengthens, 1, np.where(weakens, -1, 0))
    return out


# ======================================================================================================================
# Continuous scores
# ======================================================================================================================
class LegScore(abc.ABC):
    """A continuous per-leg quantity (to be cut into bins by ``QuantileLegFeature`` or used raw)."""

    name: str

    @abc.abstractmethod
    def scores(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray: ...


class LegSizeScore(LegScore):
    """|leg| / realised volatility at the leg's extremum: how big the swing was in units of recent noise."""

    name = "leg_size"

    def __init__(self, vol_window: Union[pd.Timedelta, int] = _DEFAULT_VOL_WINDOW):
        self.vol_window = vol_window

    def scores(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        check_legs_align(legs, market_data.close.index, "LegSizeScore")
        sig = realized_vol(market_data.close, self.vol_window).to_numpy(dtype=np.float64)
        out = np.empty(len(legs))
        for k, z in enumerate(legs):
            s = sig[z.end_bar]
            out[k] = z.magnitude / s if s > 0 else 0.0
        return out


class BOCPDBreakScore(LegScore):
    """Changepoint intensity over a leg: P(a bar's return opened a new BOCPD run), averaged (or maxed) over the leg.

    Returns are standardised by the realised volatility of the bars *before* them, so the Normal-Inverse-Gamma prior
    (unit scale) fits; the probability read is the detector's ``posterior[1] / (1 - posterior[0])``. It measures
    surprise in the return distribution and does not depend on the leg's direction: a break, not flow.
    """

    name = "bocpd_break"

    def __init__(self, hazard_rate: float = 1.0 / 250.0, r_max: int = 200, reducer: ScoreReducer = ScoreReducer.MEAN,
                 vol_window: Union[pd.Timedelta, int] = _DEFAULT_VOL_WINDOW):
        self.hazard_rate, self.r_max = float(hazard_rate), int(r_max)
        self.reducer = ScoreReducer(reducer)
        self.vol_window = vol_window

    def changepoint_prob(self, close: pd.Series) -> np.ndarray:
        """Per-bar probability that the bar's return opened a new run (bar 0 = 0)."""
        log_ret = np.diff(np.log(close.to_numpy(dtype=np.float64)))
        if log_ret.size == 0:
            return np.zeros(len(close))
        sigma = realized_vol(close, self.vol_window).to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = log_ret / sigma[:-1]                  # return at bar t over sigma known at bar t-1
        z[~np.isfinite(z)] = 0.0
        detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=self.hazard_rate,
                                                     r_max=self.r_max)
        post = detector.batch(z)
        observed = 1.0 - post[:, 0]
        cp = np.where(observed > 0, post[:, 1] / np.where(observed > 0, observed, 1.0), 1.0)
        return np.concatenate([[0.0], cp])

    def scores(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        check_legs_align(legs, market_data.close.index, "BOCPDBreakScore")
        cp = self.changepoint_prob(market_data.close)
        reduce = np.mean if self.reducer is ScoreReducer.MEAN else np.max
        out = np.empty(len(legs))
        for k, z in enumerate(legs):
            span = cp[z.start_bar + 1: z.end_bar + 1]           # the returns inside the leg
            out[k] = float(reduce(span)) if span.size else float(cp[z.end_bar])
        return out


class SignedVolumeScore(LegScore):
    """Net signed pressure over the leg (beta-CLV signed volume / volume), times the leg's sign: > 0 = with the leg."""

    name = "signed_volume"

    def scores(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        market_data.require("SignedVolumeScore", "high", "low", "volume")
        check_legs_align(legs, market_data.close.index, "SignedVolumeScore")
        pressure = np.asarray(vir(market_data.high, market_data.low, market_data.close, market_data.volume),
                              dtype=np.float64)
        vol = market_data.volume.to_numpy(dtype=np.float64)
        signed = np.nan_to_num(pressure) * np.nan_to_num(vol)
        out = np.empty(len(legs))
        for k, z in enumerate(legs):
            v = vol[z.start_bar: z.end_bar + 1]
            d = signed[z.start_bar: z.end_bar + 1]
            total = float(np.nansum(v))
            net = float(np.nansum(d)) / total if total > 0 else 0.0
            out[k] = net * (1.0 if z.direction is ZigzagDirection.UP else -1.0)
        return out


# ======================================================================================================================
# Discrete features
# ======================================================================================================================
class DiscreteLegFeature(abc.ABC):
    """A per-leg feature with ``n_levels`` levels. ``fit`` freezes anything learnt from the training legs."""

    name: str
    n_levels: int
    level_names: tuple[str, ...]

    def fit(self, legs: Sequence[Zigzag], market_data: MarketData) -> "DiscreteLegFeature":
        return self

    @abc.abstractmethod
    def transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        """(n_legs,) int levels."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(levels={list(self.level_names)})"


class LegDirection(DiscreteLegFeature):
    """0 = down leg, 1 = up leg (the ``ZigzagDirection`` value)."""

    name, n_levels, level_names = "direction", 2, ("down", "up")

    def transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        return np.array([int(z.direction) for z in legs], dtype=np.int64)


class TayalTrend(DiscreteLegFeature):
    """Tayal f1: 0 = down-trend, 1 = no trend, 2 = up-trend (lower/higher highs and lows over five turning points)."""

    name, n_levels, level_names = "trend", 3, ("down_trend", "no_trend", "up_trend")

    def transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        return tayal_structure(legs) + 1


class TayalVolume(DiscreteLegFeature):
    """Tayal f2: 0 = volume weakens, 1 = indeterminate, 2 = strengthens. Needs volume."""

    name, n_levels, level_names = "volume", 3, ("weakens", "indeterminate", "strengthens")

    def __init__(self, alpha: float = 0.25, unit: VolumeRateUnit = VolumeRateUnit.SECOND):
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        self.alpha, self.unit = float(alpha), VolumeRateUnit(unit)

    def transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        return tayal_volume_signal(legs, market_data, self.alpha, self.unit) + 1


class QuantileLegFeature(DiscreteLegFeature):
    """A continuous ``LegScore`` cut into ``n_bins`` equal-mass bins whose boundaries are frozen at ``fit``."""

    def __init__(self, score: LegScore, n_bins: int = 3, name: Optional[str] = None):
        if not isinstance(score, LegScore):
            raise TypeError("score must be a LegScore")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.score, self.n_levels = score, int(n_bins)
        self.name = name or score.name
        self.level_names = tuple(f"{self.name}_q{j}" for j in range(self.n_levels))
        self.boundaries_: Optional[np.ndarray] = None

    def fit(self, legs: Sequence[Zigzag], market_data: MarketData) -> "QuantileLegFeature":
        self.boundaries_ = quantile_boundaries(self.score.scores(legs, market_data), self.n_levels)
        return self

    def transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        if self.boundaries_ is None:
            raise RuntimeError(f"{self.name}: call fit() before transform()")
        return bucketize(self.score.scores(legs, market_data), self.boundaries_)


# ======================================================================================================================
# Symbols
# ======================================================================================================================
class SymbolEncoder:
    """Combine discrete leg features into one symbol per leg (mixed radix, first feature most significant).

    With ``[LegDirection(), TayalTrend(), TayalVolume()]`` this is Tayal's 18-symbol alphabet (down symbols 0..8, up
    symbols 9..17); without ``TayalVolume`` it is the 6-symbol structure-only alphabet.
    """

    def __init__(self, features: Sequence[DiscreteLegFeature]):
        features = list(features)
        if not features:
            raise ValueError("SymbolEncoder needs at least one feature")
        if any(not isinstance(f, DiscreteLegFeature) for f in features):
            raise TypeError("features must be DiscreteLegFeature instances")
        names = [f.name for f in features]
        if len(set(names)) != len(names):
            raise ValueError(f"feature names must be unique, got {names}")
        self.features = features
        self.radix = np.array([f.n_levels for f in features], dtype=np.int64)
        self.n_symbols = int(np.prod(self.radix))
        self._place = np.array([int(np.prod(self.radix[k + 1:])) for k in range(len(features))], dtype=np.int64)

    def fit(self, legs: Sequence[Zigzag], market_data: MarketData) -> "SymbolEncoder":
        """Fit every feature; atomic (copies are fitted and swapped in only when all succeed)."""
        fitted = [copy.deepcopy(f).fit(legs, market_data) for f in self.features]
        self.features = fitted
        return self

    def levels(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        """(n_legs, n_features) levels."""
        cols = [f.transform(legs, market_data) for f in self.features]
        for f, col in zip(self.features, cols):
            if col.size and (col.min() < 0 or col.max() >= f.n_levels):
                raise ValueError(f"Feature {f.name!r} produced levels outside 0..{f.n_levels - 1}")
        return np.column_stack(cols) if cols[0].size else np.zeros((0, len(cols)), dtype=np.int64)

    def encode(self, levels: np.ndarray) -> np.ndarray:
        return np.asarray(levels, dtype=np.int64) @ self._place

    def decode(self, symbol: int) -> tuple[int, ...]:
        if not 0 <= int(symbol) < self.n_symbols:
            raise ValueError(f"symbol must be in 0..{self.n_symbols - 1}, got {symbol}")
        return tuple(int(int(symbol) // p % r) for p, r in zip(self._place, self.radix))

    def transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        return self.encode(self.levels(legs, market_data))

    def fit_transform(self, legs: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        return self.fit(legs, market_data).transform(legs, market_data)

    def symbols_with(self, feature: str, level: int) -> np.ndarray:
        """All symbols whose ``feature`` takes ``level`` (e.g. every up-leg symbol: a leaf's support)."""
        k = [f.name for f in self.features].index(feature)
        return np.array([s for s in range(self.n_symbols) if self.decode(s)[k] == level], dtype=np.int64)

    def symbol_names(self) -> list[str]:
        return ["/".join(f.level_names[lv] for f, lv in zip(self.features, self.decode(s)))
                for s in range(self.n_symbols)]

    def __repr__(self) -> str:
        return f"SymbolEncoder({[f.name for f in self.features]}, n_symbols={self.n_symbols})"


# ======================================================================================================================
# Continuous feature frame
# ======================================================================================================================
def leg_feature_frame(legs: Sequence[Zigzag], market_data: MarketData,
                      vol_window: Union[pd.Timedelta, int] = _DEFAULT_VOL_WINDOW,
                      volume_unit: VolumeRateUnit = VolumeRateUnit.SECOND, alpha: float = 0.25) -> pd.DataFrame:
    """Continuous and signed per-leg features indexed by confirmation time (for Gaussian / Student-t emissions).

    Columns: ``direction`` (+1/-1), ``signed_size_sigma`` (signed |leg| / sigma), ``log_duration_bars`` (pivot to
    pivot), ``confirm_lag_bars`` (extremum to confirmation), ``structure`` (Tayal f1), and, when volume is present,
    ``log_volume_rate`` and ``volume_signal`` (Tayal f2).
    """
    sign = leg_signs(legs)
    size = LegSizeScore(vol_window).scores(legs, market_data)
    frame = pd.DataFrame({
        "direction": sign.astype(np.float64),
        "signed_size_sigma": sign * size,
        "log_duration_bars": np.log([max(z.end_bar - z.start_bar, 1) for z in legs]),
        "confirm_lag_bars": np.array([z.confirm_bar - z.end_bar for z in legs], dtype=np.float64),
        "structure": tayal_structure(legs).astype(np.float64),
    }, index=pd.DatetimeIndex([z.end_time for z in legs], name="confirm_time") if legs else None)
    if market_data.volume is not None:
        phi = volume_rate(legs, market_data, volume_unit)
        frame["log_volume_rate"] = np.log(np.maximum(phi, 1e-300))
        frame["volume_signal"] = tayal_volume_signal(legs, market_data, alpha, volume_unit).astype(np.float64)
    return frame
