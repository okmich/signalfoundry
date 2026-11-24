"""Zigzag legs: turn a price series into a sequence of swings (directional-change legs) for event-time models.

A leg runs from one confirmed turning point to the next. A turning point is confirmed once price has reversed by the
threshold theta (a fraction of price), so theta is the *leg size*: it sets how big a swing must be to count, and so
the time scale an event-time model reads (small theta = many short legs, large theta = few long swings). It is always
a parameter (``LegThreshold``), never hard-coded:

- ``TARGET_RATE``: calibrate theta on the training window so legs arrive at a target rate per observed hour
  (bars x bar interval, so weekends and removed sessions do not dilute the rate);
- ``K_SIGMA``: theta = k x the training window's typical per-bar volatility;
- ``FIXED``: theta given directly (e.g. 0.001 = 10 bp).

Causality. A leg is known only at its ``confirm_bar`` (the bar whose price confirmed the reversal), not at its
extremum ``end_bar``. ``ZigzagLegBuilder`` freezes theta at fit time and reuses it, so transforming a later block never
reads that block's own volatility. Legs confirmed by bar t are identical whether or not later bars are supplied.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse

_HOUR = pd.Timedelta("1h")


class ZigzagDirection(IntEnum):
    """Direction of a leg. The value doubles as the direction level of a symbol (down = 0, up = 1)."""

    DOWN = 0
    UP = 1


class VolReducer(StrEnum):
    """How a window's per-bar volatility series is reduced to one representative value."""

    MEDIAN = "median"
    MEAN = "mean"


class ThresholdMode(StrEnum):
    """How the zigzag threshold (leg size) is chosen."""

    TARGET_RATE = "target_rate"   # value = target legs per hour (float or (low, high) band)
    K_SIGMA = "k_sigma"           # value = k, theta = k * reduce(sigma) on the training window
    FIXED = "fixed"               # value = theta as a fraction of price


# ----------------------------------------------------------------------------------------------------------------------
# Realised vol and threshold
# ----------------------------------------------------------------------------------------------------------------------
def realized_vol(prices: pd.Series, window: pd.Timedelta | int, min_periods: int = 2) -> pd.Series:
    """Rolling std of log returns, forward-filled only (causal; the warm-up stays NaN, never back-filled).

    A ``pd.Timedelta`` window requires a ``DatetimeIndex``.
    """
    if isinstance(window, pd.Timedelta) and not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("A pd.Timedelta realized_vol window requires prices to have a DatetimeIndex.")
    log_ret = np.log(prices).diff()
    sigma = log_ret.rolling(window, min_periods=min_periods).std()
    return sigma.ffill()


def vol_scaled_threshold(k: float, sigma_realized: pd.Series | np.ndarray, reducer: str = VolReducer.MEDIAN) -> float:
    """Representative threshold ``theta = k * reduce(sigma_realized)`` over a window (``reducer``: median or mean)."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    values = np.asarray(sigma_realized, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("sigma_realized has no finite values")
    reducer = VolReducer(reducer)
    sigma_ref = float(np.median(values)) if reducer is VolReducer.MEDIAN else float(np.mean(values))
    if sigma_ref <= 0:
        raise ValueError(f"reduced sigma_realized must be positive, got {sigma_ref}")
    return k * sigma_ref


# ----------------------------------------------------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketData:
    """Bar data behind leg features. Only ``close`` is always required.

    Leg features read these series by bar *position*, so every series must share ``close``'s index exactly (same
    timestamps, same order) and that index must be strictly increasing; anything else is rejected here rather than
    silently pairing a leg with another bar's volume.
    """

    close: pd.Series
    high: Optional[pd.Series] = None
    low: Optional[pd.Series] = None
    volume: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        if not isinstance(self.close, pd.Series):
            raise TypeError("MarketData.close must be a pandas Series")
        index = self.close.index
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("MarketData.close needs a strictly increasing index (no duplicates, in time order)")
        for name in ("high", "low", "volume"):
            series = getattr(self, name)
            if series is None:
                continue
            if not isinstance(series, pd.Series):
                raise TypeError(f"MarketData.{name} must be a pandas Series")
            if not series.index.equals(index):
                raise ValueError(f"MarketData.{name} is not indexed like close: features read bars by position, so "
                                 f"every series must share close's exact index (reindex or align them first)")

    def require(self, who: str, *fields: str) -> None:
        """Raise a clear error if any of ``fields`` (e.g. "volume", "high") is missing."""
        missing = [f for f in fields if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{who} requires {list(fields)}; missing: {missing}")

    @classmethod
    def from_ohlcv(cls, df: pd.DataFrame, *, close: str = "close", high: str = "high", low: str = "low",
                   volume: str = "volume") -> "MarketData":
        """Build from an OHLCV frame. Missing high/low/volume columns are left as None."""
        return cls(close=df[close], high=df[high] if high in df.columns else None,
                   low=df[low] if low in df.columns else None, volume=df[volume] if volume in df.columns else None)

    def require_volume(self, who: str) -> None:
        """Raise a clear error if high/low/close/volume are not all present."""
        missing = [n for n, s in (("high", self.high), ("low", self.low), ("volume", self.volume)) if s is None]
        if missing:
            raise ValueError(f"{who} requires high/low/close/volume; missing: {missing}")


@dataclass(frozen=True)
class Zigzag:
    """One completed leg (pivot to pivot). Observable at ``confirm_bar``, so ``end_time`` is causal."""

    seq_index: int
    direction: ZigzagDirection
    start_bar: int      # position of the leg's opening pivot
    end_bar: int        # position of the leg's closing pivot (the confirmed extremum)
    start_price: float
    end_price: float
    magnitude: float    # |end_price - start_price| / start_price
    confirm_bar: int    # position of the bar that confirmed the reversal and completed this leg
    start_time: pd.Timestamp
    end_time: pd.Timestamp  # clock time the leg completed (== time at confirm_bar)


# ----------------------------------------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------------------------------------
def _extract_pivots(prices: np.ndarray, upturn_dc: np.ndarray,
                    downturn_dc: np.ndarray) -> list[tuple[int, float, bool, int]]:
    """Confirmed pivots ``(pivot_bar, pivot_price, is_peak, confirm_bar)`` from ``idc_parse`` flags, in time order."""
    pivots: list[tuple[int, float, bool, int]] = []
    started = False
    mode = 0                 # +1 uptrend (tracking a peak), -1 downtrend (tracking a trough)
    ext_bar, ext_price = 0, prices[0]
    for i in range(len(prices)):
        if not started:
            if upturn_dc[i]:
                j = int(np.argmin(prices[: i + 1]))
                pivots.append((j, float(prices[j]), False, i))
                mode, started = 1, True
                ext_bar, ext_price = i, float(prices[i])
            elif downturn_dc[i]:
                j = int(np.argmax(prices[: i + 1]))
                pivots.append((j, float(prices[j]), True, i))
                mode, started = -1, True
                ext_bar, ext_price = i, float(prices[i])
            continue
        if mode == 1:
            if prices[i] > ext_price:
                ext_price, ext_bar = float(prices[i]), i
            if downturn_dc[i]:
                pivots.append((ext_bar, ext_price, True, i))
                mode = -1
                ext_price, ext_bar = float(prices[i]), i
        else:
            if prices[i] < ext_price:
                ext_price, ext_bar = float(prices[i]), i
            if upturn_dc[i]:
                pivots.append((ext_bar, ext_price, False, i))
                mode = 1
                ext_price, ext_bar = float(prices[i]), i
    return pivots


def aggregate_zigzags(prices: pd.Series, theta: float, alpha: float = 1.0) -> list[Zigzag]:
    """Completed directional-change legs of ``prices`` at threshold ``theta``; directions strictly alternate."""
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    if len(prices) < 2:                                   # an empty or one-bar feed has no turning point yet
        return []
    df = idc_parse(prices, theta, alpha)
    prices_arr = prices.to_numpy(dtype=np.float64)
    index = prices.index
    pivots = _extract_pivots(prices_arr, df["upturn_dc"].to_numpy(), df["downturn_dc"].to_numpy())
    zigzags: list[Zigzag] = []
    for m in range(1, len(pivots)):
        start_bar, start_price, _, _ = pivots[m - 1]
        end_bar, end_price, is_peak, confirm_bar = pivots[m]
        direction = ZigzagDirection.UP if is_peak else ZigzagDirection.DOWN
        magnitude = abs(end_price - start_price) / start_price if start_price != 0 else 0.0
        zigzags.append(Zigzag(seq_index=len(zigzags), direction=direction, start_bar=start_bar, end_bar=end_bar,
                              start_price=start_price, end_price=end_price, magnitude=magnitude,
                              confirm_bar=confirm_bar, start_time=index[start_bar], end_time=index[confirm_bar]))
    return zigzags


def observed_hours(index: pd.DatetimeIndex) -> float:
    """Hours of data the bars cover: number of bars x the typical (median) bar interval.

    Gaps (weekends, removed sessions) add nothing, so a rate per observed hour is a rate per hour of market data.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 0.0
    step = pd.Series(index).diff().median()
    return len(index) * (step / _HOUR) if pd.notna(step) and step > pd.Timedelta(0) else 0.0


def events_per_hour(zigzags: Sequence[Zigzag], exposure_hours: Optional[float] = None) -> float:
    """Leg completion rate.

    With ``exposure_hours`` (e.g. ``observed_hours`` of the window the legs came from): legs / exposure, the rate over
    a defined window. Without it: the interarrival estimate (n_legs - 1) / hours between the first and last
    confirmation (two legs one hour apart give 1 per hour).
    """
    if exposure_hours is not None:
        return len(zigzags) / exposure_hours if exposure_hours > 0 else 0.0
    if len(zigzags) < 2:
        return 0.0
    hours = (zigzags[-1].end_time - zigzags[0].end_time) / _HOUR
    return (len(zigzags) - 1) / hours if hours > 0 else 0.0


def check_legs_align(legs: Sequence[Zigzag], index: pd.Index, who: str) -> None:
    """Raise unless every leg's bar positions point at its own timestamps in ``index`` (positional reads are safe)."""
    if not legs:
        return
    n = len(index)
    starts = np.array([z.start_bar for z in legs])
    ends = np.array([z.end_bar for z in legs])
    confirms = np.array([z.confirm_bar for z in legs])
    if np.any(starts < 0) or np.any(starts > ends) or np.any(ends > confirms):
        raise ValueError(f"{who}: every leg needs 0 <= start_bar <= end_bar <= confirm_bar (a feature must never read "
                         f"past the bar that confirmed the leg)")
    if confirms.max() >= n:
        raise ValueError(f"{who}: legs reach bar {int(confirms.max())} but the market data has {n} bars; build the "
                         f"legs from the same series")
    start_ok = index[starts] == pd.DatetimeIndex([z.start_time for z in legs])
    confirm_ok = index[confirms] == pd.DatetimeIndex([z.end_time for z in legs])
    if not (np.all(start_ok) and np.all(confirm_ok)):
        raise ValueError(f"{who}: the legs' bar positions do not match the market data's timestamps; build the legs "
                         f"from the same series the features read")


def calibrate_k(prices: pd.Series, sigma_realized: pd.Series, target_events_per_hour: tuple[float, float], *,
                alpha: float = 1.0, reducer: str = VolReducer.MEDIAN, k_lo: float = 0.1, k_hi: float = 20.0,
                max_iter: int = 40, tol: float = 0.05) -> float:
    """Bisect ``k`` so the leg rate lands in ``target_events_per_hour`` (legs per observed hour of ``prices``).

    The rate falls as k grows. When the band midpoint is outside what [k_lo, k_hi] can reach, the nearer end is
    returned with a ``RuntimeWarning`` (e.g. a 5-minute series cannot produce several legs per hour above its noise:
    the search stops at k_lo and the legs are bar-to-bar noise). A ``RuntimeWarning`` is also raised when bisection ends
    outside the band.
    """
    lo_target, hi_target = target_events_per_hour
    mid_target = 0.5 * (lo_target + hi_target)
    exposure = observed_hours(prices.index) if isinstance(prices.index, pd.DatetimeIndex) else None

    def rate(k: float) -> float:
        legs = aggregate_zigzags(prices, vol_scaled_threshold(k, sigma_realized, reducer), alpha)
        return events_per_hour(legs, exposure)

    a, b = k_lo, k_hi
    ra, rb = rate(a), rate(b)
    if mid_target >= ra:
        warnings.warn(f"Target {target_events_per_hour} legs/hour is not reachable: the smallest k ({k_lo}) gives "
                      f"{ra:.3f}/hour. Using k={k_lo}; on this data the legs are as small as the search allows.",
                      RuntimeWarning, stacklevel=2)
        return a
    if mid_target <= rb:
        warnings.warn(f"Target {target_events_per_hour} legs/hour is not reachable: the largest k ({k_hi}) gives "
                      f"{rb:.3f}/hour. Using k={k_hi}.", RuntimeWarning, stacklevel=2)
        return b
    best_k, best_gap, best_rate = a, abs(ra - mid_target), ra
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        rm = rate(m)
        gap = abs(rm - mid_target)
        if gap < best_gap:
            best_k, best_gap, best_rate = m, gap, rm
        if lo_target <= rm <= hi_target and gap <= tol * mid_target:
            return m
        if rm > mid_target:
            a = m
        else:
            b = m
    if not lo_target <= best_rate <= hi_target:
        warnings.warn(f"Calibration ended outside the target band {target_events_per_hour}: k={best_k:.4f} gives "
                      f"{best_rate:.3f} legs/hour.", RuntimeWarning, stacklevel=2)
    return best_k


# ----------------------------------------------------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LegThreshold:
    """How big a reversal must be to complete a leg (the leg size).

    Parameters
    ----------
    mode : ``ThresholdMode``.
    value : TARGET_RATE: legs per observed hour, a float (band +/-20%) or a (low, high) band; K_SIGMA: k; FIXED: theta
        as a fraction of price.
    vol_window : realised-vol lookback for TARGET_RATE / K_SIGMA (a ``pd.Timedelta`` or a bar count).
    reducer : how the training window's per-bar volatility is reduced to one value.
    """

    mode: ThresholdMode
    value: Union[float, tuple[float, float]]
    vol_window: Union[pd.Timedelta, int] = pd.Timedelta("24h")
    reducer: VolReducer = VolReducer.MEDIAN

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ThresholdMode(self.mode))
        object.__setattr__(self, "reducer", VolReducer(self.reducer))
        if self.mode is ThresholdMode.TARGET_RATE:
            band = self.band
            if not 0 < band[0] <= band[1]:
                raise ValueError(f"TARGET_RATE needs 0 < low <= high legs/hour, got {self.value}")
        elif not np.isscalar(self.value) or float(self.value) <= 0:
            raise ValueError(f"{self.mode.name} needs a positive scalar value, got {self.value}")

    @property
    def band(self) -> tuple[float, float]:
        if np.isscalar(self.value):
            v = float(self.value)
            return 0.8 * v, 1.2 * v
        lo, hi = self.value
        return float(lo), float(hi)

    @classmethod
    def legs_per_hour(cls, rate: Union[float, tuple[float, float]],
                      vol_window: Union[pd.Timedelta, int] = pd.Timedelta("24h")) -> "LegThreshold":
        return cls(ThresholdMode.TARGET_RATE, rate, vol_window)

    @classmethod
    def k_sigma(cls, k: float, vol_window: Union[pd.Timedelta, int] = pd.Timedelta("24h")) -> "LegThreshold":
        return cls(ThresholdMode.K_SIGMA, k, vol_window)

    @classmethod
    def fixed(cls, theta: float) -> "LegThreshold":
        return cls(ThresholdMode.FIXED, theta)


class ZigzagLegBuilder:
    """Fit/transform producer of zigzag legs with the threshold frozen at fit time.

    ``fit`` resolves the threshold on the training prices (calibrating k for TARGET_RATE); ``transform`` reuses the
    frozen ``theta_``, so a later block never reads its own volatility. ``rate_`` is the training window's legs per
    observed hour (bars x bar interval; gaps such as weekends do not count).
    """

    def __init__(self, threshold: LegThreshold, alpha: float = 1.0):
        if not isinstance(threshold, LegThreshold):
            raise TypeError("threshold must be a LegThreshold")
        self.threshold = threshold
        self.alpha = float(alpha)
        self.k_: Optional[float] = None
        self.theta_: Optional[float] = None
        self.rate_: Optional[float] = None

    def fit(self, prices: pd.Series) -> "ZigzagLegBuilder":
        """Resolve and freeze the threshold. Atomic: a failed fit leaves the previous threshold in place."""
        th = self.threshold
        if th.mode is ThresholdMode.FIXED:
            k, theta = None, float(th.value)
        else:
            sigma = realized_vol(prices, th.vol_window)
            if th.mode is ThresholdMode.TARGET_RATE:
                k = calibrate_k(prices, sigma, th.band, alpha=self.alpha, reducer=th.reducer)
            else:
                k = float(th.value)
            theta = vol_scaled_threshold(k, sigma, th.reducer)
        rate = events_per_hour(aggregate_zigzags(prices, theta, self.alpha), observed_hours(prices.index))
        self.k_, self.theta_, self.rate_ = k, theta, rate
        return self

    def transform(self, prices: pd.Series) -> list[Zigzag]:
        if self.theta_ is None:
            raise RuntimeError("ZigzagLegBuilder is not fitted; call fit() first")
        return aggregate_zigzags(prices, self.theta_, self.alpha)

    def fit_transform(self, prices: pd.Series) -> list[Zigzag]:
        return self.fit(prices).transform(prices)

    def __repr__(self) -> str:
        theta = "unfitted" if self.theta_ is None else f"{self.theta_ * 1e4:.2f}bp"
        return f"ZigzagLegBuilder({self.threshold.mode.value}={self.threshold.value}, theta={theta})"


# ----------------------------------------------------------------------------------------------------------------------
# Alignment helpers
# ----------------------------------------------------------------------------------------------------------------------
def legs_confirmed_from(legs: Sequence[Zigzag], start: pd.Timestamp) -> list[Zigzag]:
    """Legs confirmed at or after ``start`` (e.g. a test window's first bar), keeping their full history upstream."""
    return [z for z in legs if z.end_time >= start]


def legs_to_bars(values: np.ndarray, legs: Sequence[Zigzag], n_bars: int, delay: int = 0) -> np.ndarray:
    """Spread per-leg values onto bars: each leg's value holds from its confirm bar (+ ``delay``) until the next.

    ``values`` is (n_legs,) or (n_legs, k). Bars before the first leg are NaN. ``delay=1`` makes the value usable from
    the bar after confirmation (trade at the next bar's open).
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(legs):
        raise ValueError(f"values has {len(values)} rows for {len(legs)} legs")
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}")
    out = np.full((n_bars,) + values.shape[1:], np.nan)
    positions = np.array([z.confirm_bar + delay for z in legs], dtype=np.int64)
    for k, pos in enumerate(positions):
        if pos >= n_bars:
            break
        end = positions[k + 1] if k + 1 < len(positions) else n_bars
        out[pos:min(end, n_bars)] = values[k]
    return out
