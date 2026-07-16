"""
Observation pipeline for the Hierarchical HMM.

Turns a raw L1 price stream into a reproducible sequence of 18-symbol discrete zigzag
observations (plan "Week 1 — Observation pipeline"). The event-time aggregation is the
single highest-leverage design choice: directional-change zigzags strip most intraday
microstructure noise *before* the model sees it.

Pipeline
--------
1. ``realized_vol``          — slow rolling realised-vol estimate ``sigma_realized``.
2. ``vol_scaled_threshold``  — DC threshold ``theta = k * sigma_realized`` (representative scalar).
3. ``calibrate_k``           — search ``k`` to hit the group's target events/hour band.
4. ``aggregate_zigzags``     — wrap ``idc_parse`` into a pivot/leg sequence (alternating up/down).
5. ``FlowFeature``           — per-zigzag flow-support raw score (BOCPD or signed-volume).
6. discretisation            — bucket magnitude/flow into 3 bins via training-window quantiles.
7. ``ZigzagObservationPipeline`` — fit/transform that freezes ``k`` and bucket boundaries so
   live inference reuses the training-window calibration.

Threshold note
--------------
``idc_parse`` takes a scalar ``theta``. Within one fit window realised vol is treated as slowly
varying, so the DC threshold is a representative scalar ``k * reduce(sigma_realized)`` and ``k`` is
calibrated to the target event rate. The *time-local* ``sigma_realized`` is still used where it
matters most — the trend-strength normalisation ``|magnitude| / sigma_realized`` at each zigzag.
A future upgrade to a true time-varying DC threshold would not change this module's public API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse
from okmich_quant_features.microstructure.order_flow import vir

from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector, NormalInverseGammaModel

from .config import (
    AssetGroupConfig,
    FlowBucket,
    FlowFeatureKind,
    N_FLOW_BINS,
    N_STRENGTH_BINS,
    TrendStrength,
    ZigzagDirection,
    encode_symbol,
)

_HOUR = pd.Timedelta("1h")


# ----------------------------------------------------------------------------------------
# Realised vol and threshold
# ----------------------------------------------------------------------------------------
def realized_vol(prices: pd.Series, window: pd.Timedelta | int, min_periods: int = 2) -> pd.Series:
    """
    Slow rolling realised-vol estimate: rolling std of log returns.

    Parameters
    ----------
    prices
        Close price series. A ``pd.Timedelta`` ``window`` requires a ``DatetimeIndex``.
    window
        Lookback as a time span (``pd.Timedelta``, needs a ``DatetimeIndex``) or a bar count.
    min_periods
        Minimum observations in the window before a value is emitted.

    Returns
    -------
    pd.Series
        Per-bar realised vol aligned to ``prices``, **forward-filled only**. Bars before the first
        valid rolling estimate stay ``NaN`` — they are never back-filled from future returns, so the
        estimate at every bar depends only on past data (causal). Callers should supply a warmup
        prefix of at least one ``window`` before the analysis region; consumers here already tolerate
        the leading ``NaN`` (``vol_scaled_threshold`` drops non-finite values; strength normalisation
        treats a non-positive/NaN sigma as a 0.0 warmup default).
    """
    if isinstance(window, pd.Timedelta) and not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("A pd.Timedelta realized_vol window requires prices to have a DatetimeIndex.")
    log_ret = np.log(prices).diff()
    sigma = log_ret.rolling(window, min_periods=min_periods).std()
    return sigma.ffill()  # forward-only: no look-ahead. Leading NaN preserved by design.


def vol_scaled_threshold(k: float, sigma_realized: pd.Series | np.ndarray, reducer: str = "median") -> float:
    """
    Representative DC threshold ``theta = k * reduce(sigma_realized)`` for a fit window.

    ``reducer`` is one of ``{"median", "mean"}``. Raises if the reduced vol is not positive.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    values = np.asarray(sigma_realized, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("sigma_realized has no finite values")
    if reducer == "median":
        sigma_ref = float(np.median(values))
    elif reducer == "mean":
        sigma_ref = float(np.mean(values))
    else:
        raise ValueError(f"reducer must be 'median' or 'mean', got {reducer!r}")
    if sigma_ref <= 0:
        raise ValueError(f"reduced sigma_realized must be positive, got {sigma_ref}")
    return k * sigma_ref


# ----------------------------------------------------------------------------------------
# Market data + zigzag containers
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketData:
    """L1 market data backing flow-feature computation. Only ``close`` is always required."""

    close: pd.Series
    high: Optional[pd.Series] = None
    low: Optional[pd.Series] = None
    volume: Optional[pd.Series] = None

    @classmethod
    def from_ohlcv(cls, df: pd.DataFrame, *, close: str = "close", high: str = "high", low: str = "low",
                   volume: str = "volume") -> "MarketData":
        """Build from an OHLCV frame. Missing high/low/volume columns are left as None."""
        return cls(
            close=df[close],
            high=df[high] if high in df.columns else None,
            low=df[low] if low in df.columns else None,
            volume=df[volume] if volume in df.columns else None,
        )

    def require_volume(self, who: str) -> None:
        """Raise a clear error if high/low/close/volume are not all present."""
        missing = [n for n, s in (("high", self.high), ("low", self.low), ("volume", self.volume)) if s is None]
        if missing:
            raise ValueError(f"{who} requires high/low/close/volume; missing: {missing}")


@dataclass(frozen=True)
class Zigzag:
    """
    One completed directional-change leg (pivot-to-pivot).

    A leg is *observable* (completed) at ``confirm_bar`` — the DC bar that confirmed the reversal
    at ``end_bar`` — so ``end_time`` is causal: the leg's amplitude is known only then.
    """

    seq_index: int
    direction: ZigzagDirection
    start_bar: int      # position of the leg's opening pivot
    end_bar: int        # position of the leg's closing pivot (the confirmed extremum)
    start_price: float
    end_price: float
    magnitude: float    # |end_price - start_price| / start_price
    confirm_bar: int    # position of the DC confirmation that completed this leg
    start_time: pd.Timestamp
    end_time: pd.Timestamp  # clock time the leg completed (== time at confirm_bar)


# ----------------------------------------------------------------------------------------
# Zigzag aggregation
# ----------------------------------------------------------------------------------------
def _extract_pivots(prices: np.ndarray, upturn_dc: np.ndarray, downturn_dc: np.ndarray) -> list[tuple[int, float, bool, int]]:
    """
    Reconstruct confirmed zigzag pivots from ``idc_parse`` DC-confirmation flags.

    Returns a list of ``(pivot_bar, pivot_price, is_peak, confirm_bar)`` in time order. Peaks and
    troughs strictly alternate. ``idc_parse`` does not emit the extremum's bar, so the running
    extremum is tracked here; ``confirm_bar`` is the DC bar that finalised that pivot.
    """
    n = len(prices)
    pivots: list[tuple[int, float, bool, int]] = []
    started = False
    mode = 0                 # +1 uptrend (tracking a peak), -1 downtrend (tracking a trough)
    ext_bar, ext_price = 0, prices[0]

    for i in range(n):
        if not started:
            if upturn_dc[i]:
                j = int(np.argmin(prices[: i + 1]))
                pivots.append((j, float(prices[j]), False, i))  # confirmed trough
                mode, started = 1, True
                ext_bar, ext_price = i, float(prices[i])
            elif downturn_dc[i]:
                j = int(np.argmax(prices[: i + 1]))
                pivots.append((j, float(prices[j]), True, i))   # confirmed peak
                mode, started = -1, True
                ext_bar, ext_price = i, float(prices[i])
            continue

        if mode == 1:
            if prices[i] > ext_price:
                ext_price, ext_bar = float(prices[i]), i
            if downturn_dc[i]:
                pivots.append((ext_bar, ext_price, True, i))    # peak confirmed
                mode = -1
                ext_price, ext_bar = float(prices[i]), i
        else:  # mode == -1
            if prices[i] < ext_price:
                ext_price, ext_bar = float(prices[i]), i
            if upturn_dc[i]:
                pivots.append((ext_bar, ext_price, False, i))   # trough confirmed
                mode = 1
                ext_price, ext_bar = float(prices[i]), i

    return pivots


def aggregate_zigzags(prices: pd.Series, theta: float, alpha: float = 1.0) -> list[Zigzag]:
    """
    Aggregate a price series into completed directional-change zigzag legs.

    Wraps ``idc_parse`` and pairs consecutive confirmed pivots into legs. A leg whose closing
    pivot is a peak is an *up* zigzag (trough -> peak); a trough-closing leg is a *down* zigzag.
    The returned legs strictly alternate direction.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    df = idc_parse(prices, theta, alpha)
    prices_arr = prices.to_numpy(dtype=np.float64)
    index = prices.index
    upturn = df["upturn_dc"].to_numpy()
    downturn = df["downturn_dc"].to_numpy()

    pivots = _extract_pivots(prices_arr, upturn, downturn)

    zigzags: list[Zigzag] = []
    for m in range(1, len(pivots)):
        start_bar, start_price, _, _ = pivots[m - 1]
        end_bar, end_price, is_peak, confirm_bar = pivots[m]
        direction = ZigzagDirection.UP if is_peak else ZigzagDirection.DOWN
        magnitude = abs(end_price - start_price) / start_price if start_price != 0 else 0.0
        zigzags.append(
            Zigzag(
                seq_index=len(zigzags),
                direction=direction,
                start_bar=start_bar,
                end_bar=end_bar,
                start_price=start_price,
                end_price=end_price,
                magnitude=magnitude,
                confirm_bar=confirm_bar,
                start_time=index[start_bar],
                end_time=index[confirm_bar],
            )
        )
    return zigzags


def events_per_hour(zigzags: Sequence[Zigzag]) -> float:
    """Zigzag completions per hour over the span they cover. Needs >= 2 zigzags with times."""
    if len(zigzags) < 2:
        return 0.0
    t0 = zigzags[0].end_time
    t1 = zigzags[-1].end_time
    hours = (t1 - t0) / _HOUR
    if hours <= 0:
        return 0.0
    return len(zigzags) / hours


def calibrate_k(prices: pd.Series, sigma_realized: pd.Series, target_events_per_hour: tuple[float, float], *,
                alpha: float = 1.0, reducer: str = "median", k_lo: float = 0.1, k_hi: float = 20.0,
                max_iter: int = 40, tol: float = 0.05) -> float:
    """
    Search ``k`` so the zigzag event rate lands in ``target_events_per_hour``.

    Event rate is monotonically decreasing in ``k`` (a bigger threshold yields fewer zigzags), so a
    bisection converges. Returns the ``k`` whose rate is closest to the band midpoint. Falls back to
    the nearest bracket endpoint when the target is outside the achievable range on this data.
    """
    lo_target, hi_target = target_events_per_hour
    mid_target = 0.5 * (lo_target + hi_target)

    def rate(k: float) -> float:
        theta = vol_scaled_threshold(k, sigma_realized, reducer)
        return events_per_hour(aggregate_zigzags(prices, theta, alpha))

    a, b = k_lo, k_hi
    ra, rb = rate(a), rate(b)
    # Rate decreases with k: ra (small k) high, rb (large k) low. If the target midpoint is
    # outside [rb, ra], return the closest endpoint.
    if mid_target >= ra:
        return a
    if mid_target <= rb:
        return b

    best_k, best_gap = a, abs(ra - mid_target)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        rm = rate(m)
        gap = abs(rm - mid_target)
        if gap < best_gap:
            best_k, best_gap = m, gap
        if lo_target <= rm <= hi_target and gap <= tol * mid_target:
            return m
        if rm > mid_target:   # too many events -> raise k
            a = m
        else:                 # too few events -> lower k
            b = m
    return best_k


# ----------------------------------------------------------------------------------------
# Flow features
# ----------------------------------------------------------------------------------------
class FlowFeature(Protocol):
    """
    Pluggable flow-support component.

    Returns one raw score per zigzag where a *higher* score means flow more strongly *with* the
    price direction (bucketed later into AGAINST / NEUTRAL / WITH via training-window quantiles).
    """

    kind: FlowFeatureKind

    def raw_scores(self, zigzags: Sequence[Zigzag], market_data: MarketData) -> np.ndarray: ...


@dataclass
class BOCPDFlowFeature:
    """
    Default flow feature: accumulated BOCPD changepoint posterior over each zigzag.

    Low accumulated changepoint probability -> no structural break (mean-revert-like) -> AGAINST;
    high -> structural break (trend-like) -> WITH. Universal — same construction on every asset.
    """

    hazard_rate: float = 1.0 / 250.0
    r_max: int = 200
    reducer: str = "mean"  # how to accumulate per-bar changepoint prob over a leg
    kind: FlowFeatureKind = field(default=FlowFeatureKind.BOCPD, init=False)

    def _changepoint_prob(self, close: pd.Series) -> np.ndarray:
        """Per-bar changepoint probability P(run-length = 0), aligned to ``close`` (bar 0 = 0)."""
        log_ret = np.log(close.to_numpy(dtype=np.float64))
        log_ret = np.diff(log_ret)
        if log_ret.size == 0:
            return np.zeros(len(close), dtype=np.float64)
        detector = BayesianOnlineChangepointDetector(
            NormalInverseGammaModel(), hazard_rate=self.hazard_rate, r_max=self.r_max
        )
        posteriors = detector.batch(log_ret)     # (T-1, r_max)
        cp = posteriors[:, 0]
        return np.concatenate([[0.0], cp])       # prepend for the return-less first bar

    def raw_scores(self, zigzags: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        cp = self._changepoint_prob(market_data.close)
        reduce = np.mean if self.reducer == "mean" else np.max
        scores = np.empty(len(zigzags), dtype=np.float64)
        for i, z in enumerate(zigzags):
            span = cp[z.start_bar: z.end_bar + 1]
            scores[i] = float(reduce(span)) if span.size else 0.0
        return scores


@dataclass
class SignedVolumeFlowFeature:
    """
    Opt-in flow feature: β-CLV signed volume accumulated over the zigzag, signed against direction.

    Net signed pressure ``sum(vir * volume) / sum(volume)`` in ``[-1, 1]`` is multiplied by the
    zigzag's direction sign, so a positive score means volume flowed *with* the price move.
    """

    kind: FlowFeatureKind = field(default=FlowFeatureKind.SIGNED_VOLUME, init=False)

    def raw_scores(self, zigzags: Sequence[Zigzag], market_data: MarketData) -> np.ndarray:
        market_data.require_volume("SignedVolumeFlowFeature")
        pressure = np.asarray(
            vir(market_data.high, market_data.low, market_data.close, market_data.volume), dtype=np.float64
        )
        vol = market_data.volume.to_numpy(dtype=np.float64)
        signed_delta = np.nan_to_num(pressure) * np.nan_to_num(vol)
        scores = np.empty(len(zigzags), dtype=np.float64)
        for i, z in enumerate(zigzags):
            v = vol[z.start_bar: z.end_bar + 1]
            d = signed_delta[z.start_bar: z.end_bar + 1]
            total = float(np.nansum(v))
            net = float(np.nansum(d)) / total if total > 0 else 0.0
            dir_sign = 1.0 if z.direction is ZigzagDirection.UP else -1.0
            scores[i] = net * dir_sign
        return scores


def build_flow_feature(kind: FlowFeatureKind) -> FlowFeature:
    """Construct the default flow feature for a ``FlowFeatureKind``."""
    kind = FlowFeatureKind(kind)
    if kind is FlowFeatureKind.BOCPD:
        return BOCPDFlowFeature()
    return SignedVolumeFlowFeature()


# ----------------------------------------------------------------------------------------
# Discretisation
# ----------------------------------------------------------------------------------------
def quantile_boundaries(scores: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Interior quantile cut points for ``n_bins`` equal-mass buckets (``n_bins - 1`` boundaries).

    Degenerate inputs (all-equal, too few points) yield strictly increasing synthetic boundaries so
    :func:`bucketize` still returns a valid, monotonic assignment (everything lands in one bin).
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    if finite.size == 0:
        return np.arange(1, n_bins, dtype=np.float64)
    bounds = np.quantile(finite, qs)
    # Enforce strict monotonicity for np.digitize (ties collapse buckets otherwise).
    for j in range(1, len(bounds)):
        if bounds[j] <= bounds[j - 1]:
            bounds[j] = np.nextafter(bounds[j - 1], np.inf)
    return bounds


def bucketize(scores: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Assign each score to a bucket index in ``[0, len(boundaries)]`` (higher score -> higher bin)."""
    return np.digitize(np.asarray(scores, dtype=np.float64), np.asarray(boundaries, dtype=np.float64)).astype(np.int64)


# ----------------------------------------------------------------------------------------
# Observation container + pipeline
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ZigzagObservations:
    """The discretised 18-symbol observation stream and its component decomposition."""

    zigzags: list[Zigzag]
    symbols: np.ndarray       # (T,) int in [0, 18)
    directions: np.ndarray    # (T,) int ZigzagDirection
    strengths: np.ndarray     # (T,) int TrendStrength
    flows: np.ndarray         # (T,) int FlowBucket
    magnitudes: np.ndarray    # (T,) float
    event_times: np.ndarray   # (T,) datetime64 clock time each zigzag completed

    @property
    def n_zigzags(self) -> int:
        return len(self.zigzags)

    def to_model_input(self) -> np.ndarray:
        """Symbols as a ``(T, 1)`` integer matrix for the categorical HHMM."""
        return self.symbols.reshape(-1, 1).astype(np.int64)


class ZigzagObservationPipeline:
    """
    Fit/transform producer of reproducible 18-symbol zigzag streams.

    ``fit`` calibrates ``k`` and freezes the trend-strength and flow bucket boundaries on the
    training window; ``transform`` reuses that frozen calibration so live/OOS streams are encoded
    against the same thresholds. The DC threshold is re-derived per call as ``k * reduce(sigma)`` so
    the event rate stays stable as realised vol drifts.

    Causal-use caveat
    -----------------
    The DC threshold is a single scalar per ``transform`` call (``k * reduce(sigma_over_the_call)``),
    so transforming one long block folds that block's *aggregate* vol into the segmentation of its
    early bars — a mild look-ahead. Strength normalisation is time-local (causal), and bucket
    boundaries are frozen from ``fit`` (no leakage), but for a strictly causal live/backtest stream
    call ``transform`` on trailing windows (walk-forward), not once over a block that contains its
    own future. This is the documented scalar-theta approximation, not a per-bar time-varying DC.
    """

    def __init__(self, config: AssetGroupConfig, *, flow_feature: Optional[FlowFeature] = None,
                 alpha: float = 1.0, vol_reducer: str = "median"):
        self.config = config
        self.flow_feature = flow_feature if flow_feature is not None else build_flow_feature(config.flow_feature)
        self.alpha = alpha
        self.vol_reducer = vol_reducer
        self.k_: Optional[float] = None
        self.theta_: Optional[float] = None
        self.strength_boundaries_: Optional[np.ndarray] = None
        self.flow_boundaries_: Optional[np.ndarray] = None

    # -- helpers --------------------------------------------------------------------------
    def _sigma(self, prices: pd.Series) -> pd.Series:
        return realized_vol(prices, self.config.realized_vol_window)

    def _strength_scores(self, zigzags: Sequence[Zigzag], sigma: pd.Series) -> np.ndarray:
        sig = sigma.to_numpy(dtype=np.float64)
        out = np.empty(len(zigzags), dtype=np.float64)
        for i, z in enumerate(zigzags):
            s = sig[z.end_bar]
            out[i] = z.magnitude / s if s > 0 else 0.0
        return out

    def _encode(self, zigzags: Sequence[Zigzag], strengths: np.ndarray, flows: np.ndarray) -> np.ndarray:
        symbols = np.empty(len(zigzags), dtype=np.int64)
        for i, z in enumerate(zigzags):
            symbols[i] = encode_symbol(z.direction, TrendStrength(int(strengths[i])), FlowBucket(int(flows[i])))
        return symbols

    def _market_data(self, prices: pd.Series, market_data: Optional[MarketData]) -> MarketData:
        return market_data if market_data is not None else MarketData(close=prices)

    def _aggregate_and_score(self, prices: pd.Series, md: MarketData, theta: float) -> tuple:
        """Aggregate zigzags at ``theta`` and compute raw strength + flow scores. Returns the trio."""
        sigma = self._sigma(prices)
        zigzags = aggregate_zigzags(prices, theta, self.alpha)
        strength_scores = self._strength_scores(zigzags, sigma)
        flow_scores = self.flow_feature.raw_scores(zigzags, md)
        return zigzags, strength_scores, flow_scores

    def _build_observations(self, zigzags: Sequence[Zigzag], strength_scores: np.ndarray,
                            flow_scores: np.ndarray) -> ZigzagObservations:
        """Bucketize (with the frozen boundaries), encode symbols and assemble ZigzagObservations."""
        strengths = bucketize(strength_scores, self.strength_boundaries_)
        flows = bucketize(flow_scores, self.flow_boundaries_)
        symbols = self._encode(zigzags, strengths, flows)
        directions = np.array([int(z.direction) for z in zigzags], dtype=np.int64)
        magnitudes = np.array([z.magnitude for z in zigzags], dtype=np.float64)
        # UTC-naive datetime64 so session-policy time-of-day logic is unambiguous (and no per-element
        # tz-drop warning). Windows in the SessionPolicy gate are defined in UTC/GMT.
        end_times = pd.DatetimeIndex([z.end_time for z in zigzags])
        if end_times.tz is not None:
            end_times = end_times.tz_convert("UTC").tz_localize(None)
        event_times = end_times.values.astype("datetime64[ns]")
        return ZigzagObservations(
            zigzags=list(zigzags),
            symbols=symbols,
            directions=directions,
            strengths=strengths.astype(np.int64),
            flows=flows.astype(np.int64),
            magnitudes=magnitudes,
            event_times=event_times,
        )

    # -- public API -----------------------------------------------------------------------
    def _fit_collect(self, prices: pd.Series, market_data: Optional[MarketData]) -> tuple:
        """Calibrate ``k``, freeze bucket boundaries, and return the fit-window (zigzags, scores)."""
        md = self._market_data(prices, market_data)
        sigma = self._sigma(prices)
        self.k_ = calibrate_k(prices, sigma, self.config.target_events_per_hour, alpha=self.alpha, reducer=self.vol_reducer)
        self.theta_ = vol_scaled_threshold(self.k_, sigma, self.vol_reducer)
        zigzags, strength_scores, flow_scores = self._aggregate_and_score(prices, md, self.theta_)
        if len(zigzags) < N_STRENGTH_BINS:
            raise ValueError(
                f"Only {len(zigzags)} zigzags produced on the fit window; need at least {N_STRENGTH_BINS} "
                "to fit strength/flow buckets. Provide more data or widen the target event rate."
            )
        self.strength_boundaries_ = quantile_boundaries(strength_scores, N_STRENGTH_BINS)
        self.flow_boundaries_ = quantile_boundaries(flow_scores, N_FLOW_BINS)
        return zigzags, strength_scores, flow_scores

    def fit(self, prices: pd.Series, market_data: Optional[MarketData] = None) -> "ZigzagObservationPipeline":
        """Calibrate ``k`` and freeze bucket boundaries on the training window."""
        self._fit_collect(prices, market_data)
        return self

    def transform(self, prices: pd.Series, market_data: Optional[MarketData] = None) -> ZigzagObservations:
        """Encode a price stream with the frozen ``k`` and bucket boundaries."""
        if self.k_ is None or self.strength_boundaries_ is None or self.flow_boundaries_ is None:
            raise RuntimeError("Pipeline has not been fitted. Call fit() before transform().")
        md = self._market_data(prices, market_data)
        theta = vol_scaled_threshold(self.k_, self._sigma(prices), self.vol_reducer)
        zigzags, strength_scores, flow_scores = self._aggregate_and_score(prices, md, theta)
        return self._build_observations(zigzags, strength_scores, flow_scores)

    def fit_transform(self, prices: pd.Series, market_data: Optional[MarketData] = None) -> ZigzagObservations:
        """
        Fit on ``prices`` then encode the same series, computing zigzags and flow scores only once
        (``fit`` followed by ``transform`` would recompute the BOCPD/flow pass on the same data).
        """
        zigzags, strength_scores, flow_scores = self._fit_collect(prices, market_data)
        return self._build_observations(zigzags, strength_scores, flow_scores)
