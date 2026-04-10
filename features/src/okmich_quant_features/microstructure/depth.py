"""
Order Book Depth Proxy Features for Market Microstructure Analysis.

These features estimate hidden depth and absorption capacity from OHLCV data, without requiring Level 2 order book data.
They proxy:
  - Iceberg / hidden order detection (Bar Absorption Ratio)
  - Price rejection and wick pressure (Wick Imbalance)
  - Range-normalized depth (Range-Volume Depth)
  - Directional absorption score (Absorption-Weighted Depth Score)

Module Structure
----------------
bar_absorption_ratio         : V / |C-O| — iceberg order detector
wick_imbalance               : (lower_wick - upper_wick) / range — rejection proxy
range_volume_depth           : V / (H-L) with rolling z-score — inverse price impact
absorption_weighted_depth_score : sign(C-O) × AR / EMA(AR) — directional depth
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Union
from ._primitives import _check_index_aligned

ArrayLike = Union[pd.Series, np.ndarray]


@njit(cache=True)
def _bar_absorption_ratio_kernel(volume: np.ndarray, open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(volume)
    ar = np.empty(n, dtype=np.float64)

    for i in range(n):
        body = abs(close[i] - open_[i])
        if body > 1e-10:
            ar[i] = volume[i] / body
        else:
            ar[i] = np.inf

    return ar


def bar_absorption_ratio(open_: ArrayLike, close: ArrayLike, volume: ArrayLike) -> ArrayLike:
    """
    Bar Absorption Ratio (AR).

    Estimates hidden depth by measuring how much volume was traded per unit
    of price movement. A spike in AR signals that a large passive order absorbed
    the flow (iceberg / institutional resting order).

    Formula:
        AR(t) = V(t) / |C(t) - O(t)|

    For doji bars (|C-O| < ε), AR is clipped to the 99th percentile of finite values
    (maximum absorption — all volume, zero price movement).

    Parameters
    ----------
    open_, close, volume : ArrayLike
        OHLCV data (1-D float64 arrays or pd.Series, same length)

    Returns
    -------
    ArrayLike
        Bar absorption ratio. Same type as `open_`.
        - High AR → large volume, small body (hidden depth / iceberg)
        - Low AR  → small volume, large body (thin order book)
        - Capped  → doji bar (|C-O| ≈ 0), clipped to p99 of finite AR

    Interpretation
    --------------
    - AR spike > 2× rolling mean → potential iceberg detected
    - Cluster of high-AR bars at price level → strong support/resistance
    - Use `absorption_weighted_depth_score` for directional version
    """
    _check_index_aligned(open_, close, volume)

    is_series = isinstance(open_, pd.Series)
    index = open_.index if is_series else None

    o = np.asarray(open_.values if is_series else open_, dtype=np.float64)
    c = np.asarray(close.values if isinstance(close, pd.Series) else close, dtype=np.float64)
    v = np.asarray(volume.values if isinstance(volume, pd.Series) else volume, dtype=np.float64)

    ar = _bar_absorption_ratio_kernel(v, o, c)
    # Clip doji bars (Inf) to 99th percentile of finite values.
    # Doji = maximum absorption (all volume, zero price movement) — not missing data.
    inf_mask = ~np.isfinite(ar)
    if inf_mask.any():
        finite_vals = ar[~inf_mask]
        cap = np.percentile(finite_vals, 99) if len(finite_vals) > 0 else v.max()
        ar[inf_mask] = cap

    if is_series:
        return pd.Series(ar, index=index, name='bar_absorption_ratio')
    return ar


@njit(cache=True)
def _wick_imbalance_kernel(high: np.ndarray, low: np.ndarray, open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    wi = np.empty(n, dtype=np.float64)

    for i in range(n):
        body_high = close[i] if close[i] > open_[i] else open_[i]
        body_low = close[i] if close[i] < open_[i] else open_[i]

        upper_wick = high[i] - body_high
        lower_wick = body_low - low[i]

        rng = high[i] - low[i]
        if rng > 1e-10:
            wi[i] = (lower_wick - upper_wick) / rng
        else:
            wi[i] = 0.0  # Doji: no meaningful wick
    return wi


def wick_imbalance(high: ArrayLike, low: ArrayLike, open_: ArrayLike, close: ArrayLike) -> ArrayLike:
    """
    Wick Imbalance (WI). It measures the relative dominance of lower vs upper wicks within each bar.
    A long lower wick with a small body = buyers rejected the low (support).
    A long upper wick = sellers rejected the high (resistance).

    Formula:
        LowerWick = min(O, C) - L
        UpperWick = H - max(O, C)
        WI = (LowerWick - UpperWick) / (H - L)

    Parameters
    ----------
    high, low, open_, close : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series, same length)

    Returns
    -------
    ArrayLike
        WI in range [-1, +1]. Same type as `high`.
        - WI > 0  : Lower wick dominates → rejection of lows (bullish pressure)
        - WI < 0  : Upper wick dominates → rejection of highs (bearish pressure)
        - |WI| > 0.5 : Strong rejection signal
        - WI ≈ 0  : Balanced wicks (no clear rejection; consolidation)

    Interpretation
    --------------
    - High WI at key support level → buying interest (accumulation signal)
    - Low WI at key resistance → selling interest (distribution signal)
    - Combine with volume for confirmation: high WI + high volume = strong signal
    """
    is_series = isinstance(high, pd.Series)
    index = high.index if is_series else None

    h = np.asarray(high.values if is_series else high, dtype=np.float64)
    l = np.asarray(low.values if isinstance(low, pd.Series) else low, dtype=np.float64)
    o = np.asarray(open_.values if isinstance(open_, pd.Series) else open_, dtype=np.float64)
    c = np.asarray(close.values if isinstance(close, pd.Series) else close, dtype=np.float64)
    wi = _wick_imbalance_kernel(h, l, o, c)
    if is_series:
        return pd.Series(wi, index=index, name='wick_imbalance')
    return wi


@njit(cache=True)
def _rvd_kernel(volume: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    n = len(volume)
    rvd = np.empty(n, dtype=np.float64)

    for i in range(n):
        rng = high[i] - low[i]
        if rng > 1e-10:
            rvd[i] = volume[i] / rng
        else:
            rvd[i] = np.inf
    return rvd


def range_volume_depth(high: ArrayLike, low: ArrayLike, volume: ArrayLike, window: int = 20) -> pd.DataFrame:
    """
    Range-Volume Depth (RVD) with rolling z-score. it estimates the amount of volume required to move price across the bar's range.
    High RVD = deep market (many participants, price resistant to moves).
    Low RVD = thin market (few participants, easy to move price).

    Formula:
        RVD(t) = V(t) / (H(t) - L(t))
        RVD_z(t) = (RVD(t) - μ_RVD(t,N)) / σ_RVD(t,N)

    Parameters
    ----------
    high, low, volume : ArrayLike
        OHLCV data (1-D float64 arrays or pd.Series, same length)
    window : int, default=20
        Rolling window for z-score computation.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'rvd'   : Raw ratio (volume / range)
        - 'rvd_z' : Rolling z-score (anomaly detection)

    Interpretation
    --------------
    - High RVD_z (> +2) : Extreme depth — potential exhaustion or breakout failure
    - Low RVD_z (< -2)  : Thin depth — vulnerable to price moves, breakout risk
    - Rising RVD trend  : Market deepening (institutional accumulation)
    - Falling RVD trend : Market thinning (participants stepping away)
    """
    is_series = isinstance(high, pd.Series)
    index = high.index if is_series else None

    h = np.asarray(high.values if is_series else high, dtype=np.float64)
    l = np.asarray(low.values if isinstance(low, pd.Series) else low, dtype=np.float64)
    v = np.asarray(volume.values if isinstance(volume, pd.Series) else volume, dtype=np.float64)

    rvd_raw = _rvd_kernel(v, h, l)
    rvd_raw[~np.isfinite(rvd_raw)] = np.nan

    if index is None:
        index = pd.RangeIndex(len(rvd_raw))

    rvd_s = pd.Series(rvd_raw, index=index)
    rvd_filled = rvd_s.ffill()  # doji bars carry forward previous depth for z-score stability
    roll_mean = rvd_filled.rolling(window).mean()
    roll_std = rvd_filled.rolling(window).std()
    rvd_z = (rvd_filled - roll_mean) / (roll_std + 1e-10)
    return pd.DataFrame({'rvd': rvd_s, 'rvd_z': rvd_z}, index=index)


def absorption_weighted_depth_score(open_: ArrayLike, close: ArrayLike, volume: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Absorption-Weighted Depth Score (DS). It combines bar direction with absorption ratio to produce a signed depth score.
    Positive = above-average absorption on the buy side.
    Negative = above-average absorption on the sell side.

    Formula:
        AR(t) = V(t) / |C(t) - O(t)|
        EMA_AR(t) = EMA(AR, span=window)
        DS(t) = sign(C(t) - O(t)) × (AR(t) / EMA_AR(t))

    Parameters
    ----------
    open_, close, volume : ArrayLike
        OHLCV data (1-D float64 arrays or pd.Series, same length)
    window : int, default=20
        EMA span for baseline absorption normalization.

    Returns
    -------
    ArrayLike
        Signed depth score. Same type as `open_`.
        - DS > +1 : Above-average buy-side absorption (bullish hidden depth)
        - DS < -1 : Above-average sell-side absorption (bearish hidden depth)
        - |DS| > 2 : Extreme absorption event (iceberg likely)
        - DS ≈ 0  : Doji bar (sign=0, capped AR) or average-absorption bar

    Interpretation
    --------------
    - Cluster of DS > 2 → institutional buyer absorbing supply
    - DS < -2 at resistance → institutional seller distributing into rally
    - Use as confirmation for breakout: rising price + DS > 1 = real demand
    """
    is_series = isinstance(open_, pd.Series)
    index = open_.index if is_series else None

    o = np.asarray(open_.values if is_series else open_, dtype=np.float64)
    c = np.asarray(close.values if isinstance(close, pd.Series) else close, dtype=np.float64)
    v = np.asarray(volume.values if isinstance(volume, pd.Series) else volume, dtype=np.float64)

    ar_raw = _bar_absorption_ratio_kernel(v, o, c)
    inf_mask = ~np.isfinite(ar_raw)
    if inf_mask.any():
        finite_vals = ar_raw[~inf_mask]
        cap = np.percentile(finite_vals, 99) if len(finite_vals) > 0 else v.max()
        ar_raw[inf_mask] = cap

    if index is None:
        index = pd.RangeIndex(len(ar_raw))

    ar_s = pd.Series(ar_raw, index=index)
    ar_ema = ar_s.ewm(span=window, adjust=False).mean()

    sign = np.sign(c - o)  # +1 bullish bar, -1 bearish bar, 0 doji
    ds = sign * (ar_raw / (ar_ema.values + 1e-10))
    if is_series:
        return pd.Series(ds, index=index, name='absorption_depth_score')
    return ds


@njit(cache=True)
def _multi_bar_depth_pressure_kernel(depth_score: np.ndarray, ar: np.ndarray, ar_ema: np.ndarray,
                                     threshold: float, window: int) -> np.ndarray:
    n = len(depth_score)
    dp = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        total = 0.0
        for j in range(i - window + 1, i + 1):
            ar_ratio = ar[j] / (ar_ema[j] + 1e-10)
            if ar_ratio > threshold and not np.isnan(depth_score[j]):
                total += depth_score[j]
        dp[i] = total
    return dp


def multi_bar_depth_pressure(open_: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20,
                             threshold: float = 1.5) -> pd.Series:
    """
    Multi-Bar Depth Pressure (DP_N). It cumulative directional depth score over a rolling window, counting only bars where
    absorption ratio is abnormally high (AR/EMA > threshold). Reveals sustained iceberg order campaigns.

    Formula
    -------
        AR(t)   = bar_absorption_ratio
        DS(t)   = absorption_weighted_depth_score
        DP_N    = Σ_{i=t-N+1}^{t} DS(i) × 𝟙[AR(i)/EMA(AR) > θ]

    Parameters
    ----------
    open_, close, volume : pd.Series
        OHLCV data.
    window : int, default=20
        Accumulation window.
    threshold : float, default=1.5
        AR / EMA(AR) threshold for "abnormal absorption".

    Returns
    -------
    pd.Series
        Name: ``multi_bar_depth_pressure_{window}``.

        - DP > 0   : Sustained hidden buying
        - DP < 0   : Sustained hidden selling
        - |DP| > 5 : Strong iceberg campaign over window
    """
    ar = bar_absorption_ratio(open_, close, volume)
    # Handle NaN AR for doji bars (leave as NaN in EMA; fill with 0 for kernel)
    ar_filled = ar.fillna(0.0)
    ar_ema = ar_filled.ewm(span=window, adjust=False).mean()
    ds = absorption_weighted_depth_score(open_, close, volume, window)
    dp = _multi_bar_depth_pressure_kernel(ds.values, ar_filled.values, ar_ema.values, threshold, window)
    return pd.Series(dp, index=open_.index, name=f'multi_bar_depth_pressure_{window}')


@njit(cache=True)
def _stealth_trading_kernel(delta_volume: np.ndarray, abs_returns: np.ndarray, volume: np.ndarray, vol_ema: np.ndarray,
                            window: int) -> np.ndarray:
    n = len(delta_volume)
    st = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        medium_impact = 0.0
        total_impact = 0.0

        for j in range(i - window + 1, i + 1):
            if np.isnan(delta_volume[j]) or np.isnan(abs_returns[j]):
                continue

            impact = abs(delta_volume[j]) * abs_returns[j]
            total_impact += impact

            lo = 0.5 * vol_ema[j]
            hi = 2.0 * vol_ema[j]
            if lo < volume[j] < hi:
                medium_impact += delta_volume[j] * abs_returns[j]

        if total_impact > 1e-10:
            st[i] = medium_impact / total_impact
    return st


def stealth_trading_indicator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                              window: int = 20) -> pd.DataFrame:
    """
    Stealth Trading Indicator (Barclay & Warner, 1993). It detects informed traders concealing order size by trading in
    medium-volume bars — the sweet spot between too-small (slow) and too-large (market-moving).

    Formula
    -------
        medium bar: 0.5×EMA(V) < V(t) < 2×EMA(V)
        ST          = Σ[δV × |r|]_medium / Σ[|δV| × |r|]_all   (rolling N)
        ST_direction = sign(Σ δV_medium)                          (rolling N)

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.
    window : int, default=20
        Analysis window.

    Returns
    -------
    pd.DataFrame
        Columns: ``ST``, ``ST_direction``.

        - ST > 0.5 : Most price impact from medium trades → stealth
        - ST_direction +1 / −1 : Buying / selling stealth flow
        - High ST + consistent direction → institutional execution
    """
    from ._primitives import buy_sell_volume as _bsv

    v_buy, v_sell = _bsv(high.values, low.values, close.values, volume.values)
    delta_v = v_buy - v_sell

    log_ret = np.log(close / close.shift(1))
    abs_ret = np.abs(log_ret).values

    vol_ema = volume.ewm(span=window, adjust=False).mean().values

    st = _stealth_trading_kernel(delta_v, abs_ret, volume.values, vol_ema, window)
    st_s = pd.Series(st, index=high.index)

    # Direction: sign of medium-volume δV sum over rolling window
    medium_mask = (volume > pd.Series(0.5 * vol_ema, index=high.index)) & \
                  (volume < pd.Series(2.0 * vol_ema, index=high.index))
    dv_s = pd.Series(delta_v, index=high.index)
    st_dir = np.sign((dv_s * medium_mask).rolling(window).sum())
    return pd.DataFrame({'ST': st_s, 'ST_direction': st_dir}, index=high.index)
