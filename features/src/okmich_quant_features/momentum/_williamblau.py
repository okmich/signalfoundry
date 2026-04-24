from typing import Union, Tuple

import numpy as np
import pandas as pd
import talib

from ..utils import ensure_numpy_types_for_series


def _ema(values: np.ndarray, period: int):
    """
    Return EMA of `values` with `period`.
    values: 1D numpy array
    """
    out = talib.EMA(values.astype(float), timeperiod=int(period))
    return np.asarray(out, dtype=float)


def triple_ema(series: pd.Series, p1: int = 1, p2: int = 1, p3: int = 1) -> pd.Series:
    if p1 == 1:
        e1 = series
    else:
        e1 = series.ewm(span=p1, adjust=False).mean()
    if p2 == 1:
        e2 = e1
    else:
        e2 = e1.ewm(span=p2, adjust=False).mean()
    if p3 == 1:
        e3 = e2
    else:
        e3 = e2.ewm(span=p3, adjust=False).mean()
    return e3


def true_strength_index(series: Union[pd.Series, np.ndarray, list], r: int = 25, s: int = 13, signal: int = 7,
                        as_percent: bool = False, fillna: bool = False, is_series_returns: bool = False) -> Union[
Tuple[pd.Series, pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the True Strength Index (TSI) by William Blau.

    TSI = 100 * EMA( EMA(DeltaPrice, r), s ) / EMA( EMA(abs(DeltaPrice), r), s )
    signal = EMA(TSI, signal)

    Parameters
    ----------
    series : pd.Series | np.ndarray | list
        Price series (close prices). If pd.Series, index is preserved in returned DataFrame.
    r : int
        First EMA period (short). Default 25.
    s : int
        Second EMA period (long). Default 13.
    signal : int
        Signal EMA period applied to the TSI. Default 7.
    as_percent : bool
        If True, multiplies TSI by 100 (typical representation). If False returns ratio.
    fillna : bool
        If True, forward-fills initial NaNs where EMAs aren't available. Defaults to False.
    is_series_returns: bool
        If True, the series passed in is the returns and not raw prices. Defaults to False.

    Returns
    -------
    (pd.Series, pd.Series, pd.Series) or (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        If input was pd.Series returns tuple of series with columns ['tsi', 'tsi_signal', 'tsi_Diff']
        Otherwise returns tuple (tsi_array, signal_array, diff_array)
    """
    # --- Normalize input to numpy array, preserve index if pandas Series ---
    idx, x = ensure_numpy_types_for_series(series)

    if x.ndim != 1:
        raise ValueError("price must be a 1D array-like of prices")
    n = x.shape[0]
    if n < 2:
        # can't compute momentum
        tsi = np.full(n, np.nan)
        sig = np.full(n, np.nan) if signal and signal > 0 else None
    else:
        # --- momentum and absolute momentum ---
        # delta: x_t - x_{t-1}; first element will be NaN
        if is_series_returns:
            delta = x
            abs_delta = np.abs(delta)
        else:
            delta = np.empty(n, dtype=float)
            delta[0] = np.nan
            delta[1:] = x[1:] - x[:-1]
            abs_delta = np.abs(delta)

        # --- double-smoothed EMA of delta and abs(delta) ---
        ema1_delta = _ema(delta, r)
        ema2_delta = _ema(ema1_delta, s)

        ema1_abs = _ema(abs_delta, r)
        ema2_abs = _ema(ema1_abs, s)

        # --- raw tsi ratio (avoid divide by zero) ---
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = ema2_delta / ema2_abs
            raw[~np.isfinite(raw)] = np.nan  # inf or -inf -> nan

        tsi = raw * (100.0 if as_percent else 1.0)

        # --- signal line  ---
        sig = None
        if signal and signal > 0:
            sig = _ema(np.where(np.isnan(tsi), np.nan, tsi), int(signal))
        else:
            sig = None

        if fillna:
            # forward fill NaNs using pandas to preserve behavior
            tmp = pd.DataFrame({"TSI": tsi})
            tmp = tmp.fillna(0)
            tsi = tmp["TSI"].to_numpy()
            if sig is not None:
                tmp2 = pd.Series(sig).fillna(method="ffill").fillna(0)
                sig = tmp2.to_numpy()

    if idx is not None:
        tsi = pd.Series(index=idx, data=tsi, name="tsi")
        tsi_signal = None
        tsi_diff = None
        if sig is not None:
            tsi_signal = pd.Series(index=idx, data=sig, name="tsi_signal")
            tsi_diff = pd.Series(index=idx, data=tsi - tsi_signal, name="tsi_diff")
        return tsi, tsi_signal, tsi_diff
    else:
        return tsi, sig, tsi - signal


def slope_divergence_tsi(series: pd.Series, r: int = 25, s: int = 13, signal: int = 7,
                         slope_period: int = 3, is_series_returns: bool = False, method: str = "diff"):
    """
    Compute Slope Divergence of TSI (SDTSI) by William Blau.

    Parameters
    ----------
    series : pd.Series
        Input price series (close prices).
    r : int
        First EMA period (default 25).
    s : int
        Second EMA period (default 13).
    signal : int
        Signal EMA period for TSI (default 7).
    slope_period : int
        Lookback used for slope calculation (default 3).
    method : {"diff", "ols"}
        "diff"  = simple first difference slope
        "ols"   = slope from rolling OLS regression
    is_series_returns: bool
        If True, the series passed in is the returns and not raw prices. Defaults to False.

    Returns
    -------
    Tuple of pd.Series
        Columns: ["tsi", "tsi_signal", "tsi_diff", "tsi_slope", "bullish_div", "bearish_div"]
    """
    # get TSI
    tsi, tsi_signal, tsi_diff = true_strength_index(series, r=r, s=s, signal=signal, as_percent=True,
                                                    is_series_returns=is_series_returns)
    if method == "diff":
        slope = tsi.diff(periods=slope_period)
    elif method == "ols":
        # rolling slope using simple OLS
        def _ols_slope(y):
            x = np.arange(len(y))
            x = x - x.mean()
            return np.dot(x, y - y.mean()) / np.dot(x, x)

        slope = tsi.rolling(window=slope_period, min_periods=slope_period).apply(
            _ols_slope, raw=False
        )
    else:
        raise ValueError("method must be 'diff' or 'ols'")

    # divergence flags
    bullish_div = ((series.diff() < 0) & (slope > 0)).astype(int)  # price down, tsi slope up
    bearish_div = ((series.diff() > 0) & (slope < 0)).astype(int)  # price up, tsi slope down
    return tsi, tsi_signal, tsi_diff, slope, bullish_div, bearish_div


def stochastic_momentum_index(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14,
                              r: int = 3, s: int = 3, signal: int = 3, as_percent: bool = True):
    """
    Compute William Blau's Stochastic Momentum Index (SMI).

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    k_period : int
        Lookback period for highest high / lowest low (default 14)
    r : int
        First EMA smoothing period (default 3)
    s : int
        Second EMA smoothing period (default 3)
    signal : int
        Signal EMA period applied to SMI (default 3)
    as_percent : bool
        If True, multiply by 100 (standard scaling)

    Returns
    -------
    Tuple of pd.Series
        Columns: ["smi", "smi_signal", "smi_diff]
    """
    # midpoint of high-low range
    hl_mid = (high + low) / 2

    # highest high and lowest low in lookback
    hh = high.rolling(k_period, min_periods=k_period).max()
    ll = low.rolling(k_period, min_periods=k_period).min()

    # price distance from midpoint
    diff = close - (hh + ll) / 2
    hl_range = (hh - ll) / 2

    # double-smoothed EMA of diff and range
    diff_ema1 = diff.ewm(span=r, adjust=False).mean()
    diff_ema2 = diff_ema1.ewm(span=s, adjust=False).mean()

    range_ema1 = hl_range.ewm(span=r, adjust=False).mean()
    range_ema2 = range_ema1.ewm(span=s, adjust=False).mean()

    # compute SMI
    smi = diff_ema2 / range_ema2.replace(0, np.nan)
    if as_percent:
        smi *= 100.0
    # signal line
    smi_signal = smi.ewm(span=signal, adjust=False).mean()
    # output
    return smi, smi_signal, smi - smi_signal


def directional_trend_index(high: pd.Series, low: pd.Series, q: int = 2, r: int = 20, s: int = 5, u: int = 3,
                                 signal: int | None = None,
                                 as_percent: bool = True) -> Tuple[pd.Series, pd.Series | None, pd.Series | None]:
    """
    William Blau Directional Trend Index (DTI), based on Composite High/Low Momentum (HLM).

    Parameters
    ----------
    high, low : pd.Series
        High and Low price series (index aligned with close/time).
    q : int
        HLM period (default 2 in Blau's examples).
    r, s, u : int
        EMA smoothing periods (triple smoothing): default (20, 5, 3) are common Blau defaults.
    signal : int | None
        If provided, compute an EMA(signal) of the DTI as a signal line.
    as_percent : bool
        Multiply final ratio by 100 (default True).

    Returns
    -------
    (dti, dti_signal, dti_diff)
      dti : pd.Series
        Directional Trend Index values.
      dti_signal : pd.Series or None
        EMA(signal) of DTI if `signal` provided, else None.
      dti_diff : pd.Series or None
        dti - dti_signal if signal provided else None.
    """
    # --- HMU and LMD definitions per Blau ---
    # HMU(q) = High - High.shift(q-1) if positive else 0
    # LMD(q) = -(Low - Low.shift(q-1)) if negative else 0
    prev_high = high.shift(q - 1)
    prev_low = low.shift(q - 1)

    delta_high = high - prev_high
    delta_low = low - prev_low

    hmu = delta_high.where(delta_high > 0, 0.0)  # Up trend momentum
    lmd = (-delta_low).where(delta_low < 0, 0.0)  # Down trend momentum (positive values)

    hlm = hmu - lmd  # composite High/Low Momentum (can be positive or negative)

    hlm_sm = triple_ema(hlm, r, s, u)
    abs_hlm_sm = triple_ema(hlm.abs(), r, s, u)

    # --- DTI ratio (safe divide) ---
    denom = abs_hlm_sm.replace(0, np.nan)
    dti = hlm_sm / denom
    if as_percent:
        dti = dti * 100.0

    # fill NaN with 0 where denominator was zero per Blau's note
    dti = dti.fillna(0.0)

    # optional signal line
    dti_signal = None
    dti_diff = None
    if signal is not None and signal > 0:
        dti_signal = dti.ewm(span=signal, adjust=False).mean()
        dti_diff = dti - dti_signal
    return dti, dti_signal, dti_diff


def directional_efficiency_index(high: pd.Series, low: pd.Series, close: pd.Series, r: int = 14, s: int = 14,
                                 signal: int = 9, as_percent: bool = False) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Directional Efficiency Index (DEI)
    ----------------------------------

    A volatility-normalized directional strength oscillator inspired by the smoothing philosophy of William Blau
    but distinct from his canonical Directional Trend Index (DTI).

    This indicator measures *how directionally efficient* recent price movements have been relative to total volatility (True Range).
    It applies double-EMA smoothing to both directional movement (DM) and True Range (TR), then takes their ratio to produce a continuous
    signed trend-strength measure.

    ----------------------------------------------------------------------
    Formula (conceptual)
    ----------------------------------------------------------------------
        DM_t = Close_t - Close_{t-1}
        TR_t = max(High_t, Close_{t-1}) - min(Low_t, Close_{t-1})

        DM_smoothed = EMA(EMA(DM_t, r), s)
        TR_smoothed = EMA(EMA(TR_t, r), s)

        DEI_t = DM_smoothed / TR_smoothed
        (optionally scaled by 100)

        Signal_t = EMA(DEI_t, signal)
        Diff_t   = DEI_t - Signal_t

    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    high : pd.Series
        Series of high prices.
    low : pd.Series
        Series of low prices.
    close : pd.Series
        Series of close prices.
    r : int, default 14
        First EMA smoothing period (inner smoothing).
    s : int, default 14
        Second EMA smoothing period (outer smoothing).
    signal : int, default 9
        EMA period applied to DEI to create a signal line.
    as_percent : bool, default True
        If True, DEI values are scaled by 100.

    ----------------------------------------------------------------------
    Returns
    ----------------------------------------------------------------------
    Tuple of pd.Series
        (dei, dei_signal, dei_diff)
        - dei : Directional Efficiency Index values
        - dei_signal : EMA(signal) of DEI
        - dei_diff : dei - dei_signal

    ----------------------------------------------------------------------
    Interpretation
    ----------------------------------------------------------------------
    - DEI > 0 : Upside directional efficiency dominates (bullish bias)
    - DEI < 0 : Downside directional efficiency dominates (bearish bias)
    - |DEI| large : Trend conviction and volatility alignment increase
    - DEI ≈ 0 : Choppy or mean-reverting conditions

    ----------------------------------------------------------------------
    Relation to Other Indicators
    ----------------------------------------------------------------------
    * This indicator is NOT Blau's official DTI (which is based on
      High/Low Momentum). It is a *Blau-style smoothed ratio* derived from the Directional Movement (DM) and True Range (TR) concepts introduced by Welles Wilder.
    * Functionally, it parallels the True Strength Index (TSI), but normalizes directional movement by volatility rather than by
      absolute price change magnitude.
    * It can also be viewed as a volatility-normalized "trend efficiency" measure — how much of total volatility has been
      directionally consistent.

    ----------------------------------------------------------------------
    Origins and Context
    ----------------------------------------------------------------------
    - The DM/TR ratio form appeared in early Omega/TradeStation and Amibroker communities (circa 1990s–2000s) under names such as “Blau Directional Trend Index” or “Smoothed Directional Strength.”
    - It embodies Blau’s philosophy of double EMA smoothing to remove high-frequency noise while preserving directional sensitivity.
    - Not found in *Momentum, Direction, and Divergence (1995)*, but conceptually consistent with his later "True Strength Index."

    ----------------------------------------------------------------------
    Recommended Usage
    ----------------------------------------------------------------------
    - Trend confirmation oscillator (cross above/below signal line)
    - Volatility-adjusted trend filter for entry logic
    - Regime classification feature for machine learning pipelines

    ----------------------------------------------------------------------
    Author Notes
    ----------------------------------------------------------------------
    Written as a distinct, research-oriented oscillator to preserve the DM/TR concept lineage while avoiding confusion
    with Blau's original DTI.
    """

    # --- Directional Movement ---
    dm = close.diff()

    # --- True Range ---
    prev_close = close.shift(1)
    tr = np.maximum(high, prev_close) - np.minimum(low, prev_close)

    # --- Double EMA smoothing ---
    dm_ema1 = dm.ewm(span=r, adjust=False).mean()
    dm_ema2 = dm_ema1.ewm(span=s, adjust=False).mean()

    tr_ema1 = tr.ewm(span=r, adjust=False).mean()
    tr_ema2 = tr_ema1.ewm(span=s, adjust=False).mean()

    # --- Directional Efficiency Index ---
    dei = dm_ema2 / tr_ema2.replace(0, np.nan)
    if as_percent:
        dei *= 100.0

    # --- Signal line and difference ---
    dei_signal = dei.ewm(span=signal, adjust=False).mean()
    dei_diff = dei - dei_signal

    return dei, dei_signal, dei_diff
