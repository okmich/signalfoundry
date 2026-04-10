from typing import Union, List, Tuple, Optional

import numpy as np
import pandas as pd
from numba import jit
from numpy.lib.stride_tricks import sliding_window_view

# Type alias for arrays that can be either pandas Series or numpy array
ArrayLike = Union[pd.Series, np.ndarray]


def _to_numpy(data: ArrayLike) -> Tuple[np.ndarray, Optional[pd.Index]]:
    """
    Convert input to numpy array and preserve index if it was a Series.

    Returns:
        (numpy_array, index_or_None)
    """
    if isinstance(data, pd.Series):
        return data.values, data.index
    return np.asarray(data), None


def _to_output_type(data: np.ndarray, index: Optional[pd.Index], name: str = None) -> ArrayLike:
    """
    Convert numpy array back to original input type.

    If index is not None (was originally a Series), return Series.
    Otherwise return numpy array.
    """
    if index is not None:
        return pd.Series(data, index=index, name=name)
    return data


@jit(nopython=True)
def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        result[i] = np.std(arr[i - window + 1 : i + 1])
    return result


@jit(nopython=True)
def _log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices))


def _validate_prices(**arrays: np.ndarray) -> None:
    """Raise ValueError if any finite price value is non-positive."""
    for name, arr in arrays.items():
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size > 0 and np.any(finite_vals <= 0):
            raise ValueError(
                f"'{name}' contains non-positive prices; log-based estimators "
                f"require strictly positive inputs (min finite value: {finite_vals.min():.6g})"
            )


@jit(nopython=True)
def _garman_klass_volatility_nb(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 20) -> np.ndarray:
    log_open = np.log(open_)
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)

    term1 = 0.5 * (log_high - log_low) ** 2
    term2 = (2 * np.log(2) - 1) * (log_close - log_open) ** 2

    gk_single = np.sqrt(np.maximum(term1 - term2, 0.0))

    # Annualize and apply rolling window
    result = np.full_like(high, np.nan)
    for i in range(window - 1, len(high)):
        result[i] = np.sqrt(252) * np.sqrt(np.mean(gk_single[i - window + 1 : i + 1] ** 2))

    return result


@jit(nopython=True)
def _parkinson_volatility_nb(high: np.ndarray, low: np.ndarray, window: int = 20) -> np.ndarray:
    log_high = np.log(high)
    log_low = np.log(low)
    hl_ratio = log_high - log_low
    parkinson_single = np.sqrt((1.0 / (4.0 * np.log(2.0))) * hl_ratio**2)

    # Annualize and apply rolling window
    result = np.full_like(high, np.nan)
    for i in range(window - 1, len(high)):
        result[i] = np.sqrt(252) * np.sqrt(
            np.mean(parkinson_single[i - window + 1 : i + 1] ** 2)
        )

    return result


def rolling_volatility(series: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Computes rolling volatility using log returns.

    Parameters
    ----------
    series : ArrayLike
        Price series
    window : int, default=20
        Rolling window size

    Returns
    -------
    ArrayLike
        Rolling volatility (same type as input)
    """
    arr, index = _to_numpy(series)
    arr = np.asarray(arr, dtype=np.float64)
    _validate_prices(close=arr)

    # Compute log returns
    log_rets = np.diff(np.log(arr))

    # Rolling standard deviation
    result = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        result[i] = np.std(log_rets[i - window : i])

    return _to_output_type(result, index, name='rolling_vol')


def parkinson_volatility(high: ArrayLike, low: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Parkinson volatility estimator using high-low range (more efficient than close-to-close).

    SUPERIOR intra-bar volatility measure that captures the true quant range.

    Parameters
    ----------
    high : ArrayLike
        Array of high prices
    low : ArrayLike
        Array of low prices
    window : int, default=20
        Rolling window for smoothing the estimator

    Returns
    -------
    ArrayLike
        Annualized Parkinson volatility estimates (same type as input)

    Key Advantages
    --------------
    • EFFICIENCY: 5x more efficient than close-to-close volatility
    • INTRA-BAR: Captures true quant range within each period
    • SENSITIVITY: Reacts faster to changing market conditions

    Significance and Usage
    ----------------------
    • PANIC DETECTION: Spikes indicate disorderly quant and panic moves
    • GAP RISK: Excellent predictor of overnight gap probabilities
    • LIQUIDITY: High values suggest poor liquidity and wide spreads
    • OPENING AUCTION: Critical for assessing opening volatility

    Trading Applications
    --------------------
    • CRISIS DETECTION: Identify flash crashes and surge events early
    • POSITION SIZING: Adjust sizes based on intra-bar volatility
    • STOP LOSSES: Set stops based on true range rather than close-based vol
    • OVERNIGHT RISK: Manage positions before close based on Parkinson spikes

    Optimal Scenarios
    -----------------
    • Market opens and closes
    • News events and economic releases
    • Low liquidity periods (lunch time, holidays)
    • Combining with realized vol for ratio analysis
    """
    # Convert to numpy for Numba computation
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _, index = _to_numpy(high)

    _validate_prices(high=high_arr, low=low_arr)

    # Call Numba-optimized function
    result = _parkinson_volatility_nb(high_arr, low_arr, window)

    return _to_output_type(result, index, name='parkinson_vol')


def garman_klass_volatility(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 20) -> ArrayLike:
    """
    Garman-Klass volatility estimator incorporating OHLC data (most efficient OHLC estimator).

    THE MOST EFFICIENT volatility estimator using open, high, low, close data.

    Parameters
    ----------
    open_ : ArrayLike
        Array of opening prices
    high : ArrayLike
        Array of high prices
    low : ArrayLike
        Array of low prices
    close : ArrayLike
        Array of closing prices
    window : int, default=20
        Rolling window for smoothing the estimator

    Returns
    -------
    ArrayLike
        Annualized Garman-Klass volatility estimates (same type as input)

    Key Advantages
    --------------
    • MAXIMUM EFFICIENCY: Most statistically efficient OHLC-based estimator
    • COMPREHENSIVE: Incorporates all price information (O, H, L, C)
    • ACCURACY: Better reflects true volatility than close-to-close

    Significance and Usage
    ----------------------
    • OPTIONS PRICING: Superior for estimating historical vol in options models
    • RISK MODELS: More accurate for VaR and risk measurement
    • BENCHMARKING: Gold standard for comparing volatility estimation methods
    • RESEARCH: Preferred for academic and quantitative research

    Trading Applications
    --------------------
    • VOLATILITY TRADING: More accurate for volatility mean reversion strategies
    • PORTFOLIO CONSTRUCTION: Better risk estimation for optimization
    • PERFORMANCE ATTRIBUTION: More precise volatility contribution analysis
    • HEDGING: Improved hedge ratios for options and volatility products

    When to Use vs Parkinson
    ------------------------
    • USE Garman-Klass for: Options pricing, risk management, research
    • USE Parkinson for: Intraday quant, panic detection, real-time monitoring
    """
    # Convert to numpy for Numba computation
    open_arr, _ = _to_numpy(open_)
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    close_arr, index = _to_numpy(close)

    _validate_prices(open_=open_arr, high=high_arr, low=low_arr, close=close_arr)

    # Call Numba-optimized function
    result = _garman_klass_volatility_nb(open_arr, high_arr, low_arr, close_arr, window)

    return _to_output_type(result, index, name='garman_klass_vol')


@jit(nopython=True, cache=True)
def _compute_vw_vol_nb(ret_windows, vol_windows, window):
    n_windows = ret_windows.shape[0]
    result = np.full(n_windows, np.nan, dtype=np.float64)

    for i in range(n_windows):
        vols = vol_windows[i, :]
        sum_vols = np.sum(vols)

        if sum_vols == 0:
            continue  # Leave as NaN

        weights = vols / sum_vols
        rets = ret_windows[i, :]

        if np.any(np.isnan(rets)):
            continue  # Leave as NaN

        w_mean = np.sum(weights * rets)
        deviations_sq = (rets - w_mean) ** 2
        w_var = np.sum(weights * deviations_sq)
        vw_vol = np.sqrt(w_var)
        if not np.isnan(vw_vol) and not np.isinf(vw_vol):
            result[i] = vw_vol

    return result


def volume_weighted_volatility(close: pd.Series, volume: pd.Series, window: int = 20, annualize: bool = False,
                               periods_per_year: int = 252) -> pd.Series:
    """
    Volume-weighted volatility: Adjusts for liquidity conditions.

    DESCRIPTION:
    Standard deviation weighted by volume. Gives more weight to high-volume periods, reducing impact of thin/illiquid
    periods. Better representation of "tradable" volatility vs statistical volatility.

    EFFICIENCY:
    - Medium computational cost
    - Good for: Illiquid assets, crypto, pre/post-market sessions
    - Excellent for: Position sizing in variable liquidity

    REGIME DETECTION:
    ~ MODERATE - Better as feature than standalone regime detector
    - VW_Vol / Simple_Vol ratio > 1.2 = High volume volatility regime
    - Use for: Identifying when volatility is "real" vs noise

    BEST APPLICATIONS:
    - Crypto markets (variable liquidity across 24h)
    - Pre/post market hours (thin liquidity)
    - Risk management (use for stop-loss in illiquid periods)
    - ML features (better than simple vol for liquidity-sensitive models)
    """
    if len(close) < window:
        return pd.Series([np.nan] * len(close), index=close.index)

    # Compute log returns
    ret = np.log(close.values / close.shift(1).values)
    vol = volume.values.astype(np.float64)

    # Create rolling windows
    ret_windows = sliding_window_view(ret, window_shape=window)  # Shape: (N - w + 1, w)
    vol_windows = sliding_window_view(vol, window_shape=window)

    # Compute volume-weighted volatility using module-level Numba kernel
    volatility = _compute_vw_vol_nb(ret_windows, vol_windows, window)
    # are we annulazing
    if annualize:
        volatility *= np.sqrt(periods_per_year)

    result = np.full(len(close), np.nan)
    result[window - 1 :] = volatility
    return pd.Series(result, index=close.index, name="vol_volume_weighted")


def yang_zhang_volatility(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 20,
                          annualize: bool = False, periods_per_year: int = 252) -> ArrayLike:
    """
    Yang-Zhang volatility: Handles overnight gaps and drift.

    DESCRIPTION:
    Combines overnight volatility (close-to-open), open-to-close volatility,
    and Rogers-Satchell intraday component. Drift-independent and handles gaps.
    Most complete single-estimator for 24/7 markets.

    EFFICIENCY:
    - High computational cost but most complete
    - Good for: All market conditions, especially crypto/indices with gaps
    - Best for: Multi-session markets, assets with significant overnight moves

    REGIME DETECTION:
    ✓ YES - Excellent, especially for gap regimes
    - YZ/Parkinson ratio > 1.5 = High overnight gap risk
    - YZ spike = Overnight event regime
    - Use for crypto (24/7) and indices (gap risk)

    BEST APPLICATIONS:
    - Crypto volatility (continuous trading, frequent gaps)
    - Index futures (overnight gap risk management)
    - Risk models requiring gap risk component
    - Use as primary estimator for multi-session assets
    """
    # Preserve index from close; extract all arrays for price validation
    close_arr, index = _to_numpy(close)
    open_arr, _ = _to_numpy(open_)
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _validate_prices(open_=open_arr, high=high_arr, low=low_arr, close=close_arr)

    # Convert to Series for rolling operations if needed
    if index is None:
        open_ = pd.Series(open_)
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

    # Overnight volatility (close to open)
    co = np.log(open_ / close.shift(1))
    co_vol = co.rolling(window).var()

    # Open to close volatility
    oc = np.log(close / open_)
    oc_vol = oc.rolling(window).var()

    # Rogers-Satchell component (intraday)
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    rs_vol = rs.rolling(window).mean()

    # Yang-Zhang estimator with k coefficient
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_vol = np.sqrt(co_vol + k * oc_vol + (1 - k) * rs_vol)
    if annualize:
        yz_vol = yz_vol * np.sqrt(periods_per_year)
    return _to_output_type(yz_vol.values, index, name='yang_zhang_vol')


def rogers_satchell_volatility(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 20,
                               annualize: bool = False, periods_per_year: int = 252) -> ArrayLike:
    """
    Rogers-Satchell volatility: Drift-independent, no close price needed.

    DESCRIPTION:
    Geometric average of intraday movements. Eliminates drift bias, making it
    ideal for trending markets. Uses relationship between OHLC without close-to-close.
    Formula: sqrt(ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O))

    EFFICIENCY:
    - Medium computational cost
    - Good for: Trending markets, when drift matters
    - Excellent for: Strong trend identification (low RS = strong trend)

    REGIME DETECTION:
    ✓ YES - Unique for trend/consolidation regimes
    - RS/Parkinson ratio < 0.8 = Strong directional trend
    - RS/Parkinson ratio > 1.2 = Choppy/ranging market
    - Low RS = Trending regime (use trend following strategies)

    BEST APPLICATIONS:
    - Trend detection (low RS = clean trend, high RS = chop)
    - Strategy selection (low RS → trend following, high RS → mean reversion)
    - Use in combination with Parkinson for regime classification
    """
    # Preserve index from close; extract all arrays for price validation
    close_arr, index = _to_numpy(close)
    open_arr, _ = _to_numpy(open_)
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _validate_prices(open_=open_arr, high=high_arr, low=low_arr, close=close_arr)

    # Convert to Series for rolling operations if needed
    if index is None:
        open_ = pd.Series(open_)
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)

    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    rs_vol = np.sqrt(np.maximum(rs.rolling(window).mean(), 0.0))
    if annualize:
        rs_vol = rs_vol * np.sqrt(periods_per_year)
    return _to_output_type(rs_vol.values, index, name='rogers_satchell_vol')


def realized_volatility(close: ArrayLike, window: int = 60, freq_minutes: int = 5, trading_hours_per_day: float = 24,
                        trading_days_per_year: int = 252, reset_daily: bool = False, annualize: bool = True) -> pd.Series:
    """
    Compute realized volatility (single horizon) using
    the sum of squared log returns over a time-based rolling window.

    Parameters
    ----------
    close : array-like or pd.Series
        Series of closing prices indexed by DatetimeIndex.
    window : int, default=60
        Lookback window in minutes (e.g. 60 = 1 hour on 5-min data).
    freq_minutes : int, default=5
        Bar frequency in minutes.
    trading_hours_per_day : float, default=24
        Used for annualization scaling (6.5 for equities, 24 for FX/crypto).
    trading_days_per_year : int, default=252
        Trading days per year for annualization.
    reset_daily : bool, default=False
        If True, resets calculation each trading day (intraday volatility only).
    annualize : bool, default=True
        Whether to scale volatility to annualized units.

    Returns
    -------
    pd.Series
        Realized volatility series labeled as 'rv_{window}m'.

    Notes
    -----
    - Realized volatility is sqrt(sum of squared log returns)
    - Uses time-based rolling window ('Xm') for consistent horizons

    Example
    -------
    >>> rv_60m = realized_volatility(close, window=60, freq_minutes=5)
    >>> rv_60m.tail()
    """
    # --- Input handling
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("`close` must have a DatetimeIndex for time-based rolling.")

    log_rets = np.log(close / close.shift(1))
    if reset_daily:
        groups = log_rets.groupby(log_rets.index.date)
    else:
        groups = [(None, log_rets)]

    if annualize:
        minutes_per_year = trading_days_per_year * trading_hours_per_day * 60
        scale = np.sqrt(minutes_per_year / freq_minutes)
    else:
        scale = 1.0

    rv_series_list = []

    window_str = f"{window}min"
    for _, group in groups:
        rv = group.rolling(window=window_str).apply(
            lambda x: np.sqrt(np.sum(x**2)), raw=True
        )
        rv_series_list.append(rv)

    rv_concat = pd.concat(rv_series_list)
    rv_concat.name = f"rv_{window}"

    # --- Scale and return
    return rv_concat * scale


def realized_volatility_for_windows(close: ArrayLike, windows: List[int] = None, freq_minutes: int = 5,
                                    trading_hours_per_day: float = 24, trading_days_per_year: int = 252,
                                    reset_daily: bool = False, annualize: bool = True) -> pd.DataFrame:
    """
    Compute realized volatility for multiple time horizons simultaneously.

    Parameters
    ----------
    close : array-like or pd.Series
        Series of closing prices indexed by DatetimeIndex.
    windows : list of int, default=[60, 180, 360]
        Lookback windows in **minutes** (e.g. 60=1h, 180=3h, 360=6h).
        These are time-based windows, NOT bar counts. With freq_minutes=5,
        [60, 180, 360] spans 12, 36, and 72 bars respectively.
        Do NOT pass bar counts here — multiply by freq_minutes first.
    freq_minutes : int, default=5
        Bar frequency in minutes. Used only for annualization scaling.
    trading_hours_per_day : float, default=24
        Used for annualization (6.5 for equities, 24 for FX/crypto).
    trading_days_per_year : int, default=252
        Trading days per year for annualization.
    reset_daily : bool, default=False
        If True, resets calculation at daily boundaries.
    annualize : bool, default=True
        Whether to scale volatility to annualized units.

    Returns
    -------
    pd.DataFrame
        Columns: rv_{window} for each window in windows.
    """
    # --- Input handling
    if windows is None:
        windows = [60, 180, 360]
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("`close` must have a DatetimeIndex for time-based rolling.")

    log_rets = np.log(close / close.shift(1))
    log_rets.name = "log_ret"
    if reset_daily:
        grouped = log_rets.groupby(log_rets.index.date)
    else:
        grouped = [(None, log_rets)]

    if annualize:
        minutes_per_year = trading_days_per_year * trading_hours_per_day * 60
        scale = np.sqrt(minutes_per_year / freq_minutes)
    else:
        scale = 1.0

    result = pd.DataFrame(index=close.index)

    # --- Compute realized volatility per window
    for window in windows:
        window_str = f"{window}min"
        rv_series_list = []

        for _, group in grouped:
            # Realized variance = rolling sum of squared returns
            rv = group.rolling(window=window_str).apply(
                lambda x: np.sqrt(np.sum(x**2)), raw=True
            )
            rv_series_list.append(rv)

        rv_concat = pd.concat(rv_series_list)
        result[f"rv_{window}"] = rv_concat * scale
    return result


def realized_volatility_with_bipower_jump_variations(close: ArrayLike, window: int = 60, freq_minutes: int = 5,
                                                     trading_hours_per_day: float = 24, trading_days_per_year: int = 252,
                                                     reset_daily: bool = False, annualize: bool = True) -> (pd.Series, pd.Series, pd.Series):
    """
    Intraday Realized Volatility Module
    ===================================

    Computes multiple realized volatility measures:
      • Realized Volatility (RV): sqrt(sum of squared log returns)
      • Bipower Variation (BPV): robust to jumps
      • Jump Variation (JV): difference between RV² and BPV²

    Definitions
    =================================
    - Realized volatility measures how much prices actually moved over a period of time, based on intraday returns.
        It’s literally the sum of squared returns — capturing everything that happened: smooth price drift, noise, and sudden jumps.
    - Bipower variation estimates the continuous part of volatility — that is, how much prices would have fluctuated without jumps.
        It uses the product of consecutive absolute returns, which downplays the impact of large, isolated spikes which
        captures the “background turbulence” — the natural shaking of the market caused by normal trading activity
        (not news shocks).
    - Jump variation isolates discontinuous moves — large, abrupt changes that can’t be explained by normal volatility.

    Relationship
    =====================================
    Falling RV, Falling BV <=> constant JV - Quiet, low activity (Tight spreads, low volume)        - Calm market
    Rising RV, Constant BV <=> Rising JV - Sudden shock/news (Huge spreads, rising gaps, Sudden shock/news)
    Rising RV, Rising BV <=> constant JV - Trend acceleration or high activity (Active, tradable market)

    Parameters
    ----------
    close : array-like or pd.Series of close prices indexed by timestamps.
    window : int - Lookback windows in minutes (e.g. 60=1h, 180=3h, 360=6h).
    freq_minutes : int, default=5 - Bar frequency in minutes.
    trading_hours_per_day : float, default=24 - Used for annualization scaling (6.5 for equities, 24 for crypto/FX).
    trading_days_per_year : int, default=252 - Number of trading days per year.
    reset_daily : bool, default=False - If True, resets window at daily boundaries (intraday analysis).
    annualize : bool, default=True - If True, scales volatility to annualized terms.

    Returns
    -------
    Tuple of pd.Series
          rv_{window}   : realized volatility
          bpv_{window}  : bipower variation (jump-robust volatility)
          jv_{window}   : jump variation

    References
    ----------
    - Andersen, Bollerslev, Diebold & Labys (2003). *Modeling and Forecasting Realized Volatility.*
    - Barndorff-Nielsen & Shephard (2004). *Power and Bipower Variation with Stochastic Volatility and Jumps.*

    Example
    -------
    >>> rv_df = realized_volatility_with_bipower_jump_variations(close, 60, freq_minutes=5)
    >>> rv_df.tail()
    """

    # --- Input validation
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(
            "`close` must have a DatetimeIndex for time-based rolling windows."
        )

    # --- Log returns
    log_rets = np.log(close / close.shift(1))
    log_rets.name = "log_ret"

    # --- Group by day if requested
    if reset_daily:
        groups = log_rets.groupby(log_rets.index.date)
    else:
        groups = [(None, log_rets)]

    # --- Annualization factor
    if annualize:
        minutes_per_year = trading_days_per_year * trading_hours_per_day * 60
        scale = np.sqrt(minutes_per_year / freq_minutes)
    else:
        scale = 1.0

    result = pd.DataFrame(index=close.index)

    window_str = f"{window}min"
    rv_list, bpv_list, jv_list = [], [], []
    for _, group in groups:
        # Drop NaNs early
        r = group.dropna()

        # --- 1. Realized Variance (RV)
        rv = r.rolling(window=window_str).apply(
            lambda x: np.sqrt(np.sum(x**2)), raw=True
        )

        # --- 2. Bipower Variation (BPV)
        # BPV = μ1^-2 * Σ |r_i| * |r_{i-1}|
        mu1 = np.sqrt(2 / np.pi)
        abs_r = np.abs(r)
        bpv_vals = abs_r * abs_r.shift(1)
        bpv = bpv_vals.rolling(window=window_str).mean() / (mu1**2)
        bpv = np.sqrt(bpv)

        # --- 3. Jump Variation (JV)
        # JV = max(0, RV² - BPV²)
        jv = np.sqrt(np.maximum(0, rv**2 - bpv**2))

        rv_list.append(rv)
        bpv_list.append(bpv)
        jv_list.append(jv)

    # --- Combine results
    rv = pd.concat(rv_list) * scale
    bpv = pd.concat(bpv_list) * scale
    jv = pd.concat(jv_list) * scale

    return rv, bpv, jv


def realized_volatility_window_with_bipower_jump_variations(close: ArrayLike, windows: List[int] = None,
                                                            freq_minutes: int = 5, trading_hours_per_day: float = 24,
                                                            trading_days_per_year: int = 252, reset_daily: bool = False,
                                                            annualize: bool = True) -> pd.DataFrame:
    # --- Input validation: normalize first, then check index type
    if windows is None:
        windows = [60, 180, 360]
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError(
            "`close` must have a DatetimeIndex for time-based rolling windows."
        )

    # --- Log returns
    log_rets = np.log(close / close.shift(1))
    log_rets.name = "log_ret"

    # --- Group by day if requested
    if reset_daily:
        groups = log_rets.groupby(log_rets.index.date)
    else:
        groups = [(None, log_rets)]

    # --- Annualization factor
    if annualize:
        minutes_per_year = trading_days_per_year * trading_hours_per_day * 60
        scale = np.sqrt(minutes_per_year / freq_minutes)
    else:
        scale = 1.0

    result = pd.DataFrame(index=close.index)

    # --- Loop through horizons
    for window in windows:
        window_str = f"{window}min"
        rv_list, bpv_list, jv_list = [], [], []

        for _, group in groups:
            # Drop NaNs early
            r = group.dropna()

            # --- 1. Realized Variance (RV)
            rv = r.rolling(window=window_str).apply(
                lambda x: np.sqrt(np.sum(x**2)), raw=True
            )

            # --- 2. Bipower Variation (BPV)
            # BPV = μ1^-2 * Σ |r_i| * |r_{i-1}|
            mu1 = np.sqrt(2 / np.pi)
            abs_r = np.abs(r)
            bpv_vals = abs_r * abs_r.shift(1)
            bpv = bpv_vals.rolling(window=window_str).mean() / (mu1**2)
            bpv = np.sqrt(bpv)

            # --- 3. Jump Variation (JV)
            # JV = max(0, RV² - BPV²)
            jv = np.sqrt(np.maximum(0, rv**2 - bpv**2))

            rv_list.append(rv)
            bpv_list.append(bpv)
            jv_list.append(jv)

        # --- Combine results
        result[f"rv_{window}"] = pd.concat(rv_list) * scale
        result[f"bpv_{window}"] = pd.concat(bpv_list) * scale
        result[f"jv_{window}"] = pd.concat(jv_list) * scale

    return result


def volatility_signature(close: ArrayLike, short_window: int = 12, long_window: int = 72, freq_minutes: int = 5) -> ArrayLike:
    """
    Calculate the volatility signature ratio (short-term/long-term volatility).

    This is the MOST POWERFUL predictor for market regime detection as it captures
    the term structure dynamics of volatility across different time horizons.

    Parameters:
    -----------
    close : pd.Series - closing prices
    short_window : int, default=12
        Lookback period for short-term volatility (e.g., 12 periods = 1 hour for 5min bars)
    long_window : int, default=72
        Lookback period for long-term volatility (e.g., 72 periods = 6 hours for 5min bars)
    freq_minutes : int, default=5
        Bar frequency in minutes (for context)

    Returns:
    --------
    pd.Series
        Ratio of short-term to long-term volatility. Values indicate:
        - > 1.2: High volatility trending regime
        - 0.8-1.2: Transition/neutral regime
        - < 0.8: Low volatility mean reversion regime
        - > 1.5: Crisis/event-driven regime

    Significance and Regime Detection Insights:
    ------------------------------------------
    PRIMARY USE: Market regime detection and early warning system for regime shifts

    REGIME INTERPRETATION:
    • Rising ratio (short-term > long-term): Momentum/trend regime
      - Price trending with accelerating volatility
      - Ideal for trend-following strategies
      - Often precedes major directional moves

    • Falling ratio (short-term < long-term): Mean reversion/consolidation regime
      - Price oscillating in range with decaying volatility
      - Ideal for mean reversion strategies
      - Typically occurs after extended trends

    • Spiking ratio (>1.5): Crisis/event-driven regime
      - Disorderly market conditions, panic moves
      - High risk of gap events and flash crashes
      - Avoid mean reversion, reduce position sizes

    • Flat ratio (~1.0): Stable/low-vol regime
      - Efficient price discovery, normal market functioning
      - Good for high-frequency and statistical arbitrage

    WHY IT'S IDEAL FOR REGIME DETECTION:
    • Captures term structure dynamics across time horizons
    • Leading indicator (typically leads price action by several bars)
    • Normalized and comparable across instruments/timeframes
    • Non-parametric (no distribution assumptions)
    • Adaptive to current volatility environment
    • Robust across asset classes

    TRADING APPLICATIONS:
    • Entry/exit timing for trend-following strategies
    • Position sizing based on regime volatility
    • Risk management during crisis regimes
    • Strategy selection (trend vs mean reversion)
    • Overnight gap risk prediction

    OPTIMAL USAGE:
    • Market open (9:30-10:30 ET): Predicts day's range and regime
    • News events: Identifies true market impact vs noise
    • Afternoon session (2:00-4:00 ET): Predicts close volatility
    • Combined with volume: Convergence with expanding volume = strong regime shift signal

    Example:
    --------
    >>> signature = volatility_signature(close_prices, 12, 72, 5)
    >>> # High vol trending regime: signature > 1.2
    >>> # Low vol mean reversion: signature < 0.8
    >>> # Crisis mode: signature > 1.5
    """
    short_realized_vol = realized_volatility(close, short_window, freq_minutes)
    long_realized_vol = realized_volatility(close, long_window, freq_minutes)
    signature = short_realized_vol / (long_realized_vol + 1e-09)
    return signature


def volatility_ratio(high: ArrayLike, low: ArrayLike, close: ArrayLike, window: int = 20, freq_minutes: int = 5) -> ArrayLike:
    """
    Ratio of Parkinson volatility to realized volatility (intra-bar vs close-to-close).

    POWERFUL indicator of market microstructure and quant behavior.

    Parameters:
    -----------
    high : pd.Series - high prices
    low : pd.Series - low prices
    close : pd.Series - closing prices
    window : int, default=20 - Window for Parkinson and Realized volatility
    freq_minutes : int, default=20 - bars frequency

    Returns:
    --------
    pd.Series - Ratio of Parkinson to realized volatility

    Interpretation Guide:
    --------------------
    • RATIO > 1.5: Extreme panic/disorderly quant (AVOID mean reversion)
    • RATIO 1.0-1.5: Elevated intra-bar activity (CAUTION with reversions)
    • RATIO 0.8-1.0: Normal market functioning (GOOD for all strategies)
    • RATIO < 0.8: Efficient price discovery (EXCELLENT for mean reversion)

    Significance and Usage:
    ----------------------
    • MICROSTRUCTURE: Reveals the quality of price discovery
    • LIQUIDITY: High ratio = poor liquidity, wide spreads
    • PANIC METER: Measures emotional vs rational quant
    • GAP PREDICTION: Best predictor of overnight gap risk

    Trading Applications:
    --------------------
    • STRATEGY SELECTION: High ratio → avoid mean reversion strategies
    • RISK MANAGEMENT: Reduce position sizes during high ratio periods
    • ENTRY TIMING: Enter mean reversion trades only when ratio < 0.9
    • EXIT SIGNAL: High ratio spikes often precede trend reversals

    Critical Insights:
    -----------------
    • A ratio > 1 suggests intra-bar volatility exceeded close-to-close movement
    • This indicates emotional/panic quant rather than rational price discovery
    • Consistently high ratios often precede major market turning points
    """
    # Convert to numpy for Numba function
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _, index = _to_numpy(close)

    # Parkinson volatility (numba function, returns numpy)
    parkinson_vol = _parkinson_volatility_nb(high_arr, low_arr, window)
    # Realized volatility (handles both types)
    realized_vol = realized_volatility(close, window, freq_minutes, annualize=True)

    # Get realized_vol as numpy array
    if isinstance(realized_vol, pd.Series):
        realized_vol = realized_vol.values

    # Compute ratio
    ratio = parkinson_vol / (realized_vol + 1e-9)

    return _to_output_type(ratio, index, name='volatility_ratio')


def volatility_of_volatility(high: ArrayLike, low: ArrayLike, window: int = 20, vov_window: int = 20) -> ArrayLike:
    """
    Volatility of Volatility (VoV): rolling std of Parkinson volatility.

    Measures the instability of the volatility regime itself. High VoV indicates
    the market is transitioning between volatility regimes (trend → chop → crisis).

    Formula:
        park_vol(t) = Parkinson(high, low, window)
        VoV(t) = std(park_vol[t-vov_window+1 : t+1])

    Parameters
    ----------
    high, low : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series)
    window : int, default=20
        Lookback for Parkinson base volatility.
    vov_window : int, default=20
        Rolling window for computing std of Parkinson series.

    Returns
    -------
    ArrayLike
        VoV series; first (window + vov_window - 2) values are NaN.

    Interpretation
    --------------
    - Low VoV  → Stable volatility regime (momentum or consolidation)
    - High VoV → Volatile-volatility regime (regime transitions, crises)
    """
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _, index = _to_numpy(high)

    park_vol = _parkinson_volatility_nb(high_arr, low_arr, window)
    vov = _rolling_std(park_vol, vov_window)

    return _to_output_type(vov, index, name=f'vov_{window}_{vov_window}')


def vov_normalized(high: ArrayLike, low: ArrayLike, window: int = 20,
                   vov_window: int = 20) -> ArrayLike:
    """
    VoV normalized by rolling mean of Parkinson volatility.

    Adjusts VoV for the current volatility level, making it comparable across
    different volatility regimes and asset classes.

    Formula:
        vov_norm(t) = VoV(t) / mean(park_vol[t-vov_window+1 : t+1])

    Parameters
    ----------
    high, low : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series)
    window : int, default=20
        Lookback for Parkinson base volatility.
    vov_window : int, default=20
        Rolling window for computing VoV and mean vol.

    Returns
    -------
    ArrayLike
        Normalized VoV; values typically in [0, 1] for normal markets.
    """
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _, index = _to_numpy(high)

    park_vol = _parkinson_volatility_nb(high_arr, low_arr, window)
    vov = _rolling_std(park_vol, vov_window)

    # Rolling mean of Parkinson over vov_window
    park_s = pd.Series(park_vol)
    mean_vol = park_s.rolling(vov_window).mean().values

    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(mean_vol > 1e-10, vov / mean_vol, np.nan)

    return _to_output_type(result, index, name=f'vov_norm_{window}_{vov_window}')


def volatility_term_structure(high: ArrayLike, low: ArrayLike, short_window: int = 5,
                               long_window: int = 20) -> ArrayLike:
    """
    Volatility Term Structure (VTS): ratio of short-term to long-term Parkinson vol.

    Captures the shape of the volatility curve across horizons. Values above 1
    indicate short-term vol > long-term vol (backwardation / stress). Values below
    1 indicate contango (calm, short-term vol is compressed vs long-term).

    Formula:
        VTS(t) = Parkinson(short_window)(t) / Parkinson(long_window)(t)

    Parameters
    ----------
    high, low : ArrayLike
        OHLC prices (1-D float64 arrays or pd.Series)
    short_window : int, default=5
        Short lookback window for Parkinson.
    long_window : int, default=20
        Long lookback window for Parkinson.

    Returns
    -------
    ArrayLike
        VTS ratio; first (long_window - 1) values are NaN.

    Interpretation
    --------------
    - VTS < 0.7  → Contango (short-term compressed, breakout potential)
    - VTS ≈ 1.0  → Flat term structure (balanced regime)
    - VTS > 1.3  → Backwardation (near-term stress, potential exhaustion)
    """
    high_arr, _ = _to_numpy(high)
    low_arr, _ = _to_numpy(low)
    _, index = _to_numpy(high)

    short_vol = _parkinson_volatility_nb(high_arr, low_arr, short_window)
    long_vol = _parkinson_volatility_nb(high_arr, low_arr, long_window)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(long_vol > 1e-10, short_vol / long_vol, np.nan)

    return _to_output_type(result, index, name=f'vts_{short_window}_{long_window}')


def gap_risk_ratio(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike,
                   window: int = 20) -> ArrayLike:
    """Yang-Zhang / Parkinson ratio — quantifies overnight gap risk.

    Values > 1.5 indicate elevated gap risk (significant overnight component).
    Values ≈ 1.0 indicate intraday and overnight volatility are balanced.
    """
    yz_vol = yang_zhang_volatility(open_, high, low, close, window=window)
    park_vol = parkinson_volatility(high, low, window=window)
    return _to_output_type(
        np.where(park_vol > 1e-10, yz_vol / (park_vol + 1e-10), np.nan),
        _to_numpy(high)[1], name=f'gap_risk_ratio_{window}')


def trend_quality_ratio(open_: ArrayLike, high: ArrayLike, low: ArrayLike, close: ArrayLike,
                        window: int = 20) -> ArrayLike:
    """Rogers-Satchell / Parkinson ratio — measures trend vs chop.

    Values < 0.8 → clean directional trend (low drift noise).
    Values > 1.2 → choppy / ranging market.
    """
    rs_vol = rogers_satchell_volatility(open_, high, low, close, window=window)
    park_vol = parkinson_volatility(high, low, window=window)
    return _to_output_type(
        np.where(park_vol > 1e-10, rs_vol / (park_vol + 1e-10), np.nan),
        _to_numpy(high)[1], name=f'trend_quality_ratio_{window}')


def overnight_ratio(high: ArrayLike, low: ArrayLike, close: ArrayLike,
                    window: int = 20) -> ArrayLike:
    """Rolling close-to-close vol / Parkinson ratio — intraday vs overnight balance.

    Rising ratio → overnight activity increasing relative to intraday range.
    """
    roll_vol = rolling_volatility(close, window=window)
    park_vol = parkinson_volatility(high, low, window=window)
    if isinstance(roll_vol, pd.Series):
        roll_vol = roll_vol.values
    if isinstance(park_vol, pd.Series):
        park_vol = park_vol.values
    _, index = _to_numpy(close)
    return _to_output_type(
        np.where(park_vol > 1e-10, roll_vol / (park_vol + 1e-10), np.nan),
        index, name=f'overnight_ratio_{window}')


def jump_detection(close: ArrayLike, window: int = 20, threshold_sigma: float = 3.0) -> ArrayLike:
    """Rolling z-score jump detector — binary flag for return outliers.

    Returns 1 when |z-score of log return| > threshold_sigma, else 0.
    Complements bipower-variation jump detection (different sensitivity).

    Values:
        1 → jump bar (widen stops, reduce size)
        0 → normal bar
    """
    if isinstance(close, np.ndarray):
        close = pd.Series(close)
    returns = np.log(close / close.shift(1))
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    z = (returns - rolling_mean) / rolling_std.where(rolling_std > 0)
    result = (np.abs(z) > threshold_sigma).astype(int)
    _, index = _to_numpy(close)
    return _to_output_type(result.values, index, name=f'jump_{window}_{threshold_sigma}')
