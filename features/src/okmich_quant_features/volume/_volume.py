from enum import StrEnum

import numpy as np
import pandas as pd
import talib
from numba import njit, jit


def discretize_volume(volumes, bins=4):
    """
    Discretize a volume series into specified quantile-based bins.

    CRITICAL FOR: Market Facilitation Index (MFI) analysis and volume-based regime classification.

    Parameters:
    -----------
    volumes : array-like
        List or array of volume data
    bins : int, default=4
        Number of bins to create (typically 4 for quartile analysis)

    Returns:
    --------
    tuple: (bin_assignments, bin_map)
        - bin_assignments: List of bin IDs (0 to bins-1) for each volume value
        - bin_map: Dictionary mapping bin ID to (lower_bound, upper_bound) ranges

    Significance and Usage:
    ----------------------
    • VOLUME REGIMES: Creates discrete volume levels for pattern recognition
    • MFI ANALYSIS: Essential for proper Market Facilitation Index interpretation
    • NON-PARAMETRIC: Adapts to any volume distribution without assumptions
    • ROBUST: Handles outliers and changing volume characteristics over time

    Trading Applications:
    --------------------
    • VOLUME CLUSTERING: Identify institutional vs retail volume levels
    • BREAKOUT CONFIRMATION: High volume bins confirm genuine breakouts
    • EXHAUSTION DETECTION: Extreme volume bins suggest potential reversals
    • LIQUIDITY ASSESSMENT: Map volume to liquidity conditions in real-time

    Optimal Usage:
    -------------
    • Use 4 bins (quartiles) for most applications
    • Recalculate bin_map periodically to adapt to changing market conditions
    • Combine with price action for volume-price confirmation
    • Monitor transitions between volume bins for early signals

    Example:
    --------
    >>> bin_clazz, binMap = discretize_volume(volumes, bins=4)
    >>> # binMap: {0: (0, Q1), 1: (Q1, Q2), 2: (Q2, Q3), 3: (Q3, inf)}
    """
    volume = np.array(volumes)

    # Calculate bin edges using quartiles (or other method)
    bin_edges = np.histogram_bin_edges(volume, bins=bins)

    # Assign volumes to bins (0 to bins-1)
    bin_assignments = np.digitize(volume, bin_edges, right=True) - 1
    bin_assignments = np.clip(bin_assignments, 0, bins - 1)  # Ensure valid bin IDs

    # Create bin map: {bin_id: (lower_bound, upper_bound)}
    bin_map = {}
    for i in range(bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1] if i < len(bin_edges) - 1 else np.inf
        bin_map[i] = (lower_bound, upper_bound)

    return bin_assignments.tolist(), bin_map


@njit(fastmath=True, cache=True)
def _market_facilitation_index(high_prices, low_prices, volumes, bin_map):
    """
    [Internal JIT-optimized function] Calculate Market Facilitation Index with volume binning.

    Core computation engine for MFI - optimized with Numba for speed.
    """
    market_fac_index = np.where(volumes != 0, (high_prices - low_prices) / volumes, 0)

    if bin_map is None:
        return market_fac_index, None

    volume_bins = np.zeros(len(volumes), dtype=int)
    for i, vol in enumerate(volumes):
        for bin_id, (lower, upper) in bin_map.items():
            if lower <= vol < upper:
                volume_bins[i] = bin_id
                break

    return market_fac_index, volume_bins


def volume_momentum(volume_series, fast_window=5, slow_window=20):
    """
    Volume Momentum: Ratio of fast SMA to slow SMA of volume.

    MEASURES: Acceleration or deceleration of volume activity relative to recent history.

    Parameters:
    -----------
    volume_series : pd.Series
        Time series of volume data
    fast_window : int, default=5
        Short-term window for recent volume momentum
    slow_window : int, default=20
        Long-term window for baseline volume activity

    Returns:
    --------
    pd.Series
        Volume momentum ratio (values > 1 indicate increasing momentum)

    Interpretation Guide:
    --------------------
    • RATIO > 1.2: Strong volume acceleration (watch for breakouts)
    • RATIO 1.0-1.2: Moderate volume increase (trend continuation)
    • RATIO 0.8-1.0: Normal volume activity (consolidation)
    • RATIO < 0.8: Volume deceleration (exhaustion potential)

    Significance and Usage:
    ----------------------
    • EARLY WARNING: Often leads price momentum by 1-2 periods
    • BREAKOUT CONFIRMATION: Validates genuine breakouts from false ones
    • DIVERGENCE: Price making new highs with declining volume momentum = caution
    • REGIME SHIFTS: Spikes often precede volatility regime changes

    Trading Applications:
    --------------------
    • ENTRY TIMING: Enter on volume momentum confirmation
    • EXIT SIGNALS: Declining volume momentum suggests trend weakness
    • FILTERING: Filter trades based on volume momentum threshold
    • SCALPING: High volume momentum periods better for intraday quant
    """
    sma_fast = volume_series.rolling(window=fast_window, min_periods=1).mean()
    sma_slow = volume_series.rolling(window=slow_window, min_periods=1).mean()
    return sma_fast / sma_slow.replace(0, 1e-10)  # Avoid division by zero


def volume_volatility_correlation(volume_series, volatility_series, window=20):
    """
    Rolling correlation between volume and volatility.

    REVEALS: The relationship between quant activity and price movement magnitude.

    Parameters:
    -----------
    volume_series : pd.Series
        Time series of volume data
    volatility_series : pd.Series
        Time series of volatility (e.g., from realized_volatility())
    window : int, default=20
        Rolling window for correlation calculation

    Returns:
    --------
    pd.Series
        Rolling correlation coefficient between volume and volatility

    Interpretation Guide:
    --------------------
    • CORRELATION > 0.7: Strong positive relationship (healthy trends)
    • CORRELATION 0.3-0.7: Moderate relationship (normal market functioning)
    • CORRELATION 0.0-0.3: Weak relationship (choppy, inefficient markets)
    • CORRELATION < 0.0: Negative relationship (potential manipulation/exhaustion)

    Significance and Usage:
    ----------------------
    • MARKET QUALITY: High positive correlation = efficient price discovery
    • TREND HEALTH: Sustained high correlation suggests genuine trend
    • EXHAUSTION: Declining correlation often precedes reversals
    • EVENT DETECTION: Correlation spikes around news/events

    Trading Applications:
    --------------------
    • TREND CONFIRMATION: Only trade trends with high volume-volatility correlation
    • REVERSAL WARNING: Negative correlation suggests trend exhaustion
    • REGIME CLASSIFICATION: Different correlations indicate different market regimes
    • RISK ASSESSMENT: Low correlation markets = higher slippage and worse fills
    """

    @jit(nopython=True)
    def rolling_corr_numba(x, y, _window):
        n = len(x)
        result = np.full(n, np.nan)
        for i in range(_window - 1, n):
            x_window = x[i - _window + 1 : i + 1]
            y_window = y[i - _window + 1 : i + 1]
            result[i] = np.corrcoef(x_window, y_window)[0, 1]
        return result

    vol_array = volume_series.values.astype(np.float64)
    vol_ty_array = (
        volatility_series.values.astype(np.float64)
        if isinstance(volatility_series, pd.Series)
        else volatility_series
    )

    return pd.Series(
        rolling_corr_numba(vol_array, vol_ty_array, window), index=volume_series.index
    )


def volume_profile(volume_series, window=20):
    """
    Rolling standard deviation of volume - measures dispersion of volume activity.

    IDENTIFIES: Periods of volume concentration vs dispersion.

    Parameters:
    -----------
    volume_series : pd.Series
        Time series of volume data
    window : int, default=20
        Rolling window for standard deviation calculation

    Returns:
    --------
    pd.Series
        Rolling standard deviation of volume

    Interpretation Guide:
    --------------------
    • HIGH VALUES: Volume activity is dispersed (accumulation/distribution)
    • LOW VALUES: Volume activity is concentrated (consolidation/indecision)
    • RISING PROFILE: Increasing dispersion often precedes breakouts
    • FALLING PROFILE: Decreasing dispersion suggests range-bound conditions

    Significance and Usage:
    ----------------------
    • ACCUMULATION: High volume profile during sideways movement = accumulation
    • DISTRIBUTION: High volume profile at tops = distribution
    • BREAKOUT PREDICTION: Rising profile often precedes volatility expansion
    • RANGE IDENTIFICATION: Low profile suggests quant range boundaries

    Trading Applications:
    --------------------
    • BREAKOUT ANTICIPATION: Enter before breakouts when profile is rising
    • RANGE TRADING: Use low profile periods for mean reversion strategies
    • POSITION BUILDING: High profile at support = accumulation (go long)
    • POSITION REDUCING: High profile at resistance = distribution (reduce long)
    """
    return volume_series.rolling(window=window, min_periods=1).std()


@jit(nopython=True)
def volume_return_divergence_numba(returns, volume, volatility, epsilon=1e-10):
    """
    [Internal JIT-optimized function] Calculate volume-return divergence.
    """
    n = len(returns)
    result = np.full(n, np.nan)
    for i in range(n):
        if (
            not np.isnan(returns[i])
            and not np.isnan(volume[i])
            and not np.isnan(volatility[i])
        ):
            result[i] = (returns[i] * volume[i]) / (volatility[i] + epsilon)
    return result


def volume_return_divergence(return_series, volume_series, volatility_series, epsilon=1e-10):
    """
    Volume-Return Divergence: Measures the efficiency of returns per unit volume and volatility.

    CAPTURES: Whether price movement is supported by appropriate volume and volatility.

    Parameters:
    -----------
    return_series : pd.Series
        Time series of returns
    volume_series : pd.Series
        Time series of volume data
    volatility_series : pd.Series
        Time series of volatility data
    epsilon : float, default=1e-10
        Small value to avoid division by zero

    Returns:
    --------
    pd.Series
        Volume-return divergence values

    Interpretation Guide:
    --------------------
    • STRONGLY POSITIVE: Efficient bullish movement (high return, good volume/volatility)
    • STRONGLY NEGATIVE: Efficient bearish movement
    • WEAK POSITIVE/NEGATIVE: Inefficient movement (potential reversal)
    • DIVERGENCE: Price making new highs with declining divergence = exhaustion

    Significance and Usage:
    ----------------------
    • EFFICIENCY SCORE: Measures quality of price movement
    • EXHAUSTION DETECTION: Declining divergence at extremes suggests reversals
    • TRAP IDENTIFICATION: Strong moves with weak divergence often trap participants
    • MOMENTUM CONFIRMATION: High divergence confirms genuine momentum

    Trading Applications:
    --------------------
    • REVERSAL TRADING: Fade moves with deteriorating divergence
    • TREND FOLLOWING: Only ride trends with sustained high divergence
    • RISK MANAGEMENT: Reduce exposure during low divergence periods
    • ENTRY TIMING: Enter on divergence improvement after consolidation
    """
    returns = return_series.values.astype(np.float64)
    volume = volume_series.values.astype(np.float64)
    volatility = (
        volatility_series.values.astype(np.float64)
        if isinstance(volatility_series, pd.Series)
        else volatility_series
    )

    result = volume_return_divergence_numba(returns, volume, volatility, epsilon)
    return pd.Series(result, index=return_series.index)


def abnormal_volume(volume_series, window=50, z_threshold=2.0):
    """
    Z-score of current volume relative to historical average - identifies statistically unusual volume.

    DETECTS: Potential news events, institutional activity, or unusual market behavior.

    Parameters:
    -----------
    volume_series : pd.Series
        Time series of volume data
    window : int, default=50
        Lookback window for historical average calculation
    z_threshold : float, default=2.0
        Threshold for considering volume abnormal (standard deviations)

    Returns:
    --------
    pd.Series
        Z-scores of volume relative to historical average

    Interpretation Guide:
    --------------------
    • Z-SCORE > 2.0: Statistically significant high volume (98th percentile)
    • Z-SCORE < -2.0: Statistically significant low volume (2nd percentile)
    • Z-SCORE > 3.0: Extreme volume event (99.9th percentile)
    • Z-SCORE 1.0-2.0: Moderately unusual volume

    Significance and Usage:
    ----------------------
    • EVENT DETECTION: Identifies news, earnings, or event-driven volume
    • INSTITUTIONAL FLOW: Extreme volume often indicates large player activity
    • GAP PREDICTION: High abnormal volume often precedes gap moves
    • EXHAUSTION: Climax volume often marks reversal points

    Trading Applications:
    --------------------
    • NEWS TRADING: Trade in direction of abnormal volume spikes
    • GAP PLAY: Use after-hours abnormal volume to predict next day gaps
    • REVERSAL SIGNALS: Climax volume at extremes suggests exhaustion
    • FILTERING: Avoid quant during abnormally low volume (slippage risk)
    """
    rolling_mean = volume_series.rolling(window=window, min_periods=1).mean()
    rolling_std = (
        volume_series.rolling(window=window, min_periods=1).std().replace(0, 1e-10)
    )

    z_score = (volume_series - rolling_mean) / rolling_std
    return z_score


class VolumePhase(StrEnum):
    WEEKLY = "weekly"
    DAILY = "daily"


def tick_volume_phase_zscore(
    volume: pd.Series, min_periods: int = 20, phase: str | VolumePhase = VolumePhase.WEEKLY
) -> pd.Series:
    """
    Z-score of tick volume against an expanding phase baseline.

    For each bar, computes how unusual the current volume is relative to
    all prior bars that fell in the same phase bucket:
    - phase='weekly': minute-of-week bucket (10,080 buckets).
    - phase='daily': minute-of-day bucket (1,440 buckets).

    Fully causal: only uses data up to and including the current bar.

    Parameters
    ----------
    volume : pd.Series
        Tick volume with a DatetimeIndex (tz-aware or tz-naive).
    min_periods : int, default=20
        Minimum number of same-bucket observations before producing a
        z-score. Bars whose bucket has fewer observations return NaN.
    phase : {'weekly', 'daily'}, default='weekly'
        Phase granularity used for bucketing.

    Returns
    -------
    pd.Series
        Z-score. Positive = higher-than-typical volume for this time slot.
        Negative = lower-than-typical. NaN until min_periods observations
        accumulate in the bar's phase bucket.

    Raises
    ------
    ValueError
        If volume does not have a DatetimeIndex, if the index is not monotonic
        increasing, if min_periods < 1, or if phase is invalid.
    """
    if not isinstance(volume.index, pd.DatetimeIndex):
        raise ValueError("volume must have a DatetimeIndex for phase bucketing")
    if not volume.index.is_monotonic_increasing:
        raise ValueError("volume index must be monotonic increasing for causal phase bucketing")
    if min_periods < 1:
        raise ValueError("min_periods must be >= 1")

    try:
        phase_enum = VolumePhase(str(phase).lower())
    except ValueError as exc:
        raise ValueError("phase must be one of {'weekly', 'daily'}") from exc

    if phase_enum == VolumePhase.WEEKLY:
        phase_bucket = volume.index.dayofweek * 1440 + volume.index.hour * 60 + volume.index.minute
    else:
        phase_bucket = volume.index.hour * 60 + volume.index.minute

    tmp = pd.DataFrame({"vol": volume.values.astype(float), "phase_bucket": phase_bucket}, index=volume.index)
    expanding_mean = tmp.groupby("phase_bucket", sort=False)["vol"].transform(
        lambda x: x.expanding(min_periods=min_periods).mean()
    )
    expanding_std = tmp.groupby("phase_bucket", sort=False)["vol"].transform(
        lambda x: x.expanding(min_periods=min_periods).std()
    )

    z = (tmp["vol"] - expanding_mean) / (expanding_std + 1e-10)
    return pd.Series(z.values, index=volume.index, name="tick_vol_phase_zscore")


def tick_volume_how_zscore(volume: pd.Series, min_periods: int = 20) -> pd.Series:
    """Backward-compatible wrapper for weekly phase bucketing."""
    z = tick_volume_phase_zscore(volume=volume, min_periods=min_periods, phase=VolumePhase.WEEKLY)
    return pd.Series(z.values, index=volume.index, name="tick_vol_how_zscore")


def ad(df: pd.DataFrame, method: str = "williams", vol_col: str = "tick_volume", window: int = 24,) -> tuple:
    """
    Accumulation/Distribution Line with z-score and percentile rank normalization.

    CAPTURES: The relationship between price movement and volume to identify accumulation or distribution,
    with statistical normalization for regime-independent analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain columns ['open', 'high', 'low', 'close', vol_col]
    method : str, default="williams"
        Calculation method:
        - 'williams': Classic formula (2C-H-L)/(H-L)
        - 'mt': MT-style formula (C-O)/(H-L)
    vol_col : str, default="tick_volume"
        Name of the volume column
    window : int, default=24
        Rolling window for z-score and percentile rank calculations

    Returns:
    --------
    tuple: (ad_line, ad_zscore, ad_pct_rank)
        - ad_line: Cumulative A/D line values
        - ad_zscore: Rolling z-score of A/D line (clipped to [-5, 5])
        - ad_pct_rank: Rolling percentile rank of A/D line (0 to 1)

    Interpretation Guide:
    --------------------
    • A/D LINE:
      - Rising: Accumulation - buying pressure exceeds selling
      - Falling: Distribution - selling pressure exceeds buying
    • A/D Z-SCORE:
      - > 2.0: Statistically high accumulation
      - < -2.0: Statistically high distribution
      - Normalized for cross-market comparison
    • A/D PERCENTILE:
      - > 0.8: Strong accumulation phase (80th percentile)
      - < 0.2: Strong distribution phase (20th percentile)

    Significance and Usage:
    ----------------------
    • SMART MONEY TRACKING: Identifies institutional accumulation/distribution
    • TREND CONFIRMATION: A/D should move with price in healthy trends
    • DIVERGENCE DETECTION: Early warning of potential reversals
    • BREAKOUT VALIDATION: Confirms genuine breakouts vs false moves
    • REGIME NORMALIZATION: Z-score and percentile enable cross-asset comparison

    Trading Applications:
    --------------------
    • ENTRY SIGNALS: Enter long when A/D percentile > 0.7
    • EXIT SIGNALS: Exit when A/D z-score shows extreme divergence
    • BREAKOUT CONFIRMATION: Only trade breakouts with rising A/D
    • MEAN REVERSION: Trade against extreme A/D z-scores (|z| > 3)

    Method Comparison:
    -----------------
    • WILLIAMS: More sensitive to closing price relative to range
    • MT: Focuses on close vs open (intrabar direction)
    """
    h, l, c, o, v = df["high"], df["low"], df["close"], df["open"], df[vol_col]
    rng = h - l
    rng = rng.where(rng != 0, np.nan)  # guard divide-by-zero

    if method.lower() == "williams":
        mult = (2 * c - h - l) / rng
    else:  # 'mt'
        mult = (c - o) / rng

    # Compute A/D line
    ad_line = (mult * v).fillna(0).cumsum()
    ad_line = ad_line.rename(f"AD_{method.lower()}")

    # Compute rolling z-score
    roll = ad_line.rolling(window, min_periods=window)
    ad_zscore = ((ad_line - roll.mean()) / roll.std()).clip(-5, 5)
    ad_zscore = ad_zscore.rename(f"AD_{method.lower()}_zscore")

    # Compute rolling percentile rank
    ad_pct_rank = (
        ad_line.rolling(window, min_periods=window)
        .apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
        .clip(0, 1.0)
    )
    ad_pct_rank = ad_pct_rank.rename(f"AD_{method.lower()}_pct")

    return ad_line, ad_zscore, ad_pct_rank


def volume_weighted_rate_of_change(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Volume-weighted Rate of Change (VWROC).

    Formula:
        VWROC = Σ(volume_t * close_t) / Σ(volume_t-n * close_t-n) - 1

    High Impact Use:
        - Adjusts ROC by participation strength (volume).
        - Filters false breakouts with weak volume confirmation.

    Best for:
        - Equities, Indices, Crypto.
    """
    vw_price = close * volume
    return (vw_price / vw_price.shift(period)) - 1


def vwap_adjusted_roc(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute price ROC adjusted relative to VWAP shift.

    High Impact Use:
        - Tracks price movement relative to changing VWAP baseline.
        - Highlights institutional buying or selling zones.

    Best for:
        - Equities and FX (where VWAP reflects trade-weighted cost basis).
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / (volume.cumsum() + 1e-8)
    roc = talib.ROC(close, timeperiod=period)
    vwap_shift = vwap.diff(period)
    return roc - vwap_shift


def vwap_dev_momentum(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute VWAP deviation momentum.

    Formula:
        VWAP deviation = (close - VWAP) / VWAP
        Then compute rolling change in deviation.

    High Impact Use:
        - Captures how quickly price deviates from cost basis (VWAP).
        - Measures crowding or exit momentum near average cost.

    Best for:
        - FX, Equities, and Crypto.
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / (volume.cumsum() + 1e-8)
    vwap_dev = (close - vwap) / (vwap + 1e-8)
    return vwap_dev.diff(period)


def volume_surge_rate_of_change(close: pd.Series, volume: pd.Series, period: int = 14, surge_mult: float = 1.5) -> pd.Series:
    """
    Compute ROC only during volume surge periods.

    High Impact Use:
        - Detects price momentum bursts confirmed by strong participation.
        - Filters noise from low-volume environments.

    Best for:
        - FX, Crypto, Futures (liquidity-sensitive assets).
    """
    roc = pd.Series(talib.ROC(close, timeperiod=period), index=close.index)
    vol_avg = volume.rolling(period).mean()
    vol_spike = volume > vol_avg * surge_mult
    return roc.where(vol_spike, 0.0)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Money Flow Index (MFI) using TA-Lib.

    High Impact Use:
        - Combines price and volume into flow oscillator.
        - Detects momentum exhaustion or confirmation under participation.

    Best for:
        - Equities, FX, and Crypto.
    """
    mfi_values = talib.MFI(high, low, close, volume, timeperiod=period)
    return pd.Series(mfi_values, index=close.index)


def binned_mfi_delta(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                     bins: int = 5, period: int = 14) -> pd.Series:
    """
    Compute ΔMFI within each volume bin.

    High Impact Use:
        - Measures change in flow intensity conditioned on liquidity.
        - Identifies when momentum shifts occur under high/low volume regimes.

    Best for:
        - FX, Crypto.
    """
    mfi_vals = mfi(high, low, close, volume, period=period)
    vol_bin = pd.qcut(
        volume.rank(method="first"), bins, labels=False, duplicates="drop"
    )
    return mfi_vals.groupby(vol_bin).diff()


def mfi_volume_bin_ratio(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                         bins: int = 5, period: int = 14) -> pd.Series:
    """
    Compute ratio of current MFI to average MFI in its volume bin.

    High Impact Use:
        - Detects abnormal flow strength relative to liquidity regime.
        - Highlights hidden institutional footprints.

    Best for:
        - Equities, FX, and Futures.
    """
    mfi_vals = mfi(high, low, close, volume, period=period)
    vol_bin = pd.qcut(
        volume.rank(method="first"), bins, labels=False, duplicates="drop"
    )
    avg_bin_mfi = mfi_vals.groupby(vol_bin).transform("mean")
    return mfi_vals / (avg_bin_mfi + 1e-8)


def categorized_mfi_trend(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, bins: int = 5,
                          period: int = 14) -> pd.Series:
    """
    Categorize directional MFI trends (+1 for uptrend, -1 for downtrend) within each volume bin.

    High Impact Use:
        - Encodes flow direction relative to liquidity condition.
        - Useful for discrete regime labeling or model inputs.

    Best for:
        - FX and Crypto intraday models.
    """
    mfi_vals = mfi(high, low, close, volume, period=period)
    vol_bin = pd.qcut(
        volume.rank(method="first"), bins, labels=False, duplicates="drop"
    )
    grouped_delta = mfi_vals.groupby(vol_bin).diff()
    return np.sign(grouped_delta).fillna(0)


def volume_bin_mfi_persistence(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, bins: int = 5,
                               period: int = 14, lag: int = 5) -> pd.Series:
    """
    Compute autocorrelation (persistence) of MFI changes within each volume bin.

    High Impact Use:
        - Quantifies consistency of flow bias under similar liquidity.
        - Detects stable flow regimes (accumulation/distribution zones).

    Best for:
        - FX, Futures, and Commodities.
    """
    mfi_vals = mfi(high, low, close, volume, period=period)
    vol_bin = pd.qcut(
        volume.rank(method="first"), bins, labels=False, duplicates="drop"
    )

    def rolling_autocorr(x):
        return x.rolling(lag).apply(lambda y: y.autocorr(), raw=False)

    return mfi_vals.groupby(vol_bin, group_keys=False).apply(rolling_autocorr)

