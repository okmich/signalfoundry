"""
Liquidity and Market Microstructure Features

This module provides market microstructure metrics that estimate transaction costs, liquidity, and market quality from
OHLC price data without requiring trade/quote data.

Key Features:
    - Corwin-Schultz spread estimator (2012)
    - Roll spread estimator (1984)
    - Effective spread proxies
    - Market quality metrics

These spread estimators will be used by advanced microstructure features:
    - Amihud Illiquidity Ratio
    - Liquidity Drought Index (LDI)
    - Realized Liquidity Premium
    - Spread-Volatility Elasticity

References:
    - Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices." Journal of Finance, 67(2), 719-760.
    - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market." Journal of Finance, 39(4), 1127-1139.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from numba import njit
from ._primitives import beta_clv, _check_index_aligned


def corwin_schultz_spread(high: pd.Series, low: pd.Series, close: Optional[pd.Series] = None, window: int = 2,
                          return_alpha: bool = False) -> pd.Series | Tuple[pd.Series, pd.Series]:
    """
    Estimate bid-ask spread using the Corwin-Schultz (2012) high-low spread estimator.

    This method exploits the fact that observed high-low ranges contain information about
    both true volatility and bid-ask bounce. It separates these components to estimate
    the proportional bid-ask spread without needing actual trade data.

    The estimator uses the paper's exact beta and gamma terms:
    - β_i = [ln(H_i/L_i)]² + [ln(H_{i-1}/L_{i-1})]²  (sum of adjacent squared HL ratios)
    - γ_i = [ln(max(H_i, H_{i-1}) / min(L_i, L_{i-1}))]²  (two-period squared range)
    - α   = (√(2β) − √β) / (3 − 2√2) − √(γ / (3 − 2√2))
    - S   = 2(e^α − 1) / (1 + e^α), clipped to [0, 1]

    Parameters
    ----------
    high : pd.Series
        High prices for each period
    low : pd.Series
        Low prices for each period
    close : pd.Series, optional
        Kept for backward compatibility. Not used in the computation.
    window : int, default=2
        Rolling window for the estimator. Original paper uses 2.
        Larger windows provide smoother estimates but less responsive to changes.
    return_alpha : bool, default=False
        If True, returns (spread, alpha) tuple where alpha is the intermediate
        spread component. Useful for diagnostics.

    Returns
    -------
    spread : pd.Series
        Proportional bid-ask spread (0 to 1 scale)
        - 0.01 = 1% spread (100 bps)
        - 0.001 = 0.1% spread (10 bps)
        Higher values indicate lower liquidity (wider spreads)
    alpha : pd.Series (if return_alpha=True)
        Intermediate spread component used in the calculation

    Notes
    -----
    Interpretation:
        - **0.0001-0.001** (1-10 bps): Highly liquid (major FX pairs, large-cap stocks)
        - **0.001-0.005** (10-50 bps): Liquid (mid-cap stocks, active futures)
        - **0.005-0.01** (50-100 bps): Moderate liquidity (small-cap stocks)
        - **>0.01** (>100 bps): Low liquidity (illiquid instruments, thin markets)

    The estimator assumes:
        - Efficient markets (prices follow geometric Brownian motion)
        - Constant spread within the estimation window
        - No overnight gaps (for daily data)

    References
    ----------
    Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads
    from Daily High and Low Prices." Journal of Finance, 67(2), 719-760.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    _check_index_aligned(high, low)

    # Step 1: Compute squared log high-low ratios
    hl = np.log(high / low)
    high_low_sq = hl ** 2

    # Step 2: Beta — expected sum of two adjacent squared HL ratios (paper's formula)
    beta = (high_low_sq + high_low_sq.shift(1)).rolling(window).mean()

    # Step 3: Gamma — squared two-period range (paper's actual gamma term)
    two_period_high = high.combine(high.shift(1), max)
    two_period_low = low.combine(low.shift(1), min)
    gamma = (np.log(two_period_high / two_period_low) ** 2).rolling(window).mean()

    # Step 4: Alpha — isolates the spread component
    sqrt_2 = np.sqrt(2)
    denominator = 3 - 2 * sqrt_2

    # Protect sqrt from negative values (can occur with noisy estimates)
    beta_safe = beta.clip(lower=0)
    gamma_safe = gamma.clip(lower=0)

    alpha = (np.sqrt(2 * beta_safe) - np.sqrt(beta_safe)) / denominator - np.sqrt(gamma_safe / denominator)

    # Step 5: Transform alpha to proportional spread (0 to 1)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Negative spreads are theoretically impossible, clip to 0
    spread = spread.clip(lower=0)

    if return_alpha:
        return spread, alpha
    return spread


def roll_spread(close: pd.Series, window: int = 20, min_periods: Optional[int] = None) -> pd.Series:
    """
    Estimate effective bid-ask spread using Roll's (1984) implicit spread estimator.

    Roll's estimator uses the negative serial covariance of price changes to infer
    the bid-ask spread. The key insight is that in an efficient market, bid-ask
    bounce creates negative autocorrelation in returns.

    Returns a proportional spread (divided by rolling mean close) so it is
    comparable to Corwin-Schultz and scale-invariant across price levels.

    Parameters
    ----------
    close : pd.Series
        Close prices
    window : int, default=20
        Rolling window for covariance estimation
    min_periods : int, optional
        Minimum number of observations required. Defaults to window//2.

    Returns
    -------
    spread : pd.Series
        Estimated proportional bid-ask spread (dimensionless ratio, 0 to ~1 scale)

    Notes
    -----
    The Roll estimator assumes:
        - Efficient markets (prices follow random walk + bid-ask bounce)
        - Constant spread within the estimation window
        - No autocorrelation in true price changes (only bid-ask bounce)

    When autocorrelation is positive (momentum exists), the estimator returns NaN
    since the model assumptions are violated.

    References
    ----------
    Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread
    in an Efficient Market." Journal of Finance, 39(4), 1127-1139.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if min_periods is None:
        min_periods = max(window // 2, 5)

    # Compute price changes
    price_change = close.diff()

    # Compute rolling covariance of consecutive price changes
    # Cov(ΔP_t, ΔP_{t-1}) = -spread²/4 in Roll's model
    cov = price_change.rolling(window, min_periods=min_periods).cov(
        price_change.shift(1)
    )

    # Absolute spread = 2 * sqrt(-cov); proportional = divide by rolling mean close
    spread_abs = 2 * np.sqrt(-cov.clip(upper=0))
    spread = spread_abs / close.rolling(window, min_periods=1).mean()

    return spread


def effective_tick_ratio(high: pd.Series, low: pd.Series, tick_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Estimate effective spread using volume-weighted tick ratio.

    This simple proxy estimates transaction costs by computing the average
    high-low range per unit of volume. Lower values indicate better liquidity.

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    tick_volume : pd.Series
        Tick volume (number of price changes) or trade count
    window : int, default=20
        Rolling window for averaging

    Returns
    -------
    ratio : pd.Series
        Volume-weighted tick ratio. Lower values = better liquidity.

    Notes
    -----
    This is a simplified liquidity proxy that doesn't make strong assumptions
    about market efficiency. It's useful for:
        - Quick liquidity screening
        - Comparing liquidity across similar instruments
        - Detecting sudden liquidity dry-ups

    Unlike Corwin-Schultz or Roll, this metric is not a true spread estimate
    but rather a relative liquidity indicator.

    Examples
    --------
    >>> # Calculate effective tick ratio
    >>> etr = effective_tick_ratio(df['high'], df['low'], df['tick_volume'])
    >>>
    >>> # Normalize for comparison
    >>> etr_z = (etr - etr.rolling(100).mean()) / etr.rolling(100).std()
    >>> illiquid_periods = etr_z > 2  # More than 2 std above normal
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    range_val = high - low
    tick_vol_safe = tick_volume.where(tick_volume > 0, np.nan)
    ratio = (range_val / tick_vol_safe).rolling(window).mean()
    return ratio


def liquidity_score(high: pd.Series, low: pd.Series, close: pd.Series, tick_volume: Optional[pd.Series] = None,
                    window: int = 20, method: str = "composite") -> pd.Series:
    """
    Compute composite liquidity score combining multiple estimators.

    This function combines Corwin-Schultz spread, Roll spread, and optionally
    effective tick ratio into a single normalized liquidity score.

    Parameters
    ----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    tick_volume : pd.Series, optional
        Tick volume for effective tick ratio calculation.
        If None, only uses Corwin-Schultz and Roll spreads.
    window : int, default=20
        Window for rolling calculations
    method : str, default='composite'
        Aggregation method:
        - 'composite': Average of all available estimators
        - 'corwin_schultz': Use only Corwin-Schultz
        - 'roll': Use only Roll
        - 'robust': Median of available estimators

    Returns
    -------
    score : pd.Series
        Negative liquidity score where higher values = better liquidity
        (negative spread). Multiply by -1 for intuitive direction.

    Examples
    --------
    >>> # Compute composite liquidity
    >>> liq = liquidity_score(df['high'], df['low'], df['close'], df['tick_volume'])
    >>>
    >>> # Higher values = better liquidity
    >>> liq_positive = -liq  # Flip sign for intuition
    >>>
    >>> # Use in regime detection
    >>> df['liquidity_regime'] = pd.cut(liq_positive, bins=3, labels=['low', 'medium', 'high'])
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if method not in ["composite", "corwin_schultz", "roll", "robust"]:
        raise ValueError(
            f"Invalid method: '{method}'. Choose from: composite, corwin_schultz, roll, robust"
        )

    def _z(s, w):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-10)

    estimators = []

    if method in ["composite", "corwin_schultz", "robust"]:
        cs_spread = corwin_schultz_spread(high, low, close, window=2)
        cs_smooth = cs_spread.rolling(window).mean()
        estimators.append(_z(cs_smooth, window))

    if method in ["composite", "roll", "robust"]:
        roll_sp = roll_spread(close, window=window)
        estimators.append(_z(roll_sp, window))

    if tick_volume is not None and method in ["composite", "robust"]:
        etr = effective_tick_ratio(high, low, tick_volume, window=window)
        estimators.append(_z(etr, window))

    # Aggregate
    if method == "robust":
        # Use median for robustness
        score = pd.concat(estimators, axis=1).median(axis=1)
    else:
        # Use mean
        score = pd.concat(estimators, axis=1).mean(axis=1)

    # Return negative spread (higher = better liquidity)
    return -score


# Alias for backward compatibility and convenience
def cs_spread(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Shorthand alias for corwin_schultz_spread with default parameters.

    See corwin_schultz_spread() for full documentation.
    """
    return corwin_schultz_spread(high, low, close)


# --------------------------------------------------------------------------- #
# Amihud Illiquidity                                                         #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _amihud_illiquidity_kernel(returns: np.ndarray, volume: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of |r| / (V × C)."""
    n = len(returns)
    illiq = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        s = 0.0
        count = 0
        for j in range(i - window + 1, i + 1):
            if np.isnan(returns[j]) or volume[j] <= 0.0 or close[j] <= 0.0:
                continue
            s += np.abs(returns[j]) / (volume[j] * close[j])
            count += 1
        if count >= window // 2:
            illiq[i] = s / count

    return illiq


def amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 20, log_transform: bool = True) -> pd.Series:
    """
    Amihud Illiquidity Ratio — price impact per dollar of volume.

    A classic measure of market illiquidity (Amihud, 2002).  Higher values
    indicate that a given dollar of trading causes a larger price move —
    i.e., the market is thin and orders are expensive to execute.

    Formula:
        ILLIQ(t) = mean(|r(i)| / (V(i) × C(i)))   over window

    where r = log-return, V = volume, C = close price.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume (tick or real; dollar volume works but changes the scale).
    window : int, default=20
        Rolling window over which to average the per-bar illiquidity.
    log_transform : bool, default=True
        If True, apply log(ILLIQ + 1e-10) for better distributional
        properties (ILLIQ is right-skewed; log normalises it).

    Returns
    -------
    illiq : pd.Series
        Illiquidity values.  In raw form, very small numbers; log-transformed
        values are more manageable for ML features.
        - Rising ILLIQ : market thinning, transactions becoming costly
        - Falling ILLIQ : deepening liquidity, cheaper to transact

    Notes
    -----
    With tick_volume instead of real volume the absolute scale changes, but
    relative comparisons (z-scores, regime detection) remain valid.

    References
    ----------
    Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and
    time-series effects." Journal of Financial Markets, 5(1), 31-56.
    """
    returns = np.log(close / close.shift(1)).values
    c = close.values.astype(np.float64)
    v = volume.values.astype(np.float64)

    illiq = _amihud_illiquidity_kernel(returns, v, c, window)

    if log_transform:
        illiq = np.log(illiq + 1e-10)

    return pd.Series(illiq, index=close.index, name='amihud_illiq')


# --------------------------------------------------------------------------- #
# Realized Liquidity Premium                                                 #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _realized_liquidity_premium_kernel(abs_returns: np.ndarray, spread_norm: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation between |r| and normalised spread."""
    n = len(abs_returns)
    rlp = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        r_sum = 0.0
        s_sum = 0.0
        count = 0

        for j in range(i - window + 1, i + 1):
            if np.isnan(abs_returns[j]) or np.isnan(spread_norm[j]):
                continue
            r_sum += abs_returns[j]
            s_sum += spread_norm[j]
            count += 1

        if count < window // 2:
            continue

        r_mean = r_sum / count
        s_mean = s_sum / count

        num = 0.0
        r_var = 0.0
        s_var = 0.0

        for j in range(i - window + 1, i + 1):
            if np.isnan(abs_returns[j]) or np.isnan(spread_norm[j]):
                continue
            r_dev = abs_returns[j] - r_mean
            s_dev = spread_norm[j] - s_mean
            num += r_dev * s_dev
            r_var += r_dev * r_dev
            s_var += s_dev * s_dev

        if r_var > 1e-20 and s_var > 1e-20:
            rlp[i] = num / np.sqrt(r_var * s_var)
        else:
            rlp[i] = 0.0

    return rlp


def realized_liquidity_premium(close: pd.Series, spread: pd.Series, mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Realized Liquidity Premium (RLP) — correlation between |returns| and spreads.

    Measures how tightly absolute price moves are coupled to transaction costs.
    When spreads widen during large moves, the market is pricing liquidity risk
    into every trade — a sign of fragility.

    Formula:
        RLP(t) = Corr(|r(i)|, S_norm(i))   over window
        S_norm = spread / mid_price

    Parameters
    ----------
    close : pd.Series
        Close prices.
    spread : pd.Series
        Bid-ask spread estimate (from corwin_schultz_spread or actual data).
    mid_price : pd.Series
        Mid-price for normalisation ((bid+ask)/2, or (high+low)/2, or close).
    window : int, default=20
        Rolling correlation window.

    Returns
    -------
    rlp : pd.Series
        Pearson correlation in [−1, +1].
        - RLP > 0.5 : market pricing liquidity into every move (fragile)
        - RLP < 0.2 : moves happen without spread blow-out (robust)
        - Rising RLP: increasing fragility — spreads widen on any move
    """
    returns = np.log(close / close.shift(1))
    abs_returns = np.abs(returns).values.astype(np.float64)
    spread_norm = (spread / mid_price).values.astype(np.float64)

    rlp = _realized_liquidity_premium_kernel(abs_returns, spread_norm, window)

    return pd.Series(rlp, index=close.index, name='realized_liquidity_premium')


# --------------------------------------------------------------------------- #
# Spread Z-Score                                                             #
# --------------------------------------------------------------------------- #

def spread_zscore(spread: pd.Series, mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Spread Z-Score — normalised measure of current spread vs recent history.

    Detects abnormal liquidity conditions without needing absolute spread levels,
    making it comparable across instruments and regimes.

    Formula:
        z_S = (S_norm − mean(S_norm, N)) / (std(S_norm, N) + ε)
        S_norm = spread / mid_price

    Parameters
    ----------
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    window : int, default=20
        Rolling window for mean and std.

    Returns
    -------
    z_spread : pd.Series
        Z-score of normalised spread.
        - |z| > 2 : abnormal spread (stress event)
        - |z| > 3 : extreme liquidity event
        - z > 2   : liquidity withdrawal (market makers pulling back)
        - z < −2  : unusually tight spread (competitive / low-vol environment)
    """
    spread_norm = spread / mid_price
    z = (spread_norm - spread_norm.rolling(window).mean()) / \
        (spread_norm.rolling(window).std() + 1e-10)
    return pd.Series(z.values, index=spread.index, name=f'spread_z_{window}')


# --------------------------------------------------------------------------- #
# Spread Expansion Momentum                                                  #
# --------------------------------------------------------------------------- #

def spread_expansion_momentum(spread: pd.Series, mid_price: pd.Series, ema_span: int = 5) -> pd.Series:
    """
    Spread Expansion Momentum (SEM) — smoothed rate of spread widening.

    An EMA of the percentage change in normalised spread.  A positive and
    rising SEM warns that market-makers are accelerating their retreat —
    typically a leading indicator of volatility events.

    Formula:
        ΔS(t) = (S_norm(t) − S_norm(t−1)) / S_norm(t−1)
        SEM(t) = EMA_k(ΔS)

    Parameters
    ----------
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    ema_span : int, default=5
        EMA span (half-life ≈ ema_span / 1.44 bars).

    Returns
    -------
    sem : pd.Series
        Smoothed spread rate of change.
        - Persistent positive SEM : market makers pulling back (danger)
        - SEM spike                : liquidity crisis onset
        - Negative SEM             : spreads tightening (improving conditions)
    """
    spread_norm = spread / mid_price
    delta_s = spread_norm.pct_change()
    sem = delta_s.ewm(span=ema_span, adjust=False).mean()
    return pd.Series(sem.values, index=spread.index, name=f'spread_expansion_momentum_{ema_span}')


# --------------------------------------------------------------------------- #
# Spread-Volume Ratio                                                        #
# --------------------------------------------------------------------------- #

def spread_volume_ratio(spread: pd.Series, mid_price: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Spread-Volume Ratio (SVR) — liquidity stress indicator.

    Combines spread and volume into a single stress signal.  A wide spread
    accompanied by low volume is the classic signature of a drying market;
    a tight spread with high volume is healthy two-sided flow.

    Formula:
        SVR(t) = S_norm(t) / log(V(t) + 1)

    Parameters
    ----------
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    volume : pd.Series
        Volume (tick or real).

    Returns
    -------
    svr : pd.Series
        Spread-volume ratio.
        - High SVR  : wide spread + low volume (stress / illiquidity)
        - Low SVR   : tight spread + high volume (healthy)
        - Rising SVR: deteriorating liquidity
    """
    spread_norm = spread / mid_price
    log_vol = np.log(volume.where(volume > 0, np.nan) + 1.0)
    svr = spread_norm / log_vol
    return pd.Series(svr.values, index=spread.index, name='spread_volume_ratio')


# --------------------------------------------------------------------------- #
# Liquidity Drought Index                                                    #
# --------------------------------------------------------------------------- #

def liquidity_drought_index(close: pd.Series, volume: pd.Series, spread: pd.Series, mid_price: pd.Series,
                            window: int = 20) -> pd.Series:
    """
    Liquidity Drought Index (LDI) — composite multi-dimensional liquidity signal.

    Combines four z-scored components into a single index that captures
    simultaneous deterioration across price impact, spread cost, and volume.


    Formula:
        LDI = (z_ILLIQ + z_S + z_SVR − z_DV) / 4

    where each component is z-scored over the same ``window``.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Volume (tick or real).
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    window : int, default=20
        Rolling window for all z-score calculations.

    Returns
    -------
    ldi : pd.Series
        Composite liquidity drought index.
        - LDI > 1.5 : drought conditions (elevated liquidity risk)
        - LDI > 2.0 : severe drought (disorderly price action likely)
        - LDI < −1.5: excess liquidity (crowded, spread compression)

    Components
    ----------
    1. z_ILLIQ : Amihud illiquidity z-score              (positive)
    2. z_S     : Spread z-score                          (positive)
    3. z_SVR   : Spread-volume ratio z-score             (positive)
    4. z_DV    : Dollar-volume z-score                   (negative — high
                 dollar volume improves liquidity)
    """
    # Component 1: Amihud illiquidity
    illiq = amihud_illiquidity(close, volume, window=window, log_transform=True)
    z_illiq = (illiq - illiq.rolling(window).mean()) / \
              (illiq.rolling(window).std() + 1e-10)

    # Component 2: Normalised spread z-score
    spread_norm = spread / mid_price
    z_spread = (spread_norm - spread_norm.rolling(window).mean()) / \
               (spread_norm.rolling(window).std() + 1e-10)

    # Component 3: Spread-volume ratio z-score
    log_vol = np.log(volume.where(volume > 0, np.nan) + 1.0)
    svr = spread_norm / log_vol
    z_svr = (svr - svr.rolling(window).mean()) / \
            (svr.rolling(window).std() + 1e-10)

    # Component 4: Dollar-volume z-score (negative contribution)
    dollar_vol = volume * close
    z_dv = (dollar_vol - dollar_vol.rolling(window).mean()) / \
           (dollar_vol.rolling(window).std() + 1e-10)

    ldi = (z_illiq + z_spread + z_svr - z_dv) / 4.0

    return pd.Series(ldi.values, index=close.index, name='liquidity_drought_index')


# --------------------------------------------------------------------------- #
# Depth Imbalance Proxy                                                      #
# --------------------------------------------------------------------------- #

def depth_imbalance_proxy(high: pd.Series, low: pd.Series, close: pd.Series, spread: pd.Series, mid_price: pd.Series,
                          window: int = 20) -> pd.Series:
    """
    Depth Imbalance Proxy (DIP) — order-book imbalance estimated from OHLCV.

    Combines the close location (β, a proxy for which side of the book
    dominated) with the relative spread (an indicator of how tight the book
    is).  When the close is near the high AND the spread is narrow, it is
    most likely that bid-side depth was dominant — and vice versa.

    Formula:
        DIP(t) = (2β(t) − 1) × (1 − S_norm(t) / S_avg(t))

    where β = (C − L) / (H − L),  pressure = 2β − 1 ∈ [−1, +1],
    and S_avg is the rolling mean of S_norm.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC prices.
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    window : int, default=20
        Rolling window for the average spread calculation.

    Returns
    -------
    dip : pd.Series
        Depth imbalance proxy values.
        - DIP > 0 : bid depth dominates (buying pressure + tight spread)
        - DIP < 0 : ask depth dominates (selling pressure + tight spread)
        - |DIP| ≈ 0 : balanced order book or unusually wide spread

    Notes
    -----
    Strong DIP signal with tight spread → likely informed flow in that
    direction.  Combine with VPIN for confirmation.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    _check_index_aligned(high, low, close, spread, mid_price)

    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)
    c = close.values.astype(np.float64)

    beta = beta_clv(h, l, c)          # ∈ [0, 1]
    pressure = 2.0 * beta - 1.0       # ∈ [-1, +1]; negative = sell-side dominant

    spread_norm = spread / mid_price
    spread_avg = spread_norm.rolling(window).mean()
    spread_ratio = 1.0 - (spread_norm / (spread_avg + 1e-10))

    dip = pressure * spread_ratio.values

    return pd.Series(dip, index=high.index, name='depth_imbalance_proxy')


# --------------------------------------------------------------------------- #
# Liquidity Commonality                                                        #
# --------------------------------------------------------------------------- #

def liquidity_commonality(spread: pd.Series, mid_price: pd.Series, benchmark_spread: pd.Series,
                          benchmark_mid: pd.Series, window: int = 20) -> pd.Series:
    """
    Liquidity Commonality.

    Measures co-movement of asset liquidity with market liquidity via rolling
    Pearson correlation of normalised-spread changes.

    Formula:
        LC(t) = Corr[ΔS_asset, ΔS_benchmark]   over window
        S_asset  = spread / mid_price
        S_bench  = benchmark_spread / benchmark_mid

    Parameters
    ----------
    spread, mid_price : pd.Series
        Asset spread and mid-price.
    benchmark_spread, benchmark_mid : pd.Series
        Benchmark (index/sector) spread and mid-price.
    window : int, default=20
        Rolling correlation window.

    Returns
    -------
    pd.Series
        Liquidity commonality in [-1, +1].
        - LC ≈ 1 : asset liquidity moves with market (normal).
        - LC ≈ 0 or negative : decoupling → idiosyncratic event.
    """
    s_asset = spread / mid_price
    s_bench = benchmark_spread / benchmark_mid

    delta_asset = s_asset.diff()
    delta_bench = s_bench.diff()

    lc = delta_asset.rolling(window).corr(delta_bench)

    return pd.Series(lc.values, index=spread.index, name='Liquidity_Commonality')


# --------------------------------------------------------------------------- #
# Liquidity Resilience                                                         #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _ar1_coefficient_kernel(spread_norm: np.ndarray, window: int) -> np.ndarray:
    """
    AR(1) coefficient estimation via rolling OLS.

    Model: S(t) = φ·S(t-1) + intercept + ε(t)
    Returns φ computed as Cov(S_t, S_{t-1}) / Var(S_{t-1}).
    """
    n = len(spread_norm)
    phi = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n):
        y = spread_norm[i - window + 1: i + 1]   # S(t)   length window
        x = spread_norm[i - window: i]            # S(t-1) length window

        # Build valid mask (NaN-aware)
        valid_count = 0
        for j in range(window):
            if not np.isnan(y[j]) and not np.isnan(x[j]):
                valid_count += 1

        if valid_count < window // 2:
            continue

        # Means
        x_mean = 0.0
        y_mean = 0.0
        for j in range(window):
            if not np.isnan(y[j]) and not np.isnan(x[j]):
                x_mean += x[j]
                y_mean += y[j]
        x_mean /= valid_count
        y_mean /= valid_count

        # Cov & Var
        numerator = 0.0
        denominator = 0.0
        for j in range(window):
            if not np.isnan(y[j]) and not np.isnan(x[j]):
                xd = x[j] - x_mean
                yd = y[j] - y_mean
                numerator += xd * yd
                denominator += xd * xd

        if denominator > 1e-10:
            phi[i] = numerator / denominator
        else:
            phi[i] = 1.0  # constant spread -> perfect persistence

    return phi


def liquidity_resilience(spread: pd.Series, mid_price: pd.Series, window: int = 40) -> pd.DataFrame:
    """
    Liquidity Resilience (Mean Reversion Speed).

    Estimates how quickly the bid-ask spread returns to normal after a shock
    by fitting an AR(1) model to the normalised spread time series.

    Model:  S(t) = φ·S(t-1) + (1-φ)·μ + ε(t)

    Parameters
    ----------
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    window : int, default=40
        Rolling estimation window.

    Returns
    -------
    pd.DataFrame
        Columns: ['phi', 'half_life']
        - phi       : AR(1) coefficient. |phi| < 1 → mean reverting.
        - half_life : Time (in bars) to mean-revert by 50%.
                      HalfLife = -ln(2) / ln(|φ|).
        Interpretation:
            Short half-life → resilient liquidity (MMs quickly re-enter).
            Long / growing half-life → fragile liquidity.
            |phi| ≥ 1 → non-stationary (liquidity regime break).
    """
    spread_norm = (spread / mid_price).values.astype(np.float64)

    phi_arr = _ar1_coefficient_kernel(spread_norm, window)
    phi_series = pd.Series(phi_arr, index=spread.index)

    abs_phi = phi_series.abs().clip(lower=1e-10)
    half_life = -np.log(2.0) / np.log(abs_phi)
    half_life = half_life.replace([np.inf, -np.inf], np.nan).ffill()

    return pd.DataFrame({'phi': phi_series, 'half_life': half_life},
                        index=spread.index)


# --------------------------------------------------------------------------- #
# Spread-Volatility Elasticity                                                 #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _spread_vol_elasticity_kernel(log_spread: np.ndarray, log_vol: np.ndarray, window: int) -> np.ndarray:
    """
    Spread-Volatility Elasticity via rolling OLS on log-differences.

    η = Δln(S) / Δln(σ)  ≈  Cov[Δln(S), Δln(σ)] / Var[Δln(σ)]
    """
    n = len(log_spread)
    elasticity = np.full(n, np.nan, dtype=np.float64)

    # First-differences (length n-1)
    d_log_s = np.empty(n - 1, dtype=np.float64)
    d_log_v = np.empty(n - 1, dtype=np.float64)
    for k in range(n - 1):
        d_log_s[k] = log_spread[k + 1] - log_spread[k]
        d_log_v[k] = log_vol[k + 1] - log_vol[k]

    for i in range(window, n - 1):
        ds = d_log_s[i - window: i]   # window differences ending at i-1
        dv = d_log_v[i - window: i]

        valid_count = 0
        for j in range(window):
            if not np.isnan(ds[j]) and not np.isnan(dv[j]):
                valid_count += 1

        if valid_count < window // 2:
            continue

        dv_mean = 0.0
        ds_mean = 0.0
        for j in range(window):
            if not np.isnan(ds[j]) and not np.isnan(dv[j]):
                dv_mean += dv[j]
                ds_mean += ds[j]
        dv_mean /= valid_count
        ds_mean /= valid_count

        cov = 0.0
        var = 0.0
        for j in range(window):
            if not np.isnan(ds[j]) and not np.isnan(dv[j]):
                dvd = dv[j] - dv_mean
                dsd = ds[j] - ds_mean
                cov += dvd * dsd
                var += dvd * dvd

        if var > 1e-10:
            elasticity[i + 1] = cov / var   # write at bar i+1

    return elasticity


def spread_volatility_elasticity(spread: pd.Series, mid_price: pd.Series, high: pd.Series, low: pd.Series,
                                 window: int = 40) -> pd.Series:
    """
    Spread-Volatility Elasticity.

    Measures how responsive the bid-ask spread is to volatility changes.
    Estimated via rolling OLS regression of log-spread changes on log-vol changes.

    Formula:
        η_SV = ∂ln(S) / ∂ln(σ)  (rolling regression slope)

    Parameters
    ----------
    spread : pd.Series
        Bid-ask spread estimate (in price units).
    mid_price : pd.Series
        Mid-price for normalisation.
    high, low : pd.Series
        High and low prices for Parkinson volatility.
    window : int, default=40
        Rolling regression window.

    Returns
    -------
    pd.Series
        Elasticity coefficient named 'Spread_Vol_Elasticity'.
        - η > 1 : Spread over-reacts to vol (nervous MMs, fragile).
        - η ≈ 1 : Normal elasticity.
        - η < 1 : Spread under-reacts (confident MMs, robust).
    """
    from ..volatility._volatility import _parkinson_volatility_nb

    spread_norm = spread / mid_price
    log_spread = np.log(spread_norm.values.astype(np.float64) + 1e-10)

    vol = _parkinson_volatility_nb(high.values.astype(np.float64),
                                   low.values.astype(np.float64), window=20)
    log_vol = np.log(vol + 1e-10)

    elasticity = _spread_vol_elasticity_kernel(log_spread, log_vol, window)

    return pd.Series(elasticity, index=spread.index, name='Spread_Vol_Elasticity')
