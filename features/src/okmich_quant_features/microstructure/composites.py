"""
Composite Meta-Features for Market Microstructure Analysis.

High-level features that combine multiple primitives from order_flow, liquidity, depth, information, and volatility modules.
Each composite is designed to answer a specific trading question with a single number.

Module Structure
----------------
liquidity_adjusted_momentum    (LAM) : Momentum normalised by cost + risk
volume_price_divergence        (VPD) : Price-volume disagreement (exhaustion)
informed_liquidity_pressure    (ILP) : Triple product: VPIN × λ_z × z_S
institutional_footprint_score  (IFS) : Composite of institutional activity signals
regime_fragility_index         (RFI) : Master fragility / pre-crisis composite
"""

import numpy as np
import pandas as pd

from .depth import wick_imbalance as _wi, bar_absorption_ratio as _ar
from .information import smart_money_confidence_index as _smci
from ._primitives import beta_clv, buy_sell_volume, typical_price
from .order_flow import vpin as _vpin, kyles_lambda_zscore as _kl_z, volume_entropy as _ve
from .liquidity import amihud_illiquidity as _amihud, liquidity_drought_index as _ldi, spread_zscore as _sz
from ..volatility._volatility import vov_normalized as _vov_n


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with ε guard on std."""
    mu = s.rolling(window).mean()
    sigma = s.rolling(window).std()
    return (s - mu) / (sigma + 1e-10)


def liquidity_adjusted_momentum(close: pd.Series, spread: pd.Series, mid_price: pd.Series, vol_window: int = 20,
                                momentum_window: int = 20) -> pd.Series:
    """
    Liquidity-Adjusted Momentum (LAM).

    Normalises directional force by the cost and risk of participating in that move.
    A 1% move on wide spread + high vol is expensive and unreliable.
    The same move on tight spread + low vol is cheap and high-conviction.

    Formula:
        momentum = log(C(t) / C(t − N))
        rv       = std(log-returns, N) × √252          [annualised]
        S_norm   = Spread / Mid
        LAM      = momentum / (S_norm × rv + ε)

    Parameters
    ----------
    close : pd.Series
        Close prices.
    spread, mid_price : pd.Series
        Absolute spread and mid-price.
    vol_window : int, default=20
        Rolling window for realised volatility.
    momentum_window : int, default=20
        Lookback for log-return momentum.

    Returns
    -------
    pd.Series
        - High positive LAM: Strong bullish momentum, cheap to capture
        - High negative LAM: Strong bearish momentum, cheap to capture
        - Low |LAM|       : Move is expensive / noisy (low conviction)
    """
    momentum = np.log(close / close.shift(momentum_window))
    spread_norm = spread / (mid_price + 1e-10)
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(vol_window).std() * np.sqrt(252)

    lam = momentum / (spread_norm * rv + 1e-10)
    return pd.Series(lam.values, index=close.index, name=f'lam_{momentum_window}')


def volume_price_divergence(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Volume-Price Divergence (VPD).
    Detects price-volume disagreement — a classic exhaustion signal. When price moves up but volume shrinks, the move
    lacks conviction and is likely to reverse. Conversely, price down + volume down = bearish exhaustion.

    Formula:
        RVOL = V / SMA(V, N)
        VPD  = sign(ΔC) × (1 − RVOL)

    Returns
    -------
    pd.Series
        - VPD > 0 : Price and volume disagree → reversal warning
        - VPD < 0 : Price and volume agree → continuation
        - Persistent VPD > 0 : High reversal probability

    Interpretation
    --------------
    - Price up  + volume down → bullish exhaustion  (VPD > 0)
    - Price down + volume down → bearish exhaustion (VPD > 0)
    - Price up  + volume up   → healthy uptrend     (VPD < 0)
    - Price down + volume up  → healthy downtrend   (VPD < 0)
    """
    price_sign = np.sign(close.diff())
    vol_sma = volume.rolling(window).mean()
    rvol = volume / (vol_sma + 1e-10)

    vpd = price_sign * (1.0 - rvol)
    return pd.Series(vpd.values, index=close.index, name=f'vpd_{window}')


def informed_liquidity_pressure(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, spread: pd.Series,
                                mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Informed Liquidity Pressure (ILP).
    Triple product of three fragility indicators — all three must be elevated simultaneously for a strong signal, making ILP highly specific.

    Formula:
        ILP = VPIN × λ_z × z_S

    Components:
        1. VPIN  : Toxic flow (informed traders creating imbalance)
        2. λ_z   : Kyle's lambda z-score (high price impact → low depth)
        3. z_S   : Spread z-score (market makers retreating)

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.
    spread, mid_price : pd.Series
        Absolute spread and mid-price.
    window : int, default=20
        Rolling window for all components.

    Returns
    -------
    pd.Series
        - High |ILP| : Toxic flow + high impact + stressed liquidity
        - ILP spikes  : Most reliable precursor to disorderly moves
        - ILP near 0  : At least one leg is benign → no crisis signal
    """
    vpin_vals = _vpin(high, low, close, volume, window)
    lambda_z = _kl_z(high, low, close, volume, window)
    z_spread = _sz(spread, mid_price, window)
    ilp = vpin_vals * lambda_z * z_spread
    return pd.Series(ilp.values, index=high.index, name=f'ilp_{window}')


def institutional_footprint_score(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                                  window: int = 20) -> pd.Series:
    """
    Institutional Footprint Score (IFS).
    Composite of three institutional-activity indicators. Each component is z-scored within its own rolling window, then averaged.
    High IFS indicates that an institutional algorithm is likely active.

    Formula:
        IFS = (z_VCR + z_AR + inv_VE) / 3

    Components:
        1. VCR (Volume Concentration Ratio): max(V) / mean(V) over window.
           High when a single bar dominates volume (block trade).
        2. AR (Bar Absorption Ratio): V / |C−O|.
           High when price absorbs volume without moving (iceberg).
        3. inv_VE (Inverse Volume Entropy): 1 − VE_norm.
           High when volume distribution is concentrated (not uniform).

    Parameters
    ----------
    open_, high, low, close, volume : pd.Series
        OHLCV data.
    window : int, default=20
        Rolling window for z-scores and component computation.

    Returns
    -------
    pd.Series
        - IFS > 1.5  : Institutional algorithm likely active
        - IFS > 2.5  : Strong institutional presence
        - IFS < 0    : Retail-dominated flow
    """
    # Volume Concentration Ratio
    vcr = volume.rolling(window).max() / (volume.rolling(window).mean() + 1e-10)
    z_vcr = _zscore(vcr, window)
    # Absorption Ratio — z-scored
    ar = _ar(open_, close, volume)
    z_ar = _zscore(ar, window)
    # Volume Entropy (inverted: low entropy = concentrated = institutional)
    ve = _ve(volume, window)
    inv_ve = 1.0 - ve
    # Composite (3 components)
    ifs = (z_vcr + z_ar + inv_ve) / 3.0
    return pd.Series(ifs.values, index=open_.index, name=f'ifs_{window}')


def regime_fragility_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, spread: pd.Series,
                           mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Regime Fragility Index (RFI).
    Master composite of all fragility / pre-crisis indicators. Answers the question: "How likely is a regime break in the near future?"

    Formula:
        RFI = (z_VoV + z_ILLIQ + z_VPIN + λ_z + LDI) / 5

    Components (all z-scored or already z-score-like):
        1. z_VoV    : Volatility of Volatility (normalised) — regime instability
        2. z_ILLIQ  : Amihud Illiquidity — price impact rising
        3. z_VPIN   : VPIN z-score — toxic flow surging
        4. λ_z      : Kyle's Lambda z-score — market depth thinning
        5. LDI      : Liquidity Drought Index — composite fragility

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.
    spread, mid_price : pd.Series
        Absolute spread and mid-price.
    window : int, default=20
        Rolling window for all components.

    Returns
    -------
    pd.Series
        - RFI > 2  : Fragile regime (elevated probability of break)
        - RFI > 3  : Severe fragility (crisis conditions imminent)
        - RFI < 0  : Robust regime (all indicators benign)
    """
    # VoV normalised → z-score
    vov = _vov_n(high, low, window=window, vov_window=window)
    z_vov = _zscore(pd.Series(vov, index=high.index), window)

    # Amihud illiquidity → z-score
    illiq = _amihud(close, volume, window)
    z_illiq = _zscore(illiq, window)
    # VPIN → z-score
    vpin_vals = _vpin(high, low, close, volume, window)
    z_vpin = _zscore(pd.Series(vpin_vals, index=high.index), window)
    # Kyle's Lambda z-score (already a z-score)
    lambda_z = pd.Series(_kl_z(high, low, close, volume, window), index=high.index)
    # LDI (already a composite z-score)
    ldi = _ldi(close, volume, spread, mid_price, window)
    # Composite (5 components)
    rfi = (z_vov + z_illiq + z_vpin + lambda_z + ldi) / 5.0
    return pd.Series(rfi.values, index=high.index, name=f'rfi_{window}')


def supply_demand_pressure_differential(high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series,
                                        volume: pd.Series, spread: pd.Series, mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Supply/Demand Pressure Differential (SDP).
    Multi-lens composite of directional pressure combining four components:
        VAPC — Volume-At-Price Concentration
        FVD  — Flow-weighted price vs VWAP Divergence
        WI   — Wick Imbalance (EMA-smoothed)
        SMCI — Smart Money Confidence Index

    Formula:
        SDP = 0.25·VAPC_ema + 0.25·FVD + 0.25·WI_ema + 0.25·SMCI

    Parameters
    ----------
    high, low, open_, close, volume : pd.Series
        OHLCV data.
    spread, mid_price : pd.Series
        Absolute spread and mid-price.
    window : int, default=20
        EMA span / rolling window for all components.

    Returns
    -------
    pd.Series
        - SDP > 0 : demand dominates (bullish pressure).
        - SDP < 0 : supply dominates (bearish pressure).
        - |SDP| > 1 : strong pressure.
    """
    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)
    c = close.values.astype(np.float64)
    v = volume.values.astype(np.float64)

    # Component 1: VAPC (Volume-At-Price Concentration)
    beta = pd.Series(beta_clv(h, l, c), index=high.index)
    vapc = beta * high + (1.0 - beta) * low
    vapc_mid = (high + low) / 2.0
    rng = high - low
    vapc_norm = (vapc - vapc_mid) / (rng + 1e-10)
    vapc_ema = vapc_norm.ewm(span=window, adjust=False).mean()

    # Component 2: FVD (Flow-Weighted Price vs VWAP Divergence)
    v_buy, v_sell = buy_sell_volume(h, l, c, v)
    delta_v = pd.Series(v_buy - v_sell, index=high.index)
    typical_arr = typical_price(h, l, c)
    typical_s = pd.Series(typical_arr, index=high.index)

    fwp_num = (typical_s * delta_v.abs()).rolling(window).sum()
    fwp_den = delta_v.abs().rolling(window).sum()
    fwp = fwp_num / (fwp_den + 1e-10)

    vwap_num = (typical_s * volume).rolling(window).sum()
    vwap_den = volume.rolling(window).sum()
    vwap = vwap_num / (vwap_den + 1e-10)
    fvd = (fwp - vwap) / (vwap + 1e-10)
    # Component 3: WI (Wick Imbalance, EMA-smoothed)
    wi = _wi(high, low, open_, close)
    wi_ema = wi.ewm(span=window, adjust=False).mean()
    # Component 4: SMCI (Smart Money Confidence Index)
    smci = _smci(high, low, close, volume, spread, mid_price, window)
    # Equal-weight composite
    sdp = 0.25 * vapc_ema + 0.25 * fvd + 0.25 * wi_ema + 0.25 * smci
    return pd.Series(sdp.values, index=high.index, name='Supply_Demand_Pressure')


def predictive_liquidity_transition_score(high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series,
                                          volume: pd.Series, spread: pd.Series, mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Predictive Liquidity Transition Score (PLTS).
    Five-component z-scored composite that predicts impending liquidity regime transitions (breakouts, volatility spikes, liquidity crises).

    Formula:
        PLTS = (z_ΔVPIN + z_SEM + z_Δλ + z_(1-IER_avg) + z_RCR_streak) / 5

    Components:
        1. ΔVPIN       — Rising toxic flow → directional move.
        2. SEM         — Spread expansion → volatility onset.
        3. Δλ          — Rising price impact → amplified moves.
        4. (1-IER_avg) — Low efficiency → indecision → breakout.
        5. RCR_streak  — Consecutive range compression → coiled spring.

    Parameters
    ----------
    high, low, open_, close, volume : pd.Series
        OHLCV data.
    spread, mid_price : pd.Series
        Absolute spread and mid-price.
    window : int, default=20
        Rolling window for z-scoring and component computation.

    Returns
    -------
    pd.Series
        - PLTS > 2 : High probability of liquidity transition.
        - PLTS > 3 : Imminent volatility event.
    """
    from .order_flow import delta_vpin as _dvpin, kyles_lambda as _kl
    from .liquidity import spread_expansion_momentum as _sem
    from .price_structure import intrabar_efficiency_ratio as _ier, \
        range_compression_ratio as _rcr

    # Component 1: ΔVPIN
    dvpin = _dvpin(high, low, close, volume, window, lookback=5)

    # Component 2: SEM
    sem = _sem(spread, mid_price, ema_span=5)

    # Component 3: Δλ (5-bar change in Kyle's lambda)
    lambda_vals = _kl(high, low, close, volume, window)
    delta_lambda = lambda_vals.diff(5)

    # Component 4: (1 - IER_avg)
    ier = _ier(open_, high, low, close)
    ier_avg = ier.rolling(window).mean()
    inv_ier = 1.0 - ier_avg

    # Component 5: RCR streak (consecutive bars with RCR < 0.5)
    rcr = _rcr(high, low, short_span=5, long_span=window)
    rcr_compressed = (rcr < 0.5).astype(float)
    rcr_streak_arr = np.zeros(len(rcr_compressed), dtype=np.float64)
    streak = 0
    for i in range(len(rcr_compressed)):
        if rcr_compressed.iloc[i] > 0:
            streak += 1
        else:
            streak = 0
        rcr_streak_arr[i] = streak
    rcr_streak = pd.Series(rcr_streak_arr, index=high.index)

    # Z-score each component for equal contribution
    def _z(s: pd.Series) -> pd.Series:
        mu = s.rolling(window).mean()
        sigma = s.rolling(window).std()
        return (s - mu) / (sigma + 1e-10)

    dvpin_z = _z(dvpin)
    sem_z = _z(sem)
    dlambda_z = _z(delta_lambda)
    inv_ier_z = _z(inv_ier)
    streak_z = _z(rcr_streak)
    plts = (dvpin_z + sem_z + dlambda_z + inv_ier_z + streak_z) / 5.0
    return pd.Series(plts.values, index=high.index, name='Predictive_Liquidity_Transition_Score')
