"""
Market Regime Detection Features.
Rolling statistical features that characterise the current market regime by measuring co-movement relationships between volatility, volume, and spreads.

Functions
---------
volatility_volume_correlation  : ρ(σ, V)  — information-driven vs exogenous regime
spread_volume_correlation      : ρ(S, V) + Δρ — structural health + crisis precursor
return_autocorrelation_decay   : ACD — trend vs mean-reversion regime
volume_return_asymmetry        : VRA — leverage effect / fear-vs-greed regime
"""

import numpy as np
import pandas as pd
from numba import njit

from ..volatility._volatility import _parkinson_volatility_nb


# --------------------------------------------------------------------------- #
# Numba kernel: rolling correlation                                            #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _rolling_corr_kernel(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation between two arrays, skipping NaN pairs."""
    n = len(x)
    corr = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        xw = x[i - window + 1: i + 1]
        yw = y[i - window + 1: i + 1]

        valid = (~np.isnan(xw)) & (~np.isnan(yw))
        k = np.sum(valid)
        if k < window // 2:
            continue

        xv = xw[valid]
        yv = yw[valid]

        xm = np.mean(xv)
        ym = np.mean(yv)

        cov = np.mean((xv - xm) * (yv - ym))
        xs = np.std(xv)
        ys = np.std(yv)

        if xs > 1e-10 and ys > 1e-10:
            corr[i] = cov / (xs * ys)

    return corr


# --------------------------------------------------------------------------- #
# Numba kernel: autocorrelation decay                                          #
# --------------------------------------------------------------------------- #

@njit(cache=True)
def _autocorr_decay_kernel(returns: np.ndarray, max_lag: int, window: int) -> np.ndarray:
    """
    Rolling autocorrelation decay:
        ACD = [|AC(1)| - |AC(max_lag)|] / (|AC(1)| + ε)
    """
    n = len(returns)
    acd = np.full(n, np.nan, dtype=np.float64)

    for i in range(window + max_lag, n):
        r = returns[i - window - max_lag + 1: i + 1]

        valid = ~np.isnan(r)
        if np.sum(valid) < window // 2:
            continue
        rv = r[valid]
        nv = len(rv)

        if nv <= max_lag:
            continue

        # AC(1)
        c = rv[1:]
        l1 = rv[:-1]
        cm = np.mean(c)
        lm = np.mean(l1)
        cov1 = np.mean((c - cm) * (l1 - lm))
        sc = np.std(c)
        sl = np.std(l1)
        if sc < 1e-10 or sl < 1e-10:
            continue
        ac1 = cov1 / (sc * sl)

        # AC(max_lag)
        cl = rv[max_lag:]
        ll = rv[:-max_lag]
        clm = np.mean(cl)
        llm = np.mean(ll)
        covL = np.mean((cl - clm) * (ll - llm))
        scl = np.std(cl)
        sll = np.std(ll)
        if scl < 1e-10 or sll < 1e-10:
            continue
        acL = covL / (scl * sll)

        # Only compute when AC(1) is statistically significant (≥ 2 std errors);
        # when AC(1) ≈ 0 the ratio blows up and conveys no information.
        significance = 1.0 / np.sqrt(nv)
        if abs(ac1) >= significance:
            acd[i] = (abs(ac1) - abs(acL)) / (abs(ac1) + 1e-10)

    return acd


# --------------------------------------------------------------------------- #
# Volatility-Volume Correlation                                                #
# --------------------------------------------------------------------------- #

def volatility_volume_correlation(high: pd.Series, low: pd.Series, volume: pd.Series, vol_window: int = 10,
                                  window: int = 20) -> pd.Series:
    """
    Volatility-Volume Correlation Regime (ρ_σV). Measures rolling co-movement between Parkinson volatility and volume.
    High positive correlation indicates information-driven price action (informed traders active, moves backed by volume).
    Low or negative correlation suggests exogenous shocks or thin-market dynamics.

    Formula
    -------
        σ_P = Parkinson volatility (rolling `vol_window`)
        ρ_σV = Corr[σ_P, V]  over `window` bars

    Parameters
    ----------
    high, low : pd.Series
        OHLC high and low prices.
    volume : pd.Series
        Bar volume.
    vol_window : int, default=10
        Window for computing Parkinson volatility.
    window : int, default=20
        Rolling window for the correlation.

    Returns
    -------
    pd.Series
        ρ_σV in [-1, +1].  Name: ``vol_volume_corr_{window}``.

        - ρ > 0.5 : Information-driven regime (volatile bars = big volume)
        - ρ < 0.2 : Exogenous / thin-market regime (vol decoupled from volume)
        - Transition zones: regime change onset
    """
    park_vol = _parkinson_volatility_nb(high.values, low.values, window=vol_window)
    corr = _rolling_corr_kernel(park_vol, volume.values, window)

    return pd.Series(corr, index=high.index, name=f'vol_volume_corr_{window}')


# --------------------------------------------------------------------------- #
# Spread-Volume Correlation                                                    #
# --------------------------------------------------------------------------- #

def spread_volume_correlation(spread: pd.Series, mid_price: pd.Series, volume: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Spread-Volume Correlation Regime (ρ_SV). Measures the rolling correlation between normalised spread and volume.
    The *sign* of this correlation is one of the earliest structural regime-change signals available from OHLCV + spread data.

    Formula
    -------
        S_norm = spread / mid_price
        ρ_SV   = Corr[S_norm, V]      over `window` bars
        Δρ_SV  = ρ_SV(t) − ρ_SV(t − window)

    Parameters
    ----------
    spread, mid_price : pd.Series
        Absolute spread and mid-price.
    volume : pd.Series
        Bar volume.
    window : int, default=20
        Rolling correlation window.

    Returns
    -------
    pd.DataFrame
        Columns: ``rho_SV``, ``delta_rho_SV``.

        - ρ_SV < 0 : Healthy — market makers compete, high vol → tight spread
        - ρ_SV > 0 : Stressed — toxic flow, MMs widen quotes
        - Sign flip (− → +): Earliest structural regime-change signal
        - Δρ_SV rising: Regime shift accelerating
    """
    spread_norm = spread / (mid_price + 1e-10)
    rho_sv = spread_norm.rolling(window).corr(volume).ffill()
    delta_rho = rho_sv.diff(window)

    return pd.DataFrame({'rho_SV': rho_sv, 'delta_rho_SV': delta_rho}, index=spread.index)


# --------------------------------------------------------------------------- #
# Return Autocorrelation Decay                                                 #
# --------------------------------------------------------------------------- #

def return_autocorrelation_decay(close: pd.Series, max_lag: int = 5, window: int = 40) -> pd.Series:
    """
    Return Autocorrelation Decay Profile (ACD). Measures how quickly autocorrelation decays with lag,
    classifying the regime as trending (slow decay) vs mean-reverting (fast decay).

    Formula
    -------
        r(i)  = log[C(i) / C(i-1)]
        AC(ℓ) = Corr[r(i), r(i-ℓ)]   (over rolling `window`)
        ACD   = [|AC(1)| − |AC(max_lag)|] / (|AC(1)| + ε)

    Parameters
    ----------
    close : pd.Series
        Close prices.
    max_lag : int, default=5
        Maximum autocorrelation lag to compare against lag-1.
    window : int, default=72
        Rolling window for autocorrelation computation (~quarter FX session at 5-min).

    Returns
    -------
    pd.Series
        ACD values.  Name: ``ac_decay_{max_lag}_{window}``.

        - ACD ≈ 1 : Fast decay → short memory → mean-reversion regime
        - ACD ≈ 0 : Slow decay / insignificant AC(1) → no autocorrelation structure
        - ACD < 0 : Anomalous — higher AC at max_lag than lag-1
    """
    returns = np.log(close / close.shift(1)).values
    acd = _autocorr_decay_kernel(returns, max_lag, window)
    # Fill NaN with 0: insignificant AC(1) means no autocorrelation structure (ACD=0).
    # Warmup NaN (first window+max_lag rows) is also 0 — no data, no memory.
    acd = np.nan_to_num(acd, nan=0.0)
    return pd.Series(acd, index=close.index, name=f'ac_decay_{max_lag}_{window}')


# --------------------------------------------------------------------------- #
# Volume-Return Asymmetry (Leverage Effect Proxy)                             #
# --------------------------------------------------------------------------- #

def volume_return_asymmetry(close: pd.Series, volume: pd.Series, window: int = 40, causal: bool = True) -> pd.Series:
    """
    Volume-Return Asymmetry (VRA) — Leverage Effect Proxy.

    Measures whether recent returns are asymmetrically associated with past vs future volume,
    proxying the classic leverage / fear asymmetry.

    Modes
    -----
    causal=True (default, backtest-safe):
        ρ_fwd  = Corr[r(i-1), V(i)]   — does yesterday's return predict today's volume?
        ρ_bwd  = Corr[r(i),   V(i-1)] — does yesterday's volume predict today's return?
        VRA    = ρ_fwd − ρ_bwd
        No future data is used; safe for live model input.

    causal=False (research / diagnostic only):
        ρ_fwd  = Corr[r(i), V(i+1)]   — USES FUTURE volume; do NOT use as model input.
        ρ_bwd  = Corr[r(i), V(i-1)]
        VRA    = ρ_fwd − ρ_bwd

    Parameters
    ----------
    close : pd.Series
        Close prices.
    volume : pd.Series
        Bar volume.
    window : int, default=40
        Rolling correlation window.
    causal : bool, default=True
        If True, use only past/contemporaneous data (backtest-safe).
        If False, use forward-looking volume (research diagnostic only).

    Returns
    -------
    pd.Series
        VRA values.  Name: ``vol_ret_asymmetry_{window}``.

        - VRA increasingly negative: Fear-driven — down moves beget more volume
        - VRA ≈ 0 : Symmetric response
        - VRA positive : Greed-driven regime (unusual)
    """
    returns = np.log(close / close.shift(1))
    if causal:
        # r(t-1) vs V(t): does past return predict current volume?
        corr_forward = returns.shift(1).rolling(window).corr(volume)
        # r(t) vs V(t-1): does past volume predict current return?
        corr_backward = returns.rolling(window).corr(volume.shift(1))
    else:
        # Research mode: uses V(t+1) — forward-looking, leaks future information.
        corr_forward = returns.rolling(window).corr(volume.shift(-1))
        corr_backward = returns.rolling(window).corr(volume.shift(1))
    vra = corr_forward - corr_backward

    return pd.Series(vra.values, index=close.index, name=f'vol_ret_asymmetry_{window}')
