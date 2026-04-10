"""
Information Asymmetry and Adverse Selection Indicators.

These features estimate the degree of informed trading and information asymmetry from OHLCV + spread data, without requiring tick-level data.

Theoretical Foundation:
    - Easley & O'Hara (1987, 1992): PIN model — Probability of Informed Trading
    - Easley, López de Prado & O'Hara (2012): VPIN — volume-synchronized PIN
    - Glosten & Milgrom (1985): Adverse selection in bid-ask spreads
    - Huang & Stoll (1997): Decomposing the bid-ask spread

Module Structure
----------------
pin_proxy                     : VPIN / (1 + S_rel) — informed-trading proxy
adverse_selection_component   : Corr(r, sign(δV)) — adverse selection measure
smart_money_confidence_index  : Volume-weighted smart-bar directional flow
"""

import numpy as np
import pandas as pd
from numba import njit

from .order_flow import vpin as _vpin
from ._primitives import _buy_sell_volume_kernel, _beta_clv_kernel, _check_index_aligned


def pin_proxy(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, spread: pd.Series,
              mid_price: pd.Series, window: int = 20) -> pd.Series:
    """
    Probability of Informed Trading (PIN) Proxy.
    Approximates the Easley-O'Hara PIN model without tick-level data by combining VPIN (flow toxicity) with a spread-adjustment factor.
    If VPIN is high even after spreads have widened, true information asymmetry is elevated — market makers haven't yet priced in the informed flow.

    Formula:
        S_norm(t) = Spread(t) / Mid(t)
        S_avg(t)  = SMA(S_norm, window)
        S_rel(t)  = S_norm(t) / S_avg(t)
        PIN(t)    = VPIN(t) / (1 + S_rel(t))

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.
    spread : pd.Series
        Absolute bid-ask spread.
    mid_price : pd.Series
        Mid-price (or typical-price proxy).
    window : int, default=20
        Rolling window for both VPIN and spread average.

    Returns
    -------
    pd.Series
        PIN proxy in [0, ~0.5].
        - PIN > 0.3  : Elevated information asymmetry (informed traders active)
        - PIN < 0.1  : Low asymmetry (noise-dominated flow)
        - Rising PIN : Informed traders entering; expect spread widening

    Notes
    -----
    The spread adjustment deflates VPIN when market makers have already
    widened spreads to protect themselves. A high PIN means VPIN is high
    *despite* wide spreads — the strongest signal of informed trading.
    """
    vpin_vals = _vpin(high, low, close, volume, window)

    spread_norm = spread / (mid_price + 1e-10)
    spread_avg = spread_norm.rolling(window).mean()
    spread_rel = spread_norm / (spread_avg + 1e-10)

    pin = vpin_vals / (1.0 + spread_rel)

    return pd.Series(pin.values, index=high.index, name=f'pin_proxy_{window}')


@njit(cache=True)
def _adverse_selection_kernel(returns: np.ndarray, delta_volume: np.ndarray, window: int) -> np.ndarray:
    """
    [Internal Numba kernel] Adverse Selection = Corr(r, sign(δV)).

    Two-pass per window:
        Pass 1: Compute means of r and sign(δV), skipping NaN pairs.
        Pass 2: Compute Pearson correlation from deviations.

    Guards:
        - Requires at least window//2 valid pairs.
        - Returns 0.0 when variance of either series < 1e-10 (constant).
    """
    n = len(returns)
    as_vals = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        # Pass 1 — means
        r_sum = 0.0
        s_sum = 0.0
        count = 0

        for j in range(i - window + 1, i + 1):
            if np.isnan(returns[j]) or np.isnan(delta_volume[j]):
                continue
            r_sum += returns[j]
            if delta_volume[j] > 0.0:
                s_sum += 1.0
            elif delta_volume[j] < 0.0:
                s_sum -= 1.0
            # delta_volume == 0 contributes 0 to s_sum
            count += 1

        if count < max(window // 2, 3):
            continue

        r_mean = r_sum / count
        s_mean = s_sum / count

        # Pass 2 — correlation
        numerator = 0.0
        r_var = 0.0
        s_var = 0.0
        for j in range(i - window + 1, i + 1):
            if np.isnan(returns[j]) or np.isnan(delta_volume[j]):
                continue
            r_dev = returns[j] - r_mean
            if delta_volume[j] > 0.0:
                sign_val = 1.0
            elif delta_volume[j] < 0.0:
                sign_val = -1.0
            else:
                sign_val = 0.0
            s_dev = sign_val - s_mean

            numerator += r_dev * s_dev
            r_var += r_dev * r_dev
            s_var += s_dev * s_dev

        if r_var > 1e-10 and s_var > 1e-10:
            as_vals[i] = numerator / np.sqrt(r_var * s_var)
        else:
            as_vals[i] = 0.0

    return as_vals


def adverse_selection_component(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                                window: int = 20) -> pd.Series:
    """
    Adverse Selection Component (Huang-Stoll decomposition proxy). It measures the rolling Pearson correlation between
    log returns and the sign of volume delta (buy − sell). High positive correlation means returns follow
    order flow → informed traders are moving the price.

    Formula:
        r(t)  = log(C(t) / C(t-1))
        δV(t) = V_buy(t) − V_sell(t)
        AS(t) = Corr(r, sign(δV)) over [t−N+1 … t]

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.
    window : int, default=20
        Rolling window for correlation.

    Returns
    -------
    pd.Series
        AS values in [-1, +1].
        - AS > 0.5  : High adverse selection (informed traders present)
        - AS ∈ [0.2, 0.5] : Moderate adverse selection
        - AS < 0.2  : Low adverse selection (noise-dominated flow)
        - Rising AS : Information asymmetry increasing → expect spread widening

    Notes
    -----
    This is a proxy for the adverse selection component of the Huang-Stoll (1997) spread decomposition.
    It captures whether buy-initiated volume predicts positive returns (and sell-initiated predicts negative), which is the hallmark of informed trading.
    """
    _check_index_aligned(high, low, close, volume)

    log_returns = np.log(close / close.shift(1)).values.astype(np.float64)

    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)
    c = close.values.astype(np.float64)
    v = volume.values.astype(np.float64)

    beta = _beta_clv_kernel(h, l, c)
    v_buy, v_sell = _buy_sell_volume_kernel(beta, v)
    delta_v = v_buy - v_sell

    as_vals = _adverse_selection_kernel(log_returns, delta_v, window)
    return pd.Series(as_vals, index=high.index, name=f'adverse_selection_{window}')


@njit(cache=True)
def _smci_kernel(vir: np.ndarray, volume: np.ndarray, smi: np.ndarray, window: int) -> np.ndarray:
    n = len(vir)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        num = 0.0
        den = 0.0

        for j in range(i - window + 1, i + 1):
            if np.isnan(vir[j]) or np.isnan(volume[j]):
                continue
            w = smi[j] * volume[j]
            num += smi[j] * vir[j] * volume[j]
            den += w

        if den > 1e-10:
            result[i] = num / den
        else:
            result[i] = 0.0
    return result


def smart_money_confidence_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, spread: pd.Series,
                                 mid_price: pd.Series, window: int = 20, rvol_threshold: float = 2.0) -> pd.Series:
    """
    Smart Money Confidence Index (SMCI). It separates "smart money" bars (high relative volume + tight spread) from noise,
    then computes the volume-weighted directional bias of those bars.

    Logic:
        1. Compute RVOL = V / SMA(V, window)
        2. Compute S_norm = Spread / Mid
        3. Flag bar as "smart" if RVOL > threshold AND S_norm < S_avg
           (Rationale: informed traders trade in size when spreads are tight —
           they know price is wrong and liquidity is available.)
        4. SMCI = Σ[SMI × VIR × V] / Σ[SMI × V] over window

    Parameters
    ----------
    high, low, close, volume : pd.Series
        OHLCV data.
    spread : pd.Series
        Absolute bid-ask spread.
    mid_price : pd.Series
        Mid-price (or typical-price proxy).
    window : int, default=20
        Rolling window for RVOL, spread average, and SMCI.
    rvol_threshold : float, default=2.0
        Minimum relative volume to qualify as a "smart" bar.

    Returns
    -------
    pd.Series
        SMCI in [-1, +1].
        - SMCI > 0  : Smart money buying
        - SMCI < 0  : Smart money selling
        - |SMCI| > 0.5 : Strong directional conviction from informed traders
        - SMCI ≈ 0  : No clear smart-money directional bias (or no smart bars)

    Notes
    -----
    The RVOL + tight-spread filter is based on the empirical observation that institutional traders prefer to execute large orders when spreads are
    tight (lower execution cost), which creates a distinctive footprint of high volume + tight spread.
    """
    # RVOL
    vol_sma = volume.rolling(window).mean()
    rvol = volume / (vol_sma + 1e-10)

    # Spread condition
    spread_norm = spread / (mid_price + 1e-10)
    spread_avg = spread_norm.rolling(window).mean()

    # Smart money indicator: high volume + tight spread
    smi = ((rvol > rvol_threshold) & (spread_norm < spread_avg)).astype(float).values

    # VIR = 2β - 1
    h = high.values.astype(np.float64)
    l = low.values.astype(np.float64)
    c = close.values.astype(np.float64)
    beta = _beta_clv_kernel(h, l, c)
    vir = 2.0 * beta - 1.0

    v = volume.values.astype(np.float64)
    smci = _smci_kernel(vir, v, smi, window)
    return pd.Series(smci, index=high.index, name=f'smci_{window}')
