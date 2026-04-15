"""
ATDC adaptation logic — pluggable theta update rules.

AdaptationMode   — enum of built-in modes: 'rdc', 'volatility', 'tmv', 'custom'.
compute_adapted_theta — apply one adaptation step: metric → new theta.

Built-in modes
--------------
'rdc'
    Uses rolling std of RDC values from idc_parse() on the lookback window.
    High RDC std (volatile regime changes) → increase theta.
    Low RDC std → decrease theta.
    Requires a full idc_parse() call on the lookback window.

'volatility'
    Rolling std of log-returns over the lookback window.
    Cheapest to compute — no idc_parse needed.

'tmv'
    Rolling mean of per-bar TMV (Total Market Variation).
    Uses idc_parse() on the lookback window.

'custom'
    User-supplied callable: fn(prices_window, current_theta) -> float (metric signal).
    The returned signal is treated the same as the metric signal in other modes.

Adaptation formula
------------------
    signal   = (metric - baseline) / (|baseline| + 1e-10)   [% deviation from baseline]
    theta_new = clip(theta_old * (1 + adaptation_rate * signal), theta_min, theta_max)

where `baseline` is the rolling mean of past metric values (or the first window's
metric if no history exists yet).  This makes the adaptation self-normalising: the
threshold grows in high-volatility regimes and shrinks in calm ones, regardless of
the absolute scale of the metric.
"""
import enum
import numpy as np
import pandas as pd


class AdaptationMode(str, enum.Enum):
    """Supported adaptation modes for ATDC theta updates."""
    RDC = 'rdc'
    VOLATILITY = 'volatility'
    TMV = 'tmv'
    CUSTOM = 'custom'


def compute_metric(prices_window: pd.Series, current_theta: float, mode: AdaptationMode | str, alpha: float = 1.0, custom_fn=None) -> float:
    """
    Compute the adaptation metric for a given lookback window.

    Parameters
    ----------
    prices_window : pd.Series
        Price series for the lookback window.
    current_theta : float
        Current (active) DC threshold.
    mode : AdaptationMode or str
        Adaptation mode.
    alpha : float
        DC asymmetric attenuation coefficient (used for idc_parse calls).
    custom_fn : callable, optional
        Required when mode='custom'. Signature: fn(prices_window, current_theta) -> float.

    Returns
    -------
    float
        Raw metric value (not yet normalised against baseline).
    """
    from okmich_quant_features.directional_change import idc_parse

    mode = AdaptationMode(mode)
    if len(prices_window) < 2:
        return 0.0

    if mode == AdaptationMode.VOLATILITY:
        log_returns = np.diff(np.log(prices_window.values.astype(np.float64)))
        return float(np.std(log_returns)) if len(log_returns) > 1 else 0.0

    if mode == AdaptationMode.RDC:
        try:
            idc = idc_parse(prices_window, current_theta, alpha)
            rdc_vals = idc['rdc'].dropna().values
            return float(np.std(rdc_vals)) if len(rdc_vals) > 1 else 0.0
        except Exception:
            return 0.0

    if mode == AdaptationMode.TMV:
        try:
            idc = idc_parse(prices_window, current_theta, alpha)
            tmv_vals = idc['tmv'].dropna().values if 'tmv' in idc.columns else np.array([])
            return float(np.mean(tmv_vals)) if len(tmv_vals) > 0 else 0.0
        except Exception:
            return 0.0

    if mode == AdaptationMode.CUSTOM:
        if custom_fn is None:
            raise ValueError("custom_fn must be provided when mode='custom'.")
        return float(custom_fn(prices_window, current_theta))

    raise ValueError(f"Unknown adaptation mode: {mode}")


def compute_adapted_theta(prices_window: pd.Series, current_theta: float, baseline_metric: float, mode: AdaptationMode | str, adaptation_rate: float, theta_min: float, theta_max: float, alpha: float = 1.0, custom_fn=None) -> tuple[float, float]:
    """
    Compute an updated theta value using the adaptation formula.

    Adaptation formula:
        metric   = compute_metric(prices_window, current_theta, mode)
        signal   = (metric - baseline) / (|baseline| + 1e-10)
        theta_new = clip(current_theta * (1 + adaptation_rate * signal), theta_min, theta_max)

    Parameters
    ----------
    prices_window : pd.Series
        Lookback price window for metric computation.
    current_theta : float
        Current DC threshold.
    baseline_metric : float
        Baseline metric value for normalisation (updated by caller over time).
    mode : AdaptationMode or str
        Adaptation mode.
    adaptation_rate : float
        Sensitivity of theta to metric deviations. Typical range: 0.1 – 2.0.
    theta_min : float
        Lower bound on theta.
    theta_max : float
        Upper bound on theta.
    alpha : float
        DC asymmetric coefficient (used for idc_parse calls).
    custom_fn : callable, optional
        Custom metric function (mode='custom' only).

    Returns
    -------
    (theta_new, metric) : tuple[float, float]
        Updated theta and the raw metric value (for caller to update baseline).
    """
    metric = compute_metric(prices_window, current_theta, mode, alpha, custom_fn)
    signal = (metric - baseline_metric) / (abs(baseline_metric) + 1e-10)
    theta_new = current_theta * (1.0 + adaptation_rate * signal)
    theta_new = float(np.clip(theta_new, theta_min, theta_max))
    return theta_new, metric
