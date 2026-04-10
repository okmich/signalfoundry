import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _dc_live_core(arr: np.ndarray, theta: float, alpha: float):
    """
    Numba-optimised core for per-bar current TMV, T, direction, and DCC flags.

    mode encoding: 0 = initialisation, 1 = uptrend, -1 = downtrend.

    alpha : float
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC threshold = alpha * theta. alpha=1.0 → symmetric (book default).

    upward_dcc  fires when mode transitions from -1 → 1 (price rose theta from trough).
    downward_dcc fires when mode transitions from  1 → -1 (price fell alpha*theta from peak).
    Neither flag fires on the initial mode=0 → mode=±1 transition (no prior trend confirmed).
    """
    n = len(arr)
    tmv_current = np.full(n, np.nan)
    t_current = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    upward_dcc = np.zeros(n, dtype=np.bool_)
    downward_dcc = np.zeros(n, dtype=np.bool_)

    last_ext_price = arr[0]
    last_ext_pos = np.int64(0)
    running_extreme = arr[0]
    running_extreme_pos = np.int64(0)
    mode = np.int8(0)

    init_high = arr[0]
    init_high_pos = np.int64(0)
    init_low = arr[0]
    init_low_pos = np.int64(0)

    for i in range(1, n):
        p = arr[i]
        prev_mode = mode

        if mode == 0:
            if p > init_high:
                init_high = p
                init_high_pos = np.int64(i)
            if p < init_low:
                init_low = p
                init_low_pos = np.int64(i)

            if p <= init_high * (1.0 - alpha * theta):
                last_ext_price = init_high
                last_ext_pos = init_high_pos
                mode = np.int8(-1)
                running_extreme = p
                running_extreme_pos = np.int64(i)
            elif p >= init_low * (1.0 + theta):
                last_ext_price = init_low
                last_ext_pos = init_low_pos
                mode = np.int8(1)
                running_extreme = p
                running_extreme_pos = np.int64(i)

        if mode == 1:
            if p > running_extreme:
                running_extreme = p
                running_extreme_pos = np.int64(i)

            if p <= running_extreme * (1.0 - alpha * theta):
                last_ext_price = running_extreme
                last_ext_pos = running_extreme_pos
                mode = np.int8(-1)
                running_extreme = p
                running_extreme_pos = np.int64(i)
                if prev_mode == np.int8(1):
                    downward_dcc[i] = True

        elif mode == -1:
            if p < running_extreme:
                running_extreme = p
                running_extreme_pos = np.int64(i)

            if p >= running_extreme * (1.0 + theta):
                last_ext_price = running_extreme
                last_ext_pos = running_extreme_pos
                mode = np.int8(1)
                running_extreme = p
                running_extreme_pos = np.int64(i)
                if prev_mode == np.int8(-1):
                    upward_dcc[i] = True

        if mode != 0:
            tmv_current[i] = abs(p - last_ext_price) / last_ext_price / theta
            t_current[i] = float(i - last_ext_pos)
            direction[i] = float(mode)

    return tmv_current, t_current, direction, upward_dcc, downward_dcc


def log_r(trends: pd.DataFrame) -> pd.Series:
    """
    Compute log-transformed Time-Adjusted Return for each completed DC trend.

    Applied before feeding R into the HMM to compress the right-skewed distribution
    of R and satisfy the Gaussian emission assumption.

    Parameters
    ----------
    trends : pd.DataFrame
        Output of parse_dc_events(). Must contain an 'r' column.

    Returns
    -------
    pd.Series
        log(R) per trend, same index as trends. NaN where R is NaN or non-positive.

    Notes
    -----
    Book reference: Eq 3.1; Section 3.2.1. Used in Chapter 3 HMM only.
    """
    return np.log(trends['r'].where(trends['r'] > 0))


def dc_live_features(prices: pd.Series, theta: float, alpha: float = 1.0) -> pd.DataFrame:
    """
    Compute per-bar current TMV, T, direction, and DCC flags from the last confirmed extreme.

    At every bar, the current (unfinished) trend's TMV and T are computed without waiting for trend completion. These are
    the live inputs to the classifier in Chapter 5, extended with DCC event flags for the ITA algorithm (Hu et al. 2022).

    Parameters
    ----------
    prices : pd.Series
        Close price series.
    theta : float
        DC threshold as a decimal fraction (e.g., 0.002 for 0.2%).
    alpha : float, optional
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC fires when price falls alpha*theta from the last peak.
        alpha=1.0 (default) → symmetric thresholds matching the original book.

    Returns
    -------
    pd.DataFrame
        Same index as prices, with columns:
        - tmv_current  : |price - last_ext_price| / last_ext_price / θ
        - t_current    : bars elapsed since last confirmed extreme (float)
        - direction    : +1.0 (uptrend from last trough) or -1.0 (downtrend from last peak);
                         NaN before the first DC event is confirmed.
        - upward_dcc   : True on the bar an upward DC is confirmed (price rose θ from trough).
                         False on the initial mode=0 → mode=1 transition.
        - downward_dcc : True on the bar a downward DC is confirmed (price fell α×θ from peak).
                         False on the initial mode=0 → mode=-1 transition.
        - rdc_current  : Running RDC of the current in-progress half-cycle.
                         Formula: tmv_current * θ / t_current = |p - last_ext| / last_ext / t.
                         NaN during init phase and when t_current == 0.
                         Note: this is the LIVE running estimate. For RDC of completed trends,
                         use idc_parse() which emits rdc only at DC confirmation bars.

    Notes
    -----
    - last_ext_price is only updated when a new DC event is confirmed — no look-ahead.
    - At the DCC bar, tmv_current resets to ~α (downward) or ~1.0 (upward) from the new EXT.
    - T is in bars regardless of index type.
    - rdc_current uses current bar price (not running extreme) — consistent with tmv_current.
    - Book reference: Section 5.2.1; Figure 5.2. Hu et al. extension: Section 2.2.1.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if len(prices) < 2:
        return pd.DataFrame({
            'tmv_current': np.nan, 't_current': np.nan, 'direction': np.nan,
            'upward_dcc': False, 'downward_dcc': False, 'rdc_current': np.nan,
        }, index=prices.index)
    arr = prices.values.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("prices contains NaN or infinite values. Clean the data before computing live features.")
    tmv_current, t_current, direction, upward_dcc, downward_dcc = _dc_live_core(arr, theta, alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        rdc_current = np.where(t_current > 0, tmv_current * theta / t_current, np.nan)
    return pd.DataFrame({
        'tmv_current': tmv_current,
        't_current': t_current,
        'direction': direction,
        'upward_dcc': upward_dcc,
        'downward_dcc': downward_dcc,
        'rdc_current': rdc_current,
    }, index=prices.index)


def normalise_minmax(series: pd.Series, min_val: float = None, max_val: float = None) -> tuple[pd.Series, float, float]:
    """
    Min-max normalise a DC indicator series to [0, 1].

    Designed for use with TMV and T. Call with min_val=None on the training set to
    fit, then pass the returned min_val/max_val when normalising live/test data to
    avoid look-ahead bias.

    Parameters
    ----------
    series : pd.Series
        TMV or T values to normalise.
    min_val : float, optional
        Pre-computed minimum (from training set). If None, computed from series.
    max_val : float, optional
        Pre-computed maximum (from training set). If None, computed from series.

    Returns
    -------
    tuple[pd.Series, float, float]
        (normalised_series, min_val, max_val)
        min_val and max_val should be stored and reused for test/live normalisation.

    Notes
    -----
    Book reference: Eq 4.3; Section 4.2.3.
    """
    if series.isna().any() or not np.isfinite(series.values).all():
        raise ValueError("normalise_minmax: series contains NaN or infinite values. Clean the data before normalising.")
    if min_val is None:
        min_val = float(series.min())
    if max_val is None:
        max_val = float(series.max())

    denom = max_val - min_val
    if denom == 0.0:
        return pd.Series(np.zeros(len(series)), index=series.index), min_val, max_val

    normalised = (series - min_val) / denom
    return normalised, min_val, max_val
