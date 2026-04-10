import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _parse_dc_core(arr: np.ndarray, theta: float, alpha: float):
    """
    Numba-optimised DC parser core. Operates on raw numpy arrays only.

    mode encoding: 0 = initialisation, 1 = uptrend (tracking for downward DC),
                  -1 = downtrend (tracking for upward DC).
    direction encoding in output: 1 = up trend (trough→peak), -1 = down trend (peak→trough).

    alpha : float
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC threshold = alpha * theta. alpha=1.0 → symmetric (book default).
    """
    n = len(arr)
    max_trends = n

    out_direction = np.empty(max_trends, dtype=np.int8)
    out_ext_start_price = np.empty(max_trends, dtype=np.float64)
    out_ext_start_pos = np.empty(max_trends, dtype=np.int64)
    out_ext_end_price = np.empty(max_trends, dtype=np.float64)
    out_ext_end_pos = np.empty(max_trends, dtype=np.int64)
    out_dcc_price = np.empty(max_trends, dtype=np.float64)
    out_dcc_pos = np.empty(max_trends, dtype=np.int64)
    out_tmv = np.empty(max_trends, dtype=np.float64)
    out_t = np.empty(max_trends, dtype=np.int64)
    out_r = np.empty(max_trends, dtype=np.float64)
    count = 0

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
                new_ext_price = running_extreme
                new_ext_pos = running_extreme_pos
                tmv = (new_ext_price - last_ext_price) / last_ext_price / theta
                t = new_ext_pos - last_ext_pos
                r = tmv * theta / t if t > 0 else np.nan

                out_direction[count] = np.int8(1)
                out_ext_start_price[count] = last_ext_price
                out_ext_start_pos[count] = last_ext_pos
                out_ext_end_price[count] = new_ext_price
                out_ext_end_pos[count] = new_ext_pos
                out_dcc_price[count] = p
                out_dcc_pos[count] = np.int64(i)
                out_tmv[count] = tmv
                out_t[count] = t
                out_r[count] = r
                count += 1

                last_ext_price = new_ext_price
                last_ext_pos = new_ext_pos
                mode = np.int8(-1)
                running_extreme = p
                running_extreme_pos = np.int64(i)

        elif mode == -1:
            if p < running_extreme:
                running_extreme = p
                running_extreme_pos = np.int64(i)

            if p >= running_extreme * (1.0 + theta):
                new_ext_price = running_extreme
                new_ext_pos = running_extreme_pos
                tmv = (last_ext_price - new_ext_price) / last_ext_price / theta
                t = new_ext_pos - last_ext_pos
                r = tmv * theta / t if t > 0 else np.nan

                out_direction[count] = np.int8(-1)
                out_ext_start_price[count] = last_ext_price
                out_ext_start_pos[count] = last_ext_pos
                out_ext_end_price[count] = new_ext_price
                out_ext_end_pos[count] = new_ext_pos
                out_dcc_price[count] = p
                out_dcc_pos[count] = np.int64(i)
                out_tmv[count] = tmv
                out_t[count] = t
                out_r[count] = r
                count += 1

                last_ext_price = new_ext_price
                last_ext_pos = new_ext_pos
                mode = np.int8(1)
                running_extreme = p
                running_extreme_pos = np.int64(i)

    return (count, out_direction, out_ext_start_price, out_ext_start_pos,
            out_ext_end_price, out_ext_end_pos, out_dcc_price, out_dcc_pos,
            out_tmv, out_t, out_r)


def parse_dc_events(prices: pd.Series, theta: float, alpha: float = 1.0) -> pd.DataFrame:
    """
    Parse a price series into completed DC trends using the Directional Change framework.

    Each row in the output represents one completed trend (EXT to EXT), comprising
    one DC event (EXT → DCC) and one overshoot event (DCC → next EXT).

    Parameters
    ----------
    prices : pd.Series
        Close price series. Index can be integer or DatetimeIndex.
    theta : float
        DC threshold as a decimal fraction (e.g., 0.002 for 0.2%).
    alpha : float, optional
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC confirmed when price falls alpha*theta from the last peak.
        alpha=1.0 (default) → symmetric thresholds matching the original book formulation.
        alpha<1.0 → downward DC fires sooner, acting as a tighter stop-loss.

    Returns
    -------
    pd.DataFrame
        One row per completed trend with columns:
        - direction        : 'up' (trough→peak) or 'down' (peak→trough)
        - ext_start_price  : P_EXT(n-1) — price at start extreme
        - ext_start_idx    : t_EXT(n-1) — index label of start extreme
        - ext_end_price    : P_EXT(n)   — price at end extreme
        - ext_end_idx      : t_EXT(n)   — index label of end extreme
        - dcc_price        : price at DC confirmation point
        - dcc_idx          : index label at DC confirmation point
        - tmv              : Total Price Movement (multiples of θ).
                             Upward trends: always ≥ 1.0.
                             Downward trends: ≥ alpha (may be < 1.0 when alpha < 1.0).
        - t                : bars from ext_start to ext_end (integer bar count)
        - r                : Time-Adjusted Return = tmv × θ / t

    Notes
    -----
    - EXT points are confirmed retrospectively — a trough is only known when price
      rises by θ from it. No look-ahead bias is introduced.
    - T is always measured in bars regardless of the index type.
    - Book reference: Section 2.2.1; Eq 2.2–2.4; Appendix A.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if len(prices) < 2:
        return pd.DataFrame()

    arr = prices.values.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("prices contains NaN or infinite values. Clean the data before parsing.")
    idx = prices.index

    count, d, esp, espos, eep, eepos, dccp, dccpos, tmv, t, r = _parse_dc_core(arr, theta, alpha)

    if count == 0:
        return pd.DataFrame()

    return pd.DataFrame({
        'direction': np.where(d[:count] == 1, 'up', 'down'),
        'ext_start_price': esp[:count],
        'ext_start_idx': idx[espos[:count]],
        'ext_start_pos': espos[:count],
        'ext_end_price': eep[:count],
        'ext_end_idx': idx[eepos[:count]],
        'ext_end_pos': eepos[:count],
        'dcc_price': dccp[:count],
        'dcc_idx': idx[dccpos[:count]],
        'dcc_pos': dccpos[:count],
        'tmv': tmv[:count],
        't': t[:count],
        'r': r[:count],
    })
