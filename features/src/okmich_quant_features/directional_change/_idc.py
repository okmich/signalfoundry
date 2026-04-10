import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _idc_core(arr: np.ndarray, theta: float, alpha: float):
    """
    Numba-optimised IDC incremental parser. Operates on raw numpy arrays only.

    Returns per-bar event signal arrays needed for ITA Algorithm 1 (Wu & Han 2023).

    mode encoding: 0 = initialisation, 1 = upturn (watching for downward DC),
                  -1 = downturn (watching for upward DC).

    alpha : float
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC threshold = alpha * theta. alpha=1.0 -> symmetric (book default).

    Per-bar outputs
    ---------------
    direction   : int8  — 0=init, 1=upturn, -1=downturn
    ph          : float — running high (confirmed peak or running max in upturn)
    pl          : float — running low  (confirmed trough or running min in downturn)
    t_dc0       : int64 — bar offset of last extreme update in current half-cycle
    upturn_dc   : bool  — True when upward DC confirmed this bar
    downturn_dc : bool  — True when downward DC confirmed this bar
    new_high    : bool  — True when a new high is set during upturn (Rule 2 trigger)
    new_low     : bool  — True when a new low is set during downturn
    rdc         : float — RDC at DC confirmation bars, NaN for init DC and other bars
    """
    n = len(arr)

    out_direction = np.zeros(n, dtype=np.int8)
    out_ph = np.empty(n, dtype=np.float64)
    out_pl = np.empty(n, dtype=np.float64)
    out_t_dc0 = np.zeros(n, dtype=np.int64)
    out_upturn_dc = np.zeros(n, dtype=np.bool_)
    out_downturn_dc = np.zeros(n, dtype=np.bool_)
    out_new_high = np.zeros(n, dtype=np.bool_)
    out_new_low = np.zeros(n, dtype=np.bool_)
    out_rdc = np.full(n, np.nan)

    mode = np.int8(0)
    ph = arr[0]
    pl = arr[0]
    t_dc0 = np.int64(0)

    init_high = arr[0]
    init_high_pos = np.int64(0)
    init_low = arr[0]
    init_low_pos = np.int64(0)

    out_direction[0] = mode
    out_ph[0] = ph
    out_pl[0] = pl
    out_t_dc0[0] = t_dc0

    for i in range(1, n):
        p = arr[i]

        if mode == 0:
            if p > init_high:
                init_high = p
                init_high_pos = np.int64(i)
            if p < init_low:
                init_low = p
                init_low_pos = np.int64(i)

            # Keep ph/pl synced to init extremes until first DC fires
            ph = init_high
            pl = init_low

            if p <= init_high * (1.0 - alpha * theta):
                # First downward DC — mode transitions to downturn
                # ph stays at init_high (confirmed peak); pl starts at confirmation price
                pl = p
                t_dc0 = np.int64(i)
                mode = np.int8(-1)
                out_downturn_dc[i] = True
                # rdc remains NaN: no prior completed trend from init phase

            elif p >= init_low * (1.0 + theta):
                # First upward DC — mode transitions to upturn
                # pl stays at init_low (confirmed trough); ph starts at confirmation price
                ph = p
                t_dc0 = np.int64(i)
                mode = np.int8(1)
                out_upturn_dc[i] = True
                # rdc remains NaN: no prior completed trend from init phase

        if mode == 1:
            # Upturn: track running high, watch for downward DC confirmation.
            if p > ph:
                ph = p
                t_dc0 = np.int64(i)
                out_new_high[i] = True

            if p <= ph * (1.0 - alpha * theta):
                # Downward DC confirmed.
                # RDC: price drop from peak to confirmation / peak / bars-since-last-new-high
                out_downturn_dc[i] = True
                t_span = np.int64(i) - t_dc0
                out_rdc[i] = abs(p - ph) / ph / float(t_span) if t_span > 0 else np.nan
                pl = p
                t_dc0 = np.int64(i)
                mode = np.int8(-1)

        elif mode == -1:
            # Downturn: track running low, watch for upward DC confirmation.
            if p < pl:
                pl = p
                t_dc0 = np.int64(i)
                out_new_low[i] = True

            if p >= pl * (1.0 + theta):
                # Upward DC confirmed.
                # RDC: price rise from trough to confirmation / trough / bars-since-last-new-low
                out_upturn_dc[i] = True
                t_span = np.int64(i) - t_dc0
                out_rdc[i] = abs(p - pl) / pl / float(t_span) if t_span > 0 else np.nan
                ph = p
                t_dc0 = np.int64(i)
                mode = np.int8(1)

        out_direction[i] = mode
        out_ph[i] = ph
        out_pl[i] = pl
        out_t_dc0[i] = t_dc0

    return (out_direction, out_ph, out_pl, out_t_dc0,
            out_upturn_dc, out_downturn_dc, out_new_high, out_new_low, out_rdc)


def idc_parse(prices: pd.Series, theta: float, alpha: float = 1.0) -> pd.DataFrame:
    """
    Parse a price series into per-bar IDC event signals for ITA Algorithm 1.

    Unlike parse_dc_events() which returns one row per completed trend, this function
    returns one row per bar with the event signals needed for Algorithm 1: new_high and
    new_low events, DC confirmations, running ph/pl extremes, and RDC.

    Parameters
    ----------
    prices : pd.Series
        Close price series. Index can be integer or DatetimeIndex.
    theta : float
        DC threshold as a decimal fraction (e.g., 0.002 for 0.2%).
    alpha : float, optional
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC confirmed when price falls alpha*theta from the last peak.
        alpha=1.0 (default) -> symmetric thresholds matching the original book formulation.
        alpha<1.0 -> downward DC fires sooner, acting as a tighter stop-loss.

    Returns
    -------
    pd.DataFrame
        One row per bar, same index as prices, with columns:

        - direction   : 1.0 (upturn), -1.0 (downturn), NaN (initialisation phase)
        - ph          : running high — confirmed peak held fixed in downturn; updated at
                        new highs in upturn; running init_high before first DC.
        - pl          : running low  — confirmed trough held fixed in upturn; updated at
                        new lows in downturn; running init_low before first DC.
        - t_dc0       : bar offset (integer position, not index label) of the last
                        extreme update in the current half-cycle. Used for RDC / T computation.
        - upturn_dc   : True when an upward DC is confirmed (price rose theta from last trough).
                        False on the initial mode=0 -> mode=1 transition carries NaN rdc.
        - downturn_dc : True when a downward DC is confirmed (price fell alpha*theta from peak).
                        False on the initial mode=0 -> mode=-1 transition carries NaN rdc.
        - new_high    : True when a new high is set during an upturn.
                        This is the Rule 2 trigger bar in ITA Algorithm 1.
        - new_low     : True when a new low is set during a downturn.
        - rdc         : RDC indicator at DC confirmation bars (NaN for all other bars and
                        for the very first DC that transitions out of the init phase).
                        Formula: |price - extreme| / extreme / T, where extreme is ph (at
                        downturn DC) or pl (at upturn DC) and T = bars since last extreme.

    Notes
    -----
    - upturn_dc and new_high are mutually exclusive per bar.
    - downturn_dc and new_low are mutually exclusive per bar.
    - upturn_dc and downturn_dc are mutually exclusive per bar.
    - rdc at downturn DC: |confirmation_price - ph| / ph / T  (drop from peak).
    - rdc at upturn DC:   |confirmation_price - pl| / pl / T  (rise from trough).
    - Primary use: per-bar input to ITA Algorithm 1 (Wu & Han arXiv:2309.15383, 2023).
    - Book reference: Hu et al. SSRN:4048864 (2022); Chen & Tsang CRC Press (2021) Appendix A.
    """
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    if not (0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if len(prices) < 2:
        return pd.DataFrame({
            'direction': np.nan, 'ph': np.nan, 'pl': np.nan,
            't_dc0': np.int64(0), 'upturn_dc': False, 'downturn_dc': False,
            'new_high': False, 'new_low': False, 'rdc': np.nan,
        }, index=prices.index)

    arr = prices.values.astype(np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("prices contains NaN or infinite values. Clean the data before parsing.")

    direction, ph, pl, t_dc0, upturn_dc, downturn_dc, new_high, new_low, rdc = _idc_core(arr, theta, alpha)

    direction_float = np.where(direction == 0, np.nan, direction.astype(np.float64))

    return pd.DataFrame({
        'direction': direction_float,
        'ph': ph,
        'pl': pl,
        't_dc0': t_dc0,
        'upturn_dc': upturn_dc,
        'downturn_dc': downturn_dc,
        'new_high': new_high,
        'new_low': new_low,
        'rdc': rdc,
    }, index=prices.index)
