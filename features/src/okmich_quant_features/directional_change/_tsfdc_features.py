"""
TSFDC — Feature extraction for BBTheta classifier.

parse_dual_dc          — run STheta and BTheta DC parsers on the same price series.
label_bbtheta          — assign BBTheta (True/False) labels to STheta trends (training only).
extract_tsfdc_features — compute (TMV, T, OSV, COP) feature matrix for all STheta trends.

Reference: Bakhach, Tsang & Jalalian (IEEE CIFEr, 2016) Section 3;
           Bakhach, Tsang & Chinthalapati (ISAFM, 2018) Sections 3.1, 4.
"""
import numpy as np
import pandas as pd

from ._parser import parse_dc_events


def parse_dual_dc(prices: pd.Series, stheta: float, btheta: float) -> tuple:
    """
    Run STheta and BTheta DC parsers on the same price series.

    Parameters
    ----------
    prices : pd.Series
        Close price series.
    stheta : float
        Small DC threshold (e.g., 0.001 for 0.1%).
    btheta : float
        Big DC threshold (e.g., 0.0013 for 0.13%). Must be > stheta.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (trends_s, trends_b) — completed STheta and BTheta DC trends.
        Both DataFrames share the column schema of parse_dc_events().

    Raises
    ------
    ValueError
        If btheta <= stheta.
    """
    if btheta <= stheta:
        raise ValueError(f"btheta ({btheta}) must be greater than stheta ({stheta})")
    return parse_dc_events(prices, stheta), parse_dc_events(prices, btheta)


def label_bbtheta(trends_s: pd.DataFrame, trends_b: pd.DataFrame) -> pd.DataFrame:
    """
    Assign BBTheta labels to STheta trends.

    BBTheta_i = True  if the i-th STheta extreme end bar (ext_end_pos) is also
                      a BTheta extreme end bar — i.e. the STheta trend continued
                      until its magnitude reached BTheta before reversing.
    BBTheta_i = False otherwise.

    Uses bar-index matching (more robust than price matching).
    Call this function on training data only — labels require perfect foresight.

    Parameters
    ----------
    trends_s : pd.DataFrame
        STheta DC trends from parse_dc_events() — must contain ext_end_pos.
    trends_b : pd.DataFrame
        BTheta DC trends from parse_dc_events() — must contain ext_end_pos.

    Returns
    -------
    pd.DataFrame
        Copy of trends_s with added 'bbtheta' column (bool).

    Raises
    ------
    ValueError
        If required columns are missing from either DataFrame.
    """
    for df, name in [(trends_s, 'trends_s'), (trends_b, 'trends_b')]:
        if 'ext_end_pos' not in df.columns:
            raise ValueError(f"{name} missing column: ext_end_pos")

    btheta_ext_bars = set(trends_b['ext_end_pos'].values)
    result = trends_s.copy()
    result['bbtheta'] = trends_s['ext_end_pos'].isin(btheta_ext_bars)
    return result


def extract_tsfdc_features(trends_s: pd.DataFrame, trends_b: pd.DataFrame, stheta: float, btheta: float) -> pd.DataFrame:
    """
    Compute (TMV, T, OSV, COP) feature matrix for all STheta DC trends.

    Features are computed at each STheta DCC confirmation point and represent
    the completed DC event.  The first row will have OSV = NaN (no previous
    DCC).  Rows before the first BTheta DCC event will have COP = NaN.

    Feature definitions (Bakhach et al. 2018, Section 3.1):
      TMV — Total Move Value = trends_s["tmv"] (pre-computed by parse_dc_events).
      T   — Time for completion = trends_s["t"] (pre-computed, bars between extremes).
      OSV — OS Value = (dcc_price[i] - dcc_price[i-1]) / (dcc_price[i-1] * stheta).
      COP — Cross-threshold OS % = (ext_start_price[i] - last_btheta_dcc_price)
                                    / (last_btheta_dcc_price * btheta),
            where last_btheta_dcc_price is the most recent BTheta DCC price
            at or before the current STheta DCC confirmation bar.

    Parameters
    ----------
    trends_s : pd.DataFrame
        STheta DC trends from parse_dc_events() — must have: ext_start_price,
        dcc_price, dcc_pos, tmv, t.
    trends_b : pd.DataFrame
        BTheta DC trends from parse_dc_events() — must have: dcc_price, dcc_pos.
    stheta : float
        Small DC threshold used to parse trends_s.
    btheta : float
        Big DC threshold used to parse trends_b.

    Returns
    -------
    pd.DataFrame
        Same index as trends_s, columns: TMV, T, OSV, COP.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_s = {'ext_start_price', 'dcc_price', 'dcc_pos', 'tmv', 't'}
    missing = required_s - set(trends_s.columns)
    if missing:
        raise ValueError(f"trends_s missing columns: {missing}")

    required_b = {'dcc_price', 'dcc_pos'}
    missing_b = required_b - set(trends_b.columns)
    if missing_b:
        raise ValueError(f"trends_b missing columns: {missing_b}")

    # TMV and T are pre-computed by parse_dc_events
    tmv = trends_s['tmv'].values.copy()
    t = trends_s['t'].values.copy()

    # OSV = (dcc_price[i] - dcc_price[i-1]) / (dcc_price[i-1] * stheta)
    prev_dcc = trends_s['dcc_price'].shift(1)
    osv = ((trends_s['dcc_price'] - prev_dcc) / (prev_dcc * stheta)).values

    # COP = (ext_start_price[i] - last_btheta_dcc_price) / (last_btheta_dcc_price * btheta)
    cop = _compute_cop_batch(trends_s, trends_b, btheta)

    return pd.DataFrame({'TMV': tmv, 'T': t, 'OSV': osv, 'COP': cop}, index=trends_s.index)


def _compute_cop_batch(trends_s: pd.DataFrame, trends_b: pd.DataFrame, btheta: float) -> np.ndarray:
    """
    Compute COP for all STheta trends using pd.merge_asof.

    For each STheta trend i, finds the most recent BTheta DCC at or before
    trends_s['dcc_pos'][i] and computes:
        COP = (ext_start_price[i] - btheta_dcc_price) / (btheta_dcc_price * btheta)

    Returns NaN for all rows if trends_b is empty.
    """
    if len(trends_b) == 0:
        return np.full(len(trends_s), np.nan)

    s_df = pd.DataFrame({
        'dcc_pos': trends_s['dcc_pos'].values.astype(np.int64),
        'ext_start_price': trends_s['ext_start_price'].values,
    })
    b_df = pd.DataFrame({
        'dcc_pos': trends_b['dcc_pos'].values.astype(np.int64),
        'btheta_dcc_price': trends_b['dcc_price'].values,
    })

    # merge_asof requires sorted keys; DC events are naturally sequential
    merged = pd.merge_asof(s_df, b_df, on='dcc_pos', direction='backward')
    cop_vals = (
        (merged['ext_start_price'] - merged['btheta_dcc_price'])
        / (merged['btheta_dcc_price'] * btheta)
    ).values
    return cop_vals