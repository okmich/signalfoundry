"""
C+GP+TS — Feature extraction and αDC/βDC labelling for the trend reversal
prediction framework.

Provides two public functions built on top of parse_dc_events() output:

  label_alpha_beta_dc   — adds dc_length, os_length, has_os columns
  extract_dc_classification_features — extracts X1–X5 per trend

Each row i in parse_dc_events() represents a completed EXT→EXT trend whose
existence was confirmed by a closing DCC (dcc_pos[i]).  The "DC event" that
opens the *next* trend spans from ext_end_pos[i] to dcc_pos[i].  The
overshoot of the next trend spans from dcc_pos[i] to ext_end_pos[i+1].

Consequently:

  dc_length[i] = dcc_pos[i] - ext_end_pos[i]   (closing DC event for row i)
  os_length[i] = ext_end_pos[i+1] - dcc_pos[i]  (overshoot of the next trend)
  has_os[i]    = os_length[i] > 0

Features X1–X5 are computed at the closing DCC of row i (all values are
available at that point with no look-ahead) and used to classify whether the
next trend will have an overshoot.

Reference: Adegboye & Kampouridis (2020) — Tables 2, 3; Section 3.
"""
import numpy as np
import pandas as pd


def label_alpha_beta_dc(trends: pd.DataFrame) -> pd.DataFrame:
    """
    Add DC event length, OS event length, and αDC/βDC label to parse_dc_events() output.

    For each completed trend row i, the closing DC event (ext_end → dcc) has length
    dc_length[i], and the overshoot that follows (dcc[i] → ext_end[i+1]) has length
    os_length[i].  A trend is αDC when its overshoot is positive (os_length > 0).

    The last row always receives has_os=NaN (no subsequent trend to measure OS from)
    and must be excluded from classifier and GP training.

    Parameters
    ----------
    trends : pd.DataFrame
        Output of parse_dc_events().  Must contain ext_end_pos and dcc_pos columns.

    Returns
    -------
    pd.DataFrame
        Copy of trends with three additional columns:
        - dc_length : int — bars of the closing DC event (dcc_pos - ext_end_pos).
                      Always >= 1 by DC parser invariant.
        - os_length : float — bars of the overshoot that follows (ext_end_pos[i+1] - dcc_pos[i]).
                      0 means βDC (no overshoot).  NaN for the last row.
        - has_os    : object — True (αDC), False (βDC), or NaN for the last row.

    Notes
    -----
    - Requires ext_end_pos and dcc_pos columns; call parse_dc_events() first.
    - Adegboye & Kampouridis (2020) Section 3; Figure 1.
    """
    required = {'ext_end_pos', 'dcc_pos'}
    missing = required - set(trends.columns)
    if missing:
        raise ValueError(f"label_alpha_beta_dc: trends is missing columns {missing}. "
                         "Call parse_dc_events() which emits ext_end_pos and dcc_pos.")

    out = trends.copy()
    out['dc_length'] = (out['dcc_pos'] - out['ext_end_pos']).astype(int)
    out['os_length'] = out['ext_end_pos'].shift(-1) - out['dcc_pos']
    out['has_os'] = out['os_length'].apply(lambda v: (True if v > 0 else False) if pd.notna(v) else np.nan)
    return out


def extract_dc_classification_features(trends: pd.DataFrame) -> pd.DataFrame:
    """
    Extract X1–X5 classification attributes for each trend row.

    Features are computable at the closing DCC of each trend (no look-ahead).
    They characterise the DC event that just fired and are used to predict
    whether the upcoming trend will have an overshoot (αDC vs βDC).

    Input must be the output of label_alpha_beta_dc() (which itself requires
    parse_dc_events() output with integer position columns).

    Parameters
    ----------
    trends : pd.DataFrame
        Output of label_alpha_beta_dc().

    Returns
    -------
    pd.DataFrame
        Same index as trends, with columns:
        - X1 : float — |dcc_price - ext_end_price|. Price magnitude of the DC event.
        - X2 : int   — dc_length = dcc_pos - ext_end_pos. Duration of the DC event (bars).
                       Always >= 1 by DC parser invariant; clipped to 1 for safety.
        - X3 : float — X1 / X2. Price velocity of the DC event (Sigma′).
        - X4 : float — dcc_price of the preceding trend. Market level context.
                       NaN for the first row (no predecessor).
        - X5 : int   — 1 if the preceding trend was αDC, 0 otherwise.
                       0 for the first row.

    Notes
    -----
    - Adegboye & Kampouridis (2020) Table 2; Section 3.2.
    - The training label for each row is has_os from label_alpha_beta_dc().
      Drop the last row (has_os=NaN) before fitting a classifier.
    """
    required = {'dcc_price', 'ext_end_price', 'dc_length', 'has_os'}
    missing = required - set(trends.columns)
    if missing:
        raise ValueError(f"extract_dc_classification_features: trends is missing columns "
                         f"{missing}. Call label_alpha_beta_dc() first.")

    x1 = (trends['dcc_price'] - trends['ext_end_price']).abs()
    x2 = trends['dc_length'].clip(lower=1)
    x3 = x1 / x2
    x4 = trends['dcc_price'].shift(1)
    x5 = trends['has_os'].shift(1).map(lambda v: 1 if v is True else 0).astype(int)

    return pd.DataFrame({'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5},
                        index=trends.index)
