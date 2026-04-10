import pandas as pd
from typing import List, Optional


def create_lag_features(df: pd.DataFrame, n_list: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Add lagged versions of selected columns to a DataFrame for supervised time-series modelling
    (sliding-window / series-to-supervised transform).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Index is preserved as-is.
    n_list : list of int
        Lag values to generate (e.g. [1, 2, 3]).  Positive integers only.
    columns : list of str, optional
        Columns to lag.  If None, all columns in *df* are lagged.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns named
        ``{col}_lag_{n}`` for every (col, n) pair.
        Rows that cannot be filled (the first ``max(n_list)`` rows) will
        contain NaN in the lagged columns.
    """
    if not n_list:
        raise ValueError("n_list must contain at least one lag value.")
    if any(n <= 0 for n in n_list):
        raise ValueError("All lag values in n_list must be positive integers.")

    cols = columns if columns is not None else list(df.columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    lag_frames = [df]
    for n in n_list:
        shifted = df[cols].shift(n)
        shifted.columns = [f"{c}_lag_{n}" for c in cols]
        lag_frames.append(shifted)

    return pd.concat(lag_frames, axis=1)
