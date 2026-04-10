import numpy as np
import pandas as pd
from scipy import stats


def k_ratio(equity, periods_per_year=None, window=None, normalize=False):
    """
    Compute the K-Ratio (Kestner Ratio) for an equity curve.

    The K-Ratio measures the smoothness and consistency of equity growth,
    by regressing log equity on time and evaluating how linear the growth is.

    Parameters
    ----------
    equity : array-like, pd.Series, or pd.DataFrame
        Cumulative equity curve. If DataFrame, uses first column.

    periods_per_year : int or None, optional
        Number of periods per year (for annualization). None = no scaling.

    window : int or None, optional (default=None)
        If provided, compute rolling K-Ratio with this window size.

    normalize : bool, optional (default=False)
        Apply logistic normalization: 1 / (1 + exp(-k)).

    Returns
    -------
    float or pd.Series
        K-Ratio, or rolling Series if window is provided.
    """
    # --- Input normalization ---
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]
    elif not isinstance(equity, pd.Series):
        equity = pd.Series(equity)

    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(equity)
    if n < 5:
        raise ValueError("Not enough data points to compute K-Ratio.")

    def _compute_k(sub_equity):
        # Filter non-positive values but keep track of valid indices
        valid_mask = sub_equity > 0
        if valid_mask.sum() < 5:
            return np.nan

        # Use only valid equity values
        valid_equity = sub_equity[valid_mask]
        n_local = len(valid_equity)

        y = np.log(valid_equity.values)
        x = np.arange(n_local)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Standard error should not be zero
        if std_err == 0 or np.isnan(std_err):
            return np.nan

        # K-Ratio formula: (slope / std_err) * sqrt(n)
        k_val = slope / std_err * np.sqrt(n_local)

        # Annualize if periods_per_year is provided
        if periods_per_year is not None:
            k_val *= np.sqrt(periods_per_year)

        if normalize:
            k_val = 1 / (1 + np.exp(-k_val))

        return k_val

    # --- Rolling computation ---
    if window is not None:
        return equity.rolling(window=window, min_periods=5).apply(_compute_k, raw=False)

    return _compute_k(equity)
