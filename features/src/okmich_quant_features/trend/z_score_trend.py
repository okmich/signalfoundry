"""Z-Score Trend Exhaustion / Reversal Feature.

zscore_trend_features() returns the continuous rolling z-score of log returns plus its derivative and absolute
magnitude. A reversal / exhaustion signal — "is the current move statistically extreme?" — not a directional
trend labeler. Use the continuous values as ML features; do not discretize.
"""

import numpy as np
import pandas as pd


def zscore_trend_features(price_series: pd.Series, window: int = 30, deriv_window: int = 5) -> pd.DataFrame:
    """Continuous z-score features for trend exhaustion / reversal detection.

    Returns a DataFrame with three columns:
        - zscore_{window}: raw z-score value (how extreme the current move is)
        - zscore_deriv_{window}: rolling mean of z-score changes (acceleration / deceleration)
        - zscore_abs_{window}: absolute z-score (magnitude of extremity, direction-agnostic)
    """
    returns = np.log(price_series / price_series.shift())
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    zscore = (returns - rolling_mean) / (rolling_std + 1e-8)

    result = pd.DataFrame(index=price_series.index)
    result[f"zscore_{window}"] = zscore
    result[f"zscore_deriv_{window}"] = zscore.diff().rolling(window=deriv_window).mean()
    result[f"zscore_abs_{window}"] = zscore.abs()
    return result