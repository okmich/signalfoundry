"""Trend persistence labeling: vol-normalised drift over a lookback window, thresholded at +/- 0.25.

drift  = price[t] - price[t-window]
score  = drift / (price[t-window] * rolling_std(returns, window))
labels = +1 if score >= 0.25, -1 if score <= -0.25, else 0

zscore_norm=True applies a rolling standardization to the score before thresholding. Choose `window` from a
wall-clock anchor (e.g. ~1 trading day worth of bars); set `smooth ~ window // 4`.
"""

import numpy as np
import pandas as pd


def trend_persistence_labeling(price_series: pd.Series, window: int = 20, smooth: int = 5, zscore_norm: bool = True,
                               name: str = "trend_label") -> pd.Series:
    px = price_series.astype(float)
    px_shifted = px.shift(window)

    drift = px - px_shifted
    returns = px.pct_change(fill_method=None)
    vol = returns.rolling(window, min_periods=window).std()

    denominator = px_shifted * (vol + 1e-12)
    score = drift / denominator

    # Vectorized z-score normalization
    if zscore_norm:
        mu = score.rolling(window, min_periods=1).mean()
        sd = score.rolling(window, min_periods=1).std()
        score = (score - mu) / (sd + 1e-12)

    # Apply smoothing with optimized rolling mean
    score = score.rolling(smooth, min_periods=1).mean()

    labels = pd.Series(
        np.select([score >= 0.25, score <= -0.25], [1.0, -1.0], default=0.0),
        index=price_series.index,
        name=name,
    )

    return labels.fillna(0.0)
