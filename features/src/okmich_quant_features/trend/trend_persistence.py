"""Trend persistence labeling: vol-normalised drift over a lookback window, thresholded at +/- 0.25.

drift  = price[t] - price[t-window]
score  = drift / (price[t-window] * rolling_std(returns, window))
labels = +1 if score >= 0.25, -1 if score <= -0.25, 0 if |score| < 0.25, NaN if insufficient data.

zscore_norm=True applies a rolling standardization to the score before thresholding. Choose `window` from a
wall-clock anchor (e.g. ~1 trading day worth of bars); set `smooth ~ window // 4`.

Threshold note: +/-0.25 corresponds to ~0.25 standard deviations of vol-normalised drift. It is a calibrated
default — strong enough to suppress chop but loose enough to catch sustained moves before they mature. Hardcoded
because it is not naturally a per-instrument knob; the vol normalisation already adapts to instrument scale.
"""

import numpy as np
import pandas as pd


def trend_persistence_labeling(price_series: pd.Series, window: int = 20, smooth: int = 5,
                               zscore_norm: bool = True) -> pd.Series:
    """Vol-normalised drift label.

    Returns a float64 Series matching the input index, with values in {+1.0, -1.0, 0.0, NaN}.
    NaN marks warmup (insufficient data to compute the rolling vol normaliser); 0.0 marks
    bars where the smoothed score is inside +/-0.25.
    """
    px = price_series.astype(float)
    px_shifted = px.shift(window)

    drift = px - px_shifted
    returns = px.pct_change(fill_method=None)
    vol = returns.rolling(window, min_periods=window).std()

    denominator = px_shifted * (vol + 1e-12)
    score = drift / denominator

    if zscore_norm:
        mu = score.rolling(window, min_periods=1).mean()
        sd = score.rolling(window, min_periods=1).std()
        score = (score - mu) / (sd + 1e-12)

    score = score.rolling(smooth, min_periods=1).mean()

    # np.select with default=0 would bury NaN-score bars as 0 (since NaN comparisons are False).
    # Use np.where chains explicitly so NaN propagates through to the label.
    labels = np.where(score >= 0.25, 1.0, np.where(score <= -0.25, -1.0, np.where(score.isna(), np.nan, 0.0)))
    return pd.Series(labels, index=price_series.index, name="trend_label")
