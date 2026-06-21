"""Trend feature package — lean, orthogonal set.

Directional regime labels
-------------------------
  continuous_trend_labeling      Price-action state machine; +1 / -1 / 0
  trend_persistence_labeling     Vol-normalised drift; +1 / -1 / 0

Streaming CTL (O(1) per-bar; live/online use)
---------------------------------------------
  CTLState                       Persistent state mirroring continuous_trend_labeling's machine
  ctl_step                       Advance the state by one bar; returns the bar's label
  ctl_warm_up                    Replay a history once, return the live state for incremental stepping
  ctl_streaming_replay           Replay a series through the FSM (equivalence harness / batch parity)

Channels (mean line +/- vol-based half-width)
---------------------------------------------
  bollinger_band                 SMA +/- k*stdev (returns bands + %B + width)
  envelope                       SMA +/- k_atr*ATR (Bollinger analog, gap-robust)
  keltner_channels               MA +/- k_atr*ATR (MA-type selectable)

Continuous trend features
-------------------------
  zscore_trend_features          Continuous rolling z-score of log returns + derivative + |z|

Normalized MA primitives & derived features (normalized_ma.py)
---------------------------------------------------------------
  norm_sma, norm_ema, norm_lwma  log(close / MA) for SMA / EMA / linear-weighted MA
  norm_dema, norm_tema, norm_smma  log(close / MA) for DEMA / TEMA / Wilder-smoothed
  norm_vwap                      log(close / rolling VWAP)
  norm_moving_average            Dispatcher returning log(close / MA) per MovingAverageType
  MovingAverageType              Enum of supported MA flavors
  ma_slope_norm                  Slope of raw MA per bar divided by ATR — trend strength

Bundle
------
  core_trend_features            Concatenates the orthogonal core into one DataFrame.

For raw smoothing of any time series, use filters.py (smooth_ema, smooth_sma, smooth_wma, smooth_kalman, ...).
normalized_ma.py is for *trend feature* MA usage; filters.py is for *smoothing* any series.

Parameter selection: callers pass parameters directly. Anchor to interpretable scales (vol, wall-clock horizon)
rather than fitting — the package no longer ships any parameter optimizers.
"""

import pandas as pd
import numpy as np

from .channels import bollinger_band, envelope, keltner_channels
from .continuous_trend import CTLState, continuous_trend_labeling, ctl_step, ctl_streaming_replay, ctl_warm_up
from .normalized_ma import MovingAverageType, ma_slope_norm, norm_dema, norm_ema, norm_lwma, norm_moving_average, \
                            norm_sma, norm_smma, norm_tema, norm_vwap
from .trend_persistence import trend_persistence_labeling
from .z_score_trend import zscore_trend_features


def core_trend_features(df: pd.DataFrame,
    bb_window: int = 24, bb_deviation_up: float = 2.0, bb_deviation_down: float = 2.0,
    continuous_omega: float = 0.15,
    persistence_window: int = 20, persistence_smooth: int = 5, persistence_zscore_norm: bool = True,
    zscore_window: int = 30, zscore_deriv_window: int = 5, close_col: str = "close") -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)

    close_price = df[close_col]
    bb_upper, bb_middle, bb_lower, bb_percent_b, bb_width = bollinger_band(close_price, window=bb_window,
                                                                           deviation_up=bb_deviation_up,
                                                                           deviation_down=bb_deviation_down)

    safe_close = close_price.replace(0, np.nan)
    result[f"bb_upper_{bb_window}"] = bb_upper / safe_close
    result[f"bb_middle_{bb_window}"] = bb_middle / safe_close
    result[f"bb_lower_{bb_window}"] = bb_lower / safe_close
    result[f"bb_percent_b_{bb_window}"] = bb_percent_b
    result[f"bb_width_{bb_window}"] = bb_width

    result[f"continuous_trend_{continuous_omega}".replace(".", "_")] = \
        continuous_trend_labeling(close_price, omega=continuous_omega)

    result[f"trend_persistence_{persistence_window}_{persistence_smooth}"] = (
        trend_persistence_labeling(close_price, window=persistence_window, smooth=persistence_smooth,
                                   zscore_norm=persistence_zscore_norm))

    zscore_df = zscore_trend_features(close_price, window=zscore_window, deriv_window=zscore_deriv_window)
    for col in zscore_df.columns:
        result[col] = zscore_df[col]

    return result


__all__ = [
    "core_trend_features",
    # Directional labels
    "continuous_trend_labeling",
    "trend_persistence_labeling",
    # Streaming CTL (live/online)
    "CTLState",
    "ctl_step",
    "ctl_warm_up",
    "ctl_streaming_replay",
    # Channels
    "bollinger_band",
    "envelope",
    "keltner_channels",
    # Continuous trend features
    "zscore_trend_features",
    # Normalized MA primitives & derived features
    "MovingAverageType",
    "norm_sma",
    "norm_ema",
    "norm_lwma",
    "norm_dema",
    "norm_tema",
    "norm_smma",
    "norm_vwap",
    "norm_moving_average",
    "ma_slope_norm",
]